# =============================================================================
# Import required libraries
# =============================================================================
import os
import numpy as np
import cv2
from PIL import Image

import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from criteria.cosine_loss import CosineLoss
from criteria.nce_loss import NCELoss
from attention_control import AttentionControlEdit
from utils import *


@torch.enable_grad()
class Adversarial_Opt:
    def __init__(self, args, model):
        self.device = args.device
        self.dataloader = args.dataloader
        #
        self.diff_model = model
        self.diff_model.vae.requires_grad_(False)
        self.diff_model.text_encoder.requires_grad_(False)
        self.diff_model.unet.requires_grad_(False)
        #
        self.source_dir = args.source_dir
        self.protected_image_dir = args.protected_image_dir
        self.comparison_null_text = args.comparison_null_text
        #
        self.target_choice = args.target_choice
        #
        self.is_makeup = args.is_makeup
        self.source_text = args.source_text
        self.makeup_prompt = args.makeup_prompt
        self.augment = transforms.RandomPerspective(fill=0, p=1,
                                                    distortion_scale=0.5)
        #
        self.MTCNN_cropping = args.MTCNN_cropping
        #
        self.is_obfuscation = args.is_obfuscation
        #
        self.image_size = args.image_size
        self.prot_steps = args.prot_steps
        self.diffusion_steps = args.diffusion_steps
        self.start_step = args.start_step
        self.null_optimization_steps = args.null_optimization_steps
        #
        self.adv_optim_weight = args.adv_optim_weight
        self.makeup_weight = args.makeup_weight
        #
        self.augment = transforms.RandomPerspective(
            fill=0, p=1, distortion_scale=0.5)
        # Set up loss functions
        self.cosine_loss = CosineLoss(self.is_obfuscation)
        self.nce_loss = NCELoss(self.device, clip_model="ViT-B/32")
        # set up FR models
        self.surrogate_models = load_FR_models(
            args, args.surrogate_model_names)
        self.test_model_name = args.test_model_name

    def get_FR_embeddings(self, image):
        features = []
        for model_name in self.surrogate_models.keys():
            input_size = self.surrogate_models[model_name][0]
            fr_model = self.surrogate_models[model_name][1]
            emb_source = fr_model(F.interpolate(
                image, size=input_size, mode='bilinear'))
            features.append(emb_source)
        return features

    def set_attention_control(self, controller):
        def ca_forward(self, place_in_unet):

            def forward(x, context=None):
                q = self.to_q(x)
                is_cross = context is not None
                context = context if is_cross else x
                k = self.to_k(context)
                v = self.to_v(context)
                q = self.reshape_heads_to_batch_dim(q)
                k = self.reshape_heads_to_batch_dim(k)
                v = self.reshape_heads_to_batch_dim(v)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

                attn = sim.softmax(dim=-1)
                attn = controller(attn, is_cross, place_in_unet)
                out = torch.einsum("b i j, b j d -> b i d", attn, v)
                out = self.reshape_batch_dim_to_heads(out)

                out = self.to_out[0](out)
                out = self.to_out[1](out)
                return out

            return forward

        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'CrossAttention':
                net_.forward = ca_forward(net_, place_in_unet)
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = self.diff_model.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")

        controller.num_att_layers = cross_att_count

    def reset_attention_control(self):
        def ca_forward(self):
            def forward(x, context=None):
                q = self.to_q(x)
                is_cross = context is not None
                context = context if is_cross else x
                k = self.to_k(context)
                v = self.to_v(context)
                q = self.reshape_heads_to_batch_dim(q)
                k = self.reshape_heads_to_batch_dim(k)
                v = self.reshape_heads_to_batch_dim(v)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

                attn = sim.softmax(dim=-1)
                out = torch.einsum("b i j, b j d -> b i d", attn, v)
                out = self.reshape_batch_dim_to_heads(out)

                out = self.to_out[0](out)
                out = self.to_out[1](out)
                return out

            return forward

        def register_recr(net_):
            if net_.__class__.__name__ == 'CrossAttention':
                net_.forward = ca_forward(net_)
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    register_recr(net__)

        sub_nets = self.diff_model.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                register_recr(net[1])
            elif "up" in net[0]:
                register_recr(net[1])
            elif "mid" in net[0]:
                register_recr(net[1])

    def diffusion_step(self, latent, null_context, t,
                       is_null_optimization=False):
        if not is_null_optimization:
            latent_input = torch.cat([latent] * 2)
            noise_pred = self.diff_model.unet(
                latent_input, t, encoder_hidden_states=null_context)["sample"]
            noise_pred, _ = noise_pred.chunk(2)
        else:
            noise_pred = self.diff_model.unet(
                latent, t, encoder_hidden_states=null_context)["sample"]
        return self.diff_model.scheduler.step(noise_pred, t, latent)["prev_sample"]

    def null_text_embeddings(self):
        uncond_input = self.diff_model.tokenizer([""],
                                                 padding="max_length",
                                                 max_length=self.diff_model.tokenizer.model_max_length,
                                                 return_tensors="pt")
        return self.diff_model.text_encoder(uncond_input.input_ids.to(self.device))[0]

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            generator = torch.Generator().manual_seed(8888)
            gpu_generator = torch.Generator(device=image.device)
            gpu_generator.manual_seed(generator.initial_seed())
            latents = self.diff_model.vae.encode(
                image).latent_dist.sample(generator=gpu_generator)
            latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latent):
        latent = 1 / 0.18215 * latent
        image = self.diff_model.vae.decode(latent)['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def ddim_inversion(self, image):
        uncond_embeddings = self.null_text_embeddings()
        #
        self.diff_model.scheduler.set_timesteps(self.diffusion_steps)
        #
        latent = self.image2latent(image)
        all_latents = [latent]
        for i in tqdm(range(self.diffusion_steps - 1)):
            t = self.diff_model.scheduler.timesteps[self.diffusion_steps - i - 1]
            #
            noise_pred = self.diff_model.unet(latent,
                                              t,
                                              encoder_hidden_states=uncond_embeddings)["sample"]
            #
            next_timestep = t + self.diff_model.scheduler.config.num_train_timesteps // self.diff_model.scheduler.num_inference_steps
            alpha_bar_next = self.diff_model.scheduler.alphas_cumprod[next_timestep] \
                if next_timestep <= self.diff_model.scheduler.config.num_train_timesteps else torch.tensor(0.0)
            reverse_x0 = (1 / torch.sqrt(self.diff_model.scheduler.alphas_cumprod[t]) * (
                latent - noise_pred * torch.sqrt(1 - self.diff_model.scheduler.alphas_cumprod[t])))
            latent = reverse_x0 * \
                torch.sqrt(alpha_bar_next) + \
                torch.sqrt(1 - alpha_bar_next) * noise_pred
            all_latents.append(latent)

        return all_latents

    def null_optimization(self, inversion_latents):
        """
        Optimizing the unconditional embeddings based on the paper:
            Null-text Inversion for Editing Real Images using Guided Diffusion Models
        GiHub:
            https://github.com/google/prompt-to-prompt
        """
        all_uncond_embs = []

        latent = inversion_latents[self.start_step - 1]

        uncond_embeddings = self.null_text_embeddings()
        uncond_embeddings.requires_grad_(True)
        optimizer = optim.AdamW([uncond_embeddings], lr=1e-1)
        #
        # criterion torch.nn.L1Loss()
        criterion = torch.nn.MSELoss()
        #
        for i in tqdm(range(self.start_step, self.diffusion_steps)):
            t = self.diff_model.scheduler.timesteps[i]
            for _ in range(self.null_optimization_steps):
                out_latent = self.diffusion_step(latent, uncond_embeddings, t,
                                                 True)
                optimizer.zero_grad()
                loss = criterion(
                    out_latent, inversion_latents[i])
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                latent = self.diffusion_step(latent, uncond_embeddings, t,
                                             True).detach()
                all_uncond_embs.append(uncond_embeddings.detach().clone())
        #
        uncond_embeddings.requires_grad_(False)
        return all_uncond_embs

    def visualize(self, image_name, real_image, latents, controller):
        adversarial_image = self.latent2image(latents)
        adversarial_image = adversarial_image[1:]

        result_dir = self.protected_image_dir + '/' + \
            self.test_model_name[0] + '/' + \
            self.target_choice + '/' + image_name

        adversarial_img = cv2.cvtColor(adversarial_image[0], cv2.COLOR_RGB2BGR)
        cv2.imwrite(result_dir + ".png", adversarial_img)

        adversarial_image = adversarial_image.astype(np.float32) / 255

        real = (real_image / 2 + 0.5).clamp(0,
                                            1).permute(0, 2, 3, 1).cpu().numpy()
        """
        diff = adversarial_image - real

        diff_absolute = np.abs(diff)
        Image.fromarray(
            (diff_absolute[0] * 255).astype(np.uint8)).save(result_dir + "_diff_absolute.png")
        """

    def attacker(self,
                 image,
                 image_name,
                 source_embeddings,
                 target_embeddings,
                 controller,
                 null_text_dir=None,
                 bb_src1=None):
        # lat[0], lat[1], lat[2], ...
        inversion_latents = self.ddim_inversion(image)
        # reverse
        inversion_latents = inversion_latents[::-1]
        latent = inversion_latents[self.start_step - 1]
        #
        all_uncond_embs = self.null_optimization(inversion_latents)

        #######################################################################
        '''
        comparison between null_text and null_text optimized:

        '''
        if self.comparison_null_text:
            latent_holder = latent_holder_opt = latent.clone()
            uncond_embeddings = self.null_text_embeddings()
            #
            with torch.no_grad():
                for i in range(self.start_step, self.diffusion_steps):
                    t = self.diff_model.scheduler.timesteps[i]
                    #
                    latent_holder = self.diffusion_step(latent_holder,
                                                        uncond_embeddings,
                                                        t, True)
                    #
                    latent_holder_opt = self.diffusion_step(latent_holder_opt,
                                                            all_uncond_embs[i -
                                                                            self.start_step],
                                                            t, True)
                    #
                image_rec = self.latent2image(latent_holder)
                image_rec = cv2.cvtColor(image_rec[0], cv2.COLOR_RGB2BGR)
                result_dir = os.path.join(
                    null_text_dir, f"{image_name}_rec.png")
                cv2.imwrite(result_dir, image_rec)
                #
                image_rec_opt = self.latent2image(latent_holder_opt)
                image_rec_opt = cv2.cvtColor(
                    image_rec_opt[0], cv2.COLOR_RGB2BGR)
                result_dir = os.path.join(
                    null_text_dir, f"{image_name}_rec_opt.png")
                cv2.imwrite(result_dir, image_rec_opt)
                #
            return None
        #######################################################################

        if self.is_makeup:
            latent_holder = latent.clone()
            with torch.no_grad():
                for i in range(self.start_step, self.diffusion_steps):
                    t = self.diff_model.scheduler.timesteps[i]
                    latent_holder = self.diffusion_step(latent_holder,
                                                        all_uncond_embs[i -
                                                                        self.start_step],
                                                        t, True)
            fast_render_image = self.diff_model.vae.decode(
                1 / 0.18215 * latent_holder)['sample']

        #
        self.set_attention_control(controller)

        null_context_guidance = [[torch.cat([all_uncond_embs[i]] * 4)]
                        for i in range(len(all_uncond_embs))]
        null_context_guidance = [torch.cat(i) for i in null_context_guidance]

        init_latent = latent.clone()
        latent.requires_grad_(True)
        optimizer = optim.AdamW([latent], lr=1e-2)

        for _, _ in enumerate(tqdm(range(self.prot_steps))):
            controller.loss = 0
            controller.reset()

            latents = torch.cat([init_latent, latent])
            for i in range(self.start_step, self.diffusion_steps):
                t = self.diff_model.scheduler.timesteps[i]
                latents = self.diffusion_step(latents,
                                              null_context_guidance[i -
                                                           self.start_step],
                                              t)

            out_image = self.diff_model.vae.decode(
                1 / 0.18215 * latents)['sample'][1:]
            #
            if self.MTCNN_cropping:
                out_image_hold = out_image[:, :, round(bb_src1[1]):round(
                    bb_src1[3]), round(bb_src1[0]):round(bb_src1[2])]
                _, _, h, w = out_image_hold.shape
                if h != 0 and w != 0:
                    out_image = out_image_hold
            #
            if self.is_makeup:
                output_image_aug = torch.cat(
                    [self.augment(out_image) for i in range(1)], dim=0)
                clip_loss = self.nce_loss(fast_render_image,
                                          self.source_text,
                                          output_image_aug,
                                          self.makeup_prompt).sum()
                clip_loss = clip_loss * self.makeup_weight
            #
            output_embeddings = self.get_FR_embeddings(out_image)
            adv_loss = self.cosine_loss(
                output_embeddings, target_embeddings, source_embeddings) * self.adv_optim_weight
            self_attn_loss = controller.loss
            loss = adv_loss + self_attn_loss
            if self.is_makeup:
                loss += clip_loss
            #
            print()
            print('adv_loss: ', adv_loss.item())
            print('self_attn_loss: ', self_attn_loss.item())
            if self.is_makeup:
                print('clip_loss: ', clip_loss.item())
            print('loss: ', loss.item())
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            controller.loss = 0
            controller.reset()
            #
            latents = torch.cat([init_latent, latent])
            #
            for i in range(self.start_step, self.diffusion_steps):
                t = self.diff_model.scheduler.timesteps[i]
                latents = self.diffusion_step(latents,
                                              null_context_guidance[i -
                                                           self.start_step],
                                              t)
        #
        self.reset_attention_control()
        return latents.detach()

    def run(self):
        timer = MyTimer()
        time_list = []
        result_dir = self.protected_image_dir + '/' + \
            self.test_model_name[0] + '/' + self.target_choice
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        target_image, _ = get_target_test_images(
            self.target_choice, self.device, self.MTCNN_cropping)
        with torch.no_grad():
            target_embeddings = self.get_FR_embeddings(target_image)

        for i, (fname, image) in enumerate(self.dataloader):
            image_name = fname[0]
            image = image.to(self.device)
            #
            bb_src1 = None
            if self.MTCNN_cropping:
                path = self.source_dir + '/' + image_name + '.png'
                img = Image.open(path)
                if img.size[0] != self.image_size:
                    img = img.resize((self.image_size, self.image_size))
                bb_src1 = alignment(img)
            #
            controller = AttentionControlEdit(num_steps=self.diffusion_steps,
                                              self_replace_steps=1.0)
            #
            if self.comparison_null_text:
                null_text_dir = os.path.join(
                    self.protected_image_dir, "null_text_opt")
                os.makedirs(null_text_dir, exist_ok=True)
            else:
                null_text_dir = None
            #
            if self.is_obfuscation:
                image_hold = image.clone()
                if self.MTCNN_cropping:
                    out_image_hold = image_hold[:, :, round(bb_src1[1]):round(
                        bb_src1[3]), round(bb_src1[0]):round(bb_src1[2])]
                    _, _, h, w = out_image_hold.shape
                    if h != 0 and w != 0:
                        image_hold = out_image_hold
                with torch.no_grad():
                    source_embeddings = self.get_FR_embeddings(image_hold)
            else:
                source_embeddings = None
            #
            timer.tic()
            #
            latents = self.attacker(image,
                                    image_name,
                                    source_embeddings,
                                    target_embeddings,
                                    controller,
                                    null_text_dir,
                                    bb_src1)
            #
            avg_time = timer.toc()
            time_list.append(avg_time)

            if latents is not None:
                self.visualize(image_name, image, latents, controller)
        #
        print('Time: ', round(np.average(time_list), 2))
        result_fn = os.path.join(result_dir, "time.txt")
        with open(result_fn, 'a') as f:
            f.write(f"Time: {round(np.average(time_list),2)}\n")
        f.close()
