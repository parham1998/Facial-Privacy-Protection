# =============================================================================
# Import required libraries
# =============================================================================
import os
import random
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler

from dataset import ImageDataset
from adversarial_optimization import Adversarial_Opt
from tests import attack_local_models
from test_obfs import attack_local_models_obfuscation


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--source_dir',
                        default="",
                        type=str,
                        help="source images folder path for impersonation and obfuscation")
    parser.add_argument('--test_dir',
                        default="",
                        type=str,
                        help="test images folder path for obfuscation")
    parser.add_argument('--protected_image_dir',
                        default="results",
                        type=str)
    parser.add_argument('--comparison_null_text',
                        default=False,
                        type=bool)

    parser.add_argument('--target_choice',
                        default='2',
                        type=str,
                        help='Choice of target identity, as in AMT-GAN. We use 4 target identities provided by AMT-GAN. We also add a synthesized target image for obfuscation')
    parser.add_argument("--test_model_name",
                        default=['mobile_face'])
    parser.add_argument("--surrogate_model_names",
                        default=['ir152', 'facenet', 'irse50'])

    # When applying makeup to the image
    parser.add_argument('--is_makeup',
                        default=False,
                        type=bool)
    parser.add_argument('--source_text',
                        default='face',
                        type=str)
    parser.add_argument('--makeup_prompt',
                        default='red lipstick',
                        type=str)

    parser.add_argument('--MTCNN_cropping',
                        default=True,
                        type=bool)

    parser.add_argument('--is_obfuscation',
                        default=False,
                        type=bool)

    parser.add_argument('--image_size',
                        default=256,
                        type=int)
    parser.add_argument('--prot_steps',
                        default=30,
                        type=int)
    parser.add_argument('--diffusion_steps',
                        default=20,
                        type=int)
    parser.add_argument('--start_step',
                        default=17,
                        type=int,
                        help='Which DDIM step to start the protection (20 - 17 = 3)')
    parser.add_argument('--null_optimization_steps',
                        default=20,
                        type=int)

    parser.add_argument('--adv_optim_weight',
                        default=0.003,
                        type=float)
    parser.add_argument('--makeup_weight',
                        default=0,
                        type=float)

    args = parser.parse_args()
    return args


def initialize_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    #
    initialize_seed(seed=10)
    #
    args = parse_args()
    #
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the stable diffusion pretrained parameters
    diff_model = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-base').to(args.device)
    diff_model.scheduler = DDIMScheduler.from_config(
        diff_model.scheduler.config)

    # Load the dataset
    dataset = ImageDataset(
        args.source_dir,
        transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5],
                                                 [0.5, 0.5, 0.5])]
                           )
    )
    args.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    adversarial_opt = Adversarial_Opt(args, diff_model)
    adversarial_opt.run()

    if not args.comparison_null_text:
        if not args.is_obfuscation:
            attack_local_models(args, protection=False)
            attack_local_models(args, protection=True)
        else:
            attack_local_models_obfuscation(args, protection=False)
            attack_local_models_obfuscation(args, protection=True)
