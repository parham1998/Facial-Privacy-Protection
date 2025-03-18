# =============================================================================
# Import required libraries
# =============================================================================
import os
import glob
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import *


def attack_local_models(args, protection):
    # Load test model
    test_model = load_FR_models(args, args.test_model_name)
    # False acceptance rate (FAR) is set to 0.01
    th_dict = {'ir152': (0.094632, 0.166788, 0.227922),
               'irse50': (0.144840, 0.241045, 0.312703),
               'facenet': (0.256587, 0.409131, 0.591191),
               'mobile_face': (0.183635, 0.301611, 0.380878)}

    result_dir = args.protected_image_dir + '/' + \
        args.test_model_name[0] + '/' + args.target_choice
    result_fn = os.path.join(result_dir, "result.txt")

    print('Protection:', protection)
    with open(result_fn, 'a') as f:
        f.write(f"Protection: {protection}\n")
    f.close()

    combined_dir = os.path.join(result_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    size = test_model[args.test_model_name[0]][0]
    model = test_model[args.test_model_name[0]][1]
    #
    _, test_image = get_target_test_images(args.target_choice,
                                           args.device,
                                           args.MTCNN_cropping)
    test_embbeding = model(
        (F.interpolate(test_image, size=size, mode='bilinear')))

    FAR01 = 0
    FAR001 = 0
    FAR0001 = 0
    total = 0
    if protection:
        for img_path in glob.glob(os.path.join(result_dir, "*.png")):
            protectec_image = read_img(img_path, 0.5, 0.5, args.device)
            if args.MTCNN_cropping:
                bb_src1 = alignment(Image.open(img_path).convert("RGB"))
                protected_image_hold = protectec_image[:, :, round(bb_src1[1]):round(
                    bb_src1[3]), round(bb_src1[0]):round(bb_src1[2])]
                #
                _, _, h, w = protected_image_hold.shape
                if h != 0 and w != 0:
                    protectec_image = protected_image_hold
            ae_embbeding = model.forward(
                (F.interpolate(protectec_image, size=size, mode='bilinear')))
            cos_simi = torch.cosine_similarity(ae_embbeding, test_embbeding)
            if cos_simi.item() > th_dict[args.test_model_name[0]][0]:
                FAR01 += 1
            if cos_simi.item() > th_dict[args.test_model_name[0]][1]:
                FAR001 += 1
            if cos_simi.item() > th_dict[args.test_model_name[0]][2]:
                FAR0001 += 1
            total += 1

            # Combine the clean and protected images for visualization
            adv_img = cv2.imread(img_path)
            fn = img_path.split("\\")[-1].split(".")[0] + ".png"
            clean_img = cv2.imread(os.path.join(args.source_dir, fn))
            if clean_img.shape[0] != args.image_size:
                clean_img = cv2.resize(clean_img, (args.image_size, args.image_size),
                                       interpolation=cv2.INTER_LANCZOS4)
            #
            combined_img = np.concatenate([clean_img, adv_img], 1)
            combined_fn = f"{fn.split('.')[0]}_cos_simi_{cos_simi.item():.4f}.png"
            cv2.imwrite(os.path.join(combined_dir, combined_fn), combined_img)
    else:
        for img in tqdm(os.listdir(args.source_dir), desc=args.test_model_name[0] + ' clean'):
            protectec_image = read_img(os.path.join(
                args.source_dir, img), 0.5, 0.5, args.device)
            if args.MTCNN_cropping:
                bb_src1 = alignment(Image.open(os.path.join(
                    args.source_dir, img)))
                protected_image_hold = protectec_image[:, :, round(bb_src1[1]):round(
                    bb_src1[3]), round(bb_src1[0]):round(bb_src1[2])]
                #
                _, _, h, w = protected_image_hold.shape
                if h != 0 and w != 0:
                    protectec_image = protected_image_hold
            ae_embbeding = model.forward(
                (F.interpolate(protectec_image, size=size, mode='bilinear')))
            cos_simi = torch.cosine_similarity(ae_embbeding, test_embbeding)
            if cos_simi.item() > th_dict[args.test_model_name[0]][0]:
                FAR01 += 1
            if cos_simi.item() > th_dict[args.test_model_name[0]][1]:
                FAR001 += 1
            if cos_simi.item() > th_dict[args.test_model_name[0]][2]:
                FAR0001 += 1
            total += 1

    result_str = f"{args.test_model_name[0]} PSR in FAR@0.1: {FAR01/total:.4f}, PSR in FAR@0.01: {FAR001/total:.4f}, PSR in FAR@0.001: {FAR0001/total:.4f}\n"
    print(result_str)
    with open(result_fn, 'a') as f:
        f.write(result_str)
    f.close()