# =============================================================================
# Import required libraries
# =============================================================================
import os
import glob
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import *


def attack_local_models_obfuscation(args, protection):
    # Load test model
    test_model = load_FR_models(args, args.test_model_name)
    # False acceptance rate (FAR) is set to 0.01
    th_dict = {'ir152': 0.166788,
               'irse50': 0.241045,
               'facenet': 0.409131,
               'mobile_face': 0.301611}

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

    FAR001 = 0
    total = 0
    if protection:
        for img_path in glob.glob(os.path.join(result_dir, "*.png")):
            protectec_image = read_img(img_path, 0.5, 0.5, args.device)
            #
            img = os.path.basename(img_path).replace("train_", "test_")
            test_image = read_img(os.path.join(
                args.test_dir, img), 0.5, 0.5, args.device)
            #
            pro_embbeding = model.forward(
                (F.interpolate(protectec_image, size=size, mode='bilinear')))
            tes_embbeding = model.forward(
                (F.interpolate(test_image, size=size, mode='bilinear')))
            cos_simi = torch.cosine_similarity(pro_embbeding, tes_embbeding)
            if cos_simi.item() < th_dict[args.test_model_name[0]]:
                FAR001 += 1
            total += 1

            # Combine the protected and test images for visualization
            adv_img = cv2.imread(img_path)
            test_img = cv2.imread(os.path.join(args.test_dir, img))
            if test_img.shape[0] != args.image_size:
                test_img = cv2.resize(test_img, (args.image_size, args.image_size),
                                       interpolation=cv2.INTER_LANCZOS4)
            #
            combined_img = np.concatenate([adv_img, test_img], 1)
            combined_fn = f"{img.split('.')[0]}_cos_simi_{cos_simi.item():.4f}.png"
            cv2.imwrite(os.path.join(combined_dir, combined_fn), combined_img)
    else:
        for img in tqdm(os.listdir(args.source_dir), desc=args.test_model_name[0] + ' clean'):
            protectec_image = read_img(os.path.join(
                args.source_dir, img), 0.5, 0.5, args.device)
            test_image = read_img(os.path.join(
                args.test_dir, img.replace("train_", "test_")), 0.5, 0.5, args.device)
            pro_embbeding = model.forward(
                (F.interpolate(protectec_image, size=size, mode='bilinear')))
            tes_embbeding = model.forward(
                (F.interpolate(test_image, size=size, mode='bilinear')))
            cos_simi = torch.cosine_similarity(pro_embbeding, tes_embbeding)
            if cos_simi.item() < th_dict[args.test_model_name[0]]:
                FAR001 += 1
            total += 1

    result_str = f"{args.test_model_name[0]} PSR in FAR@0.01: {FAR001/total:.4f} \n"
    print(result_str)
    with open(result_fn, 'a') as f:
        f.write(result_str)
    f.close()
