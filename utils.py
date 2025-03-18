# =============================================================================
# Import required libraries
# =============================================================================
import time
import cv2

import torch
from facenet_pytorch import MTCNN

from assets.face_recognition_models import irse, ir152, facenet


class MyTimer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.


mtcnn = MTCNN(image_size=256,  # Size of the input image
              margin=0,
              post_process=False,
              select_largest=False,
              device='cuda')


def alignment(image):
    boxes, probs = mtcnn.detect(image)
    return boxes[0]


def load_FR_models(args, model_names):
    FR_models = {}
    for model_name in model_names:
        if model_name == 'ir152':
            FR_models[model_name] = []
            FR_models[model_name].append((112, 112))
            fr_model = ir152.IR_152((112, 112))
            fr_model.load_state_dict(torch.load(
                'assets/face_recognition_models/ir152.pth'))
            fr_model.to(args.device)
            fr_model.eval()
            FR_models[model_name].append(fr_model)
        if model_name == 'irse50':
            FR_models[model_name] = []
            FR_models[model_name].append((112, 112))
            fr_model = irse.Backbone(50, 0.6, 'ir_se')
            fr_model.load_state_dict(torch.load(
                'assets/face_recognition_models/irse50.pth'))
            fr_model.to(args.device)
            fr_model.eval()
            FR_models[model_name].append(fr_model)
        if model_name == 'facenet':
            FR_models[model_name] = []
            FR_models[model_name].append((160, 160))
            fr_model = facenet.InceptionResnetV1(
                num_classes=8631, device=args.device)
            fr_model.load_state_dict(torch.load(
                'assets/face_recognition_models/facenet.pth'))
            fr_model.to(args.device)
            fr_model.eval()
            FR_models[model_name].append(fr_model)
        if model_name == 'mobile_face':
            FR_models[model_name] = []
            FR_models[model_name].append((112, 112))
            fr_model = irse.MobileFaceNet(512)
            fr_model.load_state_dict(torch.load(
                'assets/face_recognition_models/mobile_face.pth'))
            fr_model.to(args.device)
            fr_model.eval()
            FR_models[model_name].append(fr_model)
    return FR_models


def preprocess(im, mean, std, device):
    if len(im.size()) == 3:
        im = im.transpose(0, 2).transpose(1, 2).unsqueeze(0)
    elif len(im.size()) == 4:
        im = im.transpose(1, 3).transpose(2, 3)
    mean = torch.tensor(mean).to(device)
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std).to(device)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    im = (im - mean) / std
    return im


def read_img(data_dir, mean, std, device):
    img = cv2.imread(data_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    img = torch.from_numpy(img).to(torch.float32).to(device)
    img = preprocess(img, mean, std, device)
    return img


def get_target_test_images(target_choice, device, MTCNN_cropping):
    # Impersonation
    if target_choice == '1':
        target_image = read_img(
            'assets/target_images/005869.jpg', 0.5, 0.5, device)
        test_image = read_img(
            'assets/test_images/008793.jpg', 0.5, 0.5, device)
        if MTCNN_cropping:
            target_image = target_image[:, :, 168:912, 205:765]
            test_image = test_image[:, :, 145:920, 202:775]
    elif target_choice == '2':
        target_image = read_img(
            'assets/target_images/085807.jpg', 0.5, 0.5, device)
        test_image = read_img(
            'assets/test_images/047073.jpg', 0.5, 0.5, device)
        if MTCNN_cropping:
            target_image = target_image[:, :, 187:891, 244:764]
            test_image = test_image[:, :, 234:905, 266:791]
    elif target_choice == '3':
        target_image = read_img(
            'assets/target_images/116481.jpg', 0.5, 0.5, device)
        test_image = read_img(
            'assets/test_images/055622.jpg', 0.5, 0.5, device)
        if MTCNN_cropping:
            target_image = target_image[:, :, 214:955, 188:773]
            test_image = test_image[:, :, 185:931, 198:780]
    elif target_choice == '4':
        target_image = read_img(
            'assets/target_images/169284.jpg', 0.5, 0.5, device)
        test_image = read_img(
            'assets/test_images/166607.jpg', 0.5, 0.5, device)
        if MTCNN_cropping:
            target_image = target_image[:, :, 173:925, 233:792]
            test_image = test_image[:, :, 172:917, 219:779]
    # Obfuscation
    elif target_choice == '5':
        target_image = read_img(
            'assets/obfs_target_images/0808002.png', 0.5, 0.5, device)
        test_image = None
    else:
        raise ValueError(
            "Invalid target choice!")
    return target_image, test_image
