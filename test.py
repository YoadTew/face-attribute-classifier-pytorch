import argparse

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torchfunc
import torchvision.transforms as transforms
import numpy as np
import random
import os
import cv2
import shutil

from models.resnet import resnet50
from data.data_manager import get_test_loader

def get_args():
    parser = argparse.ArgumentParser(description="training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument('--pretrained', action='store_true', help='Load pretrain model')
    parser.add_argument('--weights', default='', type=str,
                        help='path to latest checkpoint (default: none)')

    parser.add_argument("--img_dir", default='D:/Work/GAN/FFHQ/images', help="Images dir path")
    parser.add_argument("--test_json_path", default='labels/test.json', help="Test json path")
    parser.add_argument("--display_output_dir", default='outputs_display/server_v1', help="Image output display dir path")

    return parser.parse_args()

class Tester:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        model = resnet50(pretrained=args.pretrained, num_classes=1)
        self.model = model.to(device)

        self.test_loader = get_test_loader(args)

        if args.weights and os.path.isfile(args.weights):
            print(f'Loading checkpoint {args.weights}')

            checkpoint = torch.load(args.weights)
            self.model.load_state_dict(checkpoint['state_dict'])

            print(f'Loaded checkpoint {args.weights}')

        cudnn.benchmark = True

    def do_testing(self):
        self.model.eval()
        class_correct = 0

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # forward
                outputs = self.model(inputs)

                cls_pred = (outputs > 0.5).type(torch.int)
                labels = (targets > 0.5).type(torch.int)

                class_correct += torch.sum(cls_pred == labels)

        total = len(self.test_loader.dataset)
        class_acc = float(class_correct) / total
        print(f'Test Accuracy: {class_acc}')

    def display_outputs(self):
        self.model.eval()
        with torch.no_grad():
            dataiter = iter(self.test_loader)
            images, labels = dataiter.next()

            inputs, targets = images.to(self.device), labels.to(self.device)
            sigmoid = nn.Sigmoid()

            # forward
            outputs = self.model(inputs)
            outputs = sigmoid(outputs)

            invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                           transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                                std=[1., 1., 1.]),
                                           ])

            for i, (img, output) in enumerate(zip(images, outputs)):
                img = invTrans(img) * 255
                img = img.numpy()
                img = np.transpose(img, (1, 2, 0))
                img = img[:, :, ::-1]

                cv2.imwrite(f'{self.args.display_output_dir}/output_{i}_score_{output.item()}.png', img)

def main():
    args = get_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tester = Tester(args, device)
    # tester.do_testing()
    tester.display_outputs()


if __name__ == "__main__":
    torchfunc.cuda.reset()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    main()