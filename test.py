import argparse
import glob

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torchfunc
import torchvision.transforms as transforms
import numpy as np
import random
import os
import cv2
from data.FFHQ_dataset import pil_loader

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

    def display_from_directiory(self, dir_path):
        self.model.eval()
        with torch.no_grad():
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            img_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

            img_path_list = glob.glob(f'{dir_path}/*.jpg') + glob.glob(f'{dir_path}/*.png')
            img_list = [pil_loader(img_path) for img_path in img_path_list]
            tensor_img_list = [img_transform(img) for img in img_list]

            batch_size = 32
            for i in range(0, len(tensor_img_list), batch_size):
                inputs = torch.stack(tensor_img_list[i:i+batch_size])
                batch_imgs = img_list[i:i+batch_size]

                inputs = inputs.to(self.device)
                sigmoid = nn.Sigmoid()

                # forward
                outputs = self.model(inputs)
                outputs = sigmoid(outputs)

                for i, (img, output) in enumerate(zip(batch_imgs, outputs)):
                    img.save(f'{self.args.display_output_dir}/output_{i}_score_{output.item()}.png')

def main():
    args = get_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tester = Tester(args, device)
    # tester.do_testing()
    # tester.display_outputs()
    tester.display_from_directiory('D:\\Work\\GAN\\stylegan2-pytorch\\sample')

if __name__ == "__main__":
    torchfunc.cuda.reset()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    main()