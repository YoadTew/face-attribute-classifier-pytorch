import torch.utils.data as data
from torchvision import datasets
from torchvision import transforms

from PIL import Image
import numpy as np
import torch
import glob
import json

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #     return accimage_loader(path)
    # else:
    return pil_loader(path)

class FFHQ(data.Dataset):
    def __init__(self, json_path, img_dir, transform=None, target_transform=None, loader=default_loader):

        self.images = []
        self.targets = []

        with open(json_path) as json_file:
            all_data = json.load(json_file)

        for json_data in all_data:

            img_id = json_data['id']
            img_path = f'{img_dir}/{img_id}.png'
            self.images.append(img_path)

            smile_label = json_data['faceAttributes']['smile']
            self.targets.append(smile_label)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.images[index]
        sample = self.loader(path)
        target = self.targets[index]
        # target = torch.LongTensor(target)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.images)