import torch.utils.data as data
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def tensor_transforms(opt):
    tf = opt['Transforms']

    tf_list = []

    if tf['crop']:
        tf_list.append(transforms.CenterCrop((tf['crop'], tf['crop'])))
    if tf['resize']:
        tf_list.append(transforms.Resize((tf['resize'], tf['resize']), Image.BICUBIC))
    if tf['flip']:
        tf_list.append(transforms.RandomHorizontalFlip())

    tf_list.append(transforms.ToTensor())
    tf_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(tf_list)

class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.phase = opt['Setting']['phase']

        # ex) train : celebA, MNIST
        self.A_path = sorted(make_dataset(opt['Path']['Data_' + self.phase]))
        self.A_size = len(self.A_path)
        self.transform = tensor_transforms(opt)

        print(self.phase + ' dataset path is ' + opt['Path']['Data_' + self.phase])

    def __getitem__(self, idx):
        A_path = self.A_path[idx % self.A_size]

        A = Image.open(A_path).convert('RGB')

        if self.transform:
            A = self.transform(A)

        noise = torch.randn(self.opt['Model_Param']['nz'], 1, 1)

        return {'A': A, 'A_paths': A_path, 'noise': noise}

    def __len__(self):
        return self.A_size
