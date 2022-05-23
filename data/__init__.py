import torch.utils.data as data
import torch
import torchvision
from data.Unalign import UnalignedDataset
from data.Single import SingleDataset

def create_dataset(opt):
    if opt['Model_Param']['dataset_name'] == 'imagenet':
        data_loader = ImageNetDataset(opt)
    else:
        data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.opt_param = self.opt['Model_Param']
        if self.opt_param['model_name'] in ['cyclegan']:
            self.dataset = UnalignedDataset(opt)
        elif self.opt_param['model_name'] in ['dcgan', 'wgan', 'wgan-gp', 'sngan', 'sagan']:
            self.dataset = SingleDataset(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)

        if self.opt['Setting']['phase'] == 'train':
            shuffle = True
        else:
            shuffle = False

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt['Data_Param']['batch_size'],
            shuffle=shuffle,
            num_workers=int(opt['Data_Param']['num_threads']))

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data

class ImageNetDataset():
    def __init__(self, opt):
        self.opt = opt
        self.opt_param = self.opt['Model_Param']

        self.dataset = torchvision.datasets.ImageNet(opt['Path']['Data_train'])

        print("dataset [%s] was created" % type(self.dataset).__name__)

        if self.opt['Setting']['phase'] == 'train':
            shuffle = True
        else:
            shuffle = False

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt['Data_Param']['batch_size'],
            shuffle=shuffle,
            num_workers=int(opt['Data_Param']['num_threads']))

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data

