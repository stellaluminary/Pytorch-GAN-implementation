import torch.utils.data as data
import torch
from data.Unalign import UnalignedDataset

def create_dataset(opt):
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.dataset = UnalignedDataset(opt)
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





