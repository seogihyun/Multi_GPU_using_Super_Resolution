import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            image_lr = f['lr'][idx] / 255.
            image_lr = image_lr.transpose([2,0,1])
            image_hr = f['hr'][idx] / 255.
            image_hr = image_hr.transpose([2,0,1])
            return image_lr, image_hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            image_lr = f['lr'][str(idx)][:, :] / 255.
            image_lr = image_lr.transpose([2,0,1])
            image_hr = f['hr'][str(idx)][:, :] / 255.
            image_hr = image_hr.transpose([2,0,1])
            return image_lr, image_hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
