import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class FLDataset(Dataset):

    def __init__(self, path_Xdataset: str, path_ydataset: str):
        self.X = torch.from_numpy(np.load(path_Xdataset))/255
        self.y = torch.from_numpy(np.load(path_ydataset))
        
        self.n_samples = self.X[0]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index: int):
        return self.X[index], self.y[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

def get_dataloader(path_Xdataset: str, path_ydataset: str, batch_size: int = 32, shuffle = True):
    fl_dataset = FLDataset(path_Xdataset=path_Xdataset, path_ydataset=path_ydataset)
    dataloader = DataLoader(dataset=fl_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader