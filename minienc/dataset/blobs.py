import copy
import torch
import numpy as np
from sklearn.datasets import make_blobs
from torch.utils.data import Dataset


class BlobsDataset(Dataset):
    def __init__(
        self,
        n_samples:int,
        n_features:int,
        centers:int,
        random_state:int=42):
        data, labels = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            random_state=random_state
        )
        target:np.ndarray = copy.deepcopy(data)
        
        self.data: torch.tensor = torch.tensor(data)
        self.data.to(torch.float64)
        self.data.to(torch.device("cpu"))
        self.data.requires_grad_(True)
        
        self.target: torch.tensor = torch.tensor(target)
        self.target.to(torch.float64)
        self.data.to(torch.device("cpu"))
        self.data.requires_grad_(True)
    
    
    def __len__(self):
        return self.data.size()[0]
    
    
    def __getitem__(self, index: int):
        return self.data[index], self.target[index]