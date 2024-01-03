import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Trainer(object):
    def __init__(
        self,
        model: nn.Module,
        dataset_train: Dataset,
        dataset_vaild: Dataset,
        batch_size: int,
        n_epoches: int):
        pass
    
    