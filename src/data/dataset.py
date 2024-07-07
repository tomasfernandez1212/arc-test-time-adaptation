import torch
from torch.utils.data import Dataset
from enum import Enum 
import os
from src.data.schema import Task
from src.data.load import load_and_validate_data

class Split(Enum):
    TRAIN = "training"
    EVAL = "evaluation"

class ARCDataset(Dataset):
    def __init__(self, split: Split, data_dir: str):
        self.split = split
        self.data_dir = data_dir
        self.split_dir = os.path.join(self.data_dir, self.split.value)
        
    def load_data(self):
        self.filenames = os.listdir(self.split_dir)
        self.data = []
        for filename in self.filenames:
            task = load_and_validate_data(os.path.join(self.split_dir, filename))