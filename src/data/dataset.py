import torch
from src.data.tokenizer import TaskEncoder
from torch.utils.data import Dataset
from enum import Enum 
import os
from src.data.load import load_and_validate_data

class Split(Enum):
    TRAIN = "training"
    EVAL = "evaluation"

class ARCDataset(Dataset):
    def __init__(self, split: Split = Split.TRAIN, data_dir: str = "../ARC-AGI/data", task_encoder: TaskEncoder = TaskEncoder()):
        self.split = split
        self.data_dir = data_dir
        self.split_dir = os.path.join(self.data_dir, self.split.value)
        self.filenames = os.listdir(self.split_dir)
        self.task_encoder = task_encoder

    def __getitem__(self, index):
        filename = self.filenames[index]
        task = load_and_validate_data(os.path.join(self.split_dir, filename))
        
        encoder_seq, decoder_seq = self.task_encoder.encode_task(task)
        
        src = torch.tensor(encoder_seq, dtype=torch.long)
        tgt = torch.tensor(decoder_seq, dtype=torch.long)
        
        return src, tgt

    def __len__(self):
        return len(self.filenames)
