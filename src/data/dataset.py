import torch
from src.data.tokenizer import TaskEncoder
from torch.utils.data import Dataset
from enum import Enum 
import os
from src.data.load import load_and_validate_data
from src.data.synthetic import SyntheticTaskGenerator
from typing import Tuple
class Split(Enum):
    TRAIN = "training"
    EVAL = "evaluation"
    SYNTHETIC_MIRRORED = "synthetic_mirrored"
    SYNTHETIC_NON_MIRRORED = "synthetic_non_mirrored"

class ARCDataset(Dataset):
    def __init__(self, split: Split = Split.TRAIN, data_dir: str = "../ARC-AGI/data", task_encoder: TaskEncoder = TaskEncoder()):
        self.split = split
        self.data_dir = data_dir
        self.task_encoder = task_encoder
        
        # Initialize Synthetic Task Generator
        if self.split == Split.SYNTHETIC_MIRRORED:
            self.synthetic_dataset = SyntheticTaskGenerator(mirrored_pairs=True)
            self.num_tasks = 400
        elif self.split == Split.SYNTHETIC_NON_MIRRORED:
            self.synthetic_dataset = SyntheticTaskGenerator(mirrored_pairs=False)
            self.num_tasks = 400
        elif self.split == Split.TRAIN or self.split == Split.EVAL:
            self.split_dir = os.path.join(self.data_dir, self.split.value)
            self.filenames = os.listdir(self.split_dir)
            self.num_tasks = len(self.filenames)
        else: 
            raise ValueError(f"Invalid split: {self.split} for ARCDataset.")

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, int]:

        # Depending on the split, we either return a synthetic task or a real task 
        if self.split == Split.SYNTHETIC_MIRRORED or self.split == Split.SYNTHETIC_NON_MIRRORED:
            task = self.synthetic_dataset.generate_task()
        else:
            filename = self.filenames[index]
            task = load_and_validate_data(os.path.join(self.split_dir, filename))
        
        # Encode the task
        encoded_sequence, attention_mask, start_of_test_output_grid = self.task_encoder.encode_task(task)

        return encoded_sequence, attention_mask, start_of_test_output_grid

    def __len__(self):
        return self.num_tasks