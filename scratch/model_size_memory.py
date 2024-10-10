import sys 
import os 
if os.path.basename(os.getcwd())!="arc-test-time-adaptation":
   sys.path.append('../') # Used when running directly
else:
   sys.path.append('./') # Used when running debugger

"""
This script is used to estimate the memory usage of the model. It is not used in the training or evaluation of the model. 
"""
import os
import torch
from src.data.context import MAX_TOKENS_PER_TASK
from src.model.core import Transformer
from src.data.dataset import ARCDataset, Split
import src.model.config as config
from torch.utils.data import DataLoader


# Model parameters from config
DEVICE = config.DEVICE
VOCAB_SIZE = config.VOCAB_SIZE
D_MODEL = config.D_MODEL
NUM_HEADS = config.NUM_HEADS
NUM_LAYERS = config.NUM_LAYERS
D_FF = config.D_FF
DROPOUT = config.DROPOUT
BATCH_SIZE = config.BATCH_SIZE

# Initialize the model
model = Transformer(VOCAB_SIZE, MAX_TOKENS_PER_TASK, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT, DEVICE)
model.to(DEVICE)

# Load the Dataset
train_dataset = ARCDataset(split=Split.SYNTHETIC_MIRRORED)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Sample batch
sequence, attention_mask, start_of_test_output_grid = next(iter(train_loader))
sequence = sequence.to(DEVICE)
attention_mask = attention_mask.to(DEVICE)
print(sequence.shape)

# Prepare
decoder_input = sequence[:, :-1]
attention_mask = attention_mask[:, :-1, :-1]  # Adjust mask to match decoder input

# Run inference
model.eval()
decoder_output = model(decoder_input, attention_mask)
print(decoder_output.shape)

# Print maximum memory used
if torch.cuda.is_available():
    max_memory = torch.cuda.max_memory_allocated(DEVICE)
    print(f"Maximum memory used: {max_memory / (1024 ** 3)} GB")
else:
    print("CUDA is not available. Can't estimate memory usage.")