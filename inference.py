import os
import torch
from src.data.context import MAX_TOKENS_PER_TASK
from src.model.core import Transformer
from src.data.dataset import ARCDataset, Split
import src.model.config as config


DEVICE = config.DEVICE

# Define the path to the checkpoint you want to load
CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_DIR, "model_epoch_10.pth")  # Update with desired epoch

# Model parameters from config
VOCAB_SIZE = config.VOCAB_SIZE
D_MODEL = config.D_MODEL
NUM_HEADS = config.NUM_HEADS
NUM_LAYERS = config.NUM_LAYERS
D_FF = config.D_FF
DROPOUT = config.DROPOUT

# Initialize the model
model = Transformer(VOCAB_SIZE, MAX_TOKENS_PER_TASK, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT, DEVICE)
model.to(DEVICE)

# Load the checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode

# Send to Device
model.to(DEVICE)

# Load the Dataset
train_dataset = ARCDataset(split=Split.TRAIN)

# Sample a task
encoded_sequence, attention_mask, start_of_test_output_grid = train_dataset[0]

# Perform autoregressive inference
output = model.autoregressive_inference(encoded_sequence, attention_mask, start_of_test_output_grid)

print(output)