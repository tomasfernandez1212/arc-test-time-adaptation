import sys 
import os 
if os.path.basename(os.getcwd())!="arc-test-time-adaptation":
   sys.path.append('../') # Used when running directly
else:
   sys.path.append('./') # Used when running debugger

"""
This script is used to estimate the memory usage of the model. It is not used in the training or evaluation of the model. 
"""

from src.model.core import Transformer
import torch

src_vocab_size = 10 # Number of Colors 
tgt_vocab_size = 10 # Number of Colors 
d_model = 6 # Embeddings Dimension - Should be divisible by 2 for positional encoding. Using LLMs as a point of comparison, this should be less than number of colors/tokens.
num_heads = 3 # Number of attention heads. d_model must be divisible by num_heads. 
num_layers = 6 # Number of encoder/decoder layers. 
d_ff = 4*d_model # Feed Forward Hidden Layer Dimensionality 
max_seq_length = 2**12 # Max Grid Size is 30x30. There can be up to 10 input output pairs per task. However, a context window that size takes up a lot of memory. 4096 covers 97% of tasks.
dropout = 0.1 # Dropout probability
batch_size = 4 # Batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Example Data Dimensions - 2x2 grids, 2 input pairs, 1 output to solve: 4x2x2+4 = 20
max_pixels_in_grid = 2*2
max_grids_in_pair = 2
max_pairs_in_sample = 3
max_seq_length = max_pixels_in_grid*max_grids_in_pair*max_pairs_in_sample
grid_starts = torch.tensor([0, 4, 8, 12, 16, 20]).to(device)
grid_lengths = torch.tensor([4, 4, 4, 4, 4, 4]).to(device)
pair_starts = torch.tensor([0, 8, 16]).to(device)
pair_lengths = torch.tensor([8, 8, 8]).to(device)

# Initialize Model
transformer = Transformer(src_vocab_size, tgt_vocab_size, max_pixels_in_grid, max_grids_in_pair, max_pairs_in_sample, d_model, num_heads, num_layers, d_ff, dropout)
transformer.to(device)

# Generate Fake Data 
src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length)).to(device)  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length)).to(device)  # (batch_size, seq_length)

# Run inference
transformer.eval()
output = transformer(src_data, tgt_data, grid_starts, grid_lengths, pair_starts, pair_lengths)
print(output)

# Print maximum memory used
if torch.cuda.is_available():
    max_memory = torch.cuda.max_memory_allocated(device)
    print(f"Maximum memory used: {max_memory / (1024 ** 3)} GB")
else:
    print("CUDA is not available. Can't estimate memory usage.")