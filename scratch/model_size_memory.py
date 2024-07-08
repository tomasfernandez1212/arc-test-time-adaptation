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

src_possible_tokens = 10 # Number of Colors 
tgt_possible_tokens = 10 # Number of Colors 
d_model = 6 # Embeddings Dimension - Should be divisible by 2 for positional encoding. Using LLMs as a point of comparison, this should be less than number of colors/tokens.
num_heads = 3 # Number of attention heads. d_model must be divisible by num_heads. 
num_layers = 6 # Number of encoder/decoder layers. 
d_ff = 4*d_model # Feed Forward Hidden Layer Dimensionality 
dropout = 0.1 # Dropout probability
batch_size = 4 # Batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Example Data Dimensions - 2x2 grids, 2 input pairs, 1 output to solve: 4x2x2+4 = 20
special_tokens = ["<BOS>", "<EOS>", "<PAD>", "<start-of-pair>", "<end-of-pair>", "<start-of-grid>", "<end-of-grid>", "<start-of-row>", "<end-of-row>"]
max_pixels_in_row = 30
max_pixels_in_col = 30
max_pairs_in_sample = 10
max_tokens_per_row = 2+max_pixels_in_row # Start and End of Row
max_tokens_per_grid = 2+max_tokens_per_row*max_pixels_in_col # Start and End of Grid
max_tokens_per_pair = 2+max_tokens_per_grid*2 # Start and End of Pair
max_tokens_per_sample = 2+max_tokens_per_pair*max_pairs_in_sample


# Initialize Model
transformer = Transformer(src_possible_tokens, tgt_possible_tokens, max_tokens_per_sample, d_model, num_heads, num_layers, d_ff, dropout)
transformer.to(device)

# Generate Fake Data 
src_data = torch.randint(1, src_possible_tokens, (batch_size, max_tokens_per_sample)).to(device)  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_possible_tokens, (batch_size, max_tokens_per_sample)).to(device)  # (batch_size, seq_length)

# Run inference
transformer.eval()
output = transformer(src_data, tgt_data)
print(output)

# Print maximum memory used
if torch.cuda.is_available():
    max_memory = torch.cuda.max_memory_allocated(device)
    print(f"Maximum memory used: {max_memory / (1024 ** 3)} GB")
else:
    print("CUDA is not available. Can't estimate memory usage.")