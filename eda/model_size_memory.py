from src.model.core import Transformer
import torch

src_vocab_size = 10 # Number of Colors 
tgt_vocab_size = 10 # Number of Colors 
d_model = 8 # Embeddings Dimension - Should be divisible by 2 for positional encoding. Using LLMs as a point of comparison, this should be less than number of colors/tokens.
num_heads = 4 # Number of attention heads. d_model must be divisible by num_heads. 
num_layers = 6 # Number of encoder/decoder layers. 
d_ff = 4*d_model # Feed Forward Hidden Layer Dimensionality 
max_seq_length = 2000 # 30*30*2*10 # Max Grid Size is 30x30 and there can be up to 10 input output pairs per task. 
dropout = 0.1 # Dropout probability
batch_size = 4 # Batch size

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
transformer.to(device)

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length)).to(device)  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length)).to(device)  # (batch_size, seq_length)

# Run inference
transformer.eval()
output = transformer(src_data, tgt_data)
print(output)

# Print maximum memory used
max_memory = torch.cuda.max_memory_allocated(device)
print(f"Maximum memory used: {max_memory / (1024 ** 3)} GB")