import torch
from src.data.dataset import ARCDataset, Split
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model.core import Transformer
from src.data.tokenizer import Token, Encoding
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Model Parameters
src_possible_tokens = len(Token) # Number of Colors + Other Tokens
tgt_possible_tokens = len(Token) # Number of Colors + Other Tokens
d_model = 6 # Embeddings Dimension - Should be divisible by 2 for positional encoding. Using LLMs as a point of comparison, this should be less than number of colors/tokens.
num_heads = 3 # Number of attention heads. d_model must be divisible by num_heads. 
num_layers = 6 # Number of encoder/decoder layers. 
d_ff = 4*d_model # Feed Forward Hidden Layer Dimensionality 

# Define Context Window Size 
max_pixels_in_row = 2 # Actual is 30 
max_pixels_in_col = 2 # Actual is 30
max_pairs_in_sample = 3 # Actual is 10
max_tokens_per_row = 2+max_pixels_in_row # Start and End of Row
max_tokens_per_grid = 2+max_tokens_per_row*max_pixels_in_col # Start and End of Grid
max_tokens_per_sample = 2+max_tokens_per_grid*max_pairs_in_sample # Actual is 9622

# Define Training Parameters
batch_size = 4 # Batch size
num_epochs = 10 # Number of epochs
learning_rate = 1e-4
dropout = 0.1 # Dropout probability

# Initialize Model, Loss Function, and Optimizer
model = Transformer(src_possible_tokens, tgt_possible_tokens, max_tokens_per_sample, d_model, num_heads, num_layers, d_ff, dropout)
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=Encoding.PAD.value)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load the dataset and data loader
train_dataset = ARCDataset(split=Split.TRAIN)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        src, tgt = src.to(device), tgt.to(device)
        
        # Prepare target input and output
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Forward pass
        output = model(src, tgt_input)
        
        # Reshape output and target for loss computation
        output = output.view(-1, tgt_possible_tokens)
        tgt_output = tgt_output.view(-1)

        # Compute loss
        loss = criterion(output, tgt_output)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training complete.")