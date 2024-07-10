import torch
from src.data.dataset import ARCDataset, Split
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model.core import Transformer
from src.data.tokenizer import Token, Encoding
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Model Parameters
SRC_POSSIBLE_TOKENS = len(Token)  # Number of Colors + Other Tokens
TGT_POSSIBLE_TOKENS = len(Token)  # Number of Colors + Other Tokens
D_MODEL = 6  # Embeddings Dimension - Should be divisible by 2 for positional encoding.
NUM_HEADS = 3  # Number of attention heads. D_MODEL must be divisible by NUM_HEADS.
NUM_LAYERS = 6  # Number of encoder/decoder layers.
D_FF = 4 * D_MODEL  # Feed Forward Hidden Layer Dimensionality

# Define Context Window Size
MAX_PIXELS_IN_ROW = 2  # Actual is 30
MAX_PIXELS_IN_COL = 2  # Actual is 30
MAX_PAIRS_IN_SAMPLE = 3  # Actual is 10
MAX_TOKENS_PER_ROW = 2 + MAX_PIXELS_IN_ROW  # Start and End of Row
MAX_TOKENS_PER_GRID = 2 + MAX_TOKENS_PER_ROW * MAX_PIXELS_IN_COL  # Start and End of Grid
MAX_TOKENS_PER_SAMPLE = 2 + MAX_TOKENS_PER_GRID * MAX_PAIRS_IN_SAMPLE  # Actual is 9622

# Define Training Parameters
BATCH_SIZE = 4  # Batch size
NUM_EPOCHS = 10  # Number of epochs
LEARNING_RATE = 1e-4
DROPOUT = 0.1  # Dropout probability

# Initialize Model, Loss Function, and Optimizer
model = Transformer(SRC_POSSIBLE_TOKENS, TGT_POSSIBLE_TOKENS, MAX_TOKENS_PER_SAMPLE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT)
model.to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=Encoding.PAD.value)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load the dataset and data loader
train_dataset = ARCDataset(split=Split.TRAIN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        
        # Prepare target input and output
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Forward pass
        output = model(src, tgt_input)
        
        # Reshape output and target for loss computation
        output = output.view(-1, TGT_POSSIBLE_TOKENS)
        tgt_output = tgt_output.view(-1)

        # Compute loss
        loss = criterion(output, tgt_output)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

print("Training complete.")