import os
import torch
from src.data.dataset import ARCDataset, Split
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model.core import Transformer
from src.data.tokenizer import Token, Encoding
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Model Parameters
VOCAB_SIZE = len(Token)  # Number of Colors + Other Tokens
D_MODEL = 6  # Embeddings Dimension - Should be divisible by 2 for positional encoding.
NUM_HEADS = 3  # Number of attention heads. D_MODEL must be divisible by NUM_HEADS.
NUM_LAYERS = 5  # Number of decoder layers.
D_FF = 1 * D_MODEL  # Feed Forward Hidden Layer Dimensionality

# Define Context Window Size
MAX_PIXELS_IN_ROW = 30  # Actual is 30
MAX_PIXELS_IN_COL = 30  # Actual is 30
MAX_PAIRS_IN_TASK = 10  # Actual is 10

# Calculate Max Tokens per Sample
MAX_TOKENS_PER_ROW = 2 + MAX_PIXELS_IN_ROW  # Start and End of Row
MAX_TOKENS_PER_GRID = 2 + MAX_TOKENS_PER_ROW * MAX_PIXELS_IN_COL  # Start and End of Grid
MAX_TOKENS_PER_PAIR = 2 + 2 * MAX_TOKENS_PER_GRID  # Start and End of Pair, 2 Grids Per Pair
MAX_TOKENS_PER_TASK = 2 + MAX_TOKENS_PER_PAIR * MAX_PAIRS_IN_TASK  # Actual is 19262

# Define Training Parameters
BATCH_SIZE = 1  # Batch size
NUM_EPOCHS = 10  # Number of epochs
LEARNING_RATE = 1e-4
DROPOUT = 0.1  # Dropout probability

# Initialize Model, Loss Function, and Optimizer
model = Transformer(VOCAB_SIZE, MAX_TOKENS_PER_TASK, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT)
model.to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=Encoding.PAD.value)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load the dataset and data loader
train_dataset = ARCDataset(split=Split.SYNTHETIC_MIRRORED)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = ARCDataset(split=Split.SYNTHETIC_MIRRORED)  
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize TensorBoard writer
LOGS_DIR = "logs/pretrain"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
writer = SummaryWriter(log_dir=LOGS_DIR)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, sequence in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")):
        sequence = sequence.to(DEVICE)
        
        # Prepare decoder input and output
        decoder_input = sequence[:, :-1]  # Decoder input should be shifted left relative to output
        decoder_target = sequence[:, 1:]  # Decoder output should be shifted right relative to input

        # Forward pass
        decoder_output = model(decoder_input)
        
        # Reshape output and target for loss computation
        decoder_output = decoder_output.view(-1, VOCAB_SIZE)
        decoder_target = decoder_target.view(-1)

        # Compute loss
        loss = criterion(decoder_output, decoder_target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log batch loss
        writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
    
    # Log average training loss for the epoch
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)  
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    # Evaluation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for sequence in val_loader:
            sequence = sequence.to(DEVICE)
            
            # Prepare target input and output
            decoder_input = sequence[:, :-1] # Decoder input should be shifted left relative to output
            decoder_target = sequence[:, 1:] # Decoder output should be shifted right relative to input

            # Forward pass
            decoder_output = model(decoder_input)
            
            # Reshape output and target for loss computation
            decoder_output = decoder_output.view(-1, VOCAB_SIZE)
            decoder_target = decoder_target.view(-1)

            # Compute loss
            loss = criterion(decoder_output, decoder_target)
            total_val_loss += loss.item()
    
    # Log validation loss
    avg_val_loss = total_val_loss / len(val_loader)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)  
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}")

print("Training complete.")
writer.close()  # Close the TensorBoard writer