import os
import torch
from src.data.dataset import ARCDataset, Split
from src.data.context import MAX_TOKENS_PER_TASK
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
D_MODEL = 2**2  # Embeddings Dimension - Should a power of 2 for flex attention.
NUM_HEADS = 2  # Number of attention heads. D_MODEL must be divisible by NUM_HEADS.
NUM_LAYERS = 1  # Number of decoder layers.
D_FF = 1 * D_MODEL  # Feed Forward Hidden Layer Dimensionality

# Define Training Parameters
BATCH_SIZE = 1  # Batch size
NUM_EPOCHS = 10  # Number of epochs
LEARNING_RATE = 1e-4
DROPOUT = 0.1  # Dropout probability

# Initialize Model, Loss Function, and Optimizer
model = Transformer(VOCAB_SIZE, MAX_TOKENS_PER_TASK, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT, DEVICE)
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
    for batch_idx, (sequence, attention_mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")):
        
        # Move to device
        sequence = sequence.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        
        # Prepare decoder input and output
        decoder_input = sequence[:, :-1]  # Decoder input should be shifted left relative to output
        decoder_target = sequence[:, 1:]  # Decoder output should be shifted right relative to input

        # Prepare the attention mask
        attention_mask = attention_mask[:, :-1, :-1].to(DEVICE) # Remove the last token from the attention mask to match the decoder input

        # Forward pass
        decoder_output = model(decoder_input, attention_mask)
        
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
        for sequence, attention_mask in val_loader:

            # Move to device
            sequence = sequence.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            
            # Prepare target input and output
            decoder_input = sequence[:, :-1] # Decoder input should be shifted left relative to output
            decoder_target = sequence[:, 1:] # Decoder output should be shifted right relative to input

            # Forward pass
            decoder_output = model(decoder_input, attention_mask)
            
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