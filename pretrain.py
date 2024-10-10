import os
import torch
from src.data.dataset import ARCDataset, Split
from src.data.context import MAX_TOKENS_PER_TASK
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model.core import Transformer
from src.data.tokenizer import Encoding
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter  
import src.model.config as config

# Device configuration
DEVICE = config.DEVICE

# Model parameters from config
VOCAB_SIZE = config.VOCAB_SIZE
D_MODEL = config.D_MODEL
NUM_HEADS = config.NUM_HEADS
NUM_LAYERS = config.NUM_LAYERS
D_FF = config.D_FF
DROPOUT = config.DROPOUT

# Training parameters from config
BATCH_SIZE = config.BATCH_SIZE
NUM_EPOCHS = config.NUM_EPOCHS
LEARNING_RATE = config.LEARNING_RATE

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
LOGS_DIR = config.LOGS_DIR
CHECKPOINT_DIR = config.CHECKPOINT_DIR
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=LOGS_DIR)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, (sequence, attention_mask, start_of_test_output_grid) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")):
        
        # Move to device
        sequence = sequence.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        start_of_test_output_grid = start_of_test_output_grid.to(DEVICE)
        
        # Prepare decoder input and output
        decoder_input = sequence[:, :-1]  # Decoder input should be shifted left relative to output
        decoder_target = sequence[:, 1:]  # Decoder output should be shifted right relative to input

        # Prepare the attention mask
        attention_mask = attention_mask[:, :-1, :-1]  # Adjust mask to match decoder input

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

    # Save checkpoint at the end of each epoch
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

print("Training complete.")
writer.close()  # Close the TensorBoard writer