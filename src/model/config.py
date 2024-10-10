import torch
from src.data.tokenizer import Token

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
VOCAB_SIZE = len(Token)  # Number of Colors + Other Tokens
D_MODEL = 2**2  # Embeddings Dimension - Should a power of 2 for flex attention.
NUM_HEADS = 2  # Number of attention heads. D_MODEL must be divisible by NUM_HEADS.
NUM_LAYERS = 1  # Number of decoder layers.
D_FF = 1 * D_MODEL  # Feed Forward Hidden Layer Dimensionality
DROPOUT = 0.1  # Dropout probability

# Training parameters
BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# Directories
LOGS_DIR = "logs/pretrain"
CHECKPOINT_DIR = "checkpoints"