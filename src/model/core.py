import torch
import torch.nn as nn
import math
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from src.data.tokenizer import Encoding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, block_mask):
        # Q, K, V shape: [batch_size, num_heads, seq_length, d_k]
        
        # Apply flex_attention
        output = flex_attention(Q, K, V, block_mask=block_mask)
        
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, block_mask):

        # Split Heads
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # Apply linear transformations
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Perform attention with masking
        attn_output = self.scaled_dot_product_attention(Q, K, V, block_mask)
        
        # Combine heads and apply output transformation
        combined_output = self.combine_heads(attn_output)
        output = self.W_o(combined_output)
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_tokens_per_sample):
        super(PositionalEncoding, self).__init__()
        # Positional encoding per sample
        pe = self._generate_positional_encoding(max_tokens_per_sample, d_model)
        self.register_buffer('pe', pe)

    def _generate_positional_encoding(self, length, d_model):
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Shape: (1, length, d_model)

    def forward(self, x):
        device = x.device
        batch_size, seq_length, d_model = x.size()
        
        # Ensure pe is on the same device as x
        pe = self.pe[:, :seq_length, :].to(device)
        
        # Add the positional encoding to each batch in the input tensor
        x = x + pe
        
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, block_mask):
        attn_output = self.self_attn(x, x, x, block_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, max_seq_length, d_model, num_heads, num_layers, d_ff, dropout, device):
        super(Transformer, self).__init__()

        # Attach
        self.num_heads = num_heads  
        self.device = device

        # Initialize Layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask):

        # Raise an error if the input tensor is not of the correct shape
        if x.ndim != 2:
            raise ValueError(f"Expected input tensor to have 2 dimensions, but got {x.ndim}. Expected shape: (batch_size, sequence_length)")
        
        # Raise error if the attention mask is not of the correct shape
        if attention_mask.ndim != 3:
            raise ValueError(f"Expected attention mask tensor to have 3 dimensions, but got {attention_mask.ndim}. Expected shape: (batch_size, sequence_length, sequence_length)")
        
        # Embed the tokens
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Create a flex attention block mask based on this attention mask for this specific batch
        block_mask = self.create_flex_block_mask(attention_mask)

        # Pass Through Each Decoder Layer
        for layer in self.decoder_layers:
            x = layer(x, block_mask)

        # Pass Through The Final Linear Layer
        output = self.fc(x)

        return output
    

    def create_flex_block_mask(self, attention_mask, block_size=128):
        """
        Creates a BlockMask for FlexAttention based on the provided attention mask.

        Args:
            attention_mask (torch.Tensor): Boolean tensor of shape [batch_size, seq_len, seq_len].
            block_size (int): Size of the blocks to partition the attention mask.

        Returns:
            BlockMask: The created BlockMask object.
        """
        batch_size, seq_len, _ = attention_mask.shape

        def mask_mod(b, h, q_idx, kv_idx):
            return attention_mask[b, q_idx, kv_idx]

        # Create BlockMask
        block_mask = create_block_mask(
            mask_mod=mask_mod,
            B=batch_size,
            H=self.num_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            BLOCK_SIZE=block_size,
            device=self.device
        )
        return block_mask
        

    def autoregressive_inference(self, output_sequence: torch.Tensor, attention_mask: torch.Tensor, start_of_test_output_grid: int) -> torch.Tensor:
        """
        Performs autoregressive inference on the model for a single task starting from the start of the test's output grid.

        This method is intended for test time inference. For training, use the standard forward method.
        """

        # Copy and remove the test's output grid from the encoded sequence
        output_sequence = output_sequence[:start_of_test_output_grid].clone()
        last_token = output_sequence[-1].item()

        # Reshape - Single Sequence Unsqueeze Since Model Expects a Batch 
        output_sequence = output_sequence.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)[:,:-1,:-1]

        with torch.no_grad():
            while last_token != Encoding.END_OF_SEQUENCE.value:

                # Forward pass
                pred_scores = self.forward(output_sequence, attention_mask)

                # Sample the last token from the output (Using Max for Now)
                last_token = pred_scores[0, -1, :].argmax().item()

                # Append the last token to the sequence
                output_sequence = torch.cat([output_sequence, torch.tensor([[last_token]])], dim=1)

        return output_sequence
