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
        
        # Reshape for flex_attention
        batch_size, num_heads, seq_length, d_k = Q.size()
        Q = Q.view(batch_size * num_heads, seq_length, d_k)
        K = K.view(batch_size * num_heads, seq_length, d_k)
        V = V.view(batch_size * num_heads, seq_length, d_k)
        
        # Apply flex_attention
        output = flex_attention(Q, K, V, block_mask=block_mask)
        
        # Reshape back to original shape
        output = output.view(batch_size, num_heads, seq_length, d_k)
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
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform attention with masking
        attn_output = self.scaled_dot_product_attention(Q, K, V, block_mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
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
    def __init__(self, vocab_size, max_seq_length, d_model, num_heads, num_layers, d_ff, dropout):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads  # Store num_heads for later use
        
    def forward(self, x):
       
        # Keep reference to the tokens (integers)
        token_encodings = x  # Shape: [batch_size, seq_length]
        
        # Embed the tokens
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        batch_size, seq_length = x.size(0), x.size(1)
       
        # Identify the pair that each token belongs to.
        pair_id = None # Placeholder
        grid_id = None # Placeholder
        row_id = None # Placeholder
        
        # Use to create the hierarchical mask function with token_encodings and last_grid_start_idx from outer score
        def make_hierarchical_mask(token_encodings, last_grid_start_idx):
            def hierarchical_mask(b, h, q_idx, kv_idx):
                token_q = token_encodings[b, q_idx]
                token_k = token_encodings[b, kv_idx]
                
                # Define token type sets
                PIXEL_VALUES = set(range(Encoding.BLACK.value, Encoding.BURGUNDY.value + 1))
                START_END_ROW = {Encoding.START_OF_ROW.value, Encoding.END_OF_ROW.value}
                START_END_GRID = {Encoding.START_OF_GRID.value, Encoding.END_OF_GRID.value}
                START_END_PAIR = {Encoding.START_OF_PAIR.value, Encoding.END_OF_PAIR.value}
                START_END_SEQUENCE = {Encoding.START_OF_SEQUENCE.value, Encoding.END_OF_SEQUENCE.value}
                
                # Compute conditions
                cond1 = (token_q in PIXEL_VALUES) and (token_k in START_END_ROW)
                cond2 = (token_q in START_END_ROW) and (token_k in START_END_GRID)
                cond3 = (token_q in START_END_GRID) and (token_k in START_END_PAIR)
                cond4 = (token_q in START_END_PAIR) and (token_k in START_END_SEQUENCE)
                
                # Positions in the last grid
                is_q_in_last_grid = q_idx >= last_grid_start_idx[b]
                is_k_in_last_grid = kv_idx >= last_grid_start_idx[b]
                
                # Causal condition without using 'if'
                causal_condition = (is_q_in_last_grid and is_k_in_last_grid and (q_idx >= kv_idx)) or \
                                   (not is_q_in_last_grid or not is_k_in_last_grid)
                
                # Combine all conditions using logical OR
                mask = cond1 or cond2 or cond3 or cond4 or causal_condition
                
                return mask

            return hierarchical_mask

        # Create the block mask
        block_mask = create_block_mask(
            make_hierarchical_mask(token_encodings, last_grid_start_idx),
            B=batch_size,
            H=self.num_heads,
            Q_LEN=seq_length,
            KV_LEN=seq_length,
            BLOCK_SIZE=128,
            device=x.device
        )

        for layer in self.decoder_layers:
            x = layer(x, block_mask)

        output = self.fc(x)
        return output