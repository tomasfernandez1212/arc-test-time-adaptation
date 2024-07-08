import torch
import torch.nn as nn
import math

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
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
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
    def __init__(self, d_model, max_pairs_in_sample, max_grids_in_pair, max_pixels_in_row, max_pixels_in_col):
        super(PositionalEncoding, self).__init__()

        # Positional encoding for Pixels in Grid
        pe = self._generate_4d_positional_encoding(max_pairs_in_sample, max_grids_in_pair, max_pixels_in_row, max_pixels_in_col, d_model)
        self.register_buffer('pe', pe)

    def _generate_4d_positional_encoding(self, max_pairs_in_sample, max_grids_in_pair, max_pixels_in_row, max_pixels_in_col, d_model):

        if d_model%4!=0:
            raise ValueError("d_model must be divisible by 4")
        if d_model<8: 
            raise ValueError("d_model must be at least 8")
        
        pe = torch.zeros(max_pairs_in_sample, max_grids_in_pair, max_pixels_in_row, max_pixels_in_col, d_model)
        
        # Positional encoding for groups
        group_position = torch.arange(0, max_pairs_in_sample, dtype=torch.float).unsqueeze(1)
        div_term_group = torch.exp(torch.arange(0, d_model // 4, 2).float() * -(math.log(10000.0) / (d_model // 4)))
        pe[:, :, :, :, 0::8] = torch.sin(group_position * div_term_group).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        pe[:, :, :, :, 1::8] = torch.cos(group_position * div_term_group).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        # Positional encoding for grids within groups
        grid_position = torch.arange(0, max_grids_in_pair, dtype=torch.float).unsqueeze(1)
        div_term_grid = torch.exp(torch.arange(0, d_model // 4, 2).float() * -(math.log(10000.0) / (d_model // 4)))
        pe[:, :, :, :, 2::8] = torch.sin(grid_position * div_term_grid).unsqueeze(0).unsqueeze(2).unsqueeze(2)
        pe[:, :, :, :, 3::8] = torch.cos(grid_position * div_term_grid).unsqueeze(0).unsqueeze(2).unsqueeze(2)
        
        # Positional encoding for rows within grids
        row_position = torch.arange(0, max_pixels_in_row, dtype=torch.float).unsqueeze(1)
        div_term_row = torch.exp(torch.arange(0, d_model // 4, 2).float() * -(math.log(10000.0) / (d_model // 4)))
        pe[:, :, :, :, 4::8] = torch.sin(row_position * div_term_row).unsqueeze(0).unsqueeze(0).unsqueeze(3)
        pe[:, :, :, :, 5::8] = torch.cos(row_position * div_term_row).unsqueeze(0).unsqueeze(0).unsqueeze(3)
        
        # Positional encoding for columns within grids
        col_position = torch.arange(0, max_pixels_in_col, dtype=torch.float).unsqueeze(1)
        div_term_col = torch.exp(torch.arange(0, d_model // 4, 2).float() * -(math.log(10000.0) / (d_model // 4)))
        pe[:, :, :, :, 6::8] = torch.sin(col_position * div_term_col).unsqueeze(0).unsqueeze(0).unsqueeze(2)
        pe[:, :, :, :, 7::8] = torch.cos(col_position * div_term_col).unsqueeze(0).unsqueeze(0).unsqueeze(2)
        
        return pe.unsqueeze(0)

    def forward(self, x, 
        
        
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_possible_tokens, tgt_possible_tokens, max_pairs_in_sample, max_grids_in_pair, max_pixels_in_row, max_pixels_in_col, d_model, num_heads, num_layers, d_ff, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_possible_tokens, d_model)
        self.decoder_embedding = nn.Embedding(tgt_possible_tokens, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_pairs_in_sample, max_grids_in_pair, max_pixels_in_row, max_pixels_in_col)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_possible_tokens)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        device = src.device  # Ensure the mask is on the same device as the input tensors
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2) 
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3) 
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt, grid_starts, grid_lengths, pair_starts, pair_lengths):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src), grid_starts, grid_lengths, pair_starts, pair_lengths))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt), grid_starts, grid_lengths, pair_starts, pair_lengths))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output