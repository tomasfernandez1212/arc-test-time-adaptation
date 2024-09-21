import torch
import torch.nn as nn
import math
from src.data.tokenizer import Encoding
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
import functools

def causal_mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size=128):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Initialize BlockMask for causal masking
        self.block_size = block_size
        self.block_mask = None  # Will be set in forward based on sequence length

    def initialize_block_mask(self, seq_length, device):
        if self.block_mask is None or self.block_mask.pe.size(1) < seq_length:
            self.block_mask = create_block_mask(
                mask_mod=causal_mask_mod,
                B=None,
                H=None,
                Q_LEN=seq_length,
                KV_LEN=seq_length,
                BLOCK_SIZE=self.block_size
            ).to(device)

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: Tensor of shape (batch_size, seq_length, d_model)
        mask: Not used as FlexAttention handles masking internally via BlockMask
        """
        batch_size, seq_length, _ = Q.size()
        device = Q.device
        
        self.initialize_block_mask(seq_length, device)
        
        # Apply linear transformations
        Q = self.W_q(Q)  # (batch_size, seq_length, d_model)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_length, d_k)
        K = K.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
        # Perform FlexAttention
        attn_output = flex_attention(
            Q, K, V,
            block_mask=self.block_mask,
            score_mod=causal_mask_mod  # Using mask_mod for causal masking
        )
        
        # Combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
        # Output linear transformation
        output = self.W_o(attn_output)
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
        pe = self.pe[:, :seq_length, :].to(device)
        x = x + pe
        return x

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
    def __init__(self, src_possible_tokens, tgt_possible_tokens, max_tokens_per_sample, d_model, num_heads, num_layers, d_ff, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_possible_tokens, d_model)
        self.decoder_embedding = nn.Embedding(tgt_possible_tokens, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_tokens_per_sample)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_possible_tokens)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, mask=None)  # FlexAttention handles masking internally

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask=None, tgt_mask=None)  # Masking via FlexAttention

        output = self.fc(dec_output)
        return output