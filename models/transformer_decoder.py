import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """Positional encoding module"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Calculate positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        # Register as buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            Positional encoding: [1, seq_len, d_model]
        """
        return self.pe[:, :x.size(1)]

class TransformerDecoder(nn.Module):
    """Transformer decoder module"""
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt: Target sequence [batch_size, tgt_len, d_model]
            memory: Memory sequence (encoder output) [batch_size, src_len, d_model]
            tgt_mask: Target sequence mask [tgt_len, tgt_len]
            memory_mask: Memory sequence mask [tgt_len, src_len]
            
        Returns:
            Decoder output [batch_size, tgt_len, d_model]
        """
        # Add positional encoding
        tgt = tgt + self.pos_encoder(tgt)
        
        # Transformer decoding
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        
        return output

def generate_square_subsequent_mask(sz):
    """
    Generate square subsequent mask to prevent decoder from seeing future information
    
    Args:
        sz: Sequence length
        
    Returns:
        Mask matrix [sz, sz]
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask