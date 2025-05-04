import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class MultiVariableEmbedding(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.d_model = configs.d_model
        self.cont_dim = 13 
        self.cont_proj = nn.Linear(self.cont_dim, configs.d_model)
        # Categorical embeddings
        cat_dims = configs.cat_dims
        self.embed_month = nn.Embedding(cat_dims['month'], configs.d_model)
        self.embed_weekday = nn.Embedding(cat_dims['weekday'], configs.d_model)
        self.embed_dir = nn.Embedding(cat_dims['drive_dir'], configs.d_model)
        # Positional encoding & dropout
        self.pos_enc = PositionalEncoding(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x_cont, x_month, x_weekday, x_dir):
        cont = self.cont_proj(x_cont)  
        m = self.embed_month(x_month)
        w = self.embed_weekday(x_weekday)
        d = self.embed_dir(x_dir)
        x = cont + m + w + d + self.pos_enc(cont)
        return self.dropout(x)
