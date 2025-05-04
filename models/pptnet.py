import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from layers.Conv_Blocks import Inception_Block_V1
from models.transformer_decoder import TransformerDecoder, generate_square_subsequent_mask

class FeatureAttention(nn.Module):
    """Feature attention module for adaptive feature importance learning"""
    def __init__(self, feature_dim, d_model):
        super(FeatureAttention, self).__init__()
        self.feature_projection = nn.Linear(feature_dim, d_model)
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, feature_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        
        # Calculate feature representation
        feature_repr = torch.mean(x, dim=1) 
        
        # Project to higher dimensional space
        feature_proj = self.feature_projection(feature_repr)  
        
        # Calculate attention weights
        attention_weights = self.attention(feature_proj)  
        
        # Weight features
        attention_weights = attention_weights.unsqueeze(1).expand(-1, seq_len, -1)
        weighted_features = x * attention_weights
        
        return weighted_features

class AdaptivePeriodicLayer(nn.Module):
    """Adaptive periodic layer for discovering and processing multiple periodic patterns"""
    def __init__(self, config):
        super(AdaptivePeriodicLayer, self).__init__()
        self.d_model = config.d_model
        self.top_k = config.top_k
        self.seq_len = config.seq_len
        
        # Inception module (parameter sharing)
        self.inception = nn.Sequential(
            Inception_Block_V1(config.d_model, config.d_ff, num_kernels=config.num_kernels),
            nn.GELU(),
            Inception_Block_V1(config.d_ff, config.d_model, num_kernels=config.num_kernels)
        )
        
        # Period attention
        self.period_attention = nn.Sequential(
            nn.Linear(config.top_k, config.top_k * 2),
            nn.ReLU(),
            nn.Linear(config.top_k * 2, config.top_k),
            nn.Softmax(dim=-1)
        )
        
        # Period enhancement
        self.period_enhancement = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU()
        )
        
    def forward(self, x):
        """
        Input: [batch_size, seq_len, d_model]
        Output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # FFT to find periods
        x_fft = torch.fft.rfft(x, dim=1)
        x_fft_abs = torch.abs(x_fft)
        x_fft_avg = torch.mean(x_fft_abs, dim=-1)  
        
        # Filter DC component
        x_fft_avg[:, 0] = 0

        _, top_indices = torch.topk(x_fft_avg, self.top_k, dim=1)  
        
        batch_outputs = []
        period_weights = []
        
        for b in range(batch_size):
            sample_outputs = []
            sample_weights = []
            
            # Process each period
            for k in range(self.top_k):
                freq_idx = top_indices[b, k].item()
                if freq_idx == 0:  # Skip DC component
                    period = seq_len
                else:
                    period = max(1, seq_len // freq_idx)
                
                # Calculate period importance
                importance = x_fft_avg[b, freq_idx].item()
                sample_weights.append(importance)
                
                # Reshape to 2D
                if seq_len % period != 0:
                    pad_len = period - (seq_len % period)
                    padded = F.pad(x[b], (0, 0, 0, pad_len))
                    num_segments = (seq_len + pad_len) // period
                else:
                    padded = x[b]
                    num_segments = seq_len // period
                
                # Reshape to 2D structure 
                reshaped = padded.reshape(num_segments, period, d_model)
                reshaped = reshaped.permute(1, 0, 2)  
                
                # Convert to 4D tensor for Inception module
                reshaped = reshaped.permute(2, 0, 1).unsqueeze(0)  
                
                # Apply Inception module for 2D pattern capture
                processed = self.inception(reshaped)  
                
                # Convert back to original shape
                processed = processed.squeeze(0).permute(1, 2, 0)  
                processed = processed.permute(1, 0, 2)  
                
                # Reshape back to 1D
                flat = processed.reshape(-1, d_model)
                
                # Truncate to original length
                output = flat[:seq_len]
                
                # Period enhancement
                output = self.period_enhancement(output)
                
                sample_outputs.append(output)
            
            # Stack all period results
            if sample_outputs:
                stacked = torch.stack(sample_outputs)  
                batch_outputs.append(stacked)
                
                # Record weights
                sample_weights = torch.tensor(sample_weights, device=x.device)
                period_weights.append(sample_weights)
        
        # Stack all batch results
        if not batch_outputs:
            return x  
            
        period_outputs = torch.stack(batch_outputs) 
        
        # Stack weights
        period_weights = torch.stack(period_weights)  
        
        # Normalize period weights
        period_weights = F.softmax(period_weights, dim=1)
        
        # Calculate last time step period representation
        last_step_features = period_outputs[:, :, -1, :].mean(dim=-1)  
        
        # Apply period attention to adjust weights
        attention_weights = self.period_attention(last_step_features)  
        
        # Combine weights
        combined_weights = period_weights * attention_weights  
        combined_weights = F.softmax(combined_weights, dim=1)
        
        # Expand weight dimensions
        combined_weights = combined_weights.unsqueeze(-1).unsqueeze(-1) 
        
        # Apply weights for weighted fusion
        weighted_outputs = period_outputs * combined_weights 
        summed_outputs = weighted_outputs.sum(dim=1)  
        
        # Residual connection
        enhanced_output = summed_outputs + x
        
        return enhanced_output

class PPTNet(nn.Module):
    def __init__(self, config):
        super(PPTNet, self).__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.feature_dim = config.feature_dim
        self.d_model = config.d_model
        
        # Feature attention
        self.feature_attention = FeatureAttention(config.feature_dim, config.d_model)
        
        # Input embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(config.feature_dim, config.d_model),
            nn.GELU()
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            self._get_positional_encoding(config.seq_len, config.d_model),
            requires_grad=False
        )
        
        # Adaptive periodic layers
        self.periodic_layers = nn.ModuleList([
            AdaptivePeriodicLayer(config) for _ in range(config.e_layers)
        ])
        
        # Layer normalization
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(config.d_model) for _ in range(config.e_layers)
        ])
        
        # Transformer decoder
        self.decoder = TransformerDecoder(
            d_model=config.d_model,
            nhead=config.n_heads,
            d_ff=config.d_ff,
            num_layers=config.d_layers,
            dropout=config.dropout
        )
        
        # Decoder query input
        self.query_embed = nn.Parameter(torch.randn(1, config.pred_len, config.d_model))
        
        # Output projection
        self.projection = nn.Linear(config.d_model, 1)  
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
    
    def _get_positional_encoding(self, seq_length, d_model):
        """Generate positional encoding"""
        position = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(seq_length, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0) 
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input sequence [batch_size, seq_len, feature_dim]
            
        Returns:
            predictions [batch_size, pred_len, 1]
        """
        batch_size = x.size(0)
        
        # Feature attention
        x = self.feature_attention(x)
        
        # Embedding layer
        x = self.embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1)]
        
        # Period extraction and processing
        for i, (periodic_layer, norm_layer) in enumerate(zip(self.periodic_layers, self.norm_layers)):
            # Periodic feature extraction
            periodic_out = periodic_layer(x)
            
            # Layer normalization
            x = norm_layer(periodic_out)
        
        # Prepare decoder query
        query = self.query_embed.expand(batch_size, -1, -1)
        
        # Generate decoder mask
        tgt_mask = generate_square_subsequent_mask(self.pred_len).to(x.device)
        
        # Transformer decoder
        dec_out = self.decoder(query, x, tgt_mask=tgt_mask)
        
        # Output projection
        output = self.projection(dec_out)  
        
        return output

class PPTNetEnsemble(nn.Module):
    """PPTNet ensemble combining multiple PPTNet instances for improved robustness"""
    def __init__(self, config, num_models=3):
        super(PPTNetEnsemble, self).__init__()
        self.models = nn.ModuleList([
            PPTNet(config) for _ in range(num_models)
        ])
        
        # Model weights (learnable)
        self.model_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
    def forward(self, x):
        # Get predictions from each model
        predictions = [model(x) for model in self.models]
        
        # Stack predictions
        stacked_preds = torch.stack(predictions, dim=0)  
        
        # Get softmax weights
        weights = F.softmax(self.model_weights, dim=0)
        
        # Weighted fusion
        weighted_preds = torch.einsum('n,npqr->pqr', weights, stacked_preds)
        
        return weighted_preds
