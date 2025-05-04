import torch
import os

class Config:
    # Data paths
    raw_data_path = "./data/TFD4CHE/"
    processed_data_path = "./data/processed_TFD4CHE/"
    seq_len = 60  # Input sequence length (60 seconds of history)
    pred_len = 30  # Prediction horizon (30 seconds)
    feature_dim = 12  # Number of features
    feature_cols = ['second', 'car', 'bus', 'truck', 'G(t)', 'K(t)', 'q(t)', 
                    'xVelocity(t)', 'yVelocity(t)', 'xAcceleration(t)', 
                    'yAcceleration(t)', 'OccupancyRatio']
    target_col = 'K(t)'  # Target feature
    batch_size = 32
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    
    # Multi-GPU settings
    use_multi_gpu = True
    gpu_ids = [0, 1]
    use_amp = True
    
    # Model configuration
    d_model = 64  # Model dimension
    d_ff = 128  # Feed-forward dimension
    top_k = 6  # Number of periods in TimesNet
    num_kernels = 6  # Number of kernels in Inception block
    n_heads = 4  # Number of attention heads
    e_layers = 3  # Encoder layers
    d_layers = 2  # Decoder layers
    dropout = 0.2
    
    # Training configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 5e-3
    weight_decay = 1e-3
    epochs = 10
    patience = 5  # Early stopping patience
    
    # Feature processing
    normalize_features = True
    feature_selection = True
    
    # Metric weights
    metric_weights = {
        'MAE': 0.4,
        'MSE': 0.3,
        'RMSE': 0.3,   
    }
    
    # Optimizer settings
    optimizer = 'AdamW'
    scheduler = 'cosine'
    warmup_epochs = 5
    grad_clip_norm = 1.0
    
    # Save paths
    checkpoint_dir = './checkpoints/'
    model_save_path = os.path.join(checkpoint_dir, 'model.pth')
    results_dir = './results/'
    
    # Logging
    log_interval = 2
    
    def __init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)