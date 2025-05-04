import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import logging
import datetime
import sys
from tqdm import tqdm
from models.pptnet import PPTNet, PPTNetEnsemble
from utils.data_processor import create_dataloaders
from utils.metrics import evaluate_model, calculate_metrics
from config import Config

def setup_logging(log_dir):
    """Setup logging system"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"Training log will be saved to: {log_file}")
    return logger, log_file

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed set to: {seed}")

def prepare_batch(batch, device):
    """Move batch data to device"""
    x, y, _ = batch
    return x.to(device), y.to(device)

def train_model(config, use_ensemble=False):
    """
    Train PPTNet model
    
    Args:
        config: Configuration parameters
        use_ensemble: Whether to use ensemble model
    """
    # Setup logging
    log_dir = os.path.join(config.checkpoint_dir, 'logs')
    logger, log_file = setup_logging(log_dir)
    
    # Log configuration
    logging.info("=" * 50)
    logging.info("Training Configuration:")
    for attr, value in config.__dict__.items():
        if not attr.startswith('__'):
            logging.info(f"  {attr}: {value}")
    logging.info("=" * 50)
    
    # Create data loaders
    logging.info("Creating data loaders...")
    train_loader, val_loader, test_loader, scalers = create_dataloaders(config)
    
    # Create model
    logging.info(f"Creating {'PPTNetEnsemble' if use_ensemble else 'PPTNet'} model...")
    if use_ensemble:
        model = PPTNetEnsemble(config, num_models=3).to(config.device)
    else:
        model = PPTNet(config).to(config.device)
    
    # Multi-GPU training
    if config.use_multi_gpu and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for training")
        if hasattr(config, 'gpu_ids'):
            model = nn.DataParallel(model, device_ids=config.gpu_ids)
        else:
            model = nn.DataParallel(model)
    
    # Define loss functions
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    
    def combined_loss(pred, target):
        return 0.7 * mse_loss(pred, target) + 0.3 * mae_loss(pred, target)
    
    # Define optimizer
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    
    logging.info(f"Using optimizer: {config.optimizer}, lr: {config.lr}, weight decay: {config.weight_decay}")
    
    # Learning rate scheduler
    if config.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        logging.info(f"Using cosine annealing scheduler, T_max: {config.epochs}")
    elif config.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        logging.info("Using step scheduler, step_size: 10, gamma: 0.5")
    elif config.scheduler == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=config.lr,
            steps_per_epoch=len(train_loader),
            epochs=config.epochs
        )
        logging.info(f"Using OneCycle scheduler, max_lr: {config.lr}")
    else:
        scheduler = None
        logging.info("No learning rate scheduler used")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'train_mse': [],
        'val_mse': [],
        'train_rmse': [],
        'val_rmse': []
    }
    
    # Early stopping setup
    best_val_metric = float('inf')
    patience_counter = 0
    
    # Create checkpoint directory
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config.use_amp and torch.cuda.is_available() else None
    if scaler:
        logging.info("Using mixed precision training")
    
    # Training loop
    logging.info("\nStarting training...")
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_trues = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]") as pbar:
            for i, batch in enumerate(pbar):
                x, y = prepare_batch(batch, config.device)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(x)
                        loss = combined_loss(outputs, y)
                    
                    scaler.scale(loss).backward()
                    
                    if hasattr(config, 'grad_clip_norm'):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(x)
                    loss = combined_loss(outputs, y)
                    
                    loss.backward()
                    
                    if hasattr(config, 'grad_clip_norm'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    
                    optimizer.step()
                
                train_loss += loss.item()
                
                all_train_preds.append(outputs.detach().cpu())
                all_train_trues.append(y.detach().cpu())
                
                if config.scheduler == 'onecycle':
                    scheduler.step()
                
                if i % config.log_interval == 0:
                    pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        
        # Merge all training predictions
        all_train_preds = torch.cat(all_train_preds, dim=0)
        all_train_trues = torch.cat(all_train_trues, dim=0)
        
        # Evaluate on training set
        train_mae, train_mse, train_rmse = evaluate_model(
            model, train_loader, config.device
        )
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]") as pbar:
                for batch in pbar:
                    x, y = prepare_batch(batch, config.device)
                    
                    outputs = model(x)
                    loss = combined_loss(outputs, y)
                    
                    val_loss += loss.item()
                    
                    pbar.set_postfix(loss=loss.item())
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        
        # Evaluate on validation set
        val_mae, val_mse, val_rmse = evaluate_model(
            model, val_loader, config.device
        )
        
        # Update learning rate
        if scheduler and config.scheduler != 'onecycle':
            scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        
        # Print epoch information
        epoch_info = f"\nEpoch {epoch+1}/{config.epochs}"
        train_info = f"Train - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}"
        val_info = f"Val - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}"
        
        logging.info(epoch_info)
        logging.info(train_info)
        logging.info(val_info)
        
        # Calculate weighted validation metric
        weighted_metric = 0
        for metric_name, value in zip(['MAE', 'MSE', 'RMSE'], 
                                      [val_mae, val_mse, val_rmse]):
            if hasattr(config, 'metric_weights') and metric_name in config.metric_weights:
                weighted_metric += config.metric_weights[metric_name] * value
            else:
                default_weights = {
                    'MAE': 0.4, 'MSE': 0.3, 'RMSE': 0.3
                }
                weighted_metric += default_weights[metric_name] * value
        
        # Early stopping check
        if weighted_metric < best_val_metric:
            best_val_metric = weighted_metric
            patience_counter = 0
            
            # Save best model
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), config.model_save_path)
            else:
                torch.save(model.state_dict(), config.model_save_path)
                
            logging.info(f"Saved best model, weighted validation metric: {weighted_metric:.4f}")
        else:
            patience_counter += 1
            logging.info(f"No improvement, patience counter: {patience_counter}/{config.patience}")
            
            if patience_counter >= config.patience:
                logging.info("Early stopping triggered")
                break
    
    # Load best model for testing
    logging.info("\nLoading best model for testing...")
    if use_ensemble:
        best_model = PPTNetEnsemble(config, num_models=3).to(config.device)
    else:
        best_model = PPTNet(config).to(config.device)
    
    best_model.load_state_dict(torch.load(config.model_save_path, map_location=config.device))
    best_model.eval()
    
    # Evaluate on test set
    test_mae, test_mse, test_rmse = evaluate_model(
        best_model, test_loader, config.device
    )
    
    # Print test results
    logging.info("\nTest Results:")
    logging.info(f"MAE: {test_mae:.4f}")
    logging.info(f"MSE: {test_mse:.4f}")
    logging.info(f"RMSE: {test_rmse:.4f}")
    
    # Save training history
    history_file = os.path.join(log_dir, f"training_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    import pandas as pd
    pd.DataFrame(history).to_csv(history_file, index=False)
    logging.info(f"\nTraining history saved to: {history_file}")
    
    logging.info(f"\nTraining log saved to: {log_file}")
    
    return best_model, (test_mae, test_mse, test_rmse)

if __name__ == "__main__":
    set_seed(42)
    
    config = Config()
    
    print("Starting PPTNet training...")
    model, (mae, mse, rmse) = train_model(config, use_ensemble=False)
    
    print("\nFinal Test Results:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")