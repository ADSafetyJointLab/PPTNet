import numpy as np
import torch

def calculate_metrics(y_pred, y_true, seasonal_period=5, x=None):
    """
    Calculate evaluation metrics
    
    Args:
        y_pred: Predictions [batch_size, seq_len, feature_dim]
        y_true: Ground truth [batch_size, seq_len, feature_dim]
        
    Returns:
        mae: Mean Absolute Error
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
    """
    # Convert to numpy arrays
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    
    # Handle input dimensions
    if len(y_pred.shape) == 3 and y_pred.shape[2] == 1:
        y_pred = y_pred.squeeze(-1)
    if len(y_true.shape) == 3 and y_true.shape[2] == 1:
        y_true = y_true.squeeze(-1)
    
    # Ensure shapes match
    assert y_pred.shape == y_true.shape, "Prediction and ground truth shapes mismatch"
    
    # Calculate metrics
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean(np.square(y_pred - y_true))
    rmse = np.sqrt(mse)
    
    return mae, mse, rmse

def evaluate_model(model, data_loader, device, seasonal_period=5):
    """
    Evaluate model performance
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device (CPU or GPU)
        
    Returns:
        mae: Mean Absolute Error
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
    """
    model.eval()
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch in data_loader:
            x, y, _ = batch[0], batch[1], batch[2]
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            pred = model(x)
            
            # Save results
            all_preds.append(pred.detach().cpu())
            all_trues.append(y.detach().cpu())
    
    # Merge batch results
    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)
    
    # Calculate metrics
    mae, mse, rmse = calculate_metrics(all_preds, all_trues)
    
    return mae, mse, rmse

def calculate_weighted_metric(metrics, weights):

    if isinstance(metrics, dict):
        weighted_metric = 0.0
        for metric_name, value in metrics.items():
            if metric_name in weights:
                weighted_metric += weights[metric_name] * value
    else:
        # Assume metrics are in order: MAE, MSE, RMSE
        weighted_metric = (
            weights.get('MAE', 0.4) * metrics[0] +
            weights.get('MSE', 0.3) * metrics[1] +
            weights.get('RMSE', 0.3) * metrics[2]
        )
    
    return weighted_metric