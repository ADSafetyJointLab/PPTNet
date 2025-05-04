import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import glob

class TrafficFlowDataset(Dataset):
    def __init__(self, data_files, config, mode='train'):
        """
        Traffic flow dataset
        
        Args:
            data_files: List of data file paths
            config: Configuration parameters
            mode: 'train', 'val', or 'test'
        """
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.feature_cols = config.feature_cols
        self.target_col = config.target_col
        self.mode = mode
        self.normalize_features = config.normalize_features
        
        self.data_samples = []
        self.scalers = {}
        
        print(f"Processing {mode} dataset with {len(data_files)} files...")
        
        valid_files = 0
        total_samples = 0
        
        # Feature importance weights
        self.feature_weights = {
            'second': 0.5,
            'car': 0.8, 
            'bus': 0.7, 
            'truck': 0.7, 
            'G(t)': 0.9, 
            'K(t)': 1.0, 
            'q(t)': 0.9, 
            'xVelocity(t)': 1.0, 
            'yVelocity(t)': 0.6, 
            'xAcceleration(t)': 0.7, 
            'yAcceleration(t)': 0.6, 
            'OccupancyRatio': 0.8
        }
        
        # Process each data file
        for filename in data_files:
            try:
                if not os.path.exists(filename):
                    print(f"  Warning: File not found - {filename}")
                    continue
                
                # Read CSV file
                df = pd.read_csv(filename)
                
                # Verify required columns exist
                missing_cols = [col for col in self.feature_cols if col not in df.columns]
                if missing_cols:
                    print(f"  Warning: File {os.path.basename(filename)} missing columns: {missing_cols}")
                    continue
                
                # Ensure data is sorted by time
                if 'second' in df.columns:
                    df = df.sort_values('second').reset_index(drop=True)
                
                # Check if data length is sufficient
                if len(df) < self.seq_len + self.pred_len:
                    print(f"  Warning: File {os.path.basename(filename)} has insufficient data length")
                    continue
                
                valid_files += 1
                
                # Extract features
                features_df = df[self.feature_cols]
                
                # Handle missing values
                features_df = features_df.fillna(method='ffill').fillna(method='bfill')
                
                # Extract target column index
                target_idx = self.feature_cols.index(self.target_col) if self.target_col in self.feature_cols else None
                
                # Normalize data
                if self.normalize_features:
                    file_id = os.path.basename(filename).split('.')[0]
                    if file_id not in self.scalers:
                        self.scalers[file_id] = {}
                    
                    normalized_data = np.zeros_like(features_df.values, dtype=np.float32)
                    
                    # Normalize each feature individually
                    for i, col in enumerate(self.feature_cols):
                        if col != 'second':  
                            if mode == 'train':
                                scaler = StandardScaler()
                                col_data = features_df[col].values.reshape(-1, 1)
                                normalized_data[:, i] = scaler.fit_transform(col_data).flatten()
                                self.scalers[file_id][col] = scaler
                            else:
                                # Use training set scaler
                                if col in self.scalers[file_id]:
                                    col_data = features_df[col].values.reshape(-1, 1)
                                    normalized_data[:, i] = self.scalers[file_id][col].transform(col_data).flatten()
                                else:
                                    normalized_data[:, i] = features_df[col].values
                        else:
                            normalized_data[:, i] = features_df[col].values / max(1.0, features_df[col].max())
                else:
                    normalized_data = features_df.values.astype(np.float32)
                
                # Apply feature weighting (optional)
                if hasattr(self, 'feature_weights') and config.feature_selection:
                    for i, col in enumerate(self.feature_cols):
                        if col in self.feature_weights:
                            normalized_data[:, i] *= self.feature_weights[col]
                
                # Create sliding window samples
                file_samples = 0
                for i in range(len(df) - self.seq_len - self.pred_len + 1):
                    x_sample = normalized_data[i:i+self.seq_len]
                    
                    # Extract target values
                    if target_idx is not None:
                        if self.normalize_features and mode == 'train':
                            y_target_raw = df[self.target_col].values[i+self.seq_len:i+self.seq_len+self.pred_len]
                            y_target = y_target_raw.reshape(-1, 1)
                            
                            # Normalize target
                            file_id = os.path.basename(filename).split('.')[0]
                            if file_id in self.scalers and self.target_col in self.scalers[file_id]:
                                scaler = self.scalers[file_id][self.target_col]
                                y_target = scaler.transform(y_target.reshape(-1, 1))
                        else:
                            y_target = normalized_data[i+self.seq_len:i+self.seq_len+self.pred_len, target_idx]
                            y_target = y_target.reshape(-1, 1)
                    else:
                        # Predict all features
                        y_target = normalized_data[i+self.seq_len:i+self.seq_len+self.pred_len]
                    
                    # Store sample and metadata
                    self.data_samples.append({
                        'x': x_sample,
                        'y': y_target,
                        'file_id': os.path.basename(filename).split('.')[0]
                    })
                    file_samples += 1
                
                total_samples += file_samples
                print(f"  Processed file: {os.path.basename(filename)}, created {file_samples} samples")
                    
            except Exception as e:
                print(f"  Error: Failed to process file {os.path.basename(filename)} - {str(e)}")
        
        print(f"{mode} dataset created: {valid_files}/{len(data_files)} valid files, {len(self.data_samples)} samples total")
        
        if len(self.data_samples) == 0:
            print(f"Warning: {mode} dataset has no valid samples!")
    
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        # Convert to tensors
        x = torch.FloatTensor(sample['x'])
        y = torch.FloatTensor(sample['y'])
        
        return x, y, sample['file_id']

def create_dataloaders(config):
    """Create train, validation and test dataloaders"""
    
    # Get data directories
    train_dir = os.path.join(config.processed_data_path, 'train')
    val_dir = os.path.join(config.processed_data_path, 'val')
    test_dir = os.path.join(config.processed_data_path, 'test')
    
    # Get file lists
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.csv")))
    val_files = sorted(glob.glob(os.path.join(val_dir, "*.csv")))
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))

    if len(train_files) == 0 and len(val_files) == 0 and len(test_files) == 0:
        print("Warning: No CSV files found in preprocessed directory, attempting to create from raw data...")
        
        # Get files from raw data directory
        all_files = sorted(glob.glob(os.path.join(config.raw_data_path, "*.csv")))
        if len(all_files) > 0:
            # Create directories
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            
            # Set random seed
            np.random.seed(42)
            np.random.shuffle(all_files)
            
            # Split data
            train_size = int(len(all_files) * config.train_ratio)
            val_size = int(len(all_files) * config.val_ratio)
            
            train_src_files = all_files[:train_size]
            val_src_files = all_files[train_size:train_size+val_size]
            test_src_files = all_files[train_size+val_size:]
            
            # Copy files
            import shutil
            for src_file in train_src_files:
                dst_file = os.path.join(train_dir, os.path.basename(src_file))
                shutil.copy(src_file, dst_file)
                train_files.append(dst_file)
            
            for src_file in val_src_files:
                dst_file = os.path.join(val_dir, os.path.basename(src_file))
                shutil.copy(src_file, dst_file)
                val_files.append(dst_file)
            
            for src_file in test_src_files:
                dst_file = os.path.join(test_dir, os.path.basename(src_file))
                shutil.copy(src_file, dst_file)
                test_files.append(dst_file)
            
            print(f"Split {len(all_files)} files into preprocessed directories")
        else:
            raise FileNotFoundError(f"Error: No CSV files found in raw data directory: {config.raw_data_path}")
    
    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    
    # Create datasets
    train_dataset = TrafficFlowDataset(train_files, config, mode='train')
    val_dataset = TrafficFlowDataset(val_files, config, mode='val')
    test_dataset = TrafficFlowDataset(test_files, config, mode='test')
    
    # Verify datasets are not empty
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Error: At least one dataset is empty!")
    
    # Calculate number of workers
    num_workers = min(12, os.cpu_count() or 1)
    if hasattr(config, 'use_multi_gpu') and config.use_multi_gpu and torch.cuda.device_count() > 1:
        num_workers = min(16, num_workers * torch.cuda.device_count())
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(config.device.type == 'cuda'),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(config.device.type == 'cuda')
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(config.device.type == 'cuda')
    )
    
    print(f"Data loaders created:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, train_dataset.scalers
