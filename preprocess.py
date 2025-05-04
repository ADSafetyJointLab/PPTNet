import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import glob

# Define paths
input_dir = './data/TFD4CHE'
output_dir = './data/processed_TFD4CHE'
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

# Create output directories
for directory in [train_dir, val_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)

# Process each scenario file
csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

# Dictionary to collect all sequences
all_sequences = {}

for csv_file in csv_files:
    scenario_name = os.path.basename(csv_file).split('.')[0]
    print(f"Processing scenario: {scenario_name}")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Group by File_ID and drivingDirection
    grouped = df.groupby(['File_ID', 'drivingDirection'])
    
    for (file_id, direction), group_df in grouped:
        # Sort by second to ensure temporal order
        group_df = group_df.sort_values('second')
        
        # Create identifier for this sequence
        sequence_id = f"{scenario_name}_file{file_id}_dir{direction}"
        
        # Add to collection
        all_sequences[sequence_id] = group_df
        
# Split into train, validation, and test sets (70%, 20%, 10%)
sequence_ids = list(all_sequences.keys())
train_ids, temp_ids = train_test_split(sequence_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=1/3, random_state=42)

# Create mapping from sequence ID to target directory
target_dirs = {id: train_dir for id in train_ids}
target_dirs.update({id: val_dir for id in val_ids})
target_dirs.update({id: test_dir for id in test_ids})

# Save each sequence to target directory
for sequence_id, df in all_sequences.items():
    output_path = os.path.join(target_dirs[sequence_id], f"{sequence_id}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {sequence_id} to {output_path}")

print(f"Train sequences: {len(train_ids)}")
print(f"Validation sequences: {len(val_ids)}")
print(f"Test sequences: {len(test_ids)}")
print("Preprocessing complete!")