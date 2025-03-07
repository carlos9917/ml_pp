import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import glob
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
# CUDA availability check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

### Data Processing Classes ###

class RoadTempDataset(Dataset):
    def __init__(self, features, targets, sequence_length=24):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return (torch.FloatTensor(x).to(device),
                torch.FloatTensor(y).to(device))

class DataPreprocessor:
    """Handles data loading and preprocessing from daily parquet files"""
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def load_parquet_files(self, directory):
        """Load and combine multiple parquet files from directory"""
        parquet_files = glob.glob(os.path.join(directory, 'road_temp_*.parquet'))
        parquet_files.sort()  # Ensure chronological order

        dfs = []
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
                print(f"Loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

        return pd.concat(dfs, ignore_index=True)

    def prepare_sequences(self, df, sequence_length=24):
        """Prepare sequences for each station"""
        # Sort by timestamp and station_id
        df = df.sort_values(['timestamp', 'station_id'])

        # Get temperature columns (depth_0 to depth_14)
        temp_cols = [col for col in df.columns if col.startswith('depth_')]

        # Group by station_id
        sequences = []
        targets = []

        for station in df['station_id'].unique():
            station_data = df[df['station_id'] == station][temp_cols].values

            # Create sequences
            for i in range(len(station_data) - sequence_length):
                sequences.append(station_data[i:i + sequence_length])
                targets.append(station_data[i + sequence_length])

        return np.array(sequences), np.array(targets)

    def load_and_preprocess(self, sequence_length=24):
        """Load and preprocess both training and test data"""
        # Load training data
        print("Loading training data...")
        train_df = self.load_parquet_files(self.train_path)

        # Load test data
        print("Loading test data...")
        test_df = self.load_parquet_files(self.test_path)

        # Prepare sequences
        print("Preparing sequences...")
        train_sequences, train_targets = self.prepare_sequences(train_df, sequence_length)
        test_sequences, test_targets = self.prepare_sequences(test_df, sequence_length)

        # Fit scalers on training data only
        train_flat = train_sequences.reshape(-1, train_sequences.shape[-1])
        self.scaler_x.fit(train_flat)
        self.scaler_y.fit(train_targets)

        # Transform both training and test data
        train_sequences_scaled = np.array([self.scaler_x.transform(seq) for seq in train_sequences])
        train_targets_scaled = self.scaler_y.transform(train_targets)

        test_sequences_scaled = np.array([self.scaler_x.transform(seq) for seq in test_sequences])
        test_targets_scaled = self.scaler_y.transform(test_targets)

        return (train_sequences_scaled, train_targets_scaled,
                test_sequences_scaled, test_targets_scaled)

### Model Architecture ###
class TempProfilePredictor(pl.LightningModule):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        ).to(device)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, input_size)
        ).to(device)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

### Training Pipeline ###
def create_training_pipeline(train_path, test_path, batch_size=32, num_epochs=100):
    # ... (previous code remains the same until trainer initialization)

    # Initialize model and trainer
    model = TempProfilePredictor()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='road_temp_model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback],
        accelerator='gpu',  # Specify GPU accelerator
        devices=1,          # Number of GPUs to use
        precision=32        # Can use 16 for mixed precision training
    )


### Usage Example ###

    # Train model
    trainer.fit(model, train_loader, val_loader)

    return model, preprocessor
if __name__ == "__main__":
    # Define paths
    train_path = "/media/cap/extra_work/road_model/profiles_sample/training"
    test_path = "/media/cap/extra_work/road_model/profiles_sample/test"

    # Train model
    model, preprocessor = create_training_pipeline(train_path, test_path)

    # Save model and preprocessor
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_x': preprocessor.scaler_x,
        'scaler_y': preprocessor.scaler_y
    }, 'road_temp_model.pth')

    # Created/Modified files during execution:
    print("Created files:", [
        "road_temp_model.pth",
        "checkpoints/road_temp_model-*.ckpt"
    ])
