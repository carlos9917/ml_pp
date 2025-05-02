"""
LSTM model for predicting road temperatures (TROAD) using observation data.
This implementation uses PyTorch, ensures proper time-based train/test splitting,
includes interactive Plotly visualization, and adapts to available data.
"""
import sqlite3
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Parse command line arguments
parser = argparse.ArgumentParser(description='PyTorch LSTM model for road temperature prediction')
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help='Ratio of data to use for training (default: 0.8)')
parser.add_argument('--lookback', type=int, default=24,
                    help='Number of previous time steps to use for prediction')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=50,
                    help='Maximum number of epochs')
parser.add_argument('--patience', type=int, default=5,
                    help='Patience for early stopping')
args = parser.parse_args()

####
# data loading
DB = "/media/cap/extra_work/road_model/OBSTABLE"

def load_and_merge_data_optimized(variables, year):
    """
    Load data from SQLite databases and merge into a single dataframe.
    Uses all available data without day limits.

    Args:
        variables: List of meteorological variables to load
        year: Year of data to load

    Returns:
        DataFrame with merged data from all variables
    """
    try:
        dataframes = []

        for variable in variables:
            db_path = os.path.join(DB, f'OBSTABLE_{variable}_{year}.sqlite')
            if not os.path.exists(db_path):
                print(f"Warning: Database file not found: {db_path}")
                continue

            conn = sqlite3.connect(db_path)
            # Optimize SQLite performance
            conn.execute('PRAGMA synchronous = OFF')
            conn.execute('PRAGMA journal_mode = MEMORY')
            query = f"SELECT valid_dttm, SID, lat, lon, {variable} FROM SYNOP"

            try:
                for chunk in pd.read_sql_query(query, conn, chunksize=10000):
                    dataframes.append(chunk)
            except sqlite3.Error as e:
                print(f"SQLite error when reading {variable}: {e}")
            finally:
                conn.close()

        if not dataframes:
            raise ValueError("No data loaded from database")

        # Merge all dataframes
        full_df = pd.concat(dataframes, ignore_index=True)
        merged_df = full_df.groupby(['valid_dttm', 'SID', 'lat', 'lon']).first().reset_index()
        return merged_df

    except (sqlite3.Error, ValueError) as e:
        print(f"Error loading data: {str(e)}")
        return None

# PyTorch Dataset for time series
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# PyTorch LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)

        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out[:, -1, :])  # Take only the last output

        # Fully connected layer
        out = self.fc(lstm2_out)
        return out

def create_sequences(data, target_col, lookback):
    """
    Create sequences for LSTM model training

    Args:
        data: DataFrame with features and target
        target_col: Name of the target column
        lookback: Number of previous time steps to use

    Returns:
        X: Input sequences
        y: Target values
    """
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data.iloc[i:i+lookback].values)
        y.append(data.iloc[i+lookback][target_col])
    return np.array(X), np.array(y)

# Training function with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, device):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses

# Define the variables and year
variables = ['TROAD', 'T2m', 'Td2m', 'D10m', 'S10m', 'AccPcp12h']
year = 2023

# Load all available data
print("Loading all available data...")
df = load_and_merge_data_optimized(variables, year)

if df is None or len(df) == 0:
    raise ValueError("No data available. Please check your database connection and file paths.")

# Print data summary
print(f"Loaded {len(df)} total records")

# Drop rows with missing values in key columns
df_cleaned = df[["valid_dttm", "SID", "lat", "lon", "TROAD", "T2m","Td2m"]].dropna()
print(f"After cleaning: {len(df_cleaned)} records")

# Convert Unix timestamp to datetime for better handling
df_cleaned["dates"] = pd.to_datetime(df_cleaned["valid_dttm"], unit="s")

select_station = 630200
df_cleaned = df_cleaned[df_cleaned.SID == select_station]


# Sort data chronologically
df_cleaned = df_cleaned.sort_values("valid_dttm")

# Group by hour to reduce data size and create regular time intervals
df_cleaned['hour'] = df_cleaned['dates'].dt.floor('h')
df_hourly = df_cleaned.groupby('hour').agg({
    'valid_dttm': 'mean',
    'lat': 'mean',
    'lon': 'mean',
    'TROAD': 'mean',
    'T2m': 'mean',
    'Td2m': 'mean',
    'dates': 'first'
}).reset_index()

# Drop the hour column as it's no longer needed
df_hourly.drop('hour', axis=1, inplace=True)

print(f"After hourly aggregation: {len(df_hourly)} records")

# Check if we have enough data for modeling
min_required_samples = 72  # At least 3 days of hourly data
if len(df_hourly) < min_required_samples:
    raise ValueError(f"Not enough data for modeling. Need at least {min_required_samples} hourly records, but got {len(df_hourly)}.")

# Calculate the split point based on the train ratio
train_size = int(len(df_hourly) * args.train_ratio)
print(f"Hours to use for training out of {len(df_hourly)}: {train_size}")
hours_pred = int(len(df_hourly)) - train_size
print(f"Predicting: {hours_pred} hours")

# Ensure we have enough data for both training and testing
if train_size < 48:  # At least 2 days for training
    train_size = 48
    print(f"Warning: Adjusted training size to minimum of 48 hours")

if len(df_hourly) - train_size < 24:  # At least 1 day for testing
    train_size = len(df_hourly) - 24
    print(f"Warning: Adjusted training size to ensure at least 24 hours for testing")

# Time-based train-test split
train_df = df_hourly.iloc[:train_size]
test_df = df_hourly.iloc[train_size:]

# Calculate days for reporting
train_days = (train_df['dates'].max() - train_df['dates'].min()).total_seconds() / (24 * 3600)
test_days = (test_df['dates'].max() - test_df['dates'].min()).total_seconds() / (24 * 3600)

print(f"Training data: {len(train_df)} hours (~{train_days:.1f} days)")
print(f"Test data: {len(test_df)} hours (~{test_days:.1f} days)")

# Define features and target
#features = ['lat', 'lon', 'T2m','TROAD',"Td2m"]
features = ['T2m','TROAD',"Td2m"]
target = 'TROAD'

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit scalers on training data only
train_features = scaler_X.fit_transform(train_df[features])
train_target = scaler_y.fit_transform(train_df[[target]])

# Transform test data
test_features = scaler_X.transform(test_df[features])
test_target = scaler_y.transform(test_df[[target]])

# Create a DataFrame with scaled values
train_scaled = pd.DataFrame(train_features, columns=features)
train_scaled[target] = train_target
test_scaled = pd.DataFrame(test_features, columns=features)
test_scaled[target] = test_target

# Adjust lookback if needed
lookback = min(args.lookback, len(train_scaled) // 4)  # Ensure lookback isn't too large
if lookback != args.lookback:
    print(f"Warning: Adjusted lookback to {lookback} (from {args.lookback}) due to limited training data.")

# Create sequences for LSTM
X_train, y_train = create_sequences(train_scaled, target, lookback)
X_test, y_test = create_sequences(test_scaled, target, lookback)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Split training data into training and validation sets (80/20)
val_size = max(1, int(len(X_train) * 0.2))  # Ensure at least 1 sample for validation
X_val = X_train[-val_size:]
y_val = y_train[-val_size:]
X_train = X_train[:-val_size]
y_train = y_train[:-val_size]

# Create PyTorch datasets and dataloaders
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)

# Adjust batch size if needed
batch_size = min(args.batch_size, len(train_dataset) // 2)  # Ensure batch size isn't too large
if batch_size < 1:
    batch_size = 1
if batch_size != args.batch_size:
    print(f"Warning: Adjusted batch size to {batch_size} (from {args.batch_size}) due to limited training data.")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model
input_dim = X_train.shape[2]  # Number of features
hidden_dim1 = 50
hidden_dim2 = 30
model = LSTMModel(input_dim, hidden_dim1, hidden_dim2).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print model summary
print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# Train the model
model, train_losses, val_losses = train_model(
    model, train_loader, val_loader, criterion, optimizer,
    args.epochs, args.patience, device
)

# Evaluate on test set
model.eval()
y_pred_list = []
y_test_list = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_pred = model(X_batch)
        y_pred_list.extend(y_pred.cpu().numpy())
        y_test_list.extend(y_batch.numpy())

# Convert predictions to numpy arrays
y_pred_scaled = np.array(y_pred_list)
y_test_scaled = np.array(y_test_list).reshape(-1, 1)

# Inverse transform to get actual values
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# Create timestamps for predictions (offset by lookback)
pred_dates = test_df['dates'].iloc[lookback:lookback+len(y_pred)]

# --- Training History Plot ---
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('LSTM Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('lstm_training_history.png')
plt.show()

# --- Scatter Plot: Actual vs Predicted ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test_actual, y_pred, alpha=0.5, label='Predicted vs Actual')
plt.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()],
         'r--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Road Temperature')
plt.ylabel('Predicted Road Temperature')
plt.title('LSTM: Actual vs Predicted Road Temperature')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('lstm_scatter_plot.png')
plt.show()

# --- Interactive Plotly Time Series Plot ---
# Create a subplot with 2 rows
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=("Road Temperature Prediction", "Prediction Error"),
    row_heights=[0.7, 0.3]
)

# Add training data (last part for context)
context_size = min(72, len(train_df))  # Show last 3 days or all training data
train_context = train_df.iloc[-context_size:]

fig.add_trace(
    go.Scatter(
        x=train_context["dates"],
        y=train_context[target],
        mode='lines',
        name='Training Data',
        line=dict(color='blue', width=1),
        opacity=0.7
    ),
    row=1, col=1
)

# Add test actual data
actual_dates = pred_dates
fig.add_trace(
    go.Scatter(
        x=actual_dates,
        y=y_test_actual.flatten(),
        mode='lines',
        name='Test Actual',
        line=dict(color='green', width=2)
    ),
    row=1, col=1
)

# Add test predictions
fig.add_trace(
    go.Scatter(
        x=actual_dates,
        y=y_pred.flatten(),
        mode='lines',
        name='LSTM Predictions',
        line=dict(color='orange', width=2, dash='dash')
    ),
    row=1, col=1
)

# Add vertical line for train/test split
split_time_str = train_df["dates"].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
fig.add_vline(
    x=split_time_str,
    line_width=2,
    line_dash="dash",
    line_color="red",
    row=1, col=1
)

# Add annotation for train/test split
fig.add_annotation(
    x=split_time_str,
    y=1.05,
    text="Train/Test Split",
    showarrow=True,
    arrowhead=1,
    ax=0,
    ay=-40,
    yref="paper",
    row=1, col=1
)

# Add prediction error plot
prediction_error = y_test_actual.flatten() - y_pred.flatten()
fig.add_trace(
    go.Scatter(
        x=actual_dates,
        y=prediction_error,
        mode='lines',
        name='Prediction Error',
        line=dict(color='red', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.1)'
    ),
    row=2, col=1
)

# Add zero line for error reference
fig.add_hline(
    y=0,
    line_width=1,
    line_dash="solid",
    line_color="black",
    row=2, col=1
)

# Update layout
fig.update_layout(
    title_text=f"PyTorch LSTM Road Temperature Prediction ({train_days:.1f} days training, {test_days:.1f} days prediction)",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    height=800,
    width=1200,
    template="plotly_white"
)

# Update y-axes labels
fig.update_yaxes(title_text="Road Temperature (°C)", row=1, col=1)
fig.update_yaxes(title_text="Error (°C)", row=2, col=1)

# Update x-axis
fig.update_xaxes(
    title_text="Date",
    rangeslider_visible=True,
    row=2, col=1
)

# Add annotations for model metrics
fig.add_annotation(
    x=0.01,
    y=0.98,
    xref="paper",
    yref="paper",
    text=f"MAE: {mae:.2f}°C<br>RMSE: {rmse:.2f}°C",
    showarrow=False,
    font=dict(size=14),
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    borderpad=4
)

# Save the figure as HTML for interactive viewing
fig.write_html("pytorch_lstm_prediction.html")

# Show the figure
fig.show()

# Create a more detailed interactive dashboard
dashboard = make_subplots(
    rows=2,
    cols=2,
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter", "colspan": 2}, None]
    ],
    subplot_titles=(
        "Actual vs Predicted",
        "Training & Validation Loss",
        "Temperature Prediction Over Time"
    ),
    vertical_spacing=0.1,
    horizontal_spacing=0.1
)

# 1. Scatter plot of actual vs predicted
dashboard.add_trace(
    go.Scatter(
        x=y_test_actual.flatten(),
        y=y_pred.flatten(),
        mode='markers',
        name='Test Data Points',
        marker=dict(
            color='rgba(0, 100, 80, 0.7)',
            size=8
        )
    ),
    row=1, col=1
)

# Add identity line
min_val = min(y_test_actual.min(), y_pred.min())
max_val = max(y_test_actual.max(), y_pred.max())
dashboard.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', width=2, dash='dash')
    ),
    row=1, col=1
)

# 2. Training history
dashboard.add_trace(
    go.Scatter(
        x=list(range(1, len(train_losses) + 1)),
        y=train_losses,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='blue', width=2)
    ),
    row=1, col=2
)

dashboard.add_trace(
    go.Scatter(
        x=list(range(1, len(val_losses) + 1)),
        y=val_losses,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='red', width=2)
    ),
    row=1, col=2
)

# 3. Time series prediction with hourly pattern
# Add training data (last part for context)
dashboard.add_trace(
    go.Scatter(
        x=train_context["dates"],
        y=train_context[target],
        mode='lines',
        name='Training Data',
        line=dict(color='blue', width=1),
        opacity=0.7
    ),
    row=2, col=1
)

# Add test actual data
dashboard.add_trace(
    go.Scatter(
        x=actual_dates,
        y=y_test_actual.flatten(),
        mode='lines',
        name='Test Actual',
        line=dict(color='green', width=2)
    ),
    row=2, col=1
)

# Add test predictions
dashboard.add_trace(
    go.Scatter(
        x=actual_dates,
        y=y_pred.flatten(),
        mode='lines',
        name='LSTM Predictions',
        line=dict(color='orange', width=2, dash='dash')
    ),
    row=2, col=1
)

# Add vertical line for train/test split
dashboard.add_vline(
    x=split_time_str,
    line_width=2,
    line_dash="dash",
    line_color="red",
    row=2, col=1
)

# Add annotation for train/test split
dashboard.add_annotation(
    x=split_time_str,
    y=1.05,
    text="Train/Test Split",
    showarrow=True,
    arrowhead=1,
    ax=0,
    ay=-40,
    yref="paper",
    row=2, col=1
)

# Update layout
dashboard.update_layout(
    title_text=f"PyTorch LSTM Road Temperature Prediction Dashboard ({train_days:.1f} days → {test_days:.1f} days)",
    height=900,
    width=1200,
    template="plotly_white",
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Update axes
dashboard.update_xaxes(title_text="Actual Road Temperature", row=1, col=1)
dashboard.update_yaxes(title_text="Predicted Road Temperature", row=1, col=1)

dashboard.update_xaxes(title_text="Epoch", row=1, col=2)
dashboard.update_yaxes(title_text="Loss (MSE)", row=1, col=2)

dashboard.update_xaxes(title_text="Date", row=2, col=1)
dashboard.update_yaxes(title_text="Temperature (°C)", row=2, col=1)

# Add model architecture annotation
model_summary = f"PyTorch LSTM Model: {lookback}h lookback → 2 LSTM layers (50, 30) → Linear(1)"
dashboard.add_annotation(
    x=0.5,
    y=-0.15,
    xref="paper",
    yref="paper",
    text=model_summary,
    showarrow=False,
    font=dict(size=12),
    align="center"
)

# Save the dashboard
dashboard.write_html("pytorch_lstm_dashboard.html")

# Show the dashboard
dashboard.show()

print("\nSaved files:")
print("- lstm_training_history.png")
print("- lstm_scatter_plot.png")
print("- pytorch_lstm_prediction.html")
print("- pytorch_lstm_dashboard.html")

# Example command to run:
# python pytorch_lstm_road_temp.py --train_ratio 0.8 --lookback 24
