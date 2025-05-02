"""
Linear model for predicting road temperatures (TROAD) using observation data.
This implementation ensures proper time-based train/test splitting.
"""
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Data source path
DB = "/media/cap/extra_work/road_model/OBSTABLE"

def load_and_merge_data_optimized(variables, year):
    """
    Load data from SQLite databases and merge into a single dataframe.

    Args:
        variables: List of meteorological variables to load
        year: Year of data to load

    Returns:
        DataFrame with merged data from all variables
    """
    dataframes = []
    for variable in variables:
        conn = sqlite3.connect(os.path.join(DB, f'OBSTABLE_{variable}_{year}.sqlite'))
        # Optimize SQLite performance
        conn.execute('PRAGMA synchronous = OFF')
        conn.execute('PRAGMA journal_mode = MEMORY')
        query = f"SELECT valid_dttm, SID, lat, lon, {variable} FROM SYNOP"
        for chunk in pd.read_sql_query(query, conn, chunksize=10000):
            dataframes.append(chunk)
        conn.close()

    # Merge all dataframes
    full_df = pd.concat(dataframes, ignore_index=True)
    merged_df = full_df.groupby(['valid_dttm', 'SID', 'lat', 'lon']).first().reset_index()
    return merged_df

# Define the variables and year
variables = ['TROAD', 'T2m', 'Td2m', 'D10m', 'S10m', 'AccPcp12h']
year = 2023

# Load and merge data
df = load_and_merge_data_optimized(variables, year)

# Drop rows with missing values in key columns
df_cleaned = df[["valid_dttm", "SID", "lat", "lon", "TROAD", "T2m"]].dropna()

# Convert Unix timestamp to datetime for better handling
df_cleaned["dates"] = pd.to_datetime(df_cleaned["valid_dttm"], unit="s")

# Sort data chronologically
df_cleaned = df_cleaned.sort_values("valid_dttm")

# Time-based train-test split (e.g., using the first 80% for training, last 20% for testing)
split_idx = int(len(df_cleaned) * 0.8)
train_df = df_cleaned.iloc[:split_idx]
test_df = df_cleaned.iloc[split_idx:]

# Define features and target
features = ['lat', 'lon', 'T2m']
target = 'TROAD'

# Create train and test sets
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Create a pipeline with an imputer and linear regression
model = make_pipeline(SimpleImputer(strategy='mean'), LinearRegression())

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# --- Scatter Plot: Actual vs Predicted ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Road Temperature')
plt.ylabel('Predicted Road Temperature')
plt.title('Linear Regression: Actual vs Predicted Road Temperature')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('scatter_plot.png')
plt.show()

# --- Time Series Plot ---
plt.figure(figsize=(16, 7))

# Plot training data
plt.plot(train_df["dates"], train_df[target],
         label="Training Actual TROAD", color="blue", linewidth=1, alpha=0.7)

# Plot test actual values
plt.plot(test_df["dates"], test_df[target],
         label="Test Actual TROAD", color="green", linewidth=2)

# Plot test predictions
plt.plot(test_df["dates"], y_pred,
         label="Test Predicted TROAD", color="orange", linestyle="--", linewidth=2)

# Format x-axis for date and time
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)

# Add vertical line to show train/test split
split_time = train_df["dates"].iloc[-1]
plt.axvline(x=split_time, color='r', linestyle='-', alpha=0.5, label='Train/Test Split')

plt.xlabel("Time")
plt.ylabel("Road Temperature (TROAD)")
plt.title("Actual vs Predicted Road Temperature Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('time_series_plot.png')
plt.show()

# Optional: Print model coefficients for interpretation
linear_model = model.named_steps['linearregression']
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': linear_model.coef_
})
print("\nModel Coefficients:")
print(coefficients)
print(f"Intercept: {linear_model.intercept_}")
