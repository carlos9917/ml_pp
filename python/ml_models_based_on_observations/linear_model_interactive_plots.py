"""
Linear model for predicting road temperatures (TROAD) using observation data.
This implementation ensures proper time-based train/test splitting and includes
interactive Plotly visualization.
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

####
# data loading
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

# --- Interactive Plotly Time Series Plot ---
# Create a subplot with 2 rows
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=("Road Temperature Over Time", "Prediction Error"),
    row_heights=[0.7, 0.3]
)

# Add training data
fig.add_trace(
    go.Scatter(
        x=train_df["dates"],
        y=train_df[target],
        mode='lines',
        name='Training Data',
        line=dict(color='blue', width=1),
        opacity=0.7
    ),
    row=1, col=1
)

# Add test actual data
fig.add_trace(
    go.Scatter(
        x=test_df["dates"],
        y=test_df[target],
        mode='lines',
        name='Test Actual',
        line=dict(color='green', width=2)
    ),
    row=1, col=1
)

# Add test predictions
fig.add_trace(
    go.Scatter(
        x=test_df["dates"],
        y=y_pred,
        mode='lines',
        name='Test Predictions',
        line=dict(color='orange', width=2, dash='dash')
    ),
    row=1, col=1
)

# Add vertical line for train/test split
# Convert to string format that Plotly can handle
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
prediction_error = test_df[target].values - y_pred
fig.add_trace(
    go.Scatter(
        x=test_df["dates"],
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
    title_text="Interactive Road Temperature Prediction Analysis",
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
fig.write_html("interactive_time_series.html")

# Show the figure
fig.show()

# Create a more detailed interactive dashboard with additional plots
dashboard = make_subplots(
    rows=2,
    cols=2,
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter", "colspan": 2}, None]
    ],
    subplot_titles=(
        "Actual vs Predicted",
        "Feature Importance",
        "Temperature Distribution Over Time"
    ),
    vertical_spacing=0.1,
    horizontal_spacing=0.1
)

# 1. Scatter plot of actual vs predicted
dashboard.add_trace(
    go.Scatter(
        x=y_test,
        y=y_pred,
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
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
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

# 2. Feature importance
linear_model = model.named_steps['linearregression']
coefficients = linear_model.coef_
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': np.abs(coefficients)
})
feature_importance = feature_importance.sort_values('Importance', ascending=True)

dashboard.add_trace(
    go.Bar(
        y=feature_importance['Feature'],
        x=feature_importance['Importance'],
        orientation='h',
        marker=dict(
            color=['rgba(55, 83, 109, 0.7)', 'rgba(26, 118, 255, 0.7)', 'rgba(0, 175, 0, 0.7)']
        ),
        name='Feature Importance'
    ),
    row=1, col=2
)

# 3. Temperature distribution over time with confidence intervals
# Extract month and year for grouping
df_cleaned['month_year'] = df_cleaned['dates'].dt.strftime('%Y-%m')

# Group by month-year for better visualization
monthly_stats = df_cleaned.groupby('month_year').agg(
    mean_troad=('TROAD', 'mean'),
    std_troad=('TROAD', 'std'),
    mean_t2m=('T2m', 'mean'),
    std_t2m=('T2m', 'std'),
    date=('dates', lambda x: x.iloc[0])
).reset_index()

# Sort by date for proper time series display
monthly_stats = monthly_stats.sort_values('date')

# Add TROAD monthly average
dashboard.add_trace(
    go.Scatter(
        x=monthly_stats['date'],
        y=monthly_stats['mean_troad'],
        mode='lines+markers',
        name='Avg Road Temp',
        line=dict(color='green', width=2)
    ),
    row=2, col=1
)

# Add confidence interval for TROAD
dashboard.add_trace(
    go.Scatter(
        x=monthly_stats['date'].tolist() + monthly_stats['date'].tolist()[::-1],
        y=(monthly_stats['mean_troad'] + monthly_stats['std_troad']).tolist() +
           (monthly_stats['mean_troad'] - monthly_stats['std_troad']).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0, 100, 0, 0.2)',
        line=dict(color='rgba(0, 100, 0, 0)'),
        name='Road Temp ±1σ'
    ),
    row=2, col=1
)

# Add T2m monthly average
dashboard.add_trace(
    go.Scatter(
        x=monthly_stats['date'],
        y=monthly_stats['mean_t2m'],
        mode='lines+markers',
        name='Avg Air Temp',
        line=dict(color='blue', width=2)
    ),
    row=2, col=1
)

# Add confidence interval for T2m
dashboard.add_trace(
    go.Scatter(
        x=monthly_stats['date'].tolist() + monthly_stats['date'].tolist()[::-1],
        y=(monthly_stats['mean_t2m'] + monthly_stats['std_t2m']).tolist() +
           (monthly_stats['mean_t2m'] - monthly_stats['std_t2m']).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.1)',
        line=dict(color='rgba(0, 0, 255, 0)'),
        name='Air Temp ±1σ'
    ),
    row=2, col=1
)

# Add vertical line for train/test split in the time series plot
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
    title_text="Road Temperature Prediction Dashboard",
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

dashboard.update_xaxes(title_text="Absolute Coefficient Value", row=1, col=2)
dashboard.update_yaxes(title_text="Feature", row=1, col=2)

dashboard.update_xaxes(title_text="Date", row=2, col=1)
dashboard.update_yaxes(title_text="Temperature (°C)", row=2, col=1)

# Add model equation annotation
equation = f"TROAD = {linear_model.intercept_:.2f}"
for i, feature in enumerate(features):
    equation += f" + {linear_model.coef_[i]:.2f}×{feature}"

dashboard.add_annotation(
    x=0.5,
    y=-0.15,
    xref="paper",
    yref="paper",
    text=f"Model Equation: {equation}",
    showarrow=False,
    font=dict(size=12),
    align="center"
)

# Save the dashboard
dashboard.write_html("road_temperature_dashboard.html")

# Show the dashboard
dashboard.show()

# Print model coefficients for interpretation
coefficients_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': linear_model.coef_
})
print("\nModel Coefficients:")
print(coefficients_df)
print(f"Intercept: {linear_model.intercept_}")
