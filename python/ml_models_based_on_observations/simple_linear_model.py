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
import argparse
from datetime import datetime

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

def analyze_stations(year):
    """
    Analyze which stations have valid data for the given year.

    Args:
    year: Year to analyze

    Returns:
    DataFrame with station statistics
    """
    print(f"\nAnalyzing stations for year {year}...")

    # Load data
    variables = ['TROAD', 'T2m', 'Td2m', 'D10m', 'S10m', 'AccPcp12h']
    df = load_and_merge_data_optimized(variables, year)

    # Initialize results dictionary
    station_stats = []

    # Analyze each station
    for station in df['SID'].unique():
        station_data = df[df['SID'] == station]

        # Get the key columns we need for the model
        station_clean = station_data[["valid_dttm", "SID", "lat", "lon", "TROAD", "T2m"]].dropna()

        # Calculate statistics
        total_records = len(station_data)
        clean_records = len(station_clean)
        if clean_records > 0:
            date_range = pd.to_datetime(station_clean['valid_dttm'], unit='s')
            date_min = date_range.min()
            date_max = date_range.max()
            date_coverage = (date_max - date_min).days + 1

            # Get location info
            lat = station_clean['lat'].iloc[0]
            lon = station_clean['lon'].iloc[0]

            station_stats.append({
                'SID': int(station),  # Convert to int
                'total_records': total_records,
                'clean_records': clean_records,
                'data_completeness': clean_records / total_records * 100,
                'start_date': date_min,
                'end_date': date_max,
                'days_coverage': date_coverage,
                'latitude': lat,
                'longitude': lon
            })

    # Convert to DataFrame and sort by clean records
    stats_df = pd.DataFrame(station_stats)
    if len(stats_df) > 0:
        stats_df = stats_df.sort_values('clean_records', ascending=False)

        # Print summary
        print("\nStation Analysis Summary:")
        print(f"Total number of stations: {len(stats_df)}")
        print(f"Stations with clean data: {len(stats_df[stats_df['clean_records'] > 0])}")

        # Print detailed statistics for stations with data
        print("\nTop 10 stations with most clean records:")
        pd.set_option('display.max_columns', None)
        print(stats_df.head(10).to_string())

        # Save full statistics to CSV
        output_file = f'station_statistics_{year}.csv'
        stats_df.to_csv(output_file, index=False)
        print(f"\nFull station statistics saved to: {output_file}")

        return stats_df
    else:
        print("No valid stations found for the specified year.")
        return None

def train_and_evaluate_model(df_cleaned, features, target, station, year):
    """
    Train and evaluate the model, create plots
    """
    # Sort data chronologically
    df_cleaned = df_cleaned.sort_values("valid_dttm")

    # Time-based train-test split (80-20)
    split_idx = int(len(df_cleaned) * 0.8)
    train_df = df_cleaned.iloc[:split_idx]
    test_df = df_cleaned.iloc[split_idx:]

    # Create train and test sets
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Create and train model
    model = make_pipeline(SimpleImputer(strategy='mean'), LinearRegression())
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f'\nResults for Station {station}:')
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'Root Mean Squared Error: {rmse:.2f}')

    # Create plots
    create_scatter_plot(y_test, y_pred, station, year)
    create_time_series_plot(train_df, test_df, y_pred, target, station, year)

    # Print model coefficients
    linear_model = model.named_steps['linearregression']
    coefficients = pd.DataFrame({
        'Feature': features,
        'Coefficient': linear_model.coef_
    })
    print("\nModel Coefficients:")
    print(coefficients)
    print(f"Intercept: {linear_model.intercept_:.2f}")

def create_scatter_plot(y_test, y_pred, station, year):
    """
    Create and save scatter plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('Actual Road Temperature')
    plt.ylabel('Predicted Road Temperature')
    plt.title(f'Linear Regression: Actual vs Predicted Road Temperature\nStation: {station}, Year: {year}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'scatter_plot_{station}_{year}.png')
    plt.close()

def create_time_series_plot(train_df, test_df, y_pred, target, station, year):
    """
    Create and save time series plot
    """
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
    plt.title(f"Actual vs Predicted Road Temperature Over Time\nStation: {station}, Year: {year}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'time_series_plot_{station}_{year}.png')
    plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Road temperature prediction model')
    parser.add_argument('--year', type=int, required=True,
                       help='Year for which to run the model (e.g., 2023)')
    parser.add_argument('--station', type=int, required=True,
                       help='Station ID (SID) for which to run the prediction')
    parser.add_argument('--analyze-stations', action='store_true',
                       help='Analyze available stations before running the model')
    args = parser.parse_args()

    # If analyze-stations flag is set, run the analysis first
    if args.analyze_stations:
        stats_df = analyze_stations(args.year)
        if stats_df is None:
            return

        # Check if the requested station is in the valid stations list
        if args.station not in stats_df['SID'].values:
            print(f"\nWarning: Station {args.station} not found in the valid stations list.")
            print("Please choose from the stations listed in the statistics file.")
            return

    # Define variables and load data
    variables = ['TROAD', 'T2m', 'Td2m', 'D10m', 'S10m', 'AccPcp12h']
    print(f"Loading data for year {args.year}...")
    df = load_and_merge_data_optimized(variables, args.year)

    # Filter for specific station
    df = df[df['SID'] == args.station]

    if len(df) == 0:
        print(f"No data found for station {args.station} in year {args.year}")
        return

    print(f"Processing data for station {args.station}...")

    # Drop rows with missing values in key columns
    df_cleaned = df[["valid_dttm", "SID", "lat", "lon", "TROAD", "T2m"]].dropna()

    if len(df_cleaned) == 0:
        print(f"No valid data found for station {args.station} after cleaning")
        return

    # Convert Unix timestamp to datetime
    df_cleaned["dates"] = pd.to_datetime(df_cleaned["valid_dttm"], unit="s")

    # Define features and target
    features = ['lat', 'lon', 'T2m']
    target = 'TROAD'

    # Train and evaluate model
    train_and_evaluate_model(df_cleaned, features, target, args.station, args.year)

if __name__ == "__main__":
    main()
