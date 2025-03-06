import pandas as pd
import glob
import os
from datetime import datetime

def combine_parquet_files_by_day(input_directory, output_directory):
    """
    Combines hourly parquet files into daily files
    
    Parameters:
    input_directory: str - directory containing hourly parquet files
    output_directory: str - directory where daily files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Get all hourly parquet files in the input directory
    parquet_files = glob.glob(os.path.join(input_directory, 'road_temp_*.parquet'))
    
    # Dictionary to store DataFrames by day
    daily_data = {}
    
    print("Processing hourly files...")
    
    for file_path in parquet_files:
        try:
            # Extract date from filename
            filename = os.path.basename(file_path)
            # Assuming filename format: road_temp_YYYYMMDDHH.parquet
            date_str = filename[10:18]  # Extract YYYYMMDD part
            
            # Read the parquet file
            df = pd.read_parquet(file_path)
            
            # Group by day
            if date_str not in daily_data:
                daily_data[date_str] = []
            daily_data[date_str].append(df)
                
            print(f"Processed: {file_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    print("\nCombining and saving daily files...")
    
    # Combine and save daily files
    for date_str, df_list in daily_data.items():
        try:
            # Combine all DataFrames for this day
            combined_df = pd.concat(df_list, ignore_index=True)
            
            # Sort by timestamp and station_id
            combined_df = combined_df.sort_values(['timestamp', 'station_id'])
            
            # Remove duplicates if any
            combined_df = combined_df.drop_duplicates()
            
            # Save to parquet file
            output_file = os.path.join(output_directory, f'road_temp_{date_str}.parquet')
            combined_df.to_parquet(output_file, index=False)
            
            print(f"Created: {output_file} with {len(combined_df)} records")
            
        except Exception as e:
            print(f"Error saving day {date_str}: {str(e)}")
    
    return list(daily_data.keys())

def verify_daily_files(output_directory, date_list):
    """
    Verify the created daily files
    
    Parameters:
    output_directory: str - directory containing daily files
    date_list: list - list of dates that were processed
    """
    print("\nVerifying daily files:")
    
    for date_str in date_list:
        file_path = os.path.join(output_directory, f'road_temp_{date_str}.parquet')
        try:
            df = pd.read_parquet(file_path)
            print(f"\nDate {date_str}:")
            print(f"Number of records: {len(df)}")
            print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Number of unique stations: {df['station_id'].nunique()}")
            print(f"Number of hourly timestamps: {df['timestamp'].nunique()}")
            print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"Error verifying {file_path}: {str(e)}")

if __name__ == "__main__":
    # Define directories
    input_dir = "/data/projects/glatmodel/obs/fild8/road_profiles"
    output_dir = "/data/projects/glatmodel/obs/fild8/road_temp_daily"

    
    # Combine files
    processed_dates = combine_parquet_files_by_day(input_dir, output_dir)
    
    # Verify results
    verify_daily_files(output_dir, processed_dates)
    
