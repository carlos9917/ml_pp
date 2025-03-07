import sqlite3
import pandas as pd
import os
from collections import OrderedDict
DB = "/media/cap/extra_work/road_model/OBSTABLE"

features_cols = ['TROAD', 'T2m', 'Td2m', 'D10m', 'S10m', 'AccPcp12h']

#   valid_dttm       SID        lat        lon    TROAD      T2m     Td2m  D10m     S10m  AccPcp12h

# Define column types
column_types = {
  #'index': 'int64',  # Assuming this is the reset index column
  'valid_dttm': 'datetime64[ns]',  # For datetime columns
  'SID': 'int64', 
  'lat': 'float64',  
  'lon': 'float64',
  'TROAD': 'float64',
  'T2m': 'float64',
  'Td2m': 'float64',
  'D10m': 'float64',
  'S10m': 'float64',
  'AccPcp12h': 'float64'
}

def load_and_resample_data(variables, year, interval='30T'):
    dataframes = []
    
    for variable in variables:
        # Connect to the SQLite database for each variable
        conn = sqlite3.connect(os.path.join(DB, f'OBSTABLE_{variable}_{year}.sqlite'))
        
        # Use PRAGMA to optimize performance
        conn.execute('PRAGMA synchronous = OFF')
        conn.execute('PRAGMA journal_mode = MEMORY')
        
        query = f"SELECT valid_dttm, SID, lat, lon, {variable} FROM SYNOP"
        
        # Read data in chunks
        for chunk in pd.read_sql_query(query, conn, chunksize=10000):
            # Convert Unix time to datetime
            chunk['valid_dttm'] = pd.to_datetime(chunk['valid_dttm'], unit='s')
            dataframes.append(chunk)
        
        conn.close()
    
    # Concatenate all chunks into a single DataFrame
    full_df = pd.concat(dataframes, ignore_index=True)
    
    # Set 'valid_dttm' as the index for resampling
    full_df.set_index('valid_dttm', inplace=True)
    
    # Resample data for each station
    resampled_dataframes = []
    for sid, group in full_df.groupby('SID'):
        # Resample the group to the specified interval
        resampled_group = group.resample(interval).first()
        # Forward fill to handle missing data
        resampled_group.ffill(inplace=True)
        resampled_dataframes.append(resampled_group)
    
    # Combine all resampled dataframes
    resampled_df = pd.concat(resampled_dataframes).reset_index()
    resampled_df = resampled_df.astype(column_types)

   
    
    return resampled_df

# Define the variables and year
variables = ['TROAD', 'T2m', 'Td2m', 'D10m', 'S10m', 'AccPcp12h']
year = 2023
interval = '30T'  # Resample interval (e.g., '30T' for 30 minutes, '1H' for 1 hour)
interval = '1h'  # Resample interval (e.g., '30T' for 30 minutes, '1H' for 1 hour)
resampled_years=OrderedDict()
# Load and resample data
for year in [2022,2023]:
    resampled_years[year] = load_and_resample_data(variables, year, interval)

# Display the resampled DataFrame
resampled_data = resampled_years[2023]

# Group by SID and count missing values for each column
missing_values_by_station = resampled_data.groupby('SID')[features_cols].apply(lambda x: x.isna().sum()).reset_index()

# Calculate total missing values for each station
missing_values_by_station['total_missing'] = missing_values_by_station[features_cols].sum(axis=1)
import pdb
pdb.set_trace()
# Sort stations by total missing values (ascending)
station_ranking = missing_values_by_station.sort_values('total_missing')

# Display top 10 stations with least missing values
print("\nTop 10 stations with least missing values:")
hrint(station_ranking.head(10))
