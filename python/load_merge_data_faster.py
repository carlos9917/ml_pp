import sqlite3
import pandas as pd
import os

DB="/media/cap/extra_work/road_model/OBSTABLE"

def load_and_merge_data_optimized(variables, year):
  dataframes = []
  
  for variable in variables:
      # Connect to the SQLite database for each variable
      conn = sqlite3.connect(os.path.join(DB,f'OBSTABLE_{variable}_{year}.sqlite'))

      
      # Use PRAGMA to optimize performance
      conn.execute('PRAGMA synchronous = OFF')
      conn.execute('PRAGMA journal_mode = MEMORY')
      
      query = f"SELECT valid_dttm, SID, lat, lon, {variable} FROM SYNOP"
      
      # Read data in chunks
      for chunk in pd.read_sql_query(query, conn, chunksize=10000):
          dataframes.append(chunk)
      
      conn.close()
  
  # Concatenate all chunks into a single DataFrame
  full_df = pd.concat(dataframes, ignore_index=True)
  
  # Merge all DataFrames on 'valid_dttm', 'SID', 'lat', and 'lon'
  merged_df = full_df.groupby(['valid_dttm', 'SID', 'lat', 'lon']).first().reset_index()
  
  return merged_df

# Define the variables and year
variables = ['TROAD', 'T2m', 'Td2m', 'D10m', 'S10m', 'AccPcp12h']
year = 2023

# Load and merge data
merged_data = load_and_merge_data_optimized(variables, year)

# Display the merged DataFrame
print(merged_data.head())
