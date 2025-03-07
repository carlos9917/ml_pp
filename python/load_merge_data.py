import sqlite3
import pandas as pd
from functools import reduce
import os
import pygwalker as pyg

DB="/media/cap/extra_work/road_model/OBSTABLE"

def load_and_merge_data(variables, year):
  dataframes = []
  
  for variable in variables:
      # Connect to the SQLite database for each variable
      conn = sqlite3.connect(os.path.join(DB,f'OBSTABLE_{variable}_{year}.sqlite'))
      query = "SELECT valid_dttm, SID, lat, lon, {} FROM SYNOP".format(variable)
      df = pd.read_sql_query(query, conn)
      conn.close()
      
      # Append the DataFrame to the list
      dataframes.append(df)
  
  # Merge all DataFrames on 'valid_dttm', 'SID', 'lat', and 'lon'
  merged_df = reduce(lambda left, right: pd.merge(left, right, on=['valid_dttm', 'SID', 'lat', 'lon'], how='outer'), dataframes)
  
  return merged_df

# Define the variables and year
variables = ['TROAD', 'T2m', 'Td2m', 'D10m', 'S10m', 'AccPcp12h']
year = 2023

# Load and merge data
merged_data = load_and_merge_data(variables, year)

# Display the merged DataFrame
print(merged_data.head())
print(merged_data.shape())


#walker = pyg.walk(df)

