import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os

#def load_data(variable, year):
#  conn = sqlite3.connect(f'OBSTABLE_{variable}_{year}.sqlite')
#  query = "SELECT * FROM observations"
#  df = pd.read_sql_query(query, conn)
#  conn.close()
#  return df
#
## Load data
#variable = 'temperature'
#year = 2023
#df = load_data(variable, year)

#############################
# data loading
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
df = load_and_merge_data_optimized(variables, year)

# Display the merged DataFrame
#print(df.head())
#######################################

# Preprocess data: Assume 'temperature' is the target and others are features
# For simplicity, let's assume the DataFrame has columns ['SID', 'lat', 'lon', 'temperature', 'humidity']
features = df[['lat', 'lon', 'Td2m',"T2m"]]
target = df['TROAD']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
