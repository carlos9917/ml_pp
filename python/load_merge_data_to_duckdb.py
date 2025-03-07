import duckdb

DB="/media/cap/extra_work/road_model/OBSTABLE"

def load_and_merge_data_duckdb(variables, year):
  con = duckdb.connect(database=':memory:')
  dataframes = []
  
  for variable in variables:
      # Load data from each SQLite file into DuckDB
      query = f"SELECT valid_dttm, SID, lat, lon, {variable} FROM read_sqlite('OBSTABLE_{variable}_{year}.sqlite', 'SYNOP')"
      df = con.execute(query).df()
      dataframes.append(df)
  
  # Merge all DataFrames on 'valid_dttm', 'SID', 'lat', and 'lon'
  merged_df = dataframes[0]
  for df in dataframes[1:]:
      merged_df = merged_df.merge(df, on=['valid_dttm', 'SID', 'lat', 'lon'], how='outer')
  
  return merged_df

# Define the variables and year
variables = ['TROAD', 'T2m', 'Td2m', 'D10m', 'S10m', 'AccPcp12h']
year = 2023

# Load and merge data
merged_data = load_and_merge_data_duckdb(variables, year)

# Display the merged DataFrame
print(merged_data.head())
