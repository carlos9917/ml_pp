import pandas as pd
import numpy as np
from pykrige.uk import UniversalKriging
from sklearn.preprocessing import StandardScaler
import sqlite3
import os
from datetime import datetime

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


# Define the variables and year
DB = "/media/cap/extra_work/road_model/OBSTABLE"
variables = ['TROAD', 'T2m', 'Td2m', 'D10m', 'S10m', 'AccPcp12h']
year = 2023

# Load all available data
print("Loading all available data...")
df = load_and_merge_data_optimized(variables, year)

## load the dem data previously pre processed

# Replace 'stations.csv' with your actual CSV file path
station_gis_metrics = pd.read_csv('/media/cap/extra_work/road_model/gistools/height_calc_DSM/station_metrics.csv')

#drop these columns from the gis data, since they are already included in df
columns_to_drop = ['lat', 'lon']
cleaned = station_gis_metrics.drop(columns=columns_to_drop)

merged = df.merge(
    cleaned,
    left_on='SID',
    right_on='station_id',
    how='inner'
)

del df
merged["dates"] = pd.to_datetime(merged["valid_dttm"], unit="s")
# Stack the covariates
#X_pred = np.column_stack([elev_pred, slope_pred, aspect_pred])

df = merged[merged["dates"]==datetime(2023,1,11,2)]


#selected_points_df = show_locations_for_kriging(df)
#selected_points_df = select_locations_for_kriging(df)

# Prepare coordinates and features
coords = df[['lon', 'lat']].values
values = df['TROAD'].values

# Select covariates
covariates = ['elev_m', 'slope_deg', 'aspect_deg']
X = df[covariates].values

# Scale covariates
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Universal Kriging with specified drift terms
UK = UniversalKriging(
    df['lon'], df['lat'], df['TROAD'],
    drift_terms=['specified'],
    specified_drift=[X_scaled[:, i] for i in range(X_scaled.shape[1])],  # All your covariates
    variogram_model='exponential'
)


#selected_kriging_points = pd.read_csv('selected_kriging_points.csv')

# Load the new points where we want to predict TROAD
new_points = pd.read_csv('/media/cap/extra_work/road_model/gistools/height_calc_DSM/station_metrics_kriging_points.csv')
print(f"Loaded {len(new_points)} new points for prediction")

# Prepare coordinates for prediction
pred_coords = new_points[['lon', 'lat']].values

# Prepare covariates for prediction (same as used in training)
covariates = ['elev_m', 'slope_deg', 'aspect_deg']
X_pred = new_points[covariates].values

# Scale covariates using the same scaler used for training
X_pred_scaled = scaler.transform(X_pred)

# Predict TROAD at new locations using Universal Kriging
pred_troad, pred_var = UK.execute('points', pred_coords[:, 0], pred_coords[:, 1],
                                  specified_drift_arrays=[X_pred_scaled[:, i] for i in range(X_pred_scaled.shape[1])])

# Add predictions to the dataframe
new_points['TROAD_predicted'] = pred_troad
new_points['TROAD_variance'] = pred_var

# Save results to a new CSV file
new_points.to_csv('station_metrics_with_troad_predictions.csv', index=False)

print("Prediction complete. Results saved to 'station_metrics_with_troad_predictions.csv'")
print(new_points[['station_id', 'lat', 'lon', 'TROAD_predicted']].head())


#### plotting part

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from matplotlib.colors import Normalize
import matplotlib.lines as mlines

# Convert dataframes to GeoDataFrames
def df_to_gdf(df, lon_col='lon', lat_col='lat', crs='EPSG:4326'):
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)

# Create GeoDataFrames
gdf_original = df_to_gdf(df)
gdf_predicted = df_to_gdf(new_points)


# Add a column to identify the source
gdf_original['source'] = 'Original'
gdf_predicted['source'] = 'Predicted'

# Rename the temperature column in predicted data to match original
gdf_predicted = gdf_predicted.rename(columns={'TROAD_predicted': 'TROAD'})

# Combine the datasets
gdf_combined = pd.concat([gdf_original, gdf_predicted])

# Convert to Web Mercator for basemap compatibility
gdf_combined = gdf_combined.to_crs(epsg=3857)

# Define colormap for temperature
vmin = gdf_combined['TROAD'].min()
vmax = gdf_combined['TROAD'].max()
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.coolwarm

# Create a figure
fig, ax = plt.subplots(figsize=(15, 12))

# Plot all stations with different markers based on source
for source, marker in [('Original', 'o'), ('Predicted', '^')]:
    subset = gdf_combined[gdf_combined['source'] == source]
    subset.plot(column='TROAD', cmap=cmap, norm=norm,
                markersize=80, marker=marker, edgecolor='black',
                linewidth=1, alpha=0.8, ax=ax, zorder=2)

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zorder=0)

# Add colorbar
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label('Temperature (°C)', fontsize=14)

# Create legend handles using Line2D
orig_legend = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=10, markeredgecolor='black', label='Original Stations')
pred_legend = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                          markersize=10, markeredgecolor='black', label='Predicted Stations')
ax.legend(handles=[orig_legend, pred_legend], fontsize=12, loc='lower right')

# Add title and remove axes
ax.set_title('TROAD Values - All Stations', fontsize=16)
ax.set_axis_off()


# Create a second visualization showing the difference between nearby points
fig2, ax2 = plt.subplots(figsize=(15, 12))
# ===== SUBPLOT 2: Kriging Uncertainty =====
# Define colormap for uncertainty (variance)
vmin_var = gdf_predicted['TROAD_variance'].min()
vmax_var = gdf_predicted['TROAD_variance'].max()
norm_var = Normalize(vmin=vmin_var, vmax=vmax_var)
cmap_var = plt.cm.viridis  # Different colormap for uncertainty

# Plot only predicted stations colored by uncertainty
gdf_predicted.plot(column='TROAD_variance', cmap=cmap_var, norm=norm_var,
                           markersize=80, marker='^', edgecolor='black',
                           linewidth=1, alpha=0.8, ax=ax2, zorder=2)

# Add basemap to second subplot
ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron, zorder=0)

# Add colorbar for uncertainty
cbar2 = fig.colorbar(plt.cm.ScalarMappable(norm=norm_var, cmap=cmap_var), ax=ax2, pad=0.01, shrink=0.8)
#  cbar =  fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar2.set_label('Kriging Variance (Uncertainty)', fontsize=12)

# Create legend for uncertainty plot
uncert_legend = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                             markersize=10, markeredgecolor='black', label='Predicted Stations')
ax2.legend(handles=[uncert_legend], fontsize=10, loc='lower right')

ax2.set_title('Kriging Uncertainty (Variance) - Predicted Stations', fontsize=14)
ax2.set_axis_off()

plt.tight_layout()
plt.show()
