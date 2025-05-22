import pandas as pd
import numpy as np
from pykrige.uk import UniversalKriging
from sklearn.preprocessing import StandardScaler
import sqlite3
import os
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import json

import dash_leaflet as dl
import dash_leaflet.express as dlx

def create_leaflet_app(df, lat_col='lat', lon_col='lon'):
    # Create markers for existing stations
    markers = [
        dl.Marker(dl.Tooltip(str(row['SID']) if 'SID' in row else str(i)),
                  position=[row[lat_col], row[lon_col]])
        for i, row in df.iterrows()
    ]

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Select Locations for Kriging"),
        dl.Map(
            [
                dl.TileLayer(),
                dl.LayerGroup(markers, id="stations"),
                dl.LayerGroup(id="selected-points")
            ],
            id="map",
            center=[df[lat_col].mean(), df[lon_col].mean()],
            zoom=8,
            style={'width': '100%', 'height': '700px'}
        ),
        html.Button("Clear Selected Points", id="clear-btn"),
        html.Button("Save Selected Points", id="save-btn"),
        html.Div(id="selected-points-display"),
        html.Div(id="save-status")
    ])

    # Store selected points in a hidden div
    selected_points = []
    @app.callback(
    Output("selected-points", "children"),
    Output("selected-points-display", "children"),
    [Input("map", "click_lat_lng"),
     Input("clear-btn", "n_clicks")],
    [State("selected-points", "children")],
    prevent_initial_call=True)

    def map_click(click_lat_lng, clear_clicks, children):
        ctx = dash.callback_context
        trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
        # Clear button
        if trigger == 'clear-btn':
            return [], "No points selected yet."
    
        # Map click
        if trigger == 'map' and click_lat_lng is not None:
            lat, lon = click_lat_lng
            marker = dl.Marker(position=[lat, lon], children=dl.Tooltip(f"Lat: {lat:.6f}, Lon: {lon:.6f}"))
            children = children or []
            children.append(marker)
    
            # Display selected points
            points = [[m['props']['position'][0], m['props']['position'][1]] for m in children]
            display = html.Ul([html.Li(f"Lat: {lat:.6f}, Lon: {lon:.6f}") for lat, lon in points])
            return children, display
    
        return dash.no_update, dash.no_update

    @app.callback(
        Output("save-status", "children"),
        Input("save-btn", "n_clicks"),
        State("selected-points", "children"),
        prevent_initial_call=True
    )
    def save_points(n_clicks, children):
        if children:
            points = [[m['props']['position'][0], m['props']['position'][1]] for m in children]
            df_points = pd.DataFrame(points, columns=['lat', 'lon'])
            df_points.to_csv("selected_kriging_points.csv", index=False)
            return f"Saved {len(points)} points to selected_kriging_points.csv"
        return "No points to save."

    return app

def select_locations_for_kriging(df, lat_col='lat', lon_col='lon'):
    app = create_leaflet_app(df, lat_col, lon_col)
    print("Open http://127.0.0.1:8050/ in your browser.")
    app.run(debug=False)
    # After closing, you can load the points from CSV
    if os.path.exists('selected_kriging_points.csv'):
        return pd.read_csv('selected_kriging_points.csv')
    return None


def create_interactive_map_app(df, lat_col='lat', lon_col='lon'):
    """
    Creates a Dash web application with an interactive map for selecting points.
    
    Args:
        df: DataFrame containing station data
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        
    Returns:
        A Dash app that can be run with app.run_server()
    """
    app = dash.Dash(__name__)
    
    # Create the initial map with existing stations
    fig = px.scatter_map(
        df, 
        lat=lat_col, 
        lon=lon_col,
        hover_name=df.index if 'SID' not in df.columns else df['SID'],
        zoom=8,
        height=700
        #style="open-street-map"
    )
    
    # Add layout for the app
    app.layout = html.Div([
        html.H1("Select Locations for Kriging"),
        html.Div([
            html.P("Click on the map to select new locations for kriging prediction."),
            html.P("Existing stations are shown in blue."),
            html.Button('Clear Selected Points', id='clear-button', n_clicks=0),
            html.Button('Save Selected Points', id='save-button', n_clicks=0),
        ]),
        dcc.Graph(id='map', figure=fig),
        html.Div(id='selected-points-display'),
        # Hidden div to store the selected points
        html.Div(id='selected-points-store', style={'display': 'none'})
    ])
    
    @app.callback(
        [Output('map', 'figure'),
         Output('selected-points-store', 'children'),
         Output('selected-points-display', 'children')],
        [Input('map', 'clickData'),
         Input('clear-button', 'n_clicks')],
        [State('selected-points-store', 'children')]
    )
    def update_map(clickData, clear_clicks, stored_points):
        ctx = dash.callback_context
        
        # Initialize points list
        if stored_points is None:
            points = []
        else:
            points = json.loads(stored_points)
        
        # Handle clear button click
        if ctx.triggered_id == 'clear-button':
            points = []
        
        # Handle map click
        elif clickData is not None and ctx.triggered_id == 'map':
            point = clickData['points'][0]
            lat, lon = point['lat'], point['lon']
            points.append({'lat': lat, 'lon': lon})
        
        # Update the figure with existing stations and selected points
        fig = px.scatter_map(
            df, 
            lat=lat_col, 
            lon=lon_col,
            hover_name=df.index if 'SID' not in df.columns else df['SID'],
            zoom=8,
            height=700
            #style="open-street-map"
        )
        
        # Add selected points in red
        if points:
            selected_df = pd.DataFrame(points)
            fig.add_trace(
                px.scatter_map(
                    selected_df, 
                    lat='lat', 
                    lon='lon',
                    color_discrete_sequence=['red'],
                    size=[10] * len(selected_df)
                ).data[0]
            )
        
        # Display the selected points
        points_display = html.Div([
            html.H3(f"Selected Points ({len(points)}):"),
            html.Ul([html.Li(f"Lat: {p['lat']:.6f}, Lon: {p['lon']:.6f}") for p in points])
        ]) if points else html.Div("No points selected yet.")
        
        return fig, json.dumps(points), points_display
    
    @app.callback(
        Output('selected-points-display', 'children', allow_duplicate=True),
        Input('save-button', 'n_clicks'),
        State('selected-points-store', 'children'),
        prevent_initial_call=True
    )
    def save_points(n_clicks, stored_points):
        if n_clicks > 0 and stored_points:
            points = json.loads(stored_points)
            # Save to CSV
            pd.DataFrame(points).to_csv('selected_kriging_points.csv', index=False)
            return html.Div([
                html.H3("Points Saved!"),
                html.P("Selected points have been saved to 'selected_kriging_points.csv'"),
                html.Ul([html.Li(f"Lat: {p['lat']:.6f}, Lon: {p['lon']:.6f}") for p in points])
            ])
        return dash.no_update
    
    return app

# Function to run the app and return selected points
def show_locations_for_kriging(df, lat_col='lat', lon_col='lon'):
    """
    Run the interactive map app and return selected points.
    
    Args:
        df: DataFrame with station data
        lat_col: Column name for latitude
        lon_col: Column name for longitude
    
    Returns:
        The app will save selected points to a CSV file
    """
    app = create_interactive_map_app(df, lat_col, lon_col)
    print("Starting interactive map server. Please open a web browser and go to http://127.0.0.1:8050/")
    print("Click on the map to select points for kriging prediction.")
    print("When finished, click 'Save Selected Points' and close the browser window.")
    app.run(debug=False)
    
    # The points will be saved to CSV by the app
    print("Selected points have been saved to 'selected_kriging_points.csv'")
    
    # If you want to return the points as well
    if os.path.exists('selected_kriging_points.csv'):
        return pd.read_csv('selected_kriging_points.csv')
    return None

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
selected_points_df = select_locations_for_kriging(df)

# Prepare coordinates and features
coords = df[['lon', 'lat']].values
values = df['TROAD'].values

# Select covariates
covariates = ['elev_m', 'slope_deg', 'aspect_deg']
X = df[covariates].values

# Scale covariates
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Universal Kriging with multiple external drifts
#UK = UniversalKriging(
#    df['lon'], df['lat'], df['TROAD'],
#    drift_terms=['external_Z'], # Specifies that we're using external covariates
#    external_drift=X_scaled, # The actual array of external covariates
#    external_drift_coordinates=coords,  # Coordinates for external drift
#    variogram_model='exponential'
#)


# Universal Kriging with multiple external drifts
#UK = UniversalKriging(
#    df['lon'], df['lat'], df['TROAD'],
#    drift_terms=['external_Z'],  # Specifies that we're using external covariates
#    external_drift=X_scaled,  # The actual array of external covariates
#    variogram_model='exponential'
#)

# Universal Kriging with specified drift terms
UK = UniversalKriging(
    df['lon'], df['lat'], df['TROAD'],
    drift_terms=['specified'],
    specified_drift=[X_scaled[:, i] for i in range(X_scaled.shape[1])],  # All your covariates
    variogram_model='exponential'
)



## Define grid for prediction
#grid_lon = np.linspace(df['lon'].min(), df['lon'].max(), 100)
#grid_lat = np.linspace(df['lat'].min(), df['lat'].max(), 100)
#gridx, gridy = np.meshgrid(grid_lon, grid_lat)
#
## Prepare grid covariates (interpolated orography for grid points)
#grid_elev = ...  # Interpolated elevation for grid points
#grid_slope = ...  # Interpolated slope for grid points
#grid_aspect = ...  # Interpolated aspect for grid points
#grid_X = np.column_stack([grid_elev, grid_slope, grid_aspect])
#grid_X_scaled = scaler.transform(grid_X)
#
## Predict
#z_pred, ss = UK.execute('grid', grid_lon, grid_lat, external_drift_grid=grid_X_scaled)
