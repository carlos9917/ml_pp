import argparse
import pandas as pd
import sqlite3
import os
DB = "/media/cap/extra_work/road_model/OBSTABLE"


from geopy.distance import geodesic
import folium
from folium import plugins

def find_closest_stations(stats_df,original_df, station_id, num_stations=5, min_clean_records=1000):
    """
    Find the closest stations to a given station that have sufficient data.

    Args:
    stats_df: DataFrame with station statistics
    station_id: ID of the reference station
    num_stations: Number of closest stations to return
    min_clean_records: Minimum number of clean records required

    Returns:
    DataFrame with closest stations and their distances
    """
    # Get reference station coordinates
    ref_station = original_df[original_df['SID'] == station_id].iloc[0]
    ref_coords = (ref_station['lat'], ref_station['lon'])

    # Calculate distances for all other stations with sufficient data
    distances = []
    for _, station in stats_df[stats_df['clean_records'] >= min_clean_records].iterrows():
        if station['SID'] != station_id:
            station_coords = (station['latitude'], station['longitude'])
            distance = geodesic(ref_coords, station_coords).kilometers
            distances.append({
                'SID': station['SID'],
                'distance_km': distance,
                'latitude': station['latitude'],
                'longitude': station['longitude'],
                'clean_records': station['clean_records']
            })

    # Convert to DataFrame and sort by distance
    distances_df = pd.DataFrame(distances)
    return distances_df.nlargest(num_stations, 'clean_records').sort_values('distance_km')

def plot_stations_map(ref_station_id, closest_stations_df, stats_df,original_df):
    """
    Create an interactive map showing the reference station and closest alternatives.

    Args:
    ref_station_id: ID of the reference station
    closest_stations_df: DataFrame with closest stations
    stats_df: Original statistics DataFrame

    Returns:
    None (saves an HTML file with the map)
    """
    # Get reference station info
    ref_station = original_df[original_df['SID'] == ref_station_id].iloc[0]

    # Create map centered on reference station
    m = folium.Map(
        location=[ref_station['lat'], ref_station['lon']],
        zoom_start=8
    )

    # Add reference station marker
    folium.Marker(
        [ref_station['lat'], ref_station['lon']],
        #popup=f"Reference Station\nID: {ref_station_id}\nClean Records: {ref_station['clean_records']}",
        popup=f"Reference Station\nID: {ref_station_id}",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

    # Add closest stations markers
    for _, station in closest_stations_df.iterrows():
        folium.Marker(
            [station['latitude'], station['longitude']],
            popup=f"Station ID: {station['SID']}\nDistance: {station['distance_km']:.2f} km\nClean Records: {station['clean_records']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

        # Draw line to reference station
        folium.PolyLine(
            locations=[[ref_station['lat'], ref_station['lon']],
                      [station['latitude'], station['longitude']]],
            weight=2,
            color='gray',
            popup=f"Distance: {station['distance_km']:.2f} km"
        ).add_to(m)

    # Save map
    map_file = f'station_map_{ref_station_id}.html'
    m.save(map_file)
    print(f"\nInteractive map saved to: {map_file}")
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

def analyze_stations(year,variables = ['TROAD', 'T2m', 'Td2m', 'D10m', 'S10m', 'AccPcp12h']):
    """
    Analyze which stations have valid data for the given year.

    Args:
    year: Year to analyze

    Returns:
    DataFrame with station statistics
    """
    print(f"\nAnalyzing stations for year {year}...")

    # Load data
    #variables = ['TROAD', 'T2m', 'Td2m', 'D10m', 'S10m', 'AccPcp12h']
    df = load_and_merge_data_optimized(variables, year)

    # Initialize results dictionary
    station_stats = []

    # Analyze each station
    for station in df['SID'].unique():
        station_data = df[df['SID'] == station]

        # Get the key columns we need for the model
        station_clean = station_data[["valid_dttm", "SID", "lat", "lon"]+variables].dropna()

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


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyse stations availability and data')
    parser.add_argument('--year', type=int, required=True,
                       help='Year for which to run the model (e.g., 2023)')
    parser.add_argument('--station', type=int, required=True,
                       help='Station to check if it is in in the list')
    args = parser.parse_args()
    #it will check for these variables:
    variables = ['TROAD', 'T2m', 'Td2m', 'D10m', 'S10m', 'AccPcp12h']
    variables = ['TROAD', 'T2m', 'Td2m','S10m']
    stats_df = analyze_stations(args.year,variables)
    if stats_df is None:
        return

    # Check if the requested station is in the valid stations list
    if args.station not in stats_df['SID'].values:
        print(f"\nWarning: Station {args.station} not found in the valid stations list.")
        print("Please choose from the stations listed in the statistics file.")

        # Find closest stations with sufficient data
        print("\nFinding closest alternative stations...")
        original_df = load_and_merge_data_optimized(variables, args.year)
        closest_stations = find_closest_stations(stats_df,original_df, args.station)
        print("\nClosest alternative stations:")
        print(closest_stations.to_string())

        # Create and save interactive map
        plot_stations_map(args.station, closest_stations, stats_df,original_df)
        return


if __name__ == "__main__":
    main()
