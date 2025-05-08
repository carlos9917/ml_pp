import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Function to get station coordinates from SQLite database
def get_station_coordinates(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT SID, lon, lat FROM roadstations")
    stations = {str(sid): (lat, lon) for sid, lon, lat in cursor.fetchall()}
    conn.close()
    return stations

# Function to create and save the map
def create_station_map(stations_coords, chosen_stations, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all stations
    all_lats = []
    all_lons = []
    chosen_lats = []
    chosen_lons = []
    
    # Convert chosen station IDs to match the database format
    chosen_ids = [s.split('-')[1] for s in chosen_stations]
    
    for station_id, (lat, lon) in stations_coords.items():
        if station_id in chosen_ids:
            chosen_lats.append(lat)
            chosen_lons.append(lon)
        else:
            all_lats.append(lat)
            all_lons.append(lon)
    
    # Plot all stations in blue (smaller markers)
    ax.scatter(all_lons, all_lats, c='blue', s=30, alpha=0.5, label='All stations')
    import pdb
    pdb.set_trace()

    # Plot chosen stations in red (larger markers)
    if chosen_lats:
        ax.scatter(chosen_lons, chosen_lats, c='red', s=100, label='Selected stations')
    
    # Add labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Road Stations Map')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add some padding to the bounds
    padding = 0.5
    ax.set_xlim(min(all_lons + chosen_lons) - padding, max(all_lons + chosen_lons) + padding)
    ax.set_ylim(min(all_lats + chosen_lats) - padding, max(all_lats + chosen_lats) + padding)
    
    # Save the map
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Create a test database
#conn = sqlite3.connect('stations_coords_height.db')
#cursor = conn.cursor()
#cursor.execute('DROP TABLE IF EXISTS roadstations')
#cursor.execute('''CREATE TABLE roadstations
#                 (SID INTEGER, lon FLOAT, lat FLOAT, height FLOAT)''')
#
# Add some test data (Danish coordinates)
#test_data = [
#    (100001, 9.5, 56.2, 100),   # Somewhere in Jutland
#    (100002, 10.0, 56.5, 120),  # Near Aarhus
#    (100003, 12.8, 55.8, 90),   # Near Copenhagen
#    (100004, 11.2, 55.3, 110),  # Southern Denmark
#    (100005, 10.7, 55.6, 95)    # Odense area
#]
#cursor.executemany('INSERT INTO roadstations VALUES (?,?,?,?)', test_data)
#conn.commit()
#conn.close()

# Test the map creation
db_coords="../../../stations_coords_height.db"
stations_coords = get_station_coordinates(db_coords)
#chosen_stations = ['0-100001-0', '0-100002-0']  # Example of two chosen stations
chosen_stations= [ '15-137580-4', '49-137640-30', '16-137040-85', '26-137020-5', '7-137360-1', '30-136950-211', '26-137100-483', '14-137200-205', '15-136900-9', '20-137330-542']
chosen_stations= [ '0-630200-0']
chosen_stations = ['0-503100-0', '0-326100-0', '0-522200-0', '0-991100-0', '0-402000-0', '0-522100-0', '0-300700-0', '0-500800-0', '0-400300-0', '0-600000-0', '0-500700-0', '0-433000-0', '0-180000-0', '0-181000-0', '0-181500-0', '0-500600-0']

#chosen_stations = [s.split("-")[1] for s in chosen_stations]
fig_out = "denmark_stations_map.png"
create_station_map(stations_coords, chosen_stations, fig_out)
