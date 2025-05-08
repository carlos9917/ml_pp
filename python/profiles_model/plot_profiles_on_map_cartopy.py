import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def get_station_coordinates(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT SID, lon, lat FROM roadstations")
    stations = {str(sid): (lat, lon) for sid, lon, lat in cursor.fetchall()}
    conn.close()
    return stations

def create_station_map(stations_coords, chosen_stations, save_path):
    # Create figure with a specific projection (Mercator is good for Denmark's latitude)
    fig, ax = plt.subplots(figsize=(12, 10), 
                          subplot_kw={'projection': ccrs.Mercator()})
    
    # Get the coordinate ranges
    all_lats = [coord[0] for coord in stations_coords.values()]
    all_lons = [coord[1] for coord in stations_coords.values()]
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    # Convert chosen station IDs to match the database format
    chosen_ids = [s.split('-')[1] for s in chosen_stations]
    
    # Plot all stations and chosen stations
    for station_id, (lat, lon) in stations_coords.items():
        if station_id in chosen_ids:
            print(f'Coordinates of {station_id}: {lat},{lon}')
            ax.plot(lon, lat, 'r^', markersize=12, transform=ccrs.PlateCarree(),
                   label='Selected stations' if station_id == chosen_ids[0] else '')
            ax.annotate(station_id, xy=(lon, lat), xytext=(5, 5),
                    textcoords='offset points', transform=ccrs.PlateCarree(),
                    fontsize=8, color='black', weight='bold')
        else:
            ax.plot(lon, lat, 'bo', markersize=6, transform=ccrs.PlateCarree(),
                   label='All stations' if station_id == list(stations_coords.keys())[0] else '')



    
    # Set map extent (with some padding)
    padding = 0.5
    ax.set_extent([
        min(all_lons) - padding,
        max(all_lons) + padding,
        min(all_lats) - padding,
        max(all_lats) + padding
    ], crs=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # Add title and legend
    plt.title('Road Stations in Denmark', pad=20)
    plt.legend(loc='upper left')
    
    # Save the map
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Create test database
#conn = sqlite3.connect('stations_coords_height.db')
#cursor = conn.cursor()
#cursor.execute('DROP TABLE IF EXISTS roadstations')
#cursor.execute('''CREATE TABLE roadstations
#                 (SID INTEGER, lon FLOAT, lat FLOAT, height FLOAT)''')
#
## Add test data (Danish coordinates)
#test_data = [
#    (100001, 9.5, 56.2, 100),   # Jutland
#    (100002, 10.0, 56.5, 120),  # Near Aarhus
#    (100003, 12.8, 55.8, 90),   # Near Copenhagen
#    (100004, 11.2, 55.3, 110),  # Southern Denmark
#    (100005, 10.7, 55.6, 95),   # Odense area
#    (100006, 8.7, 56.8, 85),    # Northern Jutland
#    (100007, 12.4, 55.6, 75),   # Zealand
#    (100008, 9.8, 57.0, 115)    # More stations in Jutland
#]
#cursor.executemany('INSERT INTO roadstations VALUES (?,?,?,?)', test_data)
#conn.commit()
#conn.close()

#stations_coords = get_station_coordinates('stations_coords_height.db')
#chosen_stations = ['0-100001-0', '0-100003-0']  # Example of two chosen stations
#create_station_map(stations_coords, chosen_stations, 'denmark_stations_map_with_cartopy.png')
#print("Created detailed map at: denmark_stations_map_with_cartopy.png")


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

