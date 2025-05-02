#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from datetime import datetime, timedelta
import os
import sys

def create_parquet_from_profiles(timestamp_str, road_stretch, temp_profiles,data_path):
    """
    Convert profile data to pandas DataFrame and save as parquet

    Parameters:
    timestamp_str: str - filename in format 'fild.YYYYMMDDHH'
    road_stretch: list - list of station IDs
    temp_profiles: dict - dictionary of temperature profiles
    """
    # Extract timestamp from filename
    #date_str = timestamp_str.split('.')[1]
    timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H')
    timestamp = timestamp - timedelta(hours=2) #subtract 2h since data is from 2h ago
    new_ts_str = datetime.strftime(timestamp,"%Y%m%d%H") #update string for data to be dumped
    # Create list to hold all records
    records = []

    # Process each station's profile
    for idx, station_id in enumerate(road_stretch):
        if idx in temp_profiles:
            # Create a record with timestamp and station_id
            record = {
                'timestamp': timestamp,
                'station_id': station_id
            }

            # Add temperature values for each depth
            for depth_idx, temp_value in enumerate(temp_profiles[idx]):
                record[f'depth_{depth_idx}'] = temp_value

            records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Create directory if it doesn't exist
    
    #os.makedirs('road_temp_data', exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    # Save to parquet file
    #output_file = f'road_temp_data/road_temp_{new_ts_str}.parquet'
    output_file = os.path.join(data_path,'road_temp_{new_ts_str}.parquet')
    df.to_parquet(output_file, index=False)

    return output_file

def process_file(filename):
    stations = []
    road_stretch = [] 
    station_names = {}  # New dictionary for station details
    line_count = 0
    start_found = False
    single_column_line = None
    temp_profiles = {}
    current_profile = []
    profile_count = 0

    with open(filename, 'r') as file:
        # Skip the header line
        next(file)

        lines = file.readlines()

        for i, line in enumerate(lines, 1):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            cols = line.split()

            # First part: collect stations
            if len(cols) == 3:
                road_stretch.append("-".join([cols[0],cols[1],cols[2]]))
                if cols[0] == '0':  # Lines with "0 NUMBER 0"
                    if cols[1] == '100000':
                        start_found = True
                    if start_found:
                        station_number = cols[1]
                        stations.append(int(station_number))

                        # Process station name
                        base_station = station_number[:-2]  # First 4 digits
                        sensor_number = station_number[-2:]  # Last 2 digits

                        # Add to station_names dictionary
                        if base_station not in station_names:
                            station_names[base_station] = []
                        if sensor_number not in station_names[base_station]:
                            station_names[base_station].append(sensor_number)

                        line_count += 1

            # Rest of the code for temperature profiles...
            # Check for single column transition
            if i < len(lines) - 1:
                next_line = lines[i].strip()
                if len(line.split()) == 1 and len(next_line.split()) == 1:
                    try:
                        float(line)
                        float(next_line)
                        single_column_line = i + 1

                        # Start collecting temperature profiles
                        #for jj in range(single_column_line - 1, len(lines)):
                        for jj in range(single_column_line-2, len(lines)):
                            line_val = lines[jj].strip()
                            if not line_val:  # Skip empty lines
                                continue

                            try:
                                current_profile.append(float(line_val) - 273.15)
                                if len(current_profile) == 15:
                                    temp_profiles[profile_count] = current_profile.copy()
                                    profile_count += 1
                                    current_profile = []
                            except ValueError:
                                continue
                        if current_profile: #this one takes care of the last profile
                            temp_profiles[profile_count] = current_profile
                        break
                    except ValueError:
                        continue

    return road_stretch, stations, station_names, line_count, single_column_line, temp_profiles

def plot_temperature_profiles(road_stretch, temp_profiles, date_str, num_profiles=10):
    # Create depth layers (15 layers from 14 to 0)
    #depths = np.arange(14, -1, -1)  # Changed to go from 14 to 0
    depths = np.arange(15, 0, -1)  # Changed to go from 14 to 0

    plt.figure(figsize=(10, 8))

    # Plot each profile
    for i in range(min(num_profiles, len(temp_profiles))):
        profile = temp_profiles[i]
        station = road_stretch[i]
        plt.plot(profile,depths, label=f'Profile {station}', marker='o')
    # Customize the plot
    plt.ylabel('Layer Depth (cm)')
    plt.xlabel('Temperature (C)')
    hour = int(date_str[8:10]) - 2
    plt.title(f'Temperature Profiles vs Depth on {date_str[0:8]} at {hour} UTC')
    plt.legend()
    plt.grid(True)

    # Add minor gridlines
    plt.grid(True, which='minor', linestyle=':', alpha=0.5)
    plt.minorticks_on()

    plt.show()

def plot_temperature_selected(temp_profiles, date_str, stations):
    """
    plot temperature profiles individually
    """
    # Create depth layers (15 layers from 14 to 0)
    #depths = np.arange(14, -1, -1)  # Changed to go from 14 to 0
    depths = np.arange(15, 0, -1)  # Changed to go from 14 to 0
    depths = np.arange(-15, 0, 1)  # Changed to go from 14 to 0
    #depths = np.arange(0,15)  # Changed to go from 14 to 0

    plt.figure(figsize=(10, 8))

    # Plot each profile
    for i in stations:
        profile = temp_profiles[temp_profiles.station_id == i]
        depths = [-i for i in range(14,-1,-1)]
        temps = [profile[f"depth_{i}"].values[0] for i in range(14,-1,-1)]
        plt.plot(temps,depths, label=f'Profile {i}', marker='o')
    # Customize the plot
    plt.ylabel('Layer Depth (cm)')
    plt.xlabel('Temperature (C)')
    hour = int(date_str[8:10]) - 2
    plt.title(f'Temperature Profiles vs Depth on {date_str[0:8]} at {hour} UTC')
    plt.legend()
    plt.grid(True)

    # Add minor gridlines
    plt.grid(True, which='minor', linestyle=':', alpha=0.5)
    plt.minorticks_on()
    plt.ylim(-15,0)
    plt.show()

def plot_temperature_all(temp_profiles, date_str, station):
    """
    plot all temperatures for given station and day
    """
    plt.figure(figsize=(10, 8))

    # Plot  profile for all times
    profiles = temp_profiles[temp_profiles.station_id == station]
    depths = [-i for i in range(14,-1,-1)]
    for ts in profiles.timestamp:
        profile = profiles[profiles.timestamp == ts]
        temps = [profile[f"depth_{i}"].values[0] for i in range(14,-1,-1)]
        plt.plot(temps,depths, label=f'Profile at {ts}', marker='o')
    # Customize the plot
    plt.ylabel('Layer Depth (cm)')
    plt.xlabel('Temperature (C)')
    #hour = int(date_str[8:10]) - 2
    #plt.title(f'Temperature Profiles vs Depth on {date_str[0:8]} at {hour} UTC')
    plt.title(f'Temperature Profiles vs Depth on {date_str[0:8]}')
    plt.legend()
    plt.grid(True)

    # Add minor gridlines
    plt.grid(True, which='minor', linestyle=':', alpha=0.5)
    plt.minorticks_on()
    plt.ylim(-15,0)
    plt.show()
def main():
    #This path is in DMI. Will work only if connected via vpn
    data_path = "/data/projects/glatmodel/obs/fild8/road_profiles"
    data_path = "/data/projects/glatmodel/obs/fild8/road_temp_daily"

    filename = sys.argv[1]
    date_str = filename.split("_")[-1]
    filename = os.path.join(data_path,filename)
    #find a specific one 
    # 0 136000   0
    # 0 136001   0
    temp_profiles = pd.read_parquet(filename)
    stations = ["0-136000-0","0-136001-0","0-100001-0"]
    #plot_temperature_selected(temp_profiles, date_str, stations)
    station = "0-100001-0"
    plot_temperature_all(temp_profiles, date_str, station)
    
    
if __name__=="__main__":
    main()
