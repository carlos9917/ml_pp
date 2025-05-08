import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import sys
import os


def plot_temperature_all(temp_profiles, date_str, station):
    """
    Plot all temperatures for given station and day, with color reflecting hour of day.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    profiles = temp_profiles[temp_profiles.station_id == station]
    depths = [-i for i in range(14, -1, -1)]

    #cmap = cm.get_cmap('viridis', 24)
    cmap = plt.get_cmap('viridis', 24)
    norm = mcolors.Normalize(vmin=0, vmax=23)

    for ts in profiles.timestamp:
        hour = pd.to_datetime(ts).hour
        color = cmap(norm(hour))
        profile = profiles[profiles.timestamp == ts]
        temps = [profile[f"depth_{i}"].values[0] for i in range(14, -1, -1)]
        ax.plot(temps, depths, marker='o', color=color, linewidth=2)

    ax.set_ylabel('Layer Depth (cm)')
    ax.set_xlabel('Temperature (C)')
    ax.set_title(f'Temperature Profiles vs Depth on {date_str[0:8]} for {station}')
    ax.grid(True)
    ax.grid(True, which='minor', linestyle=':', alpha=0.5)
    ax.minorticks_on()
    ax.set_ylim(-15, 0)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=range(0, 24, 2))
    cbar.set_label('Hour of Day')
    cbar.set_ticks(range(0, 24, 2))
    cbar.set_ticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    station_str = station.split("-")[1]
    save_path = f"road_profiles_{date_str}_{station_str}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    #This path is in DMI. Will work only if connected via vpn
    data_path = "/data/projects/glatmodel/obs/fild8/road_temp_daily"
    #road_temp_20220301.parquet
    #date_chosen = sys.argv[1]
    #date_str = filename.split("_")[-1]
    date_str = str(sys.argv[1])
    filename = f"road_temp_{date_str}.parquet"
    filename = os.path.join(data_path,filename)
    temp_profiles = pd.read_parquet(filename)
    station = "0-100001-0"
    plot_temperature_all(temp_profiles, date_str, station)


if __name__=="__main__":
    main()

