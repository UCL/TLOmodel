import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.animation as animation
import math
import imageio.v2 as imageio

# Load the dataset and the variable
file_path = "/Users/rem76/Downloads/821bebfbcee0609d233c09e8b2bbc1f3/pr_Amon_UKESM1-0-LL_ssp119_r1i1p1f2_gn_20150116-20991216.nc"
dataset = Dataset(file_path, mode='r')
pr_data = dataset.variables['pr'][:] # in kg m-2 s-1 = mm s-1 x 86400 to get to day
time_data = dataset.variables['time'][:]
lat_data = dataset.variables['lat'][:]
long_data = dataset.variables['lon'][:]
years = range(2015, 2101)
## Initial plot
pr_data_time_series_grid_1 = pr_data[:,2,1]
pr_data_time_series_grid_1 *= 86400 # to get to days

# Plot the 2D data
plt.plot(pr_data_time_series_grid_1)

plt.title('Daily precipitation')
plt.ylabel('Precip (mm)')
plt.xlabel('Time')
#plt.show()

################ Now do it by specific regions #################
# get regions from facilities file
facilities_with_grid = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_districts.csv")
facilities_with_grid = gpd.read_file("/Users/rem76/Desktop/Climate_change_health/Data/facilities_with_districts.shp")
print(facilities_with_grid)
bounds_data = []
for geometry in facilities_with_grid["geometry"]:
    minx, miny, maxx, maxy = geometry.bounds  # Get bounding box
    bounds_data.append({'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy})
bounds_df = pd.DataFrame(bounds_data)
facilities_with_bounds = pd.concat([facilities_with_grid.reset_index(drop=True), bounds_df], axis=1)

# could get centroids of the grid and see if they overlap with
centroid_data = []
for geometry in facilities_with_grid["geometry"]:
    central_x = geometry.centroid.x
    central_y = geometry.centroid.y
    centroid_data.append({'central_x': central_x, 'central_y': central_y})


## can find min square distance and index to get relavent indices
diff_northern_lat = []
diff_southern_lat = []
diff_eastern_lon = []
diff_western_lon = []

max_latitudes = facilities_with_bounds['maxy'].values
min_latitudes = facilities_with_bounds['miny'].values
max_longitudes = facilities_with_bounds['maxx'].values
min_longitudes = facilities_with_bounds['minx'].values
# Iterate over each facility to calculate differences
for i in range(len(facilities_with_bounds)):
    # Calculate the square differences for latitude and longitude, to then find nearest index
    northern_diff = (lat_data - max_latitudes[i]) ** 2
    southern_diff = (lat_data - min_latitudes[i]) ** 2
    eastern_diff = (long_data - max_longitudes[i]) ** 2
    western_diff = (long_data - min_longitudes[i]) ** 2

    diff_northern_lat.append(northern_diff.argmin())
    diff_southern_lat.append(southern_diff.argmin())
    diff_eastern_lon.append(eastern_diff.argmin())
    diff_western_lon.append(western_diff.argmin())

print(diff_northern_lat)
print(diff_southern_lat)
print(diff_eastern_lon)


### Plot mean precipitation by grid by time point


data_by_gridpoint = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/ssp2_4_5/mean_projected_precip_by_timepoint_modal_resolution.csv", index_col=0)
for grid in range(len(data_by_gridpoint)):
    for long in range(len(long_data)):
        plt.plot(data_by_gridpoint.columns, data_by_gridpoint.iloc[grid], label=f"Grid {grid + 1}")

plt.xlabel("Timepoint")  # Adjust this label based on your data
plt.ylabel("Precipitation (mm)")
print(len(data_by_gridpoint.columns))
plt.xticks(ticks=np.arange(0, len(data_by_gridpoint.columns), step=365),
           labels=years)
#plt.show()


### Plot median and IQ range precipitation by grid by time point


### NOW do median and IQR
median_df = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/ssp2_4_5/median_projected_precip_by_timepoint_modal_resolution.csv", index_col=0)
percentile_25_df = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/ssp2_4_5/percentile_25_projected_precip_by_timepoint_modal_resolution.csv", index_col=0)
percentile_75_df = pd.read_csv("/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/ssp2_4_5/percentile_75_projected_precip_by_timepoint_modal_resolution.csv", index_col=0)

num_grids = len(median_df.index)
num_cols = math.ceil(math.sqrt(num_grids))
num_rows = math.ceil(num_grids / num_cols)

fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
fig.suptitle("Median Precipitation with IQR Shaded Area by Grid")
axs = axs.flatten()

for idx, grid in enumerate(median_df.index):
    median_values = median_df.loc[grid].values
    timepoints = range(len(median_values))

    axs[idx].fill_between(timepoints, percentile_25_df.loc[grid], percentile_75_df.loc[grid], alpha=0.5, color = "lightblue")
    axs[idx].plot(timepoints, median_values, color="blue")

    axs[idx].set_ylim(0,10)
    axs[idx].set_xlim(0,365*10)

    axs[idx].set_title(f"Grid {grid}")
    axs[idx].set_xlabel("Year")
    axs[idx].set_xticks(ticks=np.arange(0, len(data_by_gridpoint.columns), step=365*5),
               labels=[year for i, year in enumerate(years) if i % 5 == 0])
    axs[idx].set_ylabel("Precipitation (mm)")

for ax in axs[num_grids:]:
    ax.set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

## film for each grid point?

# Define grid layout
num_grids = len(median_df.index)
num_cols = math.ceil(math.sqrt(num_grids))
num_rows = math.ceil(num_grids / num_cols)

# Set up figure
fig, ax = plt.subplots(figsize=(10, 8))
cbar = None


# Create an animation function
def animate(i):
    global cbar
    ax.clear()  # Clear the previous frame

    # Reshape median data and IQR data for heat map
    median_values = median_df.iloc[:, i].values.reshape(num_rows, num_cols)
    iqr_values = (percentile_75_df.iloc[:, i] - percentile_25_df.iloc[:, i]).values.reshape(num_rows, num_cols)

    # Plot heatmap with IQR as a color gradient
    heatmap = ax.imshow(median_values, cmap="coolwarm", interpolation="nearest", vmin=0, vmax=40)

    # Remove old colorbar and add a new one
    if cbar:
        cbar.remove()
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Precipitation (mm)")

    # Set title
    ax.set_title(f"Median Precipitation with IQR (Time Point {i + 1})")
    ax.set_xticks([])
    ax.set_yticks([])


# Create animation
ani = animation.FuncAnimation(fig, animate, frames=median_df.shape[1], repeat=False)

# Save animation as a movie (e.g., MP4 format)
ani.save("median_precipitation_movie.mp4", writer="ffmpeg", dpi=100)
plt.show()
