import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

# Load the dataset and the variable
file_path = "/Users/rem76/Downloads/821bebfbcee0609d233c09e8b2bbc1f3/pr_Amon_UKESM1-0-LL_ssp119_r1i1p1f2_gn_20150116-20991216.nc"
dataset = Dataset(file_path, mode='r')
pr_data = dataset.variables['pr'][:]


# Average over time to get a 2D array (lat, lon)
pr_data_time_series = np.mean(pr_data, axis=(1, 2))

# Plot the 2D data
#plt.imshow(pr_data_avg, cmap='viridis', aspect='auto')
#plt.colorbar(label='Precipitation (kg m-2 s-1)')
plt.plot(pr_data_time_series)

plt.title('Average Precipitation Over Time')
plt.ylabel('Precip (mm)')
plt.xlabel('Time')
plt.show()

# Close the dataset
dataset.close()


################ Now do it by specific regions


latitudes = dataset.variables['latitude'][:]  # Adjust variable name if necessary
longitudes = dataset.variables['longitude'][:]  # Adjust variable name if necessary

# Define your coordinate bounds
lat_min, lat_max = -10, 10  # Adjust these values as needed
lon_min, lon_max = 30, 50   # Adjust these values as needed

# Create a mask for the coordinates within the specified bounds
lat_mask = (latitudes >= lat_min) & (latitudes <= lat_max)
lon_mask = (longitudes >= lon_min) & (longitudes <= lon_max)

# Use np.ix_ to create a grid of indices
lat_indices = np.where(lat_mask)[0]
lon_indices = np.where(lon_mask)[0]

# Select the data within the specified coordinate bounds
filtered_pr_data = pr_data[np.ix_(lat_indices, lon_indices)]

# Average over time to get a 2D array (lat, lon)
filtered_pr_data_time_series = np.mean(pr_data, axis=(1, 2))

# Plot the 2D data
#plt.imshow(pr_data_avg, cmap='viridis', aspect='auto')
#plt.colorbar(label='Precipitation (kg m-2 s-1)')
plt.plot(filtered_pr_data_time_series)

plt.title('Average Precipitation Over Time')
plt.ylabel('Precip (mm)')
plt.xlabel('Time')
plt.show()

# Close the dataset
dataset.close()
