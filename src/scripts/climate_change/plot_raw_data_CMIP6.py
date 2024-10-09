import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

# Load the dataset and the variable
file_path = "/Users/rem76/Downloads/821bebfbcee0609d233c09e8b2bbc1f3/pr_Amon_UKESM1-0-LL_ssp119_r1i1p1f2_gn_20150116-20991216.nc"
dataset = Dataset(file_path, mode='r')
pr_data = dataset.variables['pr'][:]
print(pr_data.dim)
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
