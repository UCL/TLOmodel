import os
import numpy as np
import xarray as xr
from glob import glob


def calculate_relative_humidity(temp_k, dewpoint_k):
    """
    Calculate relative humidity from temperature and dewpoint temperature.

    temp_k : array-like
        Temperature in Kelvin
    dewpoint_k : array-like
        Dewpoint temperature in Kelvin

    rh : array-like
        Relative humidity in percentage (0-100)
    """
    # Convert to Celsius
    temp_c = temp_k - 273.15
    dewpoint_c = dewpoint_k - 273.15

    # Calculate saturation vapor pressure using Magnus formula from https://www.omnicalculator.com/physics/relative-humidity
    # e_s = 6.112 * exp(17.67 * T / (T + 243.5))
    es_temp = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    es_dewpoint = 6.112 * np.exp(17.67 * dewpoint_c / (dewpoint_c + 243.5))

    # Relative humidity = (e / e_s) * 100
    rh = (es_dewpoint / es_temp) * 100

    # Clip to valid range [0, 100]
    rh = np.clip(rh, 0, 100)

    return rh


def process_era5_data(dewpoint_dir, temp_dir, output_dir, years):
    """
    Process ERA5 data to calculate relative humidity and monthly averages.
    Processes all years and creates a single output file.

    Parameters:
    -----------
    dewpoint_dir : str
        Base directory containing dewpoint temperature data
    temp_dir : str
        Base directory containing 2m temperature data
    output_dir : str
        Directory to save output files
    years : list
        List of years to process
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_dewpoint_files = []
    all_temp_files = []

    # Collect all files across all years
    for year in years:
        dewpoint_pattern = os.path.join(dewpoint_dir, str(year), "*.nc")
        temp_pattern = os.path.join(temp_dir, str(year), "*.nc")

        dewpoint_files = sorted(glob(dewpoint_pattern))
        temp_files = sorted(glob(temp_pattern))

        if dewpoint_files:
            all_dewpoint_files.extend(dewpoint_files)

        if temp_files:
            all_temp_files.extend(temp_files)

        return

    ds_dewpoint = xr.open_mfdataset(all_dewpoint_files, combine='by_coords')

    ds_temp = xr.open_mfdataset(all_temp_files, combine='by_coords')

    # Get variable names (may vary: d2m, t2m or 2d, 2t)
    dewpoint_var = [v for v in ds_dewpoint.data_vars if 'd2m' in v or '2d' in v.lower()][0]
    temp_var = [v for v in ds_temp.data_vars if 't2m' in v or '2t' in v.lower()][0]

    # Calculate relative humidity
    rh = calculate_relative_humidity(
        ds_temp[temp_var].values,
        ds_dewpoint[dewpoint_var].values
    )

    # Create new dataset with RH
    ds_rh = ds_temp.copy()
    ds_rh['rh'] = (ds_temp[temp_var].dims, rh)
    ds_rh['rh'].attrs = {
        'long_name': 'Relative Humidity',
        'units': '%',
        'description': 'Calculated from 2m temperature and 2m dewpoint temperature'
    }

    # Calculate monthly averages
    ds_monthly = ds_rh.resample(time='1M').mean()

    # Define year range for filename
    year_range = f"{min(years)}_{max(years)}" if len(years) > 1 else str(years[0])

    # Save hourly RH data
    hourly_output = os.path.join(output_dir, f"rh_hourly_{year_range}.nc")
    print(f"\nSaving hourly data to {hourly_output}")
    ds_rh.to_netcdf(hourly_output)

    # Save monthly average
    monthly_output = os.path.join(output_dir, f"rh_monthly_{year_range}.nc")
    print(f"Saving monthly averages to {monthly_output}")
    ds_monthly.to_netcdf(monthly_output)

    # Clean up
    ds_dewpoint.close()
    ds_temp.close()
    ds_rh.close()
    ds_monthly.close()



# Example usage
if __name__ == "__main__":
    dewpoint_base = "/Users/rem76/Desktop/Climate_change_health/Data/Temperature_data/2m_dewpoint/Historical/hourly/"
    temp_base = "/Users/rem76/Desktop/Climate_change_health/Data/Temperature_data/2m_temperature/Historical/hourly/"
    output_base = "/Users/rem76/Desktop/Climate_change_health/Data/Temperature_data/relative_humidity/Historical/"

    # Years to process
    years = range(2011, 2024)

    # Process data
    process_era5_data(dewpoint_base, temp_base, output_base, years)
