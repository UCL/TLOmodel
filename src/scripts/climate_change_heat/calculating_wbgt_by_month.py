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

    Returns
    -------
    rh : array-like
        Relative humidity in percentage (0-100)
    """
    # Convert to Celsius
    temp_c = temp_k - 273.15
    dewpoint_c = dewpoint_k - 273.15

    # Saturation vapor pressure (Magnus formula)
    es_temp = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    es_dewpoint = 6.112 * np.exp(17.67 * dewpoint_c / (dewpoint_c + 243.5))

    # Relative humidity
    rh = (es_dewpoint / es_temp) * 100
    rh = np.clip(rh, 0, 100)

    return rh


def calculate_wet_bulb_temperature(temp_c, rh):
    """
    Approximate wet bulb temperature (Tw) from temperature and RH.
    Formula from Stull (2011).
    """
    Tw = (temp_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) +
          np.arctan(temp_c + rh) -
          np.arctan(rh - 1.676331) +
          0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) -
          4.686035)
    return Tw


def calculate_wbgt(temp_k, dewpoint_k):
    """
    Approximate WBGT (indoor/shade) from ERA5 T and dewpoint.

    Parameters
    ----------
    temp_k : array-like
        Temperature in Kelvin
    dewpoint_k : array-like
        Dewpoint temperature in Kelvin

    Returns
    -------
    wbgt : array-like
        Wet Bulb Globe Temperature in Celsius
    """
    # Step 1: calculate RH
    rh = calculate_relative_humidity(temp_k, dewpoint_k)

    # Step 2: convert T to Celsius
    temp_c = temp_k - 273.15

    # Step 3: calculate wet bulb temp
    tw = calculate_wet_bulb_temperature(temp_c, rh)

    # Step 4: WBGT indoors (approximation)
    wbgt = 0.7 * tw + 0.3 * temp_c
    return wbgt


def process_era5_data(dewpoint_dir, temp_dir, output_dir, years):
    """
    Process ERA5 data to calculate WBGT and monthly averages.
    Processes all years and creates a single output file.
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

    # Open combined datasets
    ds_dewpoint = xr.open_mfdataset(all_dewpoint_files, combine='by_coords')
    ds_temp = xr.open_mfdataset(all_temp_files, combine='by_coords')

    # Get variable names
    dewpoint_var = [v for v in ds_dewpoint.data_vars if 'd2m' in v or '2d' in v.lower()][0]
    temp_var = [v for v in ds_temp.data_vars if 't2m' in v or '2t' in v.lower()][0]

    # Align the datasets to ensure matching time coordinates
    ds_dewpoint_aligned, ds_temp_aligned = xr.align(
        ds_dewpoint, ds_temp, join='inner'
    )

    print(f"Original shapes - Dewpoint: {ds_dewpoint[dewpoint_var].shape}, Temp: {ds_temp[temp_var].shape}")
    print(
        f"Aligned shapes - Dewpoint: {ds_dewpoint_aligned[dewpoint_var].shape}, Temp: {ds_temp_aligned[temp_var].shape}")

    # Calculate WBGT using aligned data
    wbgt = calculate_wbgt(
        ds_temp_aligned[temp_var].values,
        ds_dewpoint_aligned[dewpoint_var].values
    )

    # Create new dataset with WBGT only
    ds_out = ds_temp_aligned.copy()
    ds_out['wbgt'] = (ds_temp_aligned[temp_var].dims, wbgt)
    ds_out['wbgt'].attrs = {
        'long_name': 'Wet Bulb Globe Temperature (approx, shade)',
        'units': 'C',
        'description': 'Calculated from 2m temperature and dewpoint (Stull, 2011 approximation)'
    }

    # Calculate monthly averages
    ds_monthly = ds_out.resample(valid_time='1M').mean()

    # save
    year_range = f"{min(years)}_{max(years)}" if len(years) > 1 else str(years[0])

    hourly_output = os.path.join(output_dir, f"wbgt_hourly_{year_range}.nc")
    ds_out.to_netcdf(hourly_output)

    monthly_output = os.path.join(output_dir, f"wbgt_monthly_{year_range}.nc")
    ds_monthly.to_netcdf(monthly_output)


if __name__ == "__main__":
    dewpoint_base = "/Users/rem76/Desktop/Climate_change_health/Data/Temperature_data/2m_dewpoint/Historical/hourly/"
    temp_base = "/Users/rem76/Desktop/Climate_change_health/Data/Temperature_data/2m/Historical/hourly/"
    output_base = "/Users/rem76/Desktop/Climate_change_health/Data/Temperature_data/WBGT/Historical/"

    # Years to process
    years = range(2011, 2025)

    # Process data
    process_era5_data(dewpoint_base, temp_base, output_base, years)
