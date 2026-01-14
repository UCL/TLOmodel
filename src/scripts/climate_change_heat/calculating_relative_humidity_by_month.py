import os
import numpy as np
import xarray as xr
from glob import glob
from pathlib import Path


def calculate_vapor_pressure_from_huss(huss, ps=101325):
    """
    Calculate vapor pressure from specific humidity.

    Parameters:
    -----------
    huss : array-like
        Specific humidity (kg/kg)
    ps : float or array-like
        Surface pressure in Pa (default: 101325 Pa = 1 atm)

    Returns:
    --------
    e : array-like
        Vapor pressure in hPa
    """
    # e = (huss * ps) / (0.622 + 0.378 * huss)
    # Convert from Pa to hPa
    e = (huss * ps) / (0.622 + 0.378 * huss) / 100
    return e


def calculate_relative_humidity_from_hurs(hurs):
    """
    Pass through relative humidity (already in %).

    Parameters:
    -----------
    hurs : array-like
        Relative humidity (%)

    Returns:
    --------
    rh : array-like
        Relative humidity clipped to 0-100%
    """
    return np.clip(hurs, 0, 100)


def calculate_dewpoint_from_rh(temp_k, rh):
    """
    Calculate dewpoint temperature from temperature and relative humidity.

    Parameters:
    -----------
    temp_k : array-like
        Temperature in Kelvin
    rh : array-like
        Relative humidity in percentage (0-100)

    Returns:
    --------
    dewpoint_k : array-like
        Dewpoint temperature in Kelvin
    """
    temp_c = temp_k - 273.15

    # Magnus formula constants
    a = 17.27
    b = 237.7

    # Calculate gamma
    gamma = (a * temp_c / (b + temp_c)) + np.log(rh / 100.0)

    # Calculate dewpoint in Celsius
    dewpoint_c = (b * gamma) / (a - gamma)

    return dewpoint_c + 273.15


def calculate_wbgt_simplified(tas, hurs, sfcWind=None, rsds=None):
    """
    Calculate simplified Wet Bulb Globe Temperature (WBGT) for outdoor conditions.

    This uses the Stull (2011) wet bulb approximation combined with adjustments
    for solar radiation when available.

    Parameters:
    -----------
    tas : array-like
        Air temperature in Kelvin
    hurs : array-like
        Relative humidity in percentage (0-100)
    sfcWind : array-like, optional
        Surface wind speed in m/s
    rsds : array-like, optional
        Surface downwelling shortwave radiation in W/m²

    Returns:
    --------
    wbgt : array-like
        Wet Bulb Globe Temperature in Celsius
    """
    # Convert temperature to Celsius
    T = tas - 273.15
    RH = np.clip(hurs, 0, 100)

    # Calculate wet bulb temperature using Stull (2011) formula
    # Valid for RH > 5% and -20°C < T < 50°C
    Tw = T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + \
         np.arctan(T + RH) - np.arctan(RH - 1.676331) + \
         0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH) - 4.686035

    # For outdoor WBGT without globe temperature:
    # WBGT_outdoor ≈ 0.7 * Tw + 0.3 * Ta (simplified, no radiation)
    # With radiation: WBGT = 0.7 * Tw + 0.2 * Tg + 0.1 * Ta

    if rsds is not None:
        # Estimate globe temperature effect from solar radiation
        # Tg ≈ Ta + solar_factor where solar_factor depends on radiation
        # Simplified: assume globe temp increase proportional to radiation
        # At 1000 W/m² full sun, globe temp can be ~10-15°C above air temp
        solar_factor = rsds / 1000.0 * 10.0  # Max ~10°C increase at full sun

        if sfcWind is not None:
            # Wind reduces globe temperature effect
            wind_factor = 1.0 / (1.0 + 0.5 * sfcWind)
            solar_factor = solar_factor * wind_factor

        Tg = T + solar_factor

        # Full outdoor WBGT formula
        wbgt = 0.7 * Tw + 0.2 * Tg + 0.1 * T
    else:
        # Simplified indoor/shade WBGT (no radiation)
        wbgt = 0.7 * Tw + 0.3 * T

    return wbgt


def calculate_wbgt_bernard(tas, hurs, sfcWind, rsds):
    """
    Calculate WBGT using Bernard & Pourmoghani (1999) method.
    More physically-based approach using energy balance.

    Parameters:
    -----------
    tas : array-like
        Air temperature in Kelvin
    hurs : array-like
        Relative humidity in percentage (0-100)
    sfcWind : array-like
        Surface wind speed in m/s (minimum 0.5 m/s applied)
    rsds : array-like
        Surface downwelling shortwave radiation in W/m²

    Returns:
    --------
    wbgt : array-like
        Wet Bulb Globe Temperature in Celsius
    """
    # Convert to Celsius
    T = tas - 273.15
    RH = np.clip(hurs, 0, 100)

    # Ensure minimum wind speed
    V = np.maximum(sfcWind, 0.5)

    # Calculate vapor pressure (hPa)
    es = 6.112 * np.exp(17.67 * T / (T + 243.5))
    e = es * RH / 100.0

    # Wet bulb temperature (psychrometric approximation)
    Tw = T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + \
         np.arctan(T + RH) - np.arctan(RH - 1.676331) + \
         0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH) - 4.686035

    # Globe temperature estimation
    # Using simplified Liljegren approach
    # Tg = Ta + S/(h * (1 + 1.3*V^0.5)) where S is absorbed radiation
    # Assuming black globe with emissivity ~0.95
    absorbed_radiation = 0.95 * rsds * 0.25  # Globe absorbs ~25% of incoming
    h_convective = 6.3 * np.power(V, 0.6)  # Convective heat transfer coefficient

    delta_Tg = absorbed_radiation / (h_convective * (1 + 1.3 * np.sqrt(V)))
    Tg = T + np.clip(delta_Tg, 0, 20)  # Cap globe temp increase at 20°C

    # WBGT formula (outdoor)
    wbgt = 0.7 * Tw + 0.2 * Tg + 0.1 * T

    return wbgt


def process_cmip6_wbgt(data_dir, output_dir, models=None, scenario='ssp245',
                       year_start=2025, year_end=2026, wbgt_method='simplified'):
    """
    Process NEX-GDDP-CMIP6 data to calculate daily WBGT for Malawi.

    Parameters:
    -----------
    data_dir : str
        Base directory containing collated NEX-GDDP-CMIP6 data
    output_dir : str
        Directory to save WBGT output files
    models : list, optional
        List of models to process (default: all available)
    scenario : str
        SSP scenario (default: 'ssp245')
    year_start : int
        Start year (default: 2025)
    year_end : int
        End year (default: 2026)
    wbgt_method : str
        WBGT calculation method: 'simplified' or 'bernard'
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find available models
    if models is None:
        model_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        models = [d.name for d in model_dirs]

    print(f"Processing {len(models)} models for {scenario} ({year_start}-{year_end})")
    print(f"WBGT method: {wbgt_method}")
    print("=" * 60)

    # Required variables for WBGT
    required_vars = ['tas', 'hurs']
    optional_vars = ['sfcWind', 'rsds']

    results = []

    for model in models:
        print(f"\nProcessing: {model}")

        model_dir = data_dir / model / scenario
        if not model_dir.exists():
            print(f"  Skipping: No {scenario} data found")
            continue

        # Load required variables
        data = {}
        missing_required = False

        for var in required_vars:
            pattern = f"{var}_day_{model}_{scenario}_*_malawi.nc"
            files = list(model_dir.glob(pattern))

            if not files:
                print(f"  Skipping: Missing required variable {var}")
                missing_required = True
                break

            print(f"  Loading {var}...")
            ds = xr.open_dataset(files[0])

            # Subset to requested years
            ds = ds.sel(time=slice(f"{year_start}-01-01", f"{year_end}-12-31"))

            if len(ds.time) == 0:
                print(f"  Skipping: No data for {year_start}-{year_end}")
                missing_required = True
                ds.close()
                break

            data[var] = ds[var]
            ds.close()

        if missing_required:
            continue

        # Load optional variables
        for var in optional_vars:
            pattern = f"{var}_day_{model}_{scenario}_*_malawi.nc"
            files = list(model_dir.glob(pattern))

            if files:
                print(f"  Loading {var}...")
                ds = xr.open_dataset(files[0])
                ds = ds.sel(time=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
                data[var] = ds[var]
                ds.close()
            else:
                print(f"  Optional variable {var} not found")
                data[var] = None

        # Calculate WBGT
        print("  Calculating WBGT...")

        if wbgt_method == 'bernard' and data['sfcWind'] is not None and data['rsds'] is not None:
            wbgt = calculate_wbgt_bernard(
                data['tas'].values,
                data['hurs'].values,
                data['sfcWind'].values,
                data['rsds'].values
            )
        else:
            wbgt = calculate_wbgt_simplified(
                data['tas'].values,
                data['hurs'].values,
                data['sfcWind'].values if data['sfcWind'] is not None else None,
                data['rsds'].values if data['rsds'] is not None else None
            )

        # Create output dataset
        ds_out = xr.Dataset(
            {
                'wbgt': (['time', 'lat', 'lon'], wbgt),
                'tas': data['tas'],
                'hurs': data['hurs'],
            },
            coords={
                'time': data['tas'].time,
                'lat': data['tas'].lat,
                'lon': data['tas'].lon,
            }
        )

        # Add optional variables if available
        if data['sfcWind'] is not None:
            ds_out['sfcWind'] = data['sfcWind']
        if data['rsds'] is not None:
            ds_out['rsds'] = data['rsds']

        # Add attributes
        ds_out['wbgt'].attrs = {
            'long_name': 'Wet Bulb Globe Temperature',
            'units': 'degC',
            'method': wbgt_method,
            'description': 'Daily WBGT calculated from NEX-GDDP-CMIP6 data'
        }

        ds_out.attrs = {
            'title': f'Daily WBGT for Malawi - {model} {scenario}',
            'source': 'NASA NEX-GDDP-CMIP6',
            'model': model,
            'scenario': scenario,
            'year_range': f'{year_start}-{year_end}',
            'region': 'Malawi',
            'wbgt_method': wbgt_method,
            'created_by': 'calculate_wbgt_cmip6.py'
        }

        # Save output
        output_file = output_dir / f"wbgt_daily_{model}_{scenario}_{year_start}-{year_end}_malawi.nc"
        print(f"  Saving: {output_file.name}")
        ds_out.to_netcdf(output_file)

        # Store summary statistics
        results.append({
            'model': model,
            'scenario': scenario,
            'years': f'{year_start}-{year_end}',
            'wbgt_mean': float(np.nanmean(wbgt)),
            'wbgt_max': float(np.nanmax(wbgt)),
            'wbgt_min': float(np.nanmin(wbgt)),
            'n_days': len(data['tas'].time),
        })

        ds_out.close()

    # Save summary
    if results:
        import csv
        summary_file = output_dir / f"wbgt_summary_{scenario}_{year_start}-{year_end}.csv"
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSummary saved to: {summary_file}")

    print(f"\nProcessing complete! {len(results)} models processed.")
    return results


def create_ensemble_mean(output_dir, scenario='ssp245', year_start=2025, year_end=2026):
    """
    Create ensemble mean WBGT from all model outputs.

    Parameters:
    -----------
    output_dir : str
        Directory containing individual model WBGT files
    scenario : str
        SSP scenario
    year_start : int
        Start year
    year_end : int
        End year
    """
    output_dir = Path(output_dir)

    pattern = f"wbgt_daily_*_{scenario}_{year_start}-{year_end}_malawi.nc"
    files = list(output_dir.glob(pattern))

    if not files:
        print("No WBGT files found for ensemble mean")
        return None

    print(f"Creating ensemble mean from {len(files)} models...")

    # Load all models
    datasets = []
    models = []
    for f in files:
        model = f.name.split('_')[2]  # Extract model name from filename
        ds = xr.open_dataset(f)
        ds = ds.expand_dims({'model': [model]})
        datasets.append(ds)
        models.append(model)

    # Concatenate along model dimension
    ds_all = xr.concat(datasets, dim='model')

    # Calculate ensemble statistics
    ds_ensemble = xr.Dataset(
        {
            'wbgt_mean': ds_all['wbgt'].mean(dim='model'),
            'wbgt_std': ds_all['wbgt'].std(dim='model'),
            'wbgt_min': ds_all['wbgt'].min(dim='model'),
            'wbgt_max': ds_all['wbgt'].max(dim='model'),
            'wbgt_median': ds_all['wbgt'].median(dim='model'),
        },
        coords={
            'time': ds_all.time,
            'lat': ds_all.lat,
            'lon': ds_all.lon,
        }
    )

    ds_ensemble.attrs = {
        'title': f'Ensemble WBGT for Malawi - {scenario}',
        'source': 'NASA NEX-GDDP-CMIP6',
        'scenario': scenario,
        'year_range': f'{year_start}-{year_end}',
        'n_models': len(models),
        'models': ', '.join(models),
        'region': 'Malawi',
    }

    # Save ensemble
    ensemble_file = output_dir / f"wbgt_ensemble_{scenario}_{year_start}-{year_end}_malawi.nc"
    ds_ensemble.to_netcdf(ensemble_file)
    print(f"Ensemble saved to: {ensemble_file}")

    # Close datasets
    for ds in datasets:
        ds.close()
    ds_all.close()
    ds_ensemble.close()

    return ensemble_file


# Example usage
if __name__ == "__main__":
    # Paths - adjust these to your setup
    data_base = "nex_gddp_cmip6_malawi/collated"
    output_base = "wbgt_output"

    # Process all models for SSP245, 2025-2026
    results = process_cmip6_wbgt(
        data_dir=data_base,
        output_dir=output_base,
        models=None,  # Process all available models
        scenario='ssp245',
        year_start=2025,
        year_end=2045,
        wbgt_method='bernard'  # or 'bernard' for more detailed calculation
    )

    # Create ensemble mean
    if results:
        create_ensemble_mean(
            output_dir=output_base,
            scenario='ssp245',
            year_start=2025,
            year_end=2045
        )
