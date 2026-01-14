"""
Calculate WBGT from combined NEX-GDDP-CMIP6 files for Malawi.

Uses the Liljegren method for Wet Bulb Globe Temperature calculation,
which requires iterative solving for globe temperature and natural wet bulb temperature.

Based on: J. Liljegren, Argonne National Laboratory method
Reference implementation from thermofeel/ECMWF

WBGT = 0.7 * Twb + 0.2 * Tg + 0.1 * Ta

Where:
- Twb = natural wet bulb temperature
- Tg = globe temperature
- Ta = air temperature
"""

import math
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime

# ============================================================================
# Physical constants
# ============================================================================

M_AIR = 28.97  # molecular weight of dry air (g/mol)
M_H2O = 18.015  # molecular weight of water vapor (g/mol)
R_GAS = 8314.34  # ideal gas constant (J/kg mol·K)
Cp = 1003.5  # specific heat capacity of air at constant pressure (J·kg-1·K-1)
STEFANB = 5.6696e-8  # Stefan-Boltzmann constant (W·m-2·K-4)
R_AIR = R_GAS / M_AIR
RATIO = Cp * M_AIR / M_H2O
Pr = Cp / (Cp + 1.25 * R_AIR)  # Prandtl number

# Globe constants
D_GLOBE = 0.0508  # diameter of globe (m)
EMIS_GLOBE = 0.95  # emissivity of globe
ALB_GLOBE = 0.05  # albedo of globe

# Wick constants
EMIS_WICK = 0.95  # emissivity of the wick
ALB_WICK = 0.4  # albedo of the wick
D_WICK = 0.007  # diameter of the wick (m)
L_WICK = 0.0254  # length of the wick (m)

# Surface constants
EMIS_SFC = 0.999  # surface emissivity
ALB_SFC = 0.45  # surface albedo

# Iteration parameters
MAX_ITER = 50
CONVERGENCE = 0.02
MIN_SPEED = 0.13  # minimum wind speed (m/s)

MISSING_VALUE = np.nan

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(
    "/Users/rem76/Desktop/Climate_change_health/nex_gddp_cmip6_malawi_combined"
)

OUT_DIR = Path(
    "/Users/rem76/Desktop/Climate_change_health/nex_gddp_cmip6_malawi_wbgt"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["ssp245"]
YEAR_START = 2025
YEAR_END = 2040

# Variables needed for WBGT calculation
REQUIRED_VARS = ["tas", "hurs", "rsds", "sfcWind"]
OPTIONAL_VARS = ["huss", "rlds", "ps"]


# ============================================================================
# Thermodynamic helper functions
# ============================================================================

def calculate_saturation_vapour_pressure(t_k):
    """
    Calculate saturation vapor pressure using Buck equation.

    Parameters
    ----------
    t_k : array-like
        Temperature in Kelvin

    Returns
    -------
    e_sat : array-like
        Saturation vapor pressure in hPa
    """
    t_c = t_k - 273.15
    # Buck equation for liquid water
    e_sat = 6.1121 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))
    return e_sat


def calculate_dewpoint(t_k, rh_percent):
    """
    Calculate dewpoint temperature from temperature and relative humidity.

    Parameters
    ----------
    t_k : array-like
        Temperature in Kelvin
    rh_percent : array-like
        Relative humidity in percent (0-100)

    Returns
    -------
    td_k : array-like
        Dewpoint temperature in Kelvin
    """
    t_c = t_k - 273.15
    rh = np.clip(rh_percent, 1, 100) / 100.0

    # Magnus formula
    a, b = 17.27, 237.7
    alpha = (a * t_c) / (b + t_c) + np.log(rh)
    td_c = (b * alpha) / (a - alpha)

    return td_c + 273.15


def viscosity(t_k):
    """
    Calculate dynamic viscosity of air.

    Parameters
    ----------
    t_k : array-like
        Temperature in Kelvin

    Returns
    -------
    visc : array-like
        Dynamic viscosity in kg/(m·s)
    """
    omega = 1.2945 - t_k / 1141.176470588
    visc = 0.0000026693 * np.sqrt(28.97 * t_k) / (13.082689 * omega)
    return visc


def thermal_conductivity(t_k):
    """
    Calculate thermal conductivity of air.

    Parameters
    ----------
    t_k : array-like
        Temperature in Kelvin

    Returns
    -------
    tc : array-like
        Thermal conductivity in W/(m·K)
    """
    return (Cp + 1.25 * R_AIR) * viscosity(t_k)


def diffusivity(t_k, p_pa):
    """
    Calculate diffusivity of water vapor in air.

    Parameters
    ----------
    t_k : array-like
        Temperature in Kelvin
    p_pa : array-like
        Pressure in Pa

    Returns
    -------
    diff : array-like
        Diffusivity in m²/s
    """
    diff = (2.471773765165648e-05 *
            (t_k * 0.0034210563748421257) ** 2.334 *
            (p_pa / 101325) ** (-1))
    return diff


def evap(t_k):
    """
    Calculate latent heat of evaporation.

    Parameters
    ----------
    t_k : array-like
        Temperature in Kelvin

    Returns
    -------
    hevap : array-like
        Latent heat in J/kg
    """
    return 1665134.5 + 2370.0 * t_k


def emis_atm(t_k, rh_percent):
    """
    Calculate atmospheric emissivity.

    Reference: Oke (2nd edition), page 373.

    Parameters
    ----------
    t_k : array-like
        Temperature in Kelvin
    rh_percent : array-like
        Relative humidity in percent

    Returns
    -------
    emis : array-like
        Atmospheric emissivity (dimensionless)
    """
    e_sat = calculate_saturation_vapour_pressure(t_k)
    rh = rh_percent / 100.0
    e = rh * e_sat  # vapor pressure in hPa
    return 0.575 * (e ** 0.143)


def h_sphere_in_air(diameter, t_air, p_hpa, speed):
    """
    Calculate convective heat transfer coefficient for flow around a sphere.

    Reference: Bird, Stewart, and Lightfoot, page 409.

    Parameters
    ----------
    diameter : float
        Sphere diameter in m
    t_air : array-like
        Air temperature in K
    p_hpa : array-like
        Pressure in hPa
    speed : array-like
        Wind speed in m/s

    Returns
    -------
    h : array-like
        Heat transfer coefficient in W/(m²·K)
    """
    density = p_hpa * 100.0 / (R_AIR * t_air)
    speed_eff = np.maximum(speed, MIN_SPEED)
    Re = speed_eff * density * diameter / viscosity(t_air)
    Nu = 2.0 + 0.6 * np.sqrt(Re) * (Pr ** 0.3333)
    return Nu * thermal_conductivity(t_air) / diameter


def h_cylinder_in_air(diameter, length, t_air, p_hpa, speed):
    """
    Calculate convective heat transfer coefficient for flow around a cylinder.

    Parameters from Bedingfield and Drew.

    Parameters
    ----------
    diameter : float
        Cylinder diameter in m
    length : float
        Cylinder length in m (not used in calculation)
    t_air : array-like
        Air temperature in K
    p_hpa : array-like
        Pressure in hPa
    speed : array-like
        Wind speed in m/s

    Returns
    -------
    h : array-like
        Heat transfer coefficient in W/(m²·K)
    """
    a = 0.56
    b = 0.281
    c = 0.4

    density = p_hpa * 100.0 / (R_AIR * t_air)
    speed_eff = np.maximum(speed, MIN_SPEED)
    Re = speed_eff * density * diameter / viscosity(t_air)
    Nu = b * (Re ** (1.0 - c)) * (Pr ** (1.0 - a))
    return Nu * thermal_conductivity(t_air) / diameter


# ============================================================================
# Core WBGT calculations (Liljegren method)
# ============================================================================

def calculate_globe_temperature(t_air, rh, p_hpa, solar, fdir, cossza, speed):
    """
    Calculate globe temperature using Liljegren method.

    Parameters
    ----------
    t_air : array-like
        Air temperature in K
    rh : array-like
        Relative humidity in percent
    p_hpa : array-like
        Pressure in hPa
    solar : array-like
        Solar radiation in W/m²
    fdir : array-like
        Direct fraction of solar radiation (0-1)
    cossza : array-like
        Cosine of solar zenith angle
    speed : array-like
        Wind speed in m/s

    Returns
    -------
    t_globe : array-like
        Globe temperature in K
    """
    t_sfc = t_air  # assume surface temperature equals air temperature
    t_globe_prev = np.copy(t_air)  # first guess

    # Avoid division by zero for cossza
    cossza_safe = np.where(cossza > 0.01, cossza, 0.01)

    converged = np.zeros_like(t_air, dtype=bool)

    for iteration in range(MAX_ITER):
        # Evaluate properties at average temperature
        t_ref = 0.5 * (t_globe_prev + t_air)
        h = h_sphere_in_air(D_GLOBE, t_ref, p_hpa, speed)

        # Globe temperature equation
        t_globe_new = np.power(
            0.5 * (emis_atm(t_air, rh) * np.power(t_air, 4.0) +
                   EMIS_SFC * np.power(t_sfc, 4.0))
            - h / (STEFANB * EMIS_GLOBE) * (t_globe_prev - t_air)
            + solar / (2.0 * STEFANB * EMIS_GLOBE) * (1.0 - ALB_GLOBE) *
            (fdir * (1.0 / (2.0 * cossza_safe) - 1.0) + 1.0 + ALB_SFC),
            0.25
        )

        # Check convergence
        diff = np.abs(t_globe_new - t_globe_prev)
        newly_converged = diff < CONVERGENCE
        converged = converged | newly_converged

        # Update with relaxation
        t_globe_prev = np.where(
            converged,
            t_globe_prev,
            0.9 * t_globe_prev + 0.1 * t_globe_new
        )

        if np.all(converged):
            break

    # Set non-converged values to missing
    t_globe = np.where(converged, t_globe_new, MISSING_VALUE)

    return t_globe


def calculate_natural_wet_bulb(t_air, td, rh, p_hpa, speed, solar, fdir, cossza):
    """
    Calculate natural wet bulb temperature using Liljegren method.

    Parameters
    ----------
    t_air : array-like
        Air temperature in K
    td : array-like
        Dewpoint temperature in K
    rh : array-like
        Relative humidity in percent
    p_hpa : array-like
        Pressure in hPa
    speed : array-like
        Wind speed in m/s
    solar : array-like
        Solar radiation in W/m²
    fdir : array-like
        Direct fraction of solar radiation (0-1)
    cossza : array-like
        Cosine of solar zenith angle

    Returns
    -------
    t_wb : array-like
        Natural wet bulb temperature in K
    """
    a = 0.56  # from Bedingfield and Drew

    t_sfc = t_air

    # Calculate solar zenith angle (avoid division by zero)
    cossza_safe = np.clip(cossza, 0.01, 1.0)
    sza = np.arccos(cossza_safe)

    # Vapor pressure of air
    e_air = (rh / 100.0) * calculate_saturation_vapour_pressure(t_air)

    # First guess is dewpoint
    t_wb_prev = np.copy(td)

    converged = np.zeros_like(t_air, dtype=bool)

    for iteration in range(MAX_ITER):
        t_ref = 0.5 * (t_wb_prev + t_air)
        h = h_cylinder_in_air(D_WICK, L_WICK, t_ref, p_hpa, speed)

        # Atmospheric radiation term
        F_atm = (STEFANB * EMIS_WICK *
                 (0.5 * (emis_atm(t_air, rh) * np.power(t_air, 4.0) +
                         EMIS_SFC * np.power(t_sfc, 4.0))
                  - np.power(t_wb_prev, 4.0))
                 + (1.0 - ALB_WICK) * solar *
                 ((1.0 - fdir) * (1.0 + 0.25 * D_WICK / L_WICK)
                  + fdir * (np.tan(sza) / np.pi + 0.25 * D_WICK / L_WICK)
                  + ALB_SFC))

        # Saturation vapor pressure at wet bulb
        e_wick = calculate_saturation_vapour_pressure(t_wb_prev)

        # Density and Schmidt number
        density = p_hpa * 100.0 / (R_AIR * t_ref)
        Sc = viscosity(t_ref) / (density * diffusivity(t_ref, p_hpa * 100.0))

        # Wet bulb equation
        t_wb_new = (t_air
                    - evap(t_ref) / RATIO * (e_wick - e_air) / (p_hpa - e_wick) *
                    np.power(Pr / Sc, a)
                    + F_atm / h)

        # Check convergence
        diff = np.abs(t_wb_new - t_wb_prev)
        newly_converged = diff < CONVERGENCE
        converged = converged | newly_converged

        # Update with relaxation
        t_wb_prev = np.where(
            converged,
            t_wb_prev,
            0.9 * t_wb_prev + 0.1 * t_wb_new
        )

        if np.all(converged):
            break

    # Set non-converged values to missing
    t_wb = np.where(converged, t_wb_new, MISSING_VALUE)

    return t_wb


def calculate_wbgt(t_air, rh, p_hpa, solar, fdir, cossza, speed):
    """
    Calculate Wet Bulb Globe Temperature using Liljegren method.

    WBGT = 0.7 * Twb + 0.2 * Tg + 0.1 * Ta

    Parameters
    ----------
    t_air : array-like
        Air temperature in K
    rh : array-like
        Relative humidity in percent (0-100)
    p_hpa : array-like
        Pressure in hPa
    solar : array-like
        Solar radiation in W/m²
    fdir : array-like
        Direct fraction of solar radiation (0-1)
    cossza : array-like
        Cosine of solar zenith angle
    speed : array-like
        Wind speed in m/s

    Returns
    -------
    wbgt : array-like
        Wet Bulb Globe Temperature in °C
    """
    # Calculate dewpoint
    td = calculate_dewpoint(t_air, rh)

    # Calculate globe temperature
    t_globe = calculate_globe_temperature(
        t_air, rh, p_hpa, solar, fdir, cossza, speed
    )

    # Calculate natural wet bulb temperature
    t_wb = calculate_natural_wet_bulb(
        t_air, td, rh, p_hpa, speed, solar, fdir, cossza
    )

    # WBGT formula (convert to Celsius)
    wbgt = (0.7 * (t_wb - 273.15) +
            0.2 * (t_globe - 273.15) +
            0.1 * (t_air - 273.15))

    return wbgt, t_globe - 273.15, t_wb - 273.15


# ============================================================================
# Solar geometry
# ============================================================================

def calculate_cos_solar_zenith_angle(lat, lon, year, month, day, hour=12):
    """
    Calculate cosine of solar zenith angle.

    Parameters
    ----------
    lat : array-like
        Latitude in degrees
    lon : array-like
        Longitude in degrees
    year, month, day : int
        Date
    hour : float
        Hour of day (UTC)

    Returns
    -------
    cossza : array-like
        Cosine of solar zenith angle
    """
    # Day of year
    dt = datetime(year, month, day)
    doy = dt.timetuple().tm_yday

    # Solar declination (radians)
    decl = 23.45 * np.sin(np.radians(360 / 365 * (doy - 81)))
    decl_rad = np.radians(decl)

    # Hour angle (radians)
    # Solar noon at longitude 0 is at 12:00 UTC
    solar_time = hour + lon / 15.0
    hour_angle = np.radians(15 * (solar_time - 12))

    # Latitude in radians
    lat_rad = np.radians(lat)

    # Cosine of solar zenith angle
    cossza = (np.sin(lat_rad) * np.sin(decl_rad) +
              np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle))

    # Clip to valid range
    cossza = np.clip(cossza, 0, 1)

    return cossza


def estimate_daily_mean_cossza(lat, doy):
    """
    Estimate daily mean cosine of solar zenith angle.

    For daily data, we need an effective average cossza.
    This uses an approximation based on latitude and day of year.

    Parameters
    ----------
    lat : array-like
        Latitude in degrees
    doy : array-like
        Day of year (1-365)

    Returns
    -------
    cossza_mean : array-like
        Effective daily mean cossza
    """
    # Solar declination
    decl = 23.45 * np.sin(np.radians(360 / 365 * (doy - 81)))
    decl_rad = np.radians(decl)
    lat_rad = np.radians(lat)

    # Approximate daily mean (roughly 2/pi times the noon value for symmetric day)
    # This is a simplification
    cossza_noon = (np.sin(lat_rad) * np.sin(decl_rad) +
                   np.cos(lat_rad) * np.cos(decl_rad))

    # Scale factor for daily average (accounts for night and morning/evening)
    # Approximately 0.4-0.5 of noon value for tropical latitudes
    cossza_mean = 0.45 * np.clip(cossza_noon, 0, 1)

    return cossza_mean


# ============================================================================
# Data loading and processing
# ============================================================================

def load_variable(model_dir, scenario, variable, year_start, year_end):
    """Load a variable from the combined netCDF file."""
    model = model_dir.name
    filepath = (
        model_dir / scenario /
        f"{variable}_day_{model}_{scenario}_malawi_{year_start}_{year_end}.nc"
    )

    if not filepath.exists():
        return None

    ds = xr.open_dataset(filepath)
    return ds[variable]


def calculate_wbgt_for_model(model_dir, scenario, year_start, year_end):
    """
    Calculate WBGT for a single model and scenario.

    Returns xarray Dataset with WBGT, globe temperature, and wet bulb temperature.
    """
    model = model_dir.name
    print(f"\n  Loading variables for {model} / {scenario}...")

    # Load required variables
    variables = {}
    for var in REQUIRED_VARS:
        da = load_variable(model_dir, scenario, var, year_start, year_end)
        if da is None:
            print(f"    Missing required variable: {var}")
            return None
        variables[var] = da
        print(f"    Loaded {var}: {da.shape}")

    # Load optional variables
    for var in OPTIONAL_VARS:
        da = load_variable(model_dir, scenario, var, year_start, year_end)
        if da is not None:
            variables[var] = da
            print(f"    Loaded {var}: {da.shape}")

    # Extract arrays
    tas = variables["tas"]
    hurs = variables["hurs"]
    rsds = variables["rsds"]
    sfcwind = variables["sfcWind"]

    # Pressure: use ps if available, otherwise assume 850 hPa (typical for Malawi elevation)
    if "ps" in variables:
        ps_hpa = variables["ps"] / 100.0  # Convert Pa to hPa
    else:
        # Malawi average elevation ~1000m, typical pressure ~900 hPa
        ps_hpa = xr.ones_like(tas) * 900.0
        print("    Using default pressure: 900 hPa")

    # Get coordinates
    lats = tas.lat.values
    lons = tas.lon.values
    times = tas.time.values

    # Create lat/lon meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    print("  Calculating WBGT (Liljegren method)...")
    print(f"    Processing {len(times)} timesteps...")

    # Initialize output arrays
    shape = tas.shape
    wbgt_out = np.full(shape, np.nan)
    tg_out = np.full(shape, np.nan)
    twb_out = np.full(shape, np.nan)

    # Process each timestep
    for t_idx, time_val in enumerate(times):
        if t_idx % 100 == 0:
            print(f"      Timestep {t_idx}/{len(times)}")

        # Extract data for this timestep
        t_air = tas.isel(time=t_idx).values
        rh = hurs.isel(time=t_idx).values
        solar = rsds.isel(time=t_idx).values
        speed = sfcwind.isel(time=t_idx).values

        if isinstance(ps_hpa, xr.DataArray):
            p_hpa = ps_hpa.isel(time=t_idx).values
        else:
            p_hpa = ps_hpa.values if hasattr(ps_hpa, 'values') else ps_hpa

        # Get day of year for solar geometry
        time_dt = pd.Timestamp(time_val).to_pydatetime()
        doy = time_dt.timetuple().tm_yday

        # Calculate daily mean solar zenith angle
        cossza = estimate_daily_mean_cossza(lat_grid, doy)

        # Estimate direct fraction (typically 0.6-0.8 for clear skies)
        # Lower when cloudy - use ratio of actual to potential radiation as proxy
        # For simplicity, use 0.7 as default
        fdir = np.full_like(solar, 0.7)

        # Handle edge cases
        # When sun is very low or below horizon, set minimum cossza
        cossza = np.maximum(cossza, 0.05)

        # Calculate WBGT
        wbgt, tg, twb = calculate_wbgt(
            t_air, rh, p_hpa, solar, fdir, cossza, speed
        )

        wbgt_out[t_idx] = wbgt
        tg_out[t_idx] = tg
        twb_out[t_idx] = twb

    # Create output DataArrays
    coords = {"time": tas.time, "lat": tas.lat, "lon": tas.lon}

    wbgt_da = xr.DataArray(
        wbgt_out,
        coords=coords,
        dims=["time", "lat", "lon"],
        name="wbgt",
        attrs={
            "long_name": "Wet Bulb Globe Temperature",
            "units": "degC",
            "method": "Liljegren",
            "source_model": model,
            "source_scenario": scenario
        }
    )

    tg_da = xr.DataArray(
        tg_out,
        coords=coords,
        dims=["time", "lat", "lon"],
        name="tg",
        attrs={
            "long_name": "Globe Temperature",
            "units": "degC",
            "method": "Liljegren"
        }
    )

    twb_da = xr.DataArray(
        twb_out,
        coords=coords,
        dims=["time", "lat", "lon"],
        name="twb",
        attrs={
            "long_name": "Natural Wet Bulb Temperature",
            "units": "degC",
            "method": "Liljegren"
        }
    )

    # Close input datasets
    for da in variables.values():
        da.close()

    return xr.Dataset({"wbgt": wbgt_da, "tg": tg_da, "twb": twb_da})


# ============================================================================
# Main
# ============================================================================

def main():
    import pandas as pd  # for Timestamp conversion

    print("=" * 70)
    print("WBGT Calculation from NEX-GDDP-CMIP6 Data (Liljegren Method)")
    print("=" * 70)
    print(f"\nInput directory: {DATA_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Years: {YEAR_START}-{YEAR_END}")

    if not DATA_DIR.exists():
        print(f"\nError: Input directory does not exist: {DATA_DIR}")
        return

    # Find all models
    models = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    print(f"\nFound {len(models)} models: {[m.name for m in models]}")

    # Process each model and scenario
    for model_dir in sorted(models):
        model = model_dir.name
        print(f"\n{'=' * 70}")
        print(f"Processing model: {model}")
        print("=" * 70)

        for scenario in SCENARIOS:
            scenario_dir = model_dir / scenario
            if not scenario_dir.exists():
                print(f"\n  Skipping {scenario} - directory not found")
                continue

            # Calculate WBGT
            ds = calculate_wbgt_for_model(
                model_dir, scenario, YEAR_START, YEAR_END
            )

            if ds is None:
                print(f"  Could not calculate WBGT for {model}/{scenario}")
                continue

            # Save output
            out_model_dir = OUT_DIR / model / scenario
            out_model_dir.mkdir(parents=True, exist_ok=True)

            out_file = (
                out_model_dir /
                f"wbgt_day_{model}_{scenario}_malawi_{YEAR_START}_{YEAR_END}.nc"
            )

            # Encoding for compression
            encoding = {
                var: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for var in ds.data_vars
            }

            ds.to_netcdf(out_file, format="NETCDF4", encoding=encoding)

            print(f"\n  Saved: {out_file}")

            # Print summary statistics
            wbgt = ds["wbgt"]
            print(f"\n  WBGT statistics:")
            print(f"    Min:  {float(wbgt.min()):.1f}°C")
            print(f"    Mean: {float(wbgt.mean()):.1f}°C")
            print(f"    Max:  {float(wbgt.max()):.1f}°C")
            print(f"    Days > 28°C: {int((wbgt > 28).sum())} "
                  f"({100 * (wbgt > 28).mean():.1f}%)")
            print(f"    Days > 32°C: {int((wbgt > 32).sum())} "
                  f"({100 * (wbgt > 32).mean():.1f}%)")

            ds.close()

    print("\n" + "=" * 70)
    print("Done calculating WBGT for all models and scenarios.")
    print("=" * 70)


if __name__ == "__main__":
    # Need pandas for timestamp conversion
    import pandas as pd

    main()
