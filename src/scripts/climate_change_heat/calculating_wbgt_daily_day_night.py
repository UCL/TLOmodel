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
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

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
    "/Users/rachelmurray-watson/Documents/Heat_data/NASA_GDDP-CMIP6/Combined"
)

OUT_DIR = Path(
    "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/NASA_GDDP_CMIP6_Splitz"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["ssp126", "ssp245", "ssp585"]
YEAR_START = 2025
YEAR_END = 2040

# Variables needed for WBGT calculation
# tasmax/tasmin are the day/night brackets; tas is still loaded to derive a
# single representative dewpoint (see calculate_bracket_rh below).
REQUIRED_VARS = ["tas", "tasmax", "tasmin", "hurs", "rsds", "sfcWind"]
OPTIONAL_VARS = ["huss", "rlds", "ps"]

# --- ERA5-derived forcing climatology --------------------------------------
# NEX-GDDP-CMIP6 provides no surface pressure and no direct/diffuse split, so
# the day bracket otherwise falls back to the constants below. If this file
# exists (written by build_era5_forcing_climatology.py, on the CMIP6 grid), its
# per-cell mean surface pressure and per-cell MONTHLY daytime direct-beam
# fraction REPLACE those constants — an ERA5 climatology instead of a guess.
# Set ERA5_CLIMATOLOGY_FILE = None to keep the constants.
ERA5_CLIMATOLOGY_FILE = Path(
    "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/ERA5/"
    "era5_forcing_climatology_cmip6grid.nc"
)
DEFAULT_PRESSURE_HPA = 900.0   # used where climatology absent / NaN
DEFAULT_FDIR = 0.7             # used where climatology absent / NaN

# WBGT thresholds used in the summary print-out (Gohar et al., matches exposure_map.py)
WBGT_THRESHOLDS = {
    "moderate": 28,
    "high": 30,
    "severe": 32,
}


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


def calculate_bracket_rh(td_k, t_bracket_k):
    """
    Derive relative humidity at a bracket temperature (tasmax or tasmin)
    from a representative dewpoint temperature.

    Assumes dewpoint is approximately constant through the day (a standard
    simplification - dewpoint varies far less over 24h than RH does), so
    RH at the warmer/cooler bracket temperature is just the ratio of
    saturation vapor pressure at the dewpoint to saturation vapor pressure
    at the bracket temperature.

    Parameters
    ----------
    td_k : array-like
        Representative dewpoint temperature in K (from daily-mean tas/hurs)
    t_bracket_k : array-like
        Bracket air temperature in K (tasmax or tasmin)

    Returns
    -------
    rh_bracket : array-like
        Relative humidity in percent (0-100) at the bracket temperature
    """
    e_td = calculate_saturation_vapour_pressure(td_k)
    e_t = calculate_saturation_vapour_pressure(t_bracket_k)
    rh_bracket = 100.0 * e_td / e_t
    return np.clip(rh_bracket, 1, 100)


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
    """
    # Solar declination
    decl = 23.45 * np.sin(np.radians(360 / 365 * (doy - 81)))
    decl_rad = np.radians(decl)
    lat_rad = np.radians(lat)

    cossza_noon = (np.sin(lat_rad) * np.sin(decl_rad) +
                   np.cos(lat_rad) * np.cos(decl_rad))

    cossza_mean = 0.45 * np.clip(cossza_noon, 0, 1)

    return cossza_mean


# ============================================================================
# Data loading and processing
# ============================================================================

def get_time_components(time_val):
    """
    Extract (year, month, day, day_of_year) from a single time coordinate
    value, regardless of whether xarray decoded it as a numpy.datetime64
    (standard calendar) or a cftime object (e.g. DatetimeNoLeap - used by
    NEX-GDDP-CMIP6's 365-day calendar). Which one you get can depend on
    the specific date range in a given file, so both cases are handled
    rather than assuming one or the other. No pandas dependency needed.

    Parameters
    ----------
    time_val : numpy.datetime64 or cftime.datetime
        A single value from a time coordinate array.

    Returns
    -------
    year, month, day, doy : int
        Calendar components and day-of-year (1-365/366).
    """
    if isinstance(time_val, np.datetime64):
        # Standard calendar: go via datetime64[D] -> Python date, which
        # gives .year/.month/.day and .timetuple() without needing pandas.
        d = time_val.astype("datetime64[D]").astype(object)
        return d.year, d.month, d.day, d.timetuple().tm_yday
    else:
        # cftime object (e.g. DatetimeNoLeap) already exposes these directly
        return time_val.year, time_val.month, time_val.day, time_val.timetuple().tm_yday


def find_available_models(data_dir, scenario, year_start, year_end):
    """
    Discover which models have data available, by scanning for tas files
    in the flat Combined/ directory and extracting the model name from
    the filename (Combined/ has no per-model subfolders).
    """
    pattern = f"tas_day_*_{scenario}_malawi_{year_start}_{year_end}.nc"
    prefix = "tas_day_"
    suffix = f"_{scenario}_malawi_{year_start}_{year_end}.nc"

    models = []
    for f in sorted(data_dir.glob(pattern)):
        name = f.name
        if name.startswith(prefix) and name.endswith(suffix):
            model = name[len(prefix):-len(suffix)]
            models.append(model)
    return models


def load_variable(model_dir, scenario, variable, year_start, year_end):
    """Load a variable from the combined netCDF file.

    Returns None if the file doesn't exist, can't be opened, or doesn't
    contain the expected variable. This is intentionally lenient: for
    OPTIONAL_VARS (e.g. "ps", which NASA NEX-GDDP-CMIP6 doesn't publish
    at all), a missing/unreadable file just means the script falls back
    to a default rather than crashing.
    """
    model = model_dir.name
    filepath = DATA_DIR / f"{variable}_day_{model}_{scenario}_malawi_{year_start}_{year_end}.nc"

    if not filepath.exists():
        return None

    try:
        ds = xr.open_dataset(filepath)
        return ds[variable]
    except (FileNotFoundError, OSError, KeyError) as e:
        print(f"    Could not load {variable} from {filepath}: {e}")
        return None


def load_forcing_climatology(lats, lons):
    """Return (ps_hpa[ny,nx], fdir_monthly[12,ny,nx]) from the ERA5 forcing
    climatology interpolated onto this model's grid, or (None, None) if the
    file is not set / not found. NaN cells are filled with the constants."""
    if ERA5_CLIMATOLOGY_FILE is None or not Path(ERA5_CLIMATOLOGY_FILE).exists():
        return None, None
    ds = xr.open_dataset(ERA5_CLIMATOLOGY_FILE)
    # Same 0.25-deg CMIP6 grid -> nearest is exact; also robust if minor offset.
    ds = ds.interp(lat=lats, lon=lons, method="nearest")
    ps = np.asarray(ds["surface_pressure_hpa"].values, dtype=float)
    fd = np.asarray(ds["direct_fraction"].values, dtype=float)   # (12, ny, nx)
    ds.close()
    ps = np.where(np.isfinite(ps), ps, DEFAULT_PRESSURE_HPA)
    fd = np.where(np.isfinite(fd), fd, DEFAULT_FDIR)
    return ps, fd


def calculate_wbgt_for_model(model_dir, scenario, year_start, year_end):
    """
    Calculate day/night-bracketed WBGT for a single model and scenario.

    Computes WBGT twice per day:
      - using tasmax + solar radiation "on", evaluated at solar noon (day bracket)
      - using tasmin + solar radiation "off" (night bracket)

    Returns xarray Dataset with day and night WBGT, globe temperature, and
    wet bulb temperature.
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

    tas = variables["tas"]

    # Some variables (notably sfcWind) can have fewer timesteps than tas if
    # a download year failed - check for that, then reindex everything onto
    # tas's time coordinate (NaN-filling any gap) so isel(time=t_idx) below
    # can't silently pull a mismatched day for one variable vs another.
    for var_name, da in list(variables.items()):
        if var_name == "tas":
            continue
        da_times = set(da.indexes["time"].values)
        tas_times = set(tas.indexes["time"].values)
        missing = sorted(tas_times - da_times)
        if missing:
            print(f"    {len(missing)} dates missing from {var_name} - reindexing onto tas time coord (NaN-filled)")
        variables[var_name] = da.reindex(time=tas.time)

    # Extract arrays (all now aligned to tas's time coordinate)
    tasmax = variables["tasmax"]
    tasmin = variables["tasmin"]
    hurs = variables["hurs"]
    rsds = variables["rsds"]
    sfcwind = variables["sfcWind"]

    # Get coordinates
    lats = tas.lat.values
    lons = tas.lon.values
    times = tas.time.values

    # ERA5 forcing climatology (per-cell pressure + monthly direct fraction)
    clim_ps, clim_fdir = load_forcing_climatology(lats, lons)
    use_clim = clim_ps is not None

    # Pressure priority: real ps (rare for NEX-GDDP) > ERA5 climatology > constant.
    # ps_field is (ny,nx) and constant in time; ps_hpa is a per-time DataArray.
    if "ps" in variables:
        ps_hpa = variables["ps"] / 100.0  # Convert Pa to hPa
        ps_field = None
        print("    Using model surface pressure (ps)")
    elif use_clim:
        ps_hpa = None
        ps_field = clim_ps                # (ny,nx), hPa
        print("    Using ERA5 climatological surface pressure")
    else:
        ps_hpa = None
        ps_field = np.full((len(lats), len(lons)), DEFAULT_PRESSURE_HPA)
        print(f"    Using default pressure: {DEFAULT_PRESSURE_HPA} hPa")

    if use_clim:
        print("    Using ERA5 monthly direct-beam fraction (day bracket)")
    else:
        print(f"    Using default direct-beam fraction: {DEFAULT_FDIR}")

    # Create lat/lon meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    print("  Calculating day/night bracketed WBGT (Liljegren method)...")
    print(f"    Processing {len(times)} timesteps...")

    # Initialize output arrays
    shape = tas.shape
    wbgt_day_out = np.full(shape, np.nan)
    tg_day_out = np.full(shape, np.nan)
    twb_day_out = np.full(shape, np.nan)

    wbgt_night_out = np.full(shape, np.nan)
    tg_night_out = np.full(shape, np.nan)
    twb_night_out = np.full(shape, np.nan)

    # Process each timestep
    for t_idx, time_val in enumerate(times):
        if t_idx % 100 == 0:
            print(f"      Timestep {t_idx}/{len(times)}")

        # Extract data for this timestep
        t_max = tasmax.isel(time=t_idx).values
        t_min = tasmin.isel(time=t_idx).values
        rh_mean = hurs.isel(time=t_idx).values
        t_mean = tas.isel(time=t_idx).values
        solar = rsds.isel(time=t_idx).values
        speed = sfcwind.isel(time=t_idx).values

        if ps_hpa is not None:                 # real per-time model pressure
            p_hpa = ps_hpa.isel(time=t_idx).values
        else:                                  # ERA5-clim or constant (ny,nx)
            p_hpa = ps_field

        # Representative dewpoint from the daily mean - held constant across
        # the day/night brackets (see calculate_bracket_rh docstring)
        td = calculate_dewpoint(t_mean, rh_mean)

        # Derive RH at each bracket temperature from that dewpoint
        rh_day = calculate_bracket_rh(td, t_max)
        rh_night = calculate_bracket_rh(td, t_min)

        # Day of year / calendar date for solar geometry. NEX-GDDP-CMIP6
        # uses a 365-day "noleap" calendar, and depending on the file,
        # xarray may hand back either a cftime object or a numpy.datetime64
        # - get_time_components() handles both rather than assuming one
        # (and never routes through pd.Timestamp, which can't parse cftime).
        year, month, day, doy = get_time_components(time_val)

        # --- Day bracket: solar noon, solar radiation "on" ---
        cossza_day = calculate_cos_solar_zenith_angle(
            lat_grid, lon_grid, year, month, day, hour=12
        )
        cossza_day = np.maximum(cossza_day, 0.05)
        if use_clim:
            fdir_day = clim_fdir[month - 1]        # (ny,nx) for this month
        else:
            fdir_day = np.full_like(solar, DEFAULT_FDIR)

        wbgt_day, tg_day, twb_day = calculate_wbgt(
            t_max, rh_day, p_hpa, solar, fdir_day, cossza_day, speed
        )

        # --- Night bracket: no sun, radiation terms drop out ---
        solar_night = np.zeros_like(solar)
        fdir_night = np.zeros_like(solar)
        cossza_night = np.full_like(solar, 0.01)  # avoid div-by-zero; has no effect since solar=0

        wbgt_night, tg_night, twb_night = calculate_wbgt(
            t_min, rh_night, p_hpa, solar_night, fdir_night, cossza_night, speed
        )

        wbgt_day_out[t_idx] = wbgt_day
        tg_day_out[t_idx] = tg_day
        twb_day_out[t_idx] = twb_day

        wbgt_night_out[t_idx] = wbgt_night
        tg_night_out[t_idx] = tg_night
        twb_night_out[t_idx] = twb_night

    # Create output DataArrays
    coords = {"time": tas.time, "lat": tas.lat, "lon": tas.lon}

    def make_da(data, name, long_name, bracket):
        return xr.DataArray(
            data,
            coords=coords,
            dims=["time", "lat", "lon"],
            name=name,
            attrs={
                "long_name": long_name,
                "units": "degC",
                "method": "Liljegren",
                "bracket": bracket,
                "source_model": model,
                "source_scenario": scenario,
            }
        )

    ds_out = xr.Dataset({
        "wbgt_day": make_da(wbgt_day_out, "wbgt_day",
                             "Wet Bulb Globe Temperature (afternoon peak proxy, from tasmax)", "day"),
        "tg_day": make_da(tg_day_out, "tg_day", "Globe Temperature (day bracket)", "day"),
        "twb_day": make_da(twb_day_out, "twb_day", "Natural Wet Bulb Temperature (day bracket)", "day"),

        "wbgt_night": make_da(wbgt_night_out, "wbgt_night",
                               "Wet Bulb Globe Temperature (overnight low proxy, from tasmin)", "night"),
        "tg_night": make_da(tg_night_out, "tg_night", "Globe Temperature (night bracket)", "night"),
        "twb_night": make_da(twb_night_out, "twb_night", "Natural Wet Bulb Temperature (night bracket)", "night"),
    })

    # Close input datasets
    for da in variables.values():
        da.close()

    return ds_out


# ============================================================================
# Main
# ============================================================================

def main():
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

    # Combined/ is flat (no per-model subfolders), so models are discovered
    # by scanning filenames rather than listing directories.
    for scenario in SCENARIOS:
        model_names = find_available_models(DATA_DIR, scenario, YEAR_START, YEAR_END)
        print(f"\nFound {len(model_names)} models for {scenario}: {model_names}")

        for model in model_names:
            print(f"\n{'=' * 70}")
            print(f"Processing model: {model}")
            print("=" * 70)

            # calculate_wbgt_for_model / load_variable only use model_dir.name
            # to build filenames (Combined/ has no real per-model directory),
            # so a synthetic Path wrapping the model name is sufficient here.
            model_dir = Path(model)

            # Calculate WBGT
            ds = calculate_wbgt_for_model(
                model_dir, scenario, YEAR_START, YEAR_END
            )

            if ds is None:
                print(f"  Could not calculate WBGT for {model}/{scenario}")
                continue

            # Save output (organized into model/scenario subfolders even
            # though the input Combined/ directory itself is flat)
            out_model_dir = OUT_DIR / model / scenario
            out_model_dir.mkdir(parents=True, exist_ok=True)

            out_file = (
                out_model_dir /
                f"wbgt_daynight_{model}_{scenario}_malawi_{YEAR_START}_{YEAR_END}.nc"
            )

            # Encoding for compression
            encoding = {
                var: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for var in ds.data_vars
            }

            ds.to_netcdf(out_file, format="NETCDF4", encoding=encoding)

            print(f"\n  Saved: {out_file}")

            # Print summary statistics for both brackets
            for bracket in ["day", "night"]:
                wbgt = ds[f"wbgt_{bracket}"]
                print(f"\n  WBGT ({bracket} bracket) statistics:")
                print(f"    Min:  {float(wbgt.min()):.1f}°C")
                print(f"    Mean: {float(wbgt.mean()):.1f}°C")
                print(f"    Max:  {float(wbgt.max()):.1f}°C")
                for name, threshold in WBGT_THRESHOLDS.items():
                    n_exceed = int((wbgt > threshold).sum())
                    pct_exceed = 100 * (wbgt > threshold).mean().item()
                    print(f"    Days > {threshold}°C ({name}): {n_exceed} ({pct_exceed:.1f}%)")

            ds.close()

    print("\n" + "=" * 70)
    print("Done calculating WBGT for all models and scenarios.")
    print("=" * 70)


if __name__ == "__main__":
    main()
