"""
Calculate WBGT from combined ERA5 hourly reanalysis data for Malawi.

Uses the same Liljegren method (thermofeel-style) as calculating_wbgt_thermofeel.py,
applied to real ERA5 hourly fields instead of daily-mean NEX-GDDP-CMIP6 fields. This
is the same underlying formula, run on higher-fidelity inputs:

    - Relative humidity is derived from real 2m dewpoint (vapour pressure ratio),
      not inverted from RH-only data via Magnus.
    - Solar zenith angle is computed per exact hour, not approximated from a
      flat daily-mean scaling factor.
    - Direct/diffuse radiation fraction (fdir) comes from ERA5's own field,
      not assumed constant at 0.7.

WBGT = 0.7 * Twb + 0.2 * Tg + 0.1 * Ta

Where:
- Twb = natural wet bulb temperature
- Tg = globe temperature
- Ta = air temperature

Because the core Liljegren equations are unchanged from the CMIP6 pipeline,
WBGT values here are directly comparable to calculating_wbgt_thermofeel.py /
calculating_wbgt_thermofeel_daynight.py output — differences reflect input
resolution, not formula differences.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# ============================================================================
# Physical constants (unchanged from calculating_wbgt_thermofeel.py)
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

DATA_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/ERA5/Combined")
OUT_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/ERA5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_START = 2010
YEAR_END = 2024

VAR_FILES = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "ssrd": "surface_solar_radiation_downwards",
    "fdir": "total_sky_direct_solar_radiation_at_surface",
    "sp": "surface_pressure",
}


# ============================================================================
# Thermodynamic helper functions (unchanged from calculating_wbgt_thermofeel.py)
# ============================================================================

def calculate_saturation_vapour_pressure(t_k):
    """Saturation vapor pressure via Buck equation. Returns hPa."""
    t_c = t_k - 273.15
    e_sat = 6.1121 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))
    return e_sat


def viscosity(t_k):
    omega = 1.2945 - t_k / 1141.176470588
    visc = 0.0000026693 * np.sqrt(28.97 * t_k) / (13.082689 * omega)
    return visc


def thermal_conductivity(t_k):
    return (Cp + 1.25 * R_AIR) * viscosity(t_k)


def diffusivity(t_k, p_pa):
    diff = (2.471773765165648e-05 *
            (t_k * 0.0034210563748421257) ** 2.334 *
            (p_pa / 101325) ** (-1))
    return diff


def evap(t_k):
    return 1665134.5 + 2370.0 * t_k


def emis_atm(t_k, rh_percent):
    """Atmospheric emissivity. Reference: Oke (2nd edition), page 373."""
    e_sat = calculate_saturation_vapour_pressure(t_k)
    rh = rh_percent / 100.0
    e = rh * e_sat  # vapor pressure in hPa
    return 0.575 * (e ** 0.143)


def h_sphere_in_air(diameter, t_air, p_hpa, speed):
    """Convective heat transfer coefficient, flow around a sphere (Bird/Stewart/Lightfoot p.409)."""
    density = p_hpa * 100.0 / (R_AIR * t_air)
    speed_eff = np.maximum(speed, MIN_SPEED)
    Re = speed_eff * density * diameter / viscosity(t_air)
    Nu = 2.0 + 0.6 * np.sqrt(Re) * (Pr ** 0.3333)
    return Nu * thermal_conductivity(t_air) / diameter


def h_cylinder_in_air(diameter, length, t_air, p_hpa, speed):
    """Convective heat transfer coefficient, flow around a cylinder (Bedingfield and Drew)."""
    a = 0.56
    b = 0.281
    c = 0.4
    density = p_hpa * 100.0 / (R_AIR * t_air)
    speed_eff = np.maximum(speed, MIN_SPEED)
    Re = speed_eff * density * diameter / viscosity(t_air)
    Nu = b * (Re ** (1.0 - c)) * (Pr ** (1.0 - a))
    return Nu * thermal_conductivity(t_air) / diameter


# ============================================================================
# Core WBGT calculations (Liljegren method, unchanged)
# ============================================================================

def calculate_globe_temperature(t_air, rh, p_hpa, solar, fdir, cossza, speed):
    t_sfc = t_air
    t_globe_prev = np.copy(t_air)
    cossza_safe = np.where(cossza > 0.01, cossza, 0.01)
    converged = np.zeros_like(t_air, dtype=bool)

    for iteration in range(MAX_ITER):
        t_ref = 0.5 * (t_globe_prev + t_air)
        h = h_sphere_in_air(D_GLOBE, t_ref, p_hpa, speed)

        t_globe_new = np.power(
            0.5 * (emis_atm(t_air, rh) * np.power(t_air, 4.0) +
                   EMIS_SFC * np.power(t_sfc, 4.0))
            - h / (STEFANB * EMIS_GLOBE) * (t_globe_prev - t_air)
            + solar / (2.0 * STEFANB * EMIS_GLOBE) * (1.0 - ALB_GLOBE) *
            (fdir * (1.0 / (2.0 * cossza_safe) - 1.0) + 1.0 + ALB_SFC),
            0.25
        )

        diff = np.abs(t_globe_new - t_globe_prev)
        newly_converged = diff < CONVERGENCE
        converged = converged | newly_converged
        t_globe_prev = np.where(converged, t_globe_prev, 0.9 * t_globe_prev + 0.1 * t_globe_new)

        if np.all(converged):
            break

    return np.where(converged, t_globe_new, MISSING_VALUE)


def calculate_natural_wet_bulb(t_air, td, rh, p_hpa, speed, solar, fdir, cossza):
    a = 0.56  # from Bedingfield and Drew
    t_sfc = t_air
    cossza_safe = np.clip(cossza, 0.01, 1.0)
    sza = np.arccos(cossza_safe)
    e_air = (rh / 100.0) * calculate_saturation_vapour_pressure(t_air)
    t_wb_prev = np.copy(td)
    converged = np.zeros_like(t_air, dtype=bool)

    for iteration in range(MAX_ITER):
        t_ref = 0.5 * (t_wb_prev + t_air)
        h = h_cylinder_in_air(D_WICK, L_WICK, t_ref, p_hpa, speed)

        F_atm = (STEFANB * EMIS_WICK *
                 (0.5 * (emis_atm(t_air, rh) * np.power(t_air, 4.0) +
                         EMIS_SFC * np.power(t_sfc, 4.0))
                  - np.power(t_wb_prev, 4.0))
                 + (1.0 - ALB_WICK) * solar *
                 ((1.0 - fdir) * (1.0 + 0.25 * D_WICK / L_WICK)
                  + fdir * (np.tan(sza) / np.pi + 0.25 * D_WICK / L_WICK)
                  + ALB_SFC))

        e_wick = calculate_saturation_vapour_pressure(t_wb_prev)
        density = p_hpa * 100.0 / (R_AIR * t_ref)
        Sc = viscosity(t_ref) / (density * diffusivity(t_ref, p_hpa * 100.0))

        t_wb_new = (t_air
                    - evap(t_ref) / RATIO * (e_wick - e_air) / (p_hpa - e_wick) *
                    np.power(Pr / Sc, a)
                    + F_atm / h)

        diff = np.abs(t_wb_new - t_wb_prev)
        newly_converged = diff < CONVERGENCE
        converged = converged | newly_converged
        t_wb_prev = np.where(converged, t_wb_prev, 0.9 * t_wb_prev + 0.1 * t_wb_new)

        if np.all(converged):
            break

    return np.where(converged, t_wb_new, MISSING_VALUE)


def calculate_cos_solar_zenith_angle(lat, lon, year, month, day, hour=12):
    """Cosine of solar zenith angle for an exact hour (unchanged from CMIP6 script,
    but now called with the real ERA5 hour rather than a fixed noon default)."""
    from datetime import datetime
    dt = datetime(year, month, day)
    doy = dt.timetuple().tm_yday

    decl = 23.45 * np.sin(np.radians(360 / 365 * (doy - 81)))
    decl_rad = np.radians(decl)

    solar_time = hour + lon / 15.0
    hour_angle = np.radians(15 * (solar_time - 12))

    lat_rad = np.radians(lat)

    cossza = (np.sin(lat_rad) * np.sin(decl_rad) +
              np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle))

    return np.clip(cossza, 0, 1)


# ============================================================================
# ERA5-specific data loading and RH derivation
# ============================================================================

def load_era5_variable(short_name):
    long_name = VAR_FILES[short_name]
    filepath = DATA_DIR / f"{long_name}_{YEAR_START}_{YEAR_END}.nc"
    if not filepath.exists():
        raise FileNotFoundError(f"Missing combined file: {filepath}")
    ds = xr.open_dataset(filepath)
    # ERA5 netCDF variable names are usually the short CF names (t2m, d2m, u10, v10, sp)
    # but ssrd/fdir keep those names too — grab whichever data_var is present
    varname = list(ds.data_vars)[0]
    return ds[varname]


def rh_from_dewpoint(t2m_k, d2m_k):
    """RH via ratio of saturation vapour pressures — real Td, so no Magnus inversion needed."""
    e_sat_t = calculate_saturation_vapour_pressure(t2m_k)
    e_sat_td = calculate_saturation_vapour_pressure(d2m_k)
    return np.clip(100.0 * e_sat_td / e_sat_t, 1, 100)


# ============================================================================
# Main WBGT calculation for ERA5
# ============================================================================

def calculate_wbgt_era5():
    print("Loading ERA5 variables...")
    t2m = load_era5_variable("t2m")
    d2m = load_era5_variable("d2m")
    u10 = load_era5_variable("u10")
    v10 = load_era5_variable("v10")
    ssrd = load_era5_variable("ssrd")
    fdir_raw = load_era5_variable("fdir")
    sp = load_era5_variable("sp")

    speed = np.sqrt(u10 ** 2 + v10 ** 2)

    # ERA5 single-levels reanalysis: ssrd/fdir are accumulated over the hour
    # ending at the timestamp -> divide by 3600 to get W/m^2, clip negatives
    # (small negative artifacts near sunrise/sunset are common in ERA5 radiation fields)
    solar = (ssrd / 3600.0).clip(min=0)
    fdir_wm2 = (fdir_raw / 3600.0).clip(min=0)
    fdir_frac = (fdir_wm2 / solar.where(solar > 1, 1)).clip(0, 1)  # direct fraction, not raw W/m^2

    ps_hpa = sp / 100.0

    lats = t2m.lat.values
    lons = t2m.lon.values
    times = t2m.time.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    print(f"Calculating WBGT (Liljegren method) for {len(times)} hourly timesteps...")

    shape = t2m.shape
    wbgt_out = np.full(shape, np.nan)
    tg_out = np.full(shape, np.nan)
    twb_out = np.full(shape, np.nan)

    for t_idx, time_val in enumerate(times):
        if t_idx % 500 == 0:
            print(f"  Timestep {t_idx}/{len(times)}")

        ts = pd.Timestamp(time_val)
        t_air = t2m.isel(time=t_idx).values
        td = d2m.isel(time=t_idx).values
        rh = rh_from_dewpoint(t_air, td)
        p_hpa = ps_hpa.isel(time=t_idx).values
        sol = solar.isel(time=t_idx).values
        fd = fdir_frac.isel(time=t_idx).values
        spd = speed.isel(time=t_idx).values

        # real solar geometry for this exact hour — no daily-mean approximation
        cossza = calculate_cos_solar_zenith_angle(
            lat_grid, lon_grid, ts.year, ts.month, ts.day, hour=ts.hour
        )
        cossza = np.maximum(cossza, 0.05)

        t_globe = calculate_globe_temperature(t_air, rh, p_hpa, sol, fd, cossza, spd)
        t_wb = calculate_natural_wet_bulb(t_air, td, rh, p_hpa, spd, sol, fd, cossza)

        wbgt_out[t_idx] = 0.7 * (t_wb - 273.15) + 0.2 * (t_globe - 273.15) + 0.1 * (t_air - 273.15)
        tg_out[t_idx] = t_globe - 273.15
        twb_out[t_idx] = t_wb - 273.15

    coords = {"time": t2m.time, "lat": t2m.lat, "lon": t2m.lon}
    ds = xr.Dataset({
        "wbgt": xr.DataArray(
            wbgt_out, coords=coords, dims=["time", "lat", "lon"], name="wbgt",
            attrs={
                "long_name": "Wet Bulb Globe Temperature",
                "units": "degC",
                "method": "Liljegren",
                "source": "ERA5",
            },
        ),
        "tg": xr.DataArray(
            tg_out, coords=coords, dims=["time", "lat", "lon"], name="tg",
            attrs={"long_name": "Globe Temperature", "units": "degC", "method": "Liljegren"},
        ),
        "twb": xr.DataArray(
            twb_out, coords=coords, dims=["time", "lat", "lon"], name="twb",
            attrs={"long_name": "Natural Wet Bulb Temperature", "units": "degC", "method": "Liljegren"},
        ),
    })

    out_file = OUT_DIR / f"wbgt_hourly_ERA5_malawi_{YEAR_START}_{YEAR_END}.nc"
    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
    ds.to_netcdf(out_file, format="NETCDF4", encoding=encoding)
    print(f"\nSaved hourly output: {out_file}")

    # Print summary statistics
    wbgt = ds["wbgt"]
    print("\nWBGT statistics:")
    print(f"  Min:  {float(wbgt.min()):.1f}°C")
    print(f"  Mean: {float(wbgt.mean()):.1f}°C")
    print(f"  Max:  {float(wbgt.max()):.1f}°C")
    print(f"  Hours > 28°C: {int((wbgt > 28).sum())} ({100 * float((wbgt > 28).mean()):.1f}%)")
    print(f"  Hours > 32°C: {int((wbgt > 32).sum())} ({100 * float((wbgt > 32).mean()):.1f}%)")

    # Daily max aggregation, for comparability with the day/night-bracketed
    # CMIP6 pipeline (wbgt_day is also intended to capture near-peak conditions)
    daily_max = wbgt.resample(time="1D").max()
    daily_max.name = "wbgt_daily_max"
    daily_max.attrs = {
        "long_name": "Daily Maximum Wet Bulb Globe Temperature",
        "units": "degC",
        "method": "Liljegren",
        "source": "ERA5",
    }
    daily_file = OUT_DIR / f"wbgt_daily_max_ERA5_malawi_{YEAR_START}_{YEAR_END}.nc"
    daily_max.to_netcdf(daily_file, encoding={"wbgt_daily_max": {"zlib": True, "complevel": 4, "dtype": "float32"}})
    print(f"Saved daily max output: {daily_file}")

    ds.close()


if __name__ == "__main__":
    print("=" * 70)
    print("WBGT Calculation from ERA5 Hourly Reanalysis (Liljegren Method)")
    print("=" * 70)
    print(f"\nInput directory: {DATA_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Years: {YEAR_START}-{YEAR_END}")

    if not DATA_DIR.exists():
        print(f"\nError: Input directory does not exist: {DATA_DIR}")
    else:
        calculate_wbgt_era5()
        print("\n" + "=" * 70)
        print("Done calculating WBGT from ERA5.")
        print("=" * 70)
