"""
Calculate WBGT from combined ERA5 hourly reanalysis data for Malawi — vectorised.

Same Liljegren method (thermofeel-style) as calculating_wbgt_thermofeel.py, but the
per-timestep Python loop is removed: all fields are loaded into memory once and the
iterative solvers run on the full (time, lat, lon) cube in a single call. For the
Malawi domain (~500 cells x ~17.5k hourly steps) this is ~35 MB per float32 field,
so the whole computation fits in memory easily. TIME_CHUNK bounds memory anyway
in case the domain or period grows.

Changes vs. the original script (beyond vectorisation):
  - FIXED: output coords used t2m.lat / t2m.lon (ERA5 uses latitude/longitude) —
    this crashed AFTER the full loop had run. Coords are now built and validated
    BEFORE any computation (fail fast).
  - Solar zenith angle is computed at the accumulation-interval MIDPOINT
    (hour - 0.5), since ssrd/fdir are accumulated over the hour ending at the
    timestamp. Removes dawn/dusk direct-beam artifacts.
  - Explicit dimension assertions catch expver/number dims from new-CDS files.
  - load_era5_variable prefers the expected short name over data_vars[0].
  - Convergence-failure (NaN) count is reported loudly after each solve.
  - fdir fraction is 0 (not fdir/1) where solar <= 1 W/m^2.
  - Exceedance stats exclude NaNs explicitly.
  - float32 throughout (precision at 300 K is ~3e-5 K; convergence tol is 0.02 K).

WBGT = 0.7 * Twb + 0.2 * Tg + 0.1 * Ta
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# ============================================================================
# Physical constants (unchanged from calculating_wbgt_thermofeel.py)
# ============================================================================

M_AIR = 28.97
M_H2O = 18.015
R_GAS = 8314.34
Cp = 1003.5
STEFANB = 5.6696e-8
R_AIR = R_GAS / M_AIR
RATIO = Cp * M_AIR / M_H2O
Pr = Cp / (Cp + 1.25 * R_AIR)

D_GLOBE = 0.0508
EMIS_GLOBE = 0.95
ALB_GLOBE = 0.05

EMIS_WICK = 0.95
ALB_WICK = 0.4
D_WICK = 0.007
L_WICK = 0.0254

EMIS_SFC = 0.999
ALB_SFC = 0.45

MAX_ITER = 50
CONVERGENCE = 0.02
MIN_SPEED = 0.13

MISSING_VALUE = np.nan

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/ERA5/Combined")
OUT_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/ERA5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_START = 2010
YEAR_END = 2024

# Number of timesteps solved per block. The solvers are fully vectorised within
# a block; blocks exist only to bound peak memory. 2000 steps x Malawi domain
# is ~4 MB per field per block in float32 — raise freely if the domain is small.
TIME_CHUNK = 2000

DTYPE = np.float32

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
# Thermodynamic helper functions (unchanged — already fully vectorised)
# ============================================================================

def calculate_saturation_vapour_pressure(t_k):
    """Saturation vapor pressure via Buck equation. Returns hPa."""
    t_c = t_k - 273.15
    return 6.1121 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))


def viscosity(t_k):
    omega = 1.2945 - t_k / 1141.176470588
    return 0.0000026693 * np.sqrt(28.97 * t_k) / (13.082689 * omega)


def thermal_conductivity(t_k):
    return (Cp + 1.25 * R_AIR) * viscosity(t_k)


def diffusivity(t_k, p_pa):
    return (2.471773765165648e-05 *
            (t_k * 0.0034210563748421257) ** 2.334 *
            (p_pa / 101325) ** (-1))


def evap(t_k):
    return 1665134.5 + 2370.0 * t_k


def emis_atm(t_k, rh_percent):
    e_sat = calculate_saturation_vapour_pressure(t_k)
    e = (rh_percent / 100.0) * e_sat
    return 0.575 * (e ** 0.143)


def h_sphere_in_air(diameter, t_air, p_hpa, speed):
    density = p_hpa * 100.0 / (R_AIR * t_air)
    speed_eff = np.maximum(speed, MIN_SPEED)
    Re = speed_eff * density * diameter / viscosity(t_air)
    Nu = 2.0 + 0.6 * np.sqrt(Re) * (Pr ** 0.3333)
    return Nu * thermal_conductivity(t_air) / diameter


def h_cylinder_in_air(diameter, length, t_air, p_hpa, speed):
    a = 0.56
    b = 0.281
    c = 0.4
    density = p_hpa * 100.0 / (R_AIR * t_air)
    speed_eff = np.maximum(speed, MIN_SPEED)
    Re = speed_eff * density * diameter / viscosity(t_air)
    Nu = b * (Re ** (1.0 - c)) * (Pr ** (1.0 - a))
    return Nu * thermal_conductivity(t_air) / diameter


# ============================================================================
# Core WBGT solvers (Liljegren method — unchanged math, now called on 3-D cubes)
# ============================================================================

def calculate_globe_temperature(t_air, rh, p_hpa, solar, fdir, cossza, speed):
    t_sfc = t_air
    t_globe_prev = np.copy(t_air)
    cossza_safe = np.where(cossza > 0.01, cossza, 0.01)
    converged = np.zeros_like(t_air, dtype=bool)
    t_globe_new = t_globe_prev

    for _ in range(MAX_ITER):
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

        converged = converged | (np.abs(t_globe_new - t_globe_prev) < CONVERGENCE)
        t_globe_prev = np.where(converged, t_globe_prev,
                                0.9 * t_globe_prev + 0.1 * t_globe_new)
        if np.all(converged):
            break

    return np.where(converged, t_globe_new, MISSING_VALUE)


def calculate_natural_wet_bulb(t_air, td, rh, p_hpa, speed, solar, fdir, cossza):
    a = 0.56
    t_sfc = t_air
    cossza_safe = np.clip(cossza, 0.01, 1.0)
    sza = np.arccos(cossza_safe)
    e_air = (rh / 100.0) * calculate_saturation_vapour_pressure(t_air)
    t_wb_prev = np.copy(td)
    converged = np.zeros_like(t_air, dtype=bool)
    t_wb_new = t_wb_prev

    for _ in range(MAX_ITER):
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

        converged = converged | (np.abs(t_wb_new - t_wb_prev) < CONVERGENCE)
        t_wb_prev = np.where(converged, t_wb_prev,
                             0.9 * t_wb_prev + 0.1 * t_wb_new)
        if np.all(converged):
            break

    return np.where(converged, t_wb_new, MISSING_VALUE)


# ============================================================================
# Vectorised solar geometry — all timesteps at once
# ============================================================================

def cos_solar_zenith_angle_vectorised(lat_grid, lon_grid, times):
    """cossza with shape (time, lat, lon), computed at the MIDPOINT of each
    hourly accumulation interval (timestamp - 30 min), matching how ERA5
    ssrd/fdir are accumulated over the hour ending at the timestamp."""
    ti = pd.DatetimeIndex(times)
    doy = ti.dayofyear.to_numpy(dtype=DTYPE)
    # interval midpoint: hour ending at ts -> representative instant ts - 0.5 h
    hour = (ti.hour + ti.minute / 60.0).to_numpy(dtype=DTYPE) - 0.5

    decl_rad = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (doy - 81)))
                          ).astype(DTYPE)                      # (nt,)
    lat_rad = np.radians(lat_grid).astype(DTYPE)               # (ny, nx)

    solar_time = hour[:, None, None] + lon_grid[None, :, :] / 15.0   # (nt, ny, nx)
    hour_angle = np.radians(15.0 * (solar_time - 12.0))

    cossza = (np.sin(lat_rad)[None] * np.sin(decl_rad)[:, None, None] +
              np.cos(lat_rad)[None] * np.cos(decl_rad)[:, None, None] *
              np.cos(hour_angle))
    return np.clip(cossza, 0.0, 1.0).astype(DTYPE)


# ============================================================================
# ERA5-specific data loading and RH derivation
# ============================================================================

def load_era5_variable(short_name):
    long_name = VAR_FILES[short_name]
    filepath = DATA_DIR / f"{long_name}_{YEAR_START}_{YEAR_END}.nc"
    if not filepath.exists():
        raise FileNotFoundError(f"Missing combined file: {filepath}")
    ds = xr.open_dataset(filepath)

    # Prefer the expected short CF name; fall back to first data_var, but skip
    # bookkeeping variables that new-CDS files sometimes include.
    if short_name in ds.data_vars:
        varname = short_name
    else:
        candidates = [v for v in ds.data_vars if v not in ("expver", "number")]
        if not candidates:
            raise ValueError(f"No usable data variable in {filepath}: {list(ds.data_vars)}")
        varname = candidates[0]

    da = ds[varname]

    # Normalise old-CDS 'time' to 'valid_time' so the rest of the script is uniform
    if "time" in da.dims and "valid_time" not in da.dims:
        da = da.rename({"time": "valid_time"})

    # Squeeze out singleton expver/number dims; refuse non-singleton ones loudly
    for extra in ("expver", "number"):
        if extra in da.dims:
            if da.sizes[extra] == 1:
                da = da.squeeze(extra, drop=True)
            else:
                raise ValueError(
                    f"{filepath} has non-singleton '{extra}' dim (size {da.sizes[extra]}). "
                    f"Request likely spans the ERA5/ERA5T boundary — combine expver "
                    f"slices before running (e.g. ds.sel(expver=1).combine_first(ds.sel(expver=5)))."
                )

    expected = ("valid_time", "latitude", "longitude")
    if tuple(da.dims) != expected:
        raise ValueError(f"{varname} in {filepath} has dims {da.dims}, expected {expected}")

    return da.astype(DTYPE).load()  # eager load: one disk read per variable, ever


def rh_from_dewpoint(t2m_k, d2m_k):
    e_sat_t = calculate_saturation_vapour_pressure(t2m_k)
    e_sat_td = calculate_saturation_vapour_pressure(d2m_k)
    return np.clip(100.0 * e_sat_td / e_sat_t, 1, 100)


# ============================================================================
# Main WBGT calculation for ERA5 — vectorised
# ============================================================================

def calculate_wbgt_era5():
    print("Loading ERA5 variables (eager, one read per file)...")
    t2m = load_era5_variable("t2m")
    d2m = load_era5_variable("d2m")
    u10 = load_era5_variable("u10")
    v10 = load_era5_variable("v10")
    ssrd = load_era5_variable("ssrd")
    fdir_raw = load_era5_variable("fdir")
    sp = load_era5_variable("sp")

    # ---- Build output coords FIRST so any naming problem fails before compute
    times = t2m.valid_time.values
    lats = t2m.latitude.values
    lons = t2m.longitude.values
    coords = {"time": times, "lat": lats, "lon": lons}
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    nt, ny, nx = t2m.shape
    print(f"Grid: {ny} x {nx} = {ny * nx} cells; {nt} hourly timesteps")

    # ---- Pull everything out to numpy once (no per-step isel)
    t_air = t2m.values
    td = d2m.values
    speed = np.sqrt(u10.values ** 2 + v10.values ** 2)
    p_hpa = sp.values / 100.0

    # ssrd/fdir accumulated over the hour ending at the timestamp -> W/m^2
    solar = np.clip(ssrd.values / 3600.0, 0, None)
    fdir_wm2 = np.clip(fdir_raw.values / 3600.0, 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        fdir_frac = np.where(solar > 1.0, fdir_wm2 / np.where(solar > 1.0, solar, 1.0), 0.0)
    fdir_frac = np.clip(fdir_frac, 0.0, 1.0)

    rh = rh_from_dewpoint(t_air, td)

    print("Computing solar zenith angles (vectorised, interval midpoints)...")
    cossza = cos_solar_zenith_angle_vectorised(lat_grid, lon_grid, times)
    cossza = np.maximum(cossza, 0.05)

    # ---- Solve in large blocks (fully vectorised within each block)
    wbgt_out = np.full((nt, ny, nx), np.nan, dtype=DTYPE)
    tg_out = np.full((nt, ny, nx), np.nan, dtype=DTYPE)
    twb_out = np.full((nt, ny, nx), np.nan, dtype=DTYPE)

    n_blocks = int(np.ceil(nt / TIME_CHUNK))
    print(f"Solving Liljegren WBGT in {n_blocks} block(s) of up to {TIME_CHUNK} timesteps...")
    for b in range(n_blocks):
        s = slice(b * TIME_CHUNK, min((b + 1) * TIME_CHUNK, nt))
        print(f"  Block {b + 1}/{n_blocks}  [{s.start}:{s.stop}]")

        t_globe = calculate_globe_temperature(
            t_air[s], rh[s], p_hpa[s], solar[s], fdir_frac[s], cossza[s], speed[s])
        t_wb = calculate_natural_wet_bulb(
            t_air[s], td[s], rh[s], p_hpa[s], speed[s], solar[s], fdir_frac[s], cossza[s])

        wbgt_out[s] = (0.7 * (t_wb - 273.15) + 0.2 * (t_globe - 273.15)
                       + 0.1 * (t_air[s] - 273.15))
        tg_out[s] = t_globe - 273.15
        twb_out[s] = t_wb - 273.15

    # ---- Loud convergence-failure report (silent-NaN guard)
    n_nan = int(np.isnan(wbgt_out).sum())
    if n_nan:
        pct = 100.0 * n_nan / wbgt_out.size
        print(f"\n*** WARNING: {n_nan} cells ({pct:.3f}%) failed to converge within "
              f"{MAX_ITER} iterations and are NaN in the output. ***")
        print("*** Consider raising MAX_ITER or strengthening the relaxation factor. ***")
    else:
        print("\nAll cells converged.")

    ds = xr.Dataset({
        "wbgt": xr.DataArray(
            wbgt_out, coords=coords, dims=["time", "lat", "lon"], name="wbgt",
            attrs={"long_name": "Wet Bulb Globe Temperature", "units": "degC",
                   "method": "Liljegren", "source": "ERA5",
                   "solar_geometry": "cossza at accumulation-interval midpoint (ts - 30 min)"},
        ),
        "tg": xr.DataArray(
            tg_out, coords=coords, dims=["time", "lat", "lon"], name="tg",
            attrs={"long_name": "Globe Temperature", "units": "degC", "method": "Liljegren"},
        ),
        "twb": xr.DataArray(
            twb_out, coords=coords, dims=["time", "lat", "lon"], name="twb",
            attrs={"long_name": "Natural Wet Bulb Temperature", "units": "degC",
                   "method": "Liljegren"},
        ),
    })

    out_file = OUT_DIR / f"wbgt_hourly_ERA5_malawi_{YEAR_START}_{YEAR_END}.nc"
    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
    ds.to_netcdf(out_file, format="NETCDF4", encoding=encoding)
    print(f"\nSaved hourly output: {out_file}")

    # ---- Summary statistics (NaN-aware)
    wbgt = ds["wbgt"]
    valid = np.isfinite(wbgt_out)
    n_valid = int(valid.sum())
    print("\nWBGT statistics (valid cells only):")
    print(f"  Min:  {np.nanmin(wbgt_out):.1f}°C")
    print(f"  Mean: {np.nanmean(wbgt_out):.1f}°C")
    print(f"  Max:  {np.nanmax(wbgt_out):.1f}°C")
    for thresh in (28, 32):
        n_exc = int((wbgt_out > thresh).sum())  # NaN > x is False, so this counts valid exceedances
        print(f"  Hours > {thresh}°C: {n_exc} ({100.0 * n_exc / n_valid:.1f}% of valid)")

    # ---- Daily max (UTC days). Note: Malawi is UTC+2; the local afternoon peak
    # (~12-14 UTC) sits mid-UTC-day so daily MAX is safe, but do NOT reuse this
    # binning for daily min / night stats — shift time by +2 h first.
    daily_max = wbgt.resample(time="1D").max()
    daily_max.name = "wbgt_daily_max"
    daily_max.attrs = {
        "long_name": "Daily Maximum Wet Bulb Globe Temperature",
        "units": "degC", "method": "Liljegren", "source": "ERA5",
        "note": "Binned on UTC days; safe for daily max in Malawi (UTC+2), "
                "not for daily min/night statistics.",
    }
    daily_file = OUT_DIR / f"wbgt_daily_max_ERA5_malawi_{YEAR_START}_{YEAR_END}.nc"
    daily_max.to_netcdf(daily_file, encoding={
        "wbgt_daily_max": {"zlib": True, "complevel": 4, "dtype": "float32"}})
    print(f"Saved daily max output: {daily_file}")

    ds.close()


if __name__ == "__main__":
    print("=" * 70)
    print("WBGT from ERA5 Hourly Reanalysis (Liljegren, vectorised)")
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
