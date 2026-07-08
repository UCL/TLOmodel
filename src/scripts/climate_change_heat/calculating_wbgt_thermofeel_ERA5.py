"""
Calculate WBGT from ERA5 hourly reanalysis data for Malawi — vectorised.
WBGT = 0.7 * Twb + 0.2 * Tg + 0.1 * Ta  (Liljegren method)

Outputs
-------
wbgt_hourly_ERA5_malawi_<YEAR_START>_<YEAR_END>.nc   hourly wbgt, tg, twb
wbgt_daily_max_ERA5_malawi_<YEAR_START>_<YEAR_END>.nc  UTC daily max wbgt
wbgt_monthly_ERA5_historical.nc                        monthly wbgt_day/night
wbgt_wbgt5x_ERA5_historical.nc                         monthly WBGT5x day/night
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# ============================================================================
# Physical constants
# ============================================================================
M_AIR = 28.97;  M_H2O = 18.015;  R_GAS = 8314.34;  Cp = 1003.5
STEFANB = 5.6696e-8;  R_AIR = R_GAS / M_AIR;  RATIO = Cp * M_AIR / M_H2O
Pr = Cp / (Cp + 1.25 * R_AIR)
D_GLOBE = 0.0508;  EMIS_GLOBE = 0.95;  ALB_GLOBE = 0.05
EMIS_WICK = 0.95;  ALB_WICK = 0.4;  D_WICK = 0.007;  L_WICK = 0.0254
EMIS_SFC = 0.999;  ALB_SFC = 0.45
MAX_ITER = 50;  CONVERGENCE = 0.02;  MIN_SPEED = 0.13

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR   = Path("/Users/rachelmurray-watson/Documents/Heat_data/ERA5/Combined")
OUT_DIR    = Path("/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/ERA5")
CMIP6_DIR  = Path("/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/NASA_GDDP_CMIP6_Split")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CMIP6_DIR.mkdir(parents=True, exist_ok=True)

YEAR_START       = 2010
YEAR_END         = 2024
TIME_CHUNK       = 2000
DTYPE            = np.float32
UTC_OFFSET       = 2          # Malawi UTC+2
DAY_START        = 6          # local hour, inclusive
DAY_END          = 17         # local hour, inclusive
WBGT5X_WINDOW    = 5          # days

VAR_FILES = {
    "t2m": "2m_temperature",           "d2m": "2m_dewpoint_temperature",
    "u10": "10m_u_component_of_wind",  "v10": "10m_v_component_of_wind",
    "ssrd":"surface_solar_radiation_downwards",
    "fdir":"total_sky_direct_solar_radiation_at_surface",
    "sp":  "surface_pressure",
}

# ============================================================================
# Thermodynamic helpers
# ============================================================================
def svp(t):        # saturation vapour pressure (hPa), Buck equation
    c = t - 273.15
    return 6.1121 * np.exp((18.678 - c / 234.5) * (c / (257.14 + c)))

def visc(t):
    return 0.0000026693 * np.sqrt(28.97 * t) / (13.082689 * (1.2945 - t / 1141.176470588))

def kth(t):        return (Cp + 1.25 * R_AIR) * visc(t)
def diff(t, p):    return 2.471773765165648e-05 * (t * 0.0034210563748421257)**2.334 / (p / 101325)
def evap(t):       return 1665134.5 + 2370.0 * t
def emis_atm(t, rh): return 0.575 * ((rh / 100.0) * svp(t)) ** 0.143

def h_sphere(t, p, spd):
    Re = np.maximum(spd, MIN_SPEED) * p * 100 / (R_AIR * t) * D_GLOBE / visc(t)
    return (2.0 + 0.6 * Re**0.5 * Pr**0.3333) * kth(t) / D_GLOBE

def h_cyl(t, p, spd):
    Re = np.maximum(spd, MIN_SPEED) * p * 100 / (R_AIR * t) * D_WICK / visc(t)
    return 0.281 * Re**0.6 * Pr**0.44 * kth(t) / D_WICK

# ============================================================================
# WBGT solvers
# ============================================================================
def globe_temp(t, rh, p, sol, fdir, cossza, spd):
    cz  = np.where(cossza > 0.01, cossza, 0.01)
    tg  = t.copy();  conv = np.zeros_like(t, dtype=bool)
    for _ in range(MAX_ITER):
        h   = h_sphere(0.5 * (tg + t), p, spd)
        tgn = (0.5 * (emis_atm(t, rh) * t**4 + EMIS_SFC * t**4)
               - h / (STEFANB * EMIS_GLOBE) * (tg - t)
               + sol / (2 * STEFANB * EMIS_GLOBE) * (1 - ALB_GLOBE)
               * (fdir * (1 / (2 * cz) - 1) + 1 + ALB_SFC)) ** 0.25
        conv |= np.abs(tgn - tg) < CONVERGENCE
        tg   = np.where(conv, tg, 0.9 * tg + 0.1 * tgn)
        if conv.all(): break
    return np.where(conv, tgn, np.nan)

def wet_bulb(t, td, rh, p, spd, sol, fdir, cossza):
    cz   = np.clip(cossza, 0.01, 1.0)
    ea   = rh / 100.0 * svp(t)
    tw   = td.copy();  conv = np.zeros_like(t, dtype=bool)
    for _ in range(MAX_ITER):
        tr  = 0.5 * (tw + t)
        h   = h_cyl(tr, p, spd)
        Sc  = visc(tr) / (p * 100 / (R_AIR * tr) * diff(tr, p * 100))
        Fa  = (STEFANB * EMIS_WICK * (0.5 * (emis_atm(t, rh) * t**4
               + EMIS_SFC * t**4) - tw**4)
               + (1 - ALB_WICK) * sol
               * ((1 - fdir) * (1 + 0.25 * D_WICK / L_WICK)
                  + fdir * (np.tan(np.arccos(cz)) / np.pi
                            + 0.25 * D_WICK / L_WICK) + ALB_SFC))
        twn = t - evap(tr) / RATIO * (svp(tw) - ea) / (p - svp(tw)) \
              * (Pr / Sc)**0.56 + Fa / h
        conv |= np.abs(twn - tw) < CONVERGENCE
        tw   = np.where(conv, tw, 0.9 * tw + 0.1 * twn)
        if conv.all(): break
    return np.where(conv, twn, np.nan)

# ============================================================================
# Solar geometry
# ============================================================================
def cossza_grid(lat_g, lon_g, times):
    ti   = pd.DatetimeIndex(times)
    doy  = ti.dayofyear.to_numpy(DTYPE)
    hour = (ti.hour + ti.minute / 60.0).to_numpy(DTYPE) - 0.5
    decl = np.radians(23.45 * np.sin(np.radians(360 / 365 * (doy - 81)))).astype(DTYPE)
    latr = np.radians(lat_g).astype(DTYPE)
    ha   = np.radians(15 * (hour[:, None, None] + lon_g[None] / 15 - 12))
    cz   = (np.sin(latr)[None] * np.sin(decl)[:, None, None]
            + np.cos(latr)[None] * np.cos(decl)[:, None, None] * np.cos(ha))
    return np.clip(cz, 0, 1).astype(DTYPE)

# ============================================================================
# ERA5 loader
# ============================================================================
def load_var(name):
    fp = DATA_DIR / f"{VAR_FILES[name]}_{YEAR_START}_{YEAR_END}.nc"
    ds = xr.open_dataset(fp)
    vn = name if name in ds else next(v for v in ds if v not in ("expver","number"))
    da = ds[vn]
    if "time" in da.dims and "valid_time" not in da.dims:
        da = da.rename({"time": "valid_time"})
    for ex in ("expver", "number"):
        if ex in da.dims:
            da = da.squeeze(ex, drop=True)
    return da.astype(DTYPE).load()

# ============================================================================
# Aggregation: hourly -> daily mean per bracket -> monthly mean + WBGT5x
# ============================================================================
def aggregate(wbgt_h, times):
    """
    Returns four monthly (n_months, ny, nx) arrays:
        monthly_day, monthly_night, wbgt5x_day, wbgt5x_night
    and the corresponding month-start DatetimeIndex.

    Day/night split: local Malawi time (UTC+2).
    Daily series: mean of selected hours per day (not daily max) so that
    the monthly mean matches the CMIP6 bridge script output exactly.
    WBGT5x: monthly max of within-month 5-day rolling mean of daily means.
    """
    local_h  = (pd.DatetimeIndex(times) + pd.Timedelta(hours=UTC_OFFSET)).hour.to_numpy()
    day_mask = (local_h >= DAY_START) & (local_h <= DAY_END)

    dates    = pd.DatetimeIndex(times).normalize()
    periods  = dates.to_period("M")
    u_days   = pd.DatetimeIndex(sorted(set(dates)))
    u_months = periods.unique().sort_values()
    n_days   = len(u_days);  n_mon = len(u_months)
    _, ny, nx = wbgt_h.shape

    day_lbl  = np.array([u_days.get_loc(d) for d in dates], dtype=np.int32)

    def daily_mean(mask):
        s = np.zeros((n_days, ny, nx), dtype=np.float64)
        c = np.zeros(n_days, dtype=np.int32)
        np.add.at(s, day_lbl[mask], wbgt_h[mask].astype(np.float64))
        np.add.at(c, day_lbl[mask], 1)
        c = np.where(c > 0, c, 1)[:, None, None]
        return (s / c).astype(np.float32)

    def monthly_mean(daily):
        out = np.full((n_mon, ny, nx), np.nan, np.float32)
        day_periods = u_days.to_period("M")
        for k, m in enumerate(u_months):
            out[k] = np.nanmean(daily[day_periods == m], axis=0)
        return out

    def wbgt5x(daily):
        out = np.full((n_mon, ny, nx), np.nan, np.float32)
        day_periods = u_days.to_period("M")
        for k, m in enumerate(u_months):
            chunk = daily[day_periods == m].astype(np.float64)  # (n_m, ny, nx)
            n_m   = len(chunk)
            if n_m < WBGT5X_WINDOW:
                continue
            nan_m  = np.isnan(chunk)
            filled = np.where(nan_m, 0.0, chunk)
            cs     = np.cumsum(filled, axis=0)
            cv     = np.cumsum(~nan_m, axis=0, dtype=np.float64)
            rs     = cs.copy();  rv = cv.copy()
            rs[WBGT5X_WINDOW:] -= cs[:-WBGT5X_WINDOW]
            rv[WBGT5X_WINDOW:] -= cv[:-WBGT5X_WINDOW]
            rm     = np.where(rv == WBGT5X_WINDOW, rs / WBGT5X_WINDOW, np.nan)
            mx     = np.nanmax(rm, axis=0).astype(np.float32)
            mx[np.all(np.isnan(rm), axis=0)] = np.nan
            out[k] = mx
        return out

    d_day   = daily_mean(day_mask)
    d_night = daily_mean(~day_mask)
    midx    = u_months.to_timestamp(how="start")

    return (monthly_mean(d_day), monthly_mean(d_night),
            wbgt5x(d_day),       wbgt5x(d_night),       midx)

# ============================================================================
# Main
# ============================================================================
def main():
    print("Loading ERA5...")
    t2m = load_var("t2m");  d2m = load_var("d2m")
    u10 = load_var("u10");  v10 = load_var("v10")
    ssrd = load_var("ssrd"); fdir_r = load_var("fdir"); sp = load_var("sp")

    times = t2m.valid_time.values
    lats  = t2m.latitude.values;  lons = t2m.longitude.values
    lon_g, lat_g = np.meshgrid(lons, lats)
    nt, ny, nx   = t2m.shape
    print(f"Grid {ny}x{nx}, {nt} hourly steps")

    t_air  = t2m.values;  td = d2m.values
    speed  = np.sqrt(u10.values**2 + v10.values**2)
    p_hpa  = sp.values / 100.0
    solar  = np.clip(ssrd.values / 3600, 0, None)
    fdir_w = np.clip(fdir_r.values / 3600, 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        fdir_f = np.where(solar > 1, fdir_w / np.where(solar > 1, solar, 1), 0.0)
    fdir_f = np.clip(fdir_f, 0, 1)
    rh     = np.clip(100 * svp(d2m.values) / svp(t2m.values), 1, 100)

    print("Computing solar zenith angles...")
    cz = np.maximum(cossza_grid(lat_g, lon_g, times), 0.05)

    wbgt_h = np.full((nt, ny, nx), np.nan, DTYPE)
    tg_h   = np.full((nt, ny, nx), np.nan, DTYPE)
    tw_h   = np.full((nt, ny, nx), np.nan, DTYPE)

    n_blk = int(np.ceil(nt / TIME_CHUNK))
    print(f"Solving in {n_blk} block(s)...")
    for b in range(n_blk):
        s  = slice(b * TIME_CHUNK, min((b + 1) * TIME_CHUNK, nt))
        tg = globe_temp(t_air[s], rh[s], p_hpa[s], solar[s], fdir_f[s], cz[s], speed[s])
        tw = wet_bulb(t_air[s], td[s], rh[s], p_hpa[s], speed[s], solar[s], fdir_f[s], cz[s])
        wbgt_h[s] = 0.7*(tw-273.15) + 0.2*(tg-273.15) + 0.1*(t_air[s]-273.15)
        tg_h[s]   = tg - 273.15
        tw_h[s]   = tw - 273.15
        print(f"  block {b+1}/{n_blk}")

    n_nan = int(np.isnan(wbgt_h).sum())
    print(f"Converged. NaN cells: {n_nan} ({100*n_nan/wbgt_h.size:.3f}%)")

    # ---- Hourly output
    coords = {"time": times, "lat": lats, "lon": lons}
    enc    = lambda v: {v: {"zlib": True, "complevel": 4, "dtype": "float32"}}
    ds_h   = xr.Dataset({
        "wbgt": xr.DataArray(wbgt_h, coords=coords, dims=["time","lat","lon"],
                             attrs={"units":"degC","method":"Liljegren"}),
        "tg":   xr.DataArray(tg_h,   coords=coords, dims=["time","lat","lon"],
                             attrs={"units":"degC"}),
        "twb":  xr.DataArray(tw_h,   coords=coords, dims=["time","lat","lon"],
                             attrs={"units":"degC"}),
    })
    f_h = OUT_DIR / f"wbgt_hourly_ERA5_malawi_{YEAR_START}_{YEAR_END}.nc"
    ds_h.to_netcdf(f_h, encoding={v: {"zlib":True,"complevel":4,"dtype":"float32"}
                                   for v in ds_h.data_vars})
    print(f"Saved hourly       : {f_h.name}")

    # ---- Daily max
    dm = ds_h["wbgt"].resample(time="1D").max()
    dm.name = "wbgt_daily_max";  dm.attrs = {"units":"degC","method":"Liljegren"}
    f_dm = OUT_DIR / f"wbgt_daily_max_ERA5_malawi_{YEAR_START}_{YEAR_END}.nc"
    dm.to_netcdf(f_dm, encoding={"wbgt_daily_max":{"zlib":True,"complevel":4,"dtype":"float32"}})
    print(f"Saved daily max    : {f_dm.name}")

    # ---- Monthly mean + WBGT5x
    print("Aggregating to daily then monthly...")
    m_day, m_night, x_day, x_night, midx = aggregate(wbgt_h, times)

    mc = {"time": midx.values, "lat": lats, "lon": lons}
    attrs_day   = {"units": "degC", "bracket": "day",
                   "note": "monthly mean of daily-mean daytime WBGT"}
    attrs_night = {"units": "degC", "bracket": "night",
                   "note": "monthly mean of daily-mean night-time WBGT"}

    ds_m = xr.Dataset({
        "wbgt_day":   xr.DataArray(m_day,   coords=mc, dims=["time","lat","lon"], attrs=attrs_day),
        "wbgt_night": xr.DataArray(m_night, coords=mc, dims=["time","lat","lon"], attrs=attrs_night),
    })
    f_m = OUT_DIR / "wbgt_monthly_ERA5_historical.nc"
    ds_m.to_netcdf(f_m, encoding={v:{"zlib":True,"complevel":4,"dtype":"float32"}
                                   for v in ds_m.data_vars})
    print(f"Saved monthly      : {f_m.name}")

    ds_5x = xr.Dataset({
        "wbgt5x_day":   xr.DataArray(x_day,   coords=mc, dims=["time","lat","lon"],
                                     attrs={"units":"degC","bracket":"day",
                                            "note":f"monthly max of {WBGT5X_WINDOW}-day rolling mean"}),
        "wbgt5x_night": xr.DataArray(x_night, coords=mc, dims=["time","lat","lon"],
                                     attrs={"units":"degC","bracket":"night",
                                            "note":f"monthly max of {WBGT5X_WINDOW}-day rolling mean"}),
    })
    f_5x = OUT_DIR / "wbgt_wbgt5x_ERA5_historical.nc"
    ds_5x.to_netcdf(f_5x, encoding={v:{"zlib":True,"complevel":4,"dtype":"float32"}
                                     for v in ds_5x.data_vars})
    print(f"Saved WBGT5x       : {f_5x.name}")

    print(f"\nWBGT  min={np.nanmin(wbgt_h):.1f}  "
          f"mean={np.nanmean(wbgt_h):.1f}  max={np.nanmax(wbgt_h):.1f} °C")
    print(f"WBGT5x day   mean={np.nanmean(x_day):.1f} °C")
    print(f"WBGT5x night mean={np.nanmean(x_night):.1f} °C")
    ds_h.close()

if __name__ == "__main__":
    print("=" * 60)
    print("ERA5 WBGT — Liljegren vectorised")
    print("=" * 60)
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} not found")
    else:
        main()
