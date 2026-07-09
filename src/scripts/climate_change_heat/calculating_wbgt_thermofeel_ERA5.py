"""
Calculate WBGT from ERA5 hourly reanalysis data for Malawi — vectorised.
WBGT = 0.7 * Twb + 0.2 * Tg + 0.1 * Ta  (Liljegren method)


    wbgt_day / wbgt_night     monthly MEAN of daily max / min   (feeds the
                              monthly-mean facility panel)
    wbgtx_day / wbgtx_night   monthly MAX  of daily max / min   (WBGTx, mirrors
                              WBGT_extreme_indices_projections.py)
    wbgt5x_*                  monthly max of within-month 5-day rolling mean of
                              daily max / min                   (WBGT5x)

GRID CO-REGISTRATION
--------------------
ERA5 and NEX-GDDP-CMIP6 are both 0.25 deg but on grids staggered by ~0.125 deg,
so a facility can fall in different cells between branches. With REGRID_TO_CMIP6
set, the ERA5 monthly / index fields are additionally interpolated onto the
CMIP6 grid (read from an actual CMIP6 file) and written as *_cmip6grid.nc, so
facility -> cell assignment is identical in both branches. The native-grid
files are still written unchanged.

Outputs
-------
wbgt_hourly_ERA5_malawi_<Y0>_<Y1>.nc        hourly wbgt, tg, twb
wbgt_daily_max_ERA5_malawi_<Y0>_<Y1>.nc     local-day max wbgt   (peak-of-day)
wbgt_daily_min_ERA5_malawi_<Y0>_<Y1>.nc     local-day min wbgt   (trough-of-day)
wbgt_monthly_ERA5_historical.nc             monthly-mean wbgt_day/night
wbgt_extreme_indices_ERA5_historical.nc     wbgtx_* and wbgt5x_* (day + night)
  ...and, if REGRID_TO_CMIP6, *_cmip6grid.nc versions of the last two.
"""

import warnings
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

YEAR_START       = 2010
YEAR_END         = 2024
TIME_CHUNK       = 2000
DTYPE            = np.float32
UTC_OFFSET       = 2          # Malawi UTC+2 (used to define the LOCAL calendar day)
WBGT5X_WINDOW    = 5          # days

# --- Grid co-registration onto the CMIP6 grid ------------------------------
REGRID_TO_CMIP6  = True       # also write *_cmip6grid.nc versions
# None -> auto-discover a CMIP6 WBGT file under CMIP6_DIR to read the target grid.
# Point this at a specific file if auto-discovery grabs the wrong one.
CMIP6_REF_FILE   = None

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
# WBGT solvers  (unchanged Liljegren physics — identical to the CMIP6 branch)
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
# Monthly reduction of a daily bracket series
#   daily : (n_days, ny, nx) either the daily-MAX (day) or daily-MIN (night)
#   Returns monthly mean, monthly max (WBGTx), monthly WBGT5x, month index.
# ============================================================================
def monthly_reduce(daily, u_days):
    day_periods = u_days.to_period("M")
    u_months    = day_periods.unique().sort_values()
    n_mon       = len(u_months)
    _, ny, nx   = daily.shape

    mean_o = np.full((n_mon, ny, nx), np.nan, np.float32)  # monthly mean
    max_o  = np.full((n_mon, ny, nx), np.nan, np.float32)  # WBGTx
    x5_o   = np.full((n_mon, ny, nx), np.nan, np.float32)  # WBGT5x

    for k, m in enumerate(u_months):
        idx   = np.where(day_periods == m)[0]
        chunk = daily[idx].astype(np.float64)              # (n_m, ny, nx)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_o[k] = np.nanmean(chunk, axis=0)
            max_o[k]  = np.nanmax(chunk, axis=0)

        # WBGT5x: monthly max of the within-month 5-day rolling mean. Window
        # resets at the month boundary (chunk is this month's days only); a
        # window is only valid when all WBGT5X_WINDOW days are non-NaN.
        n_m = len(chunk)
        if n_m >= WBGT5X_WINDOW:
            nan_m  = np.isnan(chunk)
            filled = np.where(nan_m, 0.0, chunk)
            cs     = np.cumsum(filled, axis=0)
            cv     = np.cumsum(~nan_m, axis=0, dtype=np.float64)
            rs     = cs.copy();  rv = cv.copy()
            rs[WBGT5X_WINDOW:] -= cs[:-WBGT5X_WINDOW]
            rv[WBGT5X_WINDOW:] -= cv[:-WBGT5X_WINDOW]
            rm     = np.where(rv == WBGT5X_WINDOW, rs / WBGT5X_WINDOW, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mx = np.nanmax(rm, axis=0).astype(np.float32)
            mx[np.all(np.isnan(rm), axis=0)] = np.nan
            x5_o[k] = mx

    midx = u_months.to_timestamp(how="start")
    return mean_o, max_o, x5_o, midx

# ============================================================================
# CMIP6 grid helpers (for co-registration)
# ============================================================================
def find_cmip6_reference():
    """Locate a CMIP6 WBGT file to read the target grid from."""
    if CMIP6_REF_FILE is not None:
        p = Path(CMIP6_REF_FILE)
        return p if p.exists() else None
    for pat in ("wbgt_daynight_*.nc", "wbgt_monthly_*.nc", "*.nc"):
        cands = sorted(CMIP6_DIR.rglob(pat))
        if cands:
            return cands[0]
    return None

def read_cmip6_grid(ref_path):
    """Return (lat, lon) 1-D arrays of the CMIP6 grid, ascending."""
    ds = xr.open_dataset(ref_path, decode_times=False)
    lat_name = "lat" if "lat" in ds else ("latitude" if "latitude" in ds else None)
    lon_name = "lon" if "lon" in ds else ("longitude" if "longitude" in ds else None)
    lat = np.sort(np.asarray(ds[lat_name].values, dtype=np.float64))
    lon = np.sort(np.asarray(ds[lon_name].values, dtype=np.float64))
    ds.close()
    return lat, lon

def regrid_dataset(ds, tgt_lat, tgt_lon):
    """Bilinear interp of every var onto (tgt_lat, tgt_lon), with a nearest-
    neighbour fill for any edge cell that linear interp leaves as NaN (CMIP6
    cell centres just outside the ERA5 span). ERA5 lat/lon sorted ascending
    first so interp is well-defined."""
    ds = ds.sortby("lat").sortby("lon")
    out = {}
    for v in ds.data_vars:
        lin  = ds[v].interp(lat=tgt_lat, lon=tgt_lon, method="linear")
        near = ds[v].interp(lat=tgt_lat, lon=tgt_lon, method="nearest")
        da   = lin.fillna(near)
        da.attrs = dict(ds[v].attrs)
        da.attrs["regrid"] = "bilinear->CMIP6 grid (nearest edge-fill)"
        out[v] = da
    return xr.Dataset(out)

# ============================================================================
# I/O helper
# ============================================================================
def save(ds, path, label):
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    print(f"Saved {label:22s}: {path.name}")

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

    # ---- Hourly output (UTC times, unchanged) -----------------------------
    coords = {"time": times, "lat": lats, "lon": lons}
    ds_h = xr.Dataset({
        "wbgt": xr.DataArray(wbgt_h, coords=coords, dims=["time","lat","lon"],
                             attrs={"units":"degC","method":"Liljegren"}),
        "tg":   xr.DataArray(tg_h,   coords=coords, dims=["time","lat","lon"],
                             attrs={"units":"degC"}),
        "twb":  xr.DataArray(tw_h,   coords=coords, dims=["time","lat","lon"],
                             attrs={"units":"degC"}),
    })
    f_h = OUT_DIR / f"wbgt_hourly_ERA5_malawi_{YEAR_START}_{YEAR_END}.nc"
    save(ds_h, f_h, "hourly")

    # ---- Daily peak / trough on the LOCAL calendar day --------------------
    # Shift the time coordinate to local time so resample("1D") bins by local
    # day; the afternoon peak and pre-dawn minimum both fall safely inside a
    # local-day bin. daily MAX -> "day" bracket (matches CMIP6 tasmax),
    # daily MIN -> "night" bracket (matches CMIP6 tasmin).
    print("Building daily peak / trough (local day)...")
    local_time = pd.DatetimeIndex(times) + pd.Timedelta(hours=UTC_OFFSET)
    wbgt_local = ds_h["wbgt"].assign_coords(time=local_time)
    daily_max_da = wbgt_local.resample(time="1D").max()   # skipna=True
    daily_min_da = wbgt_local.resample(time="1D").min()

    u_days = pd.DatetimeIndex(daily_max_da.time.values)
    dmax   = daily_max_da.values
    dmin   = daily_min_da.values

    # The local-time shift spills the last UTC_OFFSET hours of the record into
    # a partial local day (and can leave the first local day a few hours short).
    # Drop any local day outside [YEAR_START, YEAR_END] so a 1-2 hour trailing
    # "day" can't create a bogus partial month with an unreliable mean/NaN WBGT5x.
    keep = (u_days.year >= YEAR_START) & (u_days.year <= YEAR_END)
    if not keep.all():
        print(f"  dropping {int((~keep).sum())} partial boundary day(s) outside "
              f"{YEAR_START}-{YEAR_END}")
        u_days = u_days[keep];  dmax = dmax[keep];  dmin = dmin[keep]

    dcoords = {"time": u_days.values, "lat": lats, "lon": lons}
    ds_dmax = xr.Dataset({"wbgt_daily_max": xr.DataArray(
        dmax, coords=dcoords, dims=["time","lat","lon"],
        attrs={"units":"degC","method":"Liljegren","bracket":"day",
               "note":"local-day max hourly WBGT (peak-of-day, matches CMIP6 tasmax bracket)"})})
    ds_dmin = xr.Dataset({"wbgt_daily_min": xr.DataArray(
        dmin, coords=dcoords, dims=["time","lat","lon"],
        attrs={"units":"degC","method":"Liljegren","bracket":"night",
               "note":"local-day min hourly WBGT (trough-of-day, matches CMIP6 tasmin bracket)"})})
    save(ds_dmax, OUT_DIR / f"wbgt_daily_max_ERA5_malawi_{YEAR_START}_{YEAR_END}.nc", "daily max")
    save(ds_dmin, OUT_DIR / f"wbgt_daily_min_ERA5_malawi_{YEAR_START}_{YEAR_END}.nc", "daily min")

    # ---- Monthly reductions (comparable to CMIP6 brackets) ----------------
    print("Reducing to monthly mean / WBGTx / WBGT5x...")
    m_day,  xx_day,  x5_day,  midx = monthly_reduce(dmax, u_days)
    m_night, xx_night, x5_night, _ = monthly_reduce(dmin, u_days)

    mc = {"time": midx.values, "lat": lats, "lon": lons}

    # monthly-mean file — feeds associating_grid_squares_weather.py (var names
    # wbgt_day / wbgt_night preserved so that script needs no change).
    ds_m = xr.Dataset({
        "wbgt_day":   xr.DataArray(m_day,   coords=mc, dims=["time","lat","lon"],
                        attrs={"units":"degC","bracket":"day",
                               "note":"monthly mean of local-day MAX WBGT (peak-of-day; matches CMIP6 tasmax bracket)"}),
        "wbgt_night": xr.DataArray(m_night, coords=mc, dims=["time","lat","lon"],
                        attrs={"units":"degC","bracket":"night",
                               "note":"monthly mean of local-day MIN WBGT (trough-of-day; matches CMIP6 tasmin bracket)"}),
    })
    f_m = OUT_DIR / "wbgt_monthly_ERA5_historical.nc"
    save(ds_m, f_m, "monthly mean")

    # extreme-index file — mirrors WBGT_extreme_indices_projections.py vars
    ds_ix = xr.Dataset({
        "wbgtx_day":    xr.DataArray(xx_day,   coords=mc, dims=["time","lat","lon"],
                          attrs={"units":"degC","bracket":"day",
                                 "note":"monthly max of local-day MAX WBGT (WBGTx)"}),
        "wbgtx_night":  xr.DataArray(xx_night, coords=mc, dims=["time","lat","lon"],
                          attrs={"units":"degC","bracket":"night",
                                 "note":"monthly max of local-day MIN WBGT (WBGTx)"}),
        "wbgt5x_day":   xr.DataArray(x5_day,   coords=mc, dims=["time","lat","lon"],
                          attrs={"units":"degC","bracket":"day",
                                 "note":f"monthly max of {WBGT5X_WINDOW}-day rolling mean of local-day MAX WBGT"}),
        "wbgt5x_night": xr.DataArray(x5_night, coords=mc, dims=["time","lat","lon"],
                          attrs={"units":"degC","bracket":"night",
                                 "note":f"monthly max of {WBGT5X_WINDOW}-day rolling mean of local-day MIN WBGT"}),
    })
    f_ix = OUT_DIR / "wbgt_extreme_indices_ERA5_historical.nc"
    save(ds_ix, f_ix, "extreme indices")

    # ---- Optional: regrid the comparable fields onto the CMIP6 grid -------
    if REGRID_TO_CMIP6:
        ref = find_cmip6_reference()
        if ref is None:
            print("REGRID_TO_CMIP6 set but no CMIP6 reference file found under "
                  f"{CMIP6_DIR} — skipping regrid. (Check the path: the day/night "
                  "calc writes to 'NASA_GDDP_CMIP6_Splitz'; this expects "
                  "'NASA_GDDP_CMIP6_Split'.)")
        else:
            print(f"Regridding onto CMIP6 grid from: {ref.name}")
            tgt_lat, tgt_lon = read_cmip6_grid(ref)
            print(f"  CMIP6 target grid: {len(tgt_lat)} lat x {len(tgt_lon)} lon")
            save(regrid_dataset(ds_m,  tgt_lat, tgt_lon),
                 OUT_DIR / "wbgt_monthly_ERA5_historical_cmip6grid.nc", "monthly (CMIP6 grid)")
            save(regrid_dataset(ds_ix, tgt_lat, tgt_lon),
                 OUT_DIR / "wbgt_extreme_indices_ERA5_historical_cmip6grid.nc", "indices (CMIP6 grid)")
            print("  -> point the historical facility panel at the *_cmip6grid.nc "
                  "files so facility->cell matching is identical to the projection.")

    # ---- Summary ----------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        print(f"\nHourly WBGT   min={np.nanmin(wbgt_h):.1f}  "
              f"mean={np.nanmean(wbgt_h):.1f}  max={np.nanmax(wbgt_h):.1f} degC")
        print(f"day  (daily max):  monthly-mean {np.nanmean(m_day):.1f}   "
              f"WBGTx {np.nanmean(xx_day):.1f}   WBGT5x {np.nanmean(x5_day):.1f} degC")
        print(f"night(daily min):  monthly-mean {np.nanmean(m_night):.1f}   "
              f"WBGTx {np.nanmean(xx_night):.1f}   WBGT5x {np.nanmean(x5_night):.1f} degC")
    ds_h.close()

if __name__ == "__main__":
    print("=" * 60)
    print("ERA5 WBGT — Liljegren vectorised (day/night = daily max/min)")
    print("=" * 60)
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} not found")
    else:
        main()
