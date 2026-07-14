"""
build_era5_forcing_climatology.py

Companion to the ERA5 WBGT script. Reads the SAME raw ERA5 inputs (surface
pressure, downward shortwave, direct shortwave) and writes a small per-cell
climatology that the CMIP6 projection WBGT calculator can use INSTEAD of its
900 hPa / fdir=0.7 constants:

  surface_pressure_hpa   (lat, lon)          per-cell mean surface pressure
  direct_fraction        (month, lat, lon)   per-cell mean DAYTIME direct-beam
                                             fraction, by calendar month

No WBGT physics here — just the two fields NEX-GDDP-CMIP6 does not provide.
Written on the native ERA5 grid AND regridded onto the CMIP6 grid (the version
the calculator reads), so cell assignment matches the projection.

Output
------
  era5_forcing_climatology.nc
  era5_forcing_climatology_cmip6grid.nc   <- point ERA5_CLIMATOLOGY_FILE here
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# CONFIG (paths mirror the ERA5 WBGT script)
# ---------------------------------------------------------------------------
DATA_DIR  = Path("/Users/rachelmurray-watson/Documents/Heat_data/ERA5/Combined")
OUT_DIR   = Path("/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/ERA5")
CMIP6_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/NASA_GDDP_CMIP6_Split")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_START = 2010
YEAR_END   = 2024
DAY_SOLAR_MIN = 50.0    # W/m2; direct fraction only averaged over sunlit hours

VAR_FILES = {
    "ssrd": "surface_solar_radiation_downwards",
    "fdir": "total_sky_direct_solar_radiation_at_surface",
    "sp":   "surface_pressure",
}


# ---------------------------------------------------------------------------
# Core computation (pure arrays -> easy to test)
# ---------------------------------------------------------------------------
def compute_climatology(sp_hpa, solar_w, fdir_frac, months,
                        day_solar_min=DAY_SOLAR_MIN):
    """sp_hpa, solar_w, fdir_frac: (nt, ny, nx). months: (nt,) 1-12.
    Returns (sp_clim[ny,nx], fdir_monthly[12,ny,nx])."""
    sp_clim = np.nanmean(sp_hpa, axis=0).astype(np.float32)

    day_hr = solar_w > day_solar_min                       # (nt,ny,nx)
    ny, nx = sp_hpa.shape[1:]
    fdir_monthly = np.full((12, ny, nx), np.nan, np.float32)
    for m in range(1, 13):
        sel = (months == m)[:, None, None] & day_hr        # (nt,ny,nx)
        den = sel.sum(axis=0)
        num = np.where(sel, fdir_frac, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore"):
            fdir_monthly[m - 1] = np.where(den > 0, num / den, np.nan)
    return sp_clim, fdir_monthly


# ---------------------------------------------------------------------------
# ERA5 loading (mirrors the ERA5 WBGT script's loader)
# ---------------------------------------------------------------------------
def load_var(name):
    fp = DATA_DIR / f"{VAR_FILES[name]}_{YEAR_START}_{YEAR_END}.nc"
    ds = xr.open_dataset(fp)
    vn = name if name in ds else next(v for v in ds if v not in ("expver", "number"))
    da = ds[vn]
    if "time" in da.dims and "valid_time" not in da.dims:
        da = da.rename({"time": "valid_time"})
    for ex in ("expver", "number"):
        if ex in da.dims:
            da = da.squeeze(ex, drop=True)
    return da.astype(np.float32).load()


# ---------------------------------------------------------------------------
# CMIP6 grid + regrid (mirrors the ERA5 WBGT script)
# ---------------------------------------------------------------------------
def find_cmip6_reference():
    for pat in ("wbgt_daynight_*.nc", "wbgt_monthly_*.nc", "*.nc"):
        cands = sorted(CMIP6_DIR.rglob(pat))
        if cands:
            return cands[0]
    return None


def read_cmip6_grid(ref_path):
    ds = xr.open_dataset(ref_path, decode_times=False)
    lat_name = "lat" if "lat" in ds else "latitude"
    lon_name = "lon" if "lon" in ds else "longitude"
    lat = np.sort(np.asarray(ds[lat_name].values, dtype=np.float64))
    lon = np.sort(np.asarray(ds[lon_name].values, dtype=np.float64))
    ds.close()
    return lat, lon


def regrid_dataset(ds, tgt_lat, tgt_lon):
    ds = ds.sortby("lat").sortby("lon")
    out = {}
    for v in ds.data_vars:
        lin = ds[v].interp(lat=tgt_lat, lon=tgt_lon, method="linear")
        near = ds[v].interp(lat=tgt_lat, lon=tgt_lon, method="nearest")
        da = lin.fillna(near)
        da.attrs = dict(ds[v].attrs)
        da.attrs["regrid"] = "bilinear->CMIP6 grid (nearest edge-fill)"
        out[v] = da
    return xr.Dataset(out)


def save(ds, path, label):
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    print(f"Saved {label:26s}: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading ERA5 pressure / radiation...")
    sp   = load_var("sp")
    ssrd = load_var("ssrd")
    fdir = load_var("fdir")

    times = sp.valid_time.values
    lats  = sp.latitude.values
    lons  = sp.longitude.values

    sp_hpa    = sp.values / 100.0
    solar_w   = np.clip(ssrd.values / 3600.0, 0, None)
    fdir_w    = np.clip(fdir.values / 3600.0, 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        fdir_frac = np.where(solar_w > 1, fdir_w / np.where(solar_w > 1, solar_w, 1), 0.0)
    fdir_frac = np.clip(fdir_frac, 0, 1)
    months = pd.DatetimeIndex(times).month.to_numpy()

    print("Computing climatology...")
    sp_clim, fdir_monthly = compute_climatology(sp_hpa, solar_w, fdir_frac, months)
    print(f"  mean surface pressure: {np.nanmean(sp_clim):.1f} hPa "
          f"(range {np.nanmin(sp_clim):.0f}-{np.nanmax(sp_clim):.0f})")
    print(f"  annual-mean daytime direct fraction: {np.nanmean(fdir_monthly):.2f} "
          f"(monthly range {np.nanmin(np.nanmean(fdir_monthly,axis=(1,2))):.2f}"
          f"-{np.nanmax(np.nanmean(fdir_monthly,axis=(1,2))):.2f})")

    clim = xr.Dataset(
        {"surface_pressure_hpa": (("lat", "lon"), sp_clim),
         "direct_fraction":      (("month", "lat", "lon"), fdir_monthly)},
        coords={"month": np.arange(1, 13), "lat": lats, "lon": lons},
        attrs={"description": "ERA5 forcing climatology for CMIP6 WBGT projection",
               "years": f"{YEAR_START}-{YEAR_END}",
               "day_solar_min_wm2": DAY_SOLAR_MIN})
    save(clim, OUT_DIR / "era5_forcing_climatology.nc", "forcing clim (native)")

    ref = find_cmip6_reference()
    if ref is None:
        print(f"⚠ no CMIP6 reference file under {CMIP6_DIR} — cannot regrid. "
              "The calculator needs the *_cmip6grid.nc version.")
    else:
        tgt_lat, tgt_lon = read_cmip6_grid(ref)
        print(f"Regridding onto CMIP6 grid from {ref.name} "
              f"({len(tgt_lat)} lat x {len(tgt_lon)} lon)")
        save(regrid_dataset(clim, tgt_lat, tgt_lon),
             OUT_DIR / "era5_forcing_climatology_cmip6grid.nc", "forcing clim (CMIP6 grid)")

    print("Done.")


if __name__ == "__main__":
    main()
