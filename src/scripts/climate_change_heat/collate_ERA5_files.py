"""Combine per-year ERA5 files into single multi-year files, one per variable."""

from pathlib import Path

import xarray as xr

BASE_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/ERA5/Periindustrial")
OUT_DIR = BASE_DIR / "Combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIABLES = [
    #"2m_temperature",
    #"2m_dewpoint_temperature",
    #"10m_u_component_of_wind",
    #"10m_v_component_of_wind",
    #"surface_solar_radiation_downwards",
    #"total_sky_direct_solar_radiation_at_surface",
    #"surface_pressure",
    "total_precipitation",
]

YEAR_START, YEAR_END = 1940, 1948

for variable in VARIABLES:
    var_dir = BASE_DIR / variable
    if not var_dir.exists():
        print(f"Skipping {variable} — directory not found")
        continue

    year_files = sorted(var_dir.glob(f"{YEAR_START}/{variable}_*.nc")) if False else []
    # files live at BASE_DIR/{variable}/{year}/{variable}_{year}.nc
    year_files = sorted(var_dir.glob(f"*/{variable}_*.nc"))

    if not year_files:
        print(f"Skipping {variable} — no year files found")
        continue

    print(f"Combining {len(year_files)} files for {variable}...")
    ds = xr.open_mfdataset(year_files, combine="by_coords", chunks={"time": 500})

    out_file = OUT_DIR / f"{variable}_{YEAR_START}_{YEAR_END}.nc"
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_file, format="NETCDF4", encoding=encoding)
    ds.close()
    print(f"  Saved: {out_file}")
