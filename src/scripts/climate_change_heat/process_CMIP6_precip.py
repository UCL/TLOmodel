"""
calculating_precip_monthly_CMIP6.py

Aggregate daily NEX-GDDP-CMIP6 pr into monthly summaries, one file per model,
on the same grid + monthly time axis as wbgt_daynight_{model}.nc.

Outputs (per model):
    precip_monthly_{model}_{scenario}.nc
        precip_month     mm / month
        precip_5day      mm  (monthly max of 5-day rolling sum of daily mm)

Runs after collate_NASA_nc_files.py and before associating_CMIP6_facilities.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# CONFIG — mirror associating_CMIP6_facilities.py
# ---------------------------------------------------------------------------
COMBINED_DIR = Path(
    "/Users/rachelmurray-watson/Documents/Heat_data/NASA_GDDP-CMIP6/Combined"
)
# Where the WBGT day/night files already live, per model, in
# {WBGT_SPLIT_DIR}/{model}/{scenario}/wbgt_daynight_*.nc — precip goes next
# to them so the projection facility script finds it with the same glob.
WBGT_SPLIT_DIR = Path(
    "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/NASA_GDDP_CMIP6_Split"
)
SCENARIO = "ssp585"

# ACTIVE_MODELS should match accessing_NASA_data.py exactly. Do not infer
# from folder listings — a folder can exist with no combined pr file.
ACTIVE_MODELS = [
    "ACCESS-ESM1-5", "CMCC-ESM2", "MPI-ESM1-2-HR", "GISS-E2-1-G",
    "CMCC-CM2-SR5", "ACCESS-CM2", "MIROC6", "CanESM5", "MIROC-ES2L",
]

TIME_COORD, LAT_COORD, LON_COORD = "time", "lat", "lon"


# ---------------------------------------------------------------------------
# Same helper as associating_CMIP6_facilities.py — handles cftime NoLeap
# ---------------------------------------------------------------------------
def get_time_index(time_values):
    """YYYY-MM string per timestep for cftime OR datetime64 time coords."""
    keys = []
    for t in time_values:
        if hasattr(t, "year") and hasattr(t, "month"):
            keys.append(f"{t.year:04d}-{t.month:02d}")
        else:
            ts = pd.Timestamp(t)
            keys.append(f"{ts.year:04d}-{ts.month:02d}")
    return pd.Index(keys, name="date")


def assert_daily(time_values, nc_path):
    if len(time_values) < 2:
        raise ValueError(f"{Path(nc_path).name}: fewer than 2 timesteps")
    diffs = []
    for i in range(min(len(time_values) - 1, 366)):
        d = time_values[i + 1] - time_values[i]
        diffs.append(d.days if hasattr(d, "days") else d / np.timedelta64(1, "D"))
    med = float(np.median(diffs))
    if med > 5:
        raise ValueError(
            f"{Path(nc_path).name}: median spacing {med:.0f} days — expected daily pr"
        )


# ---------------------------------------------------------------------------
# Per-model loop
# ---------------------------------------------------------------------------
for model in ACTIVE_MODELS:
    print(f"\n--- {model} ---")

    # Input: one combined daily pr file for this model, this scenario
    pr_files = sorted(COMBINED_DIR.glob(
        f"pr_day_{model}_{SCENARIO}_malawi_*.nc"
    ))
    if not pr_files:
        print(f"  ⚠ no combined pr file for {model} — run collate first; skipping")
        continue
    if len(pr_files) > 1:
        raise RuntimeError(f"{model}: multiple combined pr files {pr_files}")
    pr_path = pr_files[0]

    # Output: sits alongside this model's WBGT file so downstream finds it
    out_dir = WBGT_SPLIT_DIR / model / SCENARIO
    if not out_dir.exists():
        print(f"  ⚠ {out_dir} does not exist — WBGT for {model} not yet computed; skipping")
        continue
    out_path = out_dir / f"precip_monthly_{model}_{SCENARIO}.nc"

    # Open with cftime for NoLeap calendar
    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_dataset(pr_path, decode_times=time_coder)
    except AttributeError:
        ds = xr.open_dataset(pr_path, use_cftime=True)

    if "pr" not in ds:
        raise KeyError(f"{pr_path.name}: 'pr' not in {list(ds.data_vars)}")

    assert_daily(ds[TIME_COORD].values, pr_path)

    # Units: kg m-2 s-1 -> mm/day
    pr_mm = ds["pr"] * 86400.0
    pr_mm.attrs["units"] = "mm day-1"

    # 5-day rolling SUM of daily mm (aligned right; NaN for the first 4 days)
    roll5 = pr_mm.rolling({TIME_COORD: 5}, min_periods=5).sum()

    # Group both by YYYY-MM key derived from the (cftime) time axis
    month_key = xr.DataArray(
        get_time_index(ds[TIME_COORD].values).values,
        dims=TIME_COORD, name="month",
    )
    pr_month = pr_mm.groupby(month_key).sum()                # mm / month
    pr_5day_monthly_max = roll5.groupby(month_key).max()     # mm

    pr_month.name = "precip_month"
    pr_month.attrs["units"] = "mm"
    pr_5day_monthly_max.name = "precip_5day"
    pr_5day_monthly_max.attrs["units"] = "mm"
    pr_5day_monthly_max.attrs["long_name"] = (
        "monthly maximum of 5-day rolling total precipitation"
    )

    # Build the monthly time axis the projection facility script expects:
    # month-end timestamps, one per unique YYYY-MM
    month_labels = pr_month["month"].values
    monthly_dates = (pd.PeriodIndex(month_labels, freq="M")
                     .to_timestamp(how="end").normalize())

    out = xr.Dataset(
        {"precip_month": (("time", LAT_COORD, LON_COORD), pr_month.values),
         "precip_5day":  (("time", LAT_COORD, LON_COORD), pr_5day_monthly_max.values)},
        coords={"time": monthly_dates,
                LAT_COORD: ds[LAT_COORD].values,
                LON_COORD: ds[LON_COORD].values},
    )
    out.attrs["source_file"] = pr_path.name
    out.attrs["source_model"] = model
    out.attrs["scenario"] = SCENARIO

    ds.close()
    out.to_netcdf(out_path)
    print(f"  {pr_path.name} -> {out_path.name}  "
          f"({out.sizes['time']} months, "
          f"{out.sizes[LAT_COORD]}x{out.sizes[LON_COORD]} grid)")

print("\nDone. Next: re-run associating_CMIP6_facilities.py (with the precip block added).")
