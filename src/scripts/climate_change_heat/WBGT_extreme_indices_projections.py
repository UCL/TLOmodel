"""
wbgt_extreme_indices.py

Computes ETCCDI-style climate extreme indices for WBGT, in the spirit of the
EEA/Copernicus European Climate Data Explorer indices (TXx, Rx5day, etc.),
applied to the day/night bracketed WBGT output instead of raw temperature.

Indices computed (day and night bracket, separately):

  WBGTx   - Monthly maximum of daily WBGT.
  WBGT5x  - Monthly maximum of the 5-day rolling MEAN WBGT.

Two levels of aggregation are produced:

  1. Country-wide: unweighted spatial mean across in-country grid cells
     (unchanged from the original version of this script).
  2. Facility-level: same indices computed at the single grid cell nearest
     each health facility, via nearest-neighbour (argmin) matching on
     lat/long. This is intentionally the same matching approach used in
     the precip pipeline rather than the searchsorted approach in
     wbgt_facility_panel.py - simpler and robust to irregular grids, at
     the cost of being slower. Fine for this use case.

Outputs
-------
1. wbgt_extreme_indices_{model}_{scenario}.png
   Country-wide two-panel figure: top = WBGTx trend (day vs night), bottom =
   WBGT5x trend (day vs night), monthly resolution with annual max markers.
2. wbgt_extreme_indices_{model}_{scenario}.csv
   Country-wide monthly WBGTx / WBGT5x values (day + night).
3. wbgt_extreme_indices_facility_{model}_{scenario}.csv
   Facility-level monthly WBGTx / WBGT5x values (day + night), long format:
   one row per facility per month.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from netCDF4 import Dataset, num2date
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Config - mirrors conventions used elsewhere in the pipeline
# ---------------------------------------------------------------------------
min_year = 2025
max_year = 2040

WBGT_MODEL = "ACCESS-CM2"
WBGT_SCENARIO = "ssp245"

ROLLING_WINDOW_DAYS = 5  # the "5" in "5-day max" - change here to generalise to Rx3day-style etc.

WBGT_NC_PATH = Path(
    f"/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/NASA_GDDP_CMIP6/"
    f"{WBGT_MODEL}/{WBGT_SCENARIO}/wbgt_daynight_{WBGT_MODEL}_{WBGT_SCENARIO}_malawi_{min_year}_{max_year}.nc"
)

MALAWI_BOUNDARY_PATH = Path(
    "/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/mapping/"
    "ResourceFile_mwi_admbnda_adm2_nso_20181016.shp"
)

OUTPUT_DIR = Path(
    f"/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/NASA_GDDP_CMIP6/"
    f"{WBGT_MODEL}/{WBGT_SCENARIO}/"
)

# --- Facility-level config -------------------------------------------------
# Update these to match the actual Master Health Facility Registry columns
# (same placeholders as in wbgt_facility_panel.py - keep both files in sync).
FACILITIES_CSV_PATH = Path(
    "/Users/rachelmurray-watson/Documents/Heat_data/facilities_with_lat_long_region.csv"
)
FACILITY_ID_COL = "Fname"
LAT_COL = "A109__Latitude"
LON_COL = "A109__Longitude"

COMPUTE_FACILITY_LEVEL = True  # set False to skip facility-level output entirely


def load_daynight_wbgt(nc_path: Path):
    nc = Dataset(nc_path)
    wbgt_day = nc.variables['wbgt_day'][:]
    wbgt_night = nc.variables['wbgt_night'][:]
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]

    time_var = nc.variables['time']
    times = num2date(
        time_var[:], units=time_var.units, calendar=getattr(time_var, 'calendar', 'standard')
    )
    return wbgt_day, wbgt_night, lat, lon, times


def build_country_mask(lat, lon, boundary_gdf):
    """Boolean 2D mask marking grid cell centroids that fall inside Malawi."""
    boundary_union = boundary_gdf.union_all() if hasattr(boundary_gdf, "union_all") else boundary_gdf.unary_union
    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    for i, y in enumerate(lat):
        for j, x in enumerate(lon):
            if boundary_union.contains(Point(x, y)):
                mask[i, j] = True
    return mask


def country_mean_daily_series(wbgt_data, country_mask, times):
    """
    Country-wide area-mean daily WBGT series (unweighted mean across in-country
    grid cells). Returns a pandas Series indexed by date.

    NOTE: this is an unweighted spatial mean of WBGT itself (not an exceedance
    fraction), which is fine here because WBGTx/WBGT5x are about tracking the
    peak of an already-daily value, not about the nonlinear-averaging bias
    that matters when going from sub-daily to daily WBGT. If you want a
    population-weighted version instead of area-weighted, swap the plain
    np.mean() below for a weighted average using the worldpop grid populations
    (as in exposure_map.py's grid_population column) reprojected onto this
    WBGT grid.
    """
    masked = wbgt_data[:, country_mask]  # (time, n_incountry_cells)
    daily_mean = np.mean(masked, axis=1)
    dates = _times_to_datetimeindex(times)
    return pd.Series(daily_mean, index=dates).sort_index()


def _times_to_datetimeindex(times) -> pd.DatetimeIndex:
    """Shared helper so country-level and facility-level series build dates
    the same way (avoids the kind of divergence that's caused problems
    elsewhere in the pipeline)."""
    return pd.to_datetime([f"{t.year}-{t.month:02d}-{t.day:02d}" for t in times])


def nearest_grid_index(lat, lon, facility_lat, facility_lon):
    """
    Nearest-neighbour match of a facility's (lat, lon) to a grid cell index,
    via argmin on squared distance - same approach as the precip pipeline.
    Robust to irregular/non-monotonic grids; not vectorised across facilities,
    so it's O(n_lat + n_lon) per facility. Fine at Malawi facility-count scale.
    """
    lat_idx = int(np.argmin((np.asarray(lat) - facility_lat) ** 2))
    lon_idx = int(np.argmin((np.asarray(lon) - facility_lon) ** 2))
    return lat_idx, lon_idx


def facility_daily_series(wbgt_data, times, lat_idx, lon_idx) -> pd.Series:
    """Daily WBGT series at a single grid cell (nearest to one facility)."""
    series_values = wbgt_data[:, lat_idx, lon_idx]
    dates = _times_to_datetimeindex(times)
    return pd.Series(series_values, index=dates).sort_index()


def monthly_max(daily_series: pd.Series) -> pd.Series:
    """WBGTx: monthly maximum of the daily series."""
    return daily_series.resample("ME").max()


def monthly_max_of_rolling(daily_series: pd.Series, window: int) -> pd.Series:
    """
    WBGT5x: monthly maximum of the `window`-day rolling MEAN of the daily
    series.
    """
    rolling = daily_series.rolling(window, min_periods=window).mean()
    return rolling.resample("ME").max()


def annual_max(monthly_series: pd.Series) -> pd.Series:
    """The hottest month's index value in each calendar year."""
    return monthly_series.resample("YE").max()


def compute_facility_level_indices(wbgt_day, wbgt_night, lat, lon, times, window: int) -> pd.DataFrame:
    """
    Loops over facilities, matches each to its nearest grid cell, and computes
    WBGTx / WBGT5x (day + night) at facility level.

    Returns a long-format DataFrame:
        facility_id, date, wbgtx_day, wbgtx_night, wbgt{window}x_day, wbgt{window}x_night
    """
    facilities = pd.read_csv(FACILITIES_CSV_PATH)
    facilities = facilities.dropna(subset=[LAT_COL, LON_COL])

    rows = []
    n_facilities = len(facilities)
    for count, (_, row) in enumerate(facilities.iterrows(), start=1):
        facility_id = row[FACILITY_ID_COL]
        facility_lat = row[LAT_COL]
        facility_lon = row[LON_COL]

        lat_idx, lon_idx = nearest_grid_index(lat, lon, facility_lat, facility_lon)

        day_daily = facility_daily_series(wbgt_day, times, lat_idx, lon_idx)
        night_daily = facility_daily_series(wbgt_night, times, lat_idx, lon_idx)

        day_x = monthly_max(day_daily)
        night_x = monthly_max(night_daily)
        day_5x = monthly_max_of_rolling(day_daily, window)
        night_5x = monthly_max_of_rolling(night_daily, window)

        facility_df = pd.DataFrame({
            "facility_id": facility_id,
            "date": day_x.index,
            "wbgtx_day": day_x.values,
            "wbgtx_night": night_x.values,
            f"wbgt{window}x_day": day_5x.values,
            f"wbgt{window}x_night": night_5x.values,
        })
        rows.append(facility_df)

        if count % 50 == 0 or count == n_facilities:
            print(f"  facility-level WBGT indices: {count}/{n_facilities} done")

    return pd.concat(rows, ignore_index=True)


def plot_extreme_indices(day_monthly_x, night_monthly_x,
                          day_monthly_5x, night_monthly_5x,
                          output_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # --- Panel 1: WBGTx (monthly max of daily WBGT) ---
    ax = axes[0]
    ax.plot(day_monthly_x.index, day_monthly_x.values, color="#f07167", linewidth=1.3, label="Day WBGTx")
    ax.plot(night_monthly_x.index, night_monthly_x.values, color="#00afb9", linewidth=1.3, label="Night WBGTx")

    day_annual_x = annual_max(day_monthly_x)
    night_annual_x = annual_max(night_monthly_x)
    ax.scatter(day_annual_x.index, day_annual_x.values, color="#f07167", s=40, zorder=5, label="Day annual max")
    ax.scatter(night_annual_x.index, night_annual_x.values, color="#00afb9", s=40, zorder=5, label="Night annual max")

    ax.set_ylabel("WBGTx (°C)\nmonthly max of daily WBGT")
    ax.set_title(f"WBGTx - monthly maximum daily WBGT ({WBGT_MODEL} {WBGT_SCENARIO})")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

    # --- Panel 2: WBGT5x (monthly max of 5-day rolling mean) ---
    ax = axes[1]
    ax.plot(day_monthly_5x.index, day_monthly_5x.values, color="#f07167", linewidth=1.3, label=f"Day WBGT{ROLLING_WINDOW_DAYS}x")
    ax.plot(night_monthly_5x.index, night_monthly_5x.values, color="#00afb9", linewidth=1.3, label=f"Night WBGT{ROLLING_WINDOW_DAYS}x")

    day_annual_5x = annual_max(day_monthly_5x)
    night_annual_5x = annual_max(night_monthly_5x)
    ax.scatter(day_annual_5x.index, day_annual_5x.values, color="#f07167", s=40, zorder=5, label="Day annual max")
    ax.scatter(night_annual_5x.index, night_annual_5x.values, color="#00afb9", s=40, zorder=5, label="Night annual max")

    ax.set_ylabel(f"WBGT{ROLLING_WINDOW_DAYS}x (°C)\nmonthly max of {ROLLING_WINDOW_DAYS}-day rolling mean")
    ax.set_xlabel("Year")
    ax.set_title(
        f"WBGT{ROLLING_WINDOW_DAYS}x - hottest {ROLLING_WINDOW_DAYS}-day stretch each month "
        f"({WBGT_MODEL} {WBGT_SCENARIO})"
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc-path", type=Path, default=WBGT_NC_PATH)
    parser.add_argument("--boundary-path", type=Path, default=MALAWI_BOUNDARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--window", type=int, default=ROLLING_WINDOW_DAYS)
    parser.add_argument("--facilities-csv", type=Path, default=FACILITIES_CSV_PATH)
    parser.add_argument("--skip-facility-level", action="store_true", default=not COMPUTE_FACILITY_LEVEL)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.nc_path}")
    wbgt_day, wbgt_night, lat, lon, times = load_daynight_wbgt(args.nc_path)

    print(f"Loading Malawi boundary: {args.boundary_path}")
    boundary_gdf = gpd.read_file(args.boundary_path)
    country_mask = build_country_mask(lat, lon, boundary_gdf)
    print(f"{country_mask.sum()} / {country_mask.size} grid cells fall inside Malawi")

    # --- Country-wide indices (unchanged) ---
    day_daily = country_mean_daily_series(wbgt_day, country_mask, times)
    night_daily = country_mean_daily_series(wbgt_night, country_mask, times)

    day_monthly_x = monthly_max(day_daily)
    night_monthly_x = monthly_max(night_daily)
    day_monthly_5x = monthly_max_of_rolling(day_daily, args.window)
    night_monthly_5x = monthly_max_of_rolling(night_daily, args.window)

    plot_extreme_indices(
        day_monthly_x, night_monthly_x,
        day_monthly_5x, night_monthly_5x,
        args.output_dir / f"wbgt_extreme_indices_{WBGT_MODEL}_{WBGT_SCENARIO}.png",
    )

    out_df = pd.DataFrame({
        "wbgtx_day": day_monthly_x,
        "wbgtx_night": night_monthly_x,
        f"wbgt{args.window}x_day": day_monthly_5x,
        f"wbgt{args.window}x_night": night_monthly_5x,
    })
    csv_path = args.output_dir / f"wbgt_extreme_indices_{WBGT_MODEL}_{WBGT_SCENARIO}.csv"
    out_df.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    print("\nAnnual WBGTx (day), first and last year in record:")
    print(annual_max(day_monthly_x).iloc[[0, -1]])
    print("\nAnnual WBGTx (night), first and last year in record:")
    print(annual_max(night_monthly_x).iloc[[0, -1]])

    # --- Facility-level indices (new) ---
    if not args.skip_facility_level:
        print(f"\nComputing facility-level WBGT indices from {args.facilities_csv}")
        facility_df = compute_facility_level_indices(
            wbgt_day, wbgt_night, lat, lon, times, args.window
        )
        facility_csv_path = args.output_dir / f"wbgt_extreme_indices_facility_{WBGT_MODEL}_{WBGT_SCENARIO}.csv"
        facility_df.to_csv(facility_csv_path, index=False)
        print(f"Saved: {facility_csv_path}")
        print(f"  {facility_df['facility_id'].nunique()} facilities x "
              f"{facility_df['date'].nunique()} months = {len(facility_df)} rows")


if __name__ == "__main__":
    main()

# NOTE - natural follow-ups, not implemented here:
#   1. Population-weighted country mean instead of area-weighted, using the
#      worldpop grid populations already computed in exposure_map.py.
#   2. Per-grid-cell WBGTx/WBGT5x maps (rather than a single country-average
#      series) - facility-level output above is a step toward this; a full
#      per-grid-cell map would apply the same functions along the time axis
#      independently for every grid cell, then map the result the same way
#      exposure_map.py maps exceedance_pct.
#   3. Warm Spell Duration Index (WSDI) equivalent - count days in spells of
#      >= 6 consecutive days above the local 90th percentile WBGT, which is
#      the direct duration-based companion to WBGTx/WBGT5x in the ETCCDI set.
#   4. Facility matching currently uses argmin nearest-neighbour, same as the
#      precip pipeline, rather than the searchsorted approach in
#      wbgt_facility_panel.py. Worth reconciling to one method pipeline-wide
#      if/when the two scripts' facility panels need to be merged.
