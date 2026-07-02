import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # NEW: headless backend - avoids popping up a window per year during batch GIF generation
from matplotlib import pyplot as plt

import pandas as pd
from tlo import Date
from tlo.analysis.utils import extract_results, summarize
import geopandas as gpd
import numpy as np
from netCDF4 import Dataset, num2date
from shapely.geometry import Polygon
from PIL import Image  # NEW: for assembling frames into an animated GIF

min_year = 2025
max_year = 2040
spacing_of_years = 1
PREFIX_ON_FILENAME = "1"
scenario_names_all = [
    "Baseline",
    "SSP 1.26 High",
    "SSP 1.26 Low",
    "SSP 1.26 Mean",
    "SSP 2.45 High",
    "SSP 2.45 Low",
    "SSP 2.45 Mean",
    "SSP 5.85 High",
    "SSP 5.85 Low",
    "SSP 5.85 Mean",
]
scenario_names = ["SSP 2.45 Mean"]

scenario_colours = ["#0081a7", "#00afb9", "#FEB95F", "#fed9b7", "#f07167"] * 4

# WBGT thresholds (following Gohar et al.)
WBGT_THRESHOLDS = {
    'baseline': 15,
    'moderate': 28,  # Moderate work restriction
    'high': 30,      # Heavy workload restriction
    'severe': 32,    # Suspension of strenuous activity
}

# Which GCM's WBGT output to map
WBGT_MODEL = "ACCESS-CM2"
WBGT_SCENARIO = "ssp245"

# NEW: GIF settings
MAKE_GIF = True
GIF_FRAME_DURATION_MS = 800   # time each year is shown for
GIF_LOOP = 0                  # 0 = loop forever
GIF_FRAME_MAX_WIDTH = 1600    # downscale frames so the GIF file size stays reasonable
KEEP_FRAME_PNGS = True        # set False to delete the per-year PNGs once the GIF is built

## Needed for mapping
malawi_admin2 = gpd.read_file(
    "/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp"
)

worldpop_gdf = gpd.read_file(
    "/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/climate_change_impacts/worldpop_density_with_districts.shp"
)
worldpop_gdf["Z_prop"] = pd.to_numeric(worldpop_gdf["Z_prop"], errors="coerce")

# Load netCDF data - reads the day/night bracketed WBGT output
# (wbgt_daynight_..., with wbgt_day / wbgt_night variables instead of a
# single "wbgt" variable)
wbgt_nc_path = (
    f"/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/NASA_GDDP_CMIP6/"
    f"{WBGT_MODEL}/{WBGT_SCENARIO}/wbgt_daynight_{WBGT_MODEL}_{WBGT_SCENARIO}_malawi_{min_year}_{max_year}.nc"
)
nc = Dataset(wbgt_nc_path)
print(nc.variables.keys())

wbgt_day_data = nc.variables['wbgt_day'][:]
wbgt_night_data = nc.variables['wbgt_night'][:]
lat_data = nc.variables['lat'][:]
lon_data = nc.variables['lon'][:]

# Get time variable and convert to datetime
time_var = nc.variables['time']
times = num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, 'calendar', 'standard'))

# Get month and year for each timestep
months = np.array([t.month for t in times])
years = np.array([t.year for t in times])

# Create grid polygons from netCDF coordinates (do this once)
difference_lat = lat_data[1] - lat_data[0]
difference_lon = lon_data[1] - lon_data[0]

polygons = []
for i, y in enumerate(lat_data):
    for j, x in enumerate(lon_data):
        bottom_left = (x, y)
        bottom_right = (x + difference_lon, y)
        top_right = (x + difference_lon, y + difference_lat)
        top_left = (x, y + difference_lat)
        polygon = Polygon([bottom_left, bottom_right, top_right, top_left])
        polygons.append(polygon)


def calculate_threshold_exceedances(wbgt_data, threshold, time_mask, year):
    """
    Calculate number of days exceeding a WBGT threshold.

    Returns:
    --------
    exceedance_days : 2D array (lat, lon) - number of days exceeding threshold
    total_days : int - total number of days in the analysis period
    """
    year_mask = time_mask & (years == year)
    wbgt_subset = wbgt_data[year_mask, :, :]
    total_days = year_mask.sum()
    exceedance_days = np.sum(wbgt_subset > threshold, axis=0)
    print(exceedance_days)
    return exceedance_days, total_days


def create_exceedance_grid(exceedance_values, crs):
    """Create a GeoDataFrame grid from exceedance values."""
    grid_values = []
    for i in range(len(lat_data)):
        for j in range(len(lon_data)):
            grid_values.append(exceedance_values[i, j])

    return gpd.GeoDataFrame({
        'geometry': polygons,
        'value': grid_values
    }, crs=crs)


def transpose_if_needed(extracted):
    """
    If 'draw' and 'run' are column MultiIndex levels (as returned by
    extract_results), transpose so they become row index levels, then
    reset the index so they are regular columns that groupby('draw') can find.
    """
    if isinstance(extracted.columns, pd.MultiIndex) and 'draw' in extracted.columns.names:
        extracted = extracted.T
        extracted.index.names = ['draw', 'run']
        extracted = extracted.reset_index()
    return extracted


def create_gif_from_frames(frame_paths, output_path, duration_ms=800, loop=0, max_width=1600):
    """
    Assemble a sequence of PNG frames (one per year) into an animated GIF.
    Frames are downscaled to max_width to keep the GIF file size manageable -
    the source PNGs are saved at dpi=300 and would otherwise produce a huge GIF.
    """
    if not frame_paths:
        print("No frames to animate - skipping GIF creation.")
        return

    frames = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        if img.width > max_width:
            scale = max_width / img.width
            new_size = (max_width, int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)
        frames.append(img)

    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=True,
    )
    print(f"Saved animated GIF: {output_path}")


vmin = -1000
vmax = 1000
total_population = True
climate_sensitivity_analysis = False
parameter_sensitivity_analysis = True
main_text = True
mode_2 = False

if climate_sensitivity_analysis:
    scenario_names = [
        "Baseline",
        "SSP 1.26 High",
        "SSP 1.26 Low",
        "SSP 1.26 Mean",
        "SSP 2.45 High",
        "SSP 2.45 Low",
        "SSP 2.45 Mean",
        "SSP 5.85 High",
        "SSP 5.85 Low",
        "SSP 5.85 Mean",
    ]
    suffix = "climate_SA"
    scenarios_of_interest = range(len(scenario_names))

if parameter_sensitivity_analysis:
    scenario_names = list(range(0, 9, 1))  # FIX: convert range to list
    scenarios_of_interest = scenario_names
    suffix = "parameter_SA"

if main_text:
    scenario_names = [
        "Baseline",
        "SSP 2.45 Mean",
    ]
    suffix = "main_text"
    scenarios_of_interest = [0, 1]

if mode_2:
    scenario_names = [
        "Baseline",
        "SSP 5.85 Mean",
    ]
    suffix = "mode_2"
    scenarios_of_interest = [0, 1]


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce threshold exceedance maps showing days exceeding WBGT thresholds.
    - Left panel: Population distribution
    - Right panels: Days/percentage exceeding each WBGT threshold for the target year
    Also assembles the per-year frames into an animated GIF once all years are done.
    """
    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    def get_population_for_year(_df):
        """Returns the population per district in the year of interest"""
        _df["date"] = pd.to_datetime(_df["date"])
        filtered_df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
        numeric_df = filtered_df.drop(columns=["female", "male", "date", "total"], errors="ignore")
        district_sums = pd.Series(numeric_df["district_of_residence"].sum())
        return district_sums

    def get_over_65_for_year(_df):
        """Returns the population aged 65+ in the year of interest"""
        _df["date"] = pd.to_datetime(_df["date"])
        filtered_df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
        over_65_cols = ["65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95-99", "100+"]
        over_65_total = pd.Series(filtered_df[over_65_cols].sum().sum())
        return over_65_total

    target_year_sequence = range(min_year, max_year, spacing_of_years)

    # Store annual data for population
    all_years_data_population_mean = {}
    all_years_data_population_upper = {}
    all_years_data_population_lower = {}

    frame_paths = []  # NEW: collects each year's saved PNG so they can be stitched into a GIF

    for target_year in target_year_sequence:
        TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))

        # Create time mask for this target year (OND season)
        #ond_mask = (months >= 10) & (months <= 12)
        ond_mask = months !=13 #i.e. no mask

        # Store district-level data for each scenario
        all_scenarios_population_by_district_mean = {}
        all_scenarios_population_by_district_upper = {}
        all_scenarios_population_by_district_lower = {}

        for draw in range(len(scenario_names_all)):
            if draw not in scenarios_of_interest:
                continue
            scenario_name = scenario_names[draw] if draw < len(scenario_names) else f"Scenario_{draw}"

            if total_population:
                # FIX: Extract first, transpose if needed, then summarize
                extracted = extract_results(
                    results_folder,
                    module="tlo.methods.demography",
                    key="population",
                    custom_generate_series=get_population_for_year,
                    do_scaling=True,
                )
                extracted = transpose_if_needed(extracted)
                result_data_population = summarize(
                    extracted,
                    only_mean=False,        # FIX: need mean/lower/upper keys
                    collapse_columns=True,
                )[draw]

            else:
                # FIX: Extract, transpose, summarize for both M and F
                extracted_m = extract_results(
                    results_folder,
                    module="tlo.methods.demography",
                    key="age_range_m",
                    custom_generate_series=get_over_65_for_year,
                    do_scaling=True,
                )
                extracted_f = extract_results(
                    results_folder,
                    module="tlo.methods.demography",
                    key="age_range_f",
                    custom_generate_series=get_over_65_for_year,
                    do_scaling=True,
                )
                extracted_m = transpose_if_needed(extracted_m)
                extracted_f = transpose_if_needed(extracted_f)

                over_65_M = summarize(
                    extracted_m,
                    only_mean=False,
                    collapse_columns=True,
                )[draw]
                over_65_F = summarize(
                    extracted_f,
                    only_mean=False,
                    collapse_columns=True,
                )[draw]
                result_data_population = over_65_M + over_65_F

            # FIX: now correctly index mean/lower/upper
            all_scenarios_population_by_district_mean[draw]  = result_data_population['mean']
            all_scenarios_population_by_district_lower[draw] = result_data_population['lower']
            all_scenarios_population_by_district_upper[draw] = result_data_population['upper']

        # Calculate exceedances for each threshold for this target year,
        # separately for the day bracket (tasmax-based) and night bracket
        # (tasmin-based) - this is the "hot nights" layer.
        exceedance_data_by_bracket = {}
        for bracket_name, bracket_data in [("day", wbgt_day_data), ("night", wbgt_night_data)]:
            exceedance_data = {}
            for threshold_name, threshold_value in WBGT_THRESHOLDS.items():
                exceedance_days, total_days = calculate_threshold_exceedances(
                    bracket_data, threshold_value, ond_mask, target_year
                )
                exceedance_pct = (exceedance_days / total_days) * 100 if total_days > 0 else np.zeros_like(exceedance_days)
                exceedance_data[threshold_name] = {
                    'days': exceedance_days,
                    'pct': exceedance_pct,
                    'total_days': total_days,
                    'threshold': threshold_value
                }
            exceedance_data_by_bracket[bracket_name] = exceedance_data

        # Create maps: 1 population panel + 4 threshold panels, x2 rows (day/night)
        fig, axes = plt.subplots(2, 5, figsize=(20, 12))

        # Population distribution - same for both rows, so only compute the
        # column once and plot it in both row 0 and row 1's first slot
        malawi_admin2["Population"] = malawi_admin2["ADM2_EN"].map(
            all_scenarios_population_by_district_mean[scenarios_of_interest[0]]
        )
        district_pop_lookup = malawi_admin2.set_index("ADM2_EN")["Population"]
        worldpop_gdf["district_population"] = worldpop_gdf["ADM2_EN"].map(district_pop_lookup)
        worldpop_gdf["grid_population"] = np.log(
            worldpop_gdf["Z_prop"] * worldpop_gdf["district_population"]
        )

        for row_idx, bracket_name in enumerate(["day", "night"]):
            pop_ax = axes[row_idx, 0]
            pop_ax.axis("off")
            worldpop_gdf.plot(
                column="grid_population",
                ax=pop_ax,
                cmap="Greys",
                legend=True,
                legend_kwds={"label": "log(Population)", "shrink": 0.6},
            )
            malawi_admin2.boundary.plot(ax=pop_ax, edgecolor='black', linewidth=0.5)
            pop_ax.set_title(f"Population Distribution\n{target_year}", fontsize=11)

            # Panels 1-4: WBGT threshold exceedances for this bracket/year
            exceedance_data = exceedance_data_by_bracket[bracket_name]
            for idx, (threshold_name, data) in enumerate(exceedance_data.items()):
                ax = axes[row_idx, idx + 1]

                grid = create_exceedance_grid(data['pct'], malawi_admin2.crs)
                grid_clipped = gpd.overlay(grid, malawi_admin2, how='intersection')

                grid_clipped.plot(
                    column='value',
                    ax=ax,
                    cmap='YlOrRd',
                    edgecolor='grey',
                    linewidth=0.2,
                    legend=True,
                    legend_kwds={
                        'label': f"% of {bracket_name} > {data['threshold']}°C",
                        'shrink': 0.6
                    },
                    vmin=0,
                    vmax=100
                )

                malawi_admin2.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
                bracket_label = "Day (tasmax)" if bracket_name == "day" else "Night (tasmin)"
                ax.set_title(
                    f"WBGT {bracket_label} > {data['threshold']}°C ({threshold_name.capitalize()})\n"
                    f" {target_year} ({data['total_days']} days)",
                    fontsize=10
                )

        fig.suptitle(
            f"Population and WBGT Threshold Exceedances (Day vs Night) - Malawi {target_year}",
            fontsize=14, fontweight='bold'
        )
        fig.tight_layout()
        # NOTE: plt.show() removed for batch/GIF generation - re-add if running a single year interactively
        frame_path = output_folder / f"wbgt_exceedance_daynight_{target_year}_{suffix}.png"
        fig.savefig(frame_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        frame_paths.append(frame_path)  # NEW

        # Store data for this year
        all_years_data_population_mean[target_year] = all_scenarios_population_by_district_mean
        all_years_data_population_lower[target_year] = all_scenarios_population_by_district_lower
        all_years_data_population_upper[target_year] = all_scenarios_population_by_district_upper

    # NEW: build the animated GIF from all per-year frames
    if MAKE_GIF:
        gif_path = output_folder / f"wbgt_exceedance_daynight_animated_{suffix}.gif"
        create_gif_from_frames(
            frame_paths,
            gif_path,
            duration_ms=GIF_FRAME_DURATION_MS,
            loop=GIF_LOOP,
            max_width=GIF_FRAME_MAX_WIDTH,
        )
        if not KEEP_FRAME_PNGS:
            for p in frame_paths:
                Path(p).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(results_folder=args.results_folder, output_folder=args.results_folder, resourcefilepath=Path("./resources"))
