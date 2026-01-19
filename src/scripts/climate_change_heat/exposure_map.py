import argparse
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from tlo import Date
from tlo.analysis.utils import extract_results, summarize
import geopandas as gpd
import numpy as np
from netCDF4 import Dataset, num2date
from shapely.geometry import Polygon

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
    'high': 30,  # Heavy workload restriction
    'severe': 32,  # Suspension of strenuous activity
}

## Needed for mapping (using the first scenario's data for mapping)
malawi_admin2 = gpd.read_file(
    "/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp"
)
water_bodies = gpd.read_file(
    "/Users/rem76/Desktop/Climate_change_health/Data/Water_Supply_Control-Rivers-shp/Water_Supply_Control-Rivers.shp"
)

worldpop_gdf = gpd.read_file(
    "/Users/rem76/PycharmProjects/TLOmodel/resources/climate_change_impacts/worldpop_density_with_districts.shp"
)
worldpop_gdf["Z_prop"] = pd.to_numeric(worldpop_gdf["Z_prop"], errors="coerce")

# Load netCDF data
nc = Dataset(
    '/Users/rem76/Desktop/Climate_change_health/nex_gddp_cmip6_malawi_wbgt/ACCESS-CM2/ssp245/wbgt_day_ACCESS-CM2_ssp245_malawi_2025_2040.nc',
    'r')
print(nc.variables.keys())

wbgt_data = nc.variables['wbgt'][:]
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


#
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
    scenario_names = range(0, 9, 1)
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

    for target_year in target_year_sequence:
        TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))

        # Create time mask for this target year (OND season)
        ond_mask = (months >= 10) & (months <= 12)


        # Store district-level data for each scenario
        all_scenarios_population_by_district_mean = {}
        all_scenarios_population_by_district_upper = {}
        all_scenarios_population_by_district_lower = {}

        for draw in range(len(scenario_names_all)):
            if draw not in scenarios_of_interest:
                continue
            scenario_name = scenario_names[draw] if draw < len(scenario_names) else f"Scenario_{draw}"

            if total_population:
                result_data_population = summarize(
                    extract_results(
                        results_folder,
                        module="tlo.methods.demography",
                        key="population",
                        custom_generate_series=get_population_for_year,
                        do_scaling=True,
                    ),
                    only_mean=True,
                    collapse_columns=True,
                )[draw]
            else:
                over_65_M = summarize(
                    extract_results(
                        results_folder,
                        module="tlo.methods.demography",
                        key="age_range_m",
                        custom_generate_series=get_over_65_for_year,
                        do_scaling=True,
                    ),
                    only_mean=True,
                    collapse_columns=True,
                )[draw]
                over_65_F = summarize(
                    extract_results(
                        results_folder,
                        module="tlo.methods.demography",
                        key="age_range_f",
                        custom_generate_series=get_over_65_for_year,
                        do_scaling=True,
                    ),
                    only_mean=True,
                    collapse_columns=True,
                )[draw]
                result_data_population = over_65_M + over_65_F

            all_scenarios_population_by_district_mean[draw] = result_data_population['mean']
            all_scenarios_population_by_district_lower[draw] = result_data_population['lower']
            all_scenarios_population_by_district_upper[draw] = result_data_population['upper']

        # Calculate exceedances for each threshold for this target year
        exceedance_data = {}
        for threshold_name, threshold_value in WBGT_THRESHOLDS.items():
            exceedance_days, total_days = calculate_threshold_exceedances(
                wbgt_data, threshold_value, ond_mask, target_year
            )
            exceedance_pct = (exceedance_days / total_days) * 100 if total_days > 0 else np.zeros_like(exceedance_days)
            exceedance_data[threshold_name] = {
                'days': exceedance_days,
                'pct': exceedance_pct,
                'total_days': total_days,
                'threshold': threshold_value
            }

        # Create maps: 1 population panel + 4 threshold panels
        fig, axes = plt.subplots(1, 5, figsize=(20, 6))
        axes = axes.flatten()

        # Panel 0: Population distribution
        malawi_admin2["Population"] = malawi_admin2["ADM2_EN"].map(
            all_scenarios_population_by_district_mean[scenarios_of_interest[0]]
        )
        axes[0].axis("off")
        water_bodies.plot(ax=axes[0], facecolor="#7BDFF2", alpha=0.6, edgecolor="#999999", linewidth=0.5, hatch="xxx")
        water_bodies.plot(ax=axes[0], facecolor="#7BDFF2", edgecolor="black", linewidth=1)

        district_pop_lookup = malawi_admin2.set_index("ADM2_EN")["Population"]
        worldpop_gdf["district_population"] = worldpop_gdf["ADM2_EN"].map(district_pop_lookup)
        worldpop_gdf["grid_population"] = np.log(
            worldpop_gdf["Z_prop"] * worldpop_gdf["district_population"]
        )

        worldpop_gdf.plot(
            column="grid_population",
            ax=axes[0],
            cmap="Greys",
            legend=True,
            legend_kwds={"label": "log(Population)", "shrink": 0.6},
        )
        malawi_admin2.boundary.plot(ax=axes[0], edgecolor='black', linewidth=0.5)
        axes[0].set_title(f"Population Distribution\n{target_year}", fontsize=11)

        # Panels 1-3: WBGT threshold exceedances for this target year
        for idx, (threshold_name, data) in enumerate(exceedance_data.items()):
            ax = axes[idx + 1]

            # Create grid for this threshold's exceedance percentage
            grid = create_exceedance_grid(data['pct'], malawi_admin2.crs)
            grid_clipped = gpd.overlay(grid, malawi_admin2, how='intersection')

            water_bodies.plot(ax=ax, facecolor="#7BDFF2", alpha=0.6, edgecolor="#999999", linewidth=0.5, hatch="xxx")
            water_bodies.plot(ax=ax, facecolor="#7BDFF2", edgecolor="black", linewidth=1)

            grid_clipped.plot(
                column='value',
                ax=ax,
                cmap='YlOrRd',
                edgecolor='grey',
                linewidth=0.2,
                legend=True,
                legend_kwds={
                    'label': f"% of OND days > {data['threshold']}°C",
                    'shrink': 0.6
                },
                vmin=0,
                vmax=100
            )

            malawi_admin2.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
            ax.set_title(
                f"WBGT > {data['threshold']}°C ({threshold_name.capitalize()})\n"
                f"OND {target_year} ({data['total_days']} days)",
                fontsize=11
            )

        fig.suptitle(
            f"Population and WBGT Threshold Exceedances - Malawi {target_year}",
            fontsize=14, fontweight='bold'
        )
        fig.tight_layout()
        plt.show()
        fig.savefig(output_folder / f"wbgt_exceedance_{target_year}_{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Store data for this year
        all_years_data_population_mean[target_year] = all_scenarios_population_by_district_mean
        all_years_data_population_lower[target_year] = all_scenarios_population_by_district_lower
        all_years_data_population_upper[target_year] = all_scenarios_population_by_district_upper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(results_folder=args.results_folder, output_folder=args.results_folder, resourcefilepath=Path("./resources"))
