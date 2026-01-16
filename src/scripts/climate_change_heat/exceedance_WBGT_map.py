import argparse
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from tlo import Date
from tlo.analysis.utils import extract_results, summarize
import geopandas as gpd
import numpy as np
from netCDF4 import Dataset
from shapely.geometry import Polygon


min_year = 2025
max_year = 2027
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
scenario_names = [ "SSP 2.45 Mean"]

scenario_colours = ["#0081a7", "#00afb9", "#FEB95F", "#fed9b7", "#f07167"] * 4

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

# Get WBGT

# Load netCDF data
nc = Dataset('/Users/rem76/Desktop/Climate_change_health/nex_gddp_cmip6_malawi_wbgt/ACCESS-CM2/ssp245/wbgt_day_ACCESS-CM2_ssp245_malawi_2025_2040.nc', 'r')
print(nc.variables.keys())

wbgt_data = nc.variables['wbgt'][:]  # adjust variable name if needed
lat_data = nc.variables['lat'][:]
lon_data = nc.variables['lon'][:]

# Get time variable and convert to datetime
time_var = nc.variables['time']
from netCDF4 import num2date
times = num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, 'calendar', 'standard'))

# Get month for each timestep
months = np.array([t.month for t in times])

# Filter for Oct, Nov, Dec (months 10, 11, 12)
ond_mask = (months >= 10) & (months <= 12)
wbgt_mean = np.mean(wbgt_data[ond_mask, :, :], axis=0)
# Create grid polygons from netCDF coordinates
difference_lat = lat_data[1] - lat_data[0]
difference_lon = lon_data[1] - lon_data[0]

polygons = []
wbgt_values = []

for i, y in enumerate(lat_data):
    for j, x in enumerate(lon_data):
        bottom_left = (x, y)
        bottom_right = (x + difference_lon, y)
        top_right = (x + difference_lon, y + difference_lat)
        top_left = (x, y + difference_lat)
        polygon = Polygon([bottom_left, bottom_right, top_right, top_left])
        polygons.append(polygon)
        wbgt_values.append(wbgt_mean[i, j])

# Create GeoDataFrame with WBGT values
grid = gpd.GeoDataFrame({
    'geometry': polygons,
    'wbgt': wbgt_values
}, crs=malawi_admin2.crs)

# Clip grid to Malawi boundaries
grid_clipped = gpd.overlay(grid, malawi_admin2, how='intersection')


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
    """Produce a standard set of plots describing the effect of each climate scenario.
    - Generate time trend plots of deaths and DALYs by cause and district.
    - Create a final summary plot showing total deaths and DALYs per district stacked by scenario.
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

        # Store district-level data for each scenario
        all_scenarios_population_by_district_mean = {}
        all_scenarios_population_by_district_upper = {}
        all_scenarios_population_by_district_lower = {}
        for draw in range(len(scenario_names_all)):
            if draw not in scenarios_of_interest:
                continue
            scenario_name = scenario_names[draw]
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

        # Create maps for each scenario
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        axes = axes.flatten()

        for i in range(2):
            malawi_admin2["Population"] = malawi_admin2["ADM2_EN"].map(all_scenarios_population_by_district_mean[0])
            # malawi_admin2.plot(
            #     column="Population", ax=axes[i], legend=True, cmap="Blues", edgecolor="black")
            axes[i].axis("off")
            water_bodies.plot(ax=axes[i], facecolor="#7BDFF2", alpha=0.6, edgecolor="#999999", linewidth=0.5, hatch="xxx")
            water_bodies.plot(ax=axes[i], facecolor="#7BDFF2", edgecolor="black", linewidth=1)

            # set a lookup table
        district_pop_lookup = (
                malawi_admin2
                .set_index("ADM2_EN")["Population"]
            )
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
        grid_clipped.boundary.plot(ax=axes[0], edgecolor='grey', linewidth=0.3)

        # Plot clipped grid with WBGT values
        grid_clipped.plot(column='wbgt', ax=axes[1], cmap='YlOrRd',
                              edgecolor='grey', linewidth=0.2, legend=True,
                              legend_kwds={'label': 'WBGT (°C)', 'shrink': 0.7})
        fig.tight_layout()
        plt.show()
        #fig.savefig(output_folder / f"exposed_population_{target_year}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    all_years_data_population_mean[target_year] = all_scenarios_population_by_district_mean

    all_years_data_population_lower[target_year] = all_scenarios_population_by_district_lower

    all_years_data_population_upper[target_year] = all_scenarios_population_by_district_upper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(results_folder=args.results_folder, output_folder=args.results_folder, resourcefilepath=Path("./resources"))
