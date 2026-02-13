import argparse
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from tlo import Date
from tlo.analysis.utils import extract_results, summarize
import geopandas as gpd

min_year = 2025
max_year = 2031
spacing_of_years = 1
PREFIX_ON_FILENAME = "1"

scenario_colours = ["#0081a7", "#00afb9", "#FEB95F", "#fed9b7", "#f07167"] * 4

district_colours = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "navy",
    "maroon",
    "teal",
    "lime",
    "aqua",
    "fuchsia",
    "silver",
    "gold",
    "indigo",
    "violet",
    "crimson",
    "coral",
    "salmon",
    "khaki",
    "plum",
    "orchid",
    "tan",
    "wheat",
    "azure",
]

vmin = -0.065
vmax = 0.065
climate_sensitivity_analysis = False
parameter_sensitivity_analysis = False
main_text = True
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
    scenario_names = ["No disruptions", "Baseline", "Worst case"]

    suffix = "main_text"
    scenarios_of_interest = [0, 1, 2]


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce a standard set of plots describing the effect of each climate scenario.
    - Generate time trend plots of DALYs by cause and district.
    - Create a final summary plot showing total DALYs per district stacked by scenario.
    """
    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    # Definitions of general helper functions
    def get_num_dalys_by_district(_df):
        """Return total number of DALYs by district as a Series, within the TARGET PERIOD."""

        return (
            _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]
            .drop(columns=["date", "year"])
            .groupby("district_of_residence")
            .sum()
            .sum(axis=1)
        )

    def get_population_for_year(_df):
        """Returns the population per district in the year of interest"""
        _df["date"] = pd.to_datetime(_df["date"])

        filtered_df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
        numeric_df = filtered_df.drop(columns=["female", "male", "date", "total"], errors="ignore")

        district_sums = pd.Series(numeric_df["district_of_residence"].sum())
        return district_sums

    target_year_sequence = range(min_year, max_year, spacing_of_years)

    # Store district-level data for each scenario
    all_scenarios_dalys_by_district = {}
    all_scenarios_dalys_by_district_upper = {}
    all_scenarios_dalys_by_district_lower = {}
    all_scenarios_dalys_by_district_per_1000 = {}
    all_scenarios_dalys_by_district_upper_per_1000 = {}
    all_scenarios_dalys_by_district_lower_per_1000 = {}

    for draw in range(len(scenario_names)):
        if draw not in scenarios_of_interest:
            continue
        scenario_name = scenario_names[draw]
        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}_{draw}.png"  # noqa: E731

        all_years_data_dalys_mean_per_1000 = {}
        all_years_data_dalys_upper_per_1000 = {}
        all_years_data_dalys_lower_per_1000 = {}

        all_years_data_dalys_mean = {}
        all_years_data_dalys_upper = {}
        all_years_data_dalys_lower = {}

        for target_year in target_year_sequence:
            TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))

            # Absolute Number of DALYs
            result_data_dalys = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.healthburden",
                    key="dalys_by_district_stacked_by_age_and_time",
                    custom_generate_series=get_num_dalys_by_district,
                    do_scaling=True,
                ),
                only_mean=True,
                collapse_columns=True,
            )[(draw,)]

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

            all_years_data_dalys_mean_per_1000[target_year] = result_data_dalys["mean"] / result_data_population['mean']
            all_years_data_dalys_lower_per_1000[target_year] = result_data_dalys["lower"] / result_data_population[
                'lower']
            all_years_data_dalys_upper_per_1000[target_year] = result_data_dalys["upper"] / result_data_population[
                'upper']

            all_years_data_dalys_mean[target_year] = result_data_dalys["mean"]
            all_years_data_dalys_lower[target_year] = result_data_dalys["lower"]
            all_years_data_dalys_upper[target_year] = result_data_dalys["upper"]

        # Convert the accumulated data into a DataFrame for plotting
        df_all_years_DALYS_mean = pd.DataFrame(all_years_data_dalys_mean)
        df_all_years_DALYS_lower = pd.DataFrame(all_years_data_dalys_lower)
        df_all_years_DALYS_upper = pd.DataFrame(all_years_data_dalys_upper)

        df_all_years_DALYS_mean_per_1000 = pd.DataFrame(all_years_data_dalys_mean_per_1000)
        df_all_years_DALYS_lower_per_1000 = pd.DataFrame(all_years_data_dalys_lower_per_1000)
        df_all_years_DALYS_upper_per_1000 = pd.DataFrame(all_years_data_dalys_upper_per_1000)

        # Plotting - Line plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        for i, district in enumerate(df_all_years_DALYS_mean.index):
            ax.plot(
                df_all_years_DALYS_mean.columns,
                df_all_years_DALYS_mean.loc[(district)],
                marker="o",
                label=district,
                color=district_colours[i],
            )
        ax.set_xlabel("Year")
        ax.set_ylabel("DALYs per 1,000")
        ax.legend(title="District", bbox_to_anchor=(1.0, 1), loc="upper left")
        ax.grid(False)
        fig.savefig(make_graph_file_name("Trend_DALYs_by_district_All_Years_Scatter"))
        plt.close(fig)

        # BARPLOT STACKED DALYS OVER TIME
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        # Plot the stacked bar chart
        df_all_years_DALYS_mean.T.plot.bar(
            stacked=True,
            ax=ax,
            color=[district_colours[i] for i in range(len(df_all_years_DALYS_mean.index))],
        )

        ax.axhline(0.0, color="black")
        ax.set_title("DALYs by District Over Time")
        ax.set_ylabel("Number of DALYs")
        ax.set_xlabel("Year")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(title="District", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
        ax.grid(False)

        fig.tight_layout()
        fig.savefig(make_graph_file_name("Trend_DALYs_by_district_All_Years_Stacked"))
        plt.close(fig)

        print("df_all_years_DALYS_mean", df_all_years_DALYS_mean)

        district_dalys_total_per_1000 = df_all_years_DALYS_mean_per_1000.mean(axis=1)
        district_dalys_upper_per_1000 = df_all_years_DALYS_upper_per_1000.mean(axis=1)
        district_dalys_lower_per_1000 = df_all_years_DALYS_lower_per_1000.mean(axis=1)

        district_dalys_total = df_all_years_DALYS_mean.sum(axis=1)
        district_dalys_upper = df_all_years_DALYS_upper.sum(axis=1)
        district_dalys_lower = df_all_years_DALYS_lower.sum(axis=1)

        all_scenarios_dalys_by_district_per_1000[scenario_name] = district_dalys_total_per_1000
        all_scenarios_dalys_by_district_upper_per_1000[scenario_name] = district_dalys_upper_per_1000
        all_scenarios_dalys_by_district_lower_per_1000[scenario_name] = district_dalys_lower_per_1000

        all_scenarios_dalys_by_district[scenario_name] = district_dalys_total
        all_scenarios_dalys_by_district_upper[scenario_name] = district_dalys_upper
        all_scenarios_dalys_by_district_lower[scenario_name] = district_dalys_lower

    df_dalys_by_district_all_scenarios = pd.DataFrame(all_scenarios_dalys_by_district)
    df_dalys_by_district_all_scenarios_upper = pd.DataFrame(all_scenarios_dalys_by_district_upper)
    df_dalys_by_district_all_scenarios_lower = pd.DataFrame(all_scenarios_dalys_by_district_lower)

    df_dalys_by_district_all_scenarios_per_1000 = pd.DataFrame(all_scenarios_dalys_by_district_per_1000)
    df_dalys_by_district_all_scenarios_upper_per_1000 = pd.DataFrame(all_scenarios_dalys_by_district_upper_per_1000)
    df_dalys_by_district_all_scenarios_lower_per_1000 = pd.DataFrame(all_scenarios_dalys_by_district_lower_per_1000)

    # Calculate means and error bars
    dalys_means_per_1000 = df_dalys_by_district_all_scenarios_per_1000.mean(axis=0)
    dalys_upper_per_1000 = df_dalys_by_district_all_scenarios_upper_per_1000.mean(axis=0)
    dalys_lower_per_1000 = df_dalys_by_district_all_scenarios_lower_per_1000.mean(axis=0)

    dalys_means = df_dalys_by_district_all_scenarios.mean(axis=0)
    dalys_upper = df_dalys_by_district_all_scenarios_upper.mean(axis=0)
    dalys_lower = df_dalys_by_district_all_scenarios_lower.mean(axis=0)

    # Calculate error bar values (difference from mean)
    dalys_yerr_upper = dalys_upper - dalys_means
    dalys_yerr_lower = dalys_means - dalys_lower
    dalys_yerr = [dalys_yerr_lower, dalys_yerr_upper]

    # Plot with error bars
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # DALYs by scenario with error bars
    ax.bar(
        range(len(dalys_means)),
        dalys_means,
        yerr=dalys_yerr,
        capsize=5,
        color=scenario_colours[: len(scenario_names)],
        alpha=0.8,
        error_kw={"elinewidth": 2, "capthick": 2},
    )
    ax.set_title("DALYs by Scenario", fontsize=14, fontweight="bold")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("DALYs")
    ax.set_xticks(range(len(dalys_means)))
    ax.set_xticklabels(dalys_means.index, rotation=45, ha="right")
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(output_folder / "dalys_total_all_scenarios.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Additional plot: Stacked bar chart showing district contribution to total for each scenario
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Stacked DALYs by scenario
    df_dalys_by_district_all_scenarios.T.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("DALYs")
    ax.legend(title="District", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_folder / "stacked_dalys_by_district_scenario.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save data as CSV
    df_dalys_by_district_all_scenarios_per_1000.to_csv(
        output_folder / "dalys_by_district_scenario.csv", index=False
    )

    ## Now do mapping (using the first scenario's data for mapping)
    malawi_admin2 = gpd.read_file(
        "/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp"
    )
    water_bodies = gpd.read_file(
        "/Users/rem76/Desktop/Climate_Change_Health/Data/Water_Supply_Control-Rivers-shp/Water_Supply_Control-Rivers.shp"
    )
    # change names of some districts for consistency
    malawi_admin2["ADM2_EN"] = malawi_admin2["ADM2_EN"].replace("Blantyre City", "Blantyre")
    malawi_admin2["ADM2_EN"] = malawi_admin2["ADM2_EN"].replace("Mzuzu City", "Mzuzu")
    malawi_admin2["ADM2_EN"] = malawi_admin2["ADM2_EN"].replace("Lilongwe City", "Lilongwe")
    malawi_admin2["ADM2_EN"] = malawi_admin2["ADM2_EN"].replace("Zomba City", "Zomba")
    print(df_dalys_by_district_all_scenarios)

    # Create maps for each scenario
    fig, axes = plt.subplots(1, 3, figsize=(18, 18))
    axes = axes.flatten()
    for i, scenario in enumerate(scenario_names[1:], start=1):
        i = i - 1
        difference_from_baseline_per_1000 = (
            df_dalys_by_district_all_scenarios_per_1000[scenario]
            - df_dalys_by_district_all_scenarios_per_1000["No disruptions"]
        )
        malawi_admin2["DALY_Rate"] = malawi_admin2["ADM2_EN"].map(difference_from_baseline_per_1000)
        print(malawi_admin2["DALY_Rate"])
        malawi_admin2.plot(
            column="DALY_Rate", ax=axes[i], legend=True, cmap="PiYG", edgecolor="black", vmin=vmin, vmax=vmax)
        axes[i].set_title(f"DALYs per 1000 - {scenario}")
        axes[i].axis("off")
        water_bodies.plot(ax=axes[i], facecolor="#7BDFF2", alpha=0.6, edgecolor="#999999", linewidth=0.5, hatch="xxx")
        water_bodies.plot(ax=axes[i], facecolor="#7BDFF2", edgecolor="black", linewidth=1)

    fig.tight_layout()
    fig.savefig(output_folder / "dalys_maps_all_scenarios_difference.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(results_folder=args.results_folder, output_folder=args.results_folder, resourcefilepath=Path("./resources"))
