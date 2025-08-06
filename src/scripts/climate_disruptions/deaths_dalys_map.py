import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    get_color_cause_of_death_or_daly_label,
    summarize,
    parse_log_file
)
import geopandas as gpd

min_year = 2020
max_year = 2028
spacing_of_years = 1
PREFIX_ON_FILENAME = '1'

scenario_names = ["Baseline", "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean", "SSP 2.45 High", "SSP 2.45 Low", "SSP 2.45 Mean",  "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean"]
scenario_names = ["Baseline"]#, "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean", "SSP 2.45 High", "SSP 2.45 Low", "SSP 2.45 Mean",  "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean"]

scenario_colours = ['#0081a7', '#00afb9', '#FEB95F', '#fed9b7', '#f07167']*4
def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    output = parse_log_file('/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/climate_scenario_runs-2025-08-01T121521Z/0/0/climate_scenario_runs__2025-08-01T121736.log')

    """Produce a standard set of plots describing the effect of each climate scenario.

    - Estimate the epidemiological impact as the EXTRA deaths that would occur if the treatment did not occur.
    - Estimate the draw on healthcare system resources as the FEWER appointments when the treatment does not occur.
    - Generate time trend plots of deaths and DALYs by cause and district.
    - Create a final summary plot showing total deaths and DALYs per district stacked by scenario.
    """
    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    # Definitions of general helper functions
    def get_num_deaths_by_district(_df):
        """Return total number of Deaths by label (total by age-group within the TARGET_PERIOD)
        """
        return _df \
            .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .groupby(_df['district_of_residence']) \
            .size()

    def get_num_dalys_by_district(_df):
        """Return total number of DALYs by (district) as a Series, within the TARGET PERIOD."""

        return _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .drop(columns=['date', 'year']) \
            .groupby('district_of_residence') \
            .sum().sum(axis = 1)

    def get_population_for_year(_df):
        """Returns the population per district from the filtered year(s) as a Series."""
        return _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]\
            .drop(columns=['female', 'male', 'total', 'date'], errors='ignore')['district_of_residence']  \
            .apply(pd.Series) \
            .iloc[0]

    target_year_sequence = range(min_year, max_year, spacing_of_years)

    # Store district-level data for each scenario
    all_scenarios_dalys_by_district = {}
    all_scenarios_deaths_by_district = {}

    for draw in range(len(scenario_names)):
        scenario_name = scenario_names[draw]
        make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}_{draw}.png"  # noqa: E731

        all_years_data_deaths_mean = {}
        all_years_data_deaths_upper= {}
        all_years_data_deaths_lower = {}

        all_years_data_dalys_mean = {}
        all_years_data_dalys_upper = {}
        all_years_data_dalys_lower = {}

        for target_year in target_year_sequence:
            print(target_year)
            TARGET_PERIOD = (
                Date(target_year, 1, 1), Date(target_year, 12, 31))  # Corrected the year range to cover 5 years.
            # Absolute Number of Deaths and DALYs

            result_data_deaths = summarize(extract_results(
                results_folder,
                module='tlo.methods.demography.detail',
                key='properties_of_deceased_persons',
                custom_generate_series=get_num_deaths_by_district,
                do_scaling=True
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            result_data_dalys = summarize(
                extract_results(
                    results_folder,
                    module='tlo.methods.healthburden',
                    key='dalys_by_district_stacked_by_age_and_time',
                    custom_generate_series=get_num_dalys_by_district,
                    do_scaling=True
                ),
                only_mean=True,
                collapse_columns=True,
            )[(0,)]


            result_data_population = summarize(extract_results(
                results_folder,
                module='tlo.methods.demography',
                key='population',
                custom_generate_series=get_population_for_year,
                do_scaling=True
            ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            all_years_data_dalys_mean[target_year] = result_data_dalys['mean']/result_data_population['mean'] * 1000
            all_years_data_deaths_mean[target_year] = result_data_deaths['mean']/result_data_population['mean'] * 1000

            all_years_data_dalys_lower[target_year] = result_data_dalys['lower']/result_data_population['lower'] * 1000
            all_years_data_deaths_lower[target_year] = result_data_deaths['lower']/result_data_population['lower'] * 1000

            all_years_data_dalys_upper[target_year] = result_data_dalys['upper']/result_data_population['upper'] * 1000
            all_years_data_deaths_upper[target_year] = result_data_deaths['upper']/result_data_population['upper'] * 1000

        # Convert the accumulated data into a DataFrame for plotting
        df_all_years_DALYS_mean = pd.DataFrame(all_years_data_dalys_mean)
        df_all_years_DALYS_lower = pd.DataFrame(all_years_data_dalys_lower)
        df_all_years_DALYS_upper = pd.DataFrame(all_years_data_dalys_upper)
        df_all_years_deaths_mean = pd.DataFrame(all_years_data_deaths_mean)
        df_all_years_deaths_lower = pd.DataFrame(all_years_data_deaths_lower)
        df_all_years_deaths_upper = pd.DataFrame(all_years_data_deaths_upper)

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side

        # Panel A: Deaths
        for (district) in df_all_years_deaths_mean.index:
            axes[0].plot(
                df_all_years_deaths_mean.columns,
                df_all_years_deaths_mean.loc[(district)],
                marker='o',
                label=district
            )
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of deaths')
        axes[0].grid(False)
        # Panel B: DALYs
        for (district) in df_all_years_DALYS_mean.index:
            axes[1].plot(
                df_all_years_DALYS_mean.columns,
                df_all_years_DALYS_mean.loc[(district)],
                marker='o',
                label=district,
            )
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Number of DALYs')
        axes[1].legend(title='District', bbox_to_anchor=(1., 1), loc='upper left')
        axes[1].grid(False)

        fig.savefig(make_graph_file_name('Trend_Deaths_and_DALYs_by_condition_All_Years_Panel_A_and_B_Scatter'))

        # BARPLOT STACKED DEATHS AND DALYS OVER TIME
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side
        df_all_years_deaths_mean.T.plot.bar(
            stacked=True,
            ax=axes[0],
        )

        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of deaths')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].legend(title='District', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid()

        # Plot the stacked bar chart
        df_all_years_DALYS_mean.T.plot.bar(
            stacked=True,
            ax=axes[1],
        )

        axes[1].axhline(0.0, color='black')
        axes[1].set_title('Panel B: DALYs')
        axes[1].set_ylabel('Number of DALYs')
        axes[1].set_xlabel('Year')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].legend(
            title='Condition (by district)',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=8,
            ncol=2
        )
        axes[1].grid()

        fig.tight_layout()
        fig.savefig(make_graph_file_name('Trend_Deaths_and_DALYs_by_condition_All_Years_Panel_A_and_B_Stacked'))

        district_dalys_total = df_all_years_DALYS_mean.mean(axis=1)  # Average across years for each district
        district_deaths_total = df_all_years_deaths_mean.mean(axis=1)  # Average across years for each district

        all_scenarios_dalys_by_district[scenario_name] = district_dalys_total
        all_scenarios_deaths_by_district[scenario_name] = district_deaths_total

    # Create DataFrames with districts as rows and scenarios as columns
    df_dalys_by_district_all_scenarios = pd.DataFrame(all_scenarios_dalys_by_district)
    df_deaths_by_district_all_scenarios = pd.DataFrame(all_scenarios_deaths_by_district)

    # Plot DALYs by district for each scenario
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Panel A: Deaths by district for each scenario
    df_deaths_by_district_all_scenarios.plot(kind='bar', ax=axes[0], color=scenario_colours[:len(scenario_names)])
    axes[0].set_title(f'Deaths by District and Scenario ({min_year}-{max_year})')
    axes[0].set_xlabel('District')
    axes[0].set_ylabel('Deaths per 1000 population')
    axes[0].legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)

    # Panel B: DALYs by district for each scenario
    df_dalys_by_district_all_scenarios.plot(kind='bar', ax=axes[1], color=scenario_colours[:len(scenario_names)])
    axes[1].set_title(f'DALYs by District and Scenario ({min_year}-{max_year})')
    axes[1].set_xlabel('District')
    axes[1].set_ylabel('DALYs per 1000 population')
    axes[1].legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_folder / "deaths_and_dalys_by_district_all_scenarios.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Additional plot: Stacked bar chart showing district contribution to total for each scenario
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # Panel A: Stacked deaths by scenario
    df_deaths_by_district_all_scenarios.T.plot(kind='bar', stacked=True, ax=axes[0])
    axes[0].set_title(f'Total Deaths by Scenario ({min_year}-{max_year})')
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Deaths per 1000 population')
    axes[0].legend(title='District', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].tick_params(axis='x', rotation=45)

    # Panel B: Stacked DALYs by scenario
    df_dalys_by_district_all_scenarios.T.plot(kind='bar', stacked=True, ax=axes[1])
    axes[1].set_title(f'Total DALYs by Scenario ({min_year}-{max_year})')
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('DALYs per 1000 population')
    axes[1].legend(title='District', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].tick_params(axis='x', rotation=45)

    fig.tight_layout()
    fig.savefig(output_folder / "stacked_deaths_and_dalys_by_scenario.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    ## Now do mapping (using the first scenario's data for mapping)
    malawi = gpd.read_file(
        "/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm0_nso_20181016.shp")
    malawi_admin2 = gpd.read_file(
        "/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp")
    water_bodies = gpd.read_file(
        "/Users/rem76/Desktop/Climate_change_health/Data/Water_Supply_Control-Rivers-shp/Water_Supply_Control-Rivers.shp")

    # change names of some districts for consistency
    malawi_admin2['ADM2_EN'] = malawi_admin2['ADM2_EN'].replace('Blantyre City', 'Blantyre')
    malawi_admin2['ADM2_EN'] = malawi_admin2['ADM2_EN'].replace('Mzuzu City', 'Mzuzu')
    malawi_admin2['ADM2_EN'] = malawi_admin2['ADM2_EN'].replace('Lilongwe City', 'Lilongwe')
    malawi_admin2['ADM2_EN'] = malawi_admin2['ADM2_EN'].replace('Zomba City', 'Zomba')

    # Create maps for each scenario
    n_scenarios = len(scenario_names)
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    for i, scenario in enumerate(scenario_names):
            malawi_admin2['DALY_Rate'] = malawi_admin2['ADM2_EN'].map(df_dalys_by_district_all_scenarios[scenario])
            malawi_admin2.plot(column='DALY_Rate', ax=axes[i], legend=True, cmap='RdPu',edgecolor='black',)
            axes[i].set_title(f'DALYs per 1000 - {scenario}')
            axes[i].axis('off')
            water_bodies.plot(ax=axes[i], facecolor="none", edgecolor="#999999", linewidth=0.5, hatch="xxx")
            water_bodies.plot(ax=axes[i], facecolor="none", edgecolor="black", linewidth=1)

    fig.tight_layout()
    fig.savefig(output_folder / "dalys_maps_all_scenarios.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    # Save data as CSV
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()


    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
