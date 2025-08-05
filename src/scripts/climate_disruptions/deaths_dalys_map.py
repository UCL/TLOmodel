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

min_year = 2020
max_year = 2029
spacing_of_years = 1
PREFIX_ON_FILENAME = '1'

scenario_names = ["Baseline", "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean", "SSP 2.45 High", "SSP 2.45 Low", "SSP 2.45 Mean",  "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean"]
scenario_names = ["Baseline"]
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
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    def get_num_deaths_by_district(_df):
        """Return total number of Deaths by label (total by age-group within the TARGET_PERIOD)
        """
        print("death")
        print(_df \
            .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .groupby(_df['district_of_residence']) \
            .size())

        return _df \
            .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .groupby(_df['district_of_residence']) \
            .size()

    def get_num_dalys_by_district(_df):
        """Return total number of DALYs by (district) as a Series, within the TARGET PERIOD."""
        print("dalys")
        print(_df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .drop(columns=['date', 'year']) \
            .groupby('district_of_residence') \
            .sum().sum(axis = 1))

        return _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .drop(columns=['date', 'year']) \
            .groupby('district_of_residence') \
            .sum().sum(axis = 1)


    def get_population_for_year(_df):
        """Returns the population per district in the year(s) of interest"""
        _df['date'] = pd.to_datetime(_df['date'])

        # Filter the DataFrame based on the target period
        filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
        filtered_df = filtered_df.drop(columns=['female', 'male'], errors='ignore')

        records = []
        for _, row in filtered_df.iterrows():
            date = row['date']
            district_dict = row['district_of_residence']
            if isinstance(district_dict, dict):
                for district, pop in district_dict.items():
                    records.append({'date': date, 'district': district, 'population': pop})

        district_population = pd.DataFrame(records)

        district_population = district_population.groupby('district')['population'].sum()

        return district_population

    target_year_sequence = range(min_year, max_year, spacing_of_years)
    all_draws_deaths_mean = []
    all_draws_deaths_lower = []
    all_draws_deaths_upper = []

    all_draws_dalys_mean = []
    all_draws_dalys_lower = []
    all_draws_dalys_upper = []

    for draw in range(len(scenario_names)):
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
                Date(target_year, 1, 1), Date(target_year + spacing_of_years, 12, 31))  # Corrected the year range to cover 5 years.

            # %% Quantify the health gains associated with all interventions combined.

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
            print("!!!!", type(extract_results(
                results_folder,
                module='tlo.methods.demography.detail',
                key='properties_of_deceased_persons',
                custom_generate_series=get_num_deaths_by_district,
                do_scaling=True
            )))
            print("!!", type(extract_results(
                    results_folder,
                    module='tlo.methods.healthburden',
                    key='dalys_by_district_stacked_by_age_and_time',
                    custom_generate_series=get_num_dalys_by_district,
                    do_scaling=True
                )))
            result_data_dalys = summarize(
                extract_results(
                    results_folder,
                    module='tlo.methods.healthburden',
                    key='dalys_by_district_stacked_by_age_and_time',
                    custom_generate_series=get_num_dalys_by_district,
                    do_scaling=True
                ),
                only_mean=False,
                collapse_columns=True,
            )[draw]
            #
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

            all_years_data_dalys_lower[target_year] = result_data_dalys['lower']#/result_data_population['lower'] * 1000
            #all_years_data_deaths_lower[target_year] = result_data_deaths['lower']/result_data_population['lower'] * 1000

            all_years_data_dalys_upper[target_year] = result_data_dalys['upper']#/result_data_population['upper'] * 1000
            #all_years_data_deaths_upper[target_year] = result_data_deaths['upper']/result_data_population['upper'] * 1000

        # Convert the accumulated data into a DataFrame for plotting
        df_all_years_DALYS_mean = pd.DataFrame(all_years_data_dalys_mean)
        df_all_years_DALYS_lower = pd.DataFrame(all_years_data_dalys_lower)
        df_all_years_DALYS_upper = pd.DataFrame(all_years_data_dalys_upper)
        # df_all_years_deaths_mean = pd.DataFrame(all_years_data_deaths_mean)
        # df_all_years_deaths_lower = pd.DataFrame(all_years_data_deaths_lower)
        # df_all_years_deaths_upper = pd.DataFrame(all_years_data_deaths_upper)

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side

        # Panel A: Deaths
        # for i, condition in enumerate(df_all_years_deaths_mean.index):
        #     axes[0].plot(df_all_years_deaths_mean.columns, df_all_years_deaths_mean.loc[condition], marker='o',
        #                  label=condition, color=[get_color_cause_of_death_or_daly_label(_label) for _label in
        #                                          df_all_years_deaths_mean.index][i])
        axes[0].set_title('Panel A: Deaths by Cause')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of deaths')
        axes[0].grid(False)
        # Panel B: DALYs
        print(df_all_years_DALYS_mean)
        for (district) in df_all_years_DALYS_mean.index:
            axes[1].plot(
                df_all_years_DALYS_mean.columns,
                df_all_years_DALYS_mean.loc[(district)],
                marker='o',
                label=district,
            )
        axes[1].set_title('Panel B: DALYs by cause')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Number of DALYs')
        axes[1].legend(title='Condition (by district)', bbox_to_anchor=(1., 1), loc='upper left')
        axes[1].grid()


        # BARPLOT STACKED DEATHS AND DALYS OVER TIME
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Two panels side by side
        # df_all_years_deaths_mean.T.plot.bar(stacked=True, ax=axes[1],
        #                                color=[get_color_cause_of_death_or_daly_label(_label) for _label in
        #                                       df_all_years_deaths_mean.index])

        axes[0].set_title('Panel A: Deaths by Cause')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of deaths')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].legend(title='Cause', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid()

        df_plot = df_all_years_DALYS_mean.T  # shape: (years, (district, condition))

        # Plot the stacked bar chart
        df_plot.plot.bar(
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

        # Save so can compare scenarios
        all_years_data_dalys_mean = df_all_years_DALYS_mean.sum()
        # all_years_data_deaths_mean = df_all_years_deaths_mean.sum()
        all_years_data_dalys_lower = df_all_years_DALYS_lower.sum()
        # all_years_data_deaths_lower = df_all_years_deaths_lower.sum()
        all_years_data_dalys_upper = df_all_years_DALYS_upper.sum()
        # all_years_data_deaths_upper = df_all_years_deaths_upper.sum()
        all_draws_deaths_mean.append(pd.Series(all_years_data_deaths_mean, name=f'Draw {draw}'))
        all_draws_dalys_mean.append(pd.Series(all_years_data_dalys_mean, name=f'Draw {draw}'))
        all_draws_deaths_lower.append(pd.Series(all_years_data_deaths_lower, name=f'Draw {draw}'))
        all_draws_dalys_lower.append(pd.Series(all_years_data_dalys_lower, name=f'Draw {draw}'))
        all_draws_deaths_upper.append(pd.Series(all_years_data_deaths_upper, name=f'Draw {draw}'))
        all_draws_dalys_upper.append(pd.Series(all_years_data_dalys_upper, name=f'Draw {draw}'))



    df_deaths_all_draws_mean = pd.concat(all_draws_deaths_mean, axis=1)
    df_dalys_all_draws_mean = pd.concat(all_draws_dalys_mean, axis=1)
    df_deaths_all_draws_lower = pd.concat(all_draws_deaths_lower, axis=1)
    df_dalys_all_draws_lower = pd.concat(all_draws_dalys_lower, axis=1)
    df_deaths_all_draws_upper = pd.concat(all_draws_deaths_upper, axis=1)
    df_dalys_all_draws_upper = pd.concat(all_draws_dalys_upper, axis=1)

    deaths_totals_mean = df_deaths_all_draws_mean.sum()
    dalys_totals_mean = df_dalys_all_draws_mean.sum()
    deaths_totals_lower = df_deaths_all_draws_lower.sum()
    deaths_totals_upper = df_deaths_all_draws_upper.sum()
    dalys_totals_lower = df_dalys_all_draws_lower.sum()
    dalys_totals_upper = df_dalys_all_draws_upper.sum()

    deaths_totals_err = np.array([
        deaths_totals_mean - deaths_totals_lower,
        deaths_totals_upper - deaths_totals_mean
    ])

    dalys_totals_err = np.array([
        dalys_totals_mean - dalys_totals_lower,
        dalys_totals_upper - dalys_totals_mean
    ])

    # Panel A: Total Deaths
    axes[0].bar(deaths_totals_mean.index, deaths_totals_mean.values, color=scenario_colours, yerr=deaths_totals_err,
                capsize=20)
    axes[0].set_title(f'Total Deaths (2020-{max_year})')
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Total Deaths')
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].grid(False)

    # Panel B: Total DALYs
    print(dalys_totals_mean)
    axes[1].bar(dalys_totals_mean.index, dalys_totals_mean.values, color=scenario_colours, yerr=dalys_totals_err,
                capsize=20)
    axes[1].set_title(f'Total DALYs (2020-{max_year})')
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('DALYs')
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].grid(False)
    fig.tight_layout()
    fig.savefig(output_folder / "total_deaths_and_dalys_all_draws.png")
    plt.close(fig)

    ### With causes
    # Panel A: Total Deaths
    axes[0].bar(df_deaths_all_draws_mean.index, df_deaths_all_draws_mean.values,
                capsize=20)
    axes[0].set_title(f'Deaths (2020-{max_year})')
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Total Deaths')
    axes[0].set_xticklabels(scenario_names, rotation=45)
    axes[0].grid(False)

    # Panel B: Total DALYs
    print(dalys_totals_mean)
    axes[1].bar(df_deaths_all_draws_mean.index, df_deaths_all_draws_mean.values,
                capsize=20)
    axes[1].set_title(f'DALYs (2020-{max_year})')
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('DALYs')
    axes[1].set_xticklabels(scenario_names, rotation=45)
    axes[1].grid(False)
    fig.tight_layout()
    fig.savefig(output_folder / "deaths_and_dalys_all_draws.png")
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
