from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results

plt.style.use('seaborn-darkgrid')

"""This file contains functions used for wasting analyses to extract data, derive outcomes and generate plots.
(It is based on analysis_utility_function prepared by Joe for maternal_perinatal_analyses.)"""

def return_mean_95_CI_across_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with mean, lower CI, and upper CI for each year and each draw across runs.
    The output DataFrame is structured with row index ['year'] and column index ['draw'], where each cell contains
    a list of [mean, lower_ci, upper_ci].
    """
    result = pd.DataFrame(index=df.index, columns=df.columns.get_level_values('draw').unique())

    for year in df.index:
        row = df.loc[year]
        ci = row.groupby(level='draw').apply(
            lambda x: st.t.interval(0.95, len(x) - 1, loc=np.mean(x), scale=st.sem(x))
        )
        result.loc[year] = row.groupby(level='draw').mean().combine(
            ci, lambda mean, ci_interval: [mean, ci_interval[0], ci_interval[1]]
        )

    return result

def extract_birth_data_frames_and_outcomes(folder, years_of_interest, intervention_years) -> dict:
    """
    :param folder: the folder from which the outcome data will be extracted
    :param years_of_interest: years for which we want to extract the data
    :param intervention_years: out of the years of interest, during which years was intervention implemented (if any)
    :return: dictionary with list of lists with means, lower_CI, and upper_CI for all years of interest of
        (1) total_births, (2) interv_births; and a dataframe (3) with births counts for years of interests as simulated
        for draws and runs.
    """

    births_df = extract_results(
        folder,
        module="tlo.methods.demography",
        key="on_birth",
        custom_generate_series=(
            lambda df: df.assign(
                year=df['date'].dt.year).groupby(['year'])['year'].count()),
        do_scaling=True
    ).fillna(0)
    births_df = births_df.loc[years_of_interest]

    births_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(births_df)

    interv_births_df = births_df.loc[intervention_years]
    interv_births_per_year_per_draw_df = return_mean_95_CI_across_runs(interv_births_df)

    return {'births_df': births_df,
            'births_mean_ci_df': births_mean_ci_per_year_per_draw_df,
            'interv_births_df': interv_births_df, # TODO: check when and how is this used, is it really worth to return this?
            'interv_births_mean_ci_df': interv_births_per_year_per_draw_df}

def extract_death_data_frames_and_outcomes(folder, births_df, years_of_interest, intervention_years):
    # ### NEONATAL MORTALITY
    # Extract all deaths occurring during the first 28 days of life
    neonatal_deaths_df = extract_results(
        folder,
        module="tlo.methods.demography.detail",
        key="properties_of_deceased_persons",
        custom_generate_series=(
            lambda df: (filtered_by_age := df.loc[df['age_days'] < 29])
            .assign(year=filtered_by_age['date'].dt.year)
            .groupby(['year'])['year']
            .count()
            .reindex(df['date'].dt.year.unique(), fill_value=0)
        ),
        do_scaling=True).fillna(0)
    neonatal_deaths_df = neonatal_deaths_df.loc[years_of_interest]

    neo_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(neonatal_deaths_df)

    interv_neo_deaths_df = neonatal_deaths_df.loc[intervention_years]
    interv_neo_deaths_per_year_per_draw_df = return_mean_95_CI_across_runs(interv_neo_deaths_df)

    # NEONATAL MORTALITY RATE (NMR), i.e. the number of deaths of infants up to 28 days old per 1,000 live births
    nmr_df = (neonatal_deaths_df / births_df) * 1000
    nmr_per_year_per_draw_df = return_mean_95_CI_across_runs(nmr_df)

    # # TODO: rm prints when no longer needed
    # print("\nYears, and (Draws, Runs) with no neonatal death:")
    # no_neo_deaths = [(neonatal_deaths.index[row], neonatal_deaths.columns[col]) for row, col in
    #                  zip(*np.where(neonatal_deaths == 0.0))]
    # print(f"{no_neo_deaths}")
    # #

    # ### UNDER-5 MORTALITY
    # Extract all deaths occurring during the first 5 years of life
    under5_deaths_df = extract_results(
        folder,
        module="tlo.methods.demography.detail",
        key="properties_of_deceased_persons",
        custom_generate_series=(
            lambda df: (filtered_by_age := df.loc[df['age_exact_years'] < 5])
            .assign(year=filtered_by_age['date'].dt.year)
            .groupby(['year'])['year']
            .count()
            .reindex(df['date'].dt.year.unique(), fill_value=0)
        ),
        do_scaling = True).fillna(0)
    under5_deaths_df = under5_deaths_df.loc[years_of_interest]

    under5_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(under5_deaths_df)

    interv_under5_deaths_df = under5_deaths_df.loc[intervention_years]
    interv_under5_deaths_per_year_per_draw_df = return_mean_95_CI_across_runs(interv_under5_deaths_df)

    # UNDER-5 MORTALITY RATE, i.e. the number of deaths of children under 5 years old per 1,000 live births
    under5mr_df = (under5_deaths_df / births_df) * 1000
    under5mr_per_year_per_draw_df = return_mean_95_CI_across_runs(under5mr_df)

    return {'neo_deaths_df': neonatal_deaths_df,
            'neo_deaths_mean_ci_df': neo_deaths_mean_ci_per_year_per_draw_df,
            'interv_neo_deaths_df': interv_neo_deaths_df, # TODO: check when and how is this used, is it really worth to return this?
            'interv_neo_deaths_mean_ci_df': interv_neo_deaths_per_year_per_draw_df,
            'neonatal_mort_rate_df': nmr_df,
            'neo_mort_rate_mean_ci_df': nmr_per_year_per_draw_df,
            'under5_deaths_df': under5_deaths_df,
            'under5_deaths_mean_ci_df': under5_deaths_mean_ci_per_year_per_draw_df,
            'interv_under5_deaths_df': interv_under5_deaths_df,
            'interv_under5_deaths_mean_ci_df': interv_under5_deaths_per_year_per_draw_df,
            'under5_mort_rate_df': under5mr_df,
            'under5_mort_rate_mean_ci_df': under5mr_per_year_per_draw_df}

    # # TODO: rm prints when no longer needed
    # print("\nYears, and (Draws, Runs) with no under 5 death:")
    # no_under5_deaths = [(under5_deaths.index[row], under5_deaths.columns[col]) for row, col in
    #                  zip(*np.where(under5_deaths == 0.0))]
    # print(f"{no_under5_deaths}")
    # #

def get_scen_colour(scen_name: str) -> str:
    return {
        'Status Quo': '#F12AE5',
        'GM_FullAttend': '#4575B4',
        'GM_all': '#BDEBF7',
        'GM_1-2': '#91BFDB',
        'CS_10': '#9FFD17',
        'CS_30': '#61B93C',
        'CS_50': '#2D945F',
        'CS_100': '#266714',
        'FS_full': '#A90251',
        'FS_plus10': '#D4898E'
    }.get(scen_name)

def plot_mortality__by_interv_multiple_settings(cohort: str, interv_timestamps_dict: dict, scenarios_dict: dict,
                                                intervs_of_interest: list, plot_years: list, data_dict: dict,
                                                outputs_path: Path) -> None:

    def plot_scenarios(plot_interv, plot_outcome):
        scenarios_to_plot = scenarios_dict[plot_interv]
        for scen_name, draw in scenarios_to_plot.items():
            scen_colour = get_scen_colour(scen_name)
            scen_data = data_dict[plot_interv][plot_outcome][draw]

            means, ci_lower, ci_upper = zip(*scen_data.values.flatten())

            ax.plot(plot_years, means, label=scen_name, color=scen_colour)
            ax.fill_between(plot_years, ci_lower, ci_upper,
                            color=scen_colour, alpha=.1)

    # outcome to plot
    assert cohort in ['Neonatal', 'Under-5'],\
        f"Invalid value for 'cohort': expected 'Neonatal' or 'Under-5'. Received {cohort} instead."

    if cohort == 'Neonatal':
        outcome = 'neo_mort_rate_mean_ci_df'
        target = 12
        ylim_top = 40 #25
    else: #cohort == 'Under-5':
        outcome = 'under5_mort_rate_mean_ci_df'
        target = 25
        ylim_top = 100 #60

    for interv in intervs_of_interest:

        fig, ax = plt.subplots()
        plot_scenarios(interv, outcome)
        plot_scenarios('SQ', outcome)

        plt.axhline(y=target, color='black', linestyle='--', linewidth=1)
        plt.text(x=plot_years[-1] + 1, y=target, s='SDG\n3.2 target', color='black', va='center', ha='left', fontsize=8)
        plt.text(x=plot_years[0] - 1, y=target, s=target, color='black', va='center', ha='right', fontsize=8)
        plt.ylabel(f'{cohort} Deaths per 1,000 Live Births')
        plt.xlabel('Year')
        plt.title(f'{cohort} Mortality Rate: multiple settings of {interv} intervention')

        plt.gca().set_ylim(bottom=0, top=ylim_top)

        plt.legend()
        plt.xticks(plot_years, labels=plot_years, rotation=45, fontsize=8)

        # TODO: extract from iterv_folders_dict[interv] the suffix after the prefix (scenario_filename_prefix = 'wasting_analysis__minimal_model')
        plt.savefig(
            outputs_path / f"{cohort}_mort_rate_{interv}_multiple_settings__"
                           f"{interv_timestamps_dict[interv]}__{interv_timestamps_dict['SQ']}.png",
            bbox_inches='tight'
        )
        plt.show()
