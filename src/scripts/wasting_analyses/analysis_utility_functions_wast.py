"""
A helping file that contains functions used for wasting analyses to extract data, derive outcomes and generate plots.
It is not to be run by itself. Functions are called from run_interventions_analysis_wasting.py, and
heatmaps_cons_wast.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results

plt.style.use('seaborn-darkgrid')

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

def return_sum_95_CI_across_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with sum, lower CI, and upper CI for each draw across runs.
    The output DataFrame is structured with column index ['draw'], where each cell contains
    a list of [sum, lower_ci, upper_ci].
    """
    result = pd.DataFrame(index=['sum'], columns=df.columns.get_level_values('draw').unique())

    for draw in df.columns.get_level_values('draw').unique():
        draw_data = df.xs(draw, level='draw', axis=1).sum(axis=0)
        ci = st.t.interval(0.95, len(draw_data) - 1, loc=np.sum(draw_data), scale=st.sem(draw_data))
        result.at['sum', draw] = [np.sum(draw_data), ci[0], ci[1]]

    return result

def extract_birth_data_frames_and_outcomes(folder, years_of_interest, intervention_years, interv) -> dict:
    """
    :param folder: the folder from which the outcome data will be extracted
    :param years_of_interest: years for which we want to extract the data
    :param intervention_years: out of the years of interest, during which years was intervention implemented (if any)
    :return: dictionary with list of lists with means, lower_CI, and upper_CI for all years of interest of
        (1) total_births, (2) interv_births; and a dataframe (3) with births counts for years of interests as simulated
        for draws and runs.
    """

    print(f"\n{interv=}")

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
            'interv_births_df': interv_births_df,
            'interv_births_mean_ci_df': interv_births_per_year_per_draw_df}

def extract_death_data_frames_and_outcomes(folder, births_df, years_of_interest, intervention_years, interv):
    print(f"\n{interv=}")
    # ### NEONATAL MORTALITY
    # Extract all deaths occurring during the first 28 days of life
    # differentiated by cause of death
    neonatal_deaths_by_cause_df = extract_results(
        folder,
        module="tlo.methods.demography.detail",
        key="properties_of_deceased_persons",
        custom_generate_series=(
            lambda df: (filtered_by_age := df.loc[df['age_days'] < 29])
            .assign(year=filtered_by_age['date'].dt.year)
            .groupby(['year', 'cause_of_death'])['year']
            .count()
            .reindex(pd.MultiIndex.from_product([df['date'].dt.year.unique(), df['cause_of_death'].unique()],
                                                names=['year', 'cause_of_death']), fill_value=0)
        ),
        do_scaling=True).fillna(0)
    neonatal_deaths_by_cause_df = neonatal_deaths_by_cause_df.loc[years_of_interest]
    # number of deaths by any cause
    neonatal_deaths_df = neonatal_deaths_by_cause_df.groupby(['year']).sum()
    # number of deaths by specific causes
    neonatal_SAM_deaths_df = neonatal_deaths_by_cause_df.loc[
        neonatal_deaths_by_cause_df.index.get_level_values('cause_of_death') == 'Severe Acute Malnutrition'
        ].groupby(['year']).sum()
    neonatal_ALRI_deaths_df = neonatal_deaths_by_cause_df.loc[
        neonatal_deaths_by_cause_df.index.get_level_values('cause_of_death').str.startswith('ALRI_')
    ].groupby(['year']).sum()
    neonatal_Diarrhoea_deaths_df = neonatal_deaths_by_cause_df.loc[
        neonatal_deaths_by_cause_df.index.get_level_values('cause_of_death').str.startswith('Diarrhoea_')
    ].groupby(['year']).sum()

    neo_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(neonatal_deaths_df)
    neo_SAM_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(neonatal_SAM_deaths_df)
    neo_ALRI_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(neonatal_ALRI_deaths_df)
    neo_Diarrhoea_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(neonatal_Diarrhoea_deaths_df)

    interv_neo_deaths_df = neonatal_deaths_df.loc[intervention_years]
    interv_neo_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_neo_deaths_df)
    interv_neo_SAM_deaths_df = neonatal_SAM_deaths_df.loc[intervention_years]
    interv_neo_SAM_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_neo_SAM_deaths_df)
    interv_neo_ALRI_deaths_df = neonatal_ALRI_deaths_df.loc[intervention_years]
    interv_neo_ALRI_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_neo_ALRI_deaths_df)
    interv_neo_Diarrhoea_deaths_df = neonatal_Diarrhoea_deaths_df.loc[intervention_years]
    interv_neo_Diarrhoea_deaths_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_neo_Diarrhoea_deaths_df)

    # NEONATAL MORTALITY RATE (NMR), i.e. the number of deaths of infants up to 28 days old per 1,000 live births
    nmr_df = (neonatal_deaths_df / births_df) * 1000
    nmr_per_year_per_draw_df = return_mean_95_CI_across_runs(nmr_df)

    # ### UNDER-5 MORTALITY
    # Extract all deaths occurring during the first 5 years of life
    # differentiated by cause of death
    under5_deaths_by_cause_df = extract_results(
        folder,
        module="tlo.methods.demography.detail",
        key="properties_of_deceased_persons",
        custom_generate_series=(
            lambda df: (filtered_by_age := df.loc[df['age_exact_years'] < 5])
            .assign(year=filtered_by_age['date'].dt.year)
            .groupby(['year', 'cause_of_death'])['year']
            .count()
            .reindex(pd.MultiIndex.from_product([df['date'].dt.year.unique(), df['cause_of_death'].unique()],
                                                names=['year', 'cause_of_death']), fill_value=0)
        ),
        do_scaling=True).fillna(0)
    under5_deaths_by_cause_df = under5_deaths_by_cause_df.loc[years_of_interest]
    # number of deaths by any cause
    under5_deaths_df = under5_deaths_by_cause_df.groupby(['year']).sum()
    # number of deaths by specific causes
    under5_SAM_deaths_df = under5_deaths_by_cause_df.loc[
        under5_deaths_by_cause_df.index.get_level_values('cause_of_death') == 'Severe Acute Malnutrition'
        ].groupby(['year']).sum()
    under5_ALRI_deaths_df = under5_deaths_by_cause_df.loc[
        under5_deaths_by_cause_df.index.get_level_values('cause_of_death').str.startswith('ALRI_')
    ].groupby(['year']).sum()
    under5_Diarrhoea_deaths_df = under5_deaths_by_cause_df.loc[
        under5_deaths_by_cause_df.index.get_level_values('cause_of_death').str.startswith('Diarrhoea_')
    ].groupby(['year']).sum()

    under5_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(under5_deaths_df)
    under5_SAM_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(under5_SAM_deaths_df)
    under5_ALRI_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(under5_ALRI_deaths_df)
    under5_Diarrhoea_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(under5_Diarrhoea_deaths_df)

    interv_under5_deaths_df = under5_deaths_df.loc[intervention_years]
    interv_under5_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_under5_deaths_df)
    interv_under5_SAM_deaths_df = under5_SAM_deaths_df.loc[intervention_years]
    interv_under5_SAM_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_under5_SAM_deaths_df)
    interv_under5_ALRI_deaths_df = under5_ALRI_deaths_df.loc[intervention_years]
    interv_under5_ALRI_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(
        interv_under5_ALRI_deaths_df)
    interv_under5_Diarrhoea_deaths_df = under5_Diarrhoea_deaths_df.loc[intervention_years]
    interv_under5_Diarrhoea_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(
        interv_under5_Diarrhoea_deaths_df)

    # UNDER-5 MORTALITY RATE, i.e. the number of deaths of children under 5 years old per 1,000 live births
    under5mr_df = (under5_deaths_df / births_df) * 1000
    under5mr_per_year_per_draw_df = return_mean_95_CI_across_runs(under5mr_df)

    return {'neo_deaths_df': neonatal_deaths_df,
            'neo_SAM_deaths_df': neonatal_SAM_deaths_df,
            'neo_ALRI_deaths_df': neonatal_ALRI_deaths_df,
            'neo_Diarrhoea_deaths_df': neonatal_Diarrhoea_deaths_df,
            'neo_deaths_mean_ci_df': neo_deaths_mean_ci_per_year_per_draw_df,
            'neo_SAM_deaths_mean_ci_df': neo_SAM_deaths_mean_ci_per_year_per_draw_df,
            'neo_ALRI_deaths_mean_ci_df': neo_ALRI_deaths_mean_ci_per_year_per_draw_df,
            'neo_Diarrhoea_deaths_mean_ci_df': neo_Diarrhoea_deaths_mean_ci_per_year_per_draw_df,
            'interv_neo_deaths_df': interv_neo_deaths_df,
            'interv_neo_deaths_sum_ci_df': interv_neo_deaths_sum_per_draw_CI_across_runs_df,
            'interv_neo_SAM_deaths_df': interv_neo_SAM_deaths_df,
            'interv_neo_SAM_deaths_sum_ci_df': interv_neo_SAM_deaths_sum_per_draw_CI_across_runs_df,
            'interv_neo_ALRI_deaths_df': interv_neo_ALRI_deaths_df,
            'interv_neo_ALRI_deaths_sum_ci_df': interv_neo_ALRI_deaths_sum_per_draw_CI_across_runs_df,
            'interv_neo_Diarrhoea_deaths_df': interv_neo_Diarrhoea_deaths_df,
            'interv_neo_Diarrhoea_deaths_sum_ci_df': interv_neo_Diarrhoea_deaths_sum_per_draw_CI_across_runs_df,
            'neonatal_mort_rate_df': nmr_df,
            'neo_mort_rate_mean_ci_df': nmr_per_year_per_draw_df,
            'under5_deaths_df': under5_deaths_df,
            'under5_SAM_deaths_df': under5_SAM_deaths_df,
            'under5_ALRI_deaths_df': under5_ALRI_deaths_df,
            'under5_Diarrhoea_deaths_df': under5_Diarrhoea_deaths_df,
            'under5_deaths_mean_ci_df': under5_deaths_mean_ci_per_year_per_draw_df,
            'under5_SAM_deaths_mean_ci_df': under5_SAM_deaths_mean_ci_per_year_per_draw_df,
            'under5_ALRI_deaths_mean_ci_df': under5_ALRI_deaths_mean_ci_per_year_per_draw_df,
            'under5_Diarrhoea_deaths_mean_ci_df': under5_Diarrhoea_deaths_mean_ci_per_year_per_draw_df,
            'interv_under5_deaths_df': interv_under5_deaths_df,
            'interv_under5_deaths_sum_ci_df': interv_under5_deaths_sum_per_draw_CI_across_runs_df,
            'interv_under5_SAM_deaths_df': interv_under5_SAM_deaths_df,
            'interv_under5_SAM_deaths_sum_ci_df': interv_under5_SAM_deaths_sum_per_draw_CI_across_runs_df,
            'interv_under5_ALRI_deaths_df': interv_under5_ALRI_deaths_df,
            'interv_under5_ALRI_deaths_sum_ci_df': interv_under5_ALRI_deaths_sum_per_draw_CI_across_runs_df,
            'interv_under5_Diarrhoea_deaths_df': interv_under5_Diarrhoea_deaths_df,
            'interv_under5_Diarrhoea_deaths_sum_ci_df': interv_under5_Diarrhoea_deaths_sum_per_draw_CI_across_runs_df,
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
        'FS_50': '#D4898E',
        'FS_70': '#D4898E',
        'FS_Full': '#A90251'
    }.get(scen_name)

def plot_mortality_rate__by_interv_multiple_settings(cohort: str, interv_timestamps_dict: dict, scenarios_dict: dict,
                                                     intervs_of_interest: list, plot_years: list, outcomes_dict: dict,
                                                     outputs_path: Path) -> None:

    def plot_scenarios(plot_interv, plot_outcome):
        scenarios_to_plot = scenarios_dict[plot_interv]
        for scen_name, draw in scenarios_to_plot.items():
            scen_colour = get_scen_colour(scen_name)
            scen_data = outcomes_dict[plot_interv][plot_outcome][draw]

            means, ci_lower, ci_upper = zip(*scen_data.values.flatten())

            ax.plot(plot_years, means, label=scen_name, color=scen_colour)
            ax.fill_between(plot_years, ci_lower, ci_upper,
                            color=scen_colour, alpha=.1)

    # Outcome to plot, corresponding target, and y-axis limit
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

    # Plots by intervention (multiple settings within each plot)
    for interv in intervs_of_interest:

        fig, ax = plt.subplots()
        plot_scenarios(interv, outcome)

        if interv == 'SQ':

            # Add UNICEF mortality rates data
            unicef_neo_mort_rates = [27.207, 26.392, 25.463, 24.55, 23.749, 23.039, 22.392, 21.786, 21.253, 20.769,
                                     20.261, 19.778, 19.326, 18.825]
            unicef_under5_mort_rates = [83.027, 76.687, 69.793, 63.998, 60.224, 56.708, 53.503, 50.647, 47.524, 45.456,
                                        43.086, 41.527, 39.94, 38.34]
            unicef_years = list(range(2010, 2024))
            unicef_colour = '#1CABE2'

            # Add WPP 2024 mortality rates estimates (past data) and medium projection variant (future predictions)
            # only for under-5
            wpp_medium_under5_mort_rates = [
                81, 76, 70, 65, 61, 57, 53, 51, 48, 46, 44, 42, 41, 39, 38, 37, 37, 36, 35, 34, 33, 33, 32, 31, 30, 30,
                29, 28, 27, 27, 26, 26, 25, 24, 24, 23, 23, 22, 22, 21, 21, 21, 20, 20, 19, 19, 19, 18, 18, 17, 17, 17,
                16, 16, 16, 16, 16, 15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12,
                12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11
            ]

            wpp_years = list(range(2010, 2101))
            wpp_colour = '#1D73F5'

            # Filter data to include only years present in both plot_years and source years
            unicef_filtered_years = [year for year in plot_years if year in unicef_years]
            unicef_filtered_neo_rates = [rate for year, rate in zip(unicef_years, unicef_neo_mort_rates) if year in unicef_filtered_years]
            unicef_filtered_under5_rates = [rate for year, rate in zip(unicef_years, unicef_under5_mort_rates) if year in unicef_filtered_years]

            wpp_filtered_years = [year for year in plot_years if year in wpp_years]
            wpp_filtered_under5_rates = [rate for year, rate in zip(wpp_years, wpp_medium_under5_mort_rates) if year in wpp_filtered_years]

            if cohort == 'Neonatal':
                    ax.plot(unicef_filtered_years, unicef_filtered_neo_rates, label='UNICEF Data', color=unicef_colour, linestyle='--')
            elif cohort == 'Under-5':
                ax.plot(unicef_filtered_years, unicef_filtered_under5_rates, label='UNICEF Data', color=unicef_colour, linestyle='--')
                ax.plot(wpp_filtered_years, wpp_filtered_under5_rates, label='WPP 2024', color=wpp_colour, linestyle='-.')
        else:
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

        # Save plot as PNG
        if interv == 'SQ':
            plt.savefig(
                outputs_path / f"{cohort}_mort_rate_{interv}_UNICEF_WPP__"
                               f"{interv_timestamps_dict[interv]}.png",
                bbox_inches='tight'
            )

        else:
            plt.savefig(
                outputs_path / f"{cohort}_mort_rate_{interv}_multiple_settings__"
                               f"{interv_timestamps_dict[interv]}__{interv_timestamps_dict['SQ']}.png",
                bbox_inches='tight'
            )

def plot_mean_deaths_and_CIs__scenarios_comparison(cohort: str, scenarios_dict: dict, scenarios_to_compare: list,
                                                   plot_years: list, outcomes_dict: dict, outputs_path: Path,
                                                   scenarios_tocompare_prefix, timestamps_suffix: str) -> None:
    """
    Plots mean deaths and confidence intervals over time for the specified cohort for multiple scenarios.
    :param cohort: 'Neonatal' or 'Under-5'
    :param scenarios_dict: Dictionary mapping interventions to scenarios and their corresponding draw numbers
    :param scenarios_to_compare: List of scenarios to plot
    :param plot_years: List of years to plot
    :param outcomes_dict: Dictionary containing data for plotting nested as outcomes_dict[interv][outcome][draw][run]
    :param outputs_path: Path to save the plot
    :param scenarios_tocompare_prefix: Prefix for output files with names of scenarios that are compared in the plots
    :param timestamps_suffix: Timestamps to identify the log data from which the outcomes originated.
    """
    # Outcome to plot
    assert cohort in ['Neonatal', 'Under-5'], \
        f"Invalid value for 'cohort': expected 'Neonatal' or 'Under-5'. Received {cohort} instead."

    for i, cause_of_death in enumerate(['any cause', 'SAM', 'ALRI', 'Diarrhoea']):

        neonatal_outcomes = ['neo_deaths_mean_ci_df', 'neo_SAM_deaths_mean_ci_df',
                             'neo_ALRI_deaths_mean_ci_df', 'neo_Diarrhoea_deaths_mean_ci_df']
        under5_outcomes = ['under5_deaths_mean_ci_df', 'under5_SAM_deaths_mean_ci_df',
                           'under5_ALRI_deaths_mean_ci_df', 'under5_Diarrhoea_deaths_mean_ci_df']
        outcome = neonatal_outcomes[i] if cohort == 'Neonatal' else under5_outcomes[i]

        # Initialize the plot
        fig, ax = plt.subplots()

        # Iterate over scenarios to compare
        for scenario in scenarios_to_compare:
            # Find the corresponding intervention and draw number
            interv, draw = next(
                (interv, draw)
                for interv, scenarios_for_interv_dict in scenarios_dict.items()
                if scenario in scenarios_for_interv_dict
                for scen_name, draw in scenarios_for_interv_dict.items()
                if scen_name == scenario
            )

            # Extract data for the scenario
            scen_data = outcomes_dict[interv][outcome][draw]

            # Calculate means and confidence intervals
            means, ci_lower, ci_upper = zip(*scen_data.values.flatten())

            # Plot the data
            ax.plot(plot_years, means, label=scenario, color=get_scen_colour(scenario))
            ax.fill_between(plot_years, ci_lower, ci_upper, color=get_scen_colour(scenario), alpha=0.2)

        # Add labels, title, and legend
        plt.ylabel(f'{cohort} Deaths')
        plt.xlabel('Year')
        plt.title(f'{cohort} Mean deaths due to {cause_of_death} and 95% CI over time')
        plt.legend()
        plt.xticks(plot_years, labels=plot_years, rotation=45, fontsize=8)

        # Save the plot
        plt.savefig(
            outputs_path / (
                f"{cohort}_mean_{cause_of_death}_deaths_CI_scenarios_comparison__"
                f"{scenarios_tocompare_prefix}__{timestamps_suffix}.png"
            ),
            bbox_inches='tight'
        )

def plot_sum_deaths_and_CIs__intervention_period(cohort: str, scenarios_dict: dict, scenarios_to_compare: list,
                                                 outcomes_dict: dict, outputs_path: Path, scenarios_tocompare_prefix,
                                                 timestamps_suffix: str) -> None:
    """
    Plots sum of deaths and confidence intervals over the intervention period for the specified cohort for multiple
    scenarios.
    :param cohort: 'Neonatal' or 'Under-5'
    :param scenarios_dict: Dictionary mapping interventions to scenarios and their corresponding draw numbers
    :param scenarios_to_compare: List of scenarios to plot
    :param outcomes_dict: Dictionary containing data for plotting nested as outcomes_dict[interv][outcome][draw][run]
    :param outputs_path: Path to save the plot
    :param scenarios_tocompare_prefix: Prefix for output files with names of scenarios that are compared in the plots
    :param timestamps_suffix: Timestamps to identify the log data from which the outcomes originated.
    """
    # Outcome to plot
    assert cohort in ['Neonatal', 'Under-5'], \
        f"Invalid value for 'cohort': expected 'Neonatal' or 'Under-5'. Received {cohort} instead."

    for i, cause_of_death in enumerate(['any cause', 'SAM', 'ALRI', 'Diarrhoea']):

        neonatal_outcomes = ['interv_neo_deaths_sum_ci_df', 'interv_neo_SAM_deaths_sum_ci_df',
                             'interv_neo_ALRI_deaths_sum_ci_df', 'interv_neo_Diarrhoea_deaths_sum_ci_df']
        under5_outcomes = ['interv_under5_deaths_sum_ci_df', 'interv_under5_SAM_deaths_sum_ci_df',
                           'interv_under5_ALRI_deaths_sum_ci_df', 'interv_under5_Diarrhoea_deaths_sum_ci_df']
        outcome = neonatal_outcomes[i] if cohort == 'Neonatal' else under5_outcomes[i]

        # Initialize the plot
        fig, ax = plt.subplots()

        # Iterate over scenarios to compare
        for scenario in scenarios_to_compare:
            # Find the corresponding intervention and draw number
            interv, draw = next(
                (interv, draw)
                for interv, scenarios_for_interv_dict in scenarios_dict.items()
                if scenario in scenarios_for_interv_dict
                for scen_name, draw in scenarios_for_interv_dict.items()
                if scen_name == scenario
            )

            # Extract data for the scenario
            scen_data = outcomes_dict[interv][outcome][draw]

            # Calculate sum and confidence intervals
            sums, ci_lower, ci_upper = zip(*scen_data.values.flatten())

            # Plot the data
            ax.bar(scenario, sums[0], yerr=[[sums[0] - ci_lower[0]], [ci_upper[0] - sums[0]]],
                   label=scenario, color=get_scen_colour(scenario), capsize=5)

            y_top = ax.get_ylim()[1]

            # Add text labels for ci_low and ci_upper
            text_color = 'white' if scenario in ['CS_100', 'FS_Full'] else 'black'
            ax.text(scenario,
                    ci_upper[0] / 2 + ci_upper[0] / 4 if ci_upper < y_top / 2 + y_top / 15 else y_top / 2 + y_top / 15,
                    f"{ci_upper[0]:,.2f}", color=text_color, ha='center', va='top', fontsize=12)
            ax.text(scenario,
                    ci_upper[0] / 2 - ci_upper[0] / 4 if ci_upper < y_top / 2 + y_top / 15 else y_top / 2 - y_top / 15,
                    f"{ci_lower[0]:,.2f}", color=text_color, ha='center', va='bottom', fontsize=12)

            # Add horizontal lines for Status Quo scenario
            if scenario == 'Status Quo':
                ax.axhline(y=ci_lower[0], color=get_scen_colour('Status Quo'), linestyle='--', linewidth=1)
                ax.axhline(y=ci_upper[0], color=get_scen_colour('Status Quo'), linestyle='--', linewidth=1)


        # Add labels, title, and legend
        min_interv_year = outcomes_dict["SQ"]["interv_under5_deaths_df"].index.min()
        max_interv_year = outcomes_dict["SQ"]["interv_under5_deaths_df"].index.max()
        plt.ylabel(f'{cohort} Deaths (Sum over intervention period)')
        plt.xlabel('Scenario')
        plt.title(
            f'{cohort} Sum of deaths due to {cause_of_death} and 95% CI over intervention period '
            f'({min_interv_year}--{max_interv_year})')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(rotation=45, fontsize=8)

        # Save the plot
        plt.savefig(
            outputs_path / (
                f"{cohort}_sum_{cause_of_death}_deaths_CI_intervention_period_scenarios_comparison__"
                f"{scenarios_tocompare_prefix}__{timestamps_suffix}.png"
            ),
            bbox_inches='tight'
        )
# ----------------------------------------------------------------------------------------------------------------------
def plot_availability_heatmaps(outputs_path: Path) -> None:
    """
    Plots availability of
        * essential consumables,
        * treatments (i.e., probability of all consumables essential for the treatment being available)
    :param outputs_path: path where to save the plots as PNG files
    :return:
    """
    resourcefilepath = Path("./resources")

    tlo_availability_df = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'consumables' / "ResourceFile_Consumables_availability_all.csv")

    # Master Facilities List (district, facility level, region, facility id, and facility name)
    mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
    tlo_availability_df = tlo_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                                                    on=['Facility_ID'], how='left')

    # fac_levels = {'0': 'Health Post', '1a': 'Health Centers', '1b': 'Rural/Community \n Hospitals',
    #               '2': 'District Hospitals', '3': 'Central Hospitals', '4': 'Mental Hospital'}
    correct_order_of_fac_levels = ['0', '1a', '1b', '2', '3', '4']
    chosen_item_codes = [1220, 1227, 208]
    item_names_to_map = {1220:'F-75\ntherapeutic\nmilk', 1227:'RUTF', 208:'CSB++*'}

    tlo_availability_df = tlo_availability_df[tlo_availability_df.Facility_Level.isin(correct_order_of_fac_levels)]

    # HEATMAP OF CONSUMABLES AVAILABILITY
    # ###
    # Pivot the DataFrame
    aggregated_df = tlo_availability_df.groupby(['Facility_Level', 'item_code'])[['available_prop']].mean().reset_index()
    heatmap_data = aggregated_df.pivot(columns='Facility_Level', index='item_code', values='available_prop')
    # Keep chosen items
    heatmap_data = heatmap_data.loc[chosen_item_codes]
    # Add average column (availability across all facility levels)
    aggregate_col = aggregated_df.groupby('item_code')[['available_prop']].mean()
    # Order the facility levels
    heatmap_data = heatmap_data.reindex(columns=correct_order_of_fac_levels)
    heatmap_data['Average'] = aggregate_col
    # Map item codes to names
    heatmap_data.index = heatmap_data.index.map(item_names_to_map)

    # Generate the heatmap
    sns.set_theme(font_scale=1.5)
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn',
                cbar_kws={'label': 'Proportion of days on which consumable is available'})

    plt.title('Availability of essential consumables\n for acute malnutrition treatments', fontweight='bold')
    plt.xlabel('Facility Level')
    plt.ylabel('Consumable')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(outputs_path / 'consumable_availability_heatmap.png', dpi=300, bbox_inches='tight')

    # HEATMAP OF TREATMENTS AVAILABILITY
    # ###
    treatment_item_map = {
        'ITC': ['F-75\ntherapeutic\nmilk', 'RUTF'],  # 1220, 1227
        'OTP': ['RUTF'],        # 1227
        'SFP': ['CSB++*']          # 208
    }

    # Calculate availability for treatments
    treatment_availability = {}
    for treatment, items in treatment_item_map.items():
        treatment_availability[treatment] = {
            level: np.prod([heatmap_data.loc[item_code, level] for item_code in items])
            for level in correct_order_of_fac_levels
        }

    # Prepare the DataFrame
    treatment_heatmap_data = pd.DataFrame.from_dict(treatment_availability, orient='index', columns=correct_order_of_fac_levels)
    treatment_heatmap_data = treatment_heatmap_data.reindex(columns=correct_order_of_fac_levels)
    treatment_heatmap_data['Average'] = treatment_heatmap_data.mean(axis=1)

    # Generate the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(treatment_heatmap_data, annot=True, cmap='RdYlGn',
                cbar_kws={'label': 'Proportion of days on which treatment is available'})

    plt.title('Availability of treatments\n for acute malnutrition', fontweight='bold')
    plt.xlabel('Facility Level')
    plt.ylabel('Treatment')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(outputs_path / 'treatment_availability_heatmap.png', dpi=300, bbox_inches='tight')
