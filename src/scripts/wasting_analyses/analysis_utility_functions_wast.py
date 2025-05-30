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
    births_df = births_df.loc[years_of_interest[0]:years_of_interest[-1]]

    births_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(births_df)

    interv_births_df = births_df.loc[intervention_years[0]:intervention_years[-1]]
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
            lambda df: df.loc[(df['age_days'] < 29)].assign(
                year=df['date'].dt.year).groupby(['year'])['year'].count()),
        do_scaling=True).fillna(0)
    neonatal_deaths_df = neonatal_deaths_df.loc[years_of_interest[0]:years_of_interest[-1]]

    neo_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(neonatal_deaths_df)

    interv_neo_deaths_df = neonatal_deaths_df.loc[intervention_years[0]:intervention_years[-1]]
    interv_neo_deaths_per_year_per_draw_df = return_mean_95_CI_across_runs(interv_neo_deaths_df)

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
            lambda df: df.loc[(df['age_exact_years'] < 5)].assign(
                year=df['date'].dt.year).groupby(['year'])['year'].count()),
        do_scaling=True).fillna(0)
    under5_deaths_df = under5_deaths_df.loc[years_of_interest[0]:years_of_interest[-1]]

    under5_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(under5_deaths_df)

    interv_under5_deaths_df = under5_deaths_df.loc[intervention_years[0]:intervention_years[-1]]
    interv_under5_deaths_per_year_per_draw_df = return_mean_95_CI_across_runs(interv_under5_deaths_df)

    # NEONATAL MORTALITY RATE (NMR), i.e. the number of deaths of infants under 28 days old per 1,000 live births
    nmr_df = (neonatal_deaths_df / births_df) * 1000
    nmr_per_year_per_draw_df = return_mean_95_CI_across_runs(nmr_df)

    return {'neo_deaths_df': neonatal_deaths_df,
            'neo_deaths_mean_ci_df': neo_deaths_mean_ci_per_year_per_draw_df,
            'interv_neo_deaths_df': interv_neo_deaths_df, # TODO: check when and how is this used, is it really worth to return this?
            'interv_neo_deaths_mean_ci_df': interv_neo_deaths_per_year_per_draw_df,
            'under5_deaths_df': under5_deaths_df,
            'under5_deaths_mean_ci_df': under5_deaths_mean_ci_per_year_per_draw_df,
            'interv_under5_deaths_df': interv_under5_deaths_df,
            'interv_under5_deaths_mean_ci_df': interv_under5_deaths_per_year_per_draw_df,
            'neonatal_mort_rate_df': nmr_df,
            'neo_mort_rate_mean_ci_df': nmr_per_year_per_draw_df}

    # # TODO: rm prints when no longer needed
    # print("\nYears, and (Draws, Runs) with no under 5 death:")
    # no_under5_deaths = [(under5_deaths.index[row], under5_deaths.columns[col]) for row, col in
    #                  zip(*np.where(under5_deaths == 0.0))]
    # print(f"{no_under5_deaths}")
    # #

    # def plot_neonatal_mortality_multiple_scenario():
