"""
A helping file that contains functions used for wasting analyses to extract data, derive outcomes and generate plots.
It is not to be run by itself. Functions are called from run_interventions_analysis_wasting.py, and
heatmaps_cons_wast.py.
"""

import logging
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from matplotlib import lines as mpl_lines
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image
from run_costing_analysis_wast import run_costing_analysis_wast as run_costing

from src.scripts.costing.cost_estimation import apply_discounting_to_cost_data
from tlo.analysis.utils import create_pickles_locally, extract_results

plt.style.use('seaborn-darkgrid')

scenario_label_map = {
    "SQ": "Status Quo",
    "GM": "Growth Monitoring",
    "CS": "Care-Seeking",
    "FS": "Food Supplements",
    "GM_CS": "Growth Monitoring and\nCare-Seeking",
    "GM_FS": "Growth Monitoring and\nFood Supplements",
    "CS_FS": "Care-Seeking and\nFood Supplements",
    "GM_CS_FS": "Growth Monitoring and\nCare-Seeking and\nFood Supplements",
}

def get_scenario_label(scen_abbr: str) -> str:
    """Return display label for a scenario code; fall back to code if unknown."""
    return scenario_label_map.get(scen_abbr, scen_abbr)

def map_scenario_labels(scen_abbrs_list: list) -> list:
    """Map a list of scenario abbreviations to their labels to be displayed in figs."""
    return [get_scenario_label(s) for s in scen_abbrs_list]

def apply_millions_formatter_to_ax(ax, axis: str = 'x, y', x_decimals: int = 1, y_decimals: int = 1):
    if 'x' in axis:
        x_fmt = FuncFormatter(lambda v, pos: f"{v/1e6:,.{x_decimals}f}" if v != 0 else "0")
        ax.xaxis.set_major_formatter(x_fmt)
    if 'y' in axis:
        y_fmt = FuncFormatter(lambda v, pos: f"{v/1e6:,.{y_decimals}f}" if v != 0 else "0")
        ax.yaxis.set_major_formatter(y_fmt)

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
    Returns a DataFrame with sum, lower CI, and upper CI for each draw, mean across runs.
    The output DataFrame is structured with column index ['draw'], where each cell contains
    a list of [sum, lower_ci, upper_ci].
    """
    result = pd.DataFrame(index=['sum'], columns=df.columns.get_level_values('draw').unique())

    for draw in df.columns.get_level_values('draw').unique():
        draw_data = df.xs(draw, level='draw', axis=1).sum(axis=0)
        ci = st.t.interval(0.95, len(draw_data) - 1, loc=np.mean(draw_data), scale=st.sem(draw_data))
        result.at['sum', draw] = [np.mean(draw_data), ci[0], ci[1]]

    return result

def extract_birth_data_frames_and_outcomes(
    folder,
    years_of_interest,
    intervention_datayears,
    interv
) -> Dict[str, pd.DataFrame]:
    """
    Extracts and summarizes birth data.

    :param folder: Path to the folder containing outcome data.
    :param years_of_interest: List of years to extract data for.
    :param intervention_datayears: List of years for which we need data to plot means over the interventions years, ie
        from the year before interventions are implemented until the last year of interventions.
    :param interv: Name or identifier of the intervention.
    :return: Dictionary with DataFrames:
            (1) 'births_df': Birth counts for years of interest (by draw and run),
            (2) 'births_mean_ci_df': Mean and 95% CI for total births per year and draw,
            (3) 'interv_births_df': Birth counts for intervention years,
            (4) 'interv_births_mean_ci_df': Mean and 95% CI for births per year and draw for intervention_datayears.
    """

    print(f"    -{interv=}")

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

    interv_births_df = births_df.loc[intervention_datayears]
    interv_births_per_year_per_draw_df = return_mean_95_CI_across_runs(interv_births_df)

    # report during which years interventions were implemented (if any)
    interv_years = [year+1 for year in intervention_datayears[:-1]]

    return {'births_df': births_df,
            'births_mean_ci_df': births_mean_ci_per_year_per_draw_df,
            'interv_births_df': interv_births_df,
            'interv_births_mean_ci_df': interv_births_per_year_per_draw_df,
            'interv_years': interv_years}

def extract_death_data_frames_and_outcomes(
    folder,
    births_df,
    years_of_interest,
    intervention_years,
    interv
) -> Dict[str, pd.DataFrame]:
    """
    Extracts and summarizes death data (neonatal and under-5) by cause, year, and intervention period.

    :param folder: Path to the folder containing outcome data.
    :param births_df: DataFrame of births for the years of interest.
    :param years_of_interest: List of years to extract data for.
    :param intervention_years: List of years during which the intervention was implemented (if any).
    :param interv: Name or identifier of the intervention.
    :return: Dictionary with DataFrames for deaths by cause, mean and CI, and mortality rates
        for both neonatal and under-5 cohorts.
    """

    print(f"    -{interv=}")
    # ### NEONATAL MORTALITY
    # Extract all deaths occurring during the first 28 days of life
    # differentiated by cause of death and acute malnutrition state
    neonatal_deaths_by_cause_am_df = extract_results(
        folder,
        module="tlo.methods.demography.detail",
        key="properties_of_deceased_persons",
        custom_generate_series=(
            lambda df: (filtered_by_age := df.loc[df['age_days'] < 29])
            .assign(year=filtered_by_age['date'].dt.year)
            .groupby(['year', 'cause_of_death', 'un_clinical_acute_malnutrition'])['year']
            .count()
            .reindex(pd.MultiIndex.from_product([
                df['date'].dt.year.unique(), df['cause_of_death'].unique(),
                df['un_clinical_acute_malnutrition'].unique()
            ], names=['year', 'cause_of_death', 'un_clinical_acute_malnutrition']), fill_value=0)
        ),
        do_scaling=True).fillna(0)
    neonatal_deaths_by_cause_am_df = neonatal_deaths_by_cause_am_df.loc[years_of_interest]

    # number of deaths by any cause
    neonatal_deaths_df = neonatal_deaths_by_cause_am_df.groupby(['year']).sum()
    # number of deaths due to specific cause
    neonatal_SAM_deaths_df = neonatal_deaths_by_cause_am_df.loc[
        neonatal_deaths_by_cause_am_df.index.get_level_values('cause_of_death') == 'SevereAcuteMalnutrition'
        ].groupby(['year']).sum()
    neonatal_ALRI_deaths_df = neonatal_deaths_by_cause_am_df.loc[
        neonatal_deaths_by_cause_am_df.index.get_level_values('cause_of_death').str.startswith('ALRI_')
    ].groupby(['year']).sum()
    neonatal_Diarrhoea_deaths_df = neonatal_deaths_by_cause_am_df.loc[
        neonatal_deaths_by_cause_am_df.index.get_level_values('cause_of_death').str.startswith('Diarrhoea_')
    ].groupby(['year']).sum()
    # number of deaths due to specific cause with SAM
    neonatal_ALRI_deaths_with_SAM_df = neonatal_deaths_by_cause_am_df.loc[
        (neonatal_deaths_by_cause_am_df.index.get_level_values('un_clinical_acute_malnutrition') == 'SAM') &
        (neonatal_deaths_by_cause_am_df.index.get_level_values('cause_of_death').str.startswith('ALRI_'))
        ].groupby(['year']).sum()
    neonatal_Diarrhoea_deaths_with_SAM_df = neonatal_deaths_by_cause_am_df.loc[
        (neonatal_deaths_by_cause_am_df.index.get_level_values('un_clinical_acute_malnutrition') == 'SAM') &
        (neonatal_deaths_by_cause_am_df.index.get_level_values('cause_of_death').str.startswith('Diarrhoea_'))
        ].groupby(['year']).sum()

    neo_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(neonatal_deaths_df)
    neo_SAM_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(neonatal_SAM_deaths_df)
    neo_ALRI_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(neonatal_ALRI_deaths_df)
    neo_Diarrhoea_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(neonatal_Diarrhoea_deaths_df)
    neo_ALRI_deaths_with_SAM_mean_ci_per_year_per_draw_df = \
        return_mean_95_CI_across_runs(neonatal_ALRI_deaths_with_SAM_df)
    neo_Diarrhoea_deaths_with_SAM_mean_ci_per_year_per_draw_df = \
        return_mean_95_CI_across_runs(neonatal_Diarrhoea_deaths_with_SAM_df)

    # neo deaths for each year within intervention period
    interv_neo_deaths_df = neonatal_deaths_df.loc[intervention_years]
    interv_neo_SAM_deaths_df = neonatal_SAM_deaths_df.loc[intervention_years]
    interv_neo_ALRI_deaths_df = neonatal_ALRI_deaths_df.loc[intervention_years]
    interv_neo_Diarrhoea_deaths_df = neonatal_Diarrhoea_deaths_df.loc[intervention_years]
    interv_neo_ALRI_deaths_with_SAM_df = neonatal_ALRI_deaths_with_SAM_df.loc[intervention_years]
    interv_neo_Diarrhoea_deaths_with_SAM_df = neonatal_Diarrhoea_deaths_with_SAM_df.loc[intervention_years]

    # sum and CI of neo deaths over intervention period, mean across runs
    interv_neo_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_neo_deaths_df)
    interv_neo_SAM_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_neo_SAM_deaths_df)
    interv_neo_ALRI_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_neo_ALRI_deaths_df)
    interv_neo_Diarrhoea_deaths_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_neo_Diarrhoea_deaths_df)
    interv_neo_ALRI_deaths_with_SAM_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_neo_ALRI_deaths_with_SAM_df)
    interv_neo_Diarrhoea_deaths_with_SAM_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_neo_Diarrhoea_deaths_with_SAM_df)

    # NEONATAL MORTALITY RATE (NMR), i.e. the number of deaths of infants up to 28 days old per 1,000 live births
    nmr_df = (neonatal_deaths_df / births_df) * 1000
    nmr_per_year_per_draw_df = return_mean_95_CI_across_runs(nmr_df)

    # ### UNDER-5 MORTALITY
    # Extract all deaths occurring during the first 5 years of life
    # differentiated by cause of death and acute malnutrition state
    under5_deaths_by_cause_am_df = extract_results(
        folder,
        module="tlo.methods.demography.detail",
        key="properties_of_deceased_persons",
        custom_generate_series=(
            lambda df: (filtered_by_age := df.loc[df['age_exact_years'] < 5])
            .assign(year=filtered_by_age['date'].dt.year)
            .groupby(['year', 'cause_of_death', 'un_clinical_acute_malnutrition'])['year']
            .count()
            .reindex(pd.MultiIndex.from_product([
                df['date'].dt.year.unique(), df['cause_of_death'].unique(),
                df['un_clinical_acute_malnutrition'].unique()
            ], names=['year', 'cause_of_death', 'un_clinical_acute_malnutrition']), fill_value=0)
        ),
        do_scaling=True).fillna(0)
    under5_deaths_by_cause_am_df = under5_deaths_by_cause_am_df.loc[years_of_interest]

    # number of deaths by any cause
    under5_deaths_df = under5_deaths_by_cause_am_df.groupby(['year']).sum()
    # number of deaths due to specific cause
    under5_SAM_deaths_df = under5_deaths_by_cause_am_df.loc[
        under5_deaths_by_cause_am_df.index.get_level_values('cause_of_death') == 'SevereAcuteMalnutrition'
        ].groupby(['year']).sum()
    under5_ALRI_deaths_df = under5_deaths_by_cause_am_df.loc[
        under5_deaths_by_cause_am_df.index.get_level_values('cause_of_death').str.startswith('ALRI_')
    ].groupby(['year']).sum()
    under5_Diarrhoea_deaths_df = under5_deaths_by_cause_am_df.loc[
        under5_deaths_by_cause_am_df.index.get_level_values('cause_of_death').str.startswith('Diarrhoea_')
    ].groupby(['year']).sum()
    # number of deaths due to specific cause with SAM
    under5_ALRI_deaths_with_SAM_df = under5_deaths_by_cause_am_df.loc[
        (under5_deaths_by_cause_am_df.index.get_level_values('un_clinical_acute_malnutrition') == 'SAM') &
        (under5_deaths_by_cause_am_df.index.get_level_values('cause_of_death').str.startswith('ALRI_'))
    ].groupby(['year']).sum()
    under5_Diarrhoea_deaths_with_SAM_df = under5_deaths_by_cause_am_df.loc[
        (under5_deaths_by_cause_am_df.index.get_level_values('un_clinical_acute_malnutrition') == 'SAM') &
        (under5_deaths_by_cause_am_df.index.get_level_values('cause_of_death').str.startswith('Diarrhoea_'))
    ].groupby(['year']).sum()

    under5_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(under5_deaths_df)
    under5_SAM_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(under5_SAM_deaths_df)
    under5_ALRI_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(under5_ALRI_deaths_df)
    under5_Diarrhoea_deaths_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(under5_Diarrhoea_deaths_df)
    under5_ALRI_deaths_with_SAM_mean_ci_per_year_per_draw_df = \
        return_mean_95_CI_across_runs(under5_ALRI_deaths_with_SAM_df)
    under5_Diarrhoea_deaths_with_SAM_mean_ci_per_year_per_draw_df = \
        return_mean_95_CI_across_runs(under5_Diarrhoea_deaths_with_SAM_df)

    # under 5 deaths for each year within intervention period
    interv_under5_deaths_df = under5_deaths_df.loc[intervention_years]
    interv_under5_SAM_deaths_df = under5_SAM_deaths_df.loc[intervention_years]
    interv_under5_ALRI_deaths_df = under5_ALRI_deaths_df.loc[intervention_years]
    interv_under5_Diarrhoea_deaths_df = under5_Diarrhoea_deaths_df.loc[intervention_years]
    interv_under5_ALRI_deaths_with_SAM_df = under5_ALRI_deaths_with_SAM_df.loc[intervention_years]
    interv_under5_Diarrhoea_deaths_with_SAM_df = under5_Diarrhoea_deaths_with_SAM_df.loc[intervention_years]

    # sum and CI of under 5 deaths over intervention period, mean across runs
    interv_under5_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_under5_deaths_df)
    interv_under5_SAM_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_under5_SAM_deaths_df)
    interv_under5_ALRI_deaths_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_under5_ALRI_deaths_df)
    interv_under5_Diarrhoea_deaths_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_under5_Diarrhoea_deaths_df)
    interv_under5_ALRI_deaths_with_SAM_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_under5_ALRI_deaths_with_SAM_df)
    interv_under5_Diarrhoea_deaths_with_SAM_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_under5_Diarrhoea_deaths_with_SAM_df)

    # UNDER-5 MORTALITY RATE, i.e. the number of deaths of children under 5 years old per 1,000 live births
    under5mr_df = (under5_deaths_df / births_df) * 1000
    under5mr_per_year_per_draw_df = return_mean_95_CI_across_runs(under5mr_df)

    return {'neo_deaths_df': neonatal_deaths_df,
            'neo_SAM_deaths_df': neonatal_SAM_deaths_df,
            'neo_ALRI_deaths_df': neonatal_ALRI_deaths_df,
            'neo_Diarrhoea_deaths_df': neonatal_Diarrhoea_deaths_df,
            'neo_ALRI_deaths_with_SAM_df': neonatal_ALRI_deaths_with_SAM_df,
            'neo_Diarrhoea_deaths_with_SAM_df': neonatal_Diarrhoea_deaths_with_SAM_df,
            'neo_deaths_mean_ci_df': neo_deaths_mean_ci_per_year_per_draw_df,
            'neo_SAM_deaths_mean_ci_df': neo_SAM_deaths_mean_ci_per_year_per_draw_df,
            'neo_ALRI_deaths_mean_ci_df': neo_ALRI_deaths_mean_ci_per_year_per_draw_df,
            'neo_Diarrhoea_deaths_mean_ci_df': neo_Diarrhoea_deaths_mean_ci_per_year_per_draw_df,
            'neo_ALRI_deaths_with_SAM_mean_ci_df': neo_ALRI_deaths_with_SAM_mean_ci_per_year_per_draw_df,
            'neo_Diarrhoea_deaths_with_SAM_mean_ci_df': neo_Diarrhoea_deaths_with_SAM_mean_ci_per_year_per_draw_df,
            'interv_neo_deaths_df': interv_neo_deaths_df,
            'interv_neo_deaths_sum_ci_df': interv_neo_deaths_sum_per_draw_CI_across_runs_df,
            'interv_neo_SAM_deaths_df': interv_neo_SAM_deaths_df,
            'interv_neo_SAM_deaths_sum_ci_df': interv_neo_SAM_deaths_sum_per_draw_CI_across_runs_df,
            'interv_neo_ALRI_deaths_df': interv_neo_ALRI_deaths_df,
            'interv_neo_ALRI_deaths_sum_ci_df': interv_neo_ALRI_deaths_sum_per_draw_CI_across_runs_df,
            'interv_neo_Diarrhoea_deaths_df': interv_neo_Diarrhoea_deaths_df,
            'interv_neo_Diarrhoea_deaths_sum_ci_df': interv_neo_Diarrhoea_deaths_sum_per_draw_CI_across_runs_df,
            'interv_neo_ALRI_deaths_with_SAM_df': interv_neo_ALRI_deaths_with_SAM_df,
            'interv_neo_ALRI_deaths_with_SAM_sum_ci_df': interv_neo_ALRI_deaths_with_SAM_sum_per_draw_CI_across_runs_df,
            'interv_neo_Diarrhoea_deaths_with_SAM_df': interv_neo_Diarrhoea_deaths_with_SAM_df,
            'interv_neo_Diarrhoea_deaths_with_SAM_sum_ci_df':
                interv_neo_Diarrhoea_deaths_with_SAM_sum_per_draw_CI_across_runs_df,
            'neonatal_mort_rate_df': nmr_df,
            'neo_mort_rate_mean_ci_df': nmr_per_year_per_draw_df,
            'under5_deaths_df': under5_deaths_df,
            'under5_SAM_deaths_df': under5_SAM_deaths_df,
            'under5_ALRI_deaths_df': under5_ALRI_deaths_df,
            'under5_Diarrhoea_deaths_df': under5_Diarrhoea_deaths_df,
            'under5_ALRI_deaths_with_SAM_df': under5_ALRI_deaths_with_SAM_df,
            'under5_Diarrhoea_deaths_with_SAM_df': under5_Diarrhoea_deaths_with_SAM_df,
            'under5_deaths_mean_ci_df': under5_deaths_mean_ci_per_year_per_draw_df,
            'under5_SAM_deaths_mean_ci_df': under5_SAM_deaths_mean_ci_per_year_per_draw_df,
            'under5_ALRI_deaths_mean_ci_df': under5_ALRI_deaths_mean_ci_per_year_per_draw_df,
            'under5_Diarrhoea_deaths_mean_ci_df': under5_Diarrhoea_deaths_mean_ci_per_year_per_draw_df,
            'under5_ALRI_deaths_with_SAM_mean_ci_df': under5_ALRI_deaths_with_SAM_mean_ci_per_year_per_draw_df,
            'under5_Diarrhoea_deaths_with_SAM_mean_ci_df':
                under5_Diarrhoea_deaths_with_SAM_mean_ci_per_year_per_draw_df,
            'interv_under5_deaths_df': interv_under5_deaths_df,
            'interv_under5_deaths_sum_ci_df': interv_under5_deaths_sum_per_draw_CI_across_runs_df,
            'interv_under5_SAM_deaths_df': interv_under5_SAM_deaths_df,
            'interv_under5_SAM_deaths_sum_ci_df': interv_under5_SAM_deaths_sum_per_draw_CI_across_runs_df,
            'interv_under5_ALRI_deaths_df': interv_under5_ALRI_deaths_df,
            'interv_under5_ALRI_deaths_sum_ci_df': interv_under5_ALRI_deaths_sum_per_draw_CI_across_runs_df,
            'interv_under5_Diarrhoea_deaths_df': interv_under5_Diarrhoea_deaths_df,
            'interv_under5_Diarrhoea_deaths_sum_ci_df': interv_under5_Diarrhoea_deaths_sum_per_draw_CI_across_runs_df,
            'interv_under5_ALRI_deaths_with_SAM_df': interv_under5_ALRI_deaths_with_SAM_df,
            'interv_under5_ALRI_deaths_with_SAM_sum_ci_df':
                interv_under5_ALRI_deaths_with_SAM_sum_per_draw_CI_across_runs_df,
            'interv_under5_Diarrhoea_deaths_with_SAM_df':
                interv_under5_Diarrhoea_deaths_with_SAM_df,
            'interv_under5_Diarrhoea_deaths_with_SAM_sum_ci_df':
                interv_under5_Diarrhoea_deaths_with_SAM_sum_per_draw_CI_across_runs_df,
            'under5_mort_rate_df': under5mr_df,
            'under5_mort_rate_mean_ci_df': under5mr_per_year_per_draw_df,
            'interv_years': intervention_years}

def extract_daly_data_frames_and_outcomes(
    folder,
    years_of_interest,
    intervention_years,
    interv,
    sq_dalys: dict = None,
) -> Dict[str, pd.DataFrame]:
    """
    Extracts DALYs by cause for under-5s (age_range '0-4'), summed over both sexes, for the specified years.
    :param folder: the folder from which the DALY data will be extracted
    :param years_of_interest: years for which to extract the data
    :param intervention_years: List of years during which the intervention was implemented (if any).
    :param interv: Name or identifier of the intervention.
    :param sq_dalys: Dict of DataFrames with DALYs outcomes for SQ.
    :return: Dict of DataFrames with index ['year'] and columns for each (draw, run), or columns for reach draw with
        mean_ci across all runs.
    """

    print(f"    -{interv=}")
    # ### UNDER-5 DALYs
    # Extract all DALYs assigned to children under 5 --- dalys_stacked_by_age_and_time, i.e. all the year of life lost
    # are ascribed to the age of the death and the year of the death differentiated by cause of death / disability

    def extrapolate_dalys_data_from_logs(df: pd.DataFrame) -> pd.Series:
        # Melt the DataFrame to have 'cause_of_dalys' as a variable
        df_with_cause_of_dalys = df.melt(
            id_vars=['age_range', 'sex', 'year'],
            value_vars=[
                "AIDS", "COPD", "Cancer (Bladder)", "Cancer (Breast)", "Cancer (Oesophagus)", "Cancer (Other)",
                "Cancer (Prostate)", "Childhood Diarrhoea", "Childhood Undernutrition", "Congenital birth defects",
                "Depression / Self-harm", "Diabetes", "Epilepsy", "Heart Disease", "Kidney Disease", "Lower Back Pain",
                "Lower respiratory infections", "Malaria", "Maternal Disorders", "Measles", "Neonatal Disorders",
                "Other", "Schistosomiasis", "Stroke", "TB (non-AIDS)", "Transport Injuries"
            ],
            var_name='cause_of_dalys',
            value_name='dalys'
        )

        # Keep only dalys for children under-5 by year and cause_of_dalys
        under5_dalys_by_year_cause = \
            df_with_cause_of_dalys[
                (df_with_cause_of_dalys['year'].isin(years_of_interest)) &
                (df_with_cause_of_dalys['age_range'] == '0-4')
            ].groupby(['year', 'cause_of_dalys'],as_index=True)['dalys'].sum()

        return under5_dalys_by_year_cause

    under5_dalys_by_cause_df = extract_results(
        folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked_by_age_and_time",
        custom_generate_series=lambda df: extrapolate_dalys_data_from_logs(df),
        do_scaling=True
    ).fillna(0)

    # Apply 3% discount rate to DALYs. Re-indexing is required to use the discounting function,
    # so the MultiIndexes must be restored afterward.
    under5_dalys_by_cause_df__reset_index = under5_dalys_by_cause_df.reset_index()
    under5_dalys_by_cause_df__reset_index.columns = [
        f"{col[0]}_{col[1]}" if col[1] != "" else f"{col[0]}"
        for col in under5_dalys_by_cause_df__reset_index.columns.values
    ]
    for col in under5_dalys_by_cause_df__reset_index.columns:
        if col.count('_') == 1 and all(part.isdigit() for part in col.split('_')):
            under5_dalys_by_cause_df__reset_index[col] = apply_discounting_to_cost_data(
                _df=under5_dalys_by_cause_df__reset_index, _discount_rate=0.03, _column_for_discounting=col
            )[col]
    # set MultiIndex for rows
    under5_dalys_by_cause_df = under5_dalys_by_cause_df__reset_index.set_index(['year', 'cause_of_dalys'])
    # create MultiIndex for columns
    new_col_tuples = [tuple(map(int, col.split('_'))) for col in under5_dalys_by_cause_df.columns if '_' in col]
    new_col_index = pd.MultiIndex.from_tuples(new_col_tuples, names=['draw', 'run'])
    under5_dalys_by_cause_df = under5_dalys_by_cause_df[[f"{d}_{r}" for d, r in new_col_tuples]]
    under5_dalys_by_cause_df.columns = new_col_index

    # number of dalys by any cause
    under5_dalys_df = under5_dalys_by_cause_df.groupby(['year']).sum()
    # number of dalys by specific causes
    under5_SAM_dalys_df = under5_dalys_by_cause_df.xs("Childhood Undernutrition", level=1)
    under5_ALRI_dalys_df = under5_dalys_by_cause_df.xs("Lower respiratory infections", level=1)
    under5_Diarrhoea_dalys_df = under5_dalys_by_cause_df.xs("Childhood Diarrhoea", level=1)

    under5_dalys_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(under5_dalys_df)
    under5_SAM_dalys_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(under5_SAM_dalys_df)
    under5_ALRI_dalys_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(under5_ALRI_dalys_df)
    under5_Diarrhoea_dalys_mean_ci_per_year_per_draw_df = return_mean_95_CI_across_runs(under5_Diarrhoea_dalys_df)

    # under 5 DALYs for each year within intervention period
    interv_under5_dalys_df = under5_dalys_df.loc[intervention_years]
    interv_under5_SAM_dalys_df = under5_SAM_dalys_df.loc[intervention_years]
    interv_under5_ALRI_dalys_df = under5_ALRI_dalys_df.loc[intervention_years]
    interv_under5_Diarrhoea_dalys_df = under5_Diarrhoea_dalys_df.loc[intervention_years]

    # sum and CI of under 5 DALYs over intervention period, mean across runs
    interv_under5_dalys_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_under5_dalys_df)
    interv_under5_SAM_dalys_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_under5_SAM_dalys_df)
    interv_under5_ALRI_dalys_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_under5_ALRI_dalys_df)
    interv_under5_Diarrhoea_dalys_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_under5_Diarrhoea_dalys_df)

    def compute_scen_sum_and_averted(
        interv_dalys_df: pd.DataFrame, sq_sum_dalys_df_name: str, limit_to_zero: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Sum `interv_dalys_df` across years into a single-row DataFrame labeled "start-end", then compute averted DALYs
        as compared to sum of DALYs under SQ. If limit_to_zero True, averted DALYs cannot be negative.
        Returns (scen_sum_dalys_df, averted_dalys_mean_ci_df).
        """
        # Sum Series -> single-row DataFrame and set new index 'year' as "first-last"
        scen_sum_dalys = interv_dalys_df.sum(axis=0)
        start_year = str(interv_dalys_df.index.min())
        end_year = str(interv_dalys_df.index.max())
        year_label = f"{start_year}-{end_year}"
        scen_sum_dalys_df = scen_sum_dalys.to_frame().T
        scen_sum_dalys_df.index.name = "year"
        scen_sum_dalys_df.index = [year_label]

        # Obtain SQ sum DALYs
        if interv != "SQ":
            sq_sum_dalys_df = sq_dalys[sq_sum_dalys_df_name]
        else:
            sq_sum_dalys_df = scen_sum_dalys_df

        # Compute averted DALYs, mean and CI across runs
        if limit_to_zero:
            sum_averted_dalys_df = (sq_sum_dalys_df - scen_sum_dalys_df).clip(lower=0)
            assert (sum_averted_dalys_df >= 0).all().all(), "Negative averted DALYs found in under_sum_averted_dalys_df, which should be limited to zero"
        else:
            sum_averted_dalys_df = sq_sum_dalys_df - scen_sum_dalys_df
        averted_dalys_mean_ci_df = return_mean_95_CI_across_runs(sum_averted_dalys_df)

        return scen_sum_dalys_df, averted_dalys_mean_ci_df

    under5_scen_sum_dalys_df, under5_averted_dalys_mean_ci_df = compute_scen_sum_and_averted(
        interv_under5_dalys_df, 'under5_scen_sum_dalys_df', True
    )
    under5_scen_sum_SAM_dalys_df, under5_averted_SAM_dalys_mean_ci_df = compute_scen_sum_and_averted(
        interv_under5_SAM_dalys_df, 'under5_scen_sum_SAM_dalys_df', True
    )
    under5_scen_sum_ALRI_dalys_df, under5_averted_ALRI_dalys_mean_ci_df = compute_scen_sum_and_averted(
        interv_under5_ALRI_dalys_df, 'under5_scen_sum_ALRI_dalys_df', False
    )
    under5_scen_sum_Diarrhoea_dalys_df, under5_averted_Diarrhoea_dalys_mean_ci_df = compute_scen_sum_and_averted(
        interv_under5_Diarrhoea_dalys_df, 'under5_scen_sum_Diarrhoea_dalys_df', False
    )

    return {
        "under5_dalys_df": under5_dalys_df,
        "under5_SAM_dalys_df": under5_SAM_dalys_df,
        "under5_ALRI_dalys_df": under5_ALRI_dalys_df,
        "under5_Diarrhoea_dalys_df": under5_Diarrhoea_dalys_df,
        "under5_dalys_mean_ci_df": under5_dalys_mean_ci_per_year_per_draw_df,
        "under5_SAM_dalys_mean_ci_df": under5_SAM_dalys_mean_ci_per_year_per_draw_df,
        "under5_ALRI_dalys_mean_ci_df": under5_ALRI_dalys_mean_ci_per_year_per_draw_df,
        "under5_Diarrhoea_dalys_mean_ci_df": under5_Diarrhoea_dalys_mean_ci_per_year_per_draw_df,
        "interv_under5_dalys_sum_ci_df": interv_under5_dalys_sum_per_draw_CI_across_runs_df,
        "interv_under5_SAM_dalys_sum_ci_df": interv_under5_SAM_dalys_sum_per_draw_CI_across_runs_df,
        "interv_under5_ALRI_dalys_sum_ci_df": interv_under5_ALRI_dalys_sum_per_draw_CI_across_runs_df,
        "interv_under5_Diarrhoea_dalys_sum_ci_df": interv_under5_Diarrhoea_dalys_sum_per_draw_CI_across_runs_df,
        "under5_scen_sum_dalys_df": under5_scen_sum_dalys_df,
        "under5_averted_dalys_mean_ci_df": under5_averted_dalys_mean_ci_df,
        "under5_scen_sum_SAM_dalys_df": under5_scen_sum_SAM_dalys_df,
        "under5_averted_SAM_dalys_mean_ci_df": under5_averted_SAM_dalys_mean_ci_df,
        "under5_scen_sum_ALRI_dalys_df": under5_scen_sum_ALRI_dalys_df,
        "under5_averted_ALRI_dalys_mean_ci_df": under5_averted_ALRI_dalys_mean_ci_df,
        "under5_scen_sum_Diarrhoea_dalys_df": under5_scen_sum_Diarrhoea_dalys_df,
        "under5_averted_Diarrhoea_dalys_mean_ci_df": under5_averted_Diarrhoea_dalys_mean_ci_df,
        "interv_years": intervention_years,
    }

def regenerate_pickles_with_debug_logs(iterv_folders_dict) -> None:
    for interv_folder_path in iterv_folders_dict.values():
        print(f"\n{interv_folder_path=} in regenerate_wasting_pickle_with_debug_logs")
        log_to_pickle = 'wasting_analysis__full_model_'
        create_pickles_locally(interv_folder_path, compressed_file_name_prefix=log_to_pickle, level=logging.DEBUG)

def extract_tx_data_frames(
    folder,
    years_of_interest,
    intervention_years,
    interv
) -> Dict[str, pd.DataFrame]:
    """
    Extracts and summarizes treatment data by age group and year.

    :param folder: Path to the folder containing outcome data.
    :param years_of_interest: List of years to extract data for.
    :param intervention_years: List of years for which data include the interventions if any implemented.
    :param interv: Name or identifier of the intervention.
    :return: Dictionary with DataFrames:
        (1) 'tx_by_age_group_df': Counts by year, treatment, age_group (by draw and run),
        (2) 'tx_by_age_group_mean_ci_df': Mean and 95% CI for counts per year, treatment, age_group and draw,
        (3) 'tx_mean_ci_df': Mean and 95% CI for total treatments per year and draw,
        (4) 'interv_tx_by_age_group_df': Counts for intervention years,
        (5) 'interv_tx_by_age_group_mean_ci_df': Mean and 95% CI for intervention years,
        (6) 'interv_tx_mean_ci_df': Mean and 95% CI for total treatment in intervention years.
    """
    print(f"    -{interv=}")

    # Extract treatment data
    tx_by_age_group_df = extract_results(
        folder,
        module="tlo.methods.wasting",
        key="get-tx",
        custom_generate_series=(
            lambda df: (
                df.assign(year=df['date'].dt.year)
                  .groupby(['year', 'treatment', 'age_group'])['year']
                  .count()
                  .reindex(
                      pd.MultiIndex.from_product([
                          df['date'].dt.year.unique(),
                          df['treatment'].unique(),
                          df['age_group'].unique()
                      ], names=['year', 'treatment', 'age_group'])
                  )
            )
        ),
        do_scaling=True
    ).fillna(0)
    tx_by_age_group_df = tx_by_age_group_df.loc[years_of_interest]

    # Mean and CI by year, treatment, age_group
    tx_by_age_group_mean_ci_df = return_mean_95_CI_across_runs(tx_by_age_group_df)

    # Mean and CI by year and treatment (sum over age_group)
    tx_mean_df = tx_by_age_group_df.groupby(['year', 'treatment']).sum()
    tx_mean_ci_df = return_mean_95_CI_across_runs(tx_mean_df)

    # For intervention years
    interv_tx_by_age_group_df = tx_by_age_group_df.loc[intervention_years]
    interv_tx_by_age_group_mean_ci_df = return_mean_95_CI_across_runs(interv_tx_by_age_group_df)
    interv_tx_mean_df = interv_tx_by_age_group_df.groupby(['year', 'treatment']).sum()
    interv_tx_mean_ci_df = return_mean_95_CI_across_runs(interv_tx_mean_df)

    return {
        'tx_by_age_group_df': tx_by_age_group_df,
        'tx_by_age_group_mean_ci_df': tx_by_age_group_mean_ci_df,
        'tx_mean_ci_df': tx_mean_ci_df,
        'interv_tx_by_age_group_df': interv_tx_by_age_group_df,
        'interv_tx_by_age_group_mean_ci_df': interv_tx_by_age_group_mean_ci_df,
        'interv_tx_mean_ci_df': interv_tx_mean_ci_df,
        'interv_years': intervention_years
    }

def get_scen_colour(scen_name: str) -> str:
    return {
        "Status Quo": "#F12AE5",
        "SQ": "#F12AE5",
        # "GM_FullAttend": "#4575B4",
        # "GM_all": "#BDEBF7",
        # "GM_1-2": "#91BFDB",
        "GM": "#4575B4",
        # "CS_10": "#9FFD17",
        # "CS_30": "#61B93C",
        # "CS_50": "#2D945F",
        # "CS_100": "#266714",
        "CS": "#266714",
        # "FS_50": "#D4898E",
        # "FS_70": "#D4898E",
        # "FS_Full": "#A90251",
        "FS": "#A90251",
        "GM_FS": "#FD7700",
        "CS_FS": "#54DC5C",
        "GM_CS_FS": "#350D90",
        "GM_CS": "#689DEF",
    }.get(scen_name)

def plot_mortality_rate__by_interv_multiple_settings(
    cohort: str,
    interv_timestamps_dict: dict,
    scenarios_dict: dict,
    intervs_of_interest: list,
    plot_years: list,
    outcomes_dict: dict,
    outputs_path: Path
) -> None:
    """
    Plots mortality rates (neonatal or under-5) and their confidence intervals over time for multiple intervention
    settings. For the 'SQ' (Status Quo) intervention, also overlays UNICEF and WPP reference data.

    :param cohort: 'Neonatal' or 'Under-5'
    :param interv_timestamps_dict: Dictionary mapping intervention names to their timestamp identifiers
    :param scenarios_dict: Dictionary mapping interventions to scenario names and draw numbers
    :param intervs_of_interest: List of interventions to plot
    :param plot_years: List of years to plot on the x-axis
    :param outcomes_dict: Nested dictionary with outcome data for each intervention and scenario
    :param outputs_path: Path to save the generated plots
    :return: None
    """

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
            # ####
            # Load UNICEF mortality rates data from CSV
            unicef_csv_path = Path(__file__).parent / "fusion_GLOBAL_DATAFLOW_UNICEF_1.0_MWI.CME_MRY0T4+CME_MRM0...csv"
            unicef_df = pd.read_csv(unicef_csv_path)

            # Filter for neonatal and under-5 rates, total sex
            neo_mask = (
                (unicef_df['INDICATOR:Indicator'] == 'CME_MRM0: Neonatal mortality rate') &
                (unicef_df['SEX:Sex'] == '_T: Total')
            )
            under5_mask = (
                (unicef_df['INDICATOR:Indicator'] == 'CME_MRY0T4: Under-five mortality rate') &
                (unicef_df['SEX:Sex'] == '_T: Total')
            )

            unicef_neo = unicef_df.loc[neo_mask]
            unicef_under5 = unicef_df.loc[under5_mask]

            # Extract years and rates (convert to int/float)
            unicef_neo_years = unicef_neo['TIME_PERIOD:Time period'].astype(int).tolist()
            unicef_neo_rates = unicef_neo['OBS_VALUE:Observation Value'].astype(float).tolist()
            unicef_neo_lower = unicef_neo['LOWER_BOUND:Lower Bound'].astype(float).tolist()
            unicef_neo_upper = unicef_neo['UPPER_BOUND:Upper Bound'].astype(float).tolist()

            unicef_under5_years = unicef_under5['TIME_PERIOD:Time period'].astype(int).tolist()
            unicef_under5_rates = unicef_under5['OBS_VALUE:Observation Value'].astype(float).tolist()
            unicef_under5_lower = unicef_under5['LOWER_BOUND:Lower Bound'].astype(float).tolist()
            unicef_under5_upper = unicef_under5['UPPER_BOUND:Upper Bound'].astype(float).tolist()

            unicef_colour = '#1CABE2'

            # Filter data to include only years present in both plot_years and source years
            unicef_filtered_neo = [(year, mort_rate, low, upper) for year, mort_rate, low, upper in \
                                   zip(unicef_neo_years, unicef_neo_rates, unicef_neo_lower, unicef_neo_upper) if \
                                   year in plot_years]
            unicef_filtered_under5 = [(year, mort_rate, low, upper) for year, mort_rate, low, upper in
                                      zip(unicef_under5_years, unicef_under5_rates,
                    unicef_under5_lower, unicef_under5_upper) if
                year in plot_years]

            (unicef_filtered_neo_years, unicef_filtered_neo_rates,
             unicef_filtered_neo_lower, unicef_filtered_neo_upper) = \
                zip(*unicef_filtered_neo) if unicef_filtered_neo else ([], [], [], [])
            (unicef_filtered_under5_years, unicef_filtered_under5_rates,
             unicef_filtered_under5_lower, unicef_filtered_under5_upper) = \
                zip(*unicef_filtered_under5) if unicef_filtered_under5 else ([], [], [], [])

            # Add WPP 2024 mortality rates estimates (past data) and medium projection variant (future predictions)
            # ####
            wpp_medium_under5_mort_rates = [
                81, 76, 70, 65, 61, 57, 53, 51, 48, 46, 44, 42, 41, 39, 38, 37, 37, 36, 35, 34, 33, 33, 32, 31, 30, 30,
                29, 28, 27, 27, 26, 26, 25, 24, 24, 23, 23, 22, 22, 21, 21, 21, 20, 20, 19, 19, 19, 18, 18, 17, 17, 17,
                16, 16, 16, 16, 16, 15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12,
                12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11
            ]
            wpp_years = list(range(2010, 2101))
            wpp_colour = '#1D73F5'
            wpp_filtered_years = [year for year in plot_years if year in wpp_years]
            wpp_filtered_under5_rates = \
                [rate for year, rate in zip(wpp_years, wpp_medium_under5_mort_rates) if year in wpp_filtered_years]

            # Plot both data
            # ####
            if cohort == 'Neonatal':
                ax.plot(unicef_filtered_neo_years, unicef_filtered_neo_rates,
                        label='UNICEF Data', color=unicef_colour, linestyle='--')
                ax.fill_between(
                    unicef_filtered_neo_years, unicef_filtered_neo_lower, unicef_filtered_neo_upper,
                    color=unicef_colour, alpha=0.2
                )
            elif cohort == 'Under-5':
                ax.plot(unicef_filtered_under5_years, unicef_filtered_under5_rates,
                        label='UNICEF Data', color=unicef_colour, linestyle='--')
                ax.fill_between(
                    unicef_filtered_under5_years, unicef_filtered_under5_lower, unicef_filtered_under5_upper,
                    color=unicef_colour, alpha=0.2
                )
                ax.plot(wpp_filtered_years, wpp_filtered_under5_rates,
                        label='WPP 2024', color=wpp_colour, linestyle='-.')
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
            fig.savefig(
                outputs_path / f"{cohort}_mort_rate_{interv}_UNICEF_WPP__"
                               f"{interv_timestamps_dict[interv]}.png",
                bbox_inches='tight'
            )
            plt.close(fig)

        else:
            fig.savefig(
                outputs_path / f"{cohort}_mort_rate_{interv}_multiple_settings__"
                               f"{interv_timestamps_dict[interv]}__{interv_timestamps_dict['SQ']}.png",
                bbox_inches='tight'
            )
            plt.close(fig)

def plot_mean_outcome_and_CIs__scenarios_comparison(
    cohort: str,
    scenarios_dict: dict,
    scenarios_to_compare: list,
    plot_years: list,
    outcome_type: str,
    outcomes_dict: dict,
    outputs_path: Path,
    scenarios_tocompare_prefix: str,
    timestamps_suffix: str
) -> None:
    """
    Plots mean deaths or DALYs and confidence intervals over time for the specified cohort for multiple scenarios.
    :param cohort: 'Neonatal' or 'Under-5'
    :param scenarios_dict: Dictionary mapping interventions to scenarios and their corresponding draw numbers
    :param scenarios_to_compare: List of scenarios to plot
    :param plot_years: List of years to plot
    :param outcome_type: 'deaths', 'deaths_with_SAM', or 'DALYs'
    :param outcomes_dict: Dictionary containing data for plotting nested as outcomes_dict[interv][outcome][draw][run]
    :param outputs_path: Path to save the plots
    :param scenarios_tocompare_prefix: Prefix for output files with names of scenarios that are compared in the plots
    :param timestamps_suffix: Timestamps to identify the log data from which the outcomes originated.
    """
    assert cohort in ['Neonatal', 'Under-5'], \
        f"Invalid value for 'cohort': expected 'Neonatal' or 'Under-5'. Received {cohort} instead."
    assert outcome_type in ['deaths', 'deaths_with_SAM', 'DALYs'], \
        f"Invalid value for 'outcome_type': expected 'deaths' or 'DALYs'. Received {outcome_type} instead."

    for i, cause in enumerate(['any cause', 'SAM', 'ALRI', 'Diarrhoea']):
        if outcome_type == "deaths":
            neonatal_outcomes = ['neo_deaths_mean_ci_df', 'neo_SAM_deaths_mean_ci_df',
                                 'neo_ALRI_deaths_mean_ci_df', 'neo_Diarrhoea_deaths_mean_ci_df']
            under5_outcomes = ['under5_deaths_mean_ci_df', 'under5_SAM_deaths_mean_ci_df',
                               'under5_ALRI_deaths_mean_ci_df', 'under5_Diarrhoea_deaths_mean_ci_df']
        elif outcome_type == "deaths_with_SAM":
            neonatal_outcomes = [None, None,
                                 'neo_ALRI_deaths_with_SAM_mean_ci_df', 'neo_Diarrhoea_deaths_with_SAM_mean_ci_df']
            under5_outcomes = [None, None,
                               'under5_ALRI_deaths_with_SAM_mean_ci_df', 'under5_Diarrhoea_deaths_with_SAM_mean_ci_df']
        else:  # outcome_type == "DALYs":
            neonatal_outcomes = [None, None, None, None]  # No data on DALYs for neonatal
            under5_outcomes = ['under5_dalys_mean_ci_df', 'under5_SAM_dalys_mean_ci_df',
                               'under5_ALRI_dalys_mean_ci_df', 'under5_Diarrhoea_dalys_mean_ci_df']
        outcome = neonatal_outcomes[i] if cohort == 'Neonatal' else under5_outcomes[i]

        if outcome:
            # Initialize the plot
            fig, ax = plt.subplots()

            # Iterate over scenarios to compare
            for scenario in scenarios_to_compare:
                # Find the corresponding intervention and draw number
                try:
                    interv, draw = next(
                        (interv, draw)
                        for interv, scenarios_for_interv_dict in scenarios_dict.items()
                        if scenario in scenarios_for_interv_dict
                        for scen_name, draw in scenarios_for_interv_dict.items()
                        if scen_name == scenario
                    )
                except StopIteration:
                    raise ValueError(f"Scenario '{scenario}' not found in scenarios_dict")


                # Extract data for the scenario
                scen_data = outcomes_dict[interv][outcome][draw]

                # Calculate means and confidence intervals
                means, ci_lower, ci_upper = zip(*scen_data.values.flatten())

                # Plot the data
                years_to_plot = [year for year in plot_years if year-1 in scen_data.index]
                ax.plot(years_to_plot, means, label=scenario, color=get_scen_colour(scenario))
                ax.fill_between(years_to_plot, ci_lower, ci_upper, color=get_scen_colour(scenario), alpha=0.2)

            # Add labels, title, and legend
            plt.ylabel(f'{cohort} {outcome_type}')
            plt.xlabel('Year')
            plt.title(f'{cohort} Mean {outcome_type.replace("_", " ")} due to {cause} and 95% CI over time')
            plt.legend()
            plt.xticks(years_to_plot, labels=years_to_plot, rotation=45, fontsize=8)

            plt.savefig(
                outputs_path / (
                    f"{cohort}_mean_{cause}_{outcome_type}_CI_scenarios_comparison__"
                    f"{scenarios_tocompare_prefix}__{timestamps_suffix}.png"
                ),
                bbox_inches='tight'
            )
            plt.close(fig)

def plot_percentage_deaths_with_SAM(
    cohort: str,
    scenarios_dict: dict,
    scenarios_to_compare: list,
    plot_years: list,
    outcome_type: str,
    outcomes_dict: dict,
    outputs_path: Path,
    scenarios_tocompare_prefix: str,
    timestamps_suffix: str
) -> None:
    """
    Plots mean deaths or DALYs and confidence intervals over time for the specified cohort for multiple scenarios.
    :param cohort: 'Neonatal' or 'Under-5'
    :param scenarios_dict: Dictionary mapping interventions to scenarios and their corresponding draw numbers
    :param scenarios_to_compare: List of scenarios to plot
    :param plot_years: List of years to plot
    :param outcome_type: 'deaths', 'deaths_with_SAM', or 'DALYs'
    :param outcomes_dict: Dictionary containing data for plotting nested as outcomes_dict[interv][outcome][draw][run]
    :param outputs_path: Path to save the plots
    :param scenarios_tocompare_prefix: Prefix for output files with names of scenarios that are compared in the plots
    :param timestamps_suffix: Timestamps to identify the log data from which the outcomes originated.
    """

def plot_sum_outcome_and_CIs_intervention_period(
    cohort: str,
    scenarios_dict: dict,
    scenarios_to_compare: list,
    outcome_type: str,
    outcomes_dict: dict,
    outputs_path: Path,
    scenarios_tocompare_prefix: str,
    timestamps_suffix: str,
    interv_timestamps_dict: dict = None,
    births_dict: dict = None,
    force_calculation: list = None,
) -> None:
    """
    Plots sum & averted sum of averted deaths or DALYs over the intervention period for the specified cohort for
    multiple scenarios (means and confidence intervals across runs).
    If outcome is DALYs, also plots and tables the cost-effectiveness.
    :param cohort: 'Neonatal' or 'Under-5'
    :param scenarios_dict: Dictionary mapping interventions to scenarios and their corresponding draw numbers
    :param scenarios_to_compare: List of scenarios to plot
    :param outcome_type: 'deaths', 'deaths_with_SAM' or 'DALYs'
    :param outcomes_dict: Dictionary containing data for plotting nested as outcomes_dict[interv][outcome][draw][run]
    :param outputs_path: Path to save the plot
    :param scenarios_tocompare_prefix: Prefix for output files with names of scenarios that are compared in the plots
    :param timestamps_suffix: Suffix with timestamps to identify the log data from which the outcomes originated
    :param interv_timestamps_dict: Dictionary with timestamps for all the interventions
            (default: None, as needed only for outcome_type = 'DALYs' for cost-effectiveness analysis)
    :param births_dict: Dataframe containing births data from simulations nested as
            births_dict[interv][outcome][draw][run]
            (default: None, as needed only for outcome_type = 'DALYs' for cost-effectiveness analysis)
    """
    assert cohort in ['Neonatal', 'Under-5'], \
        f"Invalid value for 'cohort': expected 'Neonatal' or 'Under-5'. Received {cohort} instead."
    assert outcome_type in ['deaths', 'deaths_with_SAM', 'DALYs'], \
        f"Invalid value for 'outcome_type': expected 'deaths' or 'DALYs'. Received {outcome_type} instead."

    # Outcomes to plot
    for i, cause in enumerate(['any cause', 'SAM', 'ALRI', 'Diarrhoea']):

        if outcome_type == "deaths":
            neonatal_outcomes = ['interv_neo_deaths_sum_ci_df', 'interv_neo_SAM_deaths_sum_ci_df',
                                 'interv_neo_ALRI_deaths_sum_ci_df', 'interv_neo_Diarrhoea_deaths_sum_ci_df']
            under5_outcomes = ['interv_under5_deaths_sum_ci_df', 'interv_under5_SAM_deaths_sum_ci_df',
                               'interv_under5_ALRI_deaths_sum_ci_df', 'interv_under5_Diarrhoea_deaths_sum_ci_df']
            under5_averted_outcomes = [None, None, None, None] # only prepared for DALYs
        elif outcome_type == "deaths_with_SAM":
            neonatal_outcomes = [None, None,
                                 'interv_neo_ALRI_deaths_with_SAM_sum_ci_df',
                                 'interv_neo_Diarrhoea_deaths_with_SAM_sum_ci_df']
            under5_outcomes = [None, None,
                               'interv_under5_ALRI_deaths_with_SAM_sum_ci_df',
                               'interv_under5_Diarrhoea_deaths_with_SAM_sum_ci_df']
            under5_averted_outcomes = [None, None, None, None] # only prepared for DALYs
        else:  # outcome_type == "DALYs"
            neonatal_outcomes = [None, None, None, None]  # No DALYs for neonatal
            under5_outcomes = ['interv_under5_dalys_sum_ci_df', 'interv_under5_SAM_dalys_sum_ci_df',
                               'interv_under5_ALRI_dalys_sum_ci_df', 'interv_under5_Diarrhoea_dalys_sum_ci_df']
            under5_averted_outcomes = ['under5_averted_dalys_mean_ci_df', 'under5_averted_SAM_dalys_mean_ci_df',
                                       'under5_averted_ALRI_dalys_mean_ci_df',
                                       'under5_averted_Diarrhoea_dalys_mean_ci_df']
        outcome = under5_outcomes[i] if cohort == 'Under-5' else neonatal_outcomes[i]
        averted_outcome = under5_averted_outcomes[i] if cohort =='Under-5' else None

        if outcome:
            # Plot comparison of sum of outcome_type over intervention period (absolute numbers of outcome_type)
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
                interv_sum, interv_ci_lower, interv_ci_upper = zip(*scen_data.values.flatten())
                interv_sum, interv_ci_lower, interv_ci_upper = \
                    interv_sum[0], interv_ci_lower[0], interv_ci_upper[0]

                # Plot the data
                ax.bar(scenario, interv_sum,
                       yerr=[[interv_sum - interv_ci_lower], [interv_ci_upper - interv_sum]],
                       label=scenario, color=get_scen_colour(scenario), capsize=5)

                y_top = ax.get_ylim()[1]

                # Add text label for the bar height (sum), above the bar
                ax.text(
                    scenario,
                    interv_sum + (y_top * 0.02),  # small offset above the bar
                    f"{interv_sum:,.0f}",
                    color='black',
                    ha='center',
                    va='bottom',
                    fontsize=12.5
                )

                # Add text labels for interv_ci_lower and interv_ci_upper
                text_color = 'black' if scenario in ['Status Quo'] else 'white'
                ax.text(scenario,
                        interv_ci_upper / 2 + interv_ci_upper / 4 if \
                            interv_ci_upper < y_top / 2 + y_top / 15 else y_top / 2 + y_top / 15,
                        f"{interv_ci_upper:,.0f}", color=text_color, ha='center', va='top', fontsize=12.5)
                ax.text(scenario,
                        interv_ci_upper / 2 - interv_ci_upper / 4 if \
                            interv_ci_upper < y_top / 2 + y_top / 15 else y_top / 2 - y_top / 15,
                        f"{interv_ci_lower:,.0f}", color=text_color, ha='center', va='bottom', fontsize=12.5)

                # Add horizontal lines for Status Quo scenario
                if scenario == 'Status Quo':
                    ax.axhline(y=interv_ci_lower, color=get_scen_colour('Status Quo'), linestyle='--', linewidth=1)
                    ax.axhline(y=interv_ci_upper, color=get_scen_colour('Status Quo'), linestyle='--', linewidth=1)

            # Add labels, title, and legend
            interv_years = outcomes_dict["SQ"]['interv_years']
            min_interv_year = min(interv_years)
            max_interv_year = max(interv_years)
            plt.ylabel(f'{cohort} {outcome_type} (Sum over intervention period)')
            plt.xlabel('Scenario')
            plt.title(
                f'{cohort} Sum of {outcome_type.replace("_", " ")} due to {cause} and 95% CI over '
                f'intervention period ({min_interv_year}--{max_interv_year})')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xticks(rotation=45, fontsize=8)

            fig.savefig(
                outputs_path / (
                    f"{cohort}_sum_{cause}_{outcome_type}_CI_intervention_period_scenarios_comparison__"
                    f"{scenarios_tocompare_prefix}__{timestamps_suffix}.png"
                ),
                bbox_inches='tight'
            )
            plt.close(fig)

        # noinspection PyUnreachableCode
        if averted_outcome:
            fig2, ax2 = plt.subplots()

            averted_DALYs = {}
            for scenario in scenarios_to_compare:
                if scenario == 'Status Quo':
                    # Only horizontal lines, no bar
                    ax2.axhline(y=0, color=get_scen_colour('Status Quo'), linestyle='-', linewidth=2)
                else:
                    # Iterate over scenarios to compare
                    interv, draw = next(
                        (interv, draw)
                        for interv, scenarios_for_interv_dict in scenarios_dict.items()
                        if scenario in scenarios_for_interv_dict
                        for scen_name, draw in scenarios_for_interv_dict.items()
                        if scen_name == scenario
                    )

                    # Extract data for the scenario
                    scen_averted_data = \
                        outcomes_dict[interv][averted_outcome][draw][f'{min_interv_year}-{max_interv_year}']
                    averted_DALYs[scenario] = scen_averted_data

                    ax2.bar(scenario, scen_averted_data[0],
                            yerr=[[scen_averted_data[0] - scen_averted_data[1]],
                                  [scen_averted_data[2] - scen_averted_data[0]]],
                            label=scenario, color=get_scen_colour(scenario), capsize=5)
                    y_top2 = ax2.get_ylim()[1]
                    s1 = y_top2 * 0.02  # space between bar and value of the bar
                    ax2.text(scenario, scen_averted_data[0] + s1 if scen_averted_data[0] >= 0 else 0 + s1,
                             f"{scen_averted_data[0]:,.0f}", color=get_scen_colour(scenario), ha='center', va='bottom',
                             fontsize=12.5, fontweight='bold')

                    # # Display lower and upper CI within the bars
                    # ax2.text(scenario, averted_ci_upper / 2 + averted_ci_upper / 4 if \
                    #     averted_ci_upper < y_top2 / 2 + y_top2 / 15 else y_top2 / 2 + y_top2 / 15,
                    #          f"{averted_ci_upper:,.0f}", color='white', ha='center', va='top', fontsize=12.5)
                    # ax2.text(scenario, averted_ci_upper / 2 - averted_ci_upper / 4 if \
                    #     averted_ci_upper < y_top2 / 2 + y_top2 / 15 else y_top2 / 2 - y_top2 / 15,
                    #          f"{averted_ci_lower:,.0f}", color='white', ha='center', va='bottom', fontsize=12.5)

            if cause in ['any cause', 'SAM']:
                # Apply millions formatter to y-axis values for SAM outcomes
                plt.ylabel(f"Averted {outcome_type}, millions (Cumulative: {min_interv_year}—{max_interv_year})")
                apply_millions_formatter_to_ax(ax2, axis='y')
            else:
                plt.ylabel(f"Averted {outcome_type} (Cumulative: {min_interv_year}—{max_interv_year})")
            plt.xlabel('Scenario')
            plt.title(
                f'{cohort}: Sum of averted {outcome_type.replace("_", " ")} due to {cause} and 95% CI over '
                f'intervention period ({min_interv_year}--{max_interv_year})'
            )
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(handles2, map_scenario_labels(labels2), loc='center left', bbox_to_anchor=(1, 0.5), labelspacing=1.4)
            ax2.set_xticks(list(range(len(labels2))))
            ax2.set_xticklabels(labels2, rotation=45, fontsize=8)

            fig2.savefig(
                outputs_path / (
                    f"{cohort}_sum_averted_{cause}_{outcome_type}_CI_intervention_period_scenarios_comparison__"
                    f"{scenarios_tocompare_prefix}__{timestamps_suffix}.png"
                ),
                bbox_inches='tight'
            )
            plt.close(fig2)

            # path to outcome calculated data
            cost_outcome_folder_path = outputs_path / "outcomes_data"
            # SQ timestamp associated with scenarios for which we want the costs to be calculated
            SQ_results_timestamp = interv_timestamps_dict["SQ"]

            def plot_and_table_cost_effectiveness(
                in_averted_DALYs: dict, in_data_impl_cost_name: str, in_sharing_GM_CS: float, in_FS_multiplier: float
            ) -> None:
                ce_suffix = f"{in_data_impl_cost_name}_GM-CS-sharing{in_sharing_GM_CS}_FSmultiplier{in_FS_multiplier}"
                timestamps_and_ce_suffix = f"{timestamps_suffix}__{ce_suffix}"
                # -----------
                # Implementation costs estimates based on number of births and unit costs from REFs, discounted by 3%
                def calculate_implementation_costs():
                    # Implementation costs for education/promotion interventions (ie GM & CS)
                    # note: start_up_unit_cost is assumed to be additional cost to unit cost for the first year of
                    #       implementation

                    # Gelli et al., 2021 (incl food, not the best choice as we already have food supplements modelled
                    # separately)
                    # start_up_cost_Gelli_etal2021 = 58.41 * 0.3
                    # unit_cost_Gelli_etal2021 = 58.41 * 0.7

                    # Pearson et al., 2018 (seems to include purely implementation costs, and assume coverage for the
                    # interventions, however as we intend to reach 100% scale-up, we assume 100% coverage)
                    # coverage_educ_Pearson_etal2018 = 0.247
                    # coverage_prom_Pearson_etal2018 = 0.61
                    coverage = 1.0  # [coverage_educ_Pearson_etal2018, coverage_prom_Pearson_etal2018, 1.0]
                    # coverage with Margolies et al., 2021
                    # 0.15 => 'CS_FS': 30,484,191.749434147 > 29,853,966
                    # 0.14 => 'CS_FS': 28,451,912.299471878 < 29,853,966
                    # 0.0017 => 'GM_CS': 259,464.976512309 > 246,824.02
                    # 0.0016 => 'GM_CS': 244,202.33083511435 < 246,824.02

                    unit_cost_Pearson_etal2018 = 0.37
                    start_up_Pearson_etal2018 = 0.37 / 7 * 3 # no start-up costs included in the estimate
                    discount_incl_bool_Pearson_etal2018 = False # no discounting (1 year long programme)

                    unit_cost_Margolies_etal2021_min = 9.17
                    start_up_Margolies_etal2021 = 0  # included in their estimate of unit cost
                    discount_incl_bool_Margolies_etal2021 = True # already includes 3% annual discounting (5 years)

                    # Implementation costs GM, CS & FS:
                    # #####
                    scenarios_without_SQ = [scen for scen in scenarios_to_compare if scen != 'Status Quo']

                    if in_data_impl_cost_name == 'Pearson_etal2018':
                        unit_cost = unit_cost_Pearson_etal2018
                        start_up_unit_cost = start_up_Pearson_etal2018
                        discount_incl_bool = discount_incl_bool_Pearson_etal2018

                    elif in_data_impl_cost_name == 'Margolies_etal2021':
                        unit_cost = unit_cost_Margolies_etal2021_min
                        start_up_unit_cost = start_up_Margolies_etal2021
                        discount_incl_bool = discount_incl_bool_Margolies_etal2021

                    print(f"\nUNIT COST: {unit_cost}")
                    impl_costs = dict()
                    for scen in scenarios_without_SQ:
                        scen_coef = (
                            (in_FS_multiplier if scen == "FS" else 1) if "_" not in scen else
                            (2 - in_sharing_GM_CS + in_FS_multiplier) if scen.count("_") == 2 else
                            (2 - in_sharing_GM_CS) if scen == "GM_CS" else (1 + in_FS_multiplier)
                        )
                        impl_cost_df = pd.DataFrame(
                            {
                                "year": interv_years,
                                "impl_costs": [
                                    births_dict[scen]["births_mean_ci_df"][0].loc[year][0]
                                    * coverage
                                    * ((unit_cost + start_up_unit_cost) if year == interv_years[0] else unit_cost)
                                    * scen_coef
                                    for year in interv_years
                                ],
                            }
                        )
                        # print(f"\n--------\nscen {scen}\n--------")
                        # print("\nimpl_cost_df")
                        # print(impl_cost_df)
                        if not discount_incl_bool:
                            impl_cost_df = apply_discounting_to_cost_data(
                                _df=impl_cost_df, _discount_rate=0.03, _column_for_discounting='impl_costs'
                            )
                        sum_impl_costs_discounted = sum(impl_cost_df['impl_costs'])
                        impl_costs[scen] = sum_impl_costs_discounted
                        # print("\nimpl_cost_discounted_df")
                        # print(impl_cost_discounted_df)
                        # print("\nimpl_costs")
                        # print(impl_costs)

                    # Implementation costs FS:
                    # TODO:
                    #  interv_tx_mean_ci_df
                    #  salary [RF_Costing_HR Nutrition] at levels 1a, 1b and 2: 4573.11153 USD (annual??)

                    # Add 0 implementation costs for SQ
                    impl_costs['SQ'] = 0.0
                    return impl_costs

                def get_all_costs():
                    # 1. Medical consumable costs - discounted by 3%
                    output_costs_medical_file_path = \
                        cost_outcome_folder_path / f"output_costs_medical_outcomes_{SQ_results_timestamp}.pkl"
                    if output_costs_medical_file_path.exists() and not force_calculation[4]:
                        print("\nloading output costs medical from file ...")
                        output_costs_medical_df = pd.read_pickle(output_costs_medical_file_path)
                    else:
                        print("\noutput costs medical calculation ...")
                        run_costing(
                            cost_outcome_folder_path, SQ_results_timestamp, timestamps_suffix, force_calculation
                        )
                        output_costs_medical_df = pd.read_pickle(output_costs_medical_file_path)
                    incremental_consumable_costs = dict()
                    for scen in output_costs_medical_df.index:
                        incremental_consumable_costs[scen] = \
                            output_costs_medical_df.loc[scen, 'total'] - output_costs_medical_df.loc['SQ', 'total']
                    print("\nincremental_consumable_costs")
                    print(incremental_consumable_costs)

                    # TODO: use also incremental_consumable_costs CIs
                    #  (saved as output_costs_medical_df.loc[scen, ['lower_bound', 'upper_bound']])
                    #  How the net_health_benefit & net_monetary_benefit then will need to account for both uncertainty
                    #  around Averted DALYs and around Incremental Costs (hence similar calculation as when calculated
                    #  CI for Averted DALYs)

                    # 2. Implementation costs estimates based on number of births and unit costs from REFs,
                    # discounted by 3%
                    implementation_costs = calculate_implementation_costs()
                    print(f"\nimplementation_costs: {in_data_impl_cost_name}; GM & CS shared costs: {in_sharing_GM_CS};"
                          f" FS multiplier: {in_FS_multiplier}")
                    print(implementation_costs)

                    all_costs = pd.DataFrame(
                        {
                            "scenario": list(incremental_consumable_costs.keys()),
                            "incremental_consumable_costs": [
                                incremental_consumable_costs[scen] for scen in incremental_consumable_costs.keys()
                            ],
                            "implementation_costs": [
                                implementation_costs[scen] for scen in incremental_consumable_costs.keys()
                            ],
                            "total_cost": [
                                incremental_consumable_costs[scen] + implementation_costs[scen]
                                for scen in incremental_consumable_costs.keys()
                            ],
                        }
                    )
                    return all_costs

                output_all_costs_file_path = \
                    cost_outcome_folder_path / f"all_costs_{SQ_results_timestamp}_{ce_suffix}.pkl"
                if output_all_costs_file_path.exists() and not force_calculation[4] and not force_calculation[5] :
                    print("\nloading all costs from file ...")
                    all_costs_df = pd.read_pickle(output_all_costs_file_path)
                    print("-------------")
                    print(f"\nCE_SUFFIX={ce_suffix}")
                    # print(f"--all_costs_df:\n{all_costs_df}")
                else:
                    print("\nall costs calculation ...")
                    all_costs_df = get_all_costs()
                    print("\nsaving all costs to file ...")
                    all_costs_df.to_pickle(output_all_costs_file_path)

                # Cost-effectiveness threshold (CET) = willingness to pay
                CET = 75.9

                # PLOT cost-effectiveness plane
                # #####
                print("\nplotting CE plane ...")
                scenarios_without_SQ = [scen for scen in all_costs_df['scenario'] if scen != "SQ"]

                fig_ce, ax_ce = plt.subplots(figsize=(10, 8))
                space = 0.012 * all_costs_df.loc[all_costs_df['scenario'] == "FS", "total_cost"].values[0]
                for scen in scenarios_without_SQ:
                    ax_ce.errorbar(
                        in_averted_DALYs[scen][0],
                        all_costs_df.loc[all_costs_df['scenario'] == scen, 'total_cost'].values[0],
                        xerr=[
                            [in_averted_DALYs[scen][0] - in_averted_DALYs[scen][1]],
                            [in_averted_DALYs[scen][2] - in_averted_DALYs[scen][0]],
                        ],
                        fmt="o",
                        color=get_scen_colour(scen),
                        capsize=5,
                    )
                # add SQ point
                ax_ce.plot(0, 0, marker="o", color=get_scen_colour('SQ'))
                # define domination for scenarios and position of label in plot
                if in_data_impl_cost_name == 'Pearson_etal2018':
                    domination = {
                        "SQ": "dominated",
                        "GM": "dominated",
                        "CS": "",
                        "FS": "dominated",
                        "GM_FS": "dominated",
                        "CS_FS": "",
                        "GM_CS_FS": "",
                        "GM_CS": "extendedly dominated",
                    }
                    ha_scen = {
                        "SQ": "left",
                        "GM": "center",
                        "CS": "center",
                        "FS": "left",
                        "GM_FS": "center",
                        "CS_FS": "left",
                        "GM_CS_FS": "center",
                        "GM_CS": "left",
                    }  # ['right', 'left', 'center']
                    va_scen = {
                        "SQ": "top",
                        "GM": "bottom",
                        "CS": "top",
                        "FS": "bottom",
                        "GM_FS": "bottom",
                        "CS_FS": "top",
                        "GM_CS_FS": "bottom",
                        "GM_CS": "bottom",
                    }  # ['bottom', 'top', 'bottom']

                elif in_data_impl_cost_name == 'Margolies_etal2021':
                    if in_FS_multiplier > 0.3:
                        if in_sharing_GM_CS < 0.2 and in_FS_multiplier < 0.7:
                            domination = {
                                "SQ": "",
                                "GM": "dominated",
                                "CS": "extendedly dominated",
                                "FS": "extendedly dominated",
                                "GM_FS": "dominated",
                                "CS_FS": "",
                                "GM_CS_FS": "",
                                "GM_CS": "dominated",
                            }
                        else: # in_sharing_GM_CS >= 0.2 or in_FS_multiplier >= 0.7:
                            domination = {
                                "SQ": "",
                                "GM": "dominated",
                                "CS": "extendedly dominated",
                                "FS": "extendedly dominated",
                                "GM_FS": "dominated",
                                "CS_FS": "",
                                "GM_CS_FS": "",
                                "GM_CS": "extendedly dominated",
                            }
                    else: # in_FS_multiplier <= 0.3:
                        domination = {
                            "SQ": "",
                            "GM": "dominated",
                            "CS": "extendedly dominated",
                            "FS": "",
                            "GM_FS": "dominated",
                            "CS_FS": "",
                            "GM_CS_FS": "",
                            "GM_CS": "dominated",
                        }
                    ha_scen = {
                        "SQ": "left",
                        "GM": "center",
                        "CS": "right",
                        "FS": "right",
                        "GM_FS": "right",
                        "CS_FS": "left",
                        "GM_CS_FS": "center",
                        "GM_CS": "right",
                    }  # ['right', 'left', 'center']
                    va_scen = {
                        "SQ": "top",
                        "GM": "bottom",
                        "CS": "top",
                        "FS": "bottom",
                        "GM_FS": "bottom",
                        "CS_FS": "top",
                        "GM_CS_FS": "bottom",
                        "GM_CS": "bottom",
                    }  # ['bottom', 'top', 'bottom']
                # print("\nICER:")

                # Identify scenarios on Frontier
                scen_frontier = [scen for scen, dom in domination.items() if dom == ""]
                # Order them by total costs
                cost_map = all_costs_df.set_index('scenario')['total_cost'].to_dict()
                scen_frontier_ordered = sorted(scen_frontier, key=lambda s: cost_map.get(s, float('inf')))
                print(f"\nscen_frontier_ordered: {scen_frontier_ordered}")
                ICER = {}
                for i, scen in enumerate(scen_frontier_ordered):
                    if i == 0:
                        ICER[scen] = ""
                        continue
                    prev = scen_frontier_ordered[i - 1]
                    cost_curr = cost_map.get(scen, float("nan"))
                    cost_prev = cost_map.get(prev, float("nan"))
                    daly_curr = in_averted_DALYs[scen][0]
                    daly_prev = in_averted_DALYs[prev][0] if prev != 'SQ' else 0
                    ICER[scen] = f"${round((cost_curr - cost_prev) / (daly_curr - daly_prev), 1)}"
                print(f"\nICER: {ICER}")
                ICER_domination = {}
                for scen in all_costs_df['scenario']:
                    ICER_domination[scen] = ICER[scen] if domination[scen] == "" else domination[scen]
                print(f"\nICER_domination: {ICER_domination}")
                for scen in all_costs_df['scenario']:
                    ax_ce.text(
                        in_averted_DALYs[scen][0] if scen != 'SQ' else 0,
                        (cost_map.get(scen, float("nan")) + (space * 0.9)) if va_scen[scen] == "bottom"
                        else (cost_map.get(scen, float("nan")) - space),
                        ICER_domination[scen],
                        fontsize=12, ha=ha_scen[scen], va=va_scen[scen], color=get_scen_colour(scen),
                    )

                # Add cost-effectiveness frontier (dotted line connecting non-dominated scenarios)
                frontier_x = [in_averted_DALYs[scen][0] if scen != 'SQ' else 0 for scen in scen_frontier_ordered]
                frontier_y = [cost_map.get(scen, float("nan")) for scen in scen_frontier_ordered]
                ax_ce.plot(
                    frontier_x,
                    frontier_y,
                    linestyle=":",
                    color="black",
                    linewidth=1.5,
                    label="Cost-effectiveness frontier",
                )

                # Add axis labels
                ax_ce.set_xlabel("DALYs Averted, millions")
                ax_ce.set_ylabel("Total Incremental Costs (2023 USD), millions")
                # Format both axis to millions; x rounded to 1 decimal, but y to 0 decimals
                apply_millions_formatter_to_ax(ax_ce, y_decimals=0)
                # #TODO: uncomment only when creating this, otherwise it messes with other figures axis
                # ax_ce.set_title(
                #     f"$\\bf{{unit\\ cost:}}$ {in_data_impl_cost_name}; "
                #     f"$\\bf{{CS\\ &\\ GM\\ sharing:}}$ {in_sharing_GM_CS} prop of implem. costs; "
                #     f"$\\bf{{FS\\ multiplier:}}$ {in_FS_multiplier}",
                #     pad=12,
                # )

                # Add dashed black line for 1 DALY averted per CET
                y_vals = np.array(ax_ce.get_ylim())
                x_vals = y_vals / CET
                # x_vals = np.array(ax_ce.get_xlim())
                # y_vals = x_vals * CET
                ax_ce.plot(x_vals, y_vals, color="black", linestyle="--")
                ax_ce.text(
                    x_vals[-1] + 10, y_vals[-1] + 1e6, f"ICER = ${CET}/DALY",
                    color="black", fontsize=9, ha="left", va="top",
                )

                # Add a legend box with scenario and CEF labels
                # order legend labels by y (total_cost)
                # so the largest value appears at the top and smallest at the bottom
                ordered_scenarios = list(
                    all_costs_df.sort_values("total_cost", ascending=False)["scenario"].tolist()
                )
                # Create proxy handles with matching colours for the ordered scenarios
                proxy_handles = [
                    mpl_lines.Line2D([], [], marker="o", color=get_scen_colour(scen), linestyle="-", markersize=8)
                    for scen in ordered_scenarios
                ]
                # add a proxy handle for the cost-effectiveness frontier (dotted black line)
                frontier_handle = mpl_lines.Line2D([], [], linestyle=":", color="black", linewidth=1.5)
                proxy_handles.append(frontier_handle)

                # map scenario short codes to display full names in legend
                ax_ce.legend(proxy_handles, map_scenario_labels(ordered_scenarios) + ['cost-effectiveness\nfrontier'], loc="center left",
                             bbox_to_anchor=(1, 0.5), fontsize=12, labelspacing=1.4)

                plt.tight_layout()
                fig_ce.savefig(
                    outputs_path
                    / (
                        f"cost_effectiveness_scatter_DALYsAverted_vs_TotalCosts__"
                        f"{scenarios_tocompare_prefix}__{timestamps_and_ce_suffix}.png"
                    ),
                    bbox_inches="tight",
                )
                plt.close(fig_ce)

                # TABLE cost-effectiveness
                # #####
                print("\ncreating table with CE metrics ...")

                # Create a cost-effectiveness summary table for each scenario
                ce_table_rows = []
                for scen in all_costs_df["scenario"]:
                    averted_dalys = in_averted_DALYs[scen][0] if scen != 'SQ' else 0
                    averted_dalys_lower = in_averted_DALYs[scen][1] if scen != 'SQ' else 0
                    averted_dalys_upper = in_averted_DALYs[scen][2] if scen != 'SQ' else 0
                    incremental_costs = \
                        all_costs_df.loc[all_costs_df['scenario'] == scen, 'incremental_consumable_costs'].values[0]
                    impl_cost = all_costs_df.loc[all_costs_df['scenario'] == scen, 'implementation_costs'].values[0]
                    total_cost = incremental_costs + impl_cost

                    max_impl_costs = (averted_dalys * CET) - incremental_costs
                    max_impl_costs_lower = (averted_dalys_lower * CET) - incremental_costs
                    max_impl_costs_upper = (averted_dalys_upper * CET) - incremental_costs

                    net_health_benefit = averted_dalys - (total_cost / CET)
                    net_health_benefit_lower = averted_dalys_lower - (total_cost / CET)
                    net_health_benefit_upper = averted_dalys_upper - (total_cost / CET)

                    net_monetary_benefit = (averted_dalys * CET) - total_cost
                    net_monetary_benefit_lower = (averted_dalys_lower * CET) - total_cost
                    net_monetary_benefit_upper = (averted_dalys_upper * CET) - total_cost

                    ce_table_rows.append(
                        {
                            "Scenario": scen,
                            "Averted DALYs (95% CI)": f"{averted_dalys:,.0f} ({averted_dalys_lower:,.0f}; {averted_dalys_upper:,.0f})",
                            "Total costs (2023 USD)": f"{total_cost:,.0f}",
                            "Incremental consumable-related costs (2023 USD)": f"{incremental_costs:,.0f}",
                            "Maximum additional implementation costs (2023 USD, 95% CI)": f"{max_impl_costs:,.0f} ({max_impl_costs_lower:,.0f}; {max_impl_costs_upper:,.0f})",
                            "Implementation costs estimate (2023 USD)": f"{impl_cost:,.0f}",
                            "Incremental net health benefit (95% CI)": f"{net_health_benefit:,.0f} ({net_health_benefit_lower:,.0f}; {net_health_benefit_upper:,.0f})",
                            "Incremental net monetary benefit (95% CI)": f"{net_monetary_benefit:,.0f} ({net_monetary_benefit_lower:,.0f}; {net_monetary_benefit_upper:,.0f})",
                        }
                    )
                ce_table_df = pd.DataFrame(ce_table_rows)
                ce_table_df = ce_table_df.sort_values(
                    by="Total costs (2023 USD)", key=lambda x: x.str.replace(",", "").astype(float)
                )
                ce_table_df.to_csv(
                    outputs_path / f"cost_effectiveness_summary_table__{timestamps_and_ce_suffix}.csv",
                    index=False,
                )

            def table_effectiveness(averted_outcome: dict, outcome_cause_suffix:str) -> None:

                eff_table = pd.DataFrame(
                    {
                        "Scenario": list(averted_outcome.keys()),
                        f"Averted {outcome_type}": [
                            f"{int(round(vals[0])):,} ({int(round(vals[1])):,}; {int(round(vals[2])):,})"
                            for vals in averted_outcome.values()
                        ],
                    }
                )
                eff_table.to_csv(outputs_path / f"effectiveness_table_{outcome_cause_suffix}_{timestamps_suffix}.csv",
                                 index=False)

            if outcome_type == "DALYs":
                if cause == "any cause":
                    # sensitivity to unit cost
                    data_impl_cost_name = ['Pearson_etal2018', 'Margolies_etal2021']
                    # sensitivity to sharing implementation costs for GM and CS interventions
                    sharing_GM_CS = [0.5, 0]
                    # sensitivity to FS intervention multiplier
                    FS_multiplier = [1, 0.5, 0.18, 0.09]
                    # Individual CE planes
                    for unit_cost in data_impl_cost_name:
                        for GM_CS__multiplier in sharing_GM_CS:
                            for FS__multiplier in FS_multiplier:
                                plot_and_table_cost_effectiveness(
                                    averted_DALYs, unit_cost, GM_CS__multiplier, FS__multiplier
                                )

                    ########################################
                    # Sensitivity plot - create grid of CE figures for all unit_cost / GM_CS / FS combinations
                    ########################################
                    print("\nplotting sensitivity CE plot ...")
                    # build grid dims
                    n_rows = len(data_impl_cost_name) * len(sharing_GM_CS)
                    n_cols = len(FS_multiplier)

                    # Create figure at higher resolution (DPI) to improve exported image quality
                    # use gridspec spacing controls to reduce large gaps between columns/rows
                    fig, axes = plt.subplots(
                        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), dpi=200,
                        constrained_layout=False,
                        gridspec_kw={'wspace': 0.12, 'hspace': 0.18}
                    )

                    # Tighten left/right margins and avoid large automatic padding caused by negative text positions
                    fig.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.06)

                    # Ensure axes is a 2D array for consistent indexing
                    axes = np.atleast_2d(axes)

                    # iterate rows and columns in desired order:
                    # rows: for each unit_cost in data_impl_cost_name -> for each GM_CS in sharing_GM_CS
                    for r_idx, unit_cost in enumerate(data_impl_cost_name):
                        for g_idx, gm_cs in enumerate(sharing_GM_CS):
                            row_idx = r_idx * len(sharing_GM_CS) + g_idx
                            for c_idx, fs_mult in enumerate(FS_multiplier):
                                ax = axes[row_idx, c_idx]
                                ce_suffix = f"{unit_cost}_GM-CS-sharing{gm_cs}_FSmultiplier{fs_mult}"
                                timestamps_and_ce_suffix = f"{timestamps_suffix}__{ce_suffix}"
                                img_path = outputs_path / (
                                    f"cost_effectiveness_scatter_DALYsAverted_vs_TotalCosts__"
                                    f"{scenarios_tocompare_prefix}__{timestamps_and_ce_suffix}.png"
                                )
                                if img_path.exists():
                                    # Open image and display with preserved aspect ratio; turn axes off
                                    img = Image.open(img_path).convert("RGB")
                                    ax.imshow(img, aspect='auto', interpolation='bilinear')
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    ax.set_axis_off()
                                else:
                                    # Clear axes and show placeholder text
                                    ax.clear()
                                    ax.text(0.5, 0.5, f"Missing:\\n{img_path.name}", ha="center", va="center",
                                            fontsize=8)
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    ax.set_axis_off()

                    # add row labels on the left (unit_cost + gm_cs) using figure text to avoid expanding axes margins
                    for r in range(n_rows):
                        unit_idx = r // len(sharing_GM_CS)
                        gm_idx = r % len(sharing_GM_CS)

                        # column titles: only for first row
                        if r == 0:
                            for c in range(n_cols):
                                axes[0, c].set_title(f"$\\bf{{FS\\ multiplier:}}$ {FS_multiplier[c]}",
                                                     loc="center", x=0.43, pad=2, fontsize=8)
                        # row labels: place in left margin of the row; use transform to align with axes coordinates
                        row_label = (
                            f"$\\bf{{unit\\ cost:}}$ {data_impl_cost_name[unit_idx]};\n"
                            f"$\\bf{{GM\\ &\\ CS\\ shared\\ implem. cost\\ prop.:}}$ {sharing_GM_CS[gm_idx]}"
                        )
                        axes[r, 0].text(-0.02, 0.5, row_label, transform=axes[r, 0].transAxes,
                                        rotation=90, ha="right", va="center", fontsize=8)

                    plt.tight_layout()
                    cohort_prefix = \
                        "Neo" if cohort == "Neonatal" else "Under5" if cohort == "Under-5" else "unknown_cohort"
                    out_file = \
                        outputs_path / (f"{cohort_prefix}_cost_effectiveness_sensitivity_grid__"
                                        f"{scenarios_tocompare_prefix}__{timestamps_suffix}.png")
                    # Save at higher DPI for better quality
                    fig.savefig(out_file, bbox_inches="tight", dpi=300)
                    plt.close(fig)

                    ########################################
                    # Total cost sensitivity table
                    ########################################
                    print("\ncreating sensitivity CE table ...")
                    print("\n    per 5 years...")
                    # includes total_cost ranges (min—max) for the same grid used in the CE sensitivity plot
                    table_rows = []
                    row_labels = []
                    for unit_cost in data_impl_cost_name:
                        for gm_cs in sharing_GM_CS:
                            row_label = f"{unit_cost}; GM & CS: {gm_cs}"
                            row_labels.append(row_label)
                            row_cells = []
                            for fs_mult in FS_multiplier:
                                ce_suffix = f"{unit_cost}_GM-CS-sharing{gm_cs}_FSmultiplier{fs_mult}"
                                all_costs_path = outputs_path / "outcomes_data" / f"all_costs_{SQ_results_timestamp}_{ce_suffix}.pkl"
                                if all_costs_path.exists():
                                    try:
                                        all_costs_df_local = pd.read_pickle(all_costs_path)
                                        if "total_cost" in all_costs_df_local.columns:
                                            # Exclude SQ scenario here as well
                                            if "scenario" in all_costs_df_local.columns:
                                                df_nonSQ_local = all_costs_df_local[all_costs_df_local["scenario"] != "SQ"]
                                            else:
                                                df_nonSQ_local = all_costs_df_local.drop(index="SQ", errors="ignore")
                                            if df_nonSQ_local.empty:
                                                cell = ""
                                            else:
                                                lo = df_nonSQ_local["total_cost"].min()
                                                hi = df_nonSQ_local["total_cost"].max()
                                            cell = f"{lo:,.0f}—{hi:,.0f}"
                                        else:
                                            cell = ""
                                    except Exception:
                                        cell = ""
                                else:
                                    cell = ""
                                row_cells.append(cell)
                            table_rows.append(row_cells)

                    # Columns labelled by FS multiplier values
                    col_labels = [str(x) for x in FS_multiplier]
                    total_cost_range_table = pd.DataFrame(table_rows, index=row_labels, columns=col_labels)

                    # Save CSV summary (per 5 years)
                    out_table_path_without_csv = outputs_path / f"total_cost_sensitivity_table__{scenarios_tocompare_prefix}__{timestamps_suffix}"
                    total_cost_range_table.to_csv(Path(str(out_table_path_without_csv) + "_per5years.csv"), index=True)

                    # Create and save per-1-year table by dividing min and max values by 5
                    print("\n    per 1 year...")
                    def _divide_range_cell(cell: str) -> str:
                        if not cell:
                            return ""
                        try:
                            lo_str, hi_str = cell.split("—")
                            lo = float(lo_str.replace(",", ""))
                            hi = float(hi_str.replace(",", ""))
                            lo1 = lo / 5.0
                            hi1 = hi / 5.0
                            return f"{lo1:,.0f}—{hi1:,.0f}"
                        except Exception:
                            return ""

                    total_cost_range_table_per1 = total_cost_range_table.applymap(_divide_range_cell)
                    total_cost_range_table_per1.to_csv(Path(str(out_table_path_without_csv) + "_per1year.csv"), index=True)
                else: # cause != "any cause":
                    table_effectiveness(averted_DALYs, f"{outcome_type}_{cause}")
            # if outcome_type == "deaths":
            #     table_effectiveness(averted_deaths, f"{outcome_type}_{cause}")

# ----------------------------------------------------------------------------------------------------------------------
def plot_availability_heatmaps(outputs_path: Path) -> None:
    """
    Creates the following heatmaps of average availabilities:
    A) HEATMAP OF ESSENTIAL CONSUMABLES AVAILABILITY
    A1) average over the entire year at each facility level
    A2) average for each month at each facility level
    A3) average for each month at requested facility level
    B) HEATMAP OF TREATMENTS AVAILABILITY, i.e. probability all essential consumables for the treatments are available
    B1) average over the entire year at each facility level
    B2) average for each month at each facility level
    B3) average for each month at requested facility level

    :param outputs_path: Path to save the plots as PNG files.
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
    chosen_item_codes = [208, 1227, 1220]
    item_names_to_map = {208:'CSB++*', 1227:'RUTF', 1220:'F-75\ntherapeutic\nmilk'}

    tlo_availability_df = tlo_availability_df[tlo_availability_df.Facility_Level.isin(correct_order_of_fac_levels)]

    # Month labels used for any plot showing months (A2, A3, B2, B3)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Mapping integer month columns (1-12) to month labels
    month_map = {i + 1: month_labels[i] for i in range(12)}

    # A) HEATMAP OF ESSENTIAL CONSUMABLES AVAILABILITY
    # A1) Essential consumables: average over the entire year at each facility level
    # ###
    print("Heathmap A1...")
    # Pivot the DataFrame
    aggregated_df = \
        tlo_availability_df.groupby(['Facility_Level', 'item_code'])[['available_prop']].mean().reset_index()
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
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Proportion of days on which consumable is available'})

    plt.title('Availability of essential consumables\n for acute malnutrition treatments', fontweight='bold')
    plt.xlabel('Facility Level')
    plt.ylabel('Consumable')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(outputs_path / 'consumable_availability_heatmap.png', dpi=300, bbox_inches='tight')

    # A2) Essential consumables: average for each month at each facility level
    # ###
    print("Heathmaps A2...")
    monthly_agg_df = \
        tlo_availability_df.groupby(["Facility_Level", "item_code", "month"])[['available_prop']].mean().reset_index()
    months = range(1, 13)

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    for i, month in enumerate(months):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        month_df = monthly_agg_df[monthly_agg_df["month"] == month]
        heatmap_data_month = month_df.pivot(
            columns="Facility_Level", index="item_code", values="available_prop"
        ).reindex(index=chosen_item_codes, columns=correct_order_of_fac_levels)
        # Add average column (across all facility levels)
        aggregate_col_month = month_df.groupby('item_code')[['available_prop']].mean()
        heatmap_data_month['Average'] = aggregate_col_month
        heatmap_data_month.index = heatmap_data_month.index.map(item_names_to_map)
        sns.heatmap(heatmap_data_month, annot=True, cmap="RdYlGn", cbar=False, ax=ax, vmin=0, vmax=1)
        ax.set_title(f"Month {month}", fontweight="bold")
        if row == 2:
            ax.set_xlabel("Facility Level")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("Consumable")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(outputs_path / "consumable_monthly_availability_heatmaps.png", dpi=300, bbox_inches="tight")

    # A3) Essential consumables: average for each month at requested facility level
    # ###
    print("Heathmap A3...")
    ess_cons_requested_at = ['208_1a', '1227_1a', '1227_1b', '1220_1b']

    def split_item_level(item_level: str) -> tuple[int, str]:
        """
        :param item_level: string with item left from '_', and level right from '_'
        :return: takes the item_level, and returns what is before '_' as integer (it is the item_code)
                and what is after '_' as string (it is fac_level)
        """
        assert item_level.count('_') == 1, "the argument of split_item_level fnc must have a structure of: item_level"
        before, after = item_level.split('_', 1)
        return int(before), after

    monthly_aggregated_data_requested_fac_level = \
        pd.DataFrame(columns=['item_level', 'month', 'available_prop'])
    for item_level in ess_cons_requested_at:
        item_code, fac_level = split_item_level(item_level)
        for month in months:
            val = monthly_agg_df[
                (monthly_agg_df["item_code"] == item_code)
                & (monthly_agg_df["Facility_Level"] == fac_level)
                & (monthly_agg_df["month"] == month)
            ]["available_prop"]
            monthly_aggregated_data_requested_fac_level = \
                pd.concat([monthly_aggregated_data_requested_fac_level,
                           pd.DataFrame([{'item_level': item_level, "month": month,
                                          "available_prop": val.values[0] if not val.empty else None}])],
                          ignore_index=True)

    # Pivot for heatmap
    heatmap_data_requested_fac_level_raw = \
        monthly_aggregated_data_requested_fac_level.pivot(
            index="item_level", columns="month", values="available_prop"
        )
    # Map item codes to names
    item_level_labels_to_map = {"208_1a": "CSB++*\nfacility level 1", "1227_1a": "RUTF\nfacility level 1",
                               "1227_1b": "RUTF\nfacility level 2",
                               "1220_1b": "F-75 therapeutic milk\nfacility level 2"}
    heatmap_data_requested_fac_level = heatmap_data_requested_fac_level_raw.copy()
    heatmap_data_requested_fac_level.index = \
        heatmap_data_requested_fac_level.index.map(item_level_labels_to_map)
    heatmap_data_requested_fac_level = \
        heatmap_data_requested_fac_level.reindex([item_level_labels_to_map[item_level] for item_level in ess_cons_requested_at])
    heatmap_data_requested_fac_level["Average"] = heatmap_data_requested_fac_level.mean(axis=1)
    heatmap_data_requested_fac_level[""] = np.nan  # Add empty column for spacing
    # Reorder columns: months Jan-Dec, empty column, then Average
    heatmap_data_requested_fac_level = heatmap_data_requested_fac_level.rename(columns=month_map)
    ordered_cols = month_labels + ["", "Average"]
    heatmap_data_requested_fac_level = heatmap_data_requested_fac_level[ordered_cols]
    plt.figure(figsize=(12, 4))
    sns.heatmap(
        heatmap_data_requested_fac_level,
        annot=True, cmap="RdYlGn", vmin=0, vmax=1,cbar_kws={"label": "Proportion of days available"}
    )
    # plt.title("Monthly average availability of consumables at requested facility levels", fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Consumable")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.savefig(outputs_path / "consumable_availability_heatmap_requested_fac_level.png", dpi=300, bbox_inches="tight")

    # B) HEATMAP OF TREATMENTS AVAILABILITY, i.e. probability all essential consumables for the treatments are available
    # B1) Treatments: average over the entire year at each facility level
    # ###
    print("Heathmap B1...")
    treatment_item_map = {
        "SFP": ["CSB++*"],  # 208
        "OTP": ["RUTF"],  # 1227
        "ITC": ["F-75\ntherapeutic\nmilk", "RUTF"]  # 1220, 1227
    }

    # Calculate availability for treatments
    treatment_availability = {}
    for treatment, items in treatment_item_map.items():
        treatment_availability[treatment] = {
            level: np.prod([heatmap_data.loc[item_code, level] for item_code in items])
            for level in correct_order_of_fac_levels
        }

    # Prepare the DataFrame
    treatment_heatmap_data = \
        pd.DataFrame.from_dict(treatment_availability, orient='index',columns=correct_order_of_fac_levels)
    treatment_heatmap_data = treatment_heatmap_data.reindex(columns=correct_order_of_fac_levels)
    treatment_heatmap_data['Average'] = treatment_heatmap_data.mean(axis=1)

    # Generate the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(treatment_heatmap_data, annot=True, cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Proportion of days on which treatment is available'})

    plt.title('Availability of treatments\n for acute malnutrition', fontweight='bold')
    plt.xlabel('Facility Level')
    plt.ylabel('Treatment')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(outputs_path / 'treatment_availability_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # B2) Treatments: average for each month at each facility level
    # ###
    print("Heathmaps B2...")
    monthly_treatment_availability = {}
    for month in months:
        month_df = monthly_agg_df[monthly_agg_df["month"] == month]
        heatmap_data_month = month_df.pivot(
            columns="Facility_Level", index="item_code", values="available_prop"
        ).reindex(index=chosen_item_codes, columns=correct_order_of_fac_levels)
        heatmap_data_month.index = heatmap_data_month.index.map(item_names_to_map)

        # Calculate treatment availability for each facility level
        treatment_availability_month = {}
        for treatment, items in treatment_item_map.items():
            treatment_availability_month[treatment] = {
                level: np.prod([heatmap_data_month.loc[item_code, level] for item_code in items])
                for level in correct_order_of_fac_levels
            }
        treatment_heatmap_data_month = pd.DataFrame.from_dict(
            treatment_availability_month, orient="index", columns=correct_order_of_fac_levels
        )
        treatment_heatmap_data_month = treatment_heatmap_data_month.reindex(columns=correct_order_of_fac_levels)
        treatment_heatmap_data_month["Average"] = treatment_heatmap_data_month.mean(axis=1)
        monthly_treatment_availability[month] = treatment_heatmap_data_month

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    for i, month in enumerate(months):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        sns.heatmap(monthly_treatment_availability[month], annot=True, cmap="RdYlGn", cbar=False, ax=ax, vmin=0, vmax=1)
        ax.set_title(f"Month {month}", fontweight="bold")
        if row == 2:
            ax.set_xlabel("Facility Level")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("Treatment")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    plt.tight_layout()
    fig.savefig(outputs_path / "treatment_monthly_availability_heatmaps.png", dpi=300, bbox_inches="tight")

    # B3) Treatments: average for each month at requested facility level
    # ###
    print("Heathmap B3...")
    # Calculate availability for treatments
    treatment_item_level_map = {
        "SFP\nfacility level 1": ['208_1a'],
        "OTP\nfacility level 1": ['1227_1a'],
        "ITC\nfacility level 2": ['1227_1b', '1220_1b']
    }
    treatment_availability_requested_fac_level = {}
    for treatment, items_level in treatment_item_level_map.items():
        treatment_availability_requested_fac_level[treatment] = {
            month: np.prod(
                [heatmap_data_requested_fac_level_raw.loc[item_level, month] for item_level in items_level]
            ) for month in months
        }

    # Prepare the DataFrame
    treatment_heatmap_data_requested_fac_level = \
        pd.DataFrame.from_dict(treatment_availability_requested_fac_level, orient='index',columns=months)
    treatment_heatmap_data_requested_fac_level['Average'] = treatment_heatmap_data_requested_fac_level.mean(axis=1)
    treatment_heatmap_data_requested_fac_level[''] = np.nan
    # Reorder columns: months Jan-Dec, empty column, then Average
    treatment_heatmap_data_requested_fac_level = treatment_heatmap_data_requested_fac_level.rename(columns=month_map)
    ordered_cols = month_labels + ["", "Average"]
    treatment_heatmap_data_requested_fac_level = treatment_heatmap_data_requested_fac_level[ordered_cols]

    # Generate the heatmap
    plt.figure(figsize=(12, 4))
    sns.heatmap(
        treatment_heatmap_data_requested_fac_level,
        annot=True, cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={"label": "Proportion of days available"}
    )
    # plt.title("Monthly average availability of treatments at requested facility levels", fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Treatment")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.savefig(outputs_path / "treatment_availability_heatmap_requested_fac_level.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
