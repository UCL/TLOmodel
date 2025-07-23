"""
A helping file that contains functions used for wasting analyses to extract data, derive outcomes and generate plots.
It is not to be run by itself. Functions are called from run_interventions_analysis_wasting.py, and
heatmaps_cons_wast.py.
"""

from pathlib import Path
from typing import Dict

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
        ci = st.t.interval(0.95, len(draw_data) - 1, loc=np.mean(draw_data), scale=st.sem(draw_data))
        result.at['sum', draw] = [np.mean(draw_data), ci[0], ci[1]]

    return result

def extract_birth_data_frames_and_outcomes(
    folder,
    years_of_interest,
    intervention_years,
    interv
) -> Dict[str, pd.DataFrame]:
    """
    Extracts and summarizes birth data.

    :param folder: Path to the folder containing outcome data.
    :param years_of_interest: List of years to extract data for.
    :param intervention_years: List of years during which the intervention was implemented (if any).
    :param interv: Name or identifier of the intervention.
    :return: Dictionary with DataFrames:
            (1) 'births_df': Birth counts for years of interest (by draw and run),
            (2) 'births_mean_ci_df': Mean and 95% CI for total births per year and draw,
            (3) 'interv_births_df': Birth counts for intervention years,
            (4) 'interv_births_mean_ci_df': Mean and 95% CI for births per year and draw for intervention years.
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
            'interv_births_mean_ci_df': interv_births_per_year_per_draw_df,
            'interv_years': intervention_years}

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

    print(f"\n{interv=}")
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

    interv_neo_deaths_df = neonatal_deaths_df.loc[intervention_years]
    interv_neo_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_neo_deaths_df)
    interv_neo_SAM_deaths_df = neonatal_SAM_deaths_df.loc[intervention_years]
    interv_neo_SAM_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_neo_SAM_deaths_df)
    interv_neo_ALRI_deaths_df = neonatal_ALRI_deaths_df.loc[intervention_years]
    interv_neo_ALRI_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_neo_ALRI_deaths_df)
    interv_neo_Diarrhoea_deaths_df = neonatal_Diarrhoea_deaths_df.loc[intervention_years]
    interv_neo_Diarrhoea_deaths_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_neo_Diarrhoea_deaths_df)
    interv_neo_ALRI_deaths_with_SAM_df = neonatal_ALRI_deaths_with_SAM_df.loc[intervention_years]
    interv_neo_ALRI_deaths_with_SAM_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_neo_ALRI_deaths_with_SAM_df)
    interv_neo_Diarrhoea_deaths_with_SAM_df = neonatal_Diarrhoea_deaths_with_SAM_df.loc[intervention_years]
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

    interv_under5_deaths_df = under5_deaths_df.loc[intervention_years]
    interv_under5_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_under5_deaths_df)
    interv_under5_SAM_deaths_df = under5_SAM_deaths_df.loc[intervention_years]
    interv_under5_SAM_deaths_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_under5_SAM_deaths_df)
    interv_under5_ALRI_deaths_df = under5_ALRI_deaths_df.loc[intervention_years]
    interv_under5_ALRI_deaths_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_under5_ALRI_deaths_df)
    interv_under5_Diarrhoea_deaths_df = under5_Diarrhoea_deaths_df.loc[intervention_years]
    interv_under5_Diarrhoea_deaths_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_under5_Diarrhoea_deaths_df)
    interv_under5_ALRI_deaths_with_SAM_df = under5_ALRI_deaths_with_SAM_df.loc[intervention_years]
    interv_under5_ALRI_deaths_with_SAM_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_under5_ALRI_deaths_with_SAM_df)
    interv_under5_Diarrhoea_deaths_with_SAM_df = under5_Diarrhoea_deaths_with_SAM_df.loc[intervention_years]
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

    # # TODO: rm prints when no longer needed
    # print("\nYears, and (Draws, Runs) with no under 5 death:")
    # no_under5_deaths = [(under5_deaths.index[row], under5_deaths.columns[col]) for row, col in
    #                  zip(*np.where(under5_deaths == 0.0))]
    # print(f"{no_under5_deaths}")
    # #

def extract_daly_data_frames_and_outcomes(
    folder,
    years_of_interest,
    intervention_years,
    interv
) -> Dict[str, pd.DataFrame]:
    """
    Extracts DALYs by cause for under-5s (age_range '0-4'), summed over both sexes, for the specified years.
    :param folder: the folder from which the DALY data will be extracted
    :param years_of_interest: years for which to extract the data
    :param intervention_years: List of years during which the intervention was implemented (if any).
    :param interv: Name or identifier of the intervention.
    :return: DataFrame with index ['year'] and columns for each cause, values are summed DALYs for both sexes
    """

    print(f"\n{interv=}")
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
            df_with_cause_of_dalys[df_with_cause_of_dalys['age_range'] == '0-4']\
                .groupby(['year', 'cause_of_dalys'],as_index=True)['dalys'].sum()

        return under5_dalys_by_year_cause

    under5_dalys_by_cause_df = extract_results(
        folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked_by_age_and_time",
        custom_generate_series=lambda df: extrapolate_dalys_data_from_logs(df),
        do_scaling=True
    ).fillna(0)
    under5_dalys_by_cause_df = under5_dalys_by_cause_df.loc[years_of_interest]

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

    interv_under5_dalys_df = under5_dalys_df.loc[intervention_years]
    interv_under5_dalys_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_under5_dalys_df)
    interv_under5_SAM_dalys_df = under5_SAM_dalys_df.loc[intervention_years]
    interv_under5_SAM_dalys_sum_per_draw_CI_across_runs_df = return_sum_95_CI_across_runs(interv_under5_SAM_dalys_df)
    interv_under5_ALRI_dalys_df = under5_ALRI_dalys_df.loc[intervention_years]
    interv_under5_ALRI_dalys_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_under5_ALRI_dalys_df)
    interv_under5_Diarrhoea_dalys_df = under5_Diarrhoea_dalys_df.loc[intervention_years]
    interv_under5_Diarrhoea_dalys_sum_per_draw_CI_across_runs_df = \
        return_sum_95_CI_across_runs(interv_under5_Diarrhoea_dalys_df)

    return {'under5_dalys_df': under5_dalys_df,
            'under5_SAM_dalys_df': under5_SAM_dalys_df,
            'under5_ALRI_dalys_df': under5_ALRI_dalys_df,
            'under5_Diarrhoea_dalys_df': under5_Diarrhoea_dalys_df,
            'under5_dalys_mean_ci_df': under5_dalys_mean_ci_per_year_per_draw_df,
            'under5_SAM_dalys_mean_ci_df': under5_SAM_dalys_mean_ci_per_year_per_draw_df,
            'under5_ALRI_dalys_mean_ci_df': under5_ALRI_dalys_mean_ci_per_year_per_draw_df,
            'under5_Diarrhoea_dalys_mean_ci_df': under5_Diarrhoea_dalys_mean_ci_per_year_per_draw_df,
            'interv_under5_dalys_df': interv_under5_dalys_df,
            'interv_under5_dalys_sum_ci_df': interv_under5_dalys_sum_per_draw_CI_across_runs_df,
            'interv_under5_SAM_dalys_df': interv_under5_SAM_dalys_df,
            'interv_under5_SAM_dalys_sum_ci_df': interv_under5_SAM_dalys_sum_per_draw_CI_across_runs_df,
            'interv_under5_ALRI_dalys_df': interv_under5_ALRI_dalys_df,
            'interv_under5_ALRI_dalys_sum_ci_df': interv_under5_ALRI_dalys_sum_per_draw_CI_across_runs_df,
            'interv_under5_Diarrhoea_dalys_df': interv_under5_Diarrhoea_dalys_df,
            'interv_under5_Diarrhoea_dalys_sum_ci_df': interv_under5_Diarrhoea_dalys_sum_per_draw_CI_across_runs_df,
            'interv_years': intervention_years}

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
            plt.ylabel(f'{cohort} {outcome_type}')
            plt.xlabel('Year')
            plt.title(f'{cohort} Mean {outcome_type.replace("_", " ")} due to {cause} and 95% CI over time')
            plt.legend()
            plt.xticks(plot_years, labels=plot_years, rotation=45, fontsize=8)

            plt.savefig(
                outputs_path / (
                    f"{cohort}_mean_{cause}_{outcome_type}_CI_scenarios_comparison__"
                    f"{scenarios_tocompare_prefix}__{timestamps_suffix}.png"
                ),
                bbox_inches='tight'
            )

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

def plot_sum_outcome_and_CIs__intervention_period(
    cohort: str,
    scenarios_dict: dict,
    scenarios_to_compare: list,
    outcome_type: str,
    outcomes_dict: dict,
    outputs_path: Path,
    scenarios_tocompare_prefix: str,
    timestamps_suffix: str
) -> None:
    """
    Plots sum of deaths or DALYs and confidence intervals over the intervention period for the specified cohort for
    multiple scenarios.
    :param cohort: 'Neonatal' or 'Under-5'
    :param scenarios_dict: Dictionary mapping interventions to scenarios and their corresponding draw numbers
    :param scenarios_to_compare: List of scenarios to plot
    :param outcome_type: 'deaths', 'deaths_with_SAM' or 'DALYs'
    :param outcomes_dict: Dictionary containing data for plotting nested as outcomes_dict[interv][outcome][draw][run]
    :param outputs_path: Path to save the plot
    :param scenarios_tocompare_prefix: Prefix for output files with names of scenarios that are compared in the plots
    :param timestamps_suffix: Timestamps to identify the log data from which the outcomes originated.
    """
    assert cohort in ['Neonatal', 'Under-5'], \
        f"Invalid value for 'cohort': expected 'Neonatal' or 'Under-5'. Received {cohort} instead."
    assert outcome_type in ['deaths', 'deaths_with_SAM', 'DALYs'], \
        f"Invalid value for 'outcome_type': expected 'deaths' or 'DALYs'. Received {outcome_type} instead."

    averted_DALYs_anycause = dict()

    # Outcome to plot
    for i, cause in enumerate(['any cause', 'SAM', 'ALRI', 'Diarrhoea']):
        if outcome_type == "deaths":
            neonatal_outcomes = ['interv_neo_deaths_sum_ci_df', 'interv_neo_SAM_deaths_sum_ci_df',
                                 'interv_neo_ALRI_deaths_sum_ci_df', 'interv_neo_Diarrhoea_deaths_sum_ci_df']
            under5_outcomes = ['interv_under5_deaths_sum_ci_df', 'interv_under5_SAM_deaths_sum_ci_df',
                               'interv_under5_ALRI_deaths_sum_ci_df', 'interv_under5_Diarrhoea_deaths_sum_ci_df']
        elif outcome_type == "deaths_with_SAM":
            neonatal_outcomes = [None, None,
                                 'interv_neo_ALRI_deaths_with_SAM_sum_ci_df',
                                 'interv_neo_Diarrhoea_deaths_with_SAM_sum_ci_df']
            under5_outcomes = [None, None,
                               'interv_under5_ALRI_deaths_with_SAM_sum_ci_df',
                               'interv_under5_Diarrhoea_deaths_with_SAM_sum_ci_df']
        else:  # outcome_type == "DALYs"
            neonatal_outcomes = [None, None, None, None]  # No DALYs for neonatal
            under5_outcomes = ['interv_under5_dalys_sum_ci_df', 'interv_under5_SAM_dalys_sum_ci_df',
                               'interv_under5_ALRI_dalys_sum_ci_df', 'interv_under5_Diarrhoea_dalys_sum_ci_df']
        outcome = neonatal_outcomes[i] if cohort == 'Neonatal' else under5_outcomes[i]

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
                sum_interv_years, ci_lower, ci_upper = zip(*scen_data.values.flatten())

                # Plot the data
                ax.bar(scenario, sum_interv_years[0],
                       yerr=[[sum_interv_years[0] - ci_lower[0]], [ci_upper[0] - sum_interv_years[0]]],
                       label=scenario, color=get_scen_colour(scenario), capsize=5)

                y_top = ax.get_ylim()[1]

                # Add text label for the bar height (sum), above the bar
                ax.text(
                    scenario,
                    sum_interv_years[0] + (y_top * 0.02),  # small offset above the bar
                    f"{sum_interv_years[0]:,.0f}",
                    color='black',
                    ha='center',
                    va='bottom',
                    fontsize=12.5
                )

                # Add text labels for ci_low and ci_upper
                text_color = 'black' if scenario in ['Status Quo'] else 'white'
                ax.text(scenario,
                        ci_upper[0] / 2 + ci_upper[0] / 4 if \
                            ci_upper < y_top / 2 + y_top / 15 else y_top / 2 + y_top / 15,
                        f"{ci_upper[0]:,.0f}", color=text_color, ha='center', va='top', fontsize=12.5)
                ax.text(scenario,
                        ci_upper[0] / 2 - ci_upper[0] / 4 if \
                            ci_upper < y_top / 2 + y_top / 15 else y_top / 2 - y_top / 15,
                        f"{ci_lower[0]:,.0f}", color=text_color, ha='center', va='bottom', fontsize=12.5)

                # Add horizontal lines for Status Quo scenario
                if scenario == 'Status Quo':
                    ax.axhline(y=ci_lower[0], color=get_scen_colour('Status Quo'), linestyle='--', linewidth=1)
                    ax.axhline(y=ci_upper[0], color=get_scen_colour('Status Quo'), linestyle='--', linewidth=1)

            # Add labels, title, and legend
            min_interv_year = min(outcomes_dict["SQ"]['interv_years'])
            max_interv_year = max(outcomes_dict["SQ"]['interv_years'])
            plt.ylabel(f'{cohort} {outcome_type} (Sum over intervention period)')
            plt.xlabel('Scenario')
            plt.title(
                f'{cohort} Sum of {outcome_type.replace("_", " ")} due to {cause} and 95% CI over '
                f'intervention period ({min_interv_year}--{max_interv_year})')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xticks(rotation=45, fontsize=8)

            plt.savefig(
                outputs_path / (
                    f"{cohort}_sum_{cause}_{outcome_type}_CI_intervention_period_scenarios_comparison__"
                    f"{scenarios_tocompare_prefix}__{timestamps_suffix}.png"
                ),
                bbox_inches='tight'
            )

            # Plot sum of averted outcome_type compared to SQ over intervention period,
            # horizontal lines for SQ, bars for interventions
            fig2, ax2 = plt.subplots()

            # Get SQ values
            sq_data = outcomes_dict['SQ'][outcome][0]
            sq_sum, sq_ci_lower, sq_ci_upper = zip(*sq_data.values.flatten())
            sq_sum = sq_sum[0]
            sq_ci_lower = sq_ci_lower[0]
            sq_ci_upper = sq_ci_upper[0]

            for scenario in scenarios_to_compare:
                if scenario == 'Status Quo':
                    # Only horizontal lines, no bar
                    ax2.axhline(y=sq_sum-sq_ci_lower, color=get_scen_colour('Status Quo'), linestyle='--', linewidth=1)
                    ax2.axhline(y=sq_sum-sq_ci_upper, color=get_scen_colour('Status Quo'), linestyle='--', linewidth=1)
                    ax2.axhline(y=sq_sum-sq_sum, color=get_scen_colour('Status Quo'), linestyle='-', linewidth=2,
                                label='SQ')
                else:
                    # Find the corresponding intervention and draw number
                    interv, draw = next(
                        (interv, draw)
                        for interv, scenarios_for_interv_dict in scenarios_dict.items()
                        if scenario in scenarios_for_interv_dict
                        for scen_name, draw in scenarios_for_interv_dict.items()
                        if scen_name == scenario
                    )

                    scen_data = outcomes_dict[interv][outcome][draw]
                    sum_interv_years, ci_lower, ci_upper = zip(*scen_data.values.flatten())
                    averted_sum = sq_sum - sum_interv_years[0]
                    averted_ci_lower = sq_sum - ci_upper[0]
                    averted_ci_upper = sq_sum - ci_lower[0]
                    if outcome_type == "DALYs" and cause=='any cause':
                        averted_DALYs_anycause[scenario] = [averted_sum, averted_ci_lower, averted_ci_upper]
                    ax2.bar(scenario, averted_sum,
                            yerr=[[averted_sum - averted_ci_lower], [averted_ci_upper - averted_sum]],
                            label=scenario, color=get_scen_colour(scenario), capsize=5)
                    y_top2 = ax2.get_ylim()[1]
                    s1 = y_top2 * 0.02  # space between bar and value of the bar
                    ax2.text(scenario, averted_sum + s1 if averted_sum >= 0 else 0 + s1,
                             f"{averted_sum:,.0f}", color=get_scen_colour(scenario), ha='right', va='bottom',
                             fontsize=12.5, fontweight='bold')
                    # ax2.text(scenario, averted_ci_upper / 2 + averted_ci_upper / 4 if \
                    #     averted_ci_upper < y_top2 / 2 + y_top2 / 15 else y_top2 / 2 + y_top2 / 15,
                    #          f"{averted_ci_upper:,.0f}", color='white', ha='center', va='top', fontsize=12.5)
                    # ax2.text(scenario, averted_ci_upper / 2 - averted_ci_upper / 4 if \
                    #     averted_ci_upper < y_top2 / 2 + y_top2 / 15 else y_top2 / 2 - y_top2 / 15,
                    #          f"{averted_ci_lower:,.0f}", color='white', ha='center', va='bottom', fontsize=12.5)

            min_interv_year = min(outcomes_dict["SQ"]['interv_years'])
            max_interv_year = max(outcomes_dict["SQ"]['interv_years'])
            plt.ylabel(f'{cohort}: Averted {outcome_type}, sum over intervention period)')
            plt.xlabel('Scenario')
            plt.title(
                f'{cohort}: Sum of averted {outcome_type.replace("_", " ")} due to {cause} and 95% CI over '
                f'intervention period ({min_interv_year}--{max_interv_year})'
            )
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xticks(rotation=45, fontsize=8)

            plt.savefig(
                outputs_path / (
                    f"{cohort}_sum_averted_{cause}_{outcome_type}_CI_intervention_period_scenarios_comparison__"
                    f"{scenarios_tocompare_prefix}__{timestamps_suffix}.png"
                ),
                bbox_inches='tight'
            )

            def plot_cost_effectiveness(averted_DALYs: dict) -> None:
                # temporarily not from data, but using the numbers printed with run_costing_analysis_wast.py
                # total costs over intervention period, sizes of lower bound and upper bound for the scenarios
                # # 4K pop sim: SQ results_timestamp = '2025-07-15T223713Z'
                # scen_cons_costs_total_ci = {
                #     'Status Quo': [708882572, 133258484, 153507211],
                #     'GM_FullAttend': [711099327, 123777213, 154057510],
                #     'CS_100': [708915417, 128058833, 158695261],
                #     'FS_Full': [730978386, 127992638, 158323175]
                # }
                # 30 K pop sim (after merge): SQ results_timestamp = '2025-07-15T235608Z'
                scen_cons_costs_total_ci = {
                    'Status Quo': [705047700, 63514713, 57951785],
                    'GM_FullAttend': [711723711, 62674858, 56606637],
                    'CS_100': [708560620, 63092643, 59577695],
                    'FS_Full': [727173360, 64844879, 58676862]
                }

                incremental_costs = dict()
                for scen in scen_cons_costs_total_ci.keys():
                    if scen != 'Status Quo':
                        incremental_costs[scen] = \
                            scen_cons_costs_total_ci[scen][0] - scen_cons_costs_total_ci['Status Quo'][0]
                print(f"\naverted_DALYs:\n{averted_DALYs}")
                print(f"\nincremental_costs:\n{incremental_costs}")

                # Plot cost-effectiveness scatter plot
                fig_ce, ax_ce = plt.subplots()
                ha_scen = ['left', 'right', 'center']
                ha_i = -1
                for scen in incremental_costs.keys():
                    ha_i += 1
                    scen_cons_cost_per_DALY = incremental_costs[scen]/averted_DALYs[scen][0]
                    ax_ce.errorbar(
                        averted_DALYs[scen][0], incremental_costs[scen],
                        xerr=[[averted_DALYs[scen][0] - averted_DALYs[scen][1]],
                              [averted_DALYs[scen][2] - averted_DALYs[scen][0]]],
                        fmt='o', color=get_scen_colour(scen), capsize=5
                    )
                    # ax_ce.text(averted_DALYs[scen][0], incremental_costs[scen] + 1 * incremental_costs[scen],
                    #            scen,
                    #            fontsize=12, ha='center', va='bottom', color=get_scen_colour(scen))
                    # Add a legend box with scenario labels instead of text above points
                    ax_ce.legend([scen for scen in incremental_costs.keys()], loc='best', fontsize=12)
                    ax_ce.text(averted_DALYs[scen][0], incremental_costs[scen] + 0.5 * incremental_costs['CS_100'],
                               f"${scen_cons_cost_per_DALY:,.2f}/DALY",
                               fontsize=12, ha=ha_scen[ha_i % 3], va='bottom', color=get_scen_colour(scen))

                ax_ce.set_xlabel('DALYs Averted')
                ax_ce.set_ylabel('Incremental Costs (2023 USD)')
                ax_ce.set_title('Cost-Effectiveness: DALYs Averted vs Incremental Costs')

                # Add dashed black line for 1 DALY averted per 80 USD
                x_vals = np.array(ax_ce.get_xlim())
                y_vals = x_vals * 80
                ax_ce.plot(x_vals, y_vals, color='black', linestyle='--')
                ax_ce.text(x_vals[-1], y_vals[-1], 'ICER = $80/DALY', color='black', fontsize=9, ha='right', va='bottom')

                plt.tight_layout()
                plt.savefig(
                    outputs_path / (
                        f"cost_effectiveness_scatter_DALYsAverted_vs_IncrementalCosts__"
                        f"{scenarios_tocompare_prefix}__{timestamps_suffix}.png"
                    ),
                    bbox_inches='tight'
                )

            if outcome_type == "DALYs" and cause == 'any cause':
                plot_cost_effectiveness(averted_DALYs_anycause)


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
    treatment_heatmap_data = \
        pd.DataFrame.from_dict(treatment_availability, orient='index',columns=correct_order_of_fac_levels)
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
