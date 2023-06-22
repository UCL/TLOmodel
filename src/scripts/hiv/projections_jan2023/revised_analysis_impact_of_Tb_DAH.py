"""Analyse scenarios for impact of TB-related development assistance for health."""

# python src/scripts/hiv/projections_jan2023/revised_analysis_impact_of_Tb_DAH.py --scenario-outputs-folder outputs\nic503@york.ac.uk
import argparse
import datetime
from tlo import Date
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)
resourcefilepath = Path("./resources")
# datestamp = datetime.date.today().strftime("__%Y_%m_%d")
outputspath = Path("./outputs/nic503@york.ac.uk")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Get basic information about the results
#results_folder = (outputspath / 'impact_of_Tb_DAH_baseline-2023-06-14T133033Z')
results_folder = get_scenario_outputs("noncxr_tb_scenario-2023-06-22T153114Z", outputspath) [-1]
# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)
# get basic information about the results
info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# Defines functions to extract relevant results
tb_death_count = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df["date"].dt.year).groupby(["year"])["year"].count()
    ),
    do_scaling=False,
)
summary_deaths = summarize(tb_death_count)
summary_deaths.to_excel(outputspath / "summary_death_xpert.xlsx")

#prints dictionary keys for the TB module
print(f"Keys of log['tlo.methods.tb']: {log['tlo.methods.tb'].keys()}")
# extracts PLHIV with TB
tb_hiv_prop = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="prop_active_tb_in_plhiv",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)

tb_hiv_prop.index = tb_hiv_prop.index.year
tb_hiv_prop_with_year = pd.DataFrame(tb_hiv_prop)
tb_hiv_prop.to_excel(outputspath / "PLHIV_tb_xpert.xlsx")

#MDR TB cases
mdr_tb_cases = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_mdr",
        column="tbPropActiveCasesMdr",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
mdr_tb_cases.index = mdr_tb_cases.index.year
mdr_tb = pd.DataFrame(mdr_tb_cases)
mdr_tb.to_excel(outputspath / "mdr_tb_xpert.xlsx")

# TB treatment coverage
tb_tx = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.tb",
            key="tb_treatment",
            column="tbTreatmentCoverage",
            index="date",
            do_scaling=False,
        ),
        collapse_columns=True,
    )
tb_tx.index = tb_tx.index.year,
tb_tx_coverage = pd.DataFrame(tb_tx)
tb_tx_coverage.to_excel(outputspath / "tb_tx_coverage_xpert.xlsx")

   ###summarizing incidence and mortality########################
    # computing person years to used as denominators for mortality rate and incidence
def get_person_years(_df):
    """ extract person-years for each draw/run
    sums across men and women
    will skip column if particular run has failed
    """
    years = pd.to_datetime(_df["date"]).dt.year
    py = pd.Series(dtype="int64", index=years)
    for year in years:
        tot_py = (
            (_df.loc[pd.to_datetime(_df["date"]).dt.year == year]["M"]).apply(pd.Series) +
            (_df.loc[pd.to_datetime(_df["date"]).dt.year == year]["F"]).apply(pd.Series)
        ).transpose()
        py[year] = tot_py.sum().values[0]
    py.index = pd.to_datetime(years, format="%Y")
    return py

pyears = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="person_years",
    custom_generate_series=get_person_years,
    do_scaling=False
)

#TB Incidence
tb_inc = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
tb_inc.index = tb_inc.index.year
activeTB_inc_rate = (tb_inc.divide(pyears["mean"].values, axis=0)) * 100000
activeTB_inc_rate_df = pd.DataFrame(activeTB_inc_rate)

# write files to excel
active_tb_cases = "tb_inc_activeTB_inc_rate_xpert.xlsx"
with pd.ExcelWriter(active_tb_cases) as writer:
    tb_inc.to_excel(writer, sheet_name="tb_inc")
    activeTB_inc_rate_df.to_excel(writer, sheet_name="activeTB_inc_rate_xpert")

#######################################################################
# number of deaths and mortality rate
# deaths due to TB (not including TB-HIV)
def summarise_tb_deaths(results_folder, person_years):
    results_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(
                ["year", "cause"])["person_id"].count()
        ),
        do_scaling=False,
    )
    # removes multi-index
    results_deaths = results_deaths.reset_index()
    # select only cause AIDS_TB and AIDS_non_TB
    tmp = results_deaths.loc[results_deaths.cause == "TB"]
    # group deaths by year
    tmp2 = pd.DataFrame(tmp.groupby(["year"]).sum())
    # divide each draw/run by the respective person-years from that run
    # need to reset index as they don't match exactly (date format)
    tmp3 = tmp2.reset_index(drop=True) / (pyears.reset_index(drop=True))


# Summarizing TB DALYs
    # Define function to calculate number of DALYs
TARGET_PERIOD = (Date(2010, 1, 1), Date(2015, 1, 1))
def num_dalys(_df):
    """Return total number of DALYS (Stacked) (total by age-group within the TARGET_PERIOD)"""
    return _df \
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
        .drop(columns=['date', 'sex', 'age_range', 'cause', 'year']) \
        .sum()

# Define function to summarize TB DALYs
def tb_daly_summary(results_folder):
    dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=num_dalys,
        do_scaling=False
    )
    dalys.columns = dalys.columns.get_level_values(0)
    dalys.loc['TB (non-AIDS)'] = dalys.loc['TB (non-AIDS)'] + dalys.loc['non_AIDS_TB']
    dalys.drop(['non_AIDS_TB'], inplace=True)
    tb_daly = pd.DataFrame()
    tb_daly['median'] = dalys.median(axis=1).round(decimals=-3).astype(int)
    tb_daly['lower'] = dalys.quantile(q=0.025, axis=1).round(decimals=-3).astype(int)
    tb_daly['upper'] = dalys.quantile(q=0.975, axis=1).round(decimals=-3).astype(int)
    return tb_daly

# Call the function to get the results
tb_dalys = tb_daly_summary(results_folder)

# Export results to Excel
tb_dalys.index = tb_dalys.index.year,
summary_dalys = pd.DataFrame(tb_dalys)
summary_dalys.to_excel(outputspath / "summary_tb_dalys_xpert.xlsx")


















