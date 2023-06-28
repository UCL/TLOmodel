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
results_folder = get_scenario_outputs("baseline_tb_services_scenario-2023-06-23T213051Z", outputspath) [-1]
# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)
# get basic information about the results
info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)
draw = 0
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
summary_deaths.to_excel(outputspath / "summary_death_baseline.xlsx")

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
tb_hiv_prop.to_excel(outputspath / "PLHIV_tb_baseline.xlsx")

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
mdr_tb.to_excel(outputspath / "mdr_tb_baseline.xlsx")


# TB treatment coverage
tb_treatment = summarize(
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
#print("Before loading tb_tx")
#print(tb_treatment)
#tb_treatment.index = tb_treatment.index.year,
tb_treatment_cov = pd.DataFrame(tb_treatment)
tb_treatment_cov.to_excel(outputspath / "tb_tx_coverage_baseline.xlsx")

###summarizing incidence and mortality                  ########################
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

pyears = summarize (
    extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="person_years",
    custom_generate_series=get_person_years,
    do_scaling=False
)
)
#print(pyears)
print(pyears.columns)
tb_pyears = pd.DataFrame(pyears)
tb_pyears.to_excel(outputspath / "pyears_baseline.xlsx")

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
print(tb_inc)
tb_incidence = pd.DataFrame(tb_inc)
tb_incidence.to_excel(outputspath / "tb_incidence_baseline.xlsx")

tb_inc.index = tb_inc.index.year
print(tb_inc.head())
print(pyears.head())
tb_inc = tb_inc.reset_index(drop=True)
#pyears = pyears.reset_index(drop=True)
pyears = pyears.reset_index(drop=True)


# activeTB_inc_rate = (tb_inc["mean"] / pyears["mean"]) * 100000
# activeTB_inc_rate= pd.DataFrame(activeTB_inc_rate)
# activeTB_inc_rate.to_excel(outputspath / "tb_incidence_rate_baseline.xlsx")
#
# # Calculate the rates
# activeTB_inc_rate = (tb_inc["mean"] / pyears["mean"]) * 100000
# upper_activeTB_inc_rate = (tb_inc["upper"] / pyears["upper"]) * 100000
# lower_activeTB_inc_rate = (tb_inc["lower"] / pyears["lower"]) * 100000
#
# # Create a new dataframe with the calculated values
# summarized_incidence = pd.DataFrame({
#     "TB_inc_rate": activeTB_inc_rate,
#     "upper": upper_activeTB_inc_rate,
#     "lower": lower_activeTB_inc_rate
# })
#
# # Output the combined table to an Excel file
# summarized_incidence .to_excel(outputspath / "baseline_incidence_rate.xlsx")

#######################################################################
#number of TB deaths and mortality rate
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

# Removes multi-index
results_deaths = results_deaths.reset_index()
print(results_deaths)

tb_deaths = results_deaths.loc[results_deaths.cause.isin(["AIDS_non_TB", "AIDS_TB", "TB"])]
AIDS_TB = results_deaths.loc[results_deaths.cause == "AIDS_non_TB"]
AIDS_non_TB = results_deaths.loc[results_deaths.cause == "AIDS_TB"]
TB = results_deaths.loc[results_deaths.cause == "TB"]

combined_tb_table = pd.concat([AIDS_non_TB, AIDS_TB, TB])
#print(combined_table)
combined_tb_table.to_excel(outputspath / "combined_ttb_able_baseline.xlsx")
scaling_factor_key = log['tlo.methods.demography']['scaling_factor']

print(f"Keys of log['tlo.methods.demography']: {log['tlo.methods.demography'].keys()}")
print("Scaling Factor Key:", scaling_factor_key)
#########################################################################################

# deaths due to TB (not including TB-HIV)

def tb_mortality_rate(results_folder, pyears):
    tb_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(["year", "cause"])["person_id"].count()
        ),
        do_scaling=False,
    )

    # Print the first few rows of tb_deaths DataFrame
    # Select only causes AIDS_TB, AIDS_non_TB, and TB
    tb_deaths1 = results_deaths.loc[(results_deaths.cause == "AIDS_TB") | \
    (results_deaths.cause == "AIDS_non_TB") | (results_deaths.cause == "AIDS_non_TB")
        ]

    # Group deaths by year
    tb_deaths2 = pd.DataFrame(tb_deaths1.groupby(["year"]).sum())

    # Divide each draw/run by the respective person-years from that run
    # Need to reset index as they don't match exactly (date format)
    tb_deaths3 = tb_deaths2.reset_index(drop=True) / pyears.reset_index(drop=True)
    print("deaths3 are:", tb_deaths)
    tb_mortality_rate = {}  # empty dict
    tb_mortality_rate["year"] = tb_deaths2.index
    tb_mortality_rate["median_tb_deaths_rate_100kpy"] = tb_deaths3.quantile(0.5, axis=1) * 100000
    tb_mortality_rate["lower_tb_deaths_rate_100kpy"] = tb_deaths3.quantile(0.025, axis=1) * 100000
    tb_mortality_rate["upper_tb_deaths_rate_100kpy"] = tb_deaths3.quantile(0.975, axis=1) * 100000
    # Convert tb_mortality_rate dictionary into a DataFrame

    return tb_mortality_rate

# Call the function with appropriate arguments
mortality_rates = tb_mortality_rate(results_folder, pyears)
mortality_rates_summary = pd.DataFrame.from_dict(mortality_rates)

# Print the resulting mortality rates
mortality_rates_summary.to_excel(outputspath / "mortality_rates_summary.xlsx",index=False)
print(mortality_rates_summary)


# Summarizing TB DALYs
    # Define function to calculate number of DALYs
#TARGET_PERIOD = (Date(2010, 1, 1), Date(2015, 1, 1))


# def num_dalys(_df):
#     """Return total number of DALYS (Stacked) (total by age-group within the TARGET_PERIOD)"""
#     result = _df \
#         .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
#         .drop(columns=['date', 'sex', 'age_range', 'cause', 'year']).sum()
#     print("DALYs:", result)  # Print the result
#     return result


# def tb_daly_summary(results_folder):
#     dalys = extract_results(
#         results_folder,
#         module='tlo.methods.healthburden',
#         key='dalys_stacked',
#         custom_generate_series=num_dalys,
#         do_scaling=False
#     )
#
#     if dalys is None:
#         raise ValueError("No results were extracted. Check the implementation of extract_results().")
#
#     dalys.columns = dalys.columns.get_level_values(0)
#     print(dalys.head())
#
#     dalys.loc['TB (non-AIDS)'] = dalys.loc['TB (non-AIDS)'] + dalys.loc['non_AIDS_TB']
#     dalys.drop(['non_AIDS_TB'], inplace=True)
#     tb_daly = pd.DataFrame()
#     tb_daly['median'] = dalys.median(axis=1).round(decimals=-3).astype(int)
#     tb_daly['lower'] = dalys.quantile(q=0.025, axis=1).round(decimals=-3).astype(int)
#     tb_daly['upper'] = dalys.quantile(q=0.975, axis=1).round(decimals=-3).astype(int)
#     return tb_daly
#
#
# # Call the function with appropriate arguments
# tb_daly = tb_daly_summary(results_folder)
#
# # Print the resulting TB DALYs
# print(tb_daly)
#
# # Call the function with appropriate arguments
# tb_daly = tb_daly_summary(results_folder)
#
# # Print the resulting TB DALYs
# print(tb_daly)
#
#
# #Call the function to get the results
# tb_dalys = tb_daly_summary(results_folder)
#
# # Export results to Excel
# tb_dalys.index = tb_dalys.index.year,
# summary_dalys = pd.DataFrame(tb_dalys)
# summary_dalys.to_excel(outputspath / "summary_tb_dalys_nocxr.xlsx")
print(f"Keys of log['tlo.methods.healthburden']: {log['tlo.methods.healthburden'].keys()}")
def calculate_summary_dalys(log):
    dalys = log['tlo.methods.healthburden']['dalys']
    dalys1 = dalys.drop(columns='Other').groupby('year').sum()
    tb_dalys = pd.DataFrame(dalys1)
    #print(tb_dalys[1:10])
    summary_dalys = {}  # empty dict
    summary_dalys["year"] = tb_dalys.index
    #tb_dalys.index=tb_dalys.index.year
    summary_dalys["median"] = tb_dalys.quantile(0.5, axis=1)
    summary_dalys["lower"] = tb_dalys.quantile(0.025, axis=1)
    summary_dalys["upper"] = tb_dalys.quantile(0.975, axis=1)
    return summary_dalys

log = load_pickled_dataframes(results_folder)
summary_tb_dalys = calculate_summary_dalys(log)
#summarised_tb_dalys = summary_tb_dalys(results_folder)
summarised_tb_dalys = pd.DataFrame.from_dict(summary_tb_dalys)
print(summarised_tb_dalys)
summarised_tb_dalys.to_excel(outputspath / "tb_dalys_baseline.xlsx",index=False)

#print scaling factor  to population level estimates
print(f"The scaling factor is: {log['tlo.methods.demography']['scaling_factor']}")

def get_tb_dalys(df_):
    # get dalys of TB
    years = df_['year'].value_counts().keys()
    dalys = pd.Series(dtype='float64', index=years)
    for year in years:
        tot_dalys = df_.drop(columns='date').groupby(['year']).sum().apply(pd.Series)
        dalys[year] = tot_dalys.loc[(year, ['TB (non-AIDS)', 'non_AIDS_TB'])].sum()
    dalys.sort_index()
    return dalys


# extract dalys from model and scale
tb_dalys_count = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys",
    custom_generate_series=get_tb_dalys,
    do_scaling=False
)

# get mean / upper/ lower statistics
dalys_summary = summarize(tb_dalys_count).sort_index()
print("DALYs for TB are as follows:")
print(dalys_summary.head())
summarised_tb_dalys.to_excel(outputspath / "summarised_tb_dalys_baseline.xlsx",index=False)


#summary_tb_dalys = tb_dalys.groupby(dalys['year']).quantile([0.025, 0.5, 0.975]).unstack(level=1).rename(columns={0.025: 'lower', 0.5: 'median', 0.975: 'upper'})
#print(summary_tb_dalys[1:10])

#dalys2 = dalys[['AIDS', 'TB (non-AIDS)', 'non_AIDS_TB']].sum(axis=1).rename('TB_DALYs').reset_index()
#print(dalys2 [1:10])

# dalys2 = dalys[['AIDS', 'TB (non-AIDS)', 'non_AIDS_TB']].sum(axis=1).rename('TB_DALYs')
# print(dalys2.head())
# quantiles = dalys2.groupby(dalys['year']).quantile([0.025, 0.5, 0.975]).unstack(level=1).rename(columns={0.025: 'lower', 0.5: 'median', 0.975: 'upper'})
#print(quantiles[1:10])

#dalys['year'] = dalys2['year']  # Add the 'year' column to dalys DataFrame
#quantiles = dalys2.groupby('year').quantile([0.025, 0.5, 0.975]).unstack(level=1).rename(columns={0.025: 'lower', 0.5: 'median', 0.975: 'upper'})
#print(quantiles[1:10])

#dalys.index = dalys.index.astype(float())
#percentiles = dalys[['AIDS', 'TB (non-AIDS)', 'non_AIDS_TB']].quantile([0.025, 0.5, 0.975], axis=1).reset_index()

#print(percentiles [1:10])
#dalys1 = dalys1.sort_index()

#print(dalys1.head())

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Analyse scenario results for noxpert diagnosis pathway"
    )
    parser.add_argument(
        "--scenario-outputs-folder",
        type=Path,
        required=True,
        help="Path to folder containing scenario outputs",
    )
    parser.add_argument(
        "--show-figures",
        action="store_true",
        help="Whether to interactively show generated Matplotlib figures",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Whether to save generated Matplotlib figures to results folder",
    )
    args = parser.parse_args()
















