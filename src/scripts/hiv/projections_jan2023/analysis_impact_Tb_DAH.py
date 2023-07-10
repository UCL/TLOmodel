"""Analyse scenarios for impact of TB-related development assistance for health."""

# python src/scripts/hiv/projections_jan2023/analysis_impact_Tb_DAH.py --scenario-outputs-folder outputs\nic503@york.ac.uk
import argparse
import datetime
from tlo import Date
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple
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
outputspath = Path("./outputs/nic503@york.ac.uk")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Get basic information about the results
results_folder = get_scenario_outputs("Tb_DAH_scenarios_test_run-2023-07-09T114603Z", outputspath)[-1]
log = load_pickled_dataframes(results_folder)
info = get_scenario_info(results_folder)
print(info)
params = extract_params(results_folder)
print("the parameter info as follows")
print(params)
params.to_excel(outputspath / "parameters.xlsx")

# Choosing the draw to summarize
# number_runs = info["runs_per_draw"]
# number_draws = info['number_of_draws']
# draw = 0
# def get_person_years(draw, run):
#     log = load_pickled_dataframes(results_folder, draw, run)
#     py_ = log["tlo.methods.demography"]["person_years"]
#     years = pd.to_datetime(py_["date"]).dt.year
#     py = pd.Series(dtype="int64", index=years)
#     for year in years:
#         tot_py = (
#             (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["M"]).apply(pd.Series) +
#             (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["F"]).apply(pd.Series)
#         ).transpose()
#         py[year] = tot_py.sum().values[0]
#
#     py.index = pd.to_datetime(years, format="%Y")
#
#     return py
#
# # For draw 0, get py for all runs
# number_runs = info["runs_per_draw"]
# pyears_summary_per_run = pd.DataFrame(data=None, columns=range(number_runs))
#
# # Draw number (default = 0) is specified above
# for run in range(number_runs):
#     pyears_summary_per_run.iloc[:, run] = get_person_years(draw, run)
#
# pyears_summary = pd.DataFrame()
# pyears_summary["mean"] = pyears_summary_per_run.mean(axis=1)
# pyears_summary["lower"] = pyears_summary_per_run.quantile(0.025, axis=1).values
# pyears_summary["upper"] = pyears_summary_per_run.quantile(0.975, axis=1).values
# pyears_summary.columns = pd.MultiIndex.from_product([[draw], list(pyears_summary.columns)], names=['draw', 'stat'])
# pyears_summary.to_excel(outputspath / "pyears_baseline.xlsx")
number_runs = info["runs_per_draw"]
number_draws = info['number_of_draws']
scenario_mapping = {
    0: "Baseline",
    1: "NoXpert",
    2: "NoCXR",
    3: "Scale-up",
    4: "Outreach"
}
params = params.rename(index=scenario_mapping)

# Print the updated parameter info
print(params)
def get_person_years(draw, run):
    log = load_pickled_dataframes(results_folder, draw, run)
    py_ = log["tlo.methods.demography"]["person_years"]
    years = pd.to_datetime(py_["date"]).dt.year
    py = pd.Series(dtype="int64", index=years)
    for year in years:
        tot_py = (
            (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["M"]).apply(pd.Series) +
            (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["F"]).apply(pd.Series)
        ).transpose()
        py[year] = tot_py.sum().values[0]

    py.index = pd.to_datetime(years, format="%Y")

    return py


# Create a DataFrame to store person years per draw and run
pyears_all = pd.DataFrame()

# Iterate over draws and runs
for draw in range(number_draws):
    pyears_summary_per_run = pd.DataFrame(data=None, columns=range(number_runs))
    for run in range(number_runs):
        pyears_summary_per_run[run] = get_person_years(draw, run)

    # Calculate mean, lower, and upper percentiles
    pyears_summary = pd.DataFrame()
    pyears_summary["mean"] = pyears_summary_per_run.mean(axis=1)
    pyears_summary["lower"] = pyears_summary_per_run.quantile(0.025, axis=1).values
    pyears_summary["upper"] = pyears_summary_per_run.quantile(0.975, axis=1).values

    # Assign draw and stat columns as MultiIndex
    pyears_summary.columns = pd.MultiIndex.from_product([[draw], list(pyears_summary.columns)], names=['draw', 'stat'])

    # Append to the main DataFrame
    pyears_all = pd.concat([pyears_all, pyears_summary], axis=1)

# Print the DataFrame to Excel
pyears_all.to_excel (outputspath / "pyears_all.xlsx")

# Number of TB deaths and mortality rate
results_deaths = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df["date"].dt.year).groupby(
            ["year", "cause"])["person_id"].count()
    ),
    do_scaling=True,
)

# Removes multi-index
results_deaths = results_deaths.reset_index()
print(results_deaths)

tb_deaths = results_deaths.loc[results_deaths["cause"].isin(["AIDS_non_TB", "AIDS_TB", "TB"])]
AIDS_TB = results_deaths.loc[results_deaths["cause"] == "AIDS_TB"]
AIDS_non_TB = results_deaths.loc[results_deaths["cause"] == "AIDS_non_TB"]
TB = results_deaths.loc[results_deaths["cause"] == "TB"]

combined_tb_table = pd.concat([AIDS_non_TB, AIDS_TB, TB])
combined_tb_table.to_excel(outputspath / "combined_tb_tables.xlsx")
scaling_factor_key = log['tlo.methods.demography']['scaling_factor']
print("Scaling Factor Key:", scaling_factor_key)

def get_tb_dalys(df_):
    # Get DALYs of TB
    years = df_['year'].value_counts().keys()
    dalys = pd.Series(dtype='float64', index=years)
    for year in years:
        tot_dalys = df_.drop(columns='date').groupby(['year']).sum().apply(pd.Series)
        dalys[year] = tot_dalys.loc[(year, ['TB (non-AIDS)', 'non_AIDS_TB'])].sum()
    dalys.sort_index()
    return dalys

# Extract DALYs from model and scale
tb_dalys_count = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys",
    custom_generate_series=get_tb_dalys,
    do_scaling=True
)

# Get mean/upper/lower statistics
dalys_summary = summarize(tb_dalys_count).sort_index()
dalys_summary.index = dalys_summary.index.map(scenario_mapping)
#dalys_summary = summarize(get_tb_dalys).loc[0].unstack()
#dalys_summary = (tb_dalys_count).sort_index()
#dalys_summary.index = dalys_summary.index.map(scenario_mapping)
print("DALYs for TB are as follows:")
print(dalys_summary)
dalys_summary.to_excel(outputspath / "summarised_tb_dalys.xlsx")

# secondary outcomes
print(f"Keys of log['tlo.methods.tb']: {log['tlo.methods.tb'].keys()}")

#TB mortality rate
def tb_mortality_rate(results_folder, pyears_summary):
    tb_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(["year", "cause"])["person_id"].count()
        ),
        do_scaling=True,
    )

    # Select only causes AIDS_TB, AIDS_non_TB, and TB
    print(tb_deaths)
    #tb_deaths1 = tb_deaths.loc[tb_deaths['cause'].isin(["AIDS_TB", "TB", "AIDS_non_TB"])]
    tb_deaths1 = summarize(tb_deaths[tb_deaths.index.get_level_values('cause').isin(["AIDS_TB", "TB", "AIDS_non_TB"])])
    print("TB-related deaths as follows:",tb_deaths)

    # Group deaths by year
    #tb_deaths2 = pd.DataFrame(tb_deaths1.groupby(["year"]).sum()).reset_index()
    tb_mortality = pd.DataFrame(tb_deaths1.groupby(["year"], as_index=False).sum())


    print("Tb deaths as follows", tb_mortality)
    #tb_mortality.index= tb_mortality.index.year
    tb_mortality.to_excel(outputspath / "raw_mortality.xlsx")

    # Divide draw/run by the respective person-years from that run
    tb_mortality1 = tb_mortality.reset_index(drop=True).div(pyears_all.reset_index(drop=True), axis='rows')
    print("deaths3 are:", tb_mortality1)
    tb_mortality1.to_excel(outputspath / "mortality_rates.xlsx")

    tb_mortality_rate = {}  # empty dict
    tb_mortality_rate["year"] = tb_mortality.index
    tb_mortality_rate["median"] =tb_mortality1.quantile(0.5, axis=1) * 100000
    tb_mortality_rate["lower"] = tb_mortality1.quantile(0.025, axis=1) * 100000
    tb_mortality_rate["upper"] =tb_mortality1.quantile(0.975, axis=1) * 100000
    return tb_mortality_rate

# Call the function with appropriate arguments
mortality_rates = tb_mortality_rate(results_folder, pyears_all)
mortality_rates_summary = pd.DataFrame.from_dict(mortality_rates)

# Print the resulting mortality rates
# mortality_rates_summary.to_excel(outputspath / "mortality_rates_baseline.xlsx",index=False)
# print(mortality_rates_summary)
# Print scaling factor to population level estimates
print(f"The scaling factor is: {log['tlo.methods.demography']['scaling_factor']}")



# Extracts PLHIV with TB
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
        column="tbNewActiveMdrCases",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
mdr_tb_cases.index = mdr_tb_cases.index.year
mdr_tb = pd.DataFrame(mdr_tb_cases)
mdr_tb.to_excel(outputspath / "new_active_mdr_tb_cases.xlsx")

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

#tb_treatment.index = tb_treatment.index.year,
tb_treatment_cov = pd.DataFrame(tb_treatment)
tb_treatment_cov.to_excel(outputspath / "tb_treatment_coverage_baseline.xlsx")

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
tb_inc.index = tb_inc.index.year
tb_incidence.to_excel(outputspath / "active_tb.xlsx")
#Tb incidence rate
#Tb_inc_rate = (tb_incidence.divide(pyears_all.values, axis=0)) * 100000
Tb_inc_rate = tb_incidence.reset_index(drop=True).div(pyears_all.reset_index(drop=True), axis='rows')
Tb_inc_rate.to_excel(outputspath / "Tb_incidence_rate.xlsx")

#pyears = pyears.reset_index(drop=True)
pyears_summary = pyears_summary.reset_index(drop=True)

print(f"Keys of log['tlo.methods.tb']: {log['tlo.methods.tb'].keys()}")
mdr = log["tlo.methods.tb"]["tb_mdr"]
mdr = mdr.set_index("date")
mdr.to_excel(outputspath / "mdr_sample.xlsx")

#Active Tb prevalence
Tb_prevalence= summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_prevalence",
        column="tbPrevActive",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
Tb_prevalence.index = Tb_prevalence.index.year
Tb_prevalence.to_excel(outputspath / "Tb_prevalence_sample.xlsx")

#Active Tb prevalence in adults
adult_Tb_prevalence= summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_prevalence",
        column="tbPrevActiveAdult",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
adult_Tb_prevalence.index = adult_Tb_prevalence.index.year
adult_Tb_prevalence.to_excel(outputspath / "adult_Tb_prevalence_sample.xlsx")

#Active Tb prevalence in children
child_Tb_prevalence= summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_prevalence",
        column="tbPrevActiveChild",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
child_Tb_prevalence.index = child_Tb_prevalence.index.year
child_Tb_prevalence.to_excel(outputspath / "child_Tb_prevalence_sample.xlsx")

###### PLOTS##################################################
# Plot the bar graph
import matplotlib.pyplot as plt

# Select the columns for mean, lower, and upper
mean_values = dalys_summary.loc[:, (slice(None), 'mean')]
lower_values = dalys_summary.loc[:, (slice(None), 'lower')]
upper_values = dalys_summary.loc[:, (slice(None), 'upper')]

# Plot the bar graph
fig, ax = plt.subplots(figsize=(10, 6))
mean_values.plot.bar(ax=ax, yerr=[mean_values.values - lower_values.values, upper_values.values - mean_values.values])

# Set the labels and title
ax.set_xlabel('Draw')
ax.set_ylabel('DALYs')
ax.set_title('DALYs for TB by Draw')

# Show the plot
plt.show()





