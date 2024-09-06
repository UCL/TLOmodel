"""Analyse scenarios for impact of TB-related development assistance for health."""

# python src/scripts/hiv/projections_jan2023/analysis_tb_DAH_scenarios.py --scenario-outputs-folder outputs\newton.chagoma@york.ac.uk
import argparse
from typing import Optional, Iterable, List
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple
from tlo import Date
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize, parse_log_file,

)

resourcefilepath = Path("./resources")
outputspath = Path("./outputs/newton.chagoma@york.ac.uk")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Get basic information about the results
#Tb_DAH_scenarios_test_run09_partial-2023-09-14T125620Z
#tb_DAH_scenarios-2023-09-18T132119Z
# Tb_DAH_scenarios_test_run09_partial-2023-10-01T133822Z -looks to work fine
#Tb_DAH_scenarios_test_run13_partial-2023-10-02T144642Z xcept for CXR scaleup and outreach
#tb_DAH_impact-2023-10-07T150348Z---main results
 #Tb_DAH_impact_partial55-2023-11-13T212007Z 10k pop
results_folder = get_scenario_outputs("tb_DAH_impact-2023-12-04T222317Z", outputspath)[-1]
log = load_pickled_dataframes(results_folder)
info = get_scenario_info(results_folder)
print(info)
#info.to_excel(outputspath / "info.xlsx")
params = extract_params(results_folder)
print("the parameter info as follows")
params.to_excel(outputspath / "parameters.xlsx")

print(f"Keys of log['tlo.methods.healthsystem.summary']: {log['tlo.methods.healthsystem.summary'].keys()}")
number_runs = info["runs_per_draw"]
number_draws = info['number_of_draws']

print(f"Keys of log['tlo.methods.tb']: {log['tlo.methods.tb'].keys()}")
print(f"Keys of log['tlo.methods.healthburden']: {log['tlo.methods.healthburden'].keys()}")
# print(f"Keys of log['tlo.methods.demography']: {log['tlo.methods.demography.detail'].keys()}")


def get_parameter_names_from_scenario_file() -> Tuple[str]:
    """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
    from scripts.hiv.projections_jan2023.tb_DAH_impact01 import ImpactOfTbDaH
    e = ImpactOfTbDaH()
    return tuple(e._scenarios.keys())

def set_param_names_as_column_index_level_0(_df):
    """Set the columns index (level 0) as the param_names."""
    ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
    assert len(names_of_cols_level0) == len(_df.columns.levels[0])
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df


# %% Define parameter names
param_names = get_parameter_names_from_scenario_file()
print(param_names)
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
pyears_all = pyears_all.pipe(set_param_names_as_column_index_level_0)
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
).pipe(set_param_names_as_column_index_level_0)

# Removes multi-index
results_deaths = results_deaths.reset_index()
print("deaths as follows:")
print(results_deaths)

tb_deaths = results_deaths.loc[results_deaths["cause"].isin(["AIDS_non_TB", "AIDS_TB", "TB"])]
print(tb_deaths)
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
tb_dalys = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_stacked",
    custom_generate_series=get_tb_dalys,
    do_scaling=False
).pipe(set_param_names_as_column_index_level_0)

# Get mean/upper/lower statistics
print("Shape of tb_dalys:", tb_dalys.shape)
dalys_summary = summarize(tb_dalys).sort_index()
print("Shape after summarize:", dalys_summary.shape)

dalys_summary = summarize(tb_dalys).sort_index()
print("DALYs for TB are as follows:")
print(dalys_summary)
dalys_summary.to_excel(outputspath / "summarised_tb_dalys.xlsx")

#extracts dalys by SES groups
# def get_total_num_dalys_by_wealth_and_label(_df):
#     """
#     Return the total number of DALYS for all years by wealth and cause label.
#
#     Parameters:
#     - _df: DataFrame containing DALY data
#     """
#     wealth_cats = {5: '0-19%', 4: '20-39%', 3: '40-59%', 2: '60-79%', 1: '80-100%'}
#
#     value_sum = _df \
#         .drop(columns=['date', 'year']) \
#         .assign(
#             li_wealth=lambda x: x['li_wealth'].map(wealth_cats).astype(
#                 pd.CategoricalDtype(wealth_cats.values(), ordered=True))
#         ) \
#         .melt(id_vars=['li_wealth'], var_name='label') \
#         .groupby(by=['li_wealth', 'label'])['value'].sum()
#
#     return value_sum
#
# # Assuming you have a DataFrame named df
# total_num_dalys_by_wealth_and_label = extract_results(
#     results_folder,
#     module="tlo.methods.healthburden",
#     key="dalys_by_wealth_stacked_by_age_and_time",
#     custom_generate_series=get_total_num_dalys_by_wealth_and_label,
#     do_scaling=True
# ).pipe(set_param_names_as_column_index_level_0)
#
# print("Shape of wealth_dalys:", total_num_dalys_by_wealth_and_label.shape)


#raw mortality
def tb_mortality0(results_folder):
    tb_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(["year", "cause"])["person_id"].count()
        ),
        do_scaling=True,
    ).pipe(set_param_names_as_column_index_level_0)

    # Select only causes AIDS_TB, AIDS_non_TB, and TB
    tb_deaths = tb_deaths.sort_index()
    tb_deaths1 = summarize(tb_deaths[tb_deaths.index.get_level_values('cause').isin(["AIDS_TB", "TB", "AIDS_non_TB"])]).sort_index()
    tb_deaths1["year"] = tb_deaths1.index.get_level_values("year")  # Extract the 'year' values from the index
    tb_deaths1.reset_index(drop=True, inplace=True)

    # Group deaths by year
    tb_mortality = pd.DataFrame(tb_deaths1.groupby(["year"], as_index=False).sum())
    tb_mortality.set_index("year", inplace=True)
    return tb_mortality

#printing file to excel
tb_mortality = tb_mortality0(results_folder)
tb_mortality.to_excel(outputspath / "raw_mortality.xlsx")

#TB mortality rate
def tb_mortality_rate(results_folder, pyears_all):
    tb_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(["year", "cause"])["person_id"].count()
        ),
        do_scaling=True,
    ).pipe(set_param_names_as_column_index_level_0)

    # Select only causes AIDS_TB, AIDS_non_TB, and TB
    tb_deaths = tb_deaths.sort_index()
    tb_deaths1 = summarize(tb_deaths[tb_deaths.index.get_level_values('cause').isin(["AIDS_TB", "TB", "AIDS_non_TB"])]).sort_index()
    tb_deaths1["year"] = tb_deaths1.index.get_level_values("year")  # Extract the 'year' values from the index
    tb_deaths1.reset_index(drop=True, inplace=True)

    # Group deaths by year
    tb_mortality = pd.DataFrame(tb_deaths1.groupby(["year"], as_index=False).sum())
    tb_mortality.set_index("year", inplace=True)

    # Divide draw/run by the respective person-years from that run
    tb_mortality1 = tb_mortality.reset_index(drop=True).div(pyears_all.reset_index(drop=True), axis='rows')
    print("Tb mortality pattern as follows:", tb_mortality1)
    tb_mortality1.to_excel(outputspath / "mortality_rates.xlsx")

    tb_mortality_rate = {}  # empty dict
    tb_mortality_rate["median"] =tb_mortality1.quantile(0.5, axis=1) * 100000
    tb_mortality_rate["lower"] = tb_mortality1.quantile(0.025, axis=1) * 100000
    tb_mortality_rate["upper"] =tb_mortality1.quantile(0.975, axis=1) * 100000
    return tb_mortality_rate

# Call the function with appropriate arguments
mortality_rates = tb_mortality_rate(results_folder, pyears_all)
mortality_rates_summary = pd.DataFrame.from_dict(mortality_rates)

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
        do_scaling=True,
    ),
    collapse_columns=True,
).pipe(set_param_names_as_column_index_level_0)
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
        do_scaling=True,
    ),
    collapse_columns=True,
).pipe(set_param_names_as_column_index_level_0)

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
            do_scaling=True,
        ),
        collapse_columns=True,
    ).pipe(set_param_names_as_column_index_level_0)

#tb_treatment.index = tb_treatment.index.year,
tb_treatment_cov = pd.DataFrame(tb_treatment)
tb_treatment_cov.to_excel(outputspath / "tb_treatment_coverage.xlsx")

#TB Incidence
tb_inc = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=True,
    ),
    collapse_columns=True,
).pipe(set_param_names_as_column_index_level_0)
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
mdr.to_excel(outputspath / "mdr_summary.xlsx")



#Active Tb prevalence
Tb_prevalence= summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_prevalence",
        column="tbPrevActive",
        index="date",
        do_scaling=True,
    ),
    collapse_columns=True,
).pipe(set_param_names_as_column_index_level_0)
Tb_prevalence.index = Tb_prevalence.index.year
Tb_prevalence.to_excel(outputspath / "Tb_prevalence.xlsx")

#Active Tb prevalence in adults
adult_Tb_prevalence= summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_prevalence",
        column="tbPrevActiveAdult",
        index="date",
        do_scaling=True,
    ),
    collapse_columns=True,
).pipe(set_param_names_as_column_index_level_0)
adult_Tb_prevalence.index = adult_Tb_prevalence.index.year
adult_Tb_prevalence.to_excel(outputspath / "adult_Tb_prevalence.xlsx")

#Active Tb prevalence in children
child_Tb_prevalence= summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_prevalence",
        column="tbPrevActiveChild",
        index="date",
        do_scaling=True,
    ),
    collapse_columns=True,
).pipe(set_param_names_as_column_index_level_0)

child_Tb_prevalence.index = child_Tb_prevalence.index.year
child_Tb_prevalence.to_excel(outputspath / "child_Tb_prevalence.xlsx")

#properties of deceased
properties_of_deceased_persons = log["tlo.methods.demography.detail"]["properties_of_deceased_persons"]
properties_of_deceased_persons= properties_of_deceased_persons.set_index("date")
properties_of_deceased_persons.to_excel(outputspath / "properties_of_deceased_persons.xlsx")

# wealth_quintile = extract_results(
#     results_folder,
#     module="tlo.methods.demography.detail",
#     key="tb_properties_of_deceased_persons",
#     column="lil_wealth",
#     index="date",
#     do_scaling=False,
#     #collapse_columns=True
# ).pipe(set_param_names_as_column_index_level_0)

# wealth_quintile.index = wealth_quintile.index.year
# wealth_quintile.to_excel(outputspath / "wealth_quintiles.xlsx")

#HSE = log["tlo.methods.healthsystem.summary"]["hsi_event_details"]["Tb_Test_Screening"]

HSE = log["tlo.methods.healthsystem.summary"]["hsi_event_details"]
HSE = HSE.set_index("date")
print("Health system events as follows",HSE)
HSE.to_excel(outputspath / "HSE.xlsx")

#HSEvents = log["tlo.methods.healthsystem.summary"]["HSI_Event"]["Tb_Test_Screening"]
HSEvents = log["tlo.methods.healthsystem.summary"]["HSI_Event"]
HSEvents = HSEvents.set_index("date")
print("Health system events as follows",HSEvents)
HSEvents.to_excel(outputspath / "HSEvents.xlsx")
Tb_screening = HSEvents.loc[HSEvents['TREATMENT_ID'] == "Tb_Test_Screening"]
Tb_screening.to_excel(outputspath / "Tb_screening.xlsx")

hsi_event_counts = log["tlo.methods.healthsystem.summary"]["hsi_event_counts"]
hsi_event_counts = hsi_event_counts.set_index("date")
print("Health system events as follows",hsi_event_counts)
hsi_event_counts.to_excel(outputspath / "hsi_event_counts")
print(hsi_event_counts)

print(f"Keys of log['tlo.methods.healthsystem.summary']: {log['tlo.methods.healthsystem.summary'].keys()}")
#print(f"Keys of log['tlo.methods.tb']: {log['tlo.methods.tb'].keys()}")
# print(f"Keys of log['tlo.methods.healthburden']: {log['tlo.methods.healthburden'].keys()}")
# print(f"Keys of log['tlo.methods.demography']: {log['tlo.methods.demography.detail'].keys()}")
#
# #aiming to extract wealth quintiles for the dead
# properties_of_deceased_persons = log["tlo.methods.demography.detail"]["properties_of_deceased_persons"]
# properties_of_deceased_persons= properties_of_deceased_persons.set_index("date")
# properties_of_deceased_persons.to_excel(outputspath / "properties_of_deceased_persons.xlsx")

HSEvents = log["tlo.methods.healthsystem.summary"]["HSI_Event"]
HSEvents = HSEvents.set_index("date")
print("Health system events as follows", HSEvents)
HSEvents.to_excel(outputspath / "HSEvents.xlsx")

Tb_tx_coverage = log["tlo.methods.tb"]["tb_treatment"]
Tb_tx_coverage = Tb_tx_coverage.set_index("date")
Tb_tx_coverage.index = pd.to_datetime(Tb_tx_coverage.index)
Tb_tx_coverage.to_excel(outputspath / "Tb_tx_coverage.xlsx")

## extracts number of people screened for TB by scenario
TARGET_PERIOD = (Date(2010, 1, 1), Date(2033, 12, 31))
def get_counts_of_hsi_by_treatment_id(_df):
    """Get the counts of the TREATMENT_IDs occurring"""
    _counts_by_treatment_id = _df \
        .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'TREATMENT_ID'] \
        .apply(pd.Series) \
        .sum() \
        .astype(int)
    return _counts_by_treatment_id.groupby(level=0).sum()

counts_of_hsi_by_treatment_id = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='HSI_Event',
            custom_generate_series=get_counts_of_hsi_by_treatment_id,
            do_scaling=False,  # Counts of HSI shouldn't be scaled for this investigation
        ).pipe(set_param_names_as_column_index_level_0),
        only_mean=True,
    )
counts_of_hsi_by_treatment_id.to_excel(outputspath / "Tb_Test_Screening.xlsx")

###### PLOTS##################################################
# Calculate the sum of DALYs across years for each scenario
baseline_total = dalys_summary.loc[:, ('Baseline', 'mean')].sum()
No_Xpert_total = dalys_summary.loc[:, ('No Xpert Available', 'mean')].sum()
No_CXR_total = dalys_summary.loc[:, ('No CXR Available', 'mean')].sum()
CXR_scaleup_total = dalys_summary.loc[:, ('CXR scaleup', 'mean')].sum()
outreach_total = dalys_summary.loc[:, ('Outreach services', 'mean')].sum()

# Calculate the corresponding lower and upper bounds
baseline_lower = dalys_summary.loc[:, ('Baseline', 'lower')].sum()
baseline_upper = dalys_summary.loc[:, ('Baseline', 'upper')].sum()

No_Xpert_lower = dalys_summary.loc[:, ('No Xpert Available', 'lower')].sum()
No_Xpert_upper = dalys_summary.loc[:, ('No Xpert Available', 'upper')].sum()

No_CXR_lower = dalys_summary.loc[:, ('No CXR Available', 'lower')].sum()
No_CXR_upper = dalys_summary.loc[:, ('No CXR Available', 'upper')].sum()

CXR_scaleup_lower = dalys_summary.loc[:, ('CXR scaleup', 'lower')].sum()
CXR_scaleup_upper = dalys_summary.loc[:, ('CXR scaleup', 'upper')].sum()

outreach_lower = dalys_summary.loc[:, ('Outreach services', 'lower')].sum()
outreach_upper = dalys_summary.loc[:, ('Outreach services', 'upper')].sum()

# Plotting bar graph
x = np.arange(5)
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
bar1 = ax.bar(x[0], baseline_total, width, label='Baseline', yerr=[[baseline_total - baseline_lower], [baseline_upper - baseline_total]])
bar2 = ax.bar(x[1], No_Xpert_total, width, label='No Xpert Available', yerr=[[No_Xpert_total - No_Xpert_lower], [No_Xpert_upper - No_Xpert_total]])
bar3 = ax.bar(x[2], No_CXR_total, width, label='No CXR Available', yerr=[[No_CXR_total - No_CXR_lower], [No_CXR_upper - No_CXR_total]])
bar4 = ax.bar(x[3], CXR_scaleup_total, width, label='CXR scale_up', yerr=[[CXR_scaleup_total - CXR_scaleup_lower], [CXR_scaleup_upper - CXR_scaleup_total]])
bar5 = ax.bar(x[4], outreach_total, width, label='Outreach services', yerr=[[outreach_total - outreach_lower], [outreach_upper - outreach_total]])

# Adding labels and title
ax.set_xlabel('Scenario')
ax.set_ylabel('Total DALYs')
ax.set_title('Cumulative TB DALYs 2010-2020')
ax.set_xticks(x)
ax.set_xticklabels(['Baseline', 'No Xpert', 'No CXR', 'CXR scale_up', 'Outreach services'])
ax.legend()

# Displaying graph
plt.show()


##mortality
baseline_total = tb_mortality.loc[:, ('Baseline', 'mean')].sum()
No_Xpert_total = tb_mortality.loc[:, ('No Xpert Available', 'mean')].sum()
No_CXR_total = tb_mortality.loc[:, ('No CXR Available', 'mean')].sum()
CXR_scaleup_total = tb_mortality.loc[:, ('CXR scaleup', 'mean')].sum()
outreach_total = tb_mortality.loc[:, ('Outreach services', 'mean')].sum()

# Calculate the corresponding lower and upper bounds
baseline_lower = tb_mortality.loc[:, ('Baseline', 'lower')].sum()
baseline_upper = tb_mortality.loc[:, ('Baseline', 'upper')].sum()

No_Xpert_lower = tb_mortality.loc[:, ('No Xpert Available', 'lower')].sum()
No_Xpert_upper = tb_mortality.loc[:, ('No Xpert Available', 'upper')].sum()

No_CXR_lower = tb_mortality.loc[:, ('No CXR Available', 'lower')].sum()
No_CXR_upper = tb_mortality.loc[:, ('No CXR Available', 'upper')].sum()

CXR_scaleup_lower = tb_mortality.loc[:, ('CXR scaleup', 'lower')].sum()
CXR_scaleup_upper = tb_mortality.loc[:, ('CXR scaleup', 'upper')].sum()

outreach_lower = tb_mortality.loc[:, ('Outreach services', 'lower')].sum()
outreach_upper = tb_mortality.loc[:, ('Outreach services', 'upper')].sum()

# Plotting bar graph
x = np.arange(5)
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
bar1 = ax.bar(x[0], baseline_total, width, label='Baseline', yerr=[[baseline_total - baseline_lower], [baseline_upper - baseline_total]])
bar2 = ax.bar(x[1], No_Xpert_total, width, label='No Xpert Available', yerr=[[No_Xpert_total - No_Xpert_lower], [No_Xpert_upper - No_Xpert_total]])
bar3 = ax.bar(x[2], No_CXR_total, width, label='No CXR Available', yerr=[[No_CXR_total - No_CXR_lower], [No_CXR_upper - No_CXR_total]])
bar4 = ax.bar(x[3], CXR_scaleup_total, width, label='CXR Scale_up', yerr=[[CXR_scaleup_total - CXR_scaleup_lower], [CXR_scaleup_upper - CXR_scaleup_total]])
bar5 = ax.bar(x[4], outreach_total, width, label='Outreach services', yerr=[[outreach_total - outreach_lower], [outreach_upper - outreach_total]])

# Adding labels and title
ax.set_xlabel('Scenario')
ax.set_ylabel('TB Mortality')
ax.set_title('Cumulative TB Mortality 2010-2020')
ax.set_xticks(x)
ax.set_xticklabels(['Baseline', 'No Xpert', 'No CXR', 'CXR Scale_up', 'Outreach services'])
ax.legend()

# Displaying graph
plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate plot for each scenario",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-path",
        help="Directory to write outputs to. If not specified, outputs will be written to the results directory.",
        type=Path,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--resources-path",
        help="Directory containing resource files",
        type=Path,
        default=Path('resources'),
        required=False,
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        help="Directory containing results from running src/scripts/hiv/projections_jan2023/tb_DAH_scenario.py",
        default= None,
        required=False
    )
    args = parser.parse_args()

    # apply(
    #     results_folder=args.results_path,
    #     output_folder=args.output_path,
    #     resourcefilepath=args.resources_path
    # )
print(f"Keys of log['tlo.methods.tb']: {log['tlo.methods.tb'].keys()}")
print(f"Keys of log['tlo.methods.healthsystem.summary']: {log['tlo.methods.healthsystem.summary'].keys()}")
#print(f"Keys of log['tlo.methods.healthsystem.summary']: {log['tlo.methods.healthsystem.summary'].keys()}")

consumables_list = log['tlo.methods.healthsystem.summary']['Consumables']
# Print the list of consumables
print("List of Consumables:")
print(consumables_list)
#print(params)

print(f"Keys of log['tlo.methods.tb']: {log['tlo.methods.tb'].keys()}")
print(f"Keys of log['tlo.methods.healthsystem.summary']: {log['tlo.methods.healthsystem.summary'].keys()}")
print(f"Keys of log['tlo.methods.healthburden']: {log['tlo.methods.healthburden'].keys()}")
print(f"Keys of log['tlo.methods.demography']: {log['tlo.methods.demography'].keys()}")





