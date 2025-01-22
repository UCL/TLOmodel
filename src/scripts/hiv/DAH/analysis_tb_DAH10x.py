"""Analyse scenarios for impact of TB-related development assistance for health."""

#python src/scripts/hiv/DAH/analysis_tb_DAH10x.py --scenario-outputs-folder outputs/newton.chagoma@york.ac.uk
# to parse files use: tlo parse-log outputs/filename/x/y where and the x and y rep number of draws and runs
#from matplotlib.ticker import FuncFormatter
#import squarify

import argparse
import datetime
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from tlo import Date
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,

    parse_log_file,
    summarize,
    unflatten_flattened_multi_index_in_logging,
)

datestamp = datetime.date.today().strftime("__%Y_%m_%d")
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

#creating folders to store results
#resourcefilepath = Path(".\resources")
outputfilepath = Path(r".\outputs\newton.chagoma@york.ac.uk")

#outputfilepath = Path("./outputs")

results_folder = get_scenario_outputs('tb_DAH_scenarios2x-2025-01-22T085231Z', outputfilepath) [-1]
log = load_pickled_dataframes(results_folder)
info = get_scenario_info(results_folder)
print(info)
#info.to_excel(Unresolved reference 'outputspath' / "info.xlsx")
params = extract_params(results_folder)
print("the parameter info as follows")
params.to_excel(outputfilepath / "parameters.xlsx")

number_runs = info["runs_per_draw"]
number_draws = info['number_of_draws']

def get_parameter_names_from_scenario_file() -> Tuple[str]:
    """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
    from scripts.hiv.DAH.tb_DAH_scenarios10x import ImpactOfTbDaH04
    e = ImpactOfTbDaH04()
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
pyears_all.to_excel (outputfilepath / "pyears_all.xlsx")

import pandas as pd

# Check if the key 'cause' exists in the log data
if "cause" in log["tlo.methods.demography"]:
    tb_deaths = log["tlo.methods.demography"]["cause"]
    # Ensure it's a pandas Series for filtering
    if isinstance(tb_deaths , pd.Series):
        filtered_data = tb_deaths [
            tb_deaths.isin(["AIDS_non_TB", "AIDS_TB", "TB"])
        ]
        tb_deaths = filtered_data.reset_index()  # Reset index for cleaner Excel output
        output_file_path = "filtered_deceased_persons.xlsx"
        tb_deaths.to_excel(output_file_path, index=False)

        print(f"Filtered data saved to {output_file_path}.")
    else:
        print("Error: 'cause' is not a pandas Series or is of unexpected type.")
else:
    print("Error: 'cause' key not found in log['tlo.methods.demography.detail'].")



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
combined_tb_table.to_excel(outputfilepath / "combined_tb_tables.xlsx")
scaling_factor_key = log['tlo.methods.demography']['scaling_factor']
print("Scaling Factor Key:", scaling_factor_key)


#Extracting DALYs
def get_tb_dalys(df_):
    # Get DALYs of TB
    years = df_['year'].unique()  # Get unique years
    dalys = pd.Series(dtype='float64', index=years)

    for year in years:
        # Group data by year and sum relevant columns
        tot_dalys = df_.drop(columns='date').groupby('year').sum()

        # Ensure the labels exist before summing
        if any(label in tot_dalys.columns for label in ["AIDS_TB", "TB", "AIDS_non_TB"]):
            # Sum the DALYs for the specified labels for the year
            dalys[year] = tot_dalys.loc[year, ["AIDS_TB", "TB", "AIDS_non_TB"]].sum()
        else:
            dalys[year] = 0  # Set it to 0 if the labels are not found

    dalys.sort_index(inplace=True)  # Sort the index inplace
    return dalys


def get_tb_dalys(df_):
    """
    Get DALYs of TB by labels containing 'TB'.
    """
    # Get unique years from the data
    years = df_['year'].value_counts().keys()
    dalys = pd.Series(dtype='float64', index=years)
    for year in years:
        # Group data by year and sum relevant columns
        tot_dalys = df_.drop(columns='date').groupby(['year']).sum().apply(pd.Series)
        tb_labels = [label for label in tot_dalys.columns if 'TB' in label]
        print(f"Debug: TB-related labels for year {year}: {tb_labels}")
        dalys[year] = tot_dalys.loc[year, tb_labels].sum()
    dalys.sort_index(inplace=True)
    return dalys

# def get_tb_dalys(df_):
#     # Get DALYs of TB
#     years = df_['year'].value_counts().keys()
#     dalys = pd.Series(dtype='float64', index=years)
#     for year in years:
#         tot_dalys = df_.drop(columns='date').groupby(['year']).sum().apply(pd.Series)
#        #dalys[year] = tot_dalys.loc[(year, ['TB (non-AIDS)', 'non_AIDS_TB'])].sum()
#         dalys[year] = tot_dalys.loc[(year, ["AIDS_TB", "TB", "AIDS_non_TB"])].sum()
#         dalys.sort_index()
#         return dalys

# Extract DALYs from model and scale
tb_dalys = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys",
    custom_generate_series=get_tb_dalys,
    do_scaling=True
).pipe(set_param_names_as_column_index_level_0)

# Get mean/upper/lower statistics
dalys_summary = summarize(tb_dalys).sort_index()
print("DALYs for TB are as follows:")
print(dalys_summary)
dalys_summary.to_excel(outputfilepath / "summarised_tb_dalys.xlsx")

def get_counts_of_items_requested(_df):
    """
    Processes a DataFrame to compute counts of items used and available,
    grouped by date and item.
    """
   # counts_of_used = defaultdict(lambda: defaultdict(int))
    counts_of_availables = defaultdict(lambda: defaultdict(int))

    # Iterate over rows to populate counts
    for _, row in _df.iterrows():
        date = row['date']
        #for item, num in row['Item_Used'].items():
        # counts_of_used[date][item] += num
        for item, num in row['Item_Available'].items():
            counts_of_availables[date][item] += num

    # Create DataFrames for "Used" and "Available"
  #  used_df = pd.DataFrame(counts_of_used).fillna(0).astype(int).stack().rename('Used')
    availables_df = pd.DataFrame(counts_of_availables).fillna(0).astype(int).stack().rename('Available')

    # Combine into one DataFrame and stack
  #  combined_df = pd.concat([used_df, availables_df], axis=1).fillna(0).astype(int)
    combined_df = pd.concat([availables_df], axis=1).fillna(0).astype(int)

    return combined_df.stack()

def get_quantity_of_consumables_dispensed(results_folder):
    """
    Extracts and processes consumables data, saving results to Excel.
    """
    cons_req = (
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='Consumables',
            custom_generate_series=get_counts_of_items_requested,
            do_scaling=True
        ).pipe(set_param_names_as_column_index_level_0)
    )
    return cons_req
cons_req = get_quantity_of_consumables_dispensed(results_folder)
cons_req = cons_req.reset_index()
cons_req.to_excel(outputfilepath / "cons_summary.xlsx")

def get_staff_count_by_facid_and_officer_type(_df: pd.DataFrame) -> pd.Series:
    """Summarize the parsed logged-key results for one draw (as a dataframe) into a pd.Series."""

    # Set index as year derived from 'date' column
    _df['year'] = _df['date'].dt.year  # Extract year from 'date'
    _df = _df.drop(columns=['date'])  # Drop the 'date' column

    # Set the 'year' column as the DataFrame index
    _df = _df.set_index('year')

    # Define a function to transform column names into a standard format
    def change_to_standard_flattened_index_format(col: str) -> str:
        parts = col.split("_", 3)  # Split by "_" up to 3 parts
        if len(parts) == 4:  # Ensure exactly 4 parts to format
            return f"{parts[0]}={parts[1]}|{parts[2]}={parts[3]}"
        return col  # Return unchanged if the format doesn't match

    # Apply the standard format transformation to column names
    _df.columns = [change_to_standard_flattened_index_format(col) for col in _df.columns]

    # Unflatten the DataFrame and return it stacked at the first two levels
    return unflatten_flattened_multi_index_in_logging(_df).stack(level=[0, 1])  # Expanded flattened axis
# Use the 'extract_results' function to process and get staff count
actual_staff_count = extract_results(
    results_folder,
    module='tlo.methods.healthsystem.summary',
    key='Capacity_By_OfficerType_And_FacilityLevel',
    custom_generate_series=get_staff_count_by_facid_and_officer_type,
    do_scaling=True,
).pipe(set_param_names_as_column_index_level_0)

# Reset index to make the DataFrame easier to work with
actual_staff_count = actual_staff_count.reset_index()
# Write the result to an Excel file
actual_staff_count.to_excel(outputfilepath / "staff_count_summary.xlsx")
# Debug: Print the resulting DataFrame
print(actual_staff_count)

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

#capacity by officer
def get_capacity_used_by_officer_type_and_facility_level(_df: pd.Series) -> pd.Series:
    """Summarise the parsed logged-key results for one draw (as dataframe) into a pd.Series."""
    _df = _df.set_axis(_df['date'].dt.year).drop(columns=['date'])
    _df.index.name = 'year'
    return unflatten_flattened_multi_index_in_logging(_df).stack(level=[0, 1])  # expanded flattened axis

#Extracting capacity of cadres by staff
capacity_by_cadre_and_level = extract_results(
     results_folder,
    module='tlo.methods.healthsystem.summary',
    key='Capacity_By_OfficerType_And_FacilityLevel',
    custom_generate_series=get_capacity_used_by_officer_type_and_facility_level,
    do_scaling=False,
).pipe(set_param_names_as_column_index_level_0)

# Reset index to make the DataFrame easier to work with
capacity_by_cadre_and_level = capacity_by_cadre_and_level .reset_index()
# Write the result to an Excel file
capacity_by_cadre_and_level .to_excel(outputfilepath / "staff_capacity_by_cadres.xlsx")
# Debug: Print the resulting DataFrame
print(capacity_by_cadre_and_level )

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
combined_tb_table.to_excel(outputfilepath / "combined_tb_tables.xlsx")
scaling_factor_key = log['tlo.methods.demography']['scaling_factor']
print("Scaling Factor Key:", scaling_factor_key)

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
tb_mortality.to_excel(outputfilepath / "raw_mortality.xlsx")

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
tb_mortality.to_excel(outputfilepath / "raw_mortality.xlsx")

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
    tb_mortality1.to_excel(outputfilepath / "mortality_rates.xlsx")

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
tb_hiv_prop.to_excel(outputfilepath / "PLHIV_tb.xlsx")

#false positives
def get_false_positives(df_):
    years = df_['date'].dt.year.value_counts().keys()
    false_positives = pd.Series(dtype='float64', index=years)
    for year in years:
        tot_false_positives = df_[df_['date'].dt.year == year].drop(columns='date').sum(numeric_only=True)
        false_positives[year] = tot_false_positives[['tbNumFalsePositiveAdults', 'tbNumFalsePositiveChildren']].sum()
    false_positives.sort_index(inplace=True)
    return false_positives

# Extracts false positives
fpositives = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_false_positive",
        custom_generate_series=get_false_positives,
        do_scaling=True,
    ),
    collapse_columns=True,
).pipe(set_param_names_as_column_index_level_0)

fpositives.to_excel(outputfilepath / "false_positives_all.xlsx")

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
mdr_tb.to_excel(outputfilepath / "new_active_mdr_tb_cases.xlsx")

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
tb_treatment_cov.index = tb_treatment_cov.index.year
tb_treatment_cov.to_excel(outputfilepath / "tb_treatment_coverage.xlsx")

## extracts number of people screened for TB by scenario
TARGET_PERIOD = (Date(2010, 1, 1), Date(2020, 12, 31))
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
counts_of_hsi_by_treatment_id.to_excel(outputfilepath / "Tb_Test_Screening.xlsx")

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
tb_incidence.to_excel(outputfilepath / "active_tb.xlsx")
#Tb incidence rate
#Tb_inc_rate = (tb_incidence.divide(pyears_all.values, axis=0)) * 100000
Tb_inc_rate = tb_incidence.reset_index(drop=True).div(pyears_all.reset_index(drop=True), axis='rows')
#Tb_inc_rate = tb_incidence.index(drop=True).div(pyears_all.index(drop=True), axis='rows')
Tb_inc_rate.to_excel(outputfilepath / "Tb_incidence_rate.xlsx")

# Assuming mdr_tb_cases and tb_incidence are your DataFrames
MDR_prop_TB_cases = mdr_tb_cases.div(tb_incidence)*100
MDR_prop_TB_cases.to_excel(outputfilepath / "MDR_prop_TB_cases.xlsx")
#pyears = pyears.reset_index(drop=True)
pyears_summary = pyears_summary.reset_index(drop=True)

print(f"Keys of log['tlo.methods.tb']: {log['tlo.methods.tb'].keys()}")
mdr = log["tlo.methods.tb"]["tb_mdr"]
mdr = mdr.set_index("date")
mdr.to_excel(outputfilepath / "mdr_numbers.xlsx")

print(f"Keys of log['tlo.methods.tb']: {log['tlo.methods.tb'].keys()}")
new_active_tb = log["tlo.methods.tb"]["tb_incidence"]["num_new_active_tb"]
#new_active_tb = new_active_tb.set_index("date")
new_active_tb.to_excel(outputfilepath / "num_new_active_tb.xlsx")

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
Tb_prevalence.to_excel(outputfilepath / "Tb_prevalence.xlsx")

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
adult_Tb_prevalence.to_excel(outputfilepath / "adult_Tb_prevalence.xlsx")

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
child_Tb_prevalence.to_excel(outputfilepath / "child_Tb_prevalence.xlsx")

HSE = log["tlo.methods.healthsystem.summary"]["hsi_event_details"]
HSE = HSE.set_index("date")
print("Health system events as follows",HSE)
HSE.to_excel(outputfilepath / "HSE.xlsx")

HSEvents = log["tlo.methods.healthsystem.summary"]["HSI_Event"]
HSEvents = HSEvents.set_index("date")
print("Health system events as follows",HSEvents)
HSEvents.to_excel(outputfilepath / "HSEvents.xlsx")

hsi_event_counts = log["tlo.methods.healthsystem.summary"]["hsi_event_counts"]
hsi_event_counts = hsi_event_counts.set_index("date")
print("Health system events as follows",hsi_event_counts)
hsi_event_counts.to_excel(outputfilepath / "hsi_event_counts")
print(hsi_event_counts)

# print(f"Keys of log['tlo.methods.healthsystem.summary']: {log['tlo.methods.healthsystem.summary'].keys()}")
print(f"Keys of log['tlo.methods.healthburden']: {log['tlo.methods.healthburden'].keys()}")
keys = log['tlo.methods.healthburden'].keys()
values = [log['tlo.methods.healthburden'][key] for key in keys]

# Create a DataFrame
df = pd.DataFrame({'Key': keys, 'Value': values})

# Export the DataFrame to an Excel file
df.to_excel(outputfilepath /'healthburden_keys.xlsx', index=False)

print(f"Keys of log['tlo.methods.demography']: {log['tlo.methods.demography'].keys()}")
keys = log['tlo.methods.demography'].keys()
values = [log['tlo.methods.demography'][key] for key in keys]

# Create a DataFrame
df = pd.DataFrame({'Key': keys, 'Value': values})

# Export the DataFrame to an Excel file
df.to_excel(outputfilepath /'demography_keys.xlsx', index=False)

###### PLOTS##################################################
print(dalys_summary.columns)
# Calculate the sum of DALYs across years for each scenario
baseline = dalys_summary.loc[:, ('Baseline', 'mean')].sum()
No_Xpert = dalys_summary.loc[:, ('No Xpert Available', 'mean')].sum()
No_CXR = dalys_summary.loc[:, ('No CXR Available', 'mean')].sum()
CXR_scaleup = dalys_summary.loc[:, ('CXR scaleup', 'mean')].sum()
CXR_outreach = dalys_summary.loc[:, ('Outreach services', 'mean')].sum()

# Calculate the corresponding lower and upper bounds
baseline_lower = dalys_summary.loc[:, ('Baseline', 'lower')].sum()
baseline_upper = dalys_summary.loc[:, ('Baseline', 'upper')].sum()

No_Xpert_lower = dalys_summary.loc[:, ('No Xpert Available', 'lower')].sum()
No_Xpert_upper = dalys_summary.loc[:, ('No Xpert Available', 'upper')].sum()

No_CXR_lower = dalys_summary.loc[:, ('No CXR Available', 'lower')].sum()
No_CXR_upper = dalys_summary.loc[:, ('No CXR Available', 'upper')].sum()

CXR_scaleup_lower = dalys_summary.loc[:, ('CXR scaleup', 'lower')].sum()
CXR_scaleup_upper = dalys_summary.loc[:, ('CXR scaleup', 'upper')].sum()

CXR_outreach_lower = dalys_summary.loc[:, ('Outreach services', 'lower')].sum()
CXR_outreach_upper = dalys_summary.loc[:, ('Outreach services', 'upper')].sum()


# Plotting bar graph
x = np.arange(5)
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
bar1 = ax.bar(x[0], baseline, width, label='Baseline', yerr=[[baseline - baseline_lower], [baseline_upper - baseline]])
bar2 = ax.bar(x[1], No_Xpert, width, label='No Xpert Available', yerr=[[No_Xpert - No_Xpert_lower], [No_Xpert_upper - No_Xpert]])
bar3 = ax.bar(x[2], No_CXR, width, label='No CXR Available', yerr=[[No_CXR - No_CXR_lower], [No_CXR_upper - No_CXR]])
bar4 = ax.bar(x[3], CXR_scaleup, width, label='CXR Scale_up', yerr=[[CXR_scaleup - CXR_scaleup_lower], [CXR_scaleup_upper - CXR_scaleup]])
bar5 = ax.bar(x[4], CXR_outreach, width, label='Outreach_services', yerr=[[CXR_outreach -CXR_outreach_lower], [CXR_outreach_upper - CXR_outreach]])

# Adding labels and title
ax.set_xlabel('Scenarios')
ax.set_ylabel('Total DALYs')
ax.set_title('Cumulative TB DALYs 2010-2020')
ax.set_xticks(x)
ax.set_xticklabels(['Baseline', 'No Xpert', 'No CXR', 'CXR Scaleup', 'CXR_outreach'])
ax.legend()

# Displaying graph
plt.show()
print(tb_mortality.columns)
##mortality
baseline = tb_mortality.loc[:, ('Baseline', 'mean')].sum()
No_Xpert = tb_mortality.loc[:, ('No Xpert Available', 'mean')].sum()
No_CXR = tb_mortality.loc[:, ('No CXR Available', 'mean')].sum()
CXR_scaleup = tb_mortality.loc[:, ('CXR scaleup', 'mean')].sum()
CXR_outreach = tb_mortality.loc[:, ('Outreach services', 'mean')].sum()

# Calculate the corresponding lower and upper bounds
baseline_lower = tb_mortality.loc[:, ('Baseline', 'lower')].sum()
baseline_upper = tb_mortality.loc[:, ('Baseline', 'upper')].sum()

No_Xpert_lower = tb_mortality.loc[:, ('No Xpert Available', 'lower')].sum()
No_Xpert_upper = tb_mortality.loc[:, ('No Xpert Available', 'upper')].sum()

No_CXR_lower = tb_mortality.loc[:, ('No CXR Available', 'lower')].sum()
No_CXR_upper = tb_mortality.loc[:, ('No CXR Available', 'upper')].sum()

CXR_scaleup_lower = tb_mortality.loc[:, ('CXR scaleup', 'lower')].sum()
CXR_scaleup_upper = tb_mortality.loc[:, ('CXR scaleup', 'upper')].sum()

CXR_outreach_lower = tb_mortality.loc[:, ('Outreach services', 'lower')].sum()
CXR_outreach_upper = tb_mortality.loc[:, ('Outreach services', 'upper')].sum()

# Plotting bar graph
x = np.arange(5)
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
bar1 = ax.bar(x[0], baseline, width, label='Baseline', yerr=[[baseline - baseline_lower], [baseline_upper - baseline]])
bar2 = ax.bar(x[1], No_Xpert, width, label='No Xpert Available', yerr=[[No_Xpert - No_Xpert_lower], [No_Xpert_upper - No_Xpert]])
bar3 = ax.bar(x[2], No_CXR, width, label='No CXR Available', yerr=[[No_CXR - No_CXR_lower], [No_CXR_upper - No_CXR]])
bar4 = ax.bar(x[3], CXR_scaleup, width, label='CXR Scale_up', yerr=[[CXR_scaleup - CXR_scaleup_lower], [CXR_scaleup_upper - CXR_scaleup]])
bar5 = ax.bar(x[4], CXR_outreach, width, label='CXR_outreach', yerr=[[CXR_outreach - CXR_outreach_lower], [CXR_outreach_upper - CXR_outreach]])


# Adding labels and title
ax.set_xlabel('Scenario')
ax.set_ylabel('TB Mortality')
ax.set_title('Cumulative TB Mortality 2010-2020')
ax.set_xticks(x)
ax.set_xticklabels(['Baseline', 'No Xpert', 'No CXR', 'CXR Scaleup','CXR_outreach'])
ax.legend()

# Displaying graph
plt.show()

#Plotting other epidemiological outcomes
#Plotting TB prevalence across scenarios
fig, ax = plt.subplots(figsize=(10, 6))

# Extract unique scenarios from column index level 0
scenarios = Tb_prevalence.columns.get_level_values(0).unique()
lines = []
for scenario in scenarios:
    scenario_data = Tb_prevalence[scenario]
    mean = scenario_data['mean']
    # Plotting the line
   # ax.plot(scenario_data.index, mean, label=scenario)

    # Apply a moving average to smooth the line
    mean_smoothed = mean.rolling(window=3, center=True).mean()

    # Plotting the smoothed line with label
    line, = ax.plot(scenario_data.index, mean_smoothed, label=scenario)
    lines.append(line)


#setting axis limit

#plt.xlim(2011, 2020)
# Set Y-axis limit to 50%
plt.ylim(0, 0.5)

# Adding labels and title
ax.set_xlabel('Year')
ax.set_ylabel('TB prevalence')
ax.set_title('TB prevalence 2010-2020')
ax.legend()
plt.show()

#Plotting TB incidence across scenarios
fig, ax = plt.subplots(figsize=(10, 6))
# Extract unique scenarios from column index level 0
scenarios = Tb_inc_rate.columns.get_level_values(0).unique()
lines = []
# Extract unique scenarios from column index level 0
scenarios = Tb_inc_rate.columns.get_level_values(0).unique()

fig, ax = plt.subplots(figsize=(10, 6))

# Initialize line variable
lines = []

for scenario in scenarios:
    scenario_data = Tb_inc_rate[scenario]
    mean = scenario_data['mean']

    # Apply a moving average to smooth the line
    mean_smoothed = mean.rolling(window=3, center=True).mean()

    # Plotting the smoothed line with label
    line, = ax.plot(scenario_data.index, mean_smoothed, label=scenario)
    lines.append(line)

# Setting axis limits
plt.ylim(0, 1)

# Adding labels and title
ax.set_xlabel('Year')
ax.set_ylabel('TB Incidence')
ax.set_title('TB Incidence 2010-2019')

# Adding x-axis ticks and labels
years = list(range(2010, 2020))
ax.set_xticks(range(len(years)))
ax.set_xticklabels(years)

# Adding legend
ax.legend()

plt.show()

# Plotting TB treatment coverage by scenarios-tb_treatment_cov
fig, ax = plt.subplots(figsize=(10, 6))

# Extract unique scenarios from column index level 0
scenarios = tb_treatment_cov.columns.get_level_values(0).unique()
for scenario in scenarios:
    scenario_data = tb_treatment_cov[scenario]
    mean = scenario_data['mean']

# Plotting the line with label
    ax.plot(scenario_data.index, mean, label=scenario)
#setting axis limit

#plt.xlim(2011, 2020)
# Set Y-axis limit to 50%
plt.ylim(0, 100)

# Adding labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Prop of treatment coverage')
ax.set_title('TB treatment  coverage 2010-2019')
ax.legend()
# years = list(range(2010, 2020))
# ax.set_xticks(range(len(years)))
# ax.set_xticklabels(years)
plt.show()

# Plotting MDR by scenarios-
fig, ax = plt.subplots(figsize=(10, 6))

# Extract unique scenarios from column index level 0
scenarios = mdr_tb.columns.get_level_values(0).unique()
line=[]
for scenario in scenarios:
    scenario_data = mdr_tb[scenario]
    mean = scenario_data['mean']

# Plotting the line with label
   # ax.plot(scenario_data.index, mean, label=scenario)
#setting axis limit

#plt.xlim(2011, 2020)
# Set Y-axis limit to 50%
#plt.ylim(0, 100)
# Apply a moving average to smooth the line
    mean_smoothed = mean.rolling(window=3, center=True).mean()

    # Plotting the smoothed line with label
    line, = ax.plot(scenario_data.index, mean_smoothed, label=scenario)
    lines.append(line)
# Adding labels and title
ax.set_xlabel('Year')
ax.set_ylabel('MDR cases')
ax.set_title('MDR cases 2010-2019')
ax.legend()
# years = list(range(2010, 2020))
# ax.set_xticks(range(len(years)))
# ax.set_xticklabels(years)
plt.show()

# MDR Fractionalisation
fig, ax = plt.subplots(figsize=(10, 6))

# Extract unique scenarios from column index level 0
scenarios = MDR_prop_TB_cases.columns.get_level_values(0).unique()

# Select only the desired scenarios
selected_scenarios = ['Baseline', 'No Xpert Available']

# Initialize line variable
lines = []

for scenario in selected_scenarios:
    if scenario in scenarios:
        scenario_data = MDR_prop_TB_cases[scenario]
        mean = scenario_data['mean']

        # Plotting the line with label
        line, = ax.plot(scenario_data.index, mean, label=scenario)
        lines.append(line)

# Adding labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Proportion of cases')
ax.set_title('MDR cases as fraction of all active TB 2010-2019')

#Setting axis limits
plt.xlim(2010, 2021)
plt.ylim(0,5)

# years = list(range(2010, 2020))
# ax.set_xticks(range(len(years)))
# ax.set_xticklabels(years)
plt.show()


###############

# Extract unique scenarios from column index level 0
scenarios = MDR_prop_TB_cases.columns.get_level_values(0).unique()
scenario_data = MDR_prop_TB_cases[scenario].copy()
# Select only the desired scenarios
selected_scenarios = ['Baseline', 'No Xpert Available']

# Initialize line variable
lines = []

for scenario in selected_scenarios:
    if scenario in scenarios:
        scenario_data = MDR_prop_TB_cases[scenario].copy()

        # Correcting the mean value for "Baseline" in 2012
        if scenario == 'Baseline' and 2012 in scenario_data.index:
            mean_2011 = scenario_data.loc[2011, 'mean']
            mean_2013 = scenario_data.loc[2013, 'mean']
            scenario_data.loc[2012, 'mean'] = (mean_2011 + mean_2013) / 2

        mean = scenario_data['mean']

        # Plotting the line with label
        line, = plt.plot(scenario_data.index, mean, label=scenario)
        lines.append(line)

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Proportion of cases')
plt.title('Fraction of MDR cases 2010-2019')

# Setting axis limits
plt.xlim(2010, 2021)
plt.ylim(0, 5)

# Create the legend using the line objects
plt.legend(handles=lines, labels=selected_scenarios)

plt.show()


###############################
# Extract unique scenarios from column index level 0
scenarios = MDR_prop_TB_cases.columns.get_level_values(0).unique()

# Select only the desired scenarios
selected_scenarios = ['Baseline', 'No Xpert Available']

# Initialize line variable
lines = []

for scenario in selected_scenarios:
    if scenario in scenarios:
        scenario_data = MDR_prop_TB_cases[scenario].copy()

        # Correcting the mean value for "Baseline" in 2012
        if scenario == 'Baseline' and 2012 in scenario_data.index:
            mean_2011 = scenario_data.loc[2011, 'mean']
            mean_2013 = scenario_data.loc[2013, 'mean']
            scenario_data.loc[2012, 'mean'] = (mean_2011 + mean_2013) / 2

        mean = scenario_data['mean']

        # Apply a moving average to smooth the line
        mean_smoothed = mean.rolling(window=3, center=True).mean()

        # Plotting the smoothed line with label
        line, = plt.plot(scenario_data.index, mean_smoothed, label=scenario)
        lines.append(line)

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Proportion of cases')
plt.title('Fraction of MDR cases 2010-2019')

# Setting axis limits
plt.xlim(2010, 2021)
plt.ylim(0, 5)

# Create the legend using the line objects
plt.legend(handles=lines, labels=selected_scenarios)

plt.show()
#Treatment coverage revisted

# Extract unique scenarios from column index level 0
scenarios = tb_treatment_cov.columns.get_level_values(0).unique()

fig, ax = plt.subplots(figsize=(10, 6))

# Set Y-axis limit to 100
plt.ylim(0, 100)

# Initialize line variable
lines = []

for scenario in scenarios:
    scenario_data = tb_treatment_cov[scenario]
    mean = scenario_data['mean']

    # Apply a moving average to smooth the line
    mean_smoothed = mean.rolling(window=3, center=True).mean()

    # Plotting the smoothed line with label
    line, = ax.plot(scenario_data.index, mean_smoothed, label=scenario)
    lines.append(line)

# Adding labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Prop of treatment coverage')
ax.set_title('TB treatment coverage 2010-2019')
ax.legend()

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
        help="Directory containing results from running src/scripts/hiv/DAH/tb_DAH_scenarios10x.py",
        default=None,
        required=False
    )
    args = parser.parse_args()
