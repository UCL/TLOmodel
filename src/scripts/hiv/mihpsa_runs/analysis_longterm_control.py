"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder

if running locally need to parse log files:
tlo parse-log /Users/tmangal/PycharmProjects/TLOmodel/outputs/mihpsa_runs-2025-04-14T130655Z/0/0

mihpsa_runs-2025-04-19T220218Z

tlo parse-log 'outputs/longterm_mihpsa_runs-2025-05-16T131354Z/0/0'
when running locally

"""

import datetime
from pathlib import Path
from tlo.analysis.utils import parse_log_file

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)
from tlo import Date

datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

outputspath = Path("./outputs/")  # for local runs


# test runs
# results_folder = Path('outputs/test_runs__2025-05-15T144136.log')
# output = parse_log_file('outputs/test_runs__2025-05-15T144136.log')
#
#
# with open(outputspath / "test_runs.pickle", "wb") as f:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)
#
# # load the results
# with open(outputspath / "test_runs.pickle", "rb") as f:
#     output = pickle.load(f)




# %% Analyse results of runs

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("longterm_mihpsa_runs.py", outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder, draw=0)

# get basic information about the results
info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

scaling_factor = log['tlo.methods.population']['scaling_factor'].scaling_factor.values[0]

# -----------------------------------------------------------------
# %% Population attributes

variables = [
    "Total_00_14_M",
                "Total_15_24_M",
                "Total_25_49_M",
                "Total_50_UP_M",
                "Total_00_14_F",
                "Total_15_24_F",
                "Total_25_49_F",
                "Total_50_UP_F",
                "Total_FSW",
                "Total_MSM",
                "PLHIV_00_14_M",
                "PLHIV_15_24_M",
                "PLHIV_25_49_M",
                "PLHIV_50_UP_M",
                "PLHIV_00_14_F",
                "PLHIV_15_24_F",
                "PLHIV_25_49_F",
                "PLHIV_50_UP_F",
                "PLHIV_FSW",
                "PLHIV_MSM",
                "Diagnosed_00_14_M",
                "Diagnosed_15_24_M",
                "Diagnosed_25_49_M",
                "Diagnosed_50_UP_M",
                "Diagnosed_00_14_F",
                "Diagnosed_15_24_F",
                "Diagnosed_25_49_F",
                "Diagnosed_50_UP_F",
                "Diagnosed_FSW",
                "Diagnosed_MSM",
                "ART_00_14_M",
                "ART_15_24_M",
                "ART_25_49_M",
                "ART_50_UP_M",
                "ART_00_14_F",
                "ART_15_24_F",
                "ART_25_49_F",
                "ART_50_UP_F",
                "ART_FSW",
                "ART_MSM",
                "VLS_00_14_M",
                "VLS_15_24_M",
                "VLS_25_49_M",
                "VLS_50_UP_M",
                "VLS_00_14_F",
                "VLS_15_24_F",
                "VLS_25_49_F",
                "VLS_50_UP_F",
                "VLS_FSW",
                "VLS_MSM",
                "Birth_All",
                "Birth_HIV",
                "DeathsAll_00_14_M",
                "DeathsAll_15_24_M",
                "DeathsAll_25_49_M",
                "DeathsAll_50_UP_M",
                "DeathsAll_00_14_F",
                "DeathsAll_15_24_F",
                "DeathsAll_25_49_F",
                "DeathsAll_50_UP_F",
                "NewHIV_00_14_M",
                "NewHIV_15_24_M",
                "NewHIV_25_49_M",
                "NewHIV_50_UP_M",
                "NewHIV_00_14_F",
                "NewHIV_15_24_F",
                "NewHIV_25_49_F",
                "NewHIV_50_UP_F",
                "NewHIV_FSW",
                "NewHIV_MSM",
                "DeathsHIV_00_14_M",
                "DeathsHIV_15_24_M",
                "DeathsHIV_25_49_M",
                "DeathsHIV_50_UP_M",
                "DeathsHIV_00_14_F",
                "DeathsHIV_15_24_F",
                "DeathsHIV_25_49_F",
                "DeathsHIV_50_UP_F",
                "DALYs_Undiscounted",
                "TotalCost_Undiscounted",
                "Percent_circumcised",
                "Percent_condom use_GP",
                "PrEP_FSW",
                "PrEP_MSM",
                "PrEP_GP",
                "PrEP_Pop_GP",
                "NewHIV_PrEP_Pop_GP",
                "Percent_FSW reached",
                "Percent_MSM reached",
]

stocks_output = {}

for var in variables:
    result = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.hiv",
            key="long_term_mihpsa",
            column=var,
            index="date",
            do_scaling=False,
        ),
        collapse_columns=False,
        only_mean=True
    )

    for draw in result.columns:
        if draw not in stocks_output:
            stocks_output[draw] = pd.DataFrame()  # Initialise DataFrame for the draw if not exists

        stocks_output[draw][var] = result[draw]

with pd.ExcelWriter(results_folder / "longterm_outputs.xlsx", engine='openpyxl') as writer:
    # Iterate over the dictionary and write each DataFrame to a new sheet
    for draw, df in stocks_output.items():
        df = df.T  # Switch rows and columns
        # Writing each draw's DataFrame to a new sheet named after the draw
        df.to_excel(writer, sheet_name=f'Draw_{draw}', index=True)

# -----------------------------------------------------------------------------------

# EXTRACT DEATHS

def summarise_deaths(results_folder,
                     label=None, age=None, sex=None):
    """ returns mean deaths for each year of the simulation
    values are aggregated across the runs of each draw
    for the specified cause
    """

    results_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(
                ["year", "label", "age", "sex"])["person_id"].count()
        ),
        do_scaling=True,
    )
    # removes multi-index
    results_deaths = results_deaths.reset_index()

    # select only cause specified
    if label == 'AIDS':
        tmp = results_deaths.loc[
            (results_deaths.label == label)
        ]
    # otherwise all deaths
    else:
        tmp = results_deaths

    if age == 'children':
        tmp = tmp.loc[tmp.age < 15]
    else:
        tmp = tmp.loc[tmp.age >= 15]

    if sex == 'M':
        tmp = tmp.loc[tmp.sex == 'M']
    elif sex == 'F':
        tmp = tmp.loc[tmp.sex == 'F']

    # group deaths by year
    # Identify only the draw/run columns (MultiIndex level names as numeric)
    draw_run_columns = [col for col in tmp.columns if isinstance(col[0], int)]

    # Group by year and sum only the draw/run columns
    summed_results = tmp.groupby('year')[draw_run_columns].sum()

    # Reset the index so 'year' becomes a regular column
    summed_results = summed_results.reset_index()

    # get mean for each draw
    mean_deaths = pd.concat({'mean': summed_results.iloc[:, 1:].groupby(level=0, axis=1).mean()}, axis=1).swaplevel(
        axis=1)

    return mean_deaths


aids_deaths_children_M = summarise_deaths(results_folder,
                                        label='AIDS',
                                        age='children',
                                        sex='M')

aids_deaths_children_F = summarise_deaths(results_folder,
                                        label='AIDS',
                                        age='children',
                                        sex='F')

aids_deaths_men = summarise_deaths(results_folder,
                                   label='AIDS',
                                   age='adult',
                                   sex='M')

aids_deaths_women = summarise_deaths(results_folder,
                                     label='AIDS',
                                     age='adult',
                                     sex='F')

all_deaths_children_M = summarise_deaths(results_folder,
                                       label=None,
                                       age='children',
                                         sex='M')

all_deaths_children_F = summarise_deaths(results_folder,
                                       label=None,
                                       age='children',
                                         sex='F')

all_deaths_men = summarise_deaths(results_folder,
                                  label=None,
                                  age='adult',
                                  sex='M')

all_deaths_women = summarise_deaths(results_folder,
                                    label=None,
                                    age='adult',
                                    sex='F')

# List of dataframes to include in the workbook
dataframes = {
    "aids_deaths_children_M": aids_deaths_children_M,
    "aids_deaths_children_F": aids_deaths_children_F,
    "aids_deaths_men": aids_deaths_men,
    "aids_deaths_women": aids_deaths_women,
    "all_deaths_children_M": all_deaths_children_M,
    "all_deaths_children_F": all_deaths_children_F,
    "all_deaths_men": all_deaths_men,
    "all_deaths_women": all_deaths_women,
}

# Create a new Excel writer object
with pd.ExcelWriter(results_folder / "full_summarised_deaths.xlsx") as writer:
    # Iterate over draws (0 to 7)
    for draw in range(9):
        # Prepare data for the current draw
        sheet_data = []
        for name, df in dataframes.items():
            # Extract the "mean" column for the current draw
            row_data = [name] + df[(draw, "mean")].tolist()
            sheet_data.append(row_data)

        # Convert the sheet data to a DataFrame
        sheet_df = pd.DataFrame(sheet_data)

        # Write the DataFrame to the corresponding sheet
        sheet_df.to_excel(writer, sheet_name=f"Draw {draw}", index=False, header=False)


# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------
# DALYS AVERTED
TARGET_PERIOD = (Date(2010, 1, 1), Date(2050, 12, 31))


def get_num_dalys_by_year(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
    Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
    results from runs that crashed mid-way through the simulation.
    """
    years_needed = [i.year for i in TARGET_PERIOD]
    assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
    return pd.Series(
        data=_df
        .loc[_df.year.between(*years_needed)]
        .drop(columns=['date', 'sex', 'age_range'])
        .groupby(['year']).sum().stack()
    )


num_dalys_by_year = summarize(extract_results(
    results_folder,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=get_num_dalys_by_year,
    do_scaling=True
),
    only_mean=True)

aids_dalys = num_dalys_by_year[num_dalys_by_year.index.get_level_values(1) == 'AIDS']
with pd.ExcelWriter(results_folder / 'full_dalys.xlsx', engine='openpyxl') as writer:
    aids_dalys.to_excel(writer, sheet_name='DALYs')

num_dalys_by_year_FULL = extract_results(
    results_folder,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=get_num_dalys_by_year,
    do_scaling=True
)
aids_dalys_FULL = num_dalys_by_year_FULL[num_dalys_by_year_FULL.index.get_level_values(1) == 'AIDS']
with pd.ExcelWriter(results_folder / 'full_aids_dalys_FULL.xlsx', engine='openpyxl') as writer:
    aids_dalys_FULL.to_excel(writer, sheet_name='DALYs')

# need to get number DALYs averted compared to minimal scenario
dalys_averted = pd.DataFrame()

# Calculate differences and add to new DataFrame
for col in aids_dalys.columns[1:]:  # Start from the second column
    dalys_averted[col] = aids_dalys[1] - aids_dalys[col]

with pd.ExcelWriter(results_folder / 'full_aids_dalys_averted.xlsx', engine='openpyxl') as writer:
    dalys_averted.to_excel(writer, sheet_name='DALYs Averted')
