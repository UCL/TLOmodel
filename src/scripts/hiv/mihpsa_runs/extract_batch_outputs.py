"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder
"""

import datetime
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

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

# %% Analyse results of runs

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("mihpsa_runs.py", outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

scaling_factor = log['tlo.methods.population']['scaling_factor'].scaling_factor.values[0]

# -----------------------------------------------------------------

stock_variables = [
    "N_PLHIV_00_14_C",
    "N_PLHIV_15_24_M",
    "N_PLHIV_15_24_F",
    "N_PLHIV_25_49_M",
    "N_PLHIV_25_49_F",
    "N_PLHIV_50_UP_M",
    "N_PLHIV_50_UP_F",
    "N_Total_00_14_C",
    "N_Total_15_24_M",
    "N_Total_15_24_F",
    "N_Total_25_49_M",
    "N_Total_25_49_F",
    "N_Total_50_UP_M",
    "N_Total_50_UP_F",
    "N_Diag_00_14_C",
    "N_Diag_15_UP_M",
    "N_Diag_15_UP_F",
    "N_ART_00_14_C",
    "N_ART_15_UP_M",
    "N_ART_15_UP_F",
    "N_VLS_15_UP_M",
    "N_VLS_15_UP_F",
    "N_PLHIV_15_UP_AIDS",
    "N_PLHIV_15_UP_NO_AIDS",
]

flow_variables = [
    "N_BirthAll",
    "N_BirthHIV",
    "N_BirthART",
    "N_NewHIV_00_14_C",
    "N_NewHIV_15_24_M",
    "N_NewHIV_15_24_F",
    "N_NewHIV_25_49_M",
    "N_NewHIV_25_49_F",
    "N_NewHIV_50_UP_M",
    "N_NewHIV_50_UP_F",
    "N_DeathsHIV_00_14_C",
    "N_DeathsHIV_15_UP_M",
    "N_DeathsHIV_15_UP_F",
    "N_DeathsAll_00_14_C",
    "N_DeathsAll_15_UP_M",
    "N_DeathsAll_15_UP_F",
    "N_YLL_00_14_C",
    "N_YLL_15_UP_M",
    "N_YLL_15_UP_F",
    "N_HIVTest_Facility_NEG_15_UP",
    "N_HIVTest_Facility_POS_15_UP",
    "N_HIVTest_Index_NEG_15_UP",
    "N_HIVTest_Index_POS_15_UP",
    "N_HIVTest_Community_NEG_15_UP",
    "N_HIVTest_Community_POS_15_UP",
    "N_HIVTest_SelfTest_POS_15_UP",
    "N_HIVTest_SelfTest_Dist",
    "N_Condom_Acts",
    "N_NewVMMC",
    "PY_PREP_ORAL_AGYW",
    "PY_PREP_ORAL_FSW",
    "PY_PREP_ORAL_MSM",
    "PY_PREP_INJECT_AGYW",
    "PY_PREP_INJECT_FSW",
    "PY_PREP_INJECT_MSM",
    "N_ART_ADH_15_UP_F",
    "N_ART_ADH_15_UP_M",
    "N_VL_TEST_15_UP",
    "N_VL_TEST_00_14",
    "N_OUTREACH_FSW",
    "N_OUTREACH_MSM",
    "N_EconEmpowerment",
    "N_CSE_15_19_F",
    "N_CSE_15_19_M"]

# %% extract the intervention scenarios

stocks_output = {}

for stock in stock_variables:
    result = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.hiv",
            key="stock_variables",
            column=stock,
            index="date",
            do_scaling=True,
        ),
        collapse_columns=False,
        only_mean=True
    )
    result = result[result.index.year >= 2021]

    for draw in result.columns:
        if draw not in stocks_output:
            stocks_output[draw] = pd.DataFrame()  # Initialise DataFrame for the draw if not exists

        stocks_output[draw][stock] = result[draw]

with pd.ExcelWriter(results_folder / "stock_outputs_workbook.xlsx", engine='openpyxl') as writer:
    # Iterate over the dictionary and write each DataFrame to a new sheet
    for draw, df in stocks_output.items():
        df = df.T  # Switch rows and columns
        # Writing each draw's DataFrame to a new sheet named after the draw
        df.to_excel(writer, sheet_name=f'Draw_{draw}', index=True)


flows_output = {}

for flow in flow_variables:
    result = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.hiv",
            key="flow_variables",
            column=flow,
            index="date",
            do_scaling=True,
        ),
        collapse_columns=False,
        only_mean=True
    )
    result = result[result.index.year >= 2021]

    for draw in result.columns:
        if draw not in flows_output:
            flows_output[draw] = pd.DataFrame()  # Initialise DataFrame for the draw if not exists

        flows_output[draw][flow] = result[draw]

with pd.ExcelWriter(results_folder / 'flows_output.xlsx') as writer:
    for sheet_name, df in flows_output.items():
        df = df.T
        df.to_excel(writer, sheet_name=str(sheet_name))


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
    results_deaths = results_deaths.loc[results_deaths.index.get_level_values('year') >= 2021]
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
    mean_deaths = pd.concat({'mean': summed_results.iloc[:, 1:].groupby(level=0, axis=1).mean()}, axis=1).swaplevel(axis=1)

    return mean_deaths


aids_deaths_children = summarise_deaths(results_folder,
                                                      label='AIDS',
                                                      age='children')

aids_deaths_men = summarise_deaths(results_folder,
                                                 label='AIDS',
                                                 age='adult',
                                                 sex='M')

aids_deaths_women = summarise_deaths(results_folder,
                                                 label='AIDS',
                                                 age='adult',
                                                 sex='F')

all_deaths_children = summarise_deaths(results_folder,
                                                      label=None,
                                                      age='children')

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
    "aids_deaths_children": aids_deaths_children,
    "aids_deaths_men": aids_deaths_men,
    "aids_deaths_women": aids_deaths_women,
    "all_deaths_children": all_deaths_children,
    "all_deaths_men": all_deaths_men,
    "all_deaths_women": all_deaths_women,
}

# Create a new Excel writer object
with pd.ExcelWriter(results_folder / "summarised_deaths.xlsx") as writer:
    # Iterate over draws (0 to 7)
    for draw in range(8):
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
# HIV TESTS


def summarise_tests(results_folder, positive=None):
    """ returns mean deaths for each year of the simulation
    values are aggregated across the runs of each draw
    for the specified cause
    """

    results_tests = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_test",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(
                ["year", "hiv_diagnosed", "adult"])["person_id"].count()
        ),
        do_scaling=True,
    )

    results_tests = results_tests.loc[results_tests.index.get_level_values('year') >= 2021]
    results_tests = results_tests.loc[results_tests.index.get_level_values('adult') == True]

    if positive == 'YES':
        results_tests = results_tests.loc[results_tests.index.get_level_values('hiv_diagnosed') == True]
    else:
        results_tests = results_tests.loc[results_tests.index.get_level_values('hiv_diagnosed') == False]

    summed_results = results_tests.groupby('year').sum()
    # Reset the index so 'year' becomes a regular column
    summed_results = summed_results.reset_index()

    # get mean for each draw
    mean_tests = pd.concat({'mean': summed_results.iloc[:, 1:].groupby(level=0, axis=1).mean()}, axis=1).swaplevel(axis=1)

    return mean_tests


positive_tests = summarise_tests(results_folder, positive='YES')
negative_tests = summarise_tests(results_folder, positive='NO')

# List of dataframes to include in the workbook
dataframes = {
    "negative_tests": negative_tests,
    "positive_tests": positive_tests,
}

# Create a new Excel writer object
with pd.ExcelWriter(results_folder / "summarised_tests.xlsx") as writer:
    # Iterate over draws (0 to 7)
    for draw in range(8):
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
# VL TESTS


def summarise_vl_tests(results_folder, adult=None):
    """ returns mean deaths for each year of the simulation
    values are aggregated across the runs of each draw
    for the specified cause
    """

    results_tests = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_VLtest",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(
                ["year", "adult"])["person_id"].count()
        ),
        do_scaling=True,
    )

    results_tests = results_tests.loc[results_tests.index.get_level_values('year') >= 2021]

    if adult == 'YES':
        results_tests = results_tests.loc[results_tests.index.get_level_values('adult') == True]
    else:
        results_tests = results_tests.loc[results_tests.index.get_level_values('adult') == False]

    summed_results = results_tests.groupby('year').sum()
    # Reset the index so 'year' becomes a regular column
    summed_results = summed_results.reset_index()

    # get mean for each draw
    mean_tests = pd.concat({'mean': summed_results.iloc[:, 1:].groupby(level=0, axis=1).mean()}, axis=1).swaplevel(axis=1)

    return mean_tests


adult_tests = summarise_vl_tests(results_folder, adult='YES')
child_tests = summarise_vl_tests(results_folder, adult='NO')

# List of dataframes to include in the workbook
dataframes = {
    "adult_tests": adult_tests,
    "child_tests": child_tests,
}

# Create a new Excel writer object
with pd.ExcelWriter(results_folder / "summarised_VLtests.xlsx") as writer:
    # Iterate over draws (0 to 7)
    for draw in range(8):
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
# EXTRACT DEATHS for YLL CALCULATION

# TARGET_PERIOD = (Date(2023, 1, 1), Date(2050, 12, 31))


def output_deaths_by_age(results_folder):
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
    results_deaths = results_deaths.loc[results_deaths.index.get_level_values('year') >= 2023]

    # removes multi-index
    results_deaths = results_deaths.reset_index()

    tmp = results_deaths.loc[(results_deaths.label == 'AIDS')]
    tmp['age'] = tmp['age'].apply(lambda x: '80+' if x >= 80 else x)

    # Identify only the draw/run columns (MultiIndex level names as numeric)
    draw_run_columns = [col for col in tmp.columns if isinstance(col[0], int)]

    # Group by year and sum only the draw/run columns
    summed_results = tmp.groupby(['age', 'sex', 'year'])[draw_run_columns].sum()

    mean_by_draw = summed_results.groupby(level='draw', axis=1).mean()

    return mean_by_draw


full_deaths = output_deaths_by_age(results_folder)


def transform_full_deaths(full_deaths):
    # Reset index to use 'age', 'sex', and 'year' as columns
    full_deaths_reset = full_deaths.reset_index()

    # Pivot the data to have years as columns
    draw_columns = full_deaths_reset.columns[3:]  # Assuming columns 0-3 are 'age', 'sex', 'year'
    pivoted_data = full_deaths_reset.pivot_table(index=['age', 'sex'], columns='year', values=draw_columns)

    # Flatten the multi-index and reset it for proper format
    pivoted_data_reset = pivoted_data.reset_index()

    pivoted_data_reset = pivoted_data_reset.sort_values(by=['sex', 'age'], ascending=[False, True])

    # Select 'age' and 'sex' columns from pivoted_data_reset
    age_sex_data = pivoted_data_reset[['age', 'sex']]

    # Drop the multi-index levels from columns, but retain the column names 'age' and 'sex'
    age_sex_data.columns = age_sex_data.columns.get_level_values(0)

    # Now age_sex_data has no multi-index, and columns 'age' and 'sex' are retained

    draw_dfs = {}

    # Loop over the draw values (0 to 7)
    for draw in range(8):

        # Select the columns for the relevant 'draw' (i.e., multi-index columns for the given draw)
        draw_columns_for_current_draw = pivoted_data_reset.columns[
            pivoted_data_reset.columns.get_level_values(0) == draw]

        # Concatenate the 'age' and 'sex' columns with the selected columns for the current draw
        draw_dfs[draw] = pd.concat([age_sex_data, pivoted_data_reset[draw_columns_for_current_draw]], axis=1)

    return draw_dfs


draw_dfs = transform_full_deaths(full_deaths)

with pd.ExcelWriter(results_folder / 'full_deaths_by_age.xlsx') as writer:
    for sheet_name, df in draw_dfs.items():
        df.to_excel(writer, sheet_name=str(sheet_name))

# -----------------------------------------------------------------------------------
# DALYS AVERTED
TARGET_PERIOD = (Date(2021, 1, 1), Date(2050, 12, 31))


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
with pd.ExcelWriter(results_folder / 'dalys.xlsx', engine='openpyxl') as writer:
    aids_dalys.to_excel(writer, sheet_name='DALYs')

num_dalys_by_year_FULL = extract_results(
    results_folder,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=get_num_dalys_by_year,
    do_scaling=True
)
aids_dalys_FULL = num_dalys_by_year_FULL[num_dalys_by_year_FULL.index.get_level_values(1) == 'AIDS']
with pd.ExcelWriter(results_folder / 'dalys_FULL.xlsx', engine='openpyxl') as writer:
    aids_dalys_FULL.to_excel(writer, sheet_name='DALYs')

# need to get number DALYs averted compared to minimal scenario
dalys_averted = pd.DataFrame()

# Calculate differences and add to new DataFrame
for col in aids_dalys.columns[1:]:  # Start from the second column
    dalys_averted[col] = aids_dalys[0] - aids_dalys[col]

with pd.ExcelWriter(results_folder / 'dalys_averted.xlsx', engine='openpyxl') as writer:
    dalys_averted.to_excel(writer, sheet_name='DALYs Averted')
