"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder

if running locally need to parse log files:
tlo parse-log /Users/tmangal/PycharmProjects/TLOmodel/outputs/mihpsa_runs-2025-04-14T130655Z/0/0

mihpsa_runs-2025-04-19T220218Z

tlo parse-log 'outputs/mihpsa_runs-2025-09-20T142441Z/0/0'
when running locally


longterm_mihpsa_runs-2025-05-17T165444Z

"""

import datetime
from pathlib import Path
from tlo.analysis.utils import parse_log_file, compute_summary_statistics

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
    compute_summary_statistics
)
from tlo import Date

datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# outputspath = Path("./outputs/")  # for local runs


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
    "N_circumcised_15_24_M",
    "PrEP_AGYW_PG",
    "Total_AGYW_PG",
]



# scaled
scaled_stocks_output = {}

for var in variables:
    result = compute_summary_statistics(
        extract_results(
            results_folder,
            module="tlo.methods.hiv",
            key="long_term_mihpsa",
            column=var,
            index="date",
            do_scaling=True,
        ),
        only_central=True,
        central_measure='mean',
    )

    for draw in result.columns:
        if draw not in scaled_stocks_output:
            scaled_stocks_output[draw] = pd.DataFrame()  # Initialise DataFrame for the draw if not exists

        scaled_stocks_output[draw][var] = result[draw]



with pd.ExcelWriter(results_folder / "longterm_outputs_scaled.xlsx", engine='openpyxl') as writer:
    # Iterate over the dictionary and write each DataFrame to a new sheet
    for draw, df in scaled_stocks_output.items():
        df = df.T  # Switch rows and columns
        # Writing each draw's DataFrame to a new sheet named after the draw
        df.to_excel(writer, sheet_name=f'Draw_{draw}', index=True)



# unscaled
unscaled_stocks_output = {}

for var in variables:
    result = compute_summary_statistics(
        extract_results(
            results_folder,
            module="tlo.methods.hiv",
            key="long_term_mihpsa",
            column=var,
            index="date",
            do_scaling=False,
        ),
        only_central=True,
        central_measure='mean',
    )

    for draw in result.columns:
        if draw not in unscaled_stocks_output:
            unscaled_stocks_output[draw] = pd.DataFrame()  # Initialise DataFrame for the draw if not exists

        unscaled_stocks_output[draw][var] = result[draw]



with pd.ExcelWriter(results_folder / "longterm_outputs_unscaled.xlsx", engine='openpyxl') as writer:
    # Iterate over the dictionary and write each DataFrame to a new sheet
    for draw, df in unscaled_stocks_output.items():
        df = df.T  # Switch rows and columns
        # Writing each draw's DataFrame to a new sheet named after the draw
        df.to_excel(writer, sheet_name=f'Draw_{draw}', index=True)



# add the unscaled outputs to the scaled outputs sheet for easier transference to template
cols_to_replace = [
    "Percent_circumcised",
    "Percent_condom use_GP",
    "Percent_FSW reached",
]

merged_output = {}

for key in scaled_stocks_output:
    # make a copy so the original is untouched
    df_new = scaled_stocks_output[key].copy()

    # replace the selected variables using dict2
    df_new[cols_to_replace] = unscaled_stocks_output[key][cols_to_replace]

    merged_output[key] = df_new




# -----------------------------------------------------------------------------------

# EXTRACT DEATHS
#
# def summarise_deaths(results_folder,
#                      label=None, age=None, sex=None):
#     """ returns mean deaths for each year of the simulation
#     values are aggregated across the runs of each draw
#     for the specified cause
#     """
#
#     results_deaths = extract_results(
#         results_folder,
#         module="tlo.methods.demography",
#         key="death",
#         custom_generate_series=(
#             lambda df: df.assign(year=df["date"].dt.year).groupby(
#                 ["year", "label", "age", "sex"])["person_id"].count()
#         ),
#         do_scaling=True,
#     )
#     # removes multi-index
#     results_deaths = results_deaths.reset_index()
#
#     # select only cause specified
#     if label == 'AIDS':
#         tmp = results_deaths.loc[
#             (results_deaths.label == label)
#         ]
#     # otherwise all deaths
#     else:
#         tmp = results_deaths
#
#     if age == 'children':
#         tmp = tmp.loc[tmp.age < 15]
#     else:
#         tmp = tmp.loc[tmp.age >= 15]
#
#     if sex == 'M':
#         tmp = tmp.loc[tmp.sex == 'M']
#     elif sex == 'F':
#         tmp = tmp.loc[tmp.sex == 'F']
#
#     # group deaths by year
#     # Identify only the draw/run columns (MultiIndex level names as numeric)
#     draw_run_columns = [col for col in tmp.columns if isinstance(col[0], int)]
#
#     # Group by year and sum only the draw/run columns
#     summed_results = tmp.groupby('year')[draw_run_columns].sum()
#
#     # Reset the index so 'year' becomes a regular column
#     summed_results = summed_results.reset_index()
#
#     # get mean for each draw
#     mean_deaths = pd.concat({'mean': summed_results.iloc[:, 1:].groupby(level=0, axis=1).mean()}, axis=1).swaplevel(
#         axis=1)
#
#     return mean_deaths
#


def summarise_deaths(results_folder, label=None, age=None, sex=None, age_group_label=None):
    """
    Returns mean deaths for each year of the simulation.
    Values are aggregated across runs of each draw for the specified cause and subgroup.
    """

    # Load death results with grouped series
    results_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(
                ["year", "label", "age", "sex"]
            )["person_id"].count()
        ),
        do_scaling=True,
    ).reset_index()

    # Filter for cause of death if specified
    if label == 'AIDS':
        tmp = results_deaths.loc[results_deaths.label == label]
    else:
        tmp = results_deaths

    # Apply age and sex filtering
    if age_group_label is not None:
        # Map string labels to age and sex filters
        age_sex_criteria = {
            'Deaths_00_14_M': (0, 14, 'M'),
            'Deaths_15_24_M': (15, 24, 'M'),
            'Deaths_25_49_M': (25, 49, 'M'),
            'Deaths_50_UP_M': (50, None, 'M'),
            'Deaths_00_14_F': (0, 14, 'F'),
            'Deaths_15_24_F': (15, 24, 'F'),
            'Deaths_25_49_F': (25, 49, 'F'),
            'Deaths_50_UP_F': (50, None, 'F'),
        }

        if age_group_label not in age_sex_criteria:
            raise ValueError(f"Invalid age_group_label: {age_group_label}")

        age_min, age_max, sex_val = age_sex_criteria[age_group_label]
        if age_max is not None:
            tmp = tmp.loc[(tmp.age >= age_min) & (tmp.age <= age_max) & (tmp.sex == sex_val)]
        else:
            tmp = tmp.loc[(tmp.age >= age_min) & (tmp.sex == sex_val)]

    else:
        # Retain existing logic if no age_group_label specified
        if age == 'children':
            tmp = tmp.loc[tmp.age < 15]
        elif age == 'adults':
            tmp = tmp.loc[tmp.age >= 15]

        if sex == 'M':
            tmp = tmp.loc[tmp.sex == 'M']
        elif sex == 'F':
            tmp = tmp.loc[tmp.sex == 'F']

    # Identify draw/run columns by checking for tuple column names where first element is int
    draw_run_columns = [col for col in tmp.columns if isinstance(col, tuple) and isinstance(col[0], int)]

    # Group by year and sum across selected draw columns
    summed_results = tmp.groupby('year')[draw_run_columns].sum().reset_index()

    # Compute mean across runs for each draw
    mean_deaths = pd.concat(
        {'mean': summed_results.iloc[:, 1:].groupby(level=0, axis=1).mean()},
        axis=1
    ).swaplevel(axis=1)

    return mean_deaths


all_deaths_children_M = summarise_deaths(results_folder,
                                         label=None,
                                         age='children',
                                         sex='M')

all_deaths_15_24_M = summarise_deaths(results_folder,
                                      label=None,
                                      age_group_label='Deaths_15_24_M')

all_deaths_25_49_M = summarise_deaths(results_folder,
                                      label=None,
                                      age_group_label='Deaths_25_49_M')

all_deaths_50_M = summarise_deaths(results_folder,
                                   label=None,
                                   age_group_label='Deaths_50_UP_M')

all_deaths_children_F = summarise_deaths(results_folder,
                                         label=None,
                                         age='children',
                                         sex='F')

all_deaths_15_24_F = summarise_deaths(results_folder,
                                      label=None,
                                      age_group_label='Deaths_15_24_F')

all_deaths_25_49_F = summarise_deaths(results_folder,
                                      label=None,
                                      age_group_label='Deaths_25_49_F')

all_deaths_50_F = summarise_deaths(results_folder,
                                   label=None,
                                   age_group_label='Deaths_50_UP_F')

aids_deaths_children_M = summarise_deaths(results_folder,
                                          label='AIDS',
                                          age='children',
                                          sex='M')

aids_deaths_15_24_M = summarise_deaths(results_folder,
                                       label='AIDS',
                                       age_group_label='Deaths_15_24_M')

aids_deaths_25_49_M = summarise_deaths(results_folder,
                                       label='AIDS',
                                       age_group_label='Deaths_25_49_M')

aids_deaths_50_M = summarise_deaths(results_folder,
                                    label='AIDS',
                                    age_group_label='Deaths_50_UP_M')

aids_deaths_children_F = summarise_deaths(results_folder,
                                          label='AIDS',
                                          age='children',
                                          sex='F')

aids_deaths_15_24_F = summarise_deaths(results_folder,
                                       label='AIDS',
                                       age_group_label='Deaths_15_24_F')

aids_deaths_25_49_F = summarise_deaths(results_folder,
                                       label='AIDS',
                                       age_group_label='Deaths_25_49_F')

aids_deaths_50_F = summarise_deaths(results_folder,
                                    label='AIDS',
                                    age_group_label='Deaths_50_UP_F')

# List of dataframes to include in the workbook
dataframes = {
    "DeathsAll_00_14_M": all_deaths_children_M,
    "DeathsAll_15_24_M": all_deaths_15_24_M,
    "DeathsAll_25_49_M": all_deaths_25_49_M,
    "DeathsAll_50_UP_M": all_deaths_50_M,
    "DeathsAll_00_14_F": all_deaths_children_F,
    "DeathsAll_15_24_F": all_deaths_15_24_F,
    "DeathsAll_25_49_F": all_deaths_25_49_F,
    "DeathsAll_50_UP_F": all_deaths_50_F,

    "DeathsHIV_00_14_M": aids_deaths_children_M,
    "DeathsHIV_15_24_M": aids_deaths_15_24_M,
    "DeathsHIV_25_49_M": aids_deaths_25_49_M,
    "DeathsHIV_50_UP_M": aids_deaths_50_M,
    "DeathsHIV_00_14_F": aids_deaths_children_F,
    "DeathsHIV_15_24_F": aids_deaths_15_24_F,
    "DeathsHIV_25_49_F": aids_deaths_25_49_F,
    "DeathsHIV_50_UP_F": aids_deaths_50_F,
}

with pd.ExcelWriter(results_folder / "full_summarised_deaths.xlsx") as writer:
    wrote_at_least_one_sheet = False

    for draw in range(info['number_of_draws']):
        rows = []
        column_labels = None  # Will hold the column names (e.g., years)

        for name, df in dataframes.items():
            try:
                values = df[(draw, 'mean')]
                column_labels = values.index
            except KeyError:
                # Column (draw, 'mean') not present
                values = pd.Series([None] * df.shape[1])
                if column_labels is None:
                    column_labels = [f"col{i}" for i in range(df.shape[1])]

            # Combine scenario name with values
            row = pd.concat([pd.Series([name], index=["Scenario"]), values])
            rows.append(row)

        if rows:
            combined_df = pd.DataFrame(rows)
            combined_df.columns = ["Scenario"] + list(column_labels)
            combined_df.to_excel(writer, sheet_name=f"Draw {draw}", index=False)
            wrote_at_least_one_sheet = True

    if not wrote_at_least_one_sheet:
        raise ValueError("No sheets were written: verify (draw, 'mean') columns exist in your dataframes.")



# merge the deaths data into merged_ouput
for var_name, df_var in dataframes.items():
    # get the unique draw IDs from the first level of the column MultiIndex
    draw_ids = df_var.columns.get_level_values('draw').unique()

    for draw_id in draw_ids:
        # select the (draw_id, 'mean') column as a Series
        series = df_var[(draw_id, 'mean')]   # index = dates

        # if merged_output keys are strings, uncomment the next line:
        # draw_key = str(draw_id)
        # otherwise, use draw_id directly:
        target_df = merged_output[draw_id]

        # write this variable into the target dataframe
        target_df[var_name] = series




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


def find_difference_relative_to_comparison_series(
    _ser: pd.Series,
    comparison: str,
    scaled: bool = False,
    drop_comparison: bool = True,
):
    """Find the difference in the values in a pd.Series with a multi-index, between the draws (level 0)
    within the runs (level 1), relative to where draw = `comparison`.
    The comparison is `X - COMPARISON`."""
    return _ser \
        .unstack(level=0) \
        .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=1) \
        .drop(columns=([comparison] if drop_comparison else [])) \
        .stack()


def find_difference_relative_to_comparison_series_dataframe(_df: pd.DataFrame, **kwargs):
    """Apply `find_difference_relative_to_comparison_series` to each row in a dataframe"""
    return pd.concat({
        _idx: find_difference_relative_to_comparison_series(row, **kwargs)
        for _idx, row in _df.iterrows()
    }, axis=1).T



daly_by_cause = compute_summary_statistics(extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=get_num_dalys_by_year,
        do_scaling=True,
    ),
    central_measure='mean'
)


# dalys_labelled_diff_from_statusquo = compute_summary_statistics(
#     find_difference_relative_to_comparison_series_dataframe(
#         daly_by_cause,
#         comparison=1
#     ),
#     central_measure='mean'
# )

# 1. Keep only stat = 'mean' columns, dropping the second level
df_mean = daly_by_cause.xs('central', axis=1, level='stat', drop_level=True)

# 2. Keep only rows where level 1 (cause) == 'AIDS'
df_filtered = df_mean.xs('AIDS', axis=0, level=1, drop_level=False)
df_filtered = df_filtered.droplevel(1)


with pd.ExcelWriter(results_folder / 'aids_dalys.xlsx', engine='openpyxl') as writer:
    df_filtered.to_excel(writer, sheet_name='DALYs')



# add this to merged_outputs dict
for draw_id in df_filtered.columns:
    # adjust if merged_output keys are strings:
    key = draw_id          # or: key = str(draw_id)

    target_df = merged_output[key]

    dalys_values = df_filtered[draw_id].to_numpy()

    # optional safety check
    if len(target_df) != len(dalys_values):
        raise ValueError(
            f"Length mismatch for draw {draw_id}: "
            f"merged_output has {len(target_df)} rows, df_filtered has {len(dalys_values)}"
        )

    # assign by position, ignoring index labels
    target_df["DALYs_Undiscounted"] = dalys_values

    merged_output[key] = target_df








# ---------------------------------------------------------------------------------------------


def summarise_births(results_folder, mother_has_hiv=None):
    """
    Returns mean births for each year of the simulation.
    Values are aggregated across runs of each draw for the specified cause and subgroup.
    """

    # Load death results with grouped series
    results_births = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="on_birth",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(
                ["year", "mother_has_hiv"]
            )["child"].count()
        ),
        do_scaling=True,
    ).reset_index()

    # Filter for cause of death if specified
    if mother_has_hiv:
        tmp = results_births.loc[results_births.mother_has_hiv == True]
    else:
        tmp = results_births

    # Identify draw/run columns by checking for tuple column names where first element is int
    draw_run_columns = [col for col in tmp.columns if isinstance(col, tuple) and isinstance(col[0], int)]

    # Group by year and sum across selected draw columns
    summed_results = tmp.groupby('year')[draw_run_columns].sum().reset_index()

    # Compute mean across runs for each draw
    mean_births = pd.concat(
        {'mean': summed_results.iloc[:, 1:].groupby(level=0, axis=1).mean()},
        axis=1
    ).swaplevel(axis=1)

    return mean_births


all_births = summarise_births(results_folder, mother_has_hiv=False)
hiv_births = summarise_births(results_folder, mother_has_hiv=True)

# List of dataframes to include in the workbook
dataframes = {
    "Birth_All": all_births,
    "Birth_HIV": hiv_births,
}

with pd.ExcelWriter(results_folder / "full_summarised_births.xlsx") as writer:
    wrote_at_least_one_sheet = False

    for draw in range(info['number_of_draws']):
        rows = []
        column_labels = None  # Will hold the column names (e.g., years)

        for name, df in dataframes.items():
            try:
                values = df[(draw, 'mean')]
                column_labels = values.index
            except KeyError:
                # Column (draw, 'mean') not present
                values = pd.Series([None] * df.shape[1])
                if column_labels is None:
                    column_labels = [f"col{i}" for i in range(df.shape[1])]

            # Combine scenario name with values
            row = pd.concat([pd.Series([name], index=["Scenario"]), values])
            rows.append(row)

        if rows:
            combined_df = pd.DataFrame(rows)
            combined_df.columns = ["Scenario"] + list(column_labels)
            combined_df.to_excel(writer, sheet_name=f"Draw {draw}", index=False)
            wrote_at_least_one_sheet = True

    if not wrote_at_least_one_sheet:
        raise ValueError("No sheets were written: verify (draw, 'mean') columns exist in your dataframes.")
