
from __future__ import annotations

import datetime
from pathlib import Path

# import lacroix
from typing import Iterable, Sequence, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import seaborn as sns

from tlo import Date

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    make_age_grp_lookup,
    make_age_grp_types,
    compute_summary_statistics,
)

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")
# outputspath = Path("./outputs")

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("hiv_program_simplification", outputspath)[-1]


# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder, draw=1, run=1)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

TARGET_PERIOD = (Date(2025, 1, 1), Date(2050, 12, 31))

# extract scaling factor
scaling_factor = extract_results(
    results_folder,
    module="tlo.methods.population",
    key="scaling_factor",
    column="scaling_factor",
    index="date",
    do_scaling=False)


def target_period() -> str:
    """Returns the target period as a string of the form YYYY-YYYY"""
    return "-".join(str(t.year) for t in TARGET_PERIOD)


def get_parameter_names_from_scenario_file() -> Tuple[str]:
    """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
    from scripts.hiv.program_simplification.analysis_hiv_program_simplification import (
        HIV_Progam_Elements,
    )
    e = HIV_Progam_Elements()
    return tuple(e._scenarios.keys())

def set_param_names_as_column_index_level_0(_df):
    """Set the columns index (level 0) as the param_names."""
    ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
    assert len(names_of_cols_level0) == len(_df.columns.levels[0])
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df

param_names = get_parameter_names_from_scenario_file()




# %% -------------------------------------------------------------------------------------------------------
# EXTRACT SERVICES USED USING TREATMENT_ID

# extract numbers of appts delivered for every run within a specified draw
def sum_appt_by_id(results_folder, module, key, column, draw):
    """
    sum occurrences of each treatment_id over the simulation period for every run within a draw

    produces dataframe: rows=treatment_id, columns=counts for every run

    results are scaled to true population size
    """

    info = get_scenario_info(results_folder)
    # create emtpy dataframe
    results = pd.DataFrame()

    for run in range(info['runs_per_draw']):
        df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]

        new = df[['date', column]].copy()
        tmp = pd.DataFrame(new[column].to_list())

        # sum each column to get total appts of each type over the simulation
        tmp2 = pd.DataFrame(tmp.sum())
        # add results to dataframe for output
        results = pd.concat([results, tmp2], axis=1)

    # multiply appt numbers by scaling factor
    results = results.mul(scaling_factor.values[0][0])

    return results


# extract numbers of appts
module = "tlo.methods.healthsystem.summary"
key = 'HSI_Event'
column = 'TREATMENT_ID'

# get total counts of every appt type for each scenario
appt_sums = sum_appt_by_id(results_folder,
                           module=module, key=key, column=column, draw=0)
appt_sums.to_csv(outputspath / "Apr2024_HTMresults/appt_sums_baseline.csv")










# get HCW time by mapping appts to person-time
hcw_time = pd.read_csv("resources/healthsystem/human_resources/definitions/ResourceFile_Appt_Time_Table.csv")

# assume all services delivered at facility level 1a


# Filter mapping for facility level 1a
hcw_map = hcw_time.loc[hcw_time["Facility_Level"] == "1a"]

# Map table: Appt type → minutes per cadre
map_table = hcw_map.pivot_table(index="Appt_Type_Code",
                                columns="Officer_Category",
                                values="Time_Taken_Mins",
                                aggfunc="mean")

# Align rows with appointment counts
map_table = map_table.reindex(appt_counts.index)

# Multiply counts × minutes, sum over appt types, one total per cadre
per_cadre = {}
for cadre in map_table.columns:
    contrib = appt_counts.mul(map_table[cadre], axis=0)
    per_cadre[cadre] = contrib.sum(axis=0)

# Final dataframe: rows = cadres, columns = same as appt_counts
hcw_minutes = pd.DataFrame(per_cadre).T
hcw_hours = hcw_minutes[appt_counts.columns] / 60


# get the difference in hcw across the runs

num_hcw_hours_diff = compute_summary_statistics(
    find_difference_relative_to_comparison_series_dataframe(
        hcw_hours,
        comparison='Status Quo'
    ), central_measure='mean'
)


hcw_hours.to_csv(results_folder / f'hcw_hours_{target_period()}.csv')
num_hcw_hours_diff.to_csv(results_folder / f'num_hcw_hours_diff_{target_period()}.csv')


draw_order = hcw_minutes.columns.get_level_values("draw").unique()

# Reorder the index
num_hcw_hours_diff = num_hcw_hours_diff.reindex(draw_order, axis=1, level="draw")

# remove Status Quo columns
num_hcw_hours_diff_edit = num_hcw_hours_diff.drop(columns="Status Quo", level="draw")

