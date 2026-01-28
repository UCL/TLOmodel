
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




def summarise_appointments(df: pd.DataFrame) -> pd.Series:
    """
    Extract and sum all appointment types during the TARGET_PERIOD from the HSI_Event log.

    Returns a Series indexed by appointment type.
    """
    df["date"] = pd.to_datetime(df["date"])
    mask = df["date"].between(*TARGET_PERIOD)
    filtered = df.loc[mask, "Number_By_Appt_Type_Code"]

    # Expand list of counts per row into a DataFrame
    expanded = pd.DataFrame(filtered.tolist())

    # Sum across all rows (time points)
    summed = expanded.sum()

    # Return as Series with integer index
    summed.index.name = "AppointmentTypeCode"
    return summed

appt_counts = extract_results(
    results_folder=results_folder,
    module="tlo.methods.healthsystem.summary",
    key="HSI_Event_non_blank_appt_footprint",
    custom_generate_series=summarise_appointments,
    do_scaling=True  # to scale to national population
).pipe(set_param_names_as_column_index_level_0)

appt_counts_by_draw = appt_counts.groupby(level="draw", axis=1).mean()
appt_counts_by_draw.to_excel(results_folder / "appt_counts_by_draw.xlsx")




appt_diff_from_statusquo = compute_summary_statistics(
    find_difference_relative_to_comparison_series_dataframe(
        appt_counts,
        comparison="Status Quo"
    )
)
appt_diff_from_statusquo.to_excel(results_folder / "appt_diff_from_statusquo.xlsx")


pc_diff_appt_counts_vs_statusquo = 100.0 * compute_summary_statistics(
    pd.DataFrame(
        find_difference_relative_to_comparison_series_dataframe(
            appt_counts,
            comparison='Status Quo',
        scaled=True)
    ), only_central=True
)








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

