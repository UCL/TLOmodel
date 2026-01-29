
from __future__ import annotations

import datetime
from pathlib import Path

# import lacroix
from typing import Iterable, Sequence, Optional, Tuple, Union, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import ast
from collections import Counter, defaultdict
from pathlib import Path

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


########################
## TREATMENT_ID

# extract numbers of appts delivered for every run within a specified draw
def _parse_dict_cell(x) -> Dict[str, float]:
    if isinstance(x, dict):
        return x
    if pd.isna(x):
        return {}
    if isinstance(x, str):
        return ast.literal_eval(x)
    return {}



def make_series_treatment_counts_by_year(
    treatment_col: str = "TREATMENT_ID",
    date_col: str = "date",
    TARGET_PERIOD: Optional[Tuple[object, object]] = None,
):
    """
    Returns a function suitable for `custom_generate_series`.

    Output Series:
      - index: MultiIndex (year, treatment_id)
      - values: counts summed within year
    """
    def custom_generate_series(df: pd.DataFrame) -> pd.Series:

        # Filter to target period (your established pattern)
        if TARGET_PERIOD is not None:
            df = df.loc[pd.to_datetime(df.date).between(*TARGET_PERIOD)]

        if df.empty:
            return pd.Series(dtype=float)

        years = pd.to_datetime(df[date_col]).dt.year

        by_year: Dict[int, Counter] = {}

        for yr, cell in zip(years, df[treatment_col]):
            d = _parse_dict_cell(cell)
            by_year.setdefault(int(yr), Counter()).update(d)

        wide = pd.DataFrame({yr: dict(cnt) for yr, cnt in by_year.items()}).T
        wide.index.name = "year"
        wide = wide.sort_index()

        s = wide.stack(dropna=False)
        s.index.names = ["year", "treatment_id"]
        s.name = "count"

        return s

    return custom_generate_series


module = "tlo.methods.healthsystem.summary"
key = "HSI_Event"

custom = make_series_treatment_counts_by_year(
    treatment_col="TREATMENT_ID",
    TARGET_PERIOD=TARGET_PERIOD,
)

# treatment_by_year is a DataFrame with:
#   - index: MultiIndex (year, treatment_id)
#   - columns: MultiIndex (draw, run)
#   - values: counts
treatment_by_year = extract_results(
    results_folder=results_folder,
    module=module,
    key=key,
    custom_generate_series=custom,
    do_scaling=True,
)

treatment_by_year.to_excel(results_folder / "treatment_id_counts_by_year_draw_run.xlsx")

treatment_by_year_hiv = treatment_by_year.loc[
    treatment_by_year.index
        .get_level_values("treatment_id")
        .str.startswith("Hiv")
]
treatment_by_year_hiv.to_excel(results_folder / "treatment_by_year_hiv.xlsx")



########################
## APPT TYPES BY LEVEL


def make_series_appt_counts_by_year_and_facility(
    treatment_col: str = "Number_By_Appt_Type_Code_And_Level",
    date_col: str = "date",
    TARGET_PERIOD: Optional[Tuple[object, object]] = None,
):
    """
    For cells of the form:
        { '0': {'ConWithDCSA': 102, ...},
          '1a': {'Over5OPD': 38099, ...},
          '1b': {...},
          ... }

    Returns a function suitable for custom_generate_series, producing a pd.Series with:
      - index: MultiIndex (year, facility_level, appt_type)
      - values: counts summed within year for that facility_level/appt_type
    """
    def custom_generate_series(df: pd.DataFrame) -> pd.Series:
        # Filter to target period (your pattern)
        if TARGET_PERIOD is not None:
            df = df.loc[pd.to_datetime(df.date).between(*TARGET_PERIOD)]

        if df.empty:
            return pd.Series(dtype=float)

        years = pd.to_datetime(df[date_col]).dt.year

        # by_year_fac[year][facility_level] = Counter(appt_type -> count)
        by_year_fac = defaultdict(lambda: defaultdict(Counter))

        for yr, cell in zip(years, df[treatment_col]):
            nested = _parse_dict_cell(cell)  # dict: facility_level -> dict(appt_type -> count)

            if not isinstance(nested, dict):
                continue

            for fac_level, inner in nested.items():
                if isinstance(inner, dict) and inner:
                    by_year_fac[int(yr)][str(fac_level)].update(inner)

        # Convert to long records then to Series
        records = []
        for yr, fac_dict in by_year_fac.items():
            for fac_level, cnt in fac_dict.items():
                for appt_type, val in cnt.items():
                    records.append((yr, fac_level, appt_type, val))

        if not records:
            return pd.Series(dtype=float)

        out = pd.DataFrame(records, columns=["year", "facility_level", "appt_type", "count"])
        s = out.set_index(["year", "facility_level", "appt_type"])["count"].sort_index()
        s.index.names = ["year", "facility_level", "treatment_id"]  # keep your naming convention
        s.name = "count"
        return s

    return custom_generate_series


# appt types
key="HSI_Event_non_blank_appt_footprint"
appt_numbers = make_series_appt_counts_by_year_and_facility(
    treatment_col="Number_By_Appt_Type_Code_And_Level",
    TARGET_PERIOD=TARGET_PERIOD,
)

appt_by_year_facility = extract_results(
    results_folder=results_folder,
    module=module,
    key=key,
    custom_generate_series=appt_numbers,
    do_scaling=True,
)

appt_by_year_facility.to_excel(results_folder / "appt_by_year_facility.xlsx")

# HIV-related appts:
# VCTNegative
# VCTPositive
# MaleCirc
# NewAdult
# EstMedCom
# EstNonCom
# PMTCT
# Peds


# get HCW time by mapping appts to person-time
hcw_time = pd.read_csv("resources/healthsystem/human_resources/definitions/ResourceFile_Appt_Time_Table.csv")

# assume all services delivered at facility level 1a


# # Filter mapping for facility level 1a
# hcw_map = hcw_time.loc[
#     hcw_time["Facility_Level"].isin(["0", "1a"])
# ]


# todo add facility level
# Map table: Appt type → minutes per cadre
map_table = hcw_time.pivot_table(index="Appt_Type_Code",
                                columns="Officer_Category",
                                values="Time_Taken_Mins",
                                aggfunc="mean")



def hcw_time_by_year(
    appt_by_year: pd.DataFrame,
    map_table: pd.DataFrame,
    *,
    unit: str = "hours",   # "minutes" or "hours"
) -> dict[str, pd.DataFrame]:
    """
    Returns a dict: {officer_category: DataFrame(index=year, columns=(draw,run), values=time)}.
    """

    # 1) Appointments to tidy (one row per year, treatment_id, draw, run)
    appt_long = (
        appt_by_year
        .stack(["draw", "run"], future_stack=True)
        .rename("n_appts")
        .reset_index()
    )
    # columns now: year, treatment_id, draw, run, n_appts

    # 2) Map table to tidy weights (one row per appt type, officer category)
    weights_long = (
        map_table
        .fillna(0)
        .reset_index()  # brings Appt_Type_Code out as a column
        .melt(
            id_vars=["Appt_Type_Code"],
            var_name="Officer_Category",
            value_name="minutes_per_appt",
        )
        .rename(columns={"Appt_Type_Code": "treatment_id"})
    )

    # 3) Merge and compute time
    merged = appt_long.merge(weights_long, on="treatment_id", how="left")
    merged["minutes_per_appt"] = merged["minutes_per_appt"].fillna(0)
    merged["time_minutes"] = merged["n_appts"] * merged["minutes_per_appt"]

    # 4) Aggregate to year/draw/run/officer_category
    out_long = (
        merged
        .groupby(["Officer_Category", "year", "draw", "run"], as_index=False)["time_minutes"]
        .sum()
    )

    if unit == "hours":
        out_long["time"] = out_long["time_minutes"] / 60.0
    elif unit == "minutes":
        out_long["time"] = out_long["time_minutes"]
    else:
        raise ValueError("unit must be 'minutes' or 'hours'")

    # 5) Split into one DataFrame per officer category, pivot back to (draw,run) columns
    out = {}
    for oc, g in out_long.groupby("Officer_Category", sort=True):
        wide = (
            g.pivot(index="year", columns=["draw", "run"], values="time")
             .sort_index()
        )
        wide.columns.names = ["draw", "run"]
        out[oc] = wide

    return out


time_by_officer = hcw_time_by_year(appt_by_year_facility, map_table, unit="hours")



clinical_time = time_by_officer["Clinical"]                 # index=year, cols=(draw,run)
nursing_time  = time_by_officer["Nursing_and_Midwifery"]
pharm_time = time_by_officer["Pharmacy"]
# mental_time = time_by_officer["Mental"]
# nutrition_time = time_by_officer["Nutrition"]
# radiography_time = time_by_officer["Radiography"]
lab_time = time_by_officer["Laboratory"]
dsca_time = time_by_officer["DCSA"]
dental_time = time_by_officer["Dental"]




# read in the HRH cost sheet
hcw_costs = pd.read_csv("resources/ResourceFile_HIV/hrh_costs.csv")


# hourly-cost lookup for facility level 1a
def apply_level_1a_hourly_costs(
    time_by_officer: dict[str, pd.DataFrame],
    hcw_costs: pd.DataFrame,
    *,
    facility_level: str = "1a",
    unit: str = "hours",   # "hours" or "minutes"
) -> pd.DataFrame:
    """
    Multiply cadre time by cadre-specific hourly costs (assuming a single facility level for all activity).
    """

    # Cost lookup for the chosen facility level
    cost_level = (
        hcw_costs
        .loc[hcw_costs["Facility_Level"] == facility_level, ["Officer_Category", "Total_hourly_cost"]]
        .dropna(subset=["Total_hourly_cost"])
        .set_index("Officer_Category")["Total_hourly_cost"]
    )

    blocks = {}

    for officer_category, df_time in time_by_officer.items():
        if officer_category not in cost_level.index:
            # If no cost available for that cadre at this facility level, skip explicitly.
            # Alternatively: raise KeyError to force completeness.
            continue

        df_hours = df_time / 60.0 if unit == "minutes" else df_time
        blocks[officer_category] = df_hours * float(cost_level.loc[officer_category])

    out = pd.concat(blocks, axis=1)
    out.columns.names = ["Officer_Category", "draw", "run"]
    out.index.name = "year"

    return out



time_by_officer = {
    "Clinical": clinical_time,  # year x (draw,run)
    "Nursing_and_Midwifery": nursing_time,
    "Pharmacy": pharm_time,
    # add others you have...
}

hcw_costs_by_year = apply_level_1a_hourly_costs(
    time_by_officer=time_by_officer,
    hcw_costs=hcw_costs,
    facility_level="1a",
    unit="hours",
)

total_cost_by_year = hcw_costs_by_year.groupby(axis=1, level=["draw", "run"]).sum()














