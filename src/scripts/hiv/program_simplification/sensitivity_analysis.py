
from __future__ import annotations

import datetime
from pathlib import Path
import math

# import lacroix
from typing import Iterable, Sequence, Optional, Tuple, Union, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import ast
from collections import Counter, defaultdict
from ast import literal_eval

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

from scripts.costing.cost_estimation import (
    estimate_input_cost_of_scenarios, summarize_cost_data,
    do_stacked_bar_plot_of_cost_by_category, do_line_plot_of_cost,
    create_summary_treemap_by_cost_subgroup, estimate_projected_health_spending
)
from scripts.costing.cost_estimation import load_unit_cost_assumptions


outputspath = Path("./outputs/t.mangal@imperial.ac.uk")
# outputspath = Path("./outputs")
resourcefilepath = Path("./resources")

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


# todo change
TARGET_PERIOD = (Date(2025, 1, 1), Date(2040, 12, 31))




# %% FUNCTIONS

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





# %% DALYs


def num_dalys_by_cause(_df):
    """Return total number of DALYS (Stacked) (total by age-group within the TARGET_PERIOD)"""
    return _df \
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
        .drop(columns=['date', 'sex', 'age_range']) \
        .sum()


def num_dalys_by_cause_year(_df):
    """Return total number of DALYS by cause and year (total by age-group within the TARGET_PERIOD)"""
    result = _df \
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
        .drop(columns=['date', 'sex', 'age_range']) \
        .groupby('year') \
        .sum()

    # Stack to convert DataFrame to Series with MultiIndex (year, cause)
    return result.stack()


# extract dalys by cause for each run/draw
daly_by_cause_year = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=num_dalys_by_cause_year,
        do_scaling=True,
    ).pipe(set_param_names_as_column_index_level_0)






# %% Deaths

def extract_deaths_by_cause(results_folder):
    """ returns deaths for each year of the simulation within TARGET_PERIOD
    values are aggregated across the runs of each draw
    for the specified cause
    """

    def get_num_deaths_by_cause_label(_df):
        """Return total number of Deaths by label for each year within TARGET_PERIOD
        values are summed for all ages
        df returned: MultiIndex with rows=(label, year), values=count
        """
        # Filter by TARGET_PERIOD first, then extract year
        _df_filtered = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]
        _df_filtered = _df_filtered.copy()
        _df_filtered['year'] = pd.to_datetime(_df_filtered['date']).dt.year

        return _df_filtered \
            .groupby(['label', 'year']) \
            .size()

    num_deaths_by_label = extract_results(
        results_folder,
        module='tlo.methods.demography',
        key='death',
        custom_generate_series=get_num_deaths_by_cause_label,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    causes = {
        'AIDS': 'HIV/AIDS',
        '': 'Other',
    }

    # Map the first level of the index (label) to cause names
    # Keep the year level intact
    new_index = []
    for label, year in num_deaths_by_label.index:
        cause = causes.get(label, 'Other')
        new_index.append((cause, year))

    # Create new MultiIndex with (cause, year)
    num_deaths_by_label.index = pd.MultiIndex.from_tuples(new_index, names=['cause', 'year'])

    # Group by both cause and year to sum across original labels that map to same cause
    grouped_deaths = num_deaths_by_label.groupby(['cause', 'year']).sum()

    # Reorder based on the causes values that are in the grouped data
    ordered_causes = [cause for cause in causes.values() if cause in grouped_deaths.index.get_level_values('cause')]

    # Reindex to get consistent ordering of causes
    if ordered_causes:
        grouped_deaths = grouped_deaths.reindex(ordered_causes, level='cause')

    return grouped_deaths


num_deaths_by_cause = extract_deaths_by_cause(results_folder)




# AIDS deaths only
hiv_deaths = num_deaths_by_cause.loc[num_deaths_by_cause.index.get_level_values('cause') == 'HIV/AIDS']





#%% ####################### HS use  #######################

# get numbers by treatment ID by year
# map to expected facility level -> list of treatment IDs by facility level by year
# map to appt type by facility level
# map to cadre time required
# then can sum for the plots


#########################
# extract treatment id
#########################

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
).pipe(set_param_names_as_column_index_level_0)


treatment_by_year_hiv = treatment_by_year.loc[
    treatment_by_year.index
        .get_level_values("treatment_id")
        .str.startswith("Hiv")
]




#########################
# extract appt types
#########################


# get appt types by facility level and year
def summarise_appointments(df: pd.DataFrame) -> pd.Series:
    """
    Sum appointment type counts by calendar year *and facility level* (within TARGET_PERIOD)
    from the HSI_Event log.

    Returns
    -------
    pd.Series
        MultiIndex (year, facility_level, AppointmentTypeCode) -> total count
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])

    mask = d["date"].between(*TARGET_PERIOD)
    d = d.loc[mask, ["date", "Number_By_Appt_Type_Code_And_Level"]]

    # Collect long-form rows: (date, year, level, appt_type, count)
    rows = []
    for dt, level_dict in zip(d["date"].values, d["Number_By_Appt_Type_Code_And_Level"].values):
        if not isinstance(level_dict, dict) or len(level_dict) == 0:
            continue
        year = pd.Timestamp(dt).year

        for level, appt_dict in level_dict.items():
            if not isinstance(appt_dict, dict) or len(appt_dict) == 0:
                continue
            for appt_type, count in appt_dict.items():
                if count is None:
                    continue
                rows.append((year, str(level), appt_type, float(count)))

    if not rows:
        # Empty, but keep the expected index names
        return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], [], []],
                                                                      names=["year", "facility_level", "AppointmentTypeCode"]))

    long = pd.DataFrame(rows, columns=["year", "facility_level", "AppointmentTypeCode", "count"])

    out = (
        long.groupby(["year", "facility_level", "AppointmentTypeCode"], sort=False)["count"]
        .sum()
    )
    out.index = out.index.set_names(["year", "facility_level", "AppointmentTypeCode"])
    return out


appt_counts = (
    extract_results(
        results_folder=results_folder,
        module="tlo.methods.healthsystem.summary",
        key="HSI_Event_non_blank_appt_footprint",
        custom_generate_series=summarise_appointments,
        do_scaling=True,  # scale to national population
    )
    .pipe(set_param_names_as_column_index_level_0)
)


# remove unneeded appt types, keep only those from hiv program

# Explicit allow-list of appointment types to KEEP
KEEP_APPT_TYPES = [
    "VCTPositive",
    "VCTNegative",
    "NewAdult",
    "Peds",
    "EstNonCom",
    "MaleCirc",
]


################################
# map treatment id to appt types
################################


TREATMENT_TO_APPT_SPEC = {
    "PharmDispensing": ("Hiv_Prevention_Prep",      "1a",  1.0),
    "ConWithDCSA":     ("Hiv_Test_Selftest",    "0",   1.0),
    "IPAdmission":     ("Hiv_PalliativeCare",     "2",   2.0),
    "InpatientDays":   ("Hiv_PalliativeCare",     "2",  17.0),
}


def keep_selected_appt_types(
    appt_counts: pd.Series | pd.DataFrame,
    appt_types_to_keep: list[str],
    *,
    appt_level: str = "AppointmentTypeCode",
) -> pd.Series | pd.DataFrame:
    """
    Keep only selected appointment types (across all years and facility levels).
    """
    idx = appt_counts.index
    mask_keep = idx.get_level_values(appt_level).isin(appt_types_to_keep)
    return appt_counts.loc[mask_keep].copy()


def add_mapped_treatments_as_appt_types(
    appt_counts_base: pd.DataFrame,
    trt_counts: pd.DataFrame,
    mapping_appt_to_spec: dict[str, tuple[str, str, float]],
    *,
    year_level: str = "year",
    trt_level: str = "treatment_id",
) -> pd.DataFrame:
    """
    Sparse behaviour:
      - does NOT create a full year×level×type grid
      - only creates rows for the (year, specified facility_level, appt_type) that are inserted
      - overwrites if those rows already exist
    """
    out = appt_counts_base.copy()

    trt = trt_counts.copy().sort_index()

    for appt_type, (trt_id, facility_level, mult) in mapping_appt_to_spec.items():
        # Select treatment counts indexed by year
        trt_block = trt.xs(trt_id, level=trt_level, drop_level=True).copy()
        trt_block = trt_block * float(mult)

        years = trt_block.index.get_level_values(year_level)

        target_index = pd.MultiIndex.from_arrays(
            [
                years,
                pd.Index([str(facility_level)] * len(years)),
                pd.Index([appt_type] * len(years)),
            ],
            names=["year", "facility_level", "AppointmentTypeCode"],
        )

        src = trt_block.copy()
        src.index = target_index

        # Add rows as needed, then assign
        out = out.reindex(out.index.union(target_index))
        out.loc[target_index, :] = src

    return out


# ----
appt_counts_kept = keep_selected_appt_types(appt_counts, KEEP_APPT_TYPES)

appt_counts_final = add_mapped_treatments_as_appt_types(
    appt_counts_kept,
    treatment_by_year_hiv,
    TREATMENT_TO_APPT_SPEC,
)


################################
# map appt numbers to HCW time
################################

# get HCW time by mapping appts to person-time
hcw_time = pd.read_csv("resources/healthsystem/human_resources/definitions/ResourceFile_Appt_Time_Table.csv")



def appt_counts_to_hcw_minutes(appt_counts_final: pd.DataFrame,
                              hcw_time: pd.DataFrame,
                              fill_missing_minutes: float | None = None) -> pd.DataFrame:
    """
    Returns a DataFrame with the SAME columns as appt_counts_final, and an expanded MultiIndex:
      ['year','facility_level','AppointmentTypeCode','hcw_cadre']
    Values are: (number of appointments) × (minutes per appointment for that cadre at that facility level).
    """

    # --- normalise keys to strings for exact matching ---
    counts = appt_counts_final.copy()
    counts = counts.rename_axis(index={
        "facility_level": "Facility_Level",
        "AppointmentTypeCode": "Appt_Type_Code"
    })

    # Ensure index levels are string (facility/appt)
    counts_idx = counts.index
    counts = counts.copy()
    counts.index = pd.MultiIndex.from_arrays(
        [
            counts_idx.get_level_values("year"),
            counts_idx.get_level_values("Facility_Level").astype(str).str.strip(),
            counts_idx.get_level_values("Appt_Type_Code").astype(str).str.strip(),
        ],
        names=["year", "Facility_Level", "Appt_Type_Code"]
    )

    hcw = hcw_time.copy()
    hcw["Facility_Level"] = hcw["Facility_Level"].astype(str).str.strip()
    hcw["Appt_Type_Code"] = hcw["Appt_Type_Code"].astype(str).str.strip()
    hcw["Officer_Category"] = hcw["Officer_Category"].astype(str).str.strip()

    # --- build minutes-per-appointment map: (Facility_Level, Appt_Type_Code) × cadre ---
    map_table = hcw.pivot_table(
        index=["Facility_Level", "Appt_Type_Code"],
        columns="Officer_Category",
        values="Time_Taken_Mins",
        aggfunc="mean"
    )

    cadres = list(map_table.columns)

    # Target index for aligning minutes with counts rows (drop year)
    target = pd.MultiIndex.from_arrays(
        [
            counts.index.get_level_values("Facility_Level"),
            counts.index.get_level_values("Appt_Type_Code"),
        ],
        names=["Facility_Level", "Appt_Type_Code"]
    )

    out_parts = []
    nrows = len(counts)

    # Multiply counts (nrows × ncols) by mins_vec (nrows) per cadre
    for cadre in cadres:
        mins_c = map_table[cadre].reindex(target)  # Series aligned to counts rows (year-dropped index)

        if fill_missing_minutes is not None:
            mins_c = mins_c.fillna(fill_missing_minutes)

        # broadcasting across columns
        df_c = counts.mul(mins_c.to_numpy(), axis=0)

        # expand index with cadre level
        df_c.index = pd.MultiIndex.from_arrays(
            [
                counts.index.get_level_values("year"),
                counts.index.get_level_values("Facility_Level"),
                counts.index.get_level_values("Appt_Type_Code"),
                pd.Index([cadre] * nrows)
            ],
            names=["year", "facility_level", "AppointmentTypeCode", "hcw_cadre"]
        )

        out_parts.append(df_c)

    out = pd.concat(out_parts).sort_index()
    return out


# --- usage ---
minutes_by_appt_and_cadre = appt_counts_to_hcw_minutes(appt_counts_final, hcw_time, fill_missing_minutes=0.0)


minutes_per_year_summed_by_cadre = (
    minutes_by_appt_and_cadre
    .groupby(level=["year", "hcw_cadre"])
    .sum()
)

hours_per_year_cadre = minutes_per_year_summed_by_cadre / 60


minutes_all_years_summed_by_cadre = (
    minutes_by_appt_and_cadre
    .groupby(level=["hcw_cadre"])
    .sum()
)



################################
# add costs to HCW time
################################

# read in the HRH cost sheet
hrh_costs = pd.read_csv("resources/ResourceFile_HIV/hrh_costs.csv")


def hrh_time_costs_by_facility(
    minutes_by_appt_and_cadre: pd.DataFrame,
    hrh_costs: pd.DataFrame,
    minutes_to_hours: bool = True,
    fill_missing_cost: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute HCW time (hours) and costs by year × facility_level × cadre
    preserving columns (draw/run/scenario)
    """

    # --- 1) aggregate minutes to year × facility × cadre ---
    mins_yfc = (
        minutes_by_appt_and_cadre
        .groupby(level=["year", "facility_level", "hcw_cadre"])
        .sum()
    )

    # --- 2) convert to hours ---
    hours_yfc = mins_yfc / 60 if minutes_to_hours else mins_yfc.copy()

    # --- 3) prep cost lookup keyed on (facility_level, cadre) ---
    costs = hrh_costs.copy()
    costs["Facility_Level"] = costs["Facility_Level"].astype(str).str.strip()
    costs["Officer_Category"] = costs["Officer_Category"].astype(str).str.strip()

    # If duplicates exist per (Facility_Level, Officer_Category), average them
    cost_lookup = (
        costs
        .groupby(["Facility_Level", "Officer_Category"], as_index=True)["Total_hourly_cost"]
        .mean()
    )

    # --- 4) align lookup to hours_yfc rows ---
    idx = hours_yfc.index
    fac = idx.get_level_values("facility_level").astype(str).str.strip()
    cad = idx.get_level_values("hcw_cadre").astype(str).str.strip()

    hourly_cost_aligned = pd.MultiIndex.from_arrays(
        [fac, cad],
        names=["Facility_Level", "Officer_Category"]
    ).map(cost_lookup)

    if fill_missing_cost is not None:
        hourly_cost_aligned = hourly_cost_aligned.fillna(fill_missing_cost)

    # --- 5) cost = hours × hourly_cost (row-wise) ---
    costs_yfc = hours_yfc.mul(hourly_cost_aligned.to_numpy(), axis=0)

    return hours_yfc, costs_yfc


# --- usage ---
hours_by_year_fac_cadre, hrh_costs_by_year_fac_cadre = hrh_time_costs_by_facility(
    minutes_by_appt_and_cadre=minutes_by_appt_and_cadre,
    hrh_costs=hrh_costs,
    minutes_to_hours=True,
    fill_missing_cost=None   # or 0.0 if you want unmapped costs to contribute zero
)


# get the costs for all hcw by year
hrh_costs_by_year = (
    hrh_costs_by_year_fac_cadre
    .groupby(level="year")
    .sum()
)





####################################################################################
#%% get unit costs
####################################################################################


# Extract consumables dispensed data
def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])

def get_counts_of_items_requested(_df):
    _df = drop_outside_period(_df).copy()

    _df["year"] = pd.to_datetime(_df["date"]).dt.year
    _df["Item_Used"] = _df["Item_Used"].apply(
        lambda x: literal_eval(x) if isinstance(x, str) else x
    )

    # Turn dicts into list of (item, num) pairs and explode
    used = (
        _df[["year", "Item_Used"]]
        .assign(item_num=_df["Item_Used"].map(dict.items))
        .explode("item_num", ignore_index=True)
    )

    # Split tuple into columns
    tmp = used["item_num"].apply(
        lambda x: x if isinstance(x, tuple) else (pd.NA, 0)
    )

    used[["item", "value"]] = pd.DataFrame(tmp.tolist(), index=used.index)
    used["value"] = pd.to_numeric(used["value"], errors="coerce").fillna(0)

    # Aggregate
    return (
        used.groupby(["year", "item"], sort=False)["value"]
        .sum()
    )

# Extract results using your existing pipeline
cons_dispensed = extract_results(
    results_folder,
    module='tlo.methods.healthsystem.summary',
    key='Consumables',
    custom_generate_series=get_counts_of_items_requested,
    do_scaling=True
)



idx = pd.IndexSlice
cons_dispensed = cons_dispensed.apply(pd.to_numeric, errors="coerce")


# Add consumable name and unit cost
cons_dict = \
    pd.read_csv(resourcefilepath / 'healthsystem/consumables/ResourceFile_Consumables_Items_and_Packages.csv',
                low_memory=False,
                encoding="ISO-8859-1")[['Items', 'Item_Code']]

cons_dict = dict(zip(cons_dict['Item_Code'], cons_dict['Items']))

unit_costs = load_unit_cost_assumptions(resourcefilepath)

cons_costs_by_item_code = unit_costs["consumables"]
cons_costs_by_item_code = dict(zip(cons_costs_by_item_code['Item_Code'], cons_costs_by_item_code['Price_per_unit']))





def apply_unit_costs(
    cons_dispensed: pd.DataFrame,
    cons_costs_by_item_code: dict,
    *,
    item_level: str = "item",
    on_missing: str = "warn_nan",   # "warn_nan" | "zero" | "ignore"
) -> pd.DataFrame:
    """
    Multiply dispensed quantities by unit costs matched on the `item` level of the index.

    Missing costs will NOT raise an error.
    """

    # --- normalise cost dict keys to strings ---
    costs = {str(k).strip(): v for k, v in cons_costs_by_item_code.items()}

    # --- extract item codes from index ---
    items = cons_dispensed.index.get_level_values(item_level).astype(str).str.strip()

    # --- map unit costs ---
    unit_cost = pd.Series(items.map(costs), index=cons_dispensed.index, name="unit_cost")

    # --- handle missing costs ---
    missing_mask = unit_cost.isna()
    if missing_mask.any():
        missing_items = sorted(pd.unique(items[missing_mask]).tolist())

        if on_missing == "warn_nan":
            print(f"WARNING: Missing unit costs for {len(missing_items)} item(s): {missing_items[:20]}")
            # leave as NaN

        elif on_missing == "zero":
            print(f"WARNING: Missing unit costs set to 0 for items: {missing_items[:20]}")
            unit_cost = unit_cost.fillna(0.0)

        elif on_missing == "ignore":
            pass

    # --- multiply row-wise ---
    cons_costed = cons_dispensed.mul(unit_cost, axis=0)

    return cons_costed



cons_costed = apply_unit_costs(
    cons_dispensed,
    cons_costs_by_item_code,
    on_missing="warn_nan"
).pipe(set_param_names_as_column_index_level_0)


def get_item_codes_from_package_name(lookup_df: pd.DataFrame, package: str) -> int:
    return int(pd.unique(lookup_df.loc[lookup_df["Intervention_Pkg"] == package, "Item_Code"])[0])
    # return int(pd.unique(lookup_df.loc[lookup_df["Intervention_Pkg"] == package, "Item_Code"]))


def get_item_code_from_item_name(lookup_df: pd.DataFrame, item: str) -> int:
    """Helper function to provide the item_code (an int) when provided with the name of the item"""
    return int(pd.unique(lookup_df.loc[lookup_df["Items"] == item, "Item_Code"])[0])





## HIV consumables
# this is same as cons_dict but in df format not dict
items_list = pd.read_csv(
    resourcefilepath / "healthsystem/consumables/ResourceFile_Consumables_Items_and_Packages.csv")

hiv_item_codes_dict = dict()

# diagnostics
hiv_item_codes_dict["HIV test"] = get_item_code_from_item_name(
    items_list, "Test, HIV EIA Elisa")
hiv_item_codes_dict["Viral load"] = get_item_codes_from_package_name(
    items_list, "Viral Load")
hiv_item_codes_dict["VMMC"] = get_item_code_from_item_name(
    items_list, "male circumcision kit, consumables (10 procedures)_1_IDA")

# treatment
hiv_item_codes_dict["Adult PrEP"] = get_item_code_from_item_name(
    items_list, "Tenofovir (TDF)/Emtricitabine (FTC), tablet, 300/200 mg")
hiv_item_codes_dict["Infant PrEP"] = get_item_code_from_item_name(
    items_list, "Nevirapine, oral solution, 10 mg/ml")
hiv_item_codes_dict['First-line ART regimen: adult'] = get_item_code_from_item_name(
    items_list, "First-line ART regimen: adult")
hiv_item_codes_dict['First-line ART regimen: adult: cotrimoxazole'] = get_item_code_from_item_name(
    items_list, "Cotrimoxizole, 960mg pppy")

# ART for older children aged ("ART_age_cutoff_younger_child" < age <= "ART_age_cutoff_older_child"):
hiv_item_codes_dict['First line ART regimen: older child'] = get_item_code_from_item_name(
    items_list, "First line ART regimen: older child")

# ART for younger children aged (age < "ART_age_cutoff_younger_child"):
hiv_item_codes_dict['First line ART regimen: young child'] = get_item_code_from_item_name(
    items_list, "First line ART regimen: young child")
hiv_item_codes_dict['First line ART regimen: young child: cotrimoxazole'] = get_item_code_from_item_name(
    items_list, "Sulfamethoxazole + trimethropin, oral suspension, 240 mg, 100 ml")




# need to add self-tests
num_selftests = treatment_by_year_hiv.loc[
    treatment_by_year_hiv.index.get_level_values("treatment_id") == "Hiv_Test_Selftest"
]

selftests_costs = num_selftests * 3.14
# costs from mihpsa, cons only
# drop additional index treatment_id
selftests_costs = selftests_costs.droplevel("treatment_id")



# need to add tdf tests
def get_num_tdf(_df):
    """Return total number of TDF tests per year (from column n_tdf_tests_performed)."""

    years_needed = [i.year for i in TARGET_PERIOD]

    # ensure datetime
    df = _df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # extract year from date
    df["year"] = df["date"].dt.year

    assert set(df["year"].unique()).issuperset(years_needed), "Some years are not recorded."

    return (
        df.loc[df["year"].between(*years_needed)]
        .groupby("year")["n_tdf_tests_performed"]
        .sum()
    )



# num tdf tests
num_tdf_year = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        custom_generate_series=get_num_tdf,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)


tdf_costs = num_tdf_year * 6.86
# https://pmc.ncbi.nlm.nih.gov/articles/PMC12260116/?utm_source=chatgpt.com
# South Africa costs


# 'HIV test': 196,
#  'Viral load': 190,
#  'VMMC': 197,
#  'Adult PrEP': 1191,
#  'Infant PrEP': 198,
#  'First-line ART regimen: adult': 2671,
#  'First-line ART regimen: adult: cotrimoxazole': 204,
#  'First line ART regimen: older child': 2672,
#  'First line ART regimen: young child': 2673,
#  'First line ART regimen: young child: cotrimoxazole': 202



# select hiv item codes
# extract item codes (dict values)
hiv_item_codes = set(map(str, hiv_item_codes_dict.values()))

# filter rows where index level "item" is in that set
cons_costed_hiv = cons_costed.loc[
    cons_costed.index.get_level_values("item").isin(hiv_item_codes)
]

# add index to align with hiv cons
selftests_costs.index = pd.MultiIndex.from_product(
    [selftests_costs.index, [0]],
    names=["year", "item"]
)

tdf_costs.index = pd.MultiIndex.from_product(
    [tdf_costs.index, [1]],
    names=["year", "item"]
)


# add on the self-tests and tdf tests
cons_costed_hiv = (
    cons_costed_hiv.fillna(0)
    .add(selftests_costs.fillna(0), fill_value=0)
    .add(tdf_costs.fillna(0), fill_value=0)
    .fillna(0)
)



# sum across items, keeping only year
cons_summed_by_year = cons_costed_hiv.groupby(level="year").sum()


# total costs by year / draw / run
cons_hrh_costs_year_draw_run = cons_summed_by_year + hrh_costs_by_year


def discount_costs(costs_df, discount_rate=0.03, base_year=None):
    """
    Discount costs by specified rate each year

    Parameters:
    costs_df: DataFrame with year index and multi-column index (draw/run)
    discount_rate: Annual discount rate (default 3% = 0.03)
    base_year: Base year for discounting (default is minimum year in index)

    Returns:
    DataFrame with discounted costs, same structure as input
    """
    if base_year is None:
        base_year = costs_df.index.min()

    # Calculate discount factors for each year
    years = costs_df.index
    discount_factors = 1 / (1 + discount_rate) ** (years - base_year)

    # Apply discount factors to all columns
    discounted_costs = costs_df.multiply(discount_factors, axis=0)

    return discounted_costs


# 3% discount rate
discounted_costs_3pct = discount_costs(cons_hrh_costs_year_draw_run, discount_rate=0.03)

# 5% discount rate
discounted_costs_5pct = discount_costs(cons_hrh_costs_year_draw_run, discount_rate=0.05)



# %%
################################
#  ratios DALYs per $ saved
################################


# total DALYs by run
aids_dalys= daly_by_cause_year.loc[daly_by_cause_year.index.get_level_values(1) == 'AIDS'].droplevel(1)
aids_dalys_sum = aids_dalys.sum(axis=0)

aids_dalys_diff = find_difference_relative_to_comparison_series(
        aids_dalys_sum,
        comparison='Status Quo'
    )
# positive values indicate worse health



# total deaths by run
hiv_deaths_sum = hiv_deaths.sum(axis=0)

hiv_deaths_diff = find_difference_relative_to_comparison_series(
        hiv_deaths_sum,
        comparison='Status Quo'
    )



# total costs per run
total_costs = cons_hrh_costs_year_draw_run.sum(axis=0)
total_cost_diff = find_difference_relative_to_comparison_series(
        total_costs,
        comparison="Status Quo"
    )

total_costs_3percent = discounted_costs_3pct.sum(axis=0)
total_cost_diff_3percent = find_difference_relative_to_comparison_series(
        total_costs_3percent,
        comparison="Status Quo"
    )


total_costs_5percent = discounted_costs_5pct.sum(axis=0)
total_cost_diff_5percent = find_difference_relative_to_comparison_series(
        total_costs_5percent,
        comparison="Status Quo"
    )





def outcome_per_usd_saved(
    aids_dalys_diff: pd.Series,
    cost_diff: pd.Series,
    eps: float = 1e-9,
    per_million: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Compute USD saved per DALY lost ratio for each run/draw combination.

    Args:
        aids_dalys_diff: Series with MultiIndex (run, draw) - already summed across years
        cost_diff: Series with MultiIndex (run, draw) - already summed across years

    Returns:
        ratio: USD saved per DALY lost for each run/draw combination
        summary: Summary statistics for each draw (across runs)
    """

    # Align Series
    aids, cost = aids_dalys_diff.align(cost_diff, join="inner")

    # Calculate ratio for each run/draw
    usd_saved = -cost
    ratio = (usd_saved / aids).where(aids.abs() > eps)

    if per_million:
        ratio *= 1_000_000

    # Group by draws (level 1) to get summary statistics
    summary_data = []

    for draw in ratio.index.get_level_values(1).unique():
        # Get all runs for this draw
        draw_mask = ratio.index.get_level_values(1) == draw
        draw_ratios = ratio[draw_mask]
        draw_dalys = aids[draw_mask]
        draw_costs = cost[draw_mask]
        draw_usd_saved = usd_saved[draw_mask]

        mean_ratio = draw_ratios.mean(skipna=True)
        std_ratio = draw_ratios.std(skipna=True)

        summary_data.append({
            'draw': draw,
            'ratio_mean': mean_ratio,
            'ratio_ci_lower': mean_ratio - 1.96 * std_ratio,
            'ratio_ci_upper': mean_ratio + 1.96 * std_ratio,
            'n_cost_saving': (draw_usd_saved > 0).sum() / len(draw_usd_saved),
            'mean_dalys': draw_dalys.mean(),
            'mean_costs': draw_costs.mean(),
            'n_runs': len(draw_ratios.dropna())
        })

    summary = pd.DataFrame(summary_data).set_index('draw')

    return ratio, summary



def format_dalys_usd_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Format the DALYs per USD summary table for export with nice formatting.
    """

    # Round ratio values to nearest 100
    ratio_mean_rounded = (summary['ratio_mean'] / 100).round() * 100
    ratio_lower_rounded = (summary['ratio_ci_lower'] / 100).round() * 100
    ratio_upper_rounded = (summary['ratio_ci_upper'] / 100).round() * 100

    # Create formatted ratio column with CI
    formatted_ratio = (
        ratio_mean_rounded.astype(int).astype(str) +
        ' (' +
        ratio_lower_rounded.astype(int).astype(str) +
        ' - ' +
        ratio_upper_rounded.astype(int).astype(str) +
        ')'
    )

    # Create the formatted table
    formatted_table = pd.DataFrame({
        'Intervention': summary.index,
        'DALYs per Million USD Saved (95% CI)': formatted_ratio,
        'Cost Saving Scenarios': summary['n_cost_saving'],
        'Total Scenarios': summary['n_runs'],
        'Mean DALYs': summary['mean_dalys'].round(0).astype(int),
        'Mean Costs (USD)': summary['mean_costs'].round(0).astype(int)
    })

    # Reset index to make intervention names a regular column
    formatted_table = formatted_table.reset_index(drop=True)

    return formatted_table



### get dalys and deaths per $ saved


# $ saved per DALY lost
ratio_dalys, summary_dalys = outcome_per_usd_saved(
    aids_dalys_diff=aids_dalys_diff,
    cost_diff=total_cost_diff,
    per_million=False
)

ratio_dalys.to_excel(results_folder / f"dollars_saved_per_dalys_lost_full_{target_period()}.xlsx")
summary_dalys.to_excel(results_folder / f"dollars_saved_per_dalys_lost_{target_period()}.xlsx")


# Format for export
export_table = format_dalys_usd_table(summary_dalys)
export_table.to_excel(
    results_folder / f"dollars_saved_per_dalys_lost_summary_{target_period()}.xlsx",
    index=False
)


# deaths per $ saved
ratio_deaths, summary_deaths = outcome_per_usd_saved(
    aids_dalys_diff=hiv_deaths_diff,
    cost_diff=total_cost_diff,
    per_million=False
)

ratio_deaths.to_excel(results_folder / f"dollars_saved_per_death_full_{target_period()}.xlsx")
summary_deaths.to_excel(results_folder / f"dollars_saved_per_death_{target_period()}.xlsx")


# Format for export
export_table = format_dalys_usd_table(summary_deaths)
export_table.to_excel(
    results_folder / f"dollars_saved_per_death_summary_{target_period()}.xlsx",
    index=False
)



####### discounted ratios

# 3%
ratio_dalys_3percent, summary_dalys_3percent = outcome_per_usd_saved(
    aids_dalys_diff=aids_dalys_diff,
    cost_diff=total_cost_diff_3percent,
    per_million=False
)
ratio_dalys_3percent.to_excel(results_folder / f"dollars_saved_per_dalys_lost_full_3percent_{target_period()}.xlsx")
summary_dalys_3percent.to_excel(results_folder / f"dollars_saved_per_dalys_lost_3percent_{target_period()}.xlsx")



# Format for export
export_table = format_dalys_usd_table(summary_dalys_3percent)
export_table.to_excel(
    results_folder / f"dollars_saved_per_dalys_lost_summary_3percent_{target_period()}.xlsx",
    index=False
)


# deaths per $ saved
ratio_deaths_3percent, summary_deaths_3percent = outcome_per_usd_saved(
    aids_dalys_diff=hiv_deaths_diff,
    cost_diff=total_cost_diff_3percent,
    per_million=False
)
ratio_deaths_3percent.to_excel(results_folder / f"dollars_saved_per_death_full_3percent_{target_period()}.xlsx")
summary_deaths_3percent.to_excel(results_folder / f"dollars_saved_per_death_3percent_{target_period()}.xlsx")


# Format for export
export_table = format_dalys_usd_table(summary_deaths_3percent)
export_table.to_excel(
    results_folder / f"dollars_saved_per_death_summary_3percent_{target_period()}.xlsx",
    index=False
)



# 5% discount
ratio, summary = outcome_per_usd_saved(
    aids_dalys_diff=aids_dalys_diff,
    cost_diff=total_cost_diff_5percent,
    per_million=False
)
ratio.to_excel(results_folder / f"dollars_saved_per_dalys_lost_full_5percent_{target_period()}.xlsx")
summary.to_excel(results_folder / f"dollars_saved_per_dalys_lost_5percent_{target_period()}.xlsx")


# Format for export
export_table = format_dalys_usd_table(summary)
export_table.to_excel(
    results_folder / f"dollars_saved_per_dalys_lost_summary_5percent_{target_period()}.xlsx",
    index=False
)


# deaths per $ saved
ratio, summary = outcome_per_usd_saved(
    aids_dalys_diff=hiv_deaths_diff,
    cost_diff=total_cost_diff_5percent,
    per_million=False
)
ratio.to_excel(results_folder / f"dollars_saved_per_death_full_5percent_{target_period()}.xlsx")
summary.to_excel(results_folder / f"dollars_saved_per_death_5percent_{target_period()}.xlsx")


# Format for export
export_table = format_dalys_usd_table(summary)
export_table.to_excel(
    results_folder / f"dollars_saved_per_death_summary_5percent_{target_period()}.xlsx",
    index=False
)
