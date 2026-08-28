"""
All post-processing of a single completed TLO simulation run lives here,
separate from ask_tell_azure_example.py's submission/polling/ask-tell
orchestration. Import postprocess_run() from this file wherever a
downloaded run's outputs need turning into the four numbers SMAC/
ConstrainedEI need (dalys, cost, hr_used, stock_used).

Keeping this separate means the postprocessing pipeline can be edited,
tested, or run standalone against a downloaded output directory without
touching (or re-reading) the orchestration logic at all.

DALYs and cost are HIV-SPECIFIC, not all-cause/all-programme, both
adapted from src/scripts/hiv/program_simplification/process_outputs.py
(the tara_hiv_program_simplification branch) - see _get_hiv_dalys() and
_get_hiv_cost() below for exactly what that means and where each was
adapted from.
"""

import ast
from collections import Counter
from pathlib import Path

import pandas as pd

from tlo.analysis.utils import load_pickled_dataframes
from scripts.costing.cost_estimation import load_unit_cost_assumptions


# TARGET_PERIOD: adjust to match the evaluation window your simulation
# actually covers - this determines which years' DALYs/costs get summed.
TARGET_PERIOD = (pd.Timestamp(2010, 1, 1), pd.Timestamp(2011, 1, 1))


def postprocess_run(run_dir: Path) -> dict:
    """
    Turns a single completed run's raw output directory into the four
    numbers SMAC/ConstrainedEI need. dalys/cost adapted to be
    HIV-SPECIFIC (not all-cause/all-programme) - cost is staff time
    (_get_hiv_hrh_cost) PLUS consumables (_get_hiv_consumables_cost),
    combined in _get_hiv_cost(). Uses
    tlo.analysis.utils.load_pickled_dataframes rather than reading raw
    pickle files directly - this is the officially supported way to
    load a single run's log.

    run_dir is expected to be .../<job_id>/0/<run_number> - i.e. what
    ask_tell_azure_example.py's download_run_outputs() returns entries
    of when iterating draw_dir.
    """
    job_root = run_dir.parent.parent   # .../job_id/0/<run_number> -> .../job_id
    run_number = int(run_dir.name)
    log = load_pickled_dataframes(job_root, draw=0, run=run_number)

    dalys = _get_hiv_dalys(log)
    cost = _get_hiv_cost(log)
    hr_used = 0.1  # _get_hr_capacity_used(log) - still a stub, unrelated to this change
    stock_used = 0.1  # _get_consumables_used(log) - still a stub, unrelated to this change

    return {"dalys": dalys, "cost": cost, "hr_used": hr_used, "stock_used": stock_used}


def _get_hiv_dalys(log: dict) -> float:
    """
    HIV/AIDS-specific DALYs (stacked), summed over TARGET_PERIOD, from
    one run's log. Adapted from num_dalys_by_cause() in
    process_outputs.py: same module/key/year-completeness guard as
    _get_total_dalys() below, but does NOT collapse across cause columns
    before summing - instead selects the 'AIDS' cause specifically.

    'AIDS' is the cause label TLOmodel's healthburden module uses for
    HIV-related DALYs - confirmed by the {'AIDS': 'HIV/AIDS', '': 'Other'}
    mapping in get_total_num_dalys_by_label() in the same reference
    script, which lumps every OTHER cause into 'Other', implying 'AIDS'
    is the sole HIV-specific cause label in this model's DALY breakdown.
    """
    df = log["tlo.methods.healthburden"]["dalys_stacked"]
    years_needed = [d.year for d in TARGET_PERIOD]
    assert set(df.year.unique()).issuperset(years_needed), \
        "Some years are not recorded - run may have crashed early."
    by_cause = (
        df.loc[df.year.between(*years_needed)]
          .drop(columns=['date', 'sex', 'age_range', 'year'])
          .sum()
    )
    return float(by_cause.get('AIDS', 0.0))


def _get_total_dalys(log: dict) -> float:
    """
    ALL-CAUSE DALYs (stacked), summed over TARGET_PERIOD - kept for
    reference/diagnostics. postprocess_run() uses _get_hiv_dalys() above
    instead, since the optimisation target is HIV-specific DALYs, not
    every cause the simulation tracks.
    """
    df = log["tlo.methods.healthburden"]["dalys_stacked"]
    years_needed = [d.year for d in TARGET_PERIOD]
    assert set(df.year.unique()).issuperset(years_needed), \
        "Some years are not recorded - run may have crashed early."
    return float(
        df.loc[df.year.between(*years_needed)]
          .drop(columns=['date', 'sex', 'age_range', 'year'])
          .sum().sum()
    )


# --------------------------------------------------------------------------
# HIV-specific cost: staff (HCW) TIME x hourly wage, adapted from
# process_outputs.py's own pipeline (summarise_appointments ->
# keep_selected_appt_types -> add_mapped_treatments_as_appt_types ->
# appt_counts_to_hcw_minutes -> hrh_time_costs_by_facility). See
# _get_hiv_cost()'s docstring for the important caveat on what this does
# and doesn't capture.
#
# Note process_outputs.py imports estimate_input_cost_of_scenarios from
# scripts.costing.cost_estimation, but never actually CALLS it anywhere
# in the file - the functions adapted below are the real, working
# HIV-cost implementation there, not that import.
# --------------------------------------------------------------------------

def _parse_dict_cell(x) -> dict:
    """
    Cells in HSI/appt-footprint logs are stored as dicts when read
    straight from a live log, but sometimes come back as their string
    repr after a pickle round-trip - normalise both to a plain dict.
    Identical to _parse_dict_cell() in process_outputs.py.
    """
    if isinstance(x, dict):
        return x
    if pd.isna(x):
        return {}
    if isinstance(x, str):
        return ast.literal_eval(x)
    return {}


# Explicit allow-list of appointment types to KEEP - copied verbatim
# from process_outputs.py's KEEP_APPT_TYPES. This is the authoritative
# list; do not add/remove entries without checking that file first.
KEEP_APPT_TYPES = [
    "VCTPositive",
    "VCTNegative",
    "NewAdult",
    "Peds",
    "EstNonCom",
    "MaleCirc",
]

# Maps a TREATMENT_ID onto a synthetic appointment-type row (appt_type,
# facility_level, multiplier), copied verbatim from process_outputs.py's
# TREATMENT_TO_APPT_SPEC. These TREATMENT_IDs (PrEP dispensing, self-test
# counselling, palliative/inpatient care) use generic appt-type codes
# ALSO used by non-HIV treatments, so KEEP_APPT_TYPES filtering alone
# can't isolate their HIV-attributable share - this injects that share
# back in via the source TREATMENT_ID instead, which IS HIV-specific.
TREATMENT_TO_APPT_SPEC = {
    "PharmDispensing": ("Hiv_Prevention_Prep", "1a", 1.0),
    "ConWithDCSA": ("Hiv_Test_Selftest", "0", 1.0),
    "IPAdmission": ("Hiv_PalliativeCare", "2", 2.0),
    "InpatientDays": ("Hiv_PalliativeCare", "2", 17.0),
}

# Static reference data, loaded ONCE at import time (not per-run, since
# it never changes between runs) - same convention as TARGET_PERIOD
# above. Paths are relative to the TLOmodel repo root, same assumption
# as the rest of this pipeline.
_HCW_TIME_TABLE = pd.read_csv(
    "resources/healthsystem/human_resources/definitions/ResourceFile_Appt_Time_Table.csv"
)
_HCW_TIME_TABLE["Facility_Level"] = _HCW_TIME_TABLE["Facility_Level"].astype(str).str.strip()
_HCW_TIME_TABLE["Appt_Type_Code"] = _HCW_TIME_TABLE["Appt_Type_Code"].astype(str).str.strip()
_HCW_TIME_TABLE["Officer_Category"] = _HCW_TIME_TABLE["Officer_Category"].astype(str).str.strip()

# Facility-level-SPECIFIC minutes-per-appointment map, indexed by
# (Facility_Level, Appt_Type_Code) - NOT a blanket single-facility-level
# assumption. Matches appt_counts_to_hcw_minutes()'s map_table exactly.
_APPT_MINUTES_BY_FACILITY_MAP = _HCW_TIME_TABLE.pivot_table(
    index=["Facility_Level", "Appt_Type_Code"], columns="Officer_Category",
    values="Time_Taken_Mins", aggfunc="mean",
)

_HCW_HOURLY_COSTS = pd.read_csv("resources/ResourceFile_HIV/hrh_costs.csv")
_HCW_HOURLY_COSTS["Facility_Level"] = _HCW_HOURLY_COSTS["Facility_Level"].astype(str).str.strip()
_HCW_HOURLY_COSTS["Officer_Category"] = _HCW_HOURLY_COSTS["Officer_Category"].astype(str).str.strip()

# Hourly wage lookup by (Facility_Level, Officer_Category) - matches
# hrh_time_costs_by_facility()'s cost_lookup exactly (averaging any
# duplicate rows for the same facility_level/cadre pair).
_HCW_HOURLY_COST_LOOKUP = (
    _HCW_HOURLY_COSTS.groupby(["Facility_Level", "Officer_Category"])["Total_hourly_cost"].mean()
)


def _get_hiv_treatment_counts_by_year(log: dict) -> dict:
    """
    HIV-prefixed TREATMENT_ID counts, summed over TARGET_PERIOD, from
    one run's log. Adapted from make_series_treatment_counts_by_year()
    in process_outputs.py, filtered the same way treatment_by_year_hiv
    is built there (treatment_id starting with "Hiv"). Feeds
    TREATMENT_TO_APPT_SPEC's synthetic-row injection below.
    """
    df = log["tlo.methods.healthsystem.summary"]["HSI_Event"]
    years_needed = [d.year for d in TARGET_PERIOD]
    df = df.loc[pd.to_datetime(df["date"]).dt.year.isin(years_needed)]

    counts: Counter = Counter()
    for cell in df["TREATMENT_ID"]:
        for treatment_id, count in _parse_dict_cell(cell).items():
            if treatment_id.startswith("Hiv"):
                counts[treatment_id] += count
    return dict(counts)


def _get_hiv_appt_counts_by_facility(log: dict) -> dict:
    """
    HIV-relevant appointment counts by (facility_level, appt_type),
    summed over TARGET_PERIOD, from one run's log. Adapted from
    process_outputs.py's summarise_appointments() ->
    keep_selected_appt_types() -> add_mapped_treatments_as_appt_types()
    pipeline, adapted to operate on a single run's raw log directly
    rather than via extract_results() across many draws/runs.
    """
    df = log["tlo.methods.healthsystem.summary"]["HSI_Event_non_blank_appt_footprint"]
    years_needed = [d.year for d in TARGET_PERIOD]
    df = df.loc[pd.to_datetime(df["date"]).dt.year.isin(years_needed)]

    counts: Counter = Counter()
    for cell in df["Number_By_Appt_Type_Code_And_Level"]:
        nested = _parse_dict_cell(cell)  # {facility_level: {appt_type: count}}
        if not isinstance(nested, dict):
            continue
        for facility_level, inner in nested.items():
            if isinstance(inner, dict):
                for appt_type, count in inner.items():
                    if appt_type in KEEP_APPT_TYPES:
                        counts[(str(facility_level), appt_type)] += count

    # inject treatment-derived synthetic appointment rows - see
    # TREATMENT_TO_APPT_SPEC's comment above for why this is needed
    hiv_treatment_counts = _get_hiv_treatment_counts_by_year(log)
    for appt_type, (trt_id, facility_level, mult) in TREATMENT_TO_APPT_SPEC.items():
        if trt_id in hiv_treatment_counts:
            key = (str(facility_level), appt_type)
            counts[key] = counts.get(key, 0.0) + hiv_treatment_counts[trt_id] * mult

    return dict(counts)


def _get_hiv_appt_minutes_by_facility_and_cadre(log: dict) -> dict:
    """
    HIV-specific appointment MINUTES by (facility_level, cadre), summed
    over TARGET_PERIOD, from one run's log. Adapted from
    appt_counts_to_hcw_minutes() in process_outputs.py - uses the
    facility-level-SPECIFIC (Facility_Level, Appt_Type_Code) -> minutes
    mapping, not a blanket single-facility-level assumption.
    """
    appt_counts = _get_hiv_appt_counts_by_facility(log)

    minutes: Counter = Counter()
    for (facility_level, appt_type), count in appt_counts.items():
        key = (facility_level, appt_type)
        if key not in _APPT_MINUTES_BY_FACILITY_MAP.index:
            continue
        row = _APPT_MINUTES_BY_FACILITY_MAP.loc[key]
        for cadre, mins_per_appt in row.items():
            if pd.notna(mins_per_appt):
                minutes[(facility_level, cadre)] += count * mins_per_appt
    return dict(minutes)


def _get_hiv_hrh_cost(log: dict) -> float:
    """
    HIV-specific health-worker COST (staff time x hourly wage), summed
    over TARGET_PERIOD, from one run's log. Adapted from
    hrh_time_costs_by_facility() in process_outputs.py: hours by
    (facility_level, cadre) x the facility/cadre-specific hourly wage
    from hrh_costs.csv, summed to one total.

    Staff time ONLY - see _get_hiv_cost() below for the combined figure
    that also includes consumables, matching process_outputs.py's
    cons_hrh_costs_year_draw_run construction.
    """
    minutes_by_fac_cadre = _get_hiv_appt_minutes_by_facility_and_cadre(log)

    total_cost = 0.0
    for (facility_level, cadre), minutes in minutes_by_fac_cadre.items():
        hourly_cost = _HCW_HOURLY_COST_LOOKUP.get((facility_level, cadre))
        if hourly_cost is None or pd.isna(hourly_cost):
            continue
        total_cost += (minutes / 60.0) * hourly_cost
    return float(total_cost)


# --------------------------------------------------------------------------
# HIV-specific consumables cost, adapted from process_outputs.py's
# get_counts_of_items_requested() -> apply_unit_costs() pipeline, plus
# the two flat-rate additions it applies OUTSIDE that pipeline (self-tests
# and TDF urine tests - see cons_costed_hiv's construction there:
# item-code-filtered consumables + selftests_costs + tdf_costs).
# --------------------------------------------------------------------------

# HIV-specific consumable item codes, hardcoded from the resolved values
# shown directly in process_outputs.py's own comment block - originally
# resolved there via get_item_code_from_item_name()/
# get_item_codes_from_package_name() against
# ResourceFile_Consumables_Items_and_Packages.csv. Hardcoding the
# already-resolved codes here avoids an extra CSV load + name-lookup
# per run for values that never change.
HIV_ITEM_CODES = {
    196: "HIV test",
    190: "Viral load",
    197: "VMMC",
    1191: "Adult PrEP",
    198: "Infant PrEP",
    2671: "First-line ART regimen: adult",
    204: "First-line ART regimen: adult: cotrimoxazole",
    2672: "First line ART regimen: older child",
    2673: "First line ART regimen: young child",
    202: "First line ART regimen: young child: cotrimoxazole",
}
_HIV_ITEM_CODE_STRS = set(map(str, HIV_ITEM_CODES.keys()))

# Per-unit consumable costs, loaded ONCE via TLOmodel's own costing
# module. Unlike estimate_input_cost_of_scenarios (imported but never
# called in process_outputs.py), load_unit_cost_assumptions IS actually
# called there - a genuine, demonstrated adaptation, not a guess at an
# unused function's API.
_UNIT_COST_ASSUMPTIONS = load_unit_cost_assumptions(Path("resources"))
_CONS_UNIT_COST_BY_ITEM_CODE = dict(zip(
    _UNIT_COST_ASSUMPTIONS["consumables"]["Item_Code"],
    _UNIT_COST_ASSUMPTIONS["consumables"]["Price_per_unit"],
))

# Flat per-test costs process_outputs.py applies OUTSIDE the normal
# item-code costing pipeline, for tests not tracked as a standard
# consumable dispensation. Kept as named constants (rather than magic
# numbers) so their provenance stays visible at the call site:
# self-test cost is a MIHPSA costing assumption; TDF test cost is a
# South Africa cost estimate (see the source citation left in
# process_outputs.py next to tdf_costs).
SELFTEST_UNIT_COST = 3.14
TDF_TEST_UNIT_COST = 6.86


def _get_hiv_item_counts(log: dict) -> dict:
    """
    HIV-relevant consumable item counts, summed over TARGET_PERIOD, from
    one run's log. Adapted from get_counts_of_items_requested() in
    process_outputs.py, filtered down to HIV_ITEM_CODES.
    """
    df = log["tlo.methods.healthsystem.summary"]["Consumables"]
    years_needed = [d.year for d in TARGET_PERIOD]
    df = df.loc[pd.to_datetime(df["date"]).dt.year.isin(years_needed)]

    counts: Counter = Counter()
    for cell in df["Item_Used"]:
        for item_code, count in _parse_dict_cell(cell).items():
            if str(item_code) in _HIV_ITEM_CODE_STRS:
                counts[str(item_code)] += count
    return dict(counts)


def _get_num_tdf_tests(log: dict) -> float:
    """
    Total TDF urine tests performed over TARGET_PERIOD, from one run's
    log. Adapted from get_num_tdf() in process_outputs.py.
    """
    df = log["tlo.methods.hiv"]["hiv_program_coverage"]
    years_needed = [d.year for d in TARGET_PERIOD]
    df = df.loc[pd.to_datetime(df["date"]).dt.year.isin(years_needed)]
    return float(df["n_tdf_tests_performed"].sum())


def _get_hiv_consumables_cost(log: dict) -> float:
    """
    HIV-specific CONSUMABLES cost, summed over TARGET_PERIOD, from one
    run's log: item-code-filtered consumables (via unit costs) PLUS
    self-test and TDF-test flat-rate costs. Matches process_outputs.py's
    cons_costed_hiv construction exactly.
    """
    item_counts = _get_hiv_item_counts(log)
    item_cost = sum(
        count * _CONS_UNIT_COST_BY_ITEM_CODE.get(int(item_code), 0.0)
        for item_code, count in item_counts.items()
    )

    hiv_treatment_counts = _get_hiv_treatment_counts_by_year(log)
    selftest_cost = hiv_treatment_counts.get("Hiv_Test_Selftest", 0.0) * SELFTEST_UNIT_COST

    tdf_cost = _get_num_tdf_tests(log) * TDF_TEST_UNIT_COST

    return float(item_cost + selftest_cost + tdf_cost)


def _get_hiv_cost(log: dict) -> float:
    """
    TOTAL HIV-specific cost, summed over TARGET_PERIOD, from one run's
    log: staff time (_get_hiv_hrh_cost) PLUS consumables
    (_get_hiv_consumables_cost). Matches process_outputs.py's
    cons_hrh_costs_year_draw_run construction exactly
    (cons_summed_by_year + hrh_costs_by_year). This is what
    postprocess_run() actually calls for "cost".
    """
    return _get_hiv_hrh_cost(log) + _get_hiv_consumables_cost(log)


def _get_total_cost(log: dict) -> float:
    """
    NOT YET ADAPTED - a fully comprehensive input cost (overheads,
    equipment, programme management, etc. on top of what _get_hiv_cost()
    already covers - staff time + consumables), via TLOmodel's official
    costing module (scripts.costing.cost_estimation:
    estimate_input_cost_of_scenarios, summarize_cost_data, etc.), which
    is never actually called in process_outputs.py despite being
    imported there - so there's no working reference call to adapt from
    in that file. Worth checking scripts/costing/cost_estimation.py
    directly if overhead/equipment costs need to be captured too;
    _get_hiv_cost() (staff time + consumables) is likely sufficient for
    most purposes and is what postprocess_run() actually uses.
    """
    raise NotImplementedError("Adapt from scripts/costing/cost_estimation.py - see docstring")


def _get_hr_capacity_used(log: dict) -> float:
    """
    NOT YET ADAPTED - no HR-capacity extraction appeared in the portion
    of process_outputs.py available to adapt from. Likely lives in
    log["tlo.methods.healthsystem"] under some capability-usage key -
    worth checking that module's logged keys directly (e.g. via
    log["tlo.methods.healthsystem"].keys()) to find the right one.
    """
    raise NotImplementedError("Find the right key under log['tlo.methods.healthsystem']")


def _get_consumables_used(log: dict) -> float:
    """
    NOT YET ADAPTED - same situation as _get_hr_capacity_used: no
    consumables-usage extraction appeared in the portion of
    process_outputs.py available to adapt from. Likely also under
    log["tlo.methods.healthsystem"], possibly a consumables-specific key.
    """
    raise NotImplementedError("Find the right key under log['tlo.methods.healthsystem']")


if __name__ == "__main__":
    # Quick standalone sanity check: point this at an already-downloaded
    # run directory and confirm postprocess_run() works without needing
    # to run the full ask-tell pipeline.
    import sys
    if len(sys.argv) != 2:
        print("Usage: python postprocess_output.py <path-to-run-dir>")
        sys.exit(1)
    result = postprocess_run(Path(sys.argv[1]))
    print(result)
