"""Extract LCOA inputs and sanity-check calculations into one results pickle."""

import argparse
import pickle
from datetime import date
from pathlib import Path

import pandas as pd

from scripts.costing.cost_estimation import apply_discounting_to_cost_data, estimate_input_cost_of_scenarios
from scripts.lcoa_inputs_from_tlo_analyses.results_processing_utils import (
    find_difference_extra_relative_to_comparison,
    find_difference_relative_to_comparison,
    format_scenario_name,
    get_capacity_used_by_officer_type_and_facility_level,
    get_counts_of_hsi_by_short_treatment_id,
    get_parameter_names_from_scenario_file,
    get_total_population_by_year,
    make_get_counts_of_hsis_by_period,
    make_get_num_dalys_by_cause_label_and_period,
    make_get_num_deaths_by_cause_label_and_period,
    set_param_names_as_column_index_level_0,
)
from tlo import Date
from tlo.analysis.utils import compute_summary_statistics, extract_results
# python src/scripts/lcoa_inputs_from_tlo_analyses/analysis_effect_of_treatment_ids.py outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-10-15T115126Z outputs/generated_outputs --target-start=2026-01-01 --target-end=2040-12-31
# Do not exclude FirstAttendance_SpuriousEmergencyCare as it has non-blank
# appointment footprint
EXCLUDED_HSIs = [
    "FirstAttendance_Emergency",
    "FirstAttendance_NonEmergency",
    "Inpatient_Care"
]
SUSPENDED_RESULTS_FOLDER = Path("outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-10-09T145410Z")
YEAR_OF_SWITCH = 2026
COST_END_YEAR = 2040
DISCOUNT_RATE = 0.03

def parse_iso_date(value: str) -> Date:
    parsed = date.fromisoformat(value)
    return Date(parsed.year, parsed.month, parsed.day)

def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{value}'. Use True or False."
    )

# source: http://doi.org/https://doi.org/10.1016/j.vhri.2023.10.007
MALAWI_CET = 65.8

def analysis_metadata(target_period_tuple):
    return {
        "target_period": tuple(date.isoformat() for date in target_period_tuple),
        "discount_rate": DISCOUNT_RATE,
        "cost_years": (YEAR_OF_SWITCH, COST_END_YEAR),
        "cost_effectiveness_threshold": MALAWI_CET,
    }


def match_baseline(scenarios: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Broadcast the Nothing draw to scenario columns by original run ID."""
    matched = baseline.reindex(index=scenarios.index, columns=scenarios.columns.get_level_values("run"))
    return matched.set_axis(scenarios.columns, axis=1)


def process_scenarios(
    results_folder: Path,
    resourcefilepath: Path,
    target_period_tuple: tuple[Date, Date],
    param_names: tuple[str, ...],
    do_comparison: bool = True,
    *,
    nothing_by_run: dict | None = None,
    runs: set[int] | None = None,
    suspended_results_folder: Path = SUSPENDED_RESULTS_FOLDER,
):
    """Extract and summarise scenario metrics using an internal or exported baseline."""
    run_ids = sorted(runs) if runs is not None else None

    def extract(**kwargs):
        data = extract_results(
            results_folder, suspended_results_folder=suspended_results_folder, run_ids=run_ids, **kwargs
        )
        data = set_param_names_as_column_index_level_0(data, param_names)
        if runs is not None:
            expected = pd.MultiIndex.from_product(
                ([format_scenario_name(name) for name in param_names], run_ids), names=["draw", "run"]
            )
            missing = expected.difference(data.columns)
            if len(missing):
                raise ValueError(f"Missing {kwargs['key']} data for draw/run pairs {missing.tolist()}.")
            data = data.reindex(columns=expected)
        return data

    def with_nothing(data, key, *, fill_value=None):
        if nothing_by_run is None:
            return data
        baseline = nothing_by_run[key].loc[:, run_ids]
        combined = pd.concat([data, pd.concat({"Nothing": baseline}, axis=1, names=["draw"])], axis=1)
        return combined.fillna(fill_value) if fill_value is not None else combined

    # Extract districts and facility levels from the Master Facility List
    mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
    facility_id_levels_dict = dict(zip(mfl['Facility_ID'], mfl['Facility_Level']))

    get_num_dalys_by_cause_label_and_period = (
        make_get_num_dalys_by_cause_label_and_period(target_period_tuple,)
    )
    get_num_deaths_by_cause_label_and_period = make_get_num_deaths_by_cause_label_and_period(
        target_period_tuple,
    )
    get_num_hsi_by_period = make_get_counts_of_hsis_by_period(target_period_tuple,)

    results = {}

    # Get total population by year
    print("Extracting population data...")
    total_population_by_year = (
        extract(
            module='tlo.methods.demography',
            key='population',
            custom_generate_series=lambda _df: get_total_population_by_year(_df, target_period_tuple),
            do_scaling=True,
        )
    )

    cols_to_drop = [
        col for col in total_population_by_year.columns
        if col[0].strip().startswith(tuple(EXCLUDED_HSIs))
    ]

    print(f"Dropping columns from total_population_by_year: {cols_to_drop}")
    total_population_by_year.drop(columns=cols_to_drop, inplace=True)

    total_population_by_year = with_nothing(total_population_by_year, "total_population_by_year")
    results['total_population_by_year'] = (
        compute_summary_statistics(total_population_by_year, central_measure='median')
    )

    print("Extracting HSI event counts...")
    counts_of_hsi_by_short_treatment_id = (
        extract(
            module="tlo.methods.healthsystem.summary",
            key="HSI_Event",
            custom_generate_series=lambda _df: get_counts_of_hsi_by_short_treatment_id(_df, target_period_tuple),
            do_scaling=True,
        )
        .fillna(0.0)
        .sort_index()
    ).drop(EXCLUDED_HSIs, errors='ignore')

    counts_of_hsi_by_short_treatment_id = with_nothing(
        counts_of_hsi_by_short_treatment_id, "counts_of_hsi_by_short_treatment_id", fill_value=0.0
    )
    results['counts_of_hsi_by_short_treatment_id'] = (
        compute_summary_statistics(counts_of_hsi_by_short_treatment_id, 'median')
    )
    counts_of_hsi_by_period = (
        extract(
            module="tlo.methods.healthsystem.summary",
            key="HSI_Event",
            custom_generate_series=lambda _df: get_num_hsi_by_period(_df),
            do_scaling=True,
        )
        .fillna(0.0)
        .sort_index()
    ).drop(columns=cols_to_drop)

    counts_of_hsi_by_period = with_nothing(counts_of_hsi_by_period, "counts_of_hsi_by_period", fill_value=0.0)
    results['counts_of_hsi_by_period'] = (
        compute_summary_statistics(counts_of_hsi_by_period, 'median')
    )

    print("Extracting DALYs by label...")
    dalys = (
        extract(
            module="tlo.methods.healthburden",
            key="dalys_stacked_by_age_and_time",
            custom_generate_series=get_num_dalys_by_cause_label_and_period,
            do_scaling=True,
        )
    )

    dalys = dalys.drop(columns=cols_to_drop)

    # Because of Issue 1877 (fixed in PR 1880, not merged in feature branch),
    # the simulations for the Nothing scenario and scenarios with zero HSIs are
    # not identical, as they should be. Consequently the costs and dalys are
    # both diffrent across. As a temporary fix, we will use the costs and dalys
    # from the Nothing scenario for zero-HSIs scenarios. We expect that this
    # will not be needed when we have more runs, thereby achieving more stable
    # results.

    # Note that Appointments scheduled before the 2026 service switch can carry over into
    # 2026 specifically CardioMetabolicDisorders_Prevention_WeightLoss and
    # CervicalCancer_PalliativeCare but because these are triggered by other
    # treatment ids (not available from 2026), we will see 0 HSIs from 2027
    # onwards.
    zero_hsi_scenarios = []
    if nothing_by_run is None:
        implementation_counts = counts_of_hsi_by_period.loc[
            (counts_of_hsi_by_period.index.get_level_values("period").astype(int) >= YEAR_OF_SWITCH)
            & ~counts_of_hsi_by_period.index.get_level_values("appt_type").str.startswith(
                ("FirstAttendance_", "Inpatient_Care")
            )
        ]

        zero_hsi_by_run = implementation_counts.eq(0).all()
        zero_hsi_scenarios = zero_hsi_by_run.groupby(level="draw").all()
        zero_hsi_scenarios = zero_hsi_scenarios.index[zero_hsi_scenarios].drop("Nothing", errors="ignore")
        print(f"Zero HSI scenarios {zero_hsi_scenarios}")
    if len(zero_hsi_scenarios):
        baseline_dalys = dalys.xs("Nothing", axis=1, level="draw")
        for draw in zero_hsi_scenarios:
            columns = dalys.columns[dalys.columns.get_level_values("draw") == draw]
            dalys.loc[:, columns] = baseline_dalys.loc[
                :, columns.get_level_values("run")
            ].to_numpy()

    if do_comparison and "Nothing" in dalys.columns.get_level_values("draw"):
        post_switch = dalys.index.get_level_values("period").astype(int) >= YEAR_OF_SWITCH
        dalys_by_cause = dalys.loc[post_switch].groupby(level="label").sum(min_count=1)
        dalys_by_cause.loc["All causes"] = dalys_by_cause.sum(min_count=1)
        nothing = dalys_by_cause.xs("Nothing", axis=1, level="draw")
        scenarios = dalys_by_cause.drop(columns="Nothing", level="draw")
        matched_nothing = nothing.reindex(columns=scenarios.columns.get_level_values("run"))
        matched_nothing.columns = scenarios.columns
        results["dalys_averted_by_cause_by_run"] = matched_nothing - scenarios
        results["dalys_ratio_by_cause_by_run"] = scenarios.div(matched_nothing.where(matched_nothing.ne(0)))

    print("Extracting deaths by cause and period...")
    num_deaths = (
        extract(
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=get_num_deaths_by_cause_label_and_period,
            do_scaling=True,
        )
    )

    num_deaths = num_deaths.drop(columns=cols_to_drop, errors='ignore')
    num_deaths = with_nothing(num_deaths, "num_deaths", fill_value=0.0)

    if do_comparison:
        num_deaths_averted = compute_summary_statistics(
            -1.0 * pd.DataFrame(
                find_difference_extra_relative_to_comparison(
                    num_deaths.sum(), comparison='Nothing')).T,
            central_measure='median'
        ).iloc[0].unstack()

        pc_deaths_averted = 100.0 * compute_summary_statistics(
            -1.0 * pd.DataFrame(
                find_difference_extra_relative_to_comparison(
                    num_deaths.sum(), comparison='Nothing', scaled=True)).T,
            central_measure='median'
        ).iloc[0].unstack()
    else:
        num_deaths_averted = None
        pc_deaths_averted = None

    results['num_deaths'] = compute_summary_statistics(num_deaths, central_measure='median')
    results['num_deaths_averted'] = num_deaths_averted
    results['pc_deaths_averted'] = pc_deaths_averted

    discount_rate_dalys = DISCOUNT_RATE
    dalys_years = dalys.index.get_level_values("period").astype(int)
    discount_base_year = (
        dalys_years.min() if nothing_by_run is None else nothing_by_run["daly_discount_base_year"]
    )
    discounted_dalys = dalys.div(
        (1.0 + discount_rate_dalys) ** (dalys_years - discount_base_year),
        axis=0,
    )

    discounted_dalys = with_nothing(discounted_dalys, "discounted_dalys")
    if nothing_by_run is not None:
        by_cause = discounted_dalys.groupby(level="label").sum(min_count=1)
        scenario_causes = by_cause.drop(columns="Nothing", level="draw")
        baseline_causes = by_cause.xs("Nothing", axis=1, level="draw")
        matched_causes = match_baseline(scenario_causes, baseline_causes)
        cause_differences = matched_causes - scenario_causes
        results["dalys_averted_by_cause"] = compute_summary_statistics(cause_differences, "median")
        cause_ratios = scenario_causes.div(matched_causes.where(matched_causes.ne(0)))
        cause_differences.loc["All causes"] = cause_differences.sum(min_count=1)
        baseline_totals = matched_causes.sum(min_count=1)
        cause_ratios.loc["All causes"] = scenario_causes.sum(min_count=1).div(
            baseline_totals.where(baseline_totals.ne(0))
        )
        results["dalys_averted_by_cause_by_run"] = cause_differences
        results["dalys_ratio_by_cause_by_run"] = cause_ratios

    results['discounted_dalys'] = (
        compute_summary_statistics(discounted_dalys, central_measure='median'))

    # Costs calculation
    print("Calculating costs...")
    discount_rate_cost = DISCOUNT_RATE
    # Period relevant for costing
    TARGET_PERIOD = (Date(YEAR_OF_SWITCH, 1, 1), Date(COST_END_YEAR, 12, 31))
    relevant_period_for_costing = [i.year for i in TARGET_PERIOD]
    list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0], relevant_period_for_costing[1] + 1))
    print("List of relevant years for costing:", list_of_relevant_years_for_costing)
    input_costs = estimate_input_cost_of_scenarios(
        results_folder,
        resourcefilepath,
        suspended_results_folder=suspended_results_folder,
        _runs=run_ids,
        _years=list_of_relevant_years_for_costing,
        cost_only_used_staff=True,
        _discount_rate=discount_rate_cost,
        _metric="median",)

    # Map draw numbers to draw names in input costs
    draw_lookup = dict(enumerate(param_names))
    formatted_draw_lookup = {
        draw: format_scenario_name(name)
        for draw, name in draw_lookup.items()
    }
    input_costs.loc[:, "draw_name"] = input_costs["draw"].map(formatted_draw_lookup)
    input_costs.drop(columns=["draw"], inplace=True)
    input_costs.rename(columns={"draw_name": "draw"}, inplace=True)
    results['input_costs'] = input_costs
    if nothing_by_run is not None:
        results["input_costs"] = pd.concat(
            [input_costs, nothing_by_run["input_costs"].loc[lambda data: data["run"].isin(runs)]],
            ignore_index=True,
        )

    # Capacity_By_FacID_and_Officer logs the fraction of time used per officer
    # type.
    # To get the actual minutes, we multiply by the total available minutes.
    print("Extracting capacity fractions with clinic and facility identifiers...")
    # Set do_scaling to False as this is a fraction calculated during runtime.
    avg_frac_capacity_used_by_clinic_and_facility = extract(
        module='tlo.methods.healthsystem.summary',
        key='Capacity_By_FacID_and_Officer',
        custom_generate_series=lambda df: get_capacity_used_by_officer_type_and_facility_level(df, facility_id_levels_dict),
        do_scaling=False,
    ).drop(columns=cols_to_drop)

    avg_frac_capacity_used_by_clinic_and_facility = with_nothing(
        avg_frac_capacity_used_by_clinic_and_facility, "capacity_fractions"
    )

    # Convert each facility's fraction using its own capacity before summing.
    daily_capacity_by_cadre_and_level = pd.read_csv(
        resourcefilepath / "healthsystem" / "human_resources" / "actual" / "ResourceFile_Daily_Capabilities.csv"
    )
    annual_capacity_by_facility = (
        daily_capacity_by_cadre_and_level.groupby(['Facility_ID', 'Officer_Category'])['Total_Mins_Per_Day'].sum()
        * 365
    )
    annual_capacity_by_facility.index.names = ['FacilityID', 'OfficerType']
    facility_capacity = annual_capacity_by_facility.reindex(
        avg_frac_capacity_used_by_clinic_and_facility.index.droplevel(['year', 'FacilityLevel'])
    )
    if facility_capacity.isna().any():
        raise ValueError(f"Missing capacity for facility/cadre pairs: {list(facility_capacity[facility_capacity.isna()].index)}")
    facility_capacity.index = avg_frac_capacity_used_by_clinic_and_facility.index
    actual_capacity_used_by_clinic_and_facility = avg_frac_capacity_used_by_clinic_and_facility.mul(
        facility_capacity, axis=0
    )
    actual_capacity_used_by_cadre_and_level = actual_capacity_used_by_clinic_and_facility.groupby(
        level=['year', 'OfficerType', 'FacilityLevel']
    ).sum()
    print("Converted facility fractions to minutes and aggregated by year, cadre and level.")

    # Use the Nothing baseline for capacity scaling.
    baseline_fractions_2025 = avg_frac_capacity_used_by_clinic_and_facility.xs(
        'Nothing', level='draw', axis=1
    ).xs(2025, level='year')

    results['actual_annual_capacity_by_cadre'] = compute_summary_statistics(actual_capacity_used_by_cadre_and_level, 'median')

    # Scale each facility/cadre directly using its logged 2025 baseline fraction.
    # Retain runs until after aggregating facilities, since medians are not additive.
    baseline_capacity_2025 = facility_capacity.xs(2025, level='year')
    multiplier = baseline_fractions_2025.where(baseline_fractions_2025 != 0, other=1.0)
    print("Scaling each facility/cadre's capacity by its 2025 Nothing-baseline fraction; zero fractions use 1.0.")
    scaled_capacity_by_facility = multiplier.mul(baseline_capacity_2025, axis=0)
    scaled_capacity_by_cadre_and_level = scaled_capacity_by_facility.groupby(
        level=['OfficerType', 'FacilityLevel']
    ).sum(min_count=1)
    scaled_capacity_by_cadre = scaled_capacity_by_cadre_and_level.groupby(
        level='OfficerType'
    ).sum(min_count=1)
    annual_capacity_by_cadre = scaled_capacity_by_cadre.median(axis=1)
    results['annual_capacity_by_cadre'] = annual_capacity_by_cadre
    print("Aggregated scaled capacity within each run, then calculated median annual capacity by cadre.")

    staff_count_by_cadre = (
        daily_capacity_by_cadre_and_level.groupby('Officer_Category')['Staff_Count'].sum()
    )
    results['staff_count_by_cadre'] = staff_count_by_cadre

    # We need the time used by cadre in 2026
    actual_capacity_used_by_cadre = (
        actual_capacity_used_by_cadre_and_level
        .groupby(['OfficerType', 'year'])
        .sum()
    )
    actual_capacity_used_by_cadre_2026 = actual_capacity_used_by_cadre.xs(
        YEAR_OF_SWITCH, level='year', axis=0
    ).copy()
    if len(zero_hsi_scenarios):
        baseline_capacity = actual_capacity_used_by_cadre_2026.xs("Nothing", axis=1, level="draw")
        for draw in zero_hsi_scenarios:
            columns = actual_capacity_used_by_cadre_2026.columns[
                actual_capacity_used_by_cadre_2026.columns.get_level_values("draw") == draw
            ]
            actual_capacity_used_by_cadre_2026.loc[:, columns] = baseline_capacity.loc[
                :, columns.get_level_values("run")
            ].to_numpy()

    results['actual_capacity_used_by_cadre_2026'] = (
        compute_summary_statistics(actual_capacity_used_by_cadre_2026, 'median')
    )
    print(results['actual_capacity_used_by_cadre_2026'])
    # Load the salary per minutes by cadre and facility level
    # ResourceFile_Minute_Salary_HR.csv is produced using Bingling's code,
    # except that she was dropping Facility_Level, which I have retained.
    salary_by_cadre_and_level = (
        pd.read_csv(resourcefilepath / "costing"  / "ResourceFile_Minute_Salary_HR.csv")
    )
    salary_by_cadre_and_level =(
        salary_by_cadre_and_level
        .groupby(['Facility_Level', 'Officer_Type_Code'])['Minute_Salary_USD']
        .mean()
        .rename_axis(index={'Facility_Level': 'FacilityLevel', 'Officer_Type_Code': 'OfficerType'})
    ).reorder_levels(['OfficerType', 'FacilityLevel'])

    salary_aligned = salary_by_cadre_and_level.reindex(
        actual_capacity_used_by_cadre_and_level.index.droplevel("year")
    ).set_axis(
        actual_capacity_used_by_cadre_and_level.index
    )
    actual_cost_by_cadre_and_level = actual_capacity_used_by_cadre_and_level.mul(salary_aligned, axis=0)
    print("Calculated salary costs with cadre/level, year, draw and run aligned.")

    # The costing script calculates the HRH costs for the total cadre, not for the strength used.
    # We will remove the HRH costs from the input_costs; this includes other things besides salary
    # Then we will add to the remaining costs the salary costs calculated above. This gives us
    # the total input costs for the intervention. Note that are ignoring non-salary HRH costs.

    # Wrangling to make the actual_cost_by_cadre_and_level the same same as input_costs
    actual_cost_by_cadre_and_level = (
        actual_cost_by_cadre_and_level
        .stack(level=['draw', 'run'])
        .rename('cost')
        .reset_index()
    )

    actual_cost_by_cadre_and_level = actual_cost_by_cadre_and_level.rename(columns={
        'OfficerType':   'cost_subgroup',
        'FacilityLevel': 'Facility_Level',
    })
    actual_cost_by_cadre_and_level['cost_subcategory'] = 'salary_for_cadres_used'
    actual_cost_by_cadre_and_level['cost_category'] = 'human resources for health'

    actual_cost_by_cadre_and_level = actual_cost_by_cadre_and_level[[
        'draw', 'run', 'year',
        'cost_subcategory',
        'Facility_Level',
        'cost_subgroup',
        'cost',
        'cost_category',
    ]]

    actual_cost_by_cadre_and_level = actual_cost_by_cadre_and_level.reset_index(drop=True)
    mask = actual_cost_by_cadre_and_level['year'] >= YEAR_OF_SWITCH
    actual_cost_by_cadre_and_level = apply_discounting_to_cost_data(
        actual_cost_by_cadre_and_level.loc[mask].copy(),
        _discount_rate=discount_rate_cost,
        _initial_year=min(list_of_relevant_years_for_costing),
    )
    # Now combine with input_costs
    input_costs = input_costs[input_costs['cost_category'] != 'human resources for health']
    input_costs_with_proportional_hrh_costs = pd.concat([input_costs, actual_cost_by_cadre_and_level])

    # Because of Issue 1877 (fixed in PR 1880, not merged in feature branch),
    # the simulations for the Nothing scenario and scenarios with zero HSIs are
    # not identical, as they should be. Consequently the costs and dalys are
    # both diffrent across. As a temporary fix, we will use the costs and dalys
    # from the Nothing scenario for zero-HSIs scenarios. We expect that this
    # will not be needed when we have more runs, thereby achieving more stable
    # results.

    if len(zero_hsi_scenarios):
        baseline_costs = input_costs_with_proportional_hrh_costs.loc[
            input_costs_with_proportional_hrh_costs["draw"].eq("Nothing")
        ]
        replacements = []
        for draw in zero_hsi_scenarios:
            runs = input_costs_with_proportional_hrh_costs.loc[
                input_costs_with_proportional_hrh_costs["draw"].eq(draw), "run"
            ].unique()
            if not set(runs).issubset(baseline_costs["run"].unique()):
                raise ValueError(f"Missing Nothing cost runs for zero-HSI scenario {draw}.")
            replacements.append(baseline_costs.loc[baseline_costs["run"].isin(runs)].assign(draw=draw))
        input_costs_with_proportional_hrh_costs = pd.concat(
            [
                input_costs_with_proportional_hrh_costs.loc[
                    ~input_costs_with_proportional_hrh_costs["draw"].isin(zero_hsi_scenarios)
                ],
                *replacements,
            ],
            ignore_index=True,
        )
    if nothing_by_run is not None:
        final_costs = nothing_by_run["final_costs"]
        input_costs_with_proportional_hrh_costs = pd.concat(
            [
                input_costs_with_proportional_hrh_costs.loc[lambda data: data["draw"].ne("Nothing")],
                final_costs.loc[final_costs["run"].isin(runs)],
            ],
            ignore_index=True,
        )

    # For LCOA we need annual consumables cost per intervention in the year of the switch i.e. 2026.
    annual_cons_cost = input_costs_with_proportional_hrh_costs.groupby(['draw', 'cost_category', 'year', 'run'])['cost'].sum()
    mask = (
        (annual_cons_cost.index.get_level_values('year') == YEAR_OF_SWITCH) &
        (annual_cons_cost.index.get_level_values('cost_category') == 'medical consumables')
    )
    annual_cons_cost_2026 = annual_cons_cost[mask].droplevel(['year', 'cost_category'])
    results['annual_cons_cost_2026'] = (compute_summary_statistics(annual_cons_cost_2026.to_frame().T, 'median')).T

    # Overall costs in 2026
    mask = (input_costs_with_proportional_hrh_costs['year'] == YEAR_OF_SWITCH)

    annual_scenario_cost_2026 = input_costs_with_proportional_hrh_costs[mask].groupby(['draw', 'run'])['cost'].sum()
    results['annual_scenario_cost_2026'] = (
        compute_summary_statistics(annual_scenario_cost_2026.to_frame().T).iloc[0].unstack()
    )
    # Computing incremental costs
    if do_comparison:
        print("Computing incremental_scenario_cost...")
        total_input_cost = input_costs_with_proportional_hrh_costs.groupby(['draw', 'run'])['cost'].sum()
        diagnostic_costs = (
            input_costs_with_proportional_hrh_costs.loc[
                input_costs_with_proportional_hrh_costs["year"] >= YEAR_OF_SWITCH
            ].groupby(["draw", "run"])["cost"].sum(min_count=1).unstack("draw")
        )
        cost_baseline = diagnostic_costs["Nothing"]
        cost_scenarios = diagnostic_costs.drop(columns="Nothing")
        results["overall_cost_difference_by_run"] = cost_scenarios.sub(cost_baseline, axis=0).T
        results["overall_cost_ratio_by_run"] = cost_scenarios.div(
            cost_baseline.where(cost_baseline.ne(0)), axis=0
        ).T
        incremental_scenario_cost = (pd.DataFrame(
            find_difference_relative_to_comparison(
                total_input_cost,
                comparison='Nothing',)
        ))
        incremental_scenario_cost = (
            incremental_scenario_cost.T.reorder_levels(["draw", "run"], axis=1).sort_index(axis=1)
        )

        incremental_scenario_cost_summarized = compute_summary_statistics(incremental_scenario_cost, 'median').iloc[0].unstack()

    if do_comparison:
        if nothing_by_run is None:
            dalys_2026 = dalys.xs(str(YEAR_OF_SWITCH), level="period")
        else:
            dalys_2026 = discounted_dalys.xs(str(YEAR_OF_SWITCH), level="period") * (
                (1.0 + DISCOUNT_RATE) ** (YEAR_OF_SWITCH - discount_base_year)
            )
        dalys_averted_2026 = (
            -1.0 * pd.DataFrame(
                find_difference_extra_relative_to_comparison(dalys_2026.sum(), comparison='Nothing'))

        )
        results['dalys_averted_2026'] = compute_summary_statistics(dalys_averted_2026.T, 'median').iloc[0].unstack()

        dalys_averted = (
            -1.0 * pd.DataFrame(
                find_difference_extra_relative_to_comparison(discounted_dalys.sum(), comparison='Nothing'))

        )

        pc_dalys_averted = 100.0 * compute_summary_statistics(
            -1.0 * pd.DataFrame(
                find_difference_extra_relative_to_comparison(discounted_dalys.sum(), comparison='Nothing', scaled=True)).T,
            central_measure='median'
        ).iloc[0].unstack()

        net_health = dalys_averted.T - incremental_scenario_cost / MALAWI_CET
        net_health_summarized = (
            compute_summary_statistics(net_health, central_measure='median').iloc[0].unstack()
        )

        if nothing_by_run is not None:
            results["total_cost_by_run"] = total_input_cost
            results["dalys_averted_by_run"] = dalys_averted.T
            results["incremental_cost_by_run"] = incremental_scenario_cost
            results["net_health_by_run"] = net_health
            icer = incremental_scenario_cost.div(dalys_averted.T.where(dalys_averted.T.ne(0)))
            results["icers_summarized"] = compute_summary_statistics(icer, "median").iloc[0].unstack()
            results["annual_capacity_used_by_cadre_and_level"] = results["actual_annual_capacity_by_cadre"]
            results["annual_cost_by_cadre"] = (
                input_costs_with_proportional_hrh_costs.loc[lambda data: data["cost_category"].eq(
                    "human resources for health"
                )].groupby(["year", "cost_subgroup", "draw", "run"])["cost"].sum().unstack(["draw", "run"])
                .pipe(compute_summary_statistics, "median")
            )

        mean_dalys_averted = compute_summary_statistics(dalys_averted.T, central_measure='mean').iloc[0].unstack()
        dalys_averted = compute_summary_statistics(dalys_averted.T, central_measure='median').iloc[0].unstack()

    # Add consumables budget to this dictionary so that we have everything in one place
    # USD 225,602,946 (203136642 from donors + 22466304 from the government)
    # Revision of Malawi’s Health Benefits Package: A Critical Analysis of Policy Formulation and Implementation
    # https://doi.org/10.1016/j.vhri.2023.10.007
    results['annual_consumables_budget'] = 225602946

    results['dalys'] = compute_summary_statistics(dalys, 'median')
    results['dalys_averted'] = dalys_averted if do_comparison else None
    results['mean_dalys_averted'] = mean_dalys_averted if do_comparison else None
    results['pc_dalys_averted'] = pc_dalys_averted if do_comparison else None
    results['net_health'] = net_health_summarized if do_comparison else None
    results['incremental_scenario_cost'] = incremental_scenario_cost_summarized if do_comparison else None

    if nothing_by_run is None and "Nothing" in discounted_dalys.columns.get_level_values("draw"):
        results["nothing_by_run"] = {
            key: data.xs("Nothing", axis=1, level="draw")
            for key, data in {
                "total_population_by_year": total_population_by_year,
                "counts_of_hsi_by_short_treatment_id": counts_of_hsi_by_short_treatment_id,
                "counts_of_hsi_by_period": counts_of_hsi_by_period,
                "num_deaths": num_deaths,
                "discounted_dalys": discounted_dalys,
                "capacity_fractions": avg_frac_capacity_used_by_clinic_and_facility,
                "actual_annual_capacity_by_cadre": actual_capacity_used_by_cadre_and_level,
                "actual_capacity_used_by_cadre_2026": actual_capacity_used_by_cadre_2026,
            }.items()
        }
        results["nothing_by_run"].update(
            input_costs=results["input_costs"].loc[lambda data: data["draw"].eq("Nothing")].copy(),
            final_costs=input_costs_with_proportional_hrh_costs.loc[lambda data: data["draw"].eq("Nothing")].copy(),
            daly_discount_base_year=int(discount_base_year),
        )
    accrued = discounted_dalys.sum(min_count=1).to_frame().T
    if "Nothing" in accrued.columns.get_level_values("draw"):
        scenarios = accrued.drop(columns="Nothing", level="draw")
        matched = match_baseline(scenarios, accrued.xs("Nothing", axis=1, level="draw"))
        results["discounted_daly_totals_by_run"] = {
            "accrued": scenarios.iloc[0],
            "averted": (matched - scenarios).iloc[0],
        }
    results["analysis_metadata"] = analysis_metadata(target_period_tuple)
    return results


def apply(
    results_folder: Path,
    output_folder: Path,
    resourcefilepath: Path,
    target_period_tuple: tuple[Date, Date],
    do_comparison: bool = True,
):
    """Process treatment-ID results and attach the intervention-level LCOA inputs."""
    results = process_scenarios(
        results_folder, resourcefilepath, target_period_tuple,
        get_parameter_names_from_scenario_file(), do_comparison,
    )
    # Extract DALYs and costs from the LCOA input workbook (EHP_BasedOnLCOA sheet).
    lcoa_workbook_path = Path(__file__).resolve().parent / "ResourceFile_PriorityRanking_ALLPOLICIES_EHP_dalys_costs.xlsx"
    lcoa_df = pd.read_excel(lcoa_workbook_path, sheet_name="EHP_BasedOnLCOA")
    col_a, col_i, col_j, col_k, col_l, col_m = (
        lcoa_df.columns[0],
        lcoa_df.columns[8],
        lcoa_df.columns[9],
        lcoa_df.columns[10],
        lcoa_df.columns[11],
        lcoa_df.columns[12],
    )
    dalys_and_costs_from_lcoa = lcoa_df[[col_a, col_i, col_j, col_k, col_l, col_m]].rename(
        columns={
            col_a: "treatment_id",
            col_i: "icer",
            col_j: "dalys_per_patient",
            col_k: "cost_per_case",
            col_l: "eligible_cases",
            col_m: "lcoa_flag",
        }
    )
    for numeric_col in ["icer", "dalys_per_patient", "cost_per_case", "eligible_cases"]:
        dalys_and_costs_from_lcoa[numeric_col] = pd.to_numeric(
            dalys_and_costs_from_lcoa[numeric_col], errors="coerce"
        )
    dalys_and_costs_from_lcoa["overall_dalys"] = (
        dalys_and_costs_from_lcoa["dalys_per_patient"] * dalys_and_costs_from_lcoa["eligible_cases"]
    )
    dalys_and_costs_from_lcoa["overall_costs"] = (
        dalys_and_costs_from_lcoa["cost_per_case"] * dalys_and_costs_from_lcoa["eligible_cases"]
    )
    dalys_and_costs_from_lcoa = dalys_and_costs_from_lcoa[
        dalys_and_costs_from_lcoa["treatment_id"].notna()
        & (
            dalys_and_costs_from_lcoa["overall_dalys"].notna()
            | dalys_and_costs_from_lcoa["overall_costs"].notna()
            | dalys_and_costs_from_lcoa["icer"].notna()
        )
    ]
    dalys_and_costs_from_lcoa['net_health'] = (
        dalys_and_costs_from_lcoa['overall_dalys'] -
        dalys_and_costs_from_lcoa['overall_costs'] / MALAWI_CET
    )
    results["dalys_and_costs_from_lcoa"] = dalys_and_costs_from_lcoa

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    parser.add_argument("output_folder", type=Path, nargs="?", default=None)
    parser.add_argument("--target-start", type=str, default=None)
    parser.add_argument("--target-end", type=str, default=None)
    parser.add_argument("--do-comparison", type=parse_bool, default=True)
    args = parser.parse_args()

    if (args.target_start is None) != (args.target_end is None):
        parser.error("Provide both --target-start and --target-end, or neither.")

    target_period_tuple = (
        parse_iso_date(args.target_start),
        parse_iso_date(args.target_end),
    )
    if not target_period_tuple[0] < target_period_tuple[1]:
        parser.error("--target-start must be earlier than --target-end.")

    out = args.output_folder if args.output_folder is not None else args.results_folder
    out.mkdir(parents=True, exist_ok=True)
    results = apply(
        results_folder=args.results_folder,
        output_folder=out,
        resourcefilepath=Path("./resources"),
        target_period_tuple=target_period_tuple,
        do_comparison=args.do_comparison,
    )
    outfile = (
        f"{target_period_tuple[1].year:04d}-{target_period_tuple[1].month:02d}-{target_period_tuple[1].day:02d}"
        "_fullresults.pkl"
    )
    with open(out / outfile, 'wb') as f:
        pickle.dump(results, f)

    print(f"Analysis complete! Results saved to {out / outfile}")
