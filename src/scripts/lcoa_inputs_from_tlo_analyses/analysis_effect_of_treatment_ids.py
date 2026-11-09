"""Extract LCOA inputs and sanity-check calculations into one results pickle."""

import warnings
from time import perf_counter
from pandas.errors import (
    PerformanceWarning,
    SettingWithCopyWarning
)
import argparse
from datetime import date
import pickle
from pathlib import Path
import pandas as pd


from tlo import Date


from scripts.lcoa_inputs_from_tlo_analyses.results_processing_utils import (
    format_scenario_name,
    get_counts_of_hsi_by_short_treatment_id,
    get_num_dalys_by_cause_label,
    get_num_deaths_by_cause_label,
    get_parameter_names_from_scenario_file,
    get_periods_within_target_period,
    get_total_num_dalys_by_agegrp_and_label,
    get_total_num_death_by_agegrp_and_label,
    get_total_population_by_year,
    make_get_num_dalys_by_cause_label_and_period,
    make_get_num_deaths_by_cause_label_and_period,
    make_get_counts_of_appts_by_period,
    make_get_counts_of_hsis_by_period,
    set_param_names_as_column_index_level_0,
    target_period,
    find_difference_extra_relative_to_comparison,
    find_difference_relative_to_comparison,
    get_staff_count_by_facid_and_officer_type,
    get_capacity_used_by_officer_type_and_facility_level,
    melt_model_output_draws_and_runs
)

from scripts.costing.cost_estimation import (
    apply_discounting_to_cost_data,
    do_line_plot_of_cost,
    do_stacked_bar_plot_of_cost_by_category,
    estimate_input_cost_of_scenarios,
    estimate_projected_health_spending,
    extract_roi_at_specific_implementation_costs,
    generate_multiple_scenarios_roi_plot,
    load_unit_cost_assumptions,
    summarize_cost_data,
    tabulate_roi_estimates,
)
from tlo.analysis.utils import (
    compute_summary_statistics,
    extract_results,
    get_color_short_treatment_id,
    make_age_grp_lookup,
    summarize,
)
# python src/scripts/lcoa_inputs_from_tlo_analyses/analysis_effect_of_treatment_ids.py outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-02-12T120859Z figs/ --target-start=2010-01-01 --target-end=2025-12-31
# python src/scripts/lcoa_inputs_from_tlo_analyses/analysis_effect_of_treatment_ids.py outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-02-16T154500Z figs/ --target-start=2025-01-01 --target-end=2041-01-01
# python src/scripts/lcoa_inputs_from_tlo_analyses/analysis_effect_of_treatment_ids.py outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-04-01T130709Z --target-start=2010-01-01 --target-end=2041-01-01 --do-comparison=False
# python src/scripts/lcoa_inputs_from_tlo_analyses/analysis_effect_of_treatment_ids.py outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-combined outputs/generated_outputs --target-start=2026-01-01 --target-end=2040-12-31 --cost-checkpoint-profile=baseline --load-input-costs-from-checkpoint=True
# python src/scripts/lcoa_inputs_from_tlo_analyses/analysis_effect_of_treatment_ids.py outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-10-runs-combined outputs/generated_outputs --target-start=2010-01-01 --target-end=2040-12-31 --cost-checkpoint-profile=10runstest --load-input-costs-from-checkpoint=True
#python src/scripts/lcoa_inputs_from_tlo_analyses/analysis_effect_of_treatment_ids.py outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-02-16T154500Z outputs/generated_outputs --target-start=2026-01-01 --target-end=2040-12-31 --cost-checkpoint-profile=resumed-runs-only --load-input-costs-from-checkpoint=False
#python src/scripts/lcoa_inputs_from_tlo_analyses/analysis_effect_of_treatment_ids.py outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-09-12T063913Z --target-start=2026-01-01 --target-end=2040-01-01 --do-comparison=False
# python src/scripts/lcoa_inputs_from_tlo_analyses/analysis_effect_of_treatment_ids.py outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-10-15T115126Z outputs/generated_outputs --target-start=2026-01-01 --target-end=2040-12-31 --cost-checkpoint-profile=baseline --load-input-costs-from-checkpoint=False

# Do not exclude FirstAttendance_SpuriousEmergencyCare as it has non-blank
# appointment footprint
EXCLUDED_HSIs = [
    "FirstAttendance_Emergency",
    "FirstAttendance_NonEmergency",
    "Inpatient_Care"
]
SUSPENDED_RESULTS_FOLDER = Path("outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-10-09T145410Z")
RESUMED_RESULTS_FOLDER = Path("outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-10-15T115126Z")
YEAR_OF_SWITCH = 2026

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

def apply(
    results_folder: Path,
    output_folder: Path,
    resourcefilepath: Path,
    target_period_tuple: tuple[Date, Date],
    do_comparison: bool = True,
):
    """Process results to produce objects needed for LCOA analysis."""
    _, age_grp_lookup = make_age_grp_lookup()

    # Extract districts and facility levels from the Master Facility List
    mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
    facility_id_levels_dict = dict(zip(mfl['Facility_ID'], mfl['Facility_Level']))

    param_names = get_parameter_names_from_scenario_file()

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
        extract_results(
            results_folder,
            module='tlo.methods.demography',
            key='population',
            custom_generate_series=lambda _df: get_total_population_by_year(_df, target_period_tuple),
            do_scaling=True,
            suspended_results_folder=SUSPENDED_RESULTS_FOLDER,
        ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)
    )

    cols_to_drop = [
        col for col in total_population_by_year.columns
        if col[0].strip().startswith(tuple(EXCLUDED_HSIs))
    ]

    print(f"Dropping columns from total_population_by_year: {cols_to_drop}")
    total_population_by_year.drop(columns=cols_to_drop, inplace=True)

    results['total_population_by_year'] = (
        compute_summary_statistics(total_population_by_year, central_measure='median')
    )

    print("Extracting HSI event counts...")
    counts_of_hsi_by_short_treatment_id = (
        extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="HSI_Event",
            custom_generate_series=lambda _df: get_counts_of_hsi_by_short_treatment_id(_df, target_period_tuple),
            do_scaling=True,
            suspended_results_folder=SUSPENDED_RESULTS_FOLDER,
        )
        .pipe(set_param_names_as_column_index_level_0, param_names=param_names)
        .fillna(0.0)
        .sort_index()
    ).drop(EXCLUDED_HSIs, errors='ignore')

    results['counts_of_hsi_by_short_treatment_id'] = (
        compute_summary_statistics(counts_of_hsi_by_short_treatment_id, 'median')
    )
    counts_of_hsi_by_period = (
        extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="HSI_Event",
            custom_generate_series=lambda _df: get_num_hsi_by_period(_df),
            do_scaling=True,
            suspended_results_folder=SUSPENDED_RESULTS_FOLDER,
        )
        .pipe(set_param_names_as_column_index_level_0, param_names=param_names)
        .fillna(0.0)
        .sort_index()
    ).drop(columns=cols_to_drop)

    results['counts_of_hsi_by_period'] = (
        compute_summary_statistics(counts_of_hsi_by_period, 'median')
    )

    print("Extracting DALYs by label...")
    dalys = (
        extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked_by_age_and_time",
            custom_generate_series=get_num_dalys_by_cause_label_and_period,
            do_scaling=True,
            suspended_results_folder=SUSPENDED_RESULTS_FOLDER,
        ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)
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
        extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=get_num_deaths_by_cause_label_and_period,
            do_scaling=True,
            suspended_results_folder=SUSPENDED_RESULTS_FOLDER,
        ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)
    )

    num_deaths = num_deaths.drop(columns=cols_to_drop, errors='ignore')

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

    discount_rate_dalys = 0.03
    dalys_years = dalys.index.get_level_values("period").astype(int)
    discount_base_year = dalys_years.min()
    discounted_dalys = dalys.div(
        (1.0 + discount_rate_dalys) ** (dalys_years - discount_base_year),
        axis=0,
    )

    results['discounted_dalys'] = (
        compute_summary_statistics(discounted_dalys, central_measure='median'))

    # Costs calculation
    print("Calculating costs...")
    discount_rate_cost = 0.03
    # Period relevant for costing
    TARGET_PERIOD = (Date(YEAR_OF_SWITCH, 1, 1), Date(2040, 12, 31))
    relevant_period_for_costing = [i.year for i in TARGET_PERIOD]
    list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0], relevant_period_for_costing[1] + 1))
    print("List of relevant years for costing:", list_of_relevant_years_for_costing)
    input_costs = estimate_input_cost_of_scenarios(
        results_folder,
        resourcefilepath,
        suspended_results_folder=SUSPENDED_RESULTS_FOLDER,
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

    # Capacity_By_FacID_and_Officer logs the fraction of time used per officer
    # type.
    # To get the actual minutes, we multiply by the total available minutes.
    print("Extracting capacity fractions with clinic and facility identifiers...")
    # Set do_scaling to False as this is a fraction calculated during runtime.
    avg_frac_capacity_used_by_clinic_and_facility = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Capacity_By_FacID_and_Officer',
        custom_generate_series=lambda df: get_capacity_used_by_officer_type_and_facility_level(df, facility_id_levels_dict),
        do_scaling=False,
        suspended_results_folder=SUSPENDED_RESULTS_FOLDER,
    ).pipe(set_param_names_as_column_index_level_0, param_names=param_names).drop(columns=cols_to_drop)

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

    # Use the Nothing baseline for capacity scalin
    baseline_fractions_2025 = avg_frac_capacity_used_by_clinic_and_facility.xs(
        'Nothing', level='draw', axis=1
    ).xs(2025, level='year')
    print(f"Extracted baseline fractions for 2025 from the Nothing scenario: {baseline_fractions_2025}")

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
    actual_capacity_used_by_cadre_2026 = actual_capacity_used_by_cadre.xs(YEAR_OF_SWITCH, level='year', axis=0)

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

    salary_aligned = (
        salary_by_cadre_and_level.reindex(actual_capacity_used_by_cadre_and_level.index.droplevel("year"))
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
    # Now combine with input_costs
    input_costs = input_costs[input_costs['cost_category'] != 'human resources for health']
    input_costs_with_proportional_hrh_costs = pd.concat([input_costs, actual_cost_by_cadre_and_level[mask]])

    # For LCOA we need annual consumables cost per intervention in the year of the switch i.e. 2026.
    annual_cons_cost = input_costs.groupby(['draw', 'cost_category', 'year', 'run'])['cost'].sum()
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
        annual_scenario_cost_2026 = input_costs_with_proportional_hrh_costs.loc[
            input_costs_with_proportional_hrh_costs["year"].eq(YEAR_OF_SWITCH)
        ].groupby(["draw", "run"])["cost"].sum()
        results["annual_scenario_cost_2026"] = compute_summary_statistics(
            annual_scenario_cost_2026.to_frame().T
        ).iloc[0].unstack()

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
        dalys_2026 = dalys.xs('2026', level='period')
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

        dalys_averted = compute_summary_statistics(dalys_averted.T, central_measure='median').iloc[0].unstack()
        mean_dalys_averted = compute_summary_statistics(dalys_averted.T, central_measure='mean').iloc[0].unstack()

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
