"""Produce plots to show the impact each set of treatments."""

import warnings
from time import perf_counter
from pandas.errors import (
    PerformanceWarning,
    SettingWithCopyWarning
)
import argparse
from datetime import date
import glob
import os
import zipfile
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tlo import Date
from tlo.util import create_age_range_lookup

from scripts.lcoa_inputs_from_tlo_analyses.fig_utils import (
    do_bar_plot_with_ci,
    plot_multiindex_dot_with_interval,
)
from scripts.lcoa_inputs_from_tlo_analyses.results_processing_utils import (
    get_counts_of_appts,
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
    find_difference_relative_to_comparison
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
    squarify_neat,
    summarize,
)
# python src/scripts/lcoa_inputs_from_tlo_analyses/analysis_effect_of_treatment_ids.py outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-02-12T120859Z figs/ --target-start=2010-01-01 --target-end=2025-12-31
# python src/scripts/lcoa_inputs_from_tlo_analyses/analysis_effect_of_treatment_ids.py outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-02-16T154500Z figs/ --target-start=2025-01-01 --target-end=2041-01-01
# python src/scripts/lcoa_inputs_from_tlo_analyses/analysis_effect_of_treatment_ids.py outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-combined --target-start=2010-01-01 --target-end=2041-01-01

PERIOD_LENGTH_YEARS_FOR_BAR_PLOTS = 1
#suspended_folder = Path("outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-02-12T120859Z")
#results_folder = Path("outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-02-16T154500Z")
# SCALING_FACTOR retrieved from the suspended run in
# outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-02-12T120859Z
# SCALING_FACTOR = 58.158436
EXCLUDED_HSIs = [
    "FirstAttendance_Emergency",
    "FirstAttendance_NonEmergency",
    "FirstAttendance_SpuriousEmergencyCare",
    "Inpatient_Care"
]

def parse_iso_date(value: str) -> Date:
    parsed = date.fromisoformat(value)
    return Date(parsed.year, parsed.month, parsed.day)


def apply(
    results_folder: Path,
    output_folder: Path,
    resourcefilepath: Path,
    target_period_tuple: tuple[Date, Date]
):
    """Produce standard plots describing effect of each TREATMENT_ID."""
    _, age_grp_lookup = make_age_grp_lookup()

    param_names = get_parameter_names_from_scenario_file()
    get_num_deaths_by_cause_label_and_period = make_get_num_deaths_by_cause_label_and_period(
        PERIOD_LENGTH_YEARS_FOR_BAR_PLOTS,
        target_period_tuple,
    )
    get_num_dalys_by_cause_label_and_period = make_get_num_dalys_by_cause_label_and_period(
        PERIOD_LENGTH_YEARS_FOR_BAR_PLOTS,
        target_period_tuple,
    )
    get_num_hsi_by_period = make_get_counts_of_hsis_by_period(
        PERIOD_LENGTH_YEARS_FOR_BAR_PLOTS,
        target_period_tuple=target_period_tuple,
    )
    results = {}
    # Costs calculation
    print("Calculating costs...")
    discount_rate_cost = 0.03
    # Period relevant for costing
    TARGET_PERIOD = (Date(2026, 1, 1), Date(2040, 12, 31))  # This is the period that is costed
    relevant_period_for_costing = [i.year for i in TARGET_PERIOD]
    list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0], relevant_period_for_costing[1] + 1))
    print("List of relevant years for costing:", list_of_relevant_years_for_costing)
    start = perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
        input_costs = estimate_input_cost_of_scenarios(
                          results_folder,
                          resourcefilepath,
                          _years=list_of_relevant_years_for_costing,
                          cost_only_used_staff=True,
                          _discount_rate=discount_rate_cost,
                          _metric="median",)

    elapsed = perf_counter() - start
    print(f"\n=== TIMING: estimate_input_cost_of_scenarios took {elapsed:.3f}s ===\n", flush=True)
    results['input_costs'] = input_costs

    # Computing ICERs
    print("Computing ICERs...")
    start = perf_counter()
    total_input_cost = input_costs.groupby(['draw', 'run'])['cost'].sum()
    incremental_scenario_cost = (pd.DataFrame(
        find_difference_relative_to_comparison(
            total_input_cost,
            comparison=0,)
    ).T.iloc[0].unstack()).T

    elapsed = perf_counter() - start
    print(f"\n=== TIMING: computing icers took {elapsed:.3f}s ===\n", flush=True)

    incremental_scenario_cost_summarized = summarize_cost_data(incremental_scenario_cost, _metric='median')
    results['incremental_scenario_cost'] = incremental_scenario_cost_summarized

    # Get total population by year
    print("Extracting population data...")
    total_population_by_year = (
        extract_results(
            results_folder,
            module='tlo.methods.demography',
            key='population',
            custom_generate_series=lambda _df: get_total_population_by_year(_df, target_period_tuple),
            do_scaling=True,
            autodiscover=True
        ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)
    )

    total_population_by_year = compute_summary_statistics(total_population_by_year, central_measure='median')
    results['total_population_by_year'] = total_population_by_year

    counts_of_hsi_by_short_treatment_id = (
        extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="HSI_Event",
            custom_generate_series=lambda _df: get_counts_of_hsi_by_short_treatment_id(_df, target_period_tuple),
            do_scaling=True,
            autodiscover=True,
        )
        .pipe(set_param_names_as_column_index_level_0, param_names=param_names)
        .fillna(0.0)
        .sort_index()
    ).drop(EXCLUDED_HSIs, errors='ignore')

    counts_of_hsi_by_short_treatment_id = (
        compute_summary_statistics(counts_of_hsi_by_short_treatment_id, 'median')
    )

    results['counts_of_hsi_by_short_treatment_id'] = counts_of_hsi_by_short_treatment_id

    counts_of_hsi_by_period = (
        extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="HSI_Event",
            custom_generate_series=lambda _df: get_num_hsi_by_period(_df),
            do_scaling=True,
            autodiscover=True,
        )
        .pipe(set_param_names_as_column_index_level_0, param_names=param_names)
        .fillna(0.0)
        .sort_index()
    ).drop(EXCLUDED_HSIs, level=0, errors='ignore')

    counts_of_hsi_by_period = (
        compute_summary_statistics(counts_of_hsi_by_period, 'median')
    )
    results['counts_of_hsi_by_period'] = counts_of_hsi_by_period

    print("Extracting total deaths and DALYs by label...")
    num_deaths = (
        extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=get_num_deaths_by_cause_label_and_period,
            do_scaling=True,
            autodiscover=True,
        ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)
    )

    num_deaths_averted = summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_deaths.sum(), comparison='Nothing')).T
    ).iloc[0].unstack()

    pc_deaths_averted = 100.0 * summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_deaths.sum(), comparison='Nothing', scaled=True)).T
    ).iloc[0].unstack()

    num_deaths = compute_summary_statistics(num_deaths, central_measure='median')

    results['num_deaths'] = num_deaths
    results['num_deaths_averted'] = num_deaths_averted
    results['pc_deaths_averted'] = pc_deaths_averted

    num_dalys = (
        extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked_by_age_and_time",
            custom_generate_series=get_num_dalys_by_cause_label_and_period,
            do_scaling=True,
            autodiscover=True,
        ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)
    )

    num_dalys_averted = summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_dalys.sum(), comparison='Nothing')).T
    ).iloc[0].unstack()

    pc_dalys_averted = 100.0 * summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_dalys.sum(), comparison='Nothing', scaled=True)).T
    ).iloc[0].unstack()

    num_dalys = compute_summary_statistics(num_dalys, central_measure='median')

    results['num_dalys'] = num_dalys
    results['num_dalys_averted'] = num_dalys_averted
    results['pc_dalys_averted'] = pc_dalys_averted

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    parser.add_argument("output_folder", type=Path, nargs="?", default=None)
    parser.add_argument("--target-start", type=str, default=None)
    parser.add_argument("--target-end", type=str, default=None)
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
    )
    outfile = (
        f"{target_period_tuple[1].year:04d}-{target_period_tuple[1].month:02d}-{target_period_tuple[1].day:02d}"
        "_fullresults.pkl"
    )
    with open(out / outfile, 'wb') as f:
        pickle.dump(results, f)

    print(f"Analysis complete! Results saved to {out / outfile}")
