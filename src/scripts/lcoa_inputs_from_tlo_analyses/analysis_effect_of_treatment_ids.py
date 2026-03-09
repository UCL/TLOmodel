"""Produce plots to show the impact each set of treatments."""

import argparse
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
    extract_deaths_total,
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
    set_param_names_as_column_index_level_0,
    target_period,
    find_difference_extra_relative_to_comparison,
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

TARGET_PERIOD = (Date(2026, 1, 1), Date(2027, 1, 1))
PERIOD_LENGTH_YEARS_FOR_BAR_PLOTS = 5
suspended_folder = Path("outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-02-12T120859Z")
results_folder = Path("outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-02-16T154500Z")
# SCALING_FACTOR retrieved from the suspended run in
# outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-02-12T120859Z
# SCALING_FACTOR = 58.158436


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard plots describing effect of each TREATMENT_ID."""
    _, age_grp_lookup = make_age_grp_lookup()

    param_names = get_parameter_names_from_scenario_file()

    # Costs calculation
    print("Calculating costs...")
    discount_rate_cost = 0.03
    input_costs = estimate_input_cost_of_scenarios(
                      results_folder,
                      resourcefilepath,
                      suspended_results_folder=suspended_folder,
                      _draws=[0],
                      _runs=[0],
                      cost_only_used_staff=True,
                      _discount_rate=discount_rate_cost,
                      _metric="median",)

    # Get total population by year
    print("Extracting population data...")
    total_population_by_year = extract_results(
        results_folder,
        module='tlo.methods.demography',
        key='population',
        custom_generate_series=get_total_population_by_year,
        do_scaling=True,
        suspended_results_folder=suspended_folder,
        autodiscover=True
    )
    total_population_by_year = compute_summary_statistics(total_population_by_year, central_measure = 'median')
    total_population_by_year = set_param_names_as_column_index_level_0(total_population_by_year, param_names=param_names)
    total_population_by_year = (total_population_by_year
            .stack(level=["draw", "stat"])   # move draw & stat into index
            .reset_index()                   # turn all index levels into columns
            .rename(columns={0: "population"})    # name the value column
    ).set_index(["draw", "stat",'year'])

    total_population_by_year = total_population_by_year.rename(
        index={"central": "median"},
        level="stat"
    )

    print("Extracting total deaths and DALYs by label...")
    num_deaths = (
        extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=lambda _df: get_num_deaths_by_cause_label(_df, TARGET_PERIOD),
            do_scaling=True,
            suspended_results_folder=suspended_folder,
            autodiscover=True,
        )
        .pipe(set_param_names_as_column_index_level_0, param_names=param_names)
        .sum()
    )
    num_deaths_averted = summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_deaths, comparison='Nothing')).T,
        "median"
    ).iloc[0].unstack().sort_values(by='mean', ascending=True)

    pc_deaths_averted = 100.0 * summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_deaths, comparison='Nothing', scaled=True)).T
    ).iloc[0].unstack().sort_values(by='mean', ascending=True)

    num_dalys = (
        extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked_by_age_and_time",
            custom_generate_series=lambda _df: get_num_dalys_by_cause_label(_df, TARGET_PERIOD),
            do_scaling=True,
            suspended_results_folder=suspended_folder,
            autodiscover=True,
        )
        .pipe(set_param_names_as_column_index_level_0, param_names=param_names)
        .sum()
    )

    num_dalys_averted = summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_dalys, comparison='Nothing')).T
    ).iloc[0].unstack().sort_values(by='mean', ascending=True)

    pc_dalys_averted = 100.0 * summarize(
        pd.DataFrame(
            find_difference_extra_relative_to_comparison(num_dalys, comparison='Nothing', scaled=True)).T
    ).iloc[0].unstack().sort_values(by='mean', ascending=True)

    total_num_death_by_agegrp_and_label = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=lambda _df: get_total_num_death_by_agegrp_and_label(_df, TARGET_PERIOD),
        do_scaling=True,
        suspended_results_folder=suspended_folder,
        autodiscover=True,
    ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)

    total_num_dalys_by_agegrp_and_label = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked_by_age_and_time",
        custom_generate_series=lambda _df: get_total_num_dalys_by_agegrp_and_label(_df, TARGET_PERIOD),
        do_scaling=True,
        suspended_results_folder=suspended_folder,
        autodiscover=True,
    ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)

    counts_of_hsi_by_short_treatment_id = (
        extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="HSI_Event",
            custom_generate_series=lambda _df: get_counts_of_hsi_by_short_treatment_id(_df, TARGET_PERIOD),
            do_scaling=True,
            suspended_results_folder=suspended_folder,
            autodiscover=True,
        )
        .pipe(set_param_names_as_column_index_level_0, param_names=param_names)
        .fillna(0.0)
        .sort_index()
    )

    print("Extracting counts of appointments data...")
    counts_of_appts = (
        extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="HSI_Event",
            custom_generate_series=lambda _df: get_counts_of_appts(_df, TARGET_PERIOD),
            do_scaling=True,
            suspended_results_folder=suspended_folder,
        )
        .pipe(set_param_names_as_column_index_level_0, param_names=param_names)
        .fillna(0.0)
        .sort_index()
    )

    return {
        "total_population_by_year": total_population_by_year,
        "num_deaths": num_deaths,
        "deaths_averted": num_deaths_averted,
        "pc_deaths_averted": pc_deaths_averted,
        "num_dalys": num_dalys,
        "dalys_averted": num_dalys_averted,
        "pc_dalys_averted": pc_dalys_averted,
        "input_costs": input_costs,
        "total_num_death_by_agegrp_and_label": total_num_death_by_agegrp_and_label,
        "total_num_dalys_by_agegrp_and_label": total_num_dalys_by_agegrp_and_label,
        "counts_of_hsi_by_short_treatment_id": counts_of_hsi_by_short_treatment_id,
        "counts_of_appts": counts_of_appts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    parser.add_argument("output_folder", type=Path, nargs="?", default=None)
    args = parser.parse_args()

    out = args.output_folder if args.output_folder is not None else args.results_folder
    results = apply(results_folder=args.results_folder, output_folder=out, resourcefilepath=Path("./resources"))
    with open(args.output_folder / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("Analysis complete! Results saved to results.pkl")
