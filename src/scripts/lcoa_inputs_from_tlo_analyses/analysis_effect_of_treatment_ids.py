"""Produce plots to show the impact each set of treatments."""

import argparse
import glob
import os
import zipfile
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tlo import Date
from scripts.calibration_analyses.analysis_scripts import plot_legends
from scripts.lcoa_inputs_from_tlo_analyses.fig_utils import (
    do_bar_plot_with_ci,
    plot_multiindex_dot_with_interval,
)
from scripts.lcoa_inputs_from_tlo_analyses.results_processing_utils import (
    extract_deaths_total,
    format_scenario_name,
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
    unflatten_flattened_multi_index_in_logging,
)

TARGET_PERIOD = (Date(2026, 1, 1), Date(2041, 1, 1))
PERIOD_LENGTH_YEARS_FOR_BAR_PLOTS = 5
results_folder = Path("outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-02-16T154500Z")
# SCALING_FACTOR retrieved from the suspended run in
# outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-02-12T120859Z
SCALING_FACTOR = 58.158436


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard plots describing effect of each TREATMENT_ID."""
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    _, age_grp_lookup = make_age_grp_lookup()
    period_labels_for_bar_plots = [
        label
        for label, _ in get_periods_within_target_period(
            period_length_years=PERIOD_LENGTH_YEARS_FOR_BAR_PLOTS,
            target_period_tuple=TARGET_PERIOD,
        )
    ]
    target_period_label = target_period(TARGET_PERIOD)

    param_names = get_parameter_names_from_scenario_file()

    # Costs calculation
    alternative_discount_rates = [
        {"discount_rate_cost": 0.03, "discount_rate_health": 0, "discounting_scenario": 'WHO-CHOICE (0.03,0)'},
        {"discount_rate_cost": 0.03, "discount_rate_health": 0.03, "discounting_scenario": 'MAIN (0.03,0.03)'}
    ]

    for rates in alternative_discount_rates:
        discount_rate_cost = rates["discount_rate_cost"]
        discount_rate_health = rates["discount_rate_health"]
        input_costs = estimate_input_cost_of_scenarios(
                          results_folder,
                          resourcefilepath,
                          cost_only_used_staff=True,
                          _discount_rate=discount_rate_cost,
                          _metric="median",)




    # Get total population by year
    total_population_by_year = extract_results(
        results_folder,
        module='tlo.methods.demography',
        key='population',
        custom_generate_series=get_total_population_by_year,
        do_scaling=True,
        scaling_factor=SCALING_FACTOR,
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

    for year in [2026, 2031, 2036, 2040]:
        fig, ax = plt.subplots()
        name_of_plot = f"Population size in {year}"
        plot_multiindex_dot_with_interval(total_population_by_year / 1e6, year, ax, 'median')
        ax.set_title(name_of_plot)
        ax.set_xlabel("Treatment included")
        ax.set_ylabel("Population size (millions)")
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(" ", "_")))
        plt.close(fig)

    num_deaths_by_cause_label = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=extract_deaths_total,
            do_scaling=True,
            scaling_factor=SCALING_FACTOR,
            autodiscover=True,
        ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)
    )

    num_dalys_by_cause_label = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked_by_age_and_time",
            custom_generate_series=make_get_num_dalys_by_cause_label_and_period(
                period_length_years=PERIOD_LENGTH_YEARS_FOR_BAR_PLOTS,
                target_period_tuple=TARGET_PERIOD,
            ),
            do_scaling=True,
            scaling_factor=SCALING_FACTOR,
            autodiscover=True,
        ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)
    )

    for param in param_names:
        param_formatted = format_scenario_name(param)

        fig, ax = plt.subplots()
        name_of_plot = f"Deaths With {param_formatted}, {target_period_label}"
        do_bar_plot_with_ci(num_deaths_by_cause_label / 1e3, param_formatted, ax, period_labels_for_bar_plots, target_period_label)
        ax.set_title(name_of_plot)
        ax.set_xlabel("Cause of Death")
        ax.set_ylabel("Number of Deaths (/1000)")
        ax.set_ylim(0, 500)
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(" ", "_")))
        plt.close(fig)

        fig, ax = plt.subplots()
        name_of_plot = f"DALYS With No Services, {target_period_label}"
        do_bar_plot_with_ci(num_dalys_by_cause_label / 1e6, param_formatted, ax, period_labels_for_bar_plots, target_period_label)
        ax.set_title(name_of_plot)
        ax.set_xlabel("Cause of Disability/Death")
        ax.set_ylabel("Number of DALYS (/millions)")
        ax.set_ylim(0, 30)
        ax.set_yticks(np.arange(0, 35, 5))
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(" ", "_")))
        plt.close(fig)

    num_deaths = (
        extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=lambda _df: get_num_deaths_by_cause_label(_df, TARGET_PERIOD),
            do_scaling=True,
            scaling_factor=SCALING_FACTOR,
            autodiscover=True,
        )
        .pipe(set_param_names_as_column_index_level_0, param_names=param_names)
        .sum()
    )

    num_dalys = (
        extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked_by_age_and_time",
            custom_generate_series=lambda _df: get_num_dalys_by_cause_label(_df, TARGET_PERIOD),
            do_scaling=True,
            scaling_factor=SCALING_FACTOR,
            autodiscover=True,
        )
        .pipe(set_param_names_as_column_index_level_0, param_names=param_names)
        .sum()
    )

    total_num_death_by_agegrp_and_label = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=lambda _df: get_total_num_death_by_agegrp_and_label(_df, TARGET_PERIOD),
        do_scaling=True,
        scaling_factor=SCALING_FACTOR,
        autodiscover=True,
    ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)

    total_num_dalys_by_agegrp_and_label = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked_by_age_and_time",
        custom_generate_series=lambda _df: get_total_num_dalys_by_agegrp_and_label(_df, TARGET_PERIOD),
        do_scaling=True,
        scaling_factor=SCALING_FACTOR,
        autodiscover=True,
    ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)

    counts_of_hsi_by_short_treatment_id = (
        extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="HSI_Event",
            custom_generate_series=lambda _df: get_counts_of_hsi_by_short_treatment_id(_df, TARGET_PERIOD),
            do_scaling=True,
            scaling_factor=SCALING_FACTOR,
            autodiscover=True,
        )
        .pipe(set_param_names_as_column_index_level_0, param_names=param_names)
        .fillna(0.0)
        .sort_index()
    )

    mean_num_hsi_by_short_treatment_id = summarize(counts_of_hsi_by_short_treatment_id, only_mean=True)

    for scenario_name, _counts in mean_num_hsi_by_short_treatment_id.T.iterrows():
        _counts_non_zero = _counts[_counts > 0]

        if len(_counts_non_zero):
            fig, ax = plt.subplots()
            name_of_plot = f"HSI Events Occurring, {scenario_name}, {target_period_label}"
            squarify_neat(
                sizes=_counts_non_zero.values,
                label=_counts_non_zero.index,
                colormap=get_color_short_treatment_id,
                alpha=1,
                pad=True,
                ax=ax,
                text_kwargs={"color": "black", "size": 8},
            )
            ax.set_axis_off()
            ax.set_title(name_of_plot, {"size": 12, "color": "black"})
            fig.savefig(make_graph_file_name(name_of_plot.replace(" ", "_")))
            plt.close(fig)

    counts_of_appts = (
        extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="HSI_Event",
            custom_generate_series=lambda _df: get_counts_of_appts(_df, TARGET_PERIOD),
            do_scaling=True,
            scaling_factor=SCALING_FACTOR,
        )
        .pipe(set_param_names_as_column_index_level_0, param_names=param_names)
        .fillna(0.0)
        .sort_index()
    )


    return {
        "num_deaths": num_deaths,
        "num_dalys": num_dalys,
        "total_num_death_by_agegrp_and_label": total_num_death_by_agegrp_and_label,
        "total_num_dalys_by_agegrp_and_label": total_num_dalys_by_agegrp_and_label,
        "counts_of_hsi_by_short_treatment_id": counts_of_hsi_by_short_treatment_id,
        "counts_of_appts": counts_of_appts,
        "age_grp_lookup": age_grp_lookup,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    parser.add_argument("output_folder", type=Path, nargs="?", default=None)
    args = parser.parse_args()

    out = args.output_folder if args.output_folder is not None else args.results_folder
    apply(results_folder=args.results_folder, output_folder=out, resourcefilepath=Path("./resources"))

    plot_legends.apply(results_folder=None, output_folder=out, resourcefilepath=Path("./resources"))

    with zipfile.ZipFile(out / f"images_{out.parts[-1]}.zip", mode="w") as archive:
        for filename in sorted(glob.glob(str(out / "*.png"))):
            archive.write(filename, os.path.basename(filename))
