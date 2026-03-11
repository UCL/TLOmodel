import argparse
import glob
import os
import zipfile
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scripts.calibration_analyses.analysis_scripts import plot_legends
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
from scripts.lcoa_inputs_from_tlo_analyses.fig_utils import (
    do_bar_plot_with_ci,
    plot_multiindex_dot_with_interval,
    plot_appointment_counts_by_period_for_draw,
    plot_appointment_counts_heatmap,
)
from tlo.analysis.utils import (
    compute_summary_statistics,
)
from tlo import Date

TARGET_PERIOD = (Date(2026, 1, 1), Date(2041, 1, 1))
PERIOD_LENGTH_YEARS_FOR_BAR_PLOTS = 5

def apply(results_file: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard plots describing effect of each TREATMENT_ID."""

    param_names = get_parameter_names_from_scenario_file()
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731
    period_labels_for_bar_plots = [
        label
        for label, _ in get_periods_within_target_period(
                period_length_years=PERIOD_LENGTH_YEARS_FOR_BAR_PLOTS,
                target_period_tuple=TARGET_PERIOD,
            )
        ]
    appointment_period_labels = [
        label
        for label, _ in get_periods_within_target_period(
            period_length_years=1,
            target_period_tuple=TARGET_PERIOD,
        )
    ]

    target_period_label = target_period(TARGET_PERIOD)

    with open(results_file, "rb") as f:
        results = pickle.load(f)

    # Plot number of appointments for each draw
    counts_of_appts = results['counts_of_appts']
    fig, ax = plot_appointment_counts_heatmap(counts_of_appts)
    fig.savefig(make_graph_file_name("appointment_counts_heatmap"))
    plt.close(fig)

    counts_of_appts_by_period = results["counts_of_appts_by_period"]
    for param in param_names:
        draw = format_scenario_name(param)
        name_of_plot = f"Yearly appointment counts for {draw}"
        fig, ax = plot_appointment_counts_by_period_for_draw(
            counts_of_appts_by_period,
            draw=draw,
            period_labels=appointment_period_labels,
        )
        ax.set_title(name_of_plot)
        fig.savefig(make_graph_file_name(name_of_plot))
        plt.close(fig)

    # Plot population growth
    total_population_by_year = results['total_population_by_year']
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

    # Plot number of deaths and DALYS by cause for each parameter, with confidence intervals, for the target period
    num_deaths_by_cause_label = results['num_deaths']
    deaths_averted = results['deaths_averted']
    pc_deaths_averted = results['pc_deaths_averted']

    num_dalys_by_cause_label = results['num_dalys']
    dalys_averted = results['dalys_averted']
    pc_dalys_averted = results['pc_dalys_averted']

    for param in param_names:
        param_formatted = format_scenario_name(param)
        print(f"Plotting for {param_formatted}...")
        fig, ax = plt.subplots()
        name_of_plot = f"Deaths With {param_formatted}, {target_period_label}"
        do_bar_plot_with_ci(num_deaths_by_cause_label / 1e3, param_formatted, ax, period_labels_for_bar_plots, target_period_label)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_title(name_of_plot)
        ax.set_xlabel("Cause of Death")
        ax.set_ylabel("Number of Deaths (/1000)")
        #ax.set_ylim(0, 500)
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(" ", "_")))
        plt.close(fig)

        fig, ax = plt.subplots()
        name_of_plot = f"DALYS With {param_formatted}, {target_period_label}"
        do_bar_plot_with_ci(num_dalys_by_cause_label / 1e6, param_formatted, ax, period_labels_for_bar_plots, target_period_label)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_title(name_of_plot)
        ax.set_xlabel("Cause of Disability/Death")
        ax.set_ylabel("Number of DALYS (/millions)")
        #ax.set_ylim(0, 30)
        ##ax.set_yticks(np.arange(0, 35, 5))
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(" ", "_")))
        plt.close(fig)

    # Plot cost of each scenario, with confidence intervals, for the target period


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=Path)
    parser.add_argument("output_folder", type=Path, nargs="?", default=None)
    args = parser.parse_args()

    apply(results_file=args.results_file, output_folder=args.output_folder, resourcefilepath=Path("./resources"))

    plot_legends.apply(results_folder=None, output_folder=args.output_folder, resourcefilepath=Path("./resources"))
