import argparse
import glob
import os
import zipfile
from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scripts.calibration_analyses.analysis_scripts import plot_legends
from scripts.lcoa_inputs_from_tlo_analyses.results_processing_utils import (
    get_parameter_names_from_scenario_file,
    get_periods_within_target_period,
    format_scenario_name,
    target_period,
)
from scripts.lcoa_inputs_from_tlo_analyses.fig_utils import (
    do_bar_plot_with_ci,
    plot_deaths_by_period_for_cause,
    plot_hsi_counts_by_period_for_draw,
    plot_population_by_year,
)
from tlo import Date

TARGET_PERIOD = (Date(2026, 1, 1), Date(2041, 1, 1))
PERIOD_LENGTH_YEARS_FOR_BAR_PLOTS = 5


def load_results_files(results_files: list[Path]) -> dict[Path, dict]:
    loaded = {}
    for results_file in results_files:
        with open(results_file, "rb") as f:
            loaded[results_file] = pickle.load(f)
    return loaded

def apply(results_files: list[Path], output_folder: Path, resourcefilepath: Path = None):
    """Produce standard plots describing effect of each TREATMENT_ID."""

    def make_graph_file_name(stub):
        filename = stub.replace('*', '_star_').replace(' ', '_').lower()
        return output_folder / f"{filename}.png"

    param_names = get_parameter_names_from_scenario_file()

    period_labels_for_bar_plots = [
        label
        for label, _ in get_periods_within_target_period(
                period_length_years=PERIOD_LENGTH_YEARS_FOR_BAR_PLOTS,
                target_period_tuple=TARGET_PERIOD,
            )
        ]

    target_period_label = target_period(TARGET_PERIOD)

    all_results = load_results_files(results_files)
    results = all_results[results_files[1]]

    counts_of_hsi_in_implementation_period = all_results[results_files[1]]['counts_of_hsi_by_short_treatment_id']

    result_df = pd.DataFrame([
        {'treatment_id_included': draw, 'nonzero_hsis': treatment_id}
        for draw in counts_of_hsi_in_implementation_period.columns.get_level_values(0).unique()
        for treatment_id in ((counts_of_hsi_in_implementation_period[draw] != 0).any(axis=1))[(counts_of_hsi_in_implementation_period[draw] != 0).any(axis=1)].index
    ])
    result_df['treatment_id_included'] = result_df['treatment_id_included'].str.replace('_\\*$', '', regex=True)
    #133 rows here;
    #result_df[result_df['treatment_id_included'] != result_df['nonzero_hsis']]


    # Plot number of HSIs for each draw dropping the aggregate over the entire period
    counts_of_hsi_in_baseline = all_results[results_files[0]]['counts_of_hsi_by_period']
    counts_of_hsi_in_baseline = counts_of_hsi_in_baseline.drop(['2010-2025'], level=1)

    counts_of_hsi_in_implementation_period = all_results[results_files[1]]['counts_of_hsi_by_period']
    counts_of_hsi_in_implementation_period = counts_of_hsi_in_implementation_period.drop(['2026-2041'], level=1)
    result_df_by_period = pd.DataFrame([
        {'treatment_id_included': draw, 'nonzero_hsis': treatment_id, 'period': period}
        for draw in counts_of_hsi_in_implementation_period.columns.get_level_values(0).unique()
        for treatment_id, period in (
            ((counts_of_hsi_in_implementation_period[draw] != 0).any(axis=1))[
                (counts_of_hsi_in_implementation_period[draw] != 0).any(axis=1)
            ].index
        )
    ])
    result_df_by_period['treatment_id_included'] = result_df_by_period['treatment_id_included'].str.replace(
        '_\\*$', '', regex=True
    )

    for param in param_names:
        draw = format_scenario_name(param)
        print(f"Plotting HSI counts for {draw}...")
        name_of_plot = f"Yearly HSI counts for {draw}"
        fig, ax = plot_hsi_counts_by_period_for_draw(
            counts_of_hsi_in_implementation_period,
            draw,
            counts_of_hsi_in_baseline,
        )
        ax.set_title(name_of_plot)
        fig.savefig(make_graph_file_name(name_of_plot))
        plt.close(fig)

    # Plot population growth
    total_population_in_baseline = all_results[results_files[0]]['total_population_by_year']
    total_population_in_implementation = all_results[results_files[1]]['total_population_by_year']
    fig, ax = plot_population_by_year(total_population_in_implementation / 1e6, total_population_in_baseline / 1e6)
    name_of_plot = "Population size by year"
    ax.set_title(name_of_plot)
    ax.set_ylabel("Population size (millions)")
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

    cause_labels = num_deaths_by_cause_label.index.get_level_values("label").unique()
    for cause_label in cause_labels:
        fig, ax = plot_deaths_by_period_for_cause(num_deaths_by_cause_label / 1e3, cause_label=cause_label)
        name_of_plot = f"Deaths Over Time for {cause_label}"
        ax.set_title(name_of_plot)
        ax.set_ylabel("Number of deaths (/1000)")
        fig.savefig(make_graph_file_name(name_of_plot.replace(" ", "_")))
        plt.close(fig)

    # Plot cost of each scenario, with confidence intervals, for the target period


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_files", type=Path, nargs="+")
    parser.add_argument("--output-folder", type=Path, required=True)
    args = parser.parse_args()

    apply(results_files=args.results_files, output_folder=args.output_folder, resourcefilepath=Path("./resources"))

    plot_legends.apply(results_folder=None, output_folder=args.output_folder, resourcefilepath=Path("./resources"))
