import argparse
import glob
import os
import zipfile
from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from scripts.lcoa_inputs_from_tlo_analyses.results_processing_utils import (
    get_parameter_names_from_scenario_file,
    format_scenario_name,
)
from scripts.lcoa_inputs_from_tlo_analyses.fig_utils import (
    make_graph_file_name,
    do_barh_plot_with_ci,
    plot_cadre_time_by_draw_stacked,
    plot_deaths_by_period_for_cause,
    plot_deaths_by_period_for_draw,
    plot_hsi_counts_by_period_for_draw,
    plot_population_by_year,
)


# python src/scripts/lcoa_inputs_from_tlo_analyses/figures_effect_of_treatment_ids.py outputs/generated_outputs/2041-01-01_fullresults.pkl --output_folder=figs2


PERIOD_LENGTH_YEARS_FOR_BAR_PLOTS = 1


def load_results_files(results_files: list[Path]) -> dict[Path, dict]:
    loaded = {}
    for results_file in results_files:
        print(f"Loading results file: {results_file}")
        with open(results_file, "rb") as f:
            loaded[results_file] = pickle.load(f)
    return loaded


def apply(results_files: list[Path], output_folder: Path, resourcefilepath: Path = None):
    """Produce standard plots describing effect of each TREATMENT_ID."""
    print("Starting figure generation for treatment-ID effects.")
    print(f"Output folder: {output_folder}")

    param_names = get_parameter_names_from_scenario_file()
    print(f"Loaded parameter names: {len(param_names)}")

    all_results = load_results_files(results_files)
    primary_results = all_results[results_files[0]]
    print(f"Using primary results from: {results_files[0]}")

    num_deaths_averted = primary_results.get('num_deaths_averted')
    pc_deaths_averted = primary_results.get('pc_deaths_averted')
    dalys_averted = primary_results.get('dalys_averted')
    pc_dalys_averted = primary_results.get('pc_dalys_averted')
    icers = primary_results.get('icers_summarized')
    comparison_metrics_available = all(
        metric is not None
        for metric in (
            num_deaths_averted,
            pc_deaths_averted,
            dalys_averted,
            pc_dalys_averted,
            icers,
        )
    )
    print(f"Comparison metrics available: {comparison_metrics_available}")

    counts_of_hsi_in_implementation_period = primary_results['counts_of_hsi_by_period']
    counts_of_hsi_in_implementation_period = counts_of_hsi_in_implementation_period.drop(['2010-2041'], level=1)
    capacity_used_by_cadre = primary_results.get("capacity_used_by_cadre")

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
        if param == "Nothing":
            continue
        draw = format_scenario_name(param)
        print(f"Plotting yearly HSI counts for draw: {draw}")
        name_of_plot = f"Yearly HSI counts for {draw}"
        # Since all HSIs will be delivered before the service availability switch
        # retain only the treatment id of interest in this period to avoid plot
        # clutter.
        pre_switch_periods = (
            ['2010-2010', '2011-2011', '2012-2012', '2013-2013',
             '2014-2014', '2015-2015', '2016-2016', '2017-2017',
             '2018-2018', '2019-2019', '2020-2020', '2021-2021',
             '2022-2022', '2023-2023', '2024-2024', '2025-2025']
        )
        mask_other_periods = (
            ~counts_of_hsi_in_implementation_period.
            index.
            get_level_values("period").
            isin(pre_switch_periods)
        )
        mask_early_periods = (
            counts_of_hsi_in_implementation_period.index.get_level_values("period").isin(pre_switch_periods) &
            (counts_of_hsi_in_implementation_period.index.get_level_values("appt_type") == draw.replace("_*", ""))
        )
        plot_this = counts_of_hsi_in_implementation_period[mask_other_periods | mask_early_periods]
        fig, ax = plot_hsi_counts_by_period_for_draw(
            plot_this,
            draw,
        )
        ax.set_title(name_of_plot)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.savefig(outfile)
        plt.close(fig)

    print("Plotting capacity used by cadres across draws.")
    fig, ax = plot_cadre_time_by_draw_stacked(capacity_used_by_cadre, stat="central")
    name_of_plot = "Capacity Used by Cadres (2026-2040)"
    ax.set_title(name_of_plot)
    outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
    fig.savefig(outfile)
    plt.close(fig)

    # Plot population growth
    total_population_in_implementation = primary_results['total_population_by_year']
    print("Plotting population size by year.")
    fig, ax = plot_population_by_year(total_population_in_implementation / 1e6)
    name_of_plot = "Population size by year"
    ax.set_title(name_of_plot)
    ax.set_ylabel("Population size (millions)")
    fig.savefig(make_graph_file_name(name_of_plot.replace(" ", "_")))
    plt.close(fig)

    # Plot number of deaths and DALYS by cause for each parameter, with confidence intervals, for the target period
    num_dalys_by_cause_label_implementation = primary_results['dalys'].drop(['2010-2041'], level=1)

    num_deaths_by_cause_label_implementation = primary_results['num_deaths'].drop(['2010-2041'], level=1)
    print("Prepared deaths and DALYs by cause for plotting.")

    for param in param_names:
        draw = format_scenario_name(param)
        print(f"Plotting deaths over time by cause for draw: {draw}")
        fig, ax = plot_deaths_by_period_for_draw(
            num_deaths_by_cause_label_implementation / 1e3,
            draw,
        )
        name_of_plot = f"Deaths Over Time by Cause for {draw}"
        ax.set_title(name_of_plot)
        ax.set_ylabel("Number of deaths (/1000)")
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.savefig(outfile)
        plt.close(fig)

    cause_labels = num_deaths_by_cause_label_implementation.index.get_level_values("label").unique()
    for cause_label in cause_labels:
        print(f"Plotting cause-specific time series for: {cause_label}")
        fig, ax = plot_deaths_by_period_for_cause(
            num_deaths_by_cause_label_implementation / 1e3,
            cause_label=cause_label,
        )
        name_of_plot = f"Deaths Over Time for {cause_label}"
        ax.set_title(name_of_plot)
        ax.set_ylabel("Number of deaths (/1000)")
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.savefig(outfile)
        plt.close(fig)

        fig, ax = plot_deaths_by_period_for_cause(
            num_dalys_by_cause_label_implementation / 1e3,
            cause_label=cause_label,
        )
        name_of_plot = f"DALYs Over Time for {cause_label}"
        ax.set_title(name_of_plot)
        ax.set_ylabel("Number of DALYs (/1000)")
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.savefig(outfile)
        plt.close(fig)

    if comparison_metrics_available:
        print("Plotting comparison metrics: deaths/DALYs averted, percentages, and ICERs.")
        deaths_averted_sorted = (num_deaths_averted.sort_values(by="central", ascending=True) / 1e3)
        fig_height = max(6, min(0.28 * len(deaths_averted_sorted.index) + 4, 18))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        name_of_plot = "Deaths Averted by Each Treatment ID"
        do_barh_plot_with_ci(deaths_averted_sorted, ax)
        ax.set_title(name_of_plot)
        ax.set_xlabel("Number of deaths averted (/1000)")
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print("Saved: Deaths Averted by Each Treatment ID")

        dalys_averted_sorted = (dalys_averted.sort_values(by="central", ascending=True) / 1e3)
        fig_height = max(6, min(0.28 * len(dalys_averted_sorted.index) + 4, 18))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        name_of_plot = "DALYS Averted by Each Treatment ID"
        do_barh_plot_with_ci(dalys_averted_sorted, ax)
        ax.set_title(name_of_plot)
        ax.set_xlabel("DALYs averted (/1000)")
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print("Saved: DALYS Averted by Each Treatment ID")

        pc_deaths_averted_sorted = (pc_deaths_averted.sort_values(by="central", ascending=True))
        fig_height = max(6, min(0.28 * len(pc_deaths_averted_sorted.index) + 4, 18))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        name_of_plot = "Percentage Deaths Averted by Each Treatment ID"
        do_barh_plot_with_ci(pc_deaths_averted_sorted, ax)
        ax.set_title(name_of_plot)
        ax.set_xlabel("Percentage of deaths averted")
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print("Saved: Percentage Deaths Averted by Each Treatment ID")

        pc_dalys_averted_sorted = (pc_dalys_averted.sort_values(by="central", ascending=True))
        fig_height = max(6, min(0.28 * len(pc_dalys_averted_sorted.index) + 4, 18))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        name_of_plot = "Percentage DALYs Averted by Each Treatment ID"
        do_barh_plot_with_ci(pc_dalys_averted_sorted, ax)
        ax.set_title(name_of_plot)
        ax.set_xlabel("Percentage of DALYs averted")
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print("Saved: Percentage DALYs Averted by Each Treatment ID")

        icers_sorted = icers.sort_values(by="central", ascending=True)
        # Do not plot treatment ids with very wide uncertainty
        # CervicalCancer_Screening_Xpert_*              -110.336087   -6.192826  5064.399284
        # BreastCancer_PalliativeCare_*                  -25.104866   -5.740423  2611.046029
        # Hiv_Test_*                                   -7335.183554  248.738016   856.794914

        mask = ~icers_sorted.index.get_level_values("draw").isin(["Hiv_Test_*", "CervicalCancer_Screening_Xpert_*", "BreastCancer_PalliativeCare_*"])
        icers_sorted = icers_sorted[mask]
        fig_height = max(6, min(0.28 * len(icers_sorted.index) + 4, 18))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        name_of_plot = "ICERs for Each Treatment ID"
        do_barh_plot_with_ci(icers_sorted, ax)
        ax.set_title(name_of_plot)
        ax.set_xlabel("ICER (USD per DALY averted)")
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print("Saved: ICERs for Each Treatment ID")

    print("Finished generating figures.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_files", type=Path, nargs="+")
    parser.add_argument("--output_folder", type=Path, required=True)
    args = parser.parse_args()

    apply(results_files=args.results_files, output_folder=args.output_folder, resourcefilepath=Path("./resources"))
