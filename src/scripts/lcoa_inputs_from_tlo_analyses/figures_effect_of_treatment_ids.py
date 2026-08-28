import argparse
import glob
import os
import zipfile
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tlo.analysis.utils import (
    CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP,
    SHORT_TREATMENT_ID_TO_COLOR_MAP,
)

from scripts.lcoa_inputs_from_tlo_analyses.results_processing_utils import (
    get_parameter_names_from_scenario_file,
    format_scenario_name,
)
from scripts.lcoa_inputs_from_tlo_analyses.fig_utils import (
    make_graph_file_name,
    do_barh_plot_with_ci,
    plot_deaths_by_period_for_cause,
    plot_deaths_by_period_for_draw,
    plot_hsi_counts_by_period_for_draw,
    plot_population_by_year,
    plot_capacity_used_by_cadre_and_level_over_time_for_draw,
    plot_cost_by_cadre_over_time_for_draw,
    plot_treatment_id_include_exclude_table,
)


# python src/scripts/lcoa_inputs_from_tlo_analyses/figures_effect_of_treatment_ids.py outputs/generated_outputs/2040-12-31_fullresults.pkl --output_folder=figs-with-discounted-dalys
# python src/scripts/lcoa_inputs_from_tlo_analyses/figures_effect_of_treatment_ids.py outputs/generated_outputs/2040-12-31_fullresults.pkl --output_folder=figs10runs

PERIOD_LENGTH_YEARS_FOR_BAR_PLOTS = 1

def load_results_files(results_files: list[Path]) -> dict[Path, dict]:
    loaded = {}
    for results_file in results_files:
        print(f"Loading results file: {results_file}")
        with open(results_file, "rb") as f:
            loaded[results_file] = pickle.load(f)
    return loaded


def plot_dalys_by_cause_label_stacked_by_draw(
    _df: pd.DataFrame,
    draw_labels: list[str] | None = None,
    plot_stat: str = "central",
):
    """Plot stacked DALYs by cause label for each draw."""
    if not isinstance(_df.index, pd.MultiIndex) or _df.index.nlevels != 2:
        raise ValueError("_df index must be a 2-level MultiIndex with levels for label and period.")
    if not isinstance(_df.columns, pd.MultiIndex) or _df.columns.nlevels != 2:
        raise ValueError("_df columns must be a 2-level MultiIndex with levels for draw and stat.")

    label_level_name = "label" if "label" in _df.index.names else _df.index.names[0]
    draw_level_name = "draw" if "draw" in _df.columns.names else _df.columns.names[0]
    stat_level_name = "stat" if "stat" in _df.columns.names else _df.columns.names[1]

    available_stats = pd.Index(_df.columns.get_level_values(stat_level_name).unique())
    if plot_stat not in available_stats:
        raise ValueError(f"Statistic '{plot_stat}' not found. Available stats: {available_stats.tolist()}")

    plot_df = _df.xs(plot_stat, axis=1, level=stat_level_name)
    plot_df = plot_df.groupby(level=label_level_name).sum().T.fillna(0.0)
    plot_df.index.name = draw_level_name

    ordered_causes = [
        cause_label for cause_label in CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP.keys()
        if cause_label in plot_df.columns
    ]
    unordered_causes = sorted(
        cause_label for cause_label in plot_df.columns if cause_label not in CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP
    )
    plot_df = plot_df.loc[:, ordered_causes + unordered_causes]

    if draw_labels is not None:
        available_draws = pd.Index(plot_df.index)
        ordered_draws = [draw for draw in draw_labels if draw in available_draws]
        plot_df = plot_df.reindex(ordered_draws)

    if plot_df.empty:
        raise ValueError("No plottable DALY data remain after reshaping by draw and cause label.")

    fig_width = max(10, min(0.8 * len(plot_df.index) + 4, 24))
    fig_height = max(6, min(0.35 * len(plot_df.index) + 3, 14))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = np.arange(len(plot_df.index))
    bottom = np.zeros(len(plot_df.index), dtype=float)
    for cause_label in plot_df.columns:
        values = plot_df[cause_label].to_numpy(dtype=float)
        if not np.any(values):
            continue
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP.get(cause_label, "grey"),
            label=str(cause_label),
            width=0.8,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels([str(draw) for draw in plot_df.index], rotation=45, ha="right")
    ax.set_xlabel("Draw label")
    ax.set_ylabel("DALYs")
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        title="Cause label",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )
    fig.tight_layout()
    return fig, ax


def apply(
    results_files: list[Path],
    output_folder: Path,
    resourcefilepath: Path = None,
    include_exclude_demo: bool = False,
):
    """Produce standard plots describing effect of each TREATMENT_ID."""

    print(f"Output folder: {output_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)

    all_results = load_results_files(results_files)
    primary_results = all_results[results_files[0]]
    print(f"Using primary results from: {results_files[0]}")

    num_deaths_averted = primary_results.get('num_deaths_averted')
    pc_deaths_averted = primary_results.get('pc_deaths_averted')
    dalys_averted = primary_results.get('dalys_averted')
    pc_dalys_averted = primary_results.get('pc_dalys_averted')
    net_health = primary_results.get('net_health')
    incremental_scenario_cost = primary_results.get('incremental_scenario_cost')
    undiscounted_dalys = primary_results.get('dalys')
    discounted_dalys = primary_results.get('discounted_dalys')
    dalys_and_costs_from_lcoa = primary_results.get('dalys_and_costs_from_lcoa')
    annual_cost_by_cadre = primary_results.get('annual_cost_by_cadre')
    counts_of_hsi = primary_results['counts_of_hsi_by_period']
    annual_capacity_used_by_cadre_and_level = primary_results.get("annual_capacity_used_by_cadre_and_level")

    comparison_metrics_available = all(
        metric is not None
        for metric in (
            num_deaths_averted,
            pc_deaths_averted,
            dalys_averted,
            pc_dalys_averted,
            incremental_scenario_cost
        )
    )
    print(f"Comparison metrics available: {comparison_metrics_available}")
    # Get the list of draws directly from the results and not from scenario files
    # as some draws have been excluded at the previous step.
    param_names = primary_results['total_population_by_year'].columns.get_level_values('draw').unique()
    print(f"Loaded parameter names: {len(param_names)}")

    if isinstance(undiscounted_dalys, pd.DataFrame) and isinstance(discounted_dalys, pd.DataFrame):
        undiscounted_totals = undiscounted_dalys.sum(axis=0).unstack("stat")
        discounted_totals = discounted_dalys.sum(axis=0).unstack("stat")
        daly_totals_order = undiscounted_totals["central"].sort_values().index
        undiscounted_totals = undiscounted_totals.reindex(daly_totals_order) / 1e6
        discounted_totals = discounted_totals.reindex(daly_totals_order) / 1e6

        colors = [
            SHORT_TREATMENT_ID_TO_COLOR_MAP.get(str(draw).split("_")[0] + "*", "grey")
            for draw in daly_totals_order
        ]
        y_positions = np.arange(len(daly_totals_order))
        fig_height = max(6, min(0.28 * len(daly_totals_order) + 4, 24))
        fig, ax = plt.subplots(figsize=(12, fig_height))

        ax.barh(
            y_positions,
            undiscounted_totals["central"],
            color=colors,
            alpha=0.3,
            label="Undiscounted DALYs",
        )
        ax.barh(
            y_positions,
            discounted_totals["central"],
            color=colors,
            alpha=0.9,
            label="Discounted DALYs",
        )

        for position, color, (_, undiscounted), (_, discounted) in zip(
            y_positions,
            colors,
            undiscounted_totals.iterrows(),
            discounted_totals.iterrows(),
        ):
            for values, alpha in ((undiscounted, 0.4), (discounted, 1.0)):
                ax.errorbar(
                    values["central"],
                    position,
                    xerr=[[values["central"] - values["lower"]], [values["upper"] - values["central"]]],
                    fmt="none",
                    ecolor=color,
                    elinewidth=1,
                    capsize=2,
                    alpha=alpha,
                    zorder=3,
                )

        name_of_plot = "Discounted and Undiscounted DALYs by Treatment ID"
        ax.set_yticks(y_positions)
        ax.set_yticklabels(daly_totals_order)
        ax.set_xlabel("Total DALYs (millions)")
        ax.set_ylabel("Treatment ID")
        ax.set_title(name_of_plot)
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()
        fig.tight_layout()
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.savefig(outfile)
        plt.close(fig)
        print(f"Saved: {name_of_plot}")

    result_rows = []
    for draw in counts_of_hsi.columns.get_level_values(0).unique():
        # Keep only rows where this draw has at least one non-zero HSI count.
        non_zero_rows = (counts_of_hsi[draw] != 0).any(axis=1)
        for treatment_id, period in counts_of_hsi[draw].loc[non_zero_rows].index:
            result_rows.append(
                {
                    'treatment_id_included': draw,
                    'nonzero_hsis': treatment_id,
                    'period': period,
                }
            )

    result_df_by_period = pd.DataFrame(result_rows)
    result_df_by_period['treatment_id_included'] = result_df_by_period['treatment_id_included'].str.replace(
        '_\\*$', '', regex=True
    )

    for param in param_names:
        if param == "Nothing":
            continue
        draw = param
        print(f"Plotting yearly HSI counts for draw: {draw}")
        name_of_plot = f"Yearly HSI counts for {draw}"
        # Since all HSIs will be delivered before the service availability switch
        # retain only the treatment id of interest in this period to avoid plot
        # clutter.
        pre_switch_periods = [str(i) for i in list(range(2010, 2025))]
        # Filter rows to retain those in implementation period only
        mask_other_periods = (
            ~counts_of_hsi.
            index.
            get_level_values("period").
            isin(pre_switch_periods) &
            (counts_of_hsi > 0).any(axis=1)
        )
        # In the pre-implentation period only retain the treatment id of interest to avoid plot clutter
        mask_early_periods = (
            counts_of_hsi.index.get_level_values("period").isin(pre_switch_periods) &
            (counts_of_hsi.index.get_level_values("appt_type") == draw.replace("_*", ""))
        )
        plot_this = counts_of_hsi[mask_other_periods | mask_early_periods]
        fig, ax = plot_hsi_counts_by_period_for_draw(
            plot_this,
            draw,
        )
        ax.set_title(name_of_plot)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.savefig(outfile)
        plt.close(fig)

    if annual_capacity_used_by_cadre_and_level is not None:
        print("Plotting capacity used by cadre and facility level over time (one figure per treatment ID).")
        for param in param_names:
            if param == "Nothing":
                continue
            draw = param
            try:
                name_of_plot = f"Capacity Used by Cadre and Facility Level Over Time for {draw}"
                fig, ax = plot_capacity_used_by_cadre_and_level_over_time_for_draw(
                    annual_capacity_used_by_cadre_and_level,
                    draw,
                    title=name_of_plot,
                )
            except ValueError as exc:
                print(f"Skipping capacity-by-level plot for draw '{draw}': {exc}")
                continue

            outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
            fig.savefig(outfile)
            plt.close(fig)

    if annual_cost_by_cadre is not None:
        print("Plotting annual costs by cadre and over time (one figure per treatment ID).")
        for param in param_names:
            print(f"### {param}")
            if param == "Nothing":
                continue
            draw = format_scenario_name(param)
            try:
                name_of_plot = f"Cost by Cadre Over Time for {draw}"
                fig, ax = plot_cost_by_cadre_over_time_for_draw(
                    annual_cost_by_cadre,
                    draw,
                    title=name_of_plot,
                )
            except ValueError as exc:
                print(f"Skipping capacity-by-level plot for draw '{draw}': {exc}")
                continue

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
    outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
    fig.savefig(outfile)
    plt.close(fig)

    # Plot number of deaths and DALYS by cause for each parameter, with
    # confidence intervals, for the target period
    num_dalys_by_cause_label_implementation = primary_results['dalys']

    num_deaths_by_cause_label_implementation = primary_results['num_deaths']
    print("Prepared deaths and DALYs by cause for plotting.")

    daly_draw_labels = param_names
    print("Plotting stacked DALYs by cause label for each draw.")
    fig, ax = plot_dalys_by_cause_label_stacked_by_draw(
        num_dalys_by_cause_label_implementation,
        draw_labels=daly_draw_labels,
    )
    name_of_plot = "DALYs by Cause Label for Each Draw"
    ax.set_title(name_of_plot)
    outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print("Saved: DALYs by Cause Label for Each Draw")

    for param in param_names:
        draw = param
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
        print("Plotting comparison metrics: deaths/DALYs averted, percentages, and net health.")
        dalys_averted_sorted = (dalys_averted.sort_values(by="central", ascending=True) / 1e3)
        dalys_order = dalys_averted_sorted.index
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

        deaths_averted_sorted = (num_deaths_averted / 1e3).reindex(dalys_order)
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

        nh_sorted = net_health.sort_values(by="central", ascending=True)
        nh_sorted = nh_sorted.reindex(dalys_order.intersection(nh_sorted.index))
        fig_height = max(6, min(0.28 * len(nh_sorted.index) + 4, 18))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        name_of_plot = "Net Health for Each Treatment ID"
        do_barh_plot_with_ci(nh_sorted / 1000, ax)
        ax.set_title(name_of_plot)
        ax.set_xlabel("Net Health (/1000)")
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print("Saved: Net health for Each Treatment ID")

        incremental_cost_sorted = incremental_scenario_cost.reindex(dalys_order)
        fig_height = max(6, min(0.28 * len(incremental_cost_sorted.index) + 4, 18))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        name_of_plot = "Incremental Cost for Each Treatment ID"
        do_barh_plot_with_ci(incremental_cost_sorted, ax)
        ax.set_title(name_of_plot)
        ax.set_xlabel("Incremental cost (USD)")
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print("Saved: Incremental Cost for Each Treatment ID")

        facet_order = (
            dalys_order
            .intersection(incremental_cost_sorted.dropna().index)
            .intersection(nh_sorted.dropna().index)
        )
        dalys_facet = dalys_averted_sorted.reindex(facet_order)
        costs_facet = incremental_cost_sorted.reindex(facet_order)
        nh_facet = nh_sorted.reindex(facet_order)

        fig_height = max(6, min(0.28 * len(facet_order) + 4, 18))
        fig, axes = plt.subplots(1, 3, figsize=(20, fig_height), sharey=True)
        name_of_plot = "DALYs, Incremental Cost, and Net health by Treatment ID"

        do_barh_plot_with_ci(dalys_facet, axes[0])
        axes[0].set_title("DALYs")
        axes[0].set_xlabel("DALYs averted (/1000)")

        do_barh_plot_with_ci(costs_facet, axes[1])
        axes[1].set_title("Costs")
        axes[1].set_xlabel("Incremental cost (USD)")

        do_barh_plot_with_ci(nh_facet / 1000, axes[2])
        axes[2].set_title("Net Health")
        axes[2].set_xlabel("Net Health (/1000)")

        if isinstance(dalys_and_costs_from_lcoa, pd.DataFrame):
            lcoa_overlay = (
                dalys_and_costs_from_lcoa[["treatment_id", "overall_dalys", "overall_costs", "net_health"]]
                .dropna(subset=["treatment_id"])
                .drop_duplicates(subset=["treatment_id"], keep="first")
                .set_index("treatment_id")
            )
            facet_overlay = pd.DataFrame({"draw": facet_order})
            facet_overlay["treatment_id"] = facet_overlay["draw"].str.replace(r"_\*$", "", regex=True)
            facet_overlay = facet_overlay.join(lcoa_overlay, on="treatment_id")
            facet_overlay["short_treatment_id"] = facet_overlay["draw"].str.split("_").str[0] + "*"
            facet_overlay["color"] = (
                facet_overlay["short_treatment_id"].map(SHORT_TREATMENT_ID_TO_COLOR_MAP).fillna("grey")
            )
            facet_overlay["tlo_dalys"] = (dalys_facet["central"] * 1e3).to_numpy()
            facet_overlay["tlo_costs"] = costs_facet["central"].to_numpy()
            facet_overlay["tlo_net_health"] = nh_facet["central"].to_numpy()

            daly_overlay = facet_overlay["overall_dalys"].notna()
            if daly_overlay.any():
                # DALY bars are plotted as /1000, so convert overlay values to the same units.
                axes[0].scatter(
                    facet_overlay.loc[daly_overlay, "overall_dalys"] / 1e3,
                    facet_overlay.index[daly_overlay],
                    c="black",
                    s=16,
                    zorder=10,
                )

            cost_overlay = facet_overlay["overall_costs"].notna()
            if cost_overlay.any():
                axes[1].scatter(
                    facet_overlay.loc[cost_overlay, "overall_costs"],
                    facet_overlay.index[cost_overlay],
                    c="black",
                    s=16,
                    zorder=10,
                )

            nh_overlay = facet_overlay["net_health"].notna()
            if nh_overlay.any():
                axes[2].scatter(
                    facet_overlay.loc[nh_overlay, "net_health"] / 1e3,
                    facet_overlay.index[nh_overlay],
                    c="black",
                    s=16,
                    zorder=10,
                )

            scatter_metrics = (
                ("DALYs Averted", "overall_dalys", "tlo_dalys", 1e3, "DALYs averted (/1000)"),
                ("Incremental Cost", "overall_costs", "tlo_costs", 1.0, "Incremental cost (USD)"),
                ("Net Health", "net_health", "tlo_net_health", 1e3, "Net health (/1000)"),
            )
            for metric_name, lcoa_column, tlo_column, scale, axis_label in scatter_metrics:
                scatter_data = facet_overlay.dropna(subset=[lcoa_column, tlo_column])
                if scatter_data.empty:
                    continue

                x = scatter_data[lcoa_column] / scale
                y = scatter_data[tlo_column] / scale
                positive = x.gt(0) & y.gt(0)
                excluded_count = int((~positive).sum())
                scatter_data = scatter_data.loc[positive]
                x = x.loc[positive]
                y = y.loc[positive]
                if scatter_data.empty:
                    print(f"Skipped {metric_name} scatter plot: no positive LCOA/TLO pairs")
                    continue

                fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8))
                ax_scatter.scatter(x, y, c=scatter_data["color"], s=30)
                equality_min = min(x.min(), y.min())
                equality_max = max(x.max(), y.max())
                ax_scatter.plot(
                    [equality_min, equality_max],
                    [equality_min, equality_max],
                    color="black",
                    linestyle="--",
                    linewidth=1,
                )
                ax_scatter.set_xscale("log")
                ax_scatter.set_yscale("log")
                name_of_scatter_plot = f"LCOA vs TLO-Derived {metric_name}"
                ax_scatter.set_title(name_of_scatter_plot)
                ax_scatter.set_xlabel(f"LCOA inputs: {axis_label}")
                ax_scatter.set_ylabel(f"TLO-derived inputs: {axis_label}")
                ax_scatter.grid(alpha=0.3)
                ax_scatter.spines["top"].set_visible(False)
                ax_scatter.spines["right"].set_visible(False)
                fig_scatter.tight_layout()
                scatter_outfile = os.path.join(
                    output_folder,
                    make_graph_file_name(name_of_scatter_plot),
                )
                fig_scatter.savefig(scatter_outfile)
                plt.close(fig_scatter)
                print(f"Saved: {name_of_scatter_plot}")
                if excluded_count:
                    print(f"Excluded {excluded_count} non-positive pairs from {name_of_scatter_plot}")

        for ax in axes:
            ax.grid(axis="x")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].set_ylabel("Treatment ID")
        fig.suptitle(name_of_plot, y=1.02)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print("Saved: DALYs, Incremental Cost, and Net Health by Treatment ID")

        facet_order_without_prep = facet_order.drop("Hiv_Prevention_Prep_*", errors="ignore")
        dalys_facet_without_prep = dalys_facet.reindex(facet_order_without_prep)
        costs_facet_without_prep = costs_facet.reindex(facet_order_without_prep)
        nh_facet_without_prep = nh_facet.reindex(facet_order_without_prep)

        fig_height = max(6, min(0.28 * len(facet_order_without_prep) + 4, 18))
        fig, axes = plt.subplots(1, 3, figsize=(20, fig_height), sharey=True)
        name_of_plot = "DALYs, Incremental Cost, and Net Health Excluding Hiv Prevention Prep"

        do_barh_plot_with_ci(dalys_facet_without_prep, axes[0])
        axes[0].set_title("DALYs")
        axes[0].set_xlabel("DALYs averted (/1000)")

        do_barh_plot_with_ci(costs_facet_without_prep, axes[1])
        axes[1].set_title("Costs")
        axes[1].set_xlabel("Incremental cost (USD)")

        do_barh_plot_with_ci(nh_facet_without_prep / 1000, axes[2])
        axes[2].set_title("Net Health")
        axes[2].set_xlabel("Net Health (/1000)")

        if isinstance(dalys_and_costs_from_lcoa, pd.DataFrame):
            facet_overlay_without_prep = pd.DataFrame({"draw": facet_order_without_prep})
            facet_overlay_without_prep["treatment_id"] = facet_overlay_without_prep["draw"].str.replace(
                r"_\*$", "", regex=True
            )
            facet_overlay_without_prep = facet_overlay_without_prep.join(lcoa_overlay, on="treatment_id")

            overlay_specs = (
                (axes[0], "overall_dalys", 1e3),
                (axes[1], "overall_costs", 1.0),
                (axes[2], "net_health", 1e3),
            )
            for overlay_ax, overlay_column, scale in overlay_specs:
                has_overlay = facet_overlay_without_prep[overlay_column].notna()
                if has_overlay.any():
                    overlay_ax.scatter(
                        facet_overlay_without_prep.loc[has_overlay, overlay_column] / scale,
                        facet_overlay_without_prep.index[has_overlay],
                        c="black",
                        s=16,
                        zorder=10,
                    )

        for ax in axes:
            ax.grid(axis="x")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].set_ylabel("Treatment ID")
        fig.suptitle(name_of_plot, y=1.02)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print(f"Saved: {name_of_plot}")

    print("Finished generating figures.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_files", type=Path, nargs="*")
    parser.add_argument("--output_folder", type=Path, required=True)

    args = parser.parse_args()

    if args.results_files:
        apply(
            results_files=args.results_files,
            output_folder=args.output_folder,
            resourcefilepath=Path("./resources")
        )
