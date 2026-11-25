# python src/scripts/lcoa_inputs_from_tlo_analyses/figures_effect_of_hbps.py outputs/generated_outputs/2040-12-31_hbp_fullresults.pkl --output_folder=figs-hbps
import argparse
import os
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from scripts.lcoa_inputs_from_tlo_analyses.fig_utils import (
    make_graph_file_name,
    plot_capacity_used_by_cadre_and_level_over_time_for_draw,
    plot_cost_by_cadre_over_time_for_draw,
    plot_deaths_by_period_for_cause,
    plot_deaths_by_period_for_draw,
    plot_hsi_counts_by_period_for_draw,
    plot_population_by_year,
)
from tlo.analysis.utils import CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP

ANALYSIS_UNIT_LABEL = "Health Benefit Package"


def plot_hbp_overlap_with_fisher_test(overlap):
    """Plot the analysed package memberships and Fisher test."""
    set_labels = tuple(overlap["members"])
    first, second = overlap["members"].values()
    counts = overlap["counts"]
    odds_ratio = overlap["odds_ratio"]
    p_value = overlap["p_value"]

    def format_treatment_ids(treatment_ids):
        return "\n".join((treatment_id.removesuffix("_*") for treatment_id in sorted(treatment_ids)))

    fig, ax = plt.subplots(figsize=(15, 9))
    left_color, right_color = ("#4C78A8", "#F58518")
    ax.add_patch(Circle((0.39, 0.55), 0.34, color=left_color, alpha=0.45))
    ax.add_patch(Circle((0.71, 0.55), 0.34, color=right_color, alpha=0.45))
    ax.text(0.25, 0.55, format_treatment_ids(first - second), ha="center", va="center", fontsize=7)
    ax.text(0.55, 0.55, format_treatment_ids(first & second), ha="center", va="center", fontsize=7, fontweight="bold")
    ax.text(0.85, 0.55, format_treatment_ids(second - first), ha="center", va="center", fontsize=7)
    ax.text(0.27, 0.88, set_labels[0], ha="center", color=left_color, fontsize=11)
    ax.text(0.83, 0.88, set_labels[1], ha="center", color=right_color, fontsize=11)
    odds_ratio_text = "∞" if np.isposinf(odds_ratio) else f"{odds_ratio:.2f}"
    p_value_text = "<0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    ax.text(
        0.55,
        0.1,
        f"Fisher's exact test (two-sided): odds ratio = {odds_ratio_text}, p {p_value_text}\nNeither package: {counts['neither']} of {overlap['universe_size']} model-defined treatment IDs",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.set(xlim=(0, 1.1), ylim=(0, 1), aspect="equal")
    ax.axis("off")
    fig.tight_layout()
    return (fig, ax)


def get_draw_color_map(draw_labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    """Return a stable categorical color for each HBP draw label."""
    palettes = (
        list(plt.get_cmap("tab10").colors)
        + list(plt.get_cmap("Set2").colors)
        + list(plt.get_cmap("Dark2").colors)
        + list(plt.get_cmap("tab20").colors)
    )
    return {draw: palettes[i % len(palettes)] for i, draw in enumerate(draw_labels)}


def recolor_lines_by_draw(ax, draw_color_map: dict[str, tuple[float, float, float, float]]):
    """Apply HBP draw colors to line plots produced by shared draw-agnostic helpers."""
    for line in ax.get_lines():
        label = line.get_label()
        if label in draw_color_map:
            line.set_color(draw_color_map[label])
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            title=ANALYSIS_UNIT_LABEL,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            title_fontsize=9,
            frameon=True,
        )


def get_draw_linestyle_map(draw_labels: list[str]) -> dict[str, str]:
    """Return a stable linetype for each HBP draw label."""
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    return {draw: linestyles[i % len(linestyles)] for i, draw in enumerate(draw_labels)}


def _period_start_year(period_label) -> int:
    return int(str(period_label).split("-")[0])


def _get_periods_starting_from_2026(period_labels) -> tuple[list, list[str]]:
    ordered_period_labels = sorted(
        [period for period in pd.Index(period_labels).unique() if _period_start_year(period) >= 2026],
        key=lambda period: (_period_start_year(period), str(period)),
    )
    display_period_labels = [
        str(_period_start_year(period)) if str(period).endswith(f"-{_period_start_year(period)}") else str(period)
        for period in ordered_period_labels
    ]
    return (ordered_period_labels, display_period_labels)


def plot_population_by_year_for_hbp_draws(
    _df: pd.DataFrame, draw_color_map: dict[str, tuple[float, float, float, float]]
):
    """Use the shared population plot, then recolor draw lines for HBP labels."""
    fig, ax = plot_population_by_year(_df)
    recolor_lines_by_draw(ax, draw_color_map)
    return (fig, ax)


def plot_dalys_by_cause_label_over_time_for_hbp_draws(
    _df: pd.DataFrame, draw_labels: list[str], draw_linestyle_map: dict[str, str], plot_stat: str = "central"
):
    """Plot DALYs over time faceted by cause, with HBP draw linetype."""
    if not isinstance(_df.index, pd.MultiIndex) or _df.index.nlevels != 2:
        raise ValueError("_df index must be a 2-level MultiIndex with levels for label and period.")
    else:
        if not isinstance(_df.columns, pd.MultiIndex) or _df.columns.nlevels != 2:
            raise ValueError("_df columns must be a 2-level MultiIndex with levels for draw and stat.")
        else:
            label_level_name = "label" if "label" in _df.index.names else _df.index.names[0]
            period_level_name = "period" if "period" in _df.index.names else _df.index.names[1]
            stat_level_name = "stat" if "stat" in _df.columns.names else _df.columns.names[1]
            available_stats = pd.Index(_df.columns.get_level_values(stat_level_name).unique())
            if plot_stat not in available_stats:
                raise ValueError(f"Statistic '{plot_stat}' not found. Available stats: {available_stats.tolist()}")
            else:
                has_uncertainty = {"lower", "upper"}.issubset(set(available_stats))
                available_draws = pd.Index(_df.columns.get_level_values(0).unique())
                ordered_draws = [draw for draw in draw_labels if draw in available_draws]
                if not ordered_draws:
                    raise ValueError(f"No requested draws found. Available draws: {available_draws.tolist()}")
                else:
                    ordered_period_labels, display_period_labels = _get_periods_starting_from_2026(
                        _df.index.get_level_values(period_level_name)
                    )
                    if not ordered_period_labels:
                        raise ValueError("No periods starting in 2026 or later found.")
                    else:
                        plot_df = _df.xs(plot_stat, axis=1, level=stat_level_name)
                        lower_df = _df.xs("lower", axis=1, level=stat_level_name) if has_uncertainty else None
                        upper_df = _df.xs("upper", axis=1, level=stat_level_name) if has_uncertainty else None
                        ordered_causes = [
                            cause_label
                            for cause_label in CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP.keys()
                            if cause_label in plot_df.index.get_level_values(label_level_name)
                        ]
                        unordered_causes = sorted(
                            (
                                cause_label
                                for cause_label in plot_df.index.get_level_values(label_level_name).unique()
                                if cause_label not in CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP
                            )
                        )
                        cause_labels = ordered_causes + unordered_causes
                        n_causes = len(cause_labels)
                        ncols = 3
                        nrows = int(np.ceil(n_causes / ncols))
                        fig_width = max(13, min(1.1 * len(ordered_period_labels) * ncols, 22))
                        fig_height = max(4 * nrows, 6)
                        fig, axes = plt.subplots(
                            nrows, ncols, figsize=(fig_width, fig_height), sharex=True, sharey=False, squeeze=False
                        )
                        x = np.arange(len(ordered_period_labels))
                        axes_flat = axes.ravel()
                        plotted_axes = []
                        for axis, cause_label in zip(axes_flat, cause_labels):
                            cause_color = CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP.get(cause_label, "grey")
                            plotted_any = False
                            for draw in ordered_draws:
                                draw_df = plot_df[draw]
                                cause_df = draw_df.xs(cause_label, level=label_level_name)
                                cause_df = cause_df.reindex(ordered_period_labels)
                                values = pd.to_numeric(cause_df, errors="coerce")
                                if values.notna().sum() == 0 or not values.fillna(0.0).to_numpy().any():
                                    continue
                                else:
                                    if has_uncertainty:
                                        lower_values = pd.to_numeric(
                                            lower_df[draw]
                                            .xs(cause_label, level=label_level_name)
                                            .reindex(ordered_period_labels),
                                            errors="coerce",
                                        )
                                        upper_values = pd.to_numeric(
                                            upper_df[draw]
                                            .xs(cause_label, level=label_level_name)
                                            .reindex(ordered_period_labels),
                                            errors="coerce",
                                        )
                                        if lower_values.notna().any() and upper_values.notna().any():
                                            axis.fill_between(
                                                x,
                                                lower_values.to_numpy(dtype=float),
                                                upper_values.to_numpy(dtype=float),
                                                color=cause_color,
                                                alpha=0.12,
                                                linewidth=0,
                                            )
                                    axis.plot(
                                        x,
                                        values.to_numpy(),
                                        color=cause_color,
                                        linestyle=draw_linestyle_map[draw],
                                        linewidth=2.4,
                                        alpha=1.0,
                                    )
                                    plotted_any = True
                            axis.set_title(str(cause_label), color=cause_color, fontsize=13)
                            axis.grid(axis="y", alpha=0.3)
                            axis.spines["top"].set_visible(False)
                            axis.spines["right"].set_visible(False)
                            if plotted_any:
                                plotted_axes.append(axis)
                        for axis in axes_flat[n_causes:]:
                            axis.set_visible(False)
                        for axis in axes_flat:
                            if axis.get_visible():
                                axis.set_xticks(x)
                                axis.set_xticklabels(display_period_labels, ha="right")
                        for axis in axes[:, 0]:
                            if axis.get_visible():
                                axis.set_ylabel("DALYs (/1000)")
                        draw_handles = [
                            Line2D(
                                [0],
                                [0],
                                color="black",
                                linestyle=draw_linestyle_map[draw],
                                linewidth=2,
                            )
                            for draw in ordered_draws
                        ]
                        legend_anchor = plotted_axes[0] if plotted_axes else axes_flat[0]
                        legend_anchor.legend(
                            handles=draw_handles,
                            title=ANALYSIS_UNIT_LABEL,
                            loc="upper left",
                            bbox_to_anchor=(1.02, 1.0),
                            fontsize=8,
                            title_fontsize=9,
                            frameon=True,
                        )
                        fig.tight_layout()
                        return (fig, axes)


def plot_deaths_by_period_for_hbp_cause(
    _df: pd.DataFrame, cause_label: str, draw_color_map: dict[str, tuple[float, float, float, float]]
):
    """Use the shared cause-specific plot, then recolor draw lines for HBP labels."""
    fig, ax = plot_deaths_by_period_for_cause(_df, cause_label=cause_label)
    recolor_lines_by_draw(ax, draw_color_map)
    return (fig, ax)


def do_barh_plot_with_ci_by_draw(_df: pd.DataFrame, ax, draw_color_map: dict[str, tuple[float, float, float, float]]):
    """Make horizontal bar plot with HBP draw colors."""
    _df.plot.barh(ax=ax, y="central", legend=False, color=[draw_color_map.get(str(draw), "grey") for draw in _df.index])
    y_positions = ax.get_yticks()
    for y_position, (_, row) in zip(y_positions, _df.iterrows()):
        lower_value = row["lower"]
        upper_value = row["upper"]
        ax.hlines(y_position, lower_value, upper_value, color="black", linewidth=1.2, zorder=3)
        ax.vlines(
            [lower_value, upper_value], y_position - 0.08, y_position + 0.08, color="black", linewidth=1.2, zorder=3
        )


def plot_dalys_averted_by_cause_stacked_by_draw(
    _df: pd.DataFrame,
    additive_dalys_averted: pd.DataFrame,
    draw_labels: list[str] | None = None,
    plot_stat: str = "central",
):
    """Plot jointly modelled and additive DALYs averted by draw."""
    if "label" not in _df.index.names:
        raise ValueError("_df index must contain a 'label' level.")
    if not isinstance(_df.columns, pd.MultiIndex) or _df.columns.nlevels != 2:
        raise ValueError("_df columns must be a 2-level MultiIndex with levels for draw and stat.")

    draw_level_name = "draw" if "draw" in _df.columns.names else _df.columns.names[0]
    stat_level_name = "stat" if "stat" in _df.columns.names else _df.columns.names[1]
    available_stats = pd.Index(_df.columns.get_level_values(stat_level_name).unique())
    if plot_stat not in available_stats:
        raise ValueError(f"Statistic '{plot_stat}' not found. Available stats: {available_stats.tolist()}")

    plot_df = _df.xs(plot_stat, axis=1, level=stat_level_name)
    plot_df = plot_df.groupby(level="label").sum().T.fillna(0.0)
    plot_df.index.name = draw_level_name
    ordered_causes = [
        cause_label
        for cause_label in CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP
        if cause_label in plot_df.columns
    ]
    unordered_causes = sorted(
        cause_label
        for cause_label in plot_df.columns
        if cause_label not in CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP
    )
    plot_df = plot_df.loc[:, ordered_causes + unordered_causes]
    if draw_labels is not None:
        ordered_draws = []
        for draw in draw_labels:
            display_draw = draw
            for candidate in (draw, display_draw):
                if candidate in plot_df.index and candidate not in ordered_draws:
                    ordered_draws.append(candidate)
        plot_df = plot_df.reindex(ordered_draws)
    if plot_df.empty:
        raise ValueError("No plottable DALYs-averted data remain after reshaping by draw and cause label.")

    fig_width = max(10, min(0.8 * len(plot_df.index) + 4, 24))
    fig_height = max(6, min(0.35 * len(plot_df.index) + 3, 14))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    x = np.arange(len(plot_df.index))
    positive_bottom = np.zeros(len(plot_df.index), dtype=float)
    negative_bottom = np.zeros(len(plot_df.index), dtype=float)
    for cause_label in plot_df.columns:
        values = plot_df[cause_label].to_numpy(dtype=float)
        if not np.any(values):
            continue
        bottoms = np.where(values >= 0.0, positive_bottom, negative_bottom)
        ax.bar(
            x,
            values,
            bottom=bottoms,
            color=CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP.get(cause_label, "grey"),
            label=str(cause_label),
            width=0.8,
        )
        positive_bottom += np.where(values >= 0.0, values, 0.0)
        negative_bottom += np.where(values < 0.0, values, 0.0)

    missing_additive_draws = [
        draw
        for draw in plot_df.index
        if draw not in additive_dalys_averted.index
    ]
    if missing_additive_draws:
        raise ValueError(
            "No additive DALYs-averted estimate found for plotted draws: "
            f"{missing_additive_draws}"
        )
    for position, draw in zip(x, plot_df.index):
        additive_estimate = additive_dalys_averted.loc[draw]
        lower = float(additive_estimate["lower"])
        upper = float(additive_estimate["upper"])
        ax.add_patch(
            Rectangle(
                (position - 0.4, lower),
                width=0.8,
                height=upper - lower,
                facecolor="black",
                edgecolor="none",
                alpha=0.15,
                label="95% UI for additive estimate" if position == x[0] else None,
                zorder=4,
            )
        )
        ax.hlines(
            y=float(additive_estimate["central"]),
            xmin=position - 0.4,
            xmax=position + 0.4,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label=(
                "Sum of individually modelled DALYs averted (additive assumption)"
                if position == x[0]
                else None
            ),
            zorder=5,
        )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index)
    ax.set_xlabel("")
    ax.set_ylabel("Discounted DALYs averted")
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        title="Cause label / comparison",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )
    fig.tight_layout()
    return (fig, ax)


def plot_dalys_costs_and_icers_by_hbp(
    dalys_averted_sorted: pd.DataFrame,
    incremental_cost_sorted: pd.DataFrame,
    icers_sorted: pd.DataFrame,
    facet_order: pd.Index,
    output_folder: Path,
    name_of_plot: str,
    draw_color_map: dict[str, tuple[float, float, float, float]],
):
    """Plot DALYs, incremental costs, and ICERs in aligned horizontal facets."""
    if len(facet_order) == 0:
        print(f"Skipping {name_of_plot}: no matching HBPs.")
        return
    else:
        dalys_facet = dalys_averted_sorted.reindex(facet_order)
        costs_facet = incremental_cost_sorted.reindex(facet_order)
        icers_facet = icers_sorted.reindex(facet_order)
        fig_height = max(6, min(0.28 * len(facet_order) + 4, 18))
        fig, axes = plt.subplots(1, 3, figsize=(20, fig_height), sharey=True)
        do_barh_plot_with_ci_by_draw(dalys_facet, axes[0], draw_color_map)
        axes[0].set_title("DALYs")
        axes[0].set_xlabel("Discounted DALYs averted (/1000)")
        do_barh_plot_with_ci_by_draw(costs_facet, axes[1], draw_color_map)
        axes[1].set_title("Costs")
        axes[1].set_xlabel("Incremental cost (USD)")
        do_barh_plot_with_ci_by_draw(icers_facet, axes[2], draw_color_map)
        axes[2].set_title("ICERs")
        axes[2].set_xlabel("ICER (discounted USD per discounted DALY averted)")
        for ax in axes:
            ax.grid(axis="x")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].set_ylabel(ANALYSIS_UNIT_LABEL)
        fig.suptitle(name_of_plot, y=1.02)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print(f"Saved: {name_of_plot}")


def apply(results_file: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard plots describing effect of each health benefit package."""
    print(f"Output folder: {output_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)
    with results_file.open("rb") as file:
        results = pickle.load(file)
    print(f"Using primary results from: {results_file}")
    non_baseline_draw_labels = list(results["hbp_draw_labels"])
    draw_labels = ["Nothing", *non_baseline_draw_labels]
    draw_color_map = get_draw_color_map([*non_baseline_draw_labels, "Nothing"])
    draw_linestyle_map = get_draw_linestyle_map(non_baseline_draw_labels)
    print("Plotting overlap between the two HBP sources with Fisher's exact test.")
    fig, ax = plot_hbp_overlap_with_fisher_test(results["hbp_overlap"])
    name_of_plot = "HBP from literature-derived or TLO-derived inputs"
    ax.set_title(name_of_plot)
    outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {name_of_plot}")
    additive_dalys_averted = results["additive_dalys_summary"]["additive_averted"]
    num_deaths_averted = results.get("num_deaths_averted")
    pc_deaths_averted = results.get("pc_deaths_averted")
    dalys_averted = results.get("dalys_averted")
    pc_dalys_averted = results.get("pc_dalys_averted")
    icers = results.get("icers_summarized")
    incremental_scenario_cost = results.get("incremental_scenario_cost")
    annual_cost_by_cadre = results.get("annual_cost_by_cadre")
    counts_of_hsi = results["counts_of_hsi_by_period"]
    annual_capacity_used_by_cadre_and_level = results.get("annual_capacity_used_by_cadre_and_level")
    comparison_metrics_available = all(
        (
            metric is not None
            for metric in (
                num_deaths_averted,
                pc_deaths_averted,
                dalys_averted,
                pc_dalys_averted,
                icers,
                incremental_scenario_cost,
            )
        )
    )
    print(f"Comparison metrics available: {comparison_metrics_available}")
    counts_of_hsi = counts_of_hsi.drop(["2026-2040"], level=1, errors="ignore")
    for draw in non_baseline_draw_labels:
        print(f"Plotting yearly HSI counts for HBP draw: {draw}")
        name_of_plot = f"Yearly HSI counts for {draw}"
        non_zero_rows_for_draw = (counts_of_hsi[draw] != 0).any(axis=1)
        plot_this = counts_of_hsi.loc[non_zero_rows_for_draw]
        fig, ax = plot_hsi_counts_by_period_for_draw(plot_this, draw)
        ax.set_title(name_of_plot)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.savefig(outfile)
        plt.close(fig)
    if annual_capacity_used_by_cadre_and_level is not None:
        print(f"Plotting capacity used by cadre and facility level over time (one figure per {ANALYSIS_UNIT_LABEL}).")
        for draw in non_baseline_draw_labels:
            try:
                name_of_plot = f"Capacity Used by Cadre and Facility Level Over Time for {draw}"
                fig, ax = plot_capacity_used_by_cadre_and_level_over_time_for_draw(
                    annual_capacity_used_by_cadre_and_level, draw, title=name_of_plot
                )
            except ValueError as exc:
                print(f"Skipping capacity-by-level plot for draw '{draw}': {exc}")
                continue
            outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
            fig.savefig(outfile)
            plt.close(fig)
    if annual_cost_by_cadre is not None:
        print(f"Plotting annual costs by cadre and over time (one figure per {ANALYSIS_UNIT_LABEL}).")
        for draw in non_baseline_draw_labels:
            try:
                name_of_plot = f"Cost by Cadre Over Time for {draw}"
                fig, ax = plot_cost_by_cadre_over_time_for_draw(annual_cost_by_cadre, draw, title=name_of_plot)
            except ValueError as exc:
                print(f"Skipping cost-by-cadre plot for draw '{draw}': {exc}")
                continue
            outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
            fig.savefig(outfile)
            plt.close(fig)
    total_population_in_implementation = results["total_population_by_year"]
    print("Plotting population size by year.")
    fig, ax = plot_population_by_year_for_hbp_draws(total_population_in_implementation / 1000000.0, draw_color_map)
    name_of_plot = "Population size by year"
    ax.set_title(name_of_plot)
    ax.set_ylabel("Population size (millions)")
    outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
    fig.savefig(outfile)
    plt.close(fig)
    num_dalys_by_cause_label_implementation = results["dalys"].drop(["2026-2040"], level=1, errors="ignore")
    num_deaths_by_cause_label_implementation = results["num_deaths"].drop(["2026-2040"], level=1, errors="ignore")
    print("Prepared deaths and DALYs by cause for plotting.")
    print("Plotting DALYs by cause label over time for each draw.")
    fig, axes = plot_dalys_by_cause_label_over_time_for_hbp_draws(
        num_dalys_by_cause_label_implementation / 1000.0, draw_labels=non_baseline_draw_labels, draw_linestyle_map=draw_linestyle_map
    )
    name_of_plot = "DALYs by Cause Label Over Time by HBP"
    fig.suptitle(name_of_plot, y=1.002)
    outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {name_of_plot}")
    print("Plotting stacked DALYs averted by cause label for each draw.")
    fig, ax = plot_dalys_averted_by_cause_stacked_by_draw(
        results["dalys_averted_by_cause"],
        additive_dalys_averted=additive_dalys_averted,
        draw_labels=non_baseline_draw_labels,
    )
    name_of_plot = "DALYs Averted by Cause for each HBP"
    ax.set_title(name_of_plot)
    outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {name_of_plot}")
    for draw in draw_labels:
        print(f"Plotting deaths over time by cause for draw: {draw}")
        fig, ax = plot_deaths_by_period_for_draw(num_deaths_by_cause_label_implementation / 1000.0, draw)
        name_of_plot = f"Deaths Over Time by Cause for {draw}"
        ax.set_title(name_of_plot)
        ax.set_ylabel("Number of deaths (/1000)")
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.savefig(outfile)
        plt.close(fig)
    cause_labels = num_deaths_by_cause_label_implementation.index.get_level_values("label").unique()
    for cause_label in cause_labels:
        print(f"Plotting cause-specific time series for: {cause_label}")
        fig, ax = plot_deaths_by_period_for_hbp_cause(
            num_deaths_by_cause_label_implementation / 1000.0, cause_label=cause_label, draw_color_map=draw_color_map
        )
        name_of_plot = f"Deaths Over Time for {cause_label}"
        ax.set_title(name_of_plot)
        ax.set_ylabel("Number of deaths (/1000)")
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.savefig(outfile)
        plt.close(fig)
        fig, ax = plot_deaths_by_period_for_hbp_cause(
            num_dalys_by_cause_label_implementation / 1000.0, cause_label=cause_label, draw_color_map=draw_color_map
        )
        name_of_plot = f"DALYs Over Time for {cause_label}"
        ax.set_title(name_of_plot)
        ax.set_ylabel("Number of DALYs (/1000)")
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.savefig(outfile)
        plt.close(fig)
    if comparison_metrics_available:
        print("Plotting comparison metrics: deaths/DALYs averted, percentages, and ICERs.")
        dalys_averted_sorted = dalys_averted.sort_values(by="central", ascending=True) / 1000.0
        dalys_order = dalys_averted_sorted.index
        fig_height = max(6, min(0.28 * len(dalys_averted_sorted.index) + 4, 18))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        name_of_plot = f"DALYS Averted by Each {ANALYSIS_UNIT_LABEL}"
        do_barh_plot_with_ci_by_draw(dalys_averted_sorted, ax, draw_color_map)
        ax.set_title(name_of_plot)
        ax.set_xlabel("Discounted DALYs averted (/1000)")
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print(f"Saved: {name_of_plot}")
        deaths_averted_sorted = (num_deaths_averted / 1000.0).reindex(dalys_order)
        fig_height = max(6, min(0.28 * len(deaths_averted_sorted.index) + 4, 18))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        name_of_plot = f"Deaths Averted by Each {ANALYSIS_UNIT_LABEL}"
        do_barh_plot_with_ci_by_draw(deaths_averted_sorted, ax, draw_color_map)
        ax.set_title(name_of_plot)
        ax.set_xlabel("Number of deaths averted (/1000)")
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print(f"Saved: {name_of_plot}")
        pc_deaths_averted_sorted = pc_deaths_averted.sort_values(by="central", ascending=True)
        fig_height = max(6, min(0.28 * len(pc_deaths_averted_sorted.index) + 4, 18))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        name_of_plot = f"Percentage Deaths Averted by Each {ANALYSIS_UNIT_LABEL}"
        do_barh_plot_with_ci_by_draw(pc_deaths_averted_sorted, ax, draw_color_map)
        ax.set_title(name_of_plot)
        ax.set_xlabel("Percentage of deaths averted")
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print(f"Saved: {name_of_plot}")
        pc_dalys_averted_sorted = pc_dalys_averted.sort_values(by="central", ascending=True)
        fig_height = max(6, min(0.28 * len(pc_dalys_averted_sorted.index) + 4, 18))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        name_of_plot = f"Percentage DALYs Averted by Each {ANALYSIS_UNIT_LABEL}"
        do_barh_plot_with_ci_by_draw(pc_dalys_averted_sorted, ax, draw_color_map)
        ax.set_title(name_of_plot)
        ax.set_xlabel("Percentage of discounted DALYs averted")
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print(f"Saved: {name_of_plot}")
        icers_sorted = icers.sort_values(by="central", ascending=True)
        icers_sorted = icers_sorted.reindex(dalys_order.intersection(icers_sorted.index))
        fig_height = max(6, min(0.28 * len(icers_sorted.index) + 4, 18))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        name_of_plot = f"ICERs for Each {ANALYSIS_UNIT_LABEL}"
        do_barh_plot_with_ci_by_draw(icers_sorted, ax, draw_color_map)
        ax.set_title(name_of_plot)
        ax.set_xlabel("ICER (discounted USD per discounted DALY averted)")
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print(f"Saved: {name_of_plot}")
        incremental_cost_sorted = incremental_scenario_cost.reindex(dalys_order)
        fig_height = max(6, min(0.28 * len(incremental_cost_sorted.index) + 4, 18))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        name_of_plot = f"Incremental Cost for Each {ANALYSIS_UNIT_LABEL}"
        do_barh_plot_with_ci_by_draw(incremental_cost_sorted, ax, draw_color_map)
        ax.set_title(name_of_plot)
        ax.set_xlabel("Incremental cost (USD)")
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        outfile = os.path.join(output_folder, make_graph_file_name(name_of_plot))
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close(fig)
        print(f"Saved: {name_of_plot}")
        facet_order = dalys_order.intersection(incremental_cost_sorted.dropna().index).intersection(
            icers_sorted.dropna().index
        )
        plot_dalys_costs_and_icers_by_hbp(
            dalys_averted_sorted=dalys_averted_sorted,
            incremental_cost_sorted=incremental_cost_sorted,
            icers_sorted=icers_sorted,
            facet_order=facet_order,
            output_folder=output_folder,
            name_of_plot=f"DALYs, Incremental Cost, and ICERs by {ANALYSIS_UNIT_LABEL}",
            draw_color_map=draw_color_map,
        )
        significant_dalys_averted_order = dalys_averted_sorted.index[
            pd.to_numeric(dalys_averted_sorted["lower"], errors="coerce") > 0
        ]
        subset_order = facet_order.intersection(significant_dalys_averted_order)
        plot_dalys_costs_and_icers_by_hbp(
            dalys_averted_sorted=dalys_averted_sorted,
            incremental_cost_sorted=incremental_cost_sorted,
            icers_sorted=icers_sorted,
            facet_order=subset_order,
            output_folder=output_folder,
            name_of_plot=f"DALYs, Incremental Cost, and ICERs by {ANALYSIS_UNIT_LABEL} with Significant DALYs Averted",
            draw_color_map=draw_color_map,
        )
    print("Finished generating figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=Path, nargs=1)
    parser.add_argument("--output_folder", type=Path, required=True)
    args = parser.parse_args()
    print(f"Results file: {type(args.results_file)}")
    print(f"Results file: {args.results_file}")
    print(f"Output folder: {args.output_folder}")
    if args.results_file:
        apply(results_file=args.results_file[0], output_folder=args.output_folder, resourcefilepath=Path("./resources"))
