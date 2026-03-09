"""Plotting utilities for treatment-id analysis scripts."""

import textwrap
import warnings

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tlo.analysis.utils import (
    CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP,
    get_color_cause_of_death_or_daly_label,
    get_color_short_treatment_id,
    order_of_cause_of_death_or_daly_label,
)


def do_bar_plot_with_ci(
    _df: pd.DataFrame,
    _param,
    _ax,
    period_labels_for_bar_plots: list[str],
    target_period_label: str,
):
    """Make vertical bars by cause, decomposed into period chunks, with overall-period CI."""
    available_params = _df.columns.get_level_values(0) if isinstance(_df.columns, pd.MultiIndex) else _df.columns
    if _param not in available_params:
        warnings.warn(f"Parameter '{_param}' not found in dataframe columns. Skipping plot.", stacklevel=2)
        return

    _df_nothing = _df[_param]
    _df_nothing = _df_nothing.reindex(
        pd.MultiIndex.from_product(
            [CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP.keys(), period_labels_for_bar_plots + [target_period_label]],
            names=["label", "period"],
        ),
        fill_value=0.0,
    )
    _df_nothing = _df_nothing.sort_index(axis=0, level=0, key=order_of_cause_of_death_or_daly_label)

    cause_labels = list(_df_nothing.index.get_level_values("label").unique())

    for i, cause_label in enumerate(cause_labels):
        color = get_color_cause_of_death_or_daly_label(cause_label)
        one_cause = _df_nothing.xs(cause_label, level="label")

        bottom = 0.0
        for j, period_label in enumerate(period_labels_for_bar_plots):
            chunk_height = one_cause.loc[period_label, "mean"] if period_label in one_cause.index else 0.0
            _ax.bar(i, chunk_height, bottom=bottom, color=color, alpha=0.9 if j % 2 == 0 else 0.35)
            bottom += chunk_height

        mean_value = one_cause.loc[target_period_label, "mean"]
        lower_value = one_cause.loc[target_period_label, "lower"]
        upper_value = one_cause.loc[target_period_label, "upper"]
        overall_yerr = np.array([[mean_value - lower_value], [upper_value - mean_value]])
        _ax.errorbar(i, mean_value, yerr=overall_yerr, fmt="none", ecolor="black", capsize=2, linewidth=1.2)

    _ax.set_xticks(range(len(cause_labels)))
    _ax.set_xticklabels(cause_labels, rotation=90)
    chunk_legend_handles = [
        Patch(facecolor="grey", alpha=0.9 if i % 2 == 0 else 0.35, label=period_label)
        for i, period_label in enumerate(period_labels_for_bar_plots)
    ]
    ci_legend_handle = Line2D([0], [0], color="black", marker="|", markersize=8, linewidth=1.2, label="95% CI")
    _ax.legend(handles=chunk_legend_handles + [ci_legend_handle], loc="upper right")


def plot_multiindex_dot_with_interval(
    _df: pd.DataFrame,
    year: int,
    _ax,
    central_measure: str = "mean",
    value_col: str = "population",
    sort: bool = True,
    x_label_rotation: int = 90,
    x_tick_fontsize: int = 8,
    label_wrap_width: int = 18,
    max_xticks: int = 30,
):
    """Plot central-value dots and lower/upper intervals by category for one year."""
    if not isinstance(_df.index, pd.MultiIndex) or _df.index.nlevels < 3:
        raise ValueError("_df index must be a MultiIndex with at least 3 levels: category, stat, year.")
    if value_col not in _df.columns:
        raise ValueError(f"Column '{value_col}' not found in dataframe.")

    year_level_values = _df.index.get_level_values(2)
    available_years = pd.Index(year_level_values.unique()).sort_values()
    if year not in available_years:
        raise ValueError(f"Year '{year}' not found in index level 2. Available years: {available_years.tolist()}")

    stat_level_values = _df.index.get_level_values(1)
    required_stats = {central_measure, "lower", "upper"}
    missing_stats = required_stats.difference(set(stat_level_values))
    if missing_stats:
        raise ValueError(
            f"Missing required stat(s) in index level 1: {sorted(missing_stats)}. "
            f"Available stats: {sorted(set(stat_level_values))}"
        )

    _plot = _df.xs(year, level=2)[value_col].unstack(level=1)
    _plot = _plot.loc[:, [central_measure, "lower", "upper"]]
    _plot = _plot.dropna(subset=[central_measure, "lower", "upper"])
    if _plot.empty:
        raise ValueError(f"No plottable rows remain for year '{year}' after selecting required stats.")

    if sort:
        _plot = _plot.sort_values(by=central_measure, ascending=True)

    x = np.arange(len(_plot.index))
    _ax.vlines(x, _plot["lower"], _plot["upper"], color="black", linewidth=1.2)
    _ax.scatter(x, _plot[central_measure], color="black", s=20, zorder=3)

    _ax.figure.set_size_inches(max(12, min(0.25 * len(_plot.index), 36)), 7)
    wrapped_labels = [textwrap.fill(str(label), width=label_wrap_width) for label in _plot.index]
    if max_xticks is not None and len(x) > max_xticks:
        step = int(np.ceil(len(x) / max_xticks))
        shown_positions = x[::step]
        shown_labels = [wrapped_labels[i] for i in shown_positions]
        _ax.set_xticks(shown_positions)
        _ax.set_xticklabels(shown_labels, rotation=x_label_rotation, ha="right", fontsize=x_tick_fontsize)
    else:
        _ax.set_xticks(x)
        _ax.set_xticklabels(wrapped_labels, rotation=x_label_rotation, ha="right", fontsize=x_tick_fontsize)
    _ax.set_xlabel(_df.index.names[0] if _df.index.names[0] is not None else "category")
    _ax.set_ylabel(value_col)
    _ax.set_title(f"{value_col}: {central_measure} with lower/upper ({year})")
    _ax.grid(axis="y")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    return _ax


def do_barh_plot_with_ci(_df: pd.DataFrame, _ax):
    """Make horizontal bar plot for each treatment id."""
    errors = pd.concat([_df["mean"] - _df["lower"], _df["upper"] - _df["mean"]], axis=1).T.to_numpy()
    _df.plot.barh(ax=_ax, y="mean", xerr=errors, legend=False, color=[get_color_short_treatment_id(_id) for _id in _df.index])


def do_label_barh_plot(_df: pd.DataFrame, _ax):
    """Add text annotation from values in dataframe onto axis."""
    y_cords = {ylabel.get_text(): ytick for ytick, ylabel in zip(_ax.get_yticks(), _ax.get_yticklabels())}
    pos_on_rhs = _ax.get_xticks()[-1]

    for label, row in _df.iterrows():
        if row["mean"] > 0:
            annotation = f"{round(row['mean'], 1)} ({round(row['lower'])}-{round(row['upper'])}) %"
            _ax.annotate(
                annotation,
                xy=(pos_on_rhs, y_cords.get(label)),
                xycoords="data",
                horizontalalignment="left",
                verticalalignment="center",
                size=7,
            )
