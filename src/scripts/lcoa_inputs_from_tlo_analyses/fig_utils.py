"""Plotting utilities for treatment-id analysis scripts."""

import textwrap
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tlo.analysis.utils import (
    CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP,
    get_color_cause_of_death_or_daly_label,
    get_color_short_treatment_id,
    make_calendar_period_type,
    order_of_cause_of_death_or_daly_label,
    order_of_short_treatment_ids,
)


APPOINTMENT_TYPE_PALETTE = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("Set2").colors)
APPOINTMENT_TYPE_FIXED_COLORS = {"AccidentsandEmerg": "black"}

def make_graph_file_name(stub):
    filename = stub.replace('*', '_star_').replace(' ', '_').replace('/', '').lower()
    return f"{filename}.png"


def _coerce_include_exclude_flag(value) -> int:
    """Map a table value to 1 for include, 0 for exclude, and 2 for NA."""
    if pd.isna(value):
        return 2

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"na", "n/a", "nan", "none", ""}:
            return 2
        if normalized in {"include", "included", "yes", "true", "t", "1"}:
            return 1
        if normalized in {"exclude", "excluded", "no", "false", "f", "0"}:
            return 0
        raise ValueError(f"Unrecognized include/exclude flag value: {value!r}")

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, (int, np.integer, float, np.floating)):
        if value in {0, 0.0}:
            return 0
        if value in {1, 1.0}:
            return 1
        raise ValueError(f"Numeric include/exclude flag must be 0 or 1, found {value!r}")

    return int(bool(value))


def plot_treatment_id_include_exclude_table(
    flags_df: pd.DataFrame,
    title: str | None = None,
):
    """Plot a treatment-ID include/exclude matrix as a colored table."""
    if not isinstance(flags_df, pd.DataFrame):
        raise TypeError("`flags_df` must be a pandas DataFrame.")

    treatment_id_column = next(
        (column for column in flags_df.columns if str(column).strip().lower() == "treatment_id"),
        None,
    )
    if treatment_id_column is not None:
        ordered_flags = flags_df.copy().set_index(treatment_id_column)
    else:
        ordered_flags = flags_df.copy()

    valid_row_labels = [
        row_label
        for row_label in ordered_flags.index
        if pd.notna(row_label) and str(row_label).strip().lower() not in {"na", "n/a", "nan", "none", ""}
    ]
    ordered_flags = ordered_flags.loc[valid_row_labels]

    if ordered_flags.empty:
        raise ValueError("No plottable rows remain in the include/exclude table.")

    ordered_flags = ordered_flags.loc[
        :,
        [column for column in ordered_flags.columns if str(column).strip().lower() != "treatment_id"],
    ]

    if ordered_flags.columns.empty:
        raise ValueError("`flags_df` must have at least one column.")

    numeric_flags = ordered_flags.apply(lambda col: col.map(_coerce_include_exclude_flag))
    data = numeric_flags.to_numpy(dtype=int)

    fig_width = max(6, min(1 * len(ordered_flags.columns) + 2.0, 24))
    fig_height = max(4.5, min(1 * len(ordered_flags.index) + 1.2, 22))


    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = ListedColormap(["#d73027", "#1a9850", "#bdbdbd"])
    ax.imshow(data, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", aspect="equal")

    ax.set_xticks(np.arange(len(ordered_flags.columns)))
    ax.set_yticks(np.arange(len(ordered_flags.index)))
    ax.set_xticklabels(
        [str(column) for column in ordered_flags.columns],
        rotation=90,
        ha="center",
        rotation_mode="anchor",
        fontsize=12,
    )
    ax.set_yticklabels([str(row) for row in ordered_flags.index], fontsize=14)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=12)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_xticks(np.arange(-0.5, len(ordered_flags.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ordered_flags.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if title is not None:
        ax.set_title(title, pad=18)

    #fig.subplots_adjust(left=0.8, right=0.88, top=0.1, bottom=0)
    return fig, ax


def get_color_by_appointment_type(appointment_types) -> dict:
    """Return a deterministic color map for appointment types."""
    non_fixed_appointment_types = sorted(
        appt for appt in appointment_types if appt not in APPOINTMENT_TYPE_FIXED_COLORS
    )
    color_by_appointment_type = {
        appt: APPOINTMENT_TYPE_PALETTE[i % len(APPOINTMENT_TYPE_PALETTE)]
        for i, appt in enumerate(non_fixed_appointment_types)
    }
    color_by_appointment_type.update(
        {appt: color for appt, color in APPOINTMENT_TYPE_FIXED_COLORS.items() if appt in appointment_types}
    )
    return color_by_appointment_type


def _get_short_treatment_id_and_color(treatment_id: str) -> tuple[str, str]:
    """Return short treatment id prefix and plotting color for a treatment id."""
    short_treatment_id = str(treatment_id).split("_")[0]
    color = get_color_short_treatment_id(short_treatment_id)
    return short_treatment_id, ("grey" if pd.isna(color) else color)


def _get_ordered_short_treatment_ids(treatment_ids: pd.Index) -> list[str]:
    """Return treatment ids with recognized short ids first in standard order."""
    treatment_ids = pd.Index(treatment_ids).unique()
    recognized = [treatment_id for treatment_id in treatment_ids if not pd.isna(get_color_short_treatment_id(treatment_id))]
    unrecognized = sorted(str(treatment_id) for treatment_id in treatment_ids if pd.isna(get_color_short_treatment_id(treatment_id)))
    recognized = sorted(recognized, key=order_of_short_treatment_ids)
    return recognized + unrecognized


def _parse_period_label(period_label: str) -> tuple[int, int]:
    """Parse a period label of the form YYYY-YYYY into start/end years."""
    start_year_text, end_year_text = str(period_label).split("-", maxsplit=1)
    return int(start_year_text), int(end_year_text)


def _get_sorted_period_labels_and_display_labels(period_labels: list[str]) -> tuple[list[str], list[str]]:
    """Return chronological labels plus display labels, falling back to input order if parsing fails."""
    try:
        parsed_periods = [(label, _parse_period_label(label)) for label in period_labels]
    except (TypeError, ValueError):
        return period_labels, period_labels

    ordered_period_labels = [
        label for label, _ in sorted(parsed_periods, key=lambda item: (item[1][0], item[1][1]))
    ]
    display_labels = [
        str(start_year) if start_year == end_year else label
        for label, (start_year, end_year) in sorted(parsed_periods, key=lambda item: (item[1][0], item[1][1]))
    ]
    return ordered_period_labels, display_labels


def plot_deaths_by_period_for_cause(
    _df: pd.DataFrame,
    cause_label: str,
    plot_stat: str = "central",
):
    """Plot deaths over time for a single cause, with one line per short treatment id."""
    if not isinstance(_df.index, pd.MultiIndex) or _df.index.nlevels != 2:
        raise ValueError("_df index must be a 2-level MultiIndex with levels for label and period.")
    if not isinstance(_df.columns, pd.MultiIndex) or _df.columns.nlevels != 2:
        raise ValueError("_df columns must be a 2-level MultiIndex with levels for treatment id and stat.")

    label_level_name = "label" if "label" in _df.index.names else _df.index.names[0]
    period_level_name = "period" if "period" in _df.index.names else _df.index.names[1]
    stat_level_name = "stat" if "stat" in _df.columns.names else _df.columns.names[1]

    available_causes = pd.Index(_df.index.get_level_values(label_level_name).unique())
    if cause_label not in available_causes:
        raise ValueError(f"Cause label '{cause_label}' not found. Available causes: {available_causes.tolist()}")

    available_stats = pd.Index(_df.columns.get_level_values(stat_level_name).unique())
    if plot_stat not in available_stats:
        raise ValueError(f"Statistic '{plot_stat}' not found. Available stats: {available_stats.tolist()}")

    _plot = _df.xs(cause_label, level=label_level_name).xs(plot_stat, axis=1, level=stat_level_name)
    if _plot.empty:
        raise ValueError(f"No plottable data remain for cause '{cause_label}' using stat '{plot_stat}'.")

    _plot.index.name = period_level_name
    try:
        ordered_period_labels, display_period_labels = _get_sorted_period_labels_and_display_labels(_plot.index)
        _plot = _plot.reindex(ordered_period_labels)
    except (TypeError, ValueError):
        _plot = _plot.loc[pd.Index(_plot.index).drop_duplicates()]

    ordered_treatment_ids = _get_ordered_short_treatment_ids(_plot.columns)
    _plot = _plot.loc[:, ordered_treatment_ids]

    fig_width = max(10, min(1.4 * len(_plot.index) + 4, 18))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = np.arange(len(_plot.index))

    for treatment_id in _plot.columns:
        _, color = _get_short_treatment_id_and_color(treatment_id)
        ax.plot(
            x,
            _plot[treatment_id].to_numpy(),
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=str(treatment_id),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(display_period_labels, rotation=45, ha="right")
    ax.set_xlabel("Period")
    ax.set_ylabel("Number of deaths")
    ax.set_title(str(cause_label))
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        title="Treatment ID",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )
    return fig, ax


def plot_deaths_by_period_for_draw(
    _df: pd.DataFrame,
    draw: str,
    plot_stat: str = "central",
):
    """Plot deaths over time for a single draw, with one line per cause label."""
    if not isinstance(_df.index, pd.MultiIndex) or _df.index.nlevels != 2:
        raise ValueError("_df index must be a 2-level MultiIndex with levels for label and period.")
    if not isinstance(_df.columns, pd.MultiIndex) or _df.columns.nlevels != 2:
        raise ValueError("_df columns must be a 2-level MultiIndex with levels for draw and stat.")

    label_level_name = "label" if "label" in _df.index.names else _df.index.names[0]
    period_level_name = "period" if "period" in _df.index.names else _df.index.names[1]
    draw_level_name = "draw" if "draw" in _df.columns.names else _df.columns.names[0]
    stat_level_name = "stat" if "stat" in _df.columns.names else _df.columns.names[1]

    available_draws = pd.Index(_df.columns.get_level_values(draw_level_name).unique())
    if draw not in available_draws:
        raise ValueError(f"Draw '{draw}' not found. Available draws: {available_draws.tolist()}")
    available_stats = pd.Index(_df.columns.get_level_values(stat_level_name).unique())
    if plot_stat not in available_stats:
        raise ValueError(f"Statistic '{plot_stat}' not found. Available stats: {available_stats.tolist()}")

    _plot = _df[draw].loc[:, [plot_stat]]
    if _plot.empty:
        raise ValueError(f"No plottable data remain for draw '{draw}' using stat '{plot_stat}'.")

    _plot = _plot[plot_stat].unstack(label_level_name)
    ordered_causes = [
        cause_label for cause_label in CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP.keys()
        if cause_label in _plot.columns
    ]
    unordered_causes = sorted(
        cause_label for cause_label in _plot.columns if cause_label not in CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP
    )
    _plot = _plot.loc[:, ordered_causes + unordered_causes]

    ordered_period_labels, display_period_labels = _get_sorted_period_labels_and_display_labels(_plot.index.tolist())
    _plot = _plot.reindex(ordered_period_labels)
    if _plot.empty:
        raise ValueError(f"No plottable data remain for draw '{draw}' after reshaping by cause.")

    fig_width = max(10, min(1.4 * len(ordered_period_labels) + 4, 18))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for cause_label in _plot.columns:
        cause_values = _plot[cause_label]
        if cause_values.notna().sum() == 0:
            continue
        ax.plot(
            ordered_period_labels,
            cause_values.to_numpy(),
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=get_color_cause_of_death_or_daly_label(cause_label),
            label=str(cause_label),
        )

    ax.set_xticks(ordered_period_labels)
    ax.set_xticklabels(display_period_labels, rotation=45, ha="right")
    ax.set_xlabel("Period")
    ax.set_ylabel("Number of deaths")
    ax.set_title(str(draw))
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        title="",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )
    return fig, ax


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
            chunk_height = one_cause.loc[period_label, "central"] if period_label in one_cause.index else 0.0
            _ax.bar(i, chunk_height, bottom=bottom, color=color, alpha=0.9 if j % 2 == 0 else 0.35)
            bottom += chunk_height

        mean_value = one_cause.loc[target_period_label, "central"]
        lower_value = one_cause.loc[target_period_label, "lower"]
        upper_value = one_cause.loc[target_period_label, "upper"]
        _ax.vlines(i, lower_value, upper_value, color="black", linewidth=1.2)
        _ax.hlines([lower_value, upper_value], i - 0.08, i + 0.08, color="black", linewidth=1.2)

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
    central_measure: str = "central",
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
    _df.plot.barh(
        ax=_ax,
        y="central",
        legend=False,
        color=[_get_short_treatment_id_and_color(_id)[1] for _id in _df.index],
    )

    y_positions = _ax.get_yticks()
    for y_position, (_, row) in zip(y_positions, _df.iterrows()):
        lower_value = row["lower"]
        upper_value = row["upper"]
        _ax.hlines(y_position, lower_value, upper_value, color="black", linewidth=1.2, zorder=3)
        _ax.vlines([lower_value, upper_value], y_position - 0.08, y_position + 0.08, color="black", linewidth=1.2, zorder=3)


def do_label_barh_plot(_df: pd.DataFrame, _ax):
    """Add text annotation from values in dataframe onto axis."""
    y_cords = {ylabel.get_text(): ytick for ytick, ylabel in zip(_ax.get_yticks(), _ax.get_yticklabels())}
    pos_on_rhs = _ax.get_xticks()[-1]

    for label, row in _df.iterrows():
        if row["central"] > 0:
            annotation = f"{round(row['central'], 1)} ({round(row['lower'])}-{round(row['upper'])}) %"
            _ax.annotate(
                annotation,
                xy=(pos_on_rhs, y_cords.get(label)),
                xycoords="data",
                horizontalalignment="left",
                verticalalignment="center",
                size=7,
            )

def plot_cadre_time_by_draw_stacked(
    _df: pd.DataFrame,
    stat: str = "central",
    figsize: tuple[float, float] | None = None,
):
    """Plot horizontal stacked bars of cadre time use by draw for one summary stat."""
    if not isinstance(_df.columns, pd.MultiIndex) or _df.columns.nlevels != 2:
        raise ValueError("_df columns must be a 2-level MultiIndex with levels for draw and stat.")

    stat_level_name = "stat" if "stat" in _df.columns.names else _df.columns.names[1]
    available_stats = pd.Index(_df.columns.get_level_values(stat_level_name).unique())
    if stat not in available_stats:
        raise ValueError(f"Statistic '{stat}' not found. Available stats: {available_stats.tolist()}")

    _plot = _df.xs(stat, axis=1, level=stat_level_name).T.fillna(0.0)
    if _plot.empty:
        raise ValueError(f"No plottable data remain for stat '{stat}'.")

    _plot = _plot.loc[_plot.sum(axis=1).sort_values(ascending=True).index]

    if figsize is None:
        fig_height = max(6, min(0.35 * len(_plot.index) + 3, 20))
        figsize = (12, fig_height)
    fig, ax = plt.subplots(figsize=figsize)

    cadre_colors = list(plt.get_cmap("tab10").colors)
    left = np.zeros(len(_plot.index), dtype=float)
    y = np.arange(len(_plot.index))

    for i, cadre in enumerate(_plot.columns):
        values = _plot[cadre].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            color=cadre_colors[i % len(cadre_colors)],
            label=str(cadre),
        )
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels([str(draw) for draw in _plot.index])
    ax.set_xlabel("Time used")
    ax.set_ylabel("Draw")
    ax.grid(axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower right",
        fontsize=12,
        handlelength=2.4,
        handleheight=1.6,
        borderpad=1.0,
        labelspacing=0.8,
        frameon=True,
    )
    fig.tight_layout()
    return fig, ax

def plot_hsi_counts_stacked_bar(_df: pd.DataFrame, plot_stat: str = "central"):
    """Plot horizontal stacked bars of HSI counts by draw for a selected summary statistic."""
    if not isinstance(_df.columns, pd.MultiIndex) or _df.columns.nlevels != 2:
        raise ValueError("_df columns must be a 2-level MultiIndex with levels for draw and stat.")

    stat_level_name = "stat" if "stat" in _df.columns.names else _df.columns.names[1]
    stat_level_values = _df.columns.get_level_values(stat_level_name)
    if plot_stat not in stat_level_values:
        raise ValueError(f"The column MultiIndex does not contain '{plot_stat}' in the stat level.")

    _plot = _df.xs(plot_stat, axis=1, level=stat_level_name).T
    if _plot.empty:
        raise ValueError(f"No plottable data remain after selecting the '{plot_stat}' columns.")

    if _plot.isna().any().any():
        warnings.warn(
            f"Missing values detected after selecting '{plot_stat}'. Bars will omit missing segments.",
            stacklevel=2,
        )

    totals = _plot.sum(axis=1, skipna=True)
    _plot = _plot.loc[totals.sort_values(ascending=False).index]
    if not (_plot.gt(0).any(axis=1)).any():
        raise ValueError(f"No positive values remain after selecting the '{plot_stat}' columns.")

    fig_width = max(12, min(0.22 * len(_plot.columns) + 12, 30))
    fig_height = max(6, min(0.35 * len(_plot.index), 24))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    left = np.zeros(len(_plot.index), dtype=float)
    y = np.arange(len(_plot.index))

    for treatment_id in _plot.columns:
        values = _plot[treatment_id]
        mask = values.gt(0) & values.notna()
        if not mask.any():
            continue
        ax.barh(
            y[mask.to_numpy()],
            values.loc[mask].to_numpy(),
            left=left[mask.to_numpy()],
            color=_get_color_for_treatment_id_prefix(treatment_id),
            label=str(treatment_id),
        )
        left[mask.to_numpy()] += values.loc[mask].to_numpy()

    ax.set_yticks(y)
    ax.set_yticklabels([str(label) for label in _plot.index], fontsize=12)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax


def plot_hsi_counts_by_period_for_draw(
    _df: pd.DataFrame,
    draw: str,
):
    """Plot central values with lower/upper intervals across period chunks for one draw."""
    if not isinstance(_df.index, pd.MultiIndex) or _df.index.nlevels != 2:
        raise ValueError("_df index must be a 2-level MultiIndex with levels for short_treatment_id and period.")
    if not isinstance(_df.columns, pd.MultiIndex) or _df.columns.nlevels != 2:
        raise ValueError("_df columns must be a 2-level MultiIndex with levels for draw and stat.")
    if draw not in _df.columns.get_level_values(0):
        available_draws = sorted(set(_df.columns.get_level_values(0)))
        raise ValueError(f"Draw '{draw}' not found. Available draws: {available_draws}")

    _df = _df[draw]
    _plot = _df.reindex(
        pd.MultiIndex.from_product(
            [
                _df.index.get_level_values(0).unique(),
                _df.index.get_level_values(1).unique(),
            ],
            names=["treatment_id", "period"],
        ),
        fill_value=0.0,
    )
    period_labels = _plot.index.get_level_values(1).unique()
    _plot = _plot.loc[:, ["lower", "central", "upper"]]
    _plot = _plot.unstack("period")

    central = _plot["central"]
    lower = _plot["lower"]
    upper = _plot["upper"]
    periods_for_filtering = central.columns.difference(["2025-2025"], sort=False)
    non_zero_mask = central.loc[:, periods_for_filtering].gt(0).any(axis=1)

    ordered_period_labels, display_period_labels = _get_sorted_period_labels_and_display_labels(period_labels)
    central = central.loc[non_zero_mask, ordered_period_labels]
    lower = lower.loc[non_zero_mask, ordered_period_labels]
    upper = upper.loc[non_zero_mask, ordered_period_labels]

    if central.empty:
        print(f"No non-zero treatment ids remain for draw '{draw}'.")

    x = np.arange(len(ordered_period_labels))
    fig_width = max(10, min(1.2 * len(ordered_period_labels) + 4, 20))
    fig_height = max(6, min(0.28 * len(central.index) + 6, 18))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for treatment_id in central.index:
        central_values = central.loc[treatment_id].to_numpy()
        _, color = _get_short_treatment_id_and_color(treatment_id)
        ax.plot(
            x,
            central_values,
            marker="o",
            color=color,
            linewidth=1.2,
            markersize=4,
            label=str(treatment_id),
        )
        lower_values = lower.loc[treatment_id].to_numpy(dtype=float)
        upper_values = upper.loc[treatment_id].to_numpy(dtype=float)
        ax.vlines(x, lower_values, upper_values, color=color, linewidth=1.2)
        ax.hlines(lower_values, x - 0.08, x + 0.08, color=color, linewidth=1.2)
        ax.hlines(upper_values, x - 0.08, x + 0.08, color=color, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(display_period_labels, rotation=45, ha="right")
    ax.set_xlabel("period")
    ax.set_ylabel("HSI count")
    ax.set_title(f"HSI counts by period: {draw}")
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        title="Treatment ID",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )

    fig.tight_layout()
    return fig, ax


def plot_population_by_year(_df: pd.DataFrame):
    """Plot yearly central population values for all draws."""
    if not isinstance(_df.columns, pd.MultiIndex) or _df.columns.nlevels != 2:
        raise ValueError("_df columns must be a 2-level MultiIndex with levels for draw and stat.")

    stat_level_name = "stat" if "stat" in _df.columns.names else _df.columns.names[1]

    available_stats = pd.Index(_df.columns.get_level_values(stat_level_name).unique())
    if "central" not in available_stats:
        raise ValueError(f"Statistic 'central' not found. Available stats: {available_stats.tolist()}")

    implementation_central = _df.xs("central", axis=1, level=stat_level_name).copy()
    implementation_central.columns = implementation_central.columns.to_series().str.replace(r"_\*$", "", regex=True)
    _plot = implementation_central

    _plot = _plot.loc[:, ~_plot.columns.duplicated()]
    _plot = _plot.sort_index()

    ordered_treatment_ids = _get_ordered_short_treatment_ids(_plot.columns)
    _plot = _plot.loc[:, ordered_treatment_ids]

    if _plot.empty:
        raise ValueError("No plottable population data remain after selecting central values.")

    years = pd.Index(_plot.index)
    x = np.arange(len(years))
    fig_width = max(10, min(1.0 * len(years) + 4, 20))
    fig_height = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for treatment_id in _plot.columns:
        short_treatment_id, color = _get_short_treatment_id_and_color(treatment_id)
        ax.plot(
            x,
            _plot[treatment_id].to_numpy(),
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=short_treatment_id,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(year) for year in years], rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Population size")
    ax.set_title("Population size by year")
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    deduplicated_handles_by_label = dict(zip(labels, handles))
    ax.legend(
        handles=list(deduplicated_handles_by_label.values()),
        labels=list(deduplicated_handles_by_label.keys()),
        title="Treatment ID",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )

    fig.tight_layout()
    return fig, ax


def plot_proportion_capacity_used_by_cadre_over_time_for_draw(_df: pd.DataFrame, draw: str):
    """Plot grouped bars by year for one draw; each bar-group decomposes by cadre with CI error bars."""
    if _df is None:
        raise ValueError("`proportion_capacity_used_by_cadre` is None.")
    if not isinstance(_df.index, pd.MultiIndex) or _df.index.nlevels != 2:
        raise ValueError("Expected a 2-level index: (OfficerType, year).")
    if not isinstance(_df.columns, pd.MultiIndex) or _df.columns.nlevels != 2:
        raise ValueError("Expected a 2-level columns index: (draw, stat).")

    officer_level_name = "OfficerType" if "OfficerType" in _df.index.names else _df.index.names[0]
    year_level_name = "year" if "year" in _df.index.names else _df.index.names[1]
    draw_level_name = "draw" if "draw" in _df.columns.names else _df.columns.names[0]

    available_draws = pd.Index(_df.columns.get_level_values(draw_level_name).unique())
    if draw not in available_draws:
        raise ValueError(f"Draw '{draw}' not found. Available draws: {available_draws.tolist()}")

    draw_df = _df[draw].copy()
    required_stats = {"central", "lower", "upper"}
    if not required_stats.issubset(draw_df.columns):
        raise ValueError(
            f"Missing required stats {sorted(required_stats)} in draw '{draw}'. "
            f"Found: {draw_df.columns.tolist()}"
        )

    years = sorted(pd.Index(draw_df.index.get_level_values(year_level_name).unique()).tolist())
    cadres = sorted(pd.Index(draw_df.index.get_level_values(officer_level_name).unique()).tolist())
    x = np.arange(len(years), dtype=float)
    bar_width = min(0.8 / max(len(cadres), 1), 0.12)
    offsets = (np.arange(len(cadres)) - (len(cadres) - 1) / 2) * bar_width

    fig_width = max(10, min(1.3 * len(years) + 6, 24))
    fig_height = max(6, min(0.22 * len(cadres) + 6, 14))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for i, cadre in enumerate(cadres):
        cadre_df = draw_df.xs(cadre, level=officer_level_name).reindex(years)
        central = pd.to_numeric(cadre_df["central"], errors="coerce").fillna(0.0).to_numpy()
        lower = pd.to_numeric(cadre_df["lower"], errors="coerce").fillna(0.0).to_numpy()
        upper = pd.to_numeric(cadre_df["upper"], errors="coerce").fillna(0.0).to_numpy()
        lower_err = np.clip(central - lower, a_min=0.0, a_max=None)
        upper_err = np.clip(upper - central, a_min=0.0, a_max=None)

        ax.bar(
            x + offsets[i],
            central,
            width=bar_width,
            label=str(cadre),
            yerr=np.vstack([lower_err, upper_err]),
            capsize=2,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion of Capacity Used")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Cadre", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.tight_layout()
    return fig, ax


def plot_cost_by_cadre_over_time_for_draw(
    _df: pd.DataFrame,
    draw: str,
    title: str | None = None,
):
    """Plot capacity used over time for one draw, with one line per officer type."""
    if _df is None:
        raise ValueError("`_df` is None.")
    if not isinstance(_df.index, pd.MultiIndex) or _df.index.nlevels != 2:
        raise ValueError("Expected a 2-level index: (year, OfficerType).")
    if not isinstance(_df.columns, pd.MultiIndex) or _df.columns.nlevels != 2:
        raise ValueError("Expected a 2-level columns index: (draw, stat).")

    year_level_name = "year" if "year" in _df.index.names else _df.index.names[0]
    officer_level_name = "OfficerType" if "OfficerType" in _df.index.names else _df.index.names[1]
    draw_level_name = "draw" if "draw" in _df.columns.names else _df.columns.names[0]

    available_draws = pd.Index(_df.columns.get_level_values(draw_level_name).unique())
    if draw not in available_draws:
        raise ValueError(f"Draw '{draw}' not found. Available draws: {available_draws.tolist()}")

    draw_df = _df[draw].copy()
    required_stats = {"central", "lower", "upper"}
    if not required_stats.issubset(draw_df.columns):
        raise ValueError(
            f"Missing required stats {sorted(required_stats)} in draw '{draw}'. "
            f"Found: {draw_df.columns.tolist()}"
        )

    years = sorted(pd.Index(draw_df.index.get_level_values(year_level_name).unique()).tolist())
    officers = sorted(pd.Index(draw_df.index.get_level_values(officer_level_name).unique()).tolist(), key=str)
    x = np.arange(len(years), dtype=float)

    fig_width = max(10, min(1.3 * len(years) + 4, 24))
    fig_height = max(6, min(0.28 * len(officers) + 4, 16))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    officer_colors = list(plt.get_cmap("tab10").colors)
    plotted_any = False

    for i, officer_type in enumerate(officers):
        officer_df = draw_df.xs(officer_type, level=officer_level_name).reindex(years)
        if officer_df.empty:
            continue

        central = pd.to_numeric(officer_df["central"], errors="coerce").fillna(0.0).to_numpy()
        lower = pd.to_numeric(officer_df["lower"], errors="coerce").fillna(0.0).to_numpy()
        upper = pd.to_numeric(officer_df["upper"], errors="coerce").fillna(0.0).to_numpy()

        if not (central.any() or lower.any() or upper.any()):
            continue

        color = officer_colors[i % len(officer_colors)]
        ax.plot(
            x,
            central,
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=str(officer_type),
        )
        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        plotted_any = True

    if not plotted_any:
        raise ValueError(f"No plottable officer types remain for draw '{draw}'.")

    ax.set_xticks(x)
    ax.set_xticklabels([str(year) for year in years], rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Capacity used")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    deduplicated_handles_by_label = dict(zip(labels, handles))
    ax.legend(
        handles=list(deduplicated_handles_by_label.values()),
        labels=list(deduplicated_handles_by_label.keys()),
        title="OfficerType",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
    )

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax


def plot_capacity_used_by_cadre_and_level_over_time_for_draw(
    _df: pd.DataFrame,
    draw: str,
    title: str | None = None,
):
    """Plot line charts by year for one draw, faceted by facility level."""
    if _df is None:
        raise ValueError("`annual_capacity_used_by_cadre_and_level` is None.")
    if not isinstance(_df.index, pd.MultiIndex) or _df.index.nlevels != 3:
        raise ValueError("Expected a 3-level index: (year, OfficerType, FacilityLevel).")
    if not isinstance(_df.columns, pd.MultiIndex) or _df.columns.nlevels != 2:
        raise ValueError("Expected a 2-level columns index: (draw, stat).")

    year_level_name = "year" if "year" in _df.index.names else _df.index.names[0]
    officer_level_name = "OfficerType" if "OfficerType" in _df.index.names else _df.index.names[1]
    facility_level_name = "FacilityLevel" if "FacilityLevel" in _df.index.names else _df.index.names[2]
    draw_level_name = "draw" if "draw" in _df.columns.names else _df.columns.names[0]

    available_draws = pd.Index(_df.columns.get_level_values(draw_level_name).unique())
    if draw not in available_draws:
        raise ValueError(f"Draw '{draw}' not found. Available draws: {available_draws.tolist()}")

    draw_df = _df[draw].copy()
    required_stats = {"central", "lower", "upper"}
    if not required_stats.issubset(draw_df.columns):
        raise ValueError(
            f"Missing required stats {sorted(required_stats)} in draw '{draw}'. "
            f"Found: {draw_df.columns.tolist()}"
        )

    years = sorted(pd.Index(draw_df.index.get_level_values(year_level_name).unique()).tolist())
    cadres = sorted(pd.Index(draw_df.index.get_level_values(officer_level_name).unique()).tolist(), key=str)
    facility_levels = sorted(pd.Index(draw_df.index.get_level_values(facility_level_name).unique()).tolist(), key=str)

    if not facility_levels:
        raise ValueError(f"No facility levels found for draw '{draw}'.")

    x = np.arange(len(years), dtype=float)
    fig_width = max(11, min(1.3 * len(years) + 6, 24))
    valid_facility_levels: list[str] = []
    facility_frames: dict[str, pd.DataFrame] = {}
    for facility_level in facility_levels:
        level_df = draw_df.xs(facility_level, level=facility_level_name).reindex(
            pd.MultiIndex.from_product(
                [years, cadres],
                names=[year_level_name, officer_level_name],
            ),
            fill_value=0.0,
        )
        if level_df.empty:
            continue
        if not level_df.loc[:, ["central", "lower", "upper"]].fillna(0.0).to_numpy().any():
            continue
        valid_facility_levels.append(facility_level)
        facility_frames[facility_level] = level_df

    if not valid_facility_levels:
        raise ValueError(f"No plottable facility levels remain for draw '{draw}'.")

    fig_height = max(4.5 * len(valid_facility_levels), 5)
    fig, axes = plt.subplots(
        len(valid_facility_levels),
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    cadre_colors = list(plt.get_cmap("tab10").colors)

    for axis, facility_level in zip(axes, valid_facility_levels):
        level_df = facility_frames[facility_level]

        for i, cadre in enumerate(cadres):
            cadre_df = level_df.xs(cadre, level=officer_level_name).reindex(years)
            central = pd.to_numeric(cadre_df["central"], errors="coerce").fillna(0.0).to_numpy()
            lower = pd.to_numeric(cadre_df["lower"], errors="coerce").fillna(0.0).to_numpy()
            upper = pd.to_numeric(cadre_df["upper"], errors="coerce").fillna(0.0).to_numpy()

            axis.plot(
                x,
                central,
                label=str(cadre),
                color=cadre_colors[i % len(cadre_colors)],
                marker="o",
                linewidth=1.8,
                markersize=4,
            )
            axis.fill_between(
                x,
                lower,
                upper,
                color=cadre_colors[i % len(cadre_colors)],
                alpha=0.12,
                linewidth=0,
            )

        axis.set_ylabel("Fraction of available capacity used")
        axis.set_title(f"Facility level {facility_level}")
        axis.grid(axis="y", alpha=0.3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(y) for y in years], rotation=45, ha="right")


    handles, labels = axes[0].get_legend_handles_labels()
    deduplicated_handles_by_label = dict(zip(labels, handles))
    fig.legend(
        deduplicated_handles_by_label.values(),
        deduplicated_handles_by_label.keys(),
        loc='upper left',
        bbox_to_anchor=(0.75, 0.75),
        ncols = 2,
        fontsize=8,
        frameon=True,
    )

    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 0.86, 0.96))
    else:
        fig.tight_layout()
    return fig, axes[0]
