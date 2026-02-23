"""Produce plots to show the impact each set of treatments."""

import argparse
import glob
import os
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.calibration_analyses.analysis_scripts import plot_legends
from scripts.lcoa_inputs_from_tlo_analyses.scenario_effect_of_treatment_ids import (
    EffectOfEachTreatment,
)
from tlo import Date
from tlo.analysis.utils import (
    CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP,
    extract_results,
    get_color_cause_of_death_or_daly_label,
    get_color_short_treatment_id,
    make_age_grp_lookup,
    make_age_grp_types,
    order_of_cause_of_death_or_daly_label,
    squarify_neat,
    summarize,
    to_age_group,
)

TARGET_PERIOD = (Date(2010, 1, 1), Date(2025, 12, 31))
PERIOD_LENGTH_YEARS_FOR_BAR_PLOTS = 5


def extract_deaths_total(df: pd.DataFrame) -> pd.Series:
    return pd.Series({"Total": len(df)})

def target_period(target_period_tuple: tuple[Date, Date] = TARGET_PERIOD) -> str:
    """Returns the target period as a string of the form YYYY-YYYY."""
    return "-".join(str(t.year) for t in target_period_tuple)


def get_periods_within_target_period(
    period_length_years: int,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
) -> list[tuple[str, tuple[int, int]]]:
    """Return chunks within target period as [(label, (start_year, end_year)), ...]."""
    if period_length_years <= 0:
        raise ValueError("period_length_years must be a positive integer.")
    start_year, end_year = target_period_tuple[0].year, target_period_tuple[1].year
    periods = []
    for chunk_start in range(start_year, end_year + 1, period_length_years):
        chunk_end = min(chunk_start + period_length_years - 1, end_year)
        periods.append((f"{chunk_start}-{chunk_end}", (chunk_start, chunk_end)))
    return periods


def get_parameter_names_from_scenario_file() -> Tuple[str]:
    """Get tuple of scenario names from Scenario class used to create results."""
    e = EffectOfEachTreatment()
    return tuple(e._scenarios.keys())


def format_scenario_name(_sn: str) -> str:
    """Return reformatted scenario name ready for plotting."""
    if _sn == "Nothing":
        return "Nothing"
    return _sn.lstrip("Only ")


def set_param_names_as_column_index_level_0(_df: pd.DataFrame, param_names: tuple[str, ...]) -> pd.DataFrame:
    """Set columns index level 0 as scenario param names."""
    ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
    assert len(names_of_cols_level0) == len(_df.columns.levels[0])

    reformatted_names = map(format_scenario_name, names_of_cols_level0)
    _df.columns = _df.columns.set_levels(reformatted_names, level=0)
    return _df


def find_difference_extra_relative_to_comparison(
    _ser: pd.Series,
    comparison: str,
    scaled: bool = False,
    drop_comparison: bool = True,
):
    """Find run-wise differences relative to comparison in a series with multi-index."""
    return (
        _ser.unstack()
        .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=0)
        .drop(index=([comparison] if drop_comparison else []))
        .stack()
    )


def find_mean_difference_in_appts_relative_to_comparison(
    _df: pd.DataFrame,
    comparison: str,
    drop_comparison: bool = True,
):
    """Find mean fewer appointments when treatment does not happen relative to comparison."""
    return -summarize(
        pd.concat(
            {
                _idx: find_difference_extra_relative_to_comparison(
                    row, comparison=comparison, drop_comparison=drop_comparison
                )
                for _idx, row in _df.iterrows()
            },
            axis=1,
        ).T,
        only_mean=True,
    )


def find_mean_difference_extra_relative_to_comparison_dataframe(
    _df: pd.DataFrame,
    comparison: str,
    drop_comparison: bool = True,
):
    """Same as find_difference_extra_relative_to_comparison but for dataframe."""
    return summarize(
        pd.concat(
            {
                _idx: find_difference_extra_relative_to_comparison(
                    row, comparison=comparison, drop_comparison=drop_comparison
                )
                for _idx, row in _df.iterrows()
            },
            axis=1,
        ).T,
        only_mean=True,
    )


def get_num_deaths_by_cause_label(_df: pd.DataFrame, target_period_tuple: tuple[Date, Date] = TARGET_PERIOD) -> pd.Series:
    """Return total deaths by label within target period."""
    return _df.loc[pd.to_datetime(_df.date).between(*target_period_tuple)].groupby(_df["label"]).size()


def get_num_dalys_by_cause_label(_df: pd.DataFrame, target_period_tuple: tuple[Date, Date] = TARGET_PERIOD) -> pd.Series:
    """Return total DALYS by label within target period."""
    return (
        _df.loc[_df.year.between(*[i.year for i in target_period_tuple])]
        .drop(columns=["date", "sex", "age_range", "year"])
        .sum()
    )


def make_get_num_deaths_by_cause_label_and_period(
    period_length_years: int,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
):
    """Create helper that summarizes deaths by cause and period chunks + overall."""
    periods = get_periods_within_target_period(
        period_length_years=period_length_years,
        target_period_tuple=target_period_tuple,
    )
    period_lookup = {
        year: period_label
        for period_label, (start_year, end_year) in periods
        for year in range(start_year, end_year + 1)
    }
    target_period_label = target_period(target_period_tuple)

    def _get_num_deaths_by_cause_label_and_period(_df: pd.DataFrame) -> pd.Series:
        _df_in_target = _df.loc[pd.to_datetime(_df.date).between(*target_period_tuple)].copy()
        _df_in_target["year"] = pd.to_datetime(_df_in_target["date"]).dt.year
        _df_in_target["period"] = _df_in_target["year"].map(period_lookup)

        chunked = _df_in_target.groupby(["label", "period"]).size()
        overall = _df_in_target.groupby("label").size()
        overall.index = pd.MultiIndex.from_arrays(
            [overall.index, np.repeat(target_period_label, len(overall.index))], names=["label", "period"]
        )
        return pd.concat([chunked, overall]).sort_index()

    return _get_num_deaths_by_cause_label_and_period


def make_get_num_dalys_by_cause_label_and_period(
    period_length_years: int,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
):
    """Create helper that summarizes DALYS by cause and period chunks + overall."""
    periods = get_periods_within_target_period(
        period_length_years=period_length_years,
        target_period_tuple=target_period_tuple,
    )
    period_lookup = {
        year: period_label
        for period_label, (period_start, period_end) in periods
        for year in range(period_start, period_end + 1)
    }
    start_year, end_year = target_period_tuple[0].year, target_period_tuple[1].year
    target_period_label = target_period(target_period_tuple)

    def _get_num_dalys_by_cause_label_and_period(_df: pd.DataFrame) -> pd.Series:
        _df_in_target = _df.loc[_df.year.between(start_year, end_year)].copy()
        _df_in_target["period"] = _df_in_target["year"].map(period_lookup)

        melted = (
            _df_in_target.drop(columns=["date", "sex", "age_range"])
            .melt(id_vars=["year", "period"], var_name="label", value_name="dalys")
        )
        chunked = melted.groupby(["label", "period"])["dalys"].sum()
        overall = melted.groupby("label")["dalys"].sum()
        overall.index = pd.MultiIndex.from_arrays(
            [overall.index, np.repeat(target_period_label, len(overall.index))], names=["label", "period"]
        )
        return pd.concat([chunked, overall]).sort_index()

    return _get_num_dalys_by_cause_label_and_period


def do_bar_plot_with_ci(
    _df: pd.DataFrame,
    _ax,
    period_labels_for_bar_plots: list[str],
    target_period_label: str,
):
    """Make vertical bars by cause, decomposed into period chunks, with overall-period CI."""
    _df_nothing = _df["Contraception_Routine"]
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


def get_num_deaths_by_age_group(
    _df: pd.DataFrame,
    age_grp_lookup: dict,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
):
    """Return total deaths by age-group in target period."""
    return (
        _df.loc[pd.to_datetime(_df.date).between(*target_period_tuple)]
        .groupby(_df["age"].map(age_grp_lookup).astype(make_age_grp_types()))
        .size()
    )


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


def get_total_num_death_by_agegrp_and_label(
    _df: pd.DataFrame,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
) -> pd.Series:
    """Return deaths in target period by age-group and cause label."""
    _df_limited_to_dates = _df.loc[_df["date"].between(*target_period_tuple)]
    age_group = to_age_group(_df_limited_to_dates["age"])
    return _df_limited_to_dates.groupby([age_group, "label"])["person_id"].size()


def get_total_num_dalys_by_agegrp_and_label(
    _df: pd.DataFrame,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
) -> pd.Series:
    """Return DALYS in target period by age-group and cause label."""
    return (
        _df.loc[_df.year.between(*[i.year for i in target_period_tuple])]
        .assign(age_group=_df["age_range"])
        .drop(columns=["date", "year", "sex", "age_range"])
        .melt(id_vars=["age_group"], var_name="label", value_name="dalys")
        .groupby(by=["age_group", "label"])["dalys"]
        .sum()
    )


def get_counts_of_hsi_by_short_treatment_id(
    _df: pd.DataFrame,
    target_period_tuple: tuple[Date, Date] = TARGET_PERIOD,
) -> pd.Series:
    """Get counts of short treatment ids occurring in target period."""
    _counts_by_treatment_id = (
        _df.loc[pd.to_datetime(_df["date"]).between(*target_period_tuple), "TREATMENT_ID"]
        .apply(pd.Series)
        .sum()
        .astype(int)
    )
    _short_treatment_id = _counts_by_treatment_id.index.map(lambda x: x.split("_")[0] + "*")
    return _counts_by_treatment_id.groupby(by=_short_treatment_id).sum()


def get_counts_of_appts(_df: pd.DataFrame, target_period_tuple: tuple[Date, Date] = TARGET_PERIOD) -> pd.Series:
    """Get counts of appointments of each type being used in target period."""
    return (
        _df.loc[pd.to_datetime(_df["date"]).between(*target_period_tuple), "Number_By_Appt_Type_Code"]
        .apply(pd.Series)
        .sum()
        .astype(int)
    )


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

    num_deaths_by_cause_label = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=extract_deaths_total,
            do_scaling=True,
        ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)[["Contraception_Routine"]]
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
        ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)[["Contraception_Routine"]]
    )

    fig, ax = plt.subplots()
    name_of_plot = f"Deaths With No Services, {target_period_label}"
    do_bar_plot_with_ci(num_deaths_by_cause_label / 1e3, ax, period_labels_for_bar_plots, target_period_label)
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
    do_bar_plot_with_ci(num_dalys_by_cause_label / 1e6, ax, period_labels_for_bar_plots, target_period_label)
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
    ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)

    total_num_dalys_by_agegrp_and_label = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked_by_age_and_time",
        custom_generate_series=lambda _df: get_total_num_dalys_by_agegrp_and_label(_df, TARGET_PERIOD),
        do_scaling=True,
    ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)

    counts_of_hsi_by_short_treatment_id = (
        extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="HSI_Event",
            custom_generate_series=lambda _df: get_counts_of_hsi_by_short_treatment_id(_df, TARGET_PERIOD),
            do_scaling=True,
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
