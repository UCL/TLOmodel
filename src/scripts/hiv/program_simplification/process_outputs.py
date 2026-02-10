
from __future__ import annotations

import datetime
from pathlib import Path

# import lacroix
from typing import Iterable, Sequence, Optional, Tuple, Union, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import ast
from collections import Counter, defaultdict

import seaborn as sns

from tlo import Date

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    make_age_grp_lookup,
    make_age_grp_types,
    compute_summary_statistics,
)

from scripts.costing.cost_estimation import (
    estimate_input_cost_of_scenarios, summarize_cost_data,
    do_stacked_bar_plot_of_cost_by_category, do_line_plot_of_cost,
    create_summary_treemap_by_cost_subgroup, estimate_projected_health_spending
)


outputspath = Path("./outputs/t.mangal@imperial.ac.uk")
# outputspath = Path("./outputs")

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("hiv_program_simplification", outputspath)[-1]


# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder, draw=1, run=1)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

TARGET_PERIOD = (Date(2025, 1, 1), Date(2050, 12, 31))


def target_period() -> str:
    """Returns the target period as a string of the form YYYY-YYYY"""
    return "-".join(str(t.year) for t in TARGET_PERIOD)


def get_parameter_names_from_scenario_file() -> Tuple[str]:
    """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
    from scripts.hiv.program_simplification.analysis_hiv_program_simplification import (
        HIV_Progam_Elements,
    )
    e = HIV_Progam_Elements()
    return tuple(e._scenarios.keys())

def set_param_names_as_column_index_level_0(_df):
    """Set the columns index (level 0) as the param_names."""
    ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
    assert len(names_of_cols_level0) == len(_df.columns.levels[0])
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df

param_names = get_parameter_names_from_scenario_file()


def find_difference_relative_to_comparison_series(
    _ser: pd.Series,
    comparison: str,
    scaled: bool = False,
    drop_comparison: bool = True,
):
    """Find the difference in the values in a pd.Series with a multi-index, between the draws (level 0)
    within the runs (level 1), relative to where draw = `comparison`.
    The comparison is `X - COMPARISON`."""
    return _ser \
        .unstack(level=0) \
        .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=1) \
        .drop(columns=([comparison] if drop_comparison else [])) \
        .stack()


def find_difference_relative_to_comparison_series_dataframe(_df: pd.DataFrame, **kwargs):
    """Apply `find_difference_relative_to_comparison_series` to each row in a dataframe"""
    return pd.concat({
        _idx: find_difference_relative_to_comparison_series(row, **kwargs)
        for _idx, row in _df.iterrows()
    }, axis=1).T



# set3_palette = sns.color_palette("Mako", 11).as_hex()
# colours = sns.color_palette("mako", n_colors=12)


colours = [
    "#3B4252",  # brighter slate
    "#5B708C",  # mid grey-blue
    "#4C89D9",  # vivid blue
    "#76A9E0",  # lighter sky
    "#4CCED9",  # aqua
    "#3FA7A3",  # muted teal
    "#66D9C1",  # mint
    "#9CD48C",  # fresh green
    "#F2D98D",  # golden sand
    "#F29E74",  # warm coral
    "#E06B75",  # lively crimson
    "#7A5FD6",  # royal violet (new; adds depth/contrast)
    "#C58CCB",  # pastel lilac
    "#A6A5FF",  # periwinkle (light but visible)
]



scenario_colours = dict(zip(param_names, colours))



def plot_with_ci(
    df,
    variable: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    colour_map: dict | None = None,
    percent_df=None,          # accepts Series or DF; wide (draw,stat) or flat
    ax=None,
    annotate_labels: list[str] | set[str] | None = None,
):

    def _pick_stat_name(cols):
        stats = list(pd.Index(cols.get_level_values("stat")).unique())
        for s in ("central", "mean", "median"):
            if s in stats:
                return s
        raise ValueError(f"No suitable statistic in 'stat' level; found {stats}")

    # --- normalise df to flat (index = draws; cols = central/lower/upper) ---
    if isinstance(df.columns, pd.MultiIndex) and "stat" in (df.columns.names or []):
        row = df.index[0] if variable is None else variable
        stat_c = _pick_stat_name(df.columns)
        central = df.xs(stat_c, axis=1, level="stat").loc[row]
        lower   = df.xs("lower", axis=1, level="stat").loc[row]
        upper   = df.xs("upper", axis=1, level="stat").loc[row]
        flat = pd.concat({"central": central, "lower": lower, "upper": upper}, axis=1)
    else:
        flat = df.copy()
        flat.columns = [str(c).lower() for c in flat.columns]
        if "central" not in flat.columns:
            if "mean" in flat.columns:   flat = flat.rename(columns={"mean": "central"})
            elif "median" in flat.columns: flat = flat.rename(columns={"median": "central"})
            else: raise ValueError("Flat input must have 'central' (or 'mean'/'median'), plus 'lower' and 'upper'.")
        for need in ("lower","upper"):
            if need not in flat.columns:
                raise ValueError(f"Missing column '{need}' in flat input.")

    central = flat["central"]
    lower   = flat["lower"]
    upper   = flat["upper"]
    yerr = [central - lower, upper - central]

    # --- percentages (optional) → Series indexed by draw, aligned ---
    percent_vals = None
    if percent_df is not None:
        if isinstance(percent_df, pd.DataFrame) and isinstance(percent_df.columns, pd.MultiIndex) and "stat" in (percent_df.columns.names or []):
            p_stat = _pick_stat_name(percent_df.columns)
            p = percent_df.xs(p_stat, axis=1, level="stat")
            if (variable is not None) and (variable in p.index): p = p.loc[variable]
            elif len(p.index) == 1: p = p.iloc[0]
            percent_vals = p.reindex(central.index)
        elif isinstance(percent_df, pd.DataFrame):
            q = percent_df.copy()
            q.columns = [str(c).lower() for c in q.columns]
            cname = "central" if "central" in q.columns else ("mean" if "mean" in q.columns else ("median" if "median" in q.columns else None))
            q = q[cname] if cname is not None else q.squeeze()
            percent_vals = q.reindex(central.index)
        else:  # Series
            percent_vals = percent_df.reindex(central.index)

    # --- colours ---
    colours = [colour_map.get(s, "grey") for s in central.index] if colour_map else None

    # --- plotting ---
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(central))
    bars = ax.bar(x, central.values, yerr=yerr, capsize=5, color=colours)
    ax.set_ylabel(ylabel if ylabel else (variable.replace("_"," ").title() if variable else "Value"))
    ax.set_title(title if title else (variable.replace("_"," ").title() if variable else ""))
    ax.set_xticks(x)
    ax.set_xticklabels(central.index, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axhline(0, color="grey", linewidth=1)

    if percent_vals is not None:
        # labels to annotate: by default annotate all; if provided, restrict
        allowed = set(central.index) if annotate_labels is None else set(annotate_labels) & set(central.index)

        rng = float(upper.max() - lower.min()) if len(upper) and len(lower) else 0.0
        offset = 0.05 * rng

        for xi, label, bar in zip(x, central.index, bars):
            if label not in allowed:
                continue
            v = percent_vals.get(label, np.nan)
            if pd.notna(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    upper[label] + offset,
                    f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=9
                )

    # y-limits with padding
    span = float(upper.max() - lower.min()) if len(upper) else 0.0
    pad = 0.2 * span
    ax.set_ylim((lower - pad).min(), (upper + pad).max())

    if created_fig:
        plt.tight_layout()
        plt.savefig(results_folder / f"{title}.png")
        plt.show()
    return ax




def lineplot_over_time_with_ci(
    df,
    param_names,
    scenario_colours,
    target_period,
    title=None,
    ax=None,
    alpha=0.25,
    linewidth=2,
    ylim=(0, 1),
):
    """
    Plot central ART coverage with confidence intervals for each scenario,
    restricted to a target period.
    """

    start_date, end_date = target_period
    df = df.loc[start_date:end_date]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    draws = df.columns.get_level_values(0).unique()
    draws = [d for d in param_names if d in draws]

    for draw in draws:
        central = df[(draw, "central")]
        lower   = df[(draw, "lower")]
        upper   = df[(draw, "upper")]

        colour = scenario_colours.get(draw, "grey")

        ax.plot(
            df.index,
            central,
            color=colour,
            linewidth=linewidth,
            label=draw,
        )

        ax.fill_between(
            df.index,
            lower,
            upper,
            color=colour,
            alpha=alpha,
            linewidth=0,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("ART coverage")
    if title is not None:
        ax.set_title(title, loc="center")

    ax.set_ylim(*ylim)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    ax.figure.subplots_adjust(right=0.78)

    return ax




# %% extract outputs

# number HIV tests
def get_num_tests(_df):
    """Return total number of adults tested across the TARGET_PERIOD.
    Throws error if data is missing for any year in TARGET_PERIOD.
    """
    _df["date"] = pd.to_datetime(_df["date"])
    years_needed = set(i.year for i in TARGET_PERIOD)
    recorded_years = set(_df["date"].dt.year.unique())
    assert recorded_years.issuperset(years_needed), "Some years are not recorded."

    mask = _df["date"].between(*TARGET_PERIOD)
    total_tested = _df.loc[mask, "number_adults_tested"].sum()

    return pd.Series(data={"number_adults_tested": total_tested})


def make_column_summariser(column_name: str):
    """
    Returns a function that computes the total for a specified column over the TARGET_PERIOD,
    validating that all years in the period are present.
    """

    def summariser(_df):
        _df["date"] = pd.to_datetime(_df["date"])
        years_needed = set(i.year for i in TARGET_PERIOD)
        recorded_years = set(_df["date"].dt.year.unique())
        assert recorded_years.issuperset(years_needed), "Some years are not recorded."

        mask = _df["date"].between(*TARGET_PERIOD)
        total = _df.loc[mask, column_name].sum()

        return pd.Series(data={column_name: total})

    return summariser


get_num_tests = make_column_summariser("number_adults_tested")

hiv_tests = compute_summary_statistics(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        custom_generate_series=get_num_tests,
        do_scaling=True
    ), central_measure="mean", use_standard_error=True
).pipe(set_param_names_as_column_index_level_0)


plot_with_ci(
    hiv_tests,
    variable="number_adults_tested",
    title=f"Number of Adults Tested for HIV {target_period()}",
    ylabel="Number of Adults Tested",
    colour_map=scenario_colours
)



# FSW PY on PrEP
get_prep_fsw_py = make_column_summariser("PY_PREP_ORAL_FSW")

py_fsw_prep = compute_summary_statistics(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        custom_generate_series=get_prep_fsw_py,
        do_scaling=True
    ), central_measure="mean", use_standard_error=True
).pipe(set_param_names_as_column_index_level_0)


plot_with_ci(
    py_fsw_prep,
    variable="PY_PREP_ORAL_FSW",
    title=f"Number of person-years on Oral PrEP for FSW {target_period()}",
    ylabel="Person-years on PrEP",
    colour_map=scenario_colours
)


# AGYW PY on PrEP
get_prep_agyw_py = make_column_summariser("PY_PREP_ORAL_AGYW")

py_agyw_prep = compute_summary_statistics(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        custom_generate_series=get_prep_agyw_py,
        do_scaling=True
    ), central_measure="mean", use_standard_error=True
).pipe(set_param_names_as_column_index_level_0)


plot_with_ci(
    py_agyw_prep,
    variable="PY_PREP_ORAL_AGYW",
    title=f"Number of person-years on Oral PrEP for AGYW {target_period()}",
    ylabel="Person-years on PrEP",
    colour_map=scenario_colours
)


# FSW PY on INJECTABLE PrEP
get_inj_prep_fsw_py = make_column_summariser("PY_PREP_INJ_FSW")

py_inj_fsw_prep = compute_summary_statistics(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        custom_generate_series=get_inj_prep_fsw_py,
        do_scaling=True
    ), central_measure="mean", use_standard_error=True
).pipe(set_param_names_as_column_index_level_0)


plot_with_ci(
    py_inj_fsw_prep,
    variable="PY_PREP_INJ_FSW",
    title=f"Number of person-years on Injectable PrEP for FSW {target_period()}",
    ylabel="Person-years on PrEP",
    colour_map=scenario_colours
)



# AGYW PY on Injectable PrEP
get_inj_prep_agyw_py = make_column_summariser("PY_PREP_INJ_AGYW")

py_inj_agyw_prep = compute_summary_statistics(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        custom_generate_series=get_inj_prep_agyw_py,
        do_scaling=True
    ), central_measure="mean", use_standard_error=True
).pipe(set_param_names_as_column_index_level_0)


plot_with_ci(
    py_inj_agyw_prep,
    variable="PY_PREP_INJ_AGYW",
    title=f"Number of person-years on Injectable PrEP for AGYW {target_period()}",
    ylabel="Person-years on PrEP",
    colour_map=scenario_colours
)


# num men circ
get_num_vmmc = make_column_summariser("N_NewVMMC")

num_men_circ = compute_summary_statistics(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        custom_generate_series=get_num_vmmc,
        do_scaling=True
    ), central_measure="mean", use_standard_error=True
).pipe(set_param_names_as_column_index_level_0)

plot_with_ci(
    num_men_circ,
    variable="N_NewVMMC",
    title=f"Number of VMMC performed {target_period()}",
    ylabel="Number new VMMC",
    colour_map=scenario_colours
)



# ART coverage
art_cov = compute_summary_statistics(extract_results(
        results_folder,
        module='tlo.methods.hiv',
        key='hiv_program_coverage',
        column="art_coverage_adult",
        index='date',
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0), central_measure='median')



ax = lineplot_over_time_with_ci(
    df=art_cov,
    param_names=param_names,
    scenario_colours=scenario_colours,
    target_period=TARGET_PERIOD,
    ylim=(0.6, 1),
    title='ART coverage in adults with HIV'
)
plt.show()


art_cov_full = extract_results(
        results_folder,
        module='tlo.methods.hiv',
        key='hiv_program_coverage',
        column="art_coverage_adult",
        index='date',
        do_scaling=False
    ).pipe(set_param_names_as_column_index_level_0)






# num tdf tests
get_num_tdf = make_column_summariser("n_tdf_tests_performed")

num_tdf = compute_summary_statistics(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        custom_generate_series=get_num_tdf,
        do_scaling=True
    ), central_measure="mean", use_standard_error=True
).pipe(set_param_names_as_column_index_level_0)


plot_with_ci(
    num_tdf,
    variable="n_tdf_tests_performed",
    title=f"Number of TDF Urine tests performed {target_period()}",
    ylabel="Number TDF tests",
    colour_map=scenario_colours
)


# Num new infections
get_num_inf = make_column_summariser("n_new_infections_adult_1549")

num_new_infections = compute_summary_statistics(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        custom_generate_series=get_num_inf,
        do_scaling=True,
    ), central_measure="mean", use_standard_error=True
).pipe(set_param_names_as_column_index_level_0)

plot_with_ci(
    num_new_infections,
    variable="n_new_infections_adult_1549",
    title=f"Number new HIV infections {target_period()}",
    ylabel="Number HIV infections",
    colour_map=scenario_colours
)


num_new_infections_full = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        custom_generate_series=get_num_inf
).pipe(set_param_names_as_column_index_level_0)



inc_diff_from_statusquo = compute_summary_statistics(
    find_difference_relative_to_comparison_series_dataframe(
        num_new_infections_full,
        comparison='Status Quo'
    ),
    central_measure="mean", use_standard_error=True
)


inc_diff_from_statusquo.to_excel(results_folder / "inc_diff_from_statusquo.xlsx")


pc_inc_diff_from_statusquo = 100.0 * compute_summary_statistics(
    pd.DataFrame(
        find_difference_relative_to_comparison_series_dataframe(
            num_new_infections,
            comparison='Status Quo',
            scaled=True)
    ), central_measure="mean", use_standard_error=True
)

pc_inc_diff_from_statusquo.to_excel(results_folder / "pc_inc_diff_from_statusquo.xlsx")


# Get the draw order from the original DataFrame
draw_order = num_new_infections.columns.get_level_values("draw").unique()

# Select the row, keep the (draw, stat) wide format, but reorder the draws
infect_ci = (
    inc_diff_from_statusquo
    .loc[["n_new_infections_adult_1549"]]
    .reindex(draw_order, axis=1, level="draw")
)

pc_infect_ci = (
    pc_inc_diff_from_statusquo
    .loc[["n_new_infections_adult_1549"]]
    .reindex(draw_order, axis=1, level="draw")
)



plot_with_ci(
    infect_ci,
    variable="n_new_infections_adult_1549",
    title=f"Difference in number new HIV infections {target_period()}",
    ylabel="Number HIV infections",
    percent_df=pc_infect_ci,
    colour_map=scenario_colours
)



# # 6MMD
#
# mmd_M = compute_summary_statistics(extract_results(
#         results_folder,
#         module="tlo.methods.hiv",
#         key="arv_dispensing_intervals",
#         column="group=adult_male|interval=6+",
#         index="date",
#         do_scaling=False
# ), central_measure="mean", use_standard_error=True
# )
#
#
# mmd_F = extract_results(
#         results_folder,
#         module="tlo.methods.hiv",
#         key="arv_dispensing_intervals",
#         column="group=adult_female|interval=6+",
#         index="date",
#         do_scaling=False
# )
#
#







# DALYs
def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
    Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
    results from runs that crashed mid-way through the simulation.
    """
    years_needed = [i.year for i in TARGET_PERIOD]
    assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
    return pd.Series(
        data=_df
        .loc[_df.year.between(*years_needed)]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum().sum()
    )


total_num_dalys = compute_summary_statistics(
    extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=get_num_dalys,
        do_scaling=True
    ), central_measure="mean", use_standard_error=True
).pipe(set_param_names_as_column_index_level_0)

plot_with_ci(
    total_num_dalys,
    variable=0,
    title=f"Number DALYs {target_period()}",
    ylabel="DALYs",
    colour_map=scenario_colours
)



num_dalys = extract_results(
    results_folder,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=get_num_dalys,
    do_scaling=True
).pipe(set_param_names_as_column_index_level_0)


dalys_diff_from_statusquo = compute_summary_statistics(
    pd.DataFrame(
        find_difference_relative_to_comparison_series(
            num_dalys.loc[0],
            comparison='Status Quo')
    ).T
).iloc[0].unstack().reindex(param_names).drop(['Status Quo'])
dalys_diff_from_statusquo.to_excel(results_folder / "dalys_diff_from_statusquo.xlsx")


pc_dalys_diff_from_statusquo = 100.0 * compute_summary_statistics(
    pd.DataFrame(
        find_difference_relative_to_comparison_series(
            num_dalys.loc[0],
            comparison='Status Quo',
            scaled=True)
    ).T
).iloc[0].unstack().reindex(param_names).drop(['Status Quo'])



plot_with_ci(
    df=dalys_diff_from_statusquo,
    title=f"Difference in All-Cause DALYs from Status Quo {target_period()}",
    ylabel="Difference from Status Quo",
    percent_df=pc_dalys_diff_from_statusquo,
    colour_map=scenario_colours
)





def num_dalys_by_cause(_df):
    """Return total number of DALYS (Stacked) (total by age-group within the TARGET_PERIOD)"""
    return _df \
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
        .drop(columns=['date', 'sex', 'age_range', 'year']) \
        .sum()

# extract dalys by cause with mean and upper/lower intervals
# With 'collapse_columns', if number of draws is 1, then collapse columns multi-index:

daly_by_cause = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True,
    ).pipe(set_param_names_as_column_index_level_0)
daly_by_cause.to_excel(results_folder / "daly_by_cause.xlsx")



dalys_labelled_diff_from_statusquo = compute_summary_statistics(
    find_difference_relative_to_comparison_series_dataframe(
        daly_by_cause,
        comparison='Status Quo'
    ),
    central_measure='mean'
)


dalys_labelled_diff_from_statusquo.to_excel(results_folder / "dalys_labelled_diff_from_statusquo.xlsx")


pc_dalys_diff_from_statusquo = 100.0 * compute_summary_statistics(
    pd.DataFrame(
        find_difference_relative_to_comparison_series_dataframe(
            daly_by_cause,
            comparison='Status Quo',
            scaled=True)
    )
)





# AIDS DALYS only
# Get the order of draws as they appear in the original dataframe
draw_order = daly_by_cause.columns.get_level_values("draw").unique()

aids_ci = dalys_labelled_diff_from_statusquo.loc['AIDS'].unstack('stat')
# Reorder the index
aids_ci = aids_ci.reindex(draw_order)

pc_aids_ci = pc_dalys_diff_from_statusquo.loc['AIDS'].unstack('stat')
pc_aids_ci = pc_aids_ci.reindex(draw_order)

aids_ci.to_excel(results_folder / "aids_dalys_diff_from_statusquo.xlsx")
pc_aids_ci.to_excel(results_folder / "pc_aids_dalys_diff_from_statusquo.xlsx")

# Plot
plot_with_ci(
    df=aids_ci,   # now a (draw × stat) wide frame with required cols
    title=f"Difference in AIDS DALYs from Status Quo {target_period()}",
    ylabel="Difference from Status Quo",
    percent_df=pc_aids_ci,
    colour_map=scenario_colours
)





# DALYs by cause
def get_total_num_dalys_by_label(_df):
    """Return the total number of DALYS in the TARGET_PERIOD by wealth and cause label."""
    y = _df \
        .loc[_df['year'].between(*[d.year for d in TARGET_PERIOD])] \
        .drop(columns=['date', 'year', 'li_wealth']) \
        .sum(axis=0)

    # define course cause mapper for HIV, TB, MALARIA and OTHER
    causes = {
        'AIDS': 'HIV/AIDS',
        '': 'Other',  # defined in order to use this dict to determine ordering of the causes in output
    }
    causes_relabels = y.index.map(causes).fillna('Other')

    return y.groupby(by=causes_relabels).sum()[list(causes.values())]


total_num_dalys_by_label_results = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_by_wealth_stacked_by_age_and_time",
    custom_generate_series=get_total_num_dalys_by_label,
    do_scaling=True,
).pipe(set_param_names_as_column_index_level_0)

diff_num_dalys_by_label_vs_statusquo = summarize(
    find_difference_relative_to_comparison_series_dataframe(
        total_num_dalys_by_label_results,
        comparison='Status Quo'
    ),
    only_mean=True
)








def plot_stacked_dalys_by_cause(
    df_by_cause: pd.DataFrame,
    param_names,
    title: str = "DALYs Averted by Scenario (Stacked by Cause)",
    ylabel: str = "Difference in DALYs from Status Quo",
    figsize: tuple = (12, 6),
    cause_palette: str = "Set1",
    percent_df: pd.DataFrame = None
):
    """
    Plot a stacked bar chart of DALYs averted by scenario and cause, with proper y-axis padding
    and optional percentage annotations.
    """
    param_names = list(param_names)
    ordered_names = [name for name in param_names if name in df_by_cause.columns]
    df_plot = df_by_cause[ordered_names]

    # Generate colours for each cause
    cause_colours = dict(zip(df_plot.index, sns.color_palette(cause_palette, n_colors=len(df_plot.index))))

    fig, ax = plt.subplots(figsize=figsize)
    bottom = pd.Series(0, index=ordered_names, dtype=float)

    cumulative_positive = pd.Series(0, index=ordered_names, dtype=float)
    cumulative_negative = pd.Series(0, index=ordered_names, dtype=float)

    for cause in df_plot.index:
        heights = df_plot.loc[cause]
        ax.bar(ordered_names, heights, bottom=bottom, label=cause, color=cause_colours[cause])
        cumulative_positive += heights.where(heights > 0, 0)
        cumulative_negative += heights.where(heights < 0, 0)
        bottom += heights

    # Adjust y-axis limits
    y_min = cumulative_negative.min()
    y_max = cumulative_positive.max()
    y_pad = 0.3 * (y_max - y_min if y_max != y_min else abs(y_max) or 1)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Annotate percentage difference (above stacked bar)
    if percent_df is not None:
        for scenario in ordered_names:
            pc = percent_df[(scenario, 'central')]
            total_height = cumulative_positive[scenario] if cumulative_positive[scenario] > 0 else cumulative_negative[scenario]
            offset = 0.02 * (y_max - y_min if y_max != y_min else abs(y_max) or 1)
            y_pos = total_height + offset if total_height > 0 else total_height - offset
            ax.text(scenario, y_pos, f"{pc:.1f}%", ha='center', va='bottom' if total_height > 0 else 'top', fontsize=9)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(ordered_names)))
    ax.set_xticklabels(ordered_names, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axhline(0, color="grey", linewidth=1)
    ax.legend(title="Cause", loc="upper right")

    plt.subplots_adjust(left=0.15)
    plt.tight_layout()
    plt.savefig(results_folder / f'{title}.png')
    plt.show()



plot_stacked_dalys_by_cause(
    df_by_cause=diff_num_dalys_by_label_vs_statusquo,
    param_names=param_names,
    title="Difference in DALYs Compared to Status Quo (by Cause)",
    ylabel="Difference in DALYs"
)






# Deaths
def extract_deaths_by_cause(results_folder):
    """ returns mean deaths for each year of the simulation
    values are aggregated across the runs of each draw
    for the specified cause
    """

    def get_num_deaths_by_cause_label(_df):
        """Return total number of Deaths by label within the TARGET_PERIOD
        values are summed for all ages
        df returned: rows=COD, columns=draw
        """
        return _df \
            .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .groupby(_df['label']) \
            .size()

    num_deaths_by_label = extract_results(
        results_folder,
        module='tlo.methods.demography',
        key='death',
        custom_generate_series=get_num_deaths_by_cause_label,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    causes = {
        'AIDS': 'HIV/AIDS',
        '': 'Other',
    }

    causes_relabels = num_deaths_by_label.index.map(causes).fillna('Other')

    grouped_deaths = num_deaths_by_label.groupby(causes_relabels).sum()
    # Reorder based on the causes keys that are in the grouped data
    ordered_causes = [cause for cause in causes.values() if cause in grouped_deaths.index]
    test = grouped_deaths.reindex(ordered_causes)

    return test


num_deaths_by_cause = extract_deaths_by_cause(results_folder)

summary_num_deaths_by_cause = compute_summary_statistics(extract_deaths_by_cause(results_folder),
                                                 central_measure='mean')

summary_num_deaths_by_cause.to_csv(results_folder / f'num_deaths_by_cause_{target_period()}.csv')


num_deaths_averted = compute_summary_statistics(
    find_difference_relative_to_comparison_series_dataframe(
        num_deaths_by_cause,
        comparison='Status Quo'
    ), central_measure='mean'
)

num_deaths_averted.to_csv(results_folder / f'num_deaths_averted_{target_period()}.csv')

pc_deaths_averted = 100.0 * compute_summary_statistics(
    pd.DataFrame(
        find_difference_relative_to_comparison_series_dataframe(
            num_deaths_by_cause,
            comparison='Status Quo',
            scaled=True)
    ), central_measure='mean',
)

pc_deaths_averted.to_csv(results_folder / f'pc_deaths_averted_{target_period()}.csv')



# AIDS deaths only
# Get the order of draws as they appear in the original dataframe
aids_deaths_ci = num_deaths_averted.loc['HIV/AIDS'].unstack('stat')
aids_deaths_ci = aids_deaths_ci.reindex(draw_order)

pc_aids_deaths_ci = pc_deaths_averted.loc['HIV/AIDS'].unstack('stat')
pc_aids_deaths_ci = pc_aids_deaths_ci.reindex(draw_order)

aids_deaths_ci.to_csv(results_folder / f'aids_deaths_averted{target_period()}.csv')
pc_aids_deaths_ci.to_csv(results_folder / f'pc_aids_deaths_averted_{target_period()}.csv')


# Plot
plot_with_ci(
    df=aids_deaths_ci,   # now a (draw × stat) wide frame with required cols
    title=f"Difference in AIDS deaths from Status Quo {target_period()}",
    ylabel="Difference from Status Quo",
    percent_df=pc_aids_deaths_ci,
    colour_map=scenario_colours,
)







# multi-panel plot
# remove first columns from some df
aids_ci_edit = aids_ci.drop("Status Quo")
aids_deaths_ci_edit = aids_deaths_ci.drop("Status Quo")



fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# annotate labels selects which bars to give % values, if overlap 0 then don't annotate
plot_with_ci(
    df=infect_ci,
    variable="n_new_infections_adult_1549",
    title=f"Difference in number new HIV infections {target_period()}",
    ylabel="Number HIV infections",
    percent_df=pc_infect_ci,
    colour_map=scenario_colours,
    ax=axes[0],
    annotate_labels={'Reduce HIV testing',
                     'Remove Viral Load Testing',
                     'Target Viral Load Testing',
                     'Replace Viral Load Testing',
                     'Remove VMMC',
                     'Target All Elements',
                     'Reduce All Elements',
                     'Program Scale-up'}
)


plot_with_ci(
    df=(aids_ci_edit / 1_000_000),
    variable="AIDS",   # row name in that table
    title=f"Difference in AIDS DALYs from Status Quo {target_period()}",
    ylabel="Difference from Status Quo, millions",
    percent_df=pc_aids_ci,
    colour_map=scenario_colours,
    ax=axes[1],
    annotate_labels={'Reduce HIV testing',
                     'Remove Viral Load Testing',
                     'Target Viral Load Testing',
                     'Replace Viral Load Testing',
                     'Target All Elements',
                     'Reduce All Elements',
                     'Program Scale-up'}
)


plot_with_ci(
    df=aids_deaths_ci_edit / 1000,
    variable="HIV/AIDS",   # row name in that table
    title=f"Difference in AIDS deaths from Status Quo {target_period()}",
    ylabel="Difference from Status Quo, thousands",
    percent_df=pc_aids_deaths_ci,
    colour_map=scenario_colours,
    ax=axes[2],
    annotate_labels={'Reduce HIV testing',
                     'Remove Viral Load Testing',
                     'Target Viral Load Testing',
                     'Replace Viral Load Testing',
                     'Target All Elements',
                     'Reduce All Elements',
                     'Program Scale-up'}
)

# Only hide top two labels
for ax in axes[:-1]:
    ax.tick_params(axis="x", labelbottom=False)

fig.tight_layout()
fig.subplots_adjust(bottom=0.2)   # enough room for rotation
plt.savefig(results_folder / "infections_dalys_deaths_vs_baseline.png")

plt.show()







#%% ####################### HS use  #######################

# get numbers by treatment ID by year
# map to expected facility level -> list of treatment IDs by facility level by year
# map to appt type by facility level
# map to cadre time required
# then can sum for the plots


#########################
# extract treatment id
#########################

# extract numbers of appts delivered for every run within a specified draw
def _parse_dict_cell(x) -> Dict[str, float]:
    if isinstance(x, dict):
        return x
    if pd.isna(x):
        return {}
    if isinstance(x, str):
        return ast.literal_eval(x)
    return {}



def make_series_treatment_counts_by_year(
    treatment_col: str = "TREATMENT_ID",
    date_col: str = "date",
    TARGET_PERIOD: Optional[Tuple[object, object]] = None,
):
    """
    Returns a function suitable for `custom_generate_series`.

    Output Series:
      - index: MultiIndex (year, treatment_id)
      - values: counts summed within year
    """
    def custom_generate_series(df: pd.DataFrame) -> pd.Series:

        # Filter to target period (your established pattern)
        if TARGET_PERIOD is not None:
            df = df.loc[pd.to_datetime(df.date).between(*TARGET_PERIOD)]

        if df.empty:
            return pd.Series(dtype=float)

        years = pd.to_datetime(df[date_col]).dt.year

        by_year: Dict[int, Counter] = {}

        for yr, cell in zip(years, df[treatment_col]):
            d = _parse_dict_cell(cell)
            by_year.setdefault(int(yr), Counter()).update(d)

        wide = pd.DataFrame({yr: dict(cnt) for yr, cnt in by_year.items()}).T
        wide.index.name = "year"
        wide = wide.sort_index()

        s = wide.stack(dropna=False)
        s.index.names = ["year", "treatment_id"]
        s.name = "count"

        return s

    return custom_generate_series


module = "tlo.methods.healthsystem.summary"
key = "HSI_Event"

custom = make_series_treatment_counts_by_year(
    treatment_col="TREATMENT_ID",
    TARGET_PERIOD=TARGET_PERIOD,
)

# treatment_by_year is a DataFrame with:
#   - index: MultiIndex (year, treatment_id)
#   - columns: MultiIndex (draw, run)
#   - values: counts
treatment_by_year = extract_results(
    results_folder=results_folder,
    module=module,
    key=key,
    custom_generate_series=custom,
    do_scaling=True,
).pipe(set_param_names_as_column_index_level_0)

treatment_by_year.to_excel(results_folder / "treatment_id_counts_by_year_draw_run.xlsx")

treatment_by_year_hiv = treatment_by_year.loc[
    treatment_by_year.index
        .get_level_values("treatment_id")
        .str.startswith("Hiv")
]
treatment_by_year_hiv.to_excel(results_folder / "treatment_by_year_hiv.xlsx")


#########################
# extract appt types
#########################


# get appt types by facility level and year
def summarise_appointments(df: pd.DataFrame) -> pd.Series:
    """
    Sum appointment type counts by calendar year *and facility level* (within TARGET_PERIOD)
    from the HSI_Event log.

    Returns
    -------
    pd.Series
        MultiIndex (year, facility_level, AppointmentTypeCode) -> total count
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])

    mask = d["date"].between(*TARGET_PERIOD)
    d = d.loc[mask, ["date", "Number_By_Appt_Type_Code_And_Level"]]

    # Collect long-form rows: (date, year, level, appt_type, count)
    rows = []
    for dt, level_dict in zip(d["date"].values, d["Number_By_Appt_Type_Code_And_Level"].values):
        if not isinstance(level_dict, dict) or len(level_dict) == 0:
            continue
        year = pd.Timestamp(dt).year

        for level, appt_dict in level_dict.items():
            if not isinstance(appt_dict, dict) or len(appt_dict) == 0:
                continue
            for appt_type, count in appt_dict.items():
                if count is None:
                    continue
                rows.append((year, str(level), appt_type, float(count)))

    if not rows:
        # Empty, but keep the expected index names
        return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], [], []],
                                                                      names=["year", "facility_level", "AppointmentTypeCode"]))

    long = pd.DataFrame(rows, columns=["year", "facility_level", "AppointmentTypeCode", "count"])

    out = (
        long.groupby(["year", "facility_level", "AppointmentTypeCode"], sort=False)["count"]
        .sum()
    )
    out.index = out.index.set_names(["year", "facility_level", "AppointmentTypeCode"])
    return out


appt_counts = (
    extract_results(
        results_folder=results_folder,
        module="tlo.methods.healthsystem.summary",
        key="HSI_Event_non_blank_appt_footprint",
        custom_generate_series=summarise_appointments,
        do_scaling=True,  # scale to national population
    )
    .pipe(set_param_names_as_column_index_level_0)
)


# remove unneeded appt types, keep only those from hiv program

# Explicit allow-list of appointment types to KEEP
KEEP_APPT_TYPES = [
    "VCTPositive",
    "VCTNegative",
    "NewAdult",
    "Peds",
    "EstNonCom",
    "MaleCirc",
]


################################
# map treatment id to appt types
################################


TREATMENT_TO_APPT_SPEC = {
    "PharmDispensing": ("Hiv_Test_Selftest",      "1a",  1.0),
    "ConWithDCSA":     ("Hiv_Prevention_Prep",    "0",   1.0),
    "IPAdmission":     ("Hiv_PalliativeCare",     "2",   2.0),
    "InpatientDays":   ("Hiv_PalliativeCare",     "2",  17.0),
}

def keep_selected_appt_types(
    appt_counts: pd.Series | pd.DataFrame,
    appt_types_to_keep: list[str],
    *,
    appt_level: str = "AppointmentTypeCode",
) -> pd.Series | pd.DataFrame:
    """
    Keep only selected appointment types (across all years and facility levels).
    """
    idx = appt_counts.index
    mask_keep = idx.get_level_values(appt_level).isin(appt_types_to_keep)
    return appt_counts.loc[mask_keep].copy()


def add_mapped_treatments_as_appt_types(
    appt_counts_base: pd.DataFrame,
    trt_counts: pd.DataFrame,
    mapping_appt_to_spec: dict[str, tuple[str, str, float]],
    *,
    year_level: str = "year",
    trt_level: str = "treatment_id",
) -> pd.DataFrame:
    """
    Sparse behaviour:
      - does NOT create a full year×level×type grid
      - only creates rows for the (year, specified facility_level, appt_type) that are inserted
      - overwrites if those rows already exist
    """
    out = appt_counts_base.copy()

    trt = trt_counts.copy().sort_index()

    for appt_type, (trt_id, facility_level, mult) in mapping_appt_to_spec.items():
        # Select treatment counts indexed by year
        trt_block = trt.xs(trt_id, level=trt_level, drop_level=True).copy()
        trt_block = trt_block * float(mult)

        years = trt_block.index.get_level_values(year_level)

        target_index = pd.MultiIndex.from_arrays(
            [
                years,
                pd.Index([str(facility_level)] * len(years)),
                pd.Index([appt_type] * len(years)),
            ],
            names=["year", "facility_level", "AppointmentTypeCode"],
        )

        src = trt_block.copy()
        src.index = target_index

        # Add rows as needed, then assign
        out = out.reindex(out.index.union(target_index))
        out.loc[target_index, :] = src

    return out


# ----
appt_counts_kept = keep_selected_appt_types(appt_counts, KEEP_APPT_TYPES)

appt_counts_final = add_mapped_treatments_as_appt_types(
    appt_counts_kept,
    treatment_by_year_hiv,
    TREATMENT_TO_APPT_SPEC,
)


################################
# map appt numbers to HCW time
################################

# get HCW time by mapping appts to person-time
hcw_time = pd.read_csv("resources/healthsystem/human_resources/definitions/ResourceFile_Appt_Time_Table.csv")



def appt_counts_to_hcw_minutes(appt_counts_final: pd.DataFrame,
                              hcw_time: pd.DataFrame,
                              fill_missing_minutes: float | None = None) -> pd.DataFrame:
    """
    Returns a DataFrame with the SAME columns as appt_counts_final, and an expanded MultiIndex:
      ['year','facility_level','AppointmentTypeCode','hcw_cadre']
    Values are: (number of appointments) × (minutes per appointment for that cadre at that facility level).
    """

    # --- normalise keys to strings for exact matching ---
    counts = appt_counts_final.copy()
    counts = counts.rename_axis(index={
        "facility_level": "Facility_Level",
        "AppointmentTypeCode": "Appt_Type_Code"
    })

    # Ensure index levels are string (facility/appt)
    counts_idx = counts.index
    counts = counts.copy()
    counts.index = pd.MultiIndex.from_arrays(
        [
            counts_idx.get_level_values("year"),
            counts_idx.get_level_values("Facility_Level").astype(str).str.strip(),
            counts_idx.get_level_values("Appt_Type_Code").astype(str).str.strip(),
        ],
        names=["year", "Facility_Level", "Appt_Type_Code"]
    )

    hcw = hcw_time.copy()
    hcw["Facility_Level"] = hcw["Facility_Level"].astype(str).str.strip()
    hcw["Appt_Type_Code"] = hcw["Appt_Type_Code"].astype(str).str.strip()
    hcw["Officer_Category"] = hcw["Officer_Category"].astype(str).str.strip()

    # --- build minutes-per-appointment map: (Facility_Level, Appt_Type_Code) × cadre ---
    map_table = hcw.pivot_table(
        index=["Facility_Level", "Appt_Type_Code"],
        columns="Officer_Category",
        values="Time_Taken_Mins",
        aggfunc="mean"
    )

    cadres = list(map_table.columns)

    # Target index for aligning minutes with counts rows (drop year)
    target = pd.MultiIndex.from_arrays(
        [
            counts.index.get_level_values("Facility_Level"),
            counts.index.get_level_values("Appt_Type_Code"),
        ],
        names=["Facility_Level", "Appt_Type_Code"]
    )

    out_parts = []
    nrows = len(counts)

    # Multiply counts (nrows × ncols) by mins_vec (nrows) per cadre
    for cadre in cadres:
        mins_c = map_table[cadre].reindex(target)  # Series aligned to counts rows (year-dropped index)

        if fill_missing_minutes is not None:
            mins_c = mins_c.fillna(fill_missing_minutes)

        # broadcasting across columns
        df_c = counts.mul(mins_c.to_numpy(), axis=0)

        # expand index with cadre level
        df_c.index = pd.MultiIndex.from_arrays(
            [
                counts.index.get_level_values("year"),
                counts.index.get_level_values("Facility_Level"),
                counts.index.get_level_values("Appt_Type_Code"),
                pd.Index([cadre] * nrows)
            ],
            names=["year", "facility_level", "AppointmentTypeCode", "hcw_cadre"]
        )

        out_parts.append(df_c)

    out = pd.concat(out_parts).sort_index()
    return out


# --- usage ---
minutes_by_appt_and_cadre = appt_counts_to_hcw_minutes(appt_counts_final, hcw_time, fill_missing_minutes=0.0)
minutes_by_appt_and_cadre.to_excel(results_folder / "minutes_by_appt_and_cadre.xlsx")


minutes_per_year_summed_by_cadre = (
    minutes_by_appt_and_cadre
    .groupby(level=["year", "hcw_cadre"])
    .sum()
)
minutes_per_year_summed_by_cadre.to_excel(results_folder / "minutes_per_year_summed_by_cadre.xlsx")

hours_per_year_cadre = minutes_per_year_summed_by_cadre / 60
hours_per_year_cadre.to_excel(results_folder / "hours_per_year_cadre.xlsx")

minutes_all_years_summed_by_cadre = (
    minutes_by_appt_and_cadre
    .groupby(level=["hcw_cadre"])
    .sum()
)
minutes_all_years_summed_by_cadre.to_excel(results_folder / "minutes_all_years_summed_by_cadre.xlsx")
hours_all_years_cadre = minutes_all_years_summed_by_cadre / 60
hours_all_years_cadre.to_excel(results_folder / "hours_all_years_cadre.xlsx")


################################
# add costs to HCW time
################################

# read in the HRH cost sheet
hcw_costs = pd.read_csv("resources/ResourceFile_HIV/hrh_costs.csv")


def hcw_time_costs_by_facility(
    minutes_by_appt_and_cadre: pd.DataFrame,
    hcw_costs: pd.DataFrame,
    minutes_to_hours: bool = True,
    fill_missing_cost: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute HCW time (hours) and costs by year × facility_level × cadre
    preserving columns (draw/run/scenario)
    """

    # --- 1) aggregate minutes to year × facility × cadre ---
    mins_yfc = (
        minutes_by_appt_and_cadre
        .groupby(level=["year", "facility_level", "hcw_cadre"])
        .sum()
    )

    # --- 2) convert to hours ---
    hours_yfc = mins_yfc / 60 if minutes_to_hours else mins_yfc.copy()

    # --- 3) prep cost lookup keyed on (facility_level, cadre) ---
    costs = hcw_costs.copy()
    costs["Facility_Level"] = costs["Facility_Level"].astype(str).str.strip()
    costs["Officer_Category"] = costs["Officer_Category"].astype(str).str.strip()

    # If duplicates exist per (Facility_Level, Officer_Category), average them
    cost_lookup = (
        costs
        .groupby(["Facility_Level", "Officer_Category"], as_index=True)["Total_hourly_cost"]
        .mean()
    )

    # --- 4) align lookup to hours_yfc rows ---
    idx = hours_yfc.index
    fac = idx.get_level_values("facility_level").astype(str).str.strip()
    cad = idx.get_level_values("hcw_cadre").astype(str).str.strip()

    hourly_cost_aligned = pd.MultiIndex.from_arrays(
        [fac, cad],
        names=["Facility_Level", "Officer_Category"]
    ).map(cost_lookup)

    if fill_missing_cost is not None:
        hourly_cost_aligned = hourly_cost_aligned.fillna(fill_missing_cost)

    # --- 5) cost = hours × hourly_cost (row-wise) ---
    costs_yfc = hours_yfc.mul(hourly_cost_aligned.to_numpy(), axis=0)

    return hours_yfc, costs_yfc


# --- usage ---
hours_by_year_fac_cadre, costs_by_year_fac_cadre = hcw_time_costs_by_facility(
    minutes_by_appt_and_cadre=minutes_by_appt_and_cadre,
    hcw_costs=hcw_costs,
    minutes_to_hours=True,
    fill_missing_cost=None   # or 0.0 if you want unmapped costs to contribute zero
)

hours_by_year_fac_cadre.to_excel(results_folder / "hours_by_year_fac_cadre.xlsx")
costs_by_year_fac_cadre.to_excel(results_folder / "costs_by_year_fac_cadre.xlsx")


# get the costs for all hcw by year
total_costs_by_year = (
    costs_by_year_fac_cadre
    .groupby(level="year")
    .sum()
)
total_costs_by_year.to_excel(results_folder / "total_costs_by_year.xlsx")

# get total costs across all the years
total_costs_all_years = costs_by_year_fac_cadre.sum()
total_costs_all_years.to_excel(results_folder / "total_costs_all_years.xlsx")


# total_costs_all_years is a Series with MultiIndex (e.g. run, draw) or (draw, run)
total_costs_all_years_df = total_costs_all_years.to_frame().T


cost_diff_from_statusquo = compute_summary_statistics(
    find_difference_relative_to_comparison_series_dataframe(
        total_costs_all_years_df,
        comparison="Status Quo"
    ), central_measure='mean',
)

cost_diff_from_statusquo.to_excel(results_folder / "cost_diff_from_statusquo.xlsx")


pc_cost_diff_from_statusquo = 100.0 * compute_summary_statistics(
    pd.DataFrame(
        find_difference_relative_to_comparison_series_dataframe(
            total_costs_all_years_df,
            comparison='Status Quo',
        scaled=True)
    ), only_central=True
)
pc_cost_diff_from_statusquo.to_excel(results_folder / "pc_cost_diff_from_statusquo.xlsx")




###############################################################################
# PLOTS - infections / deaths vs HCW hours

def plot_cadre_hours_vs_outcomes(
    num_hcw_hours_diff_edit,
    epi_df,
    scenario_colours: dict | None = None,
    cadres=("Clinical", "Nursing_and_Midwifery", "Pharmacy"),
    figsize=(10, 12),
    ylabel=None,
    exclude_scenarios: list[str] | None = None,
):
    """
    Panel plot: scatter cadre hours (x) vs new infections (y) with asymmetric CI whiskers.
    Includes one legend for scenarios across the figure.
    """

    # Extract infections (assume single row)
    y_c = epi_df.xs("central", axis=1, level="stat").iloc[0]
    y_l = epi_df.xs("lower",   axis=1, level="stat").iloc[0]
    y_u = epi_df.xs("upper",   axis=1, level="stat").iloc[0]

    fig, axes = plt.subplots(len(cadres), 1, figsize=figsize, sharex=False)

    if len(cadres) == 1:
        axes = [axes]

    # Use draw order from the first cadre’s row
    first_cadre = cadres[0]
    draws_order = num_hcw_hours_diff_edit.xs("central", axis=1, level="stat").loc[first_cadre].index

    # Drop any excluded scenarios
    if exclude_scenarios:
        draws_order = [d for d in draws_order if d not in exclude_scenarios]

    for ax, cadre in zip(axes, cadres):
        x_c = num_hcw_hours_diff_edit.xs("central", axis=1, level="stat").loc[cadre]
        x_l = num_hcw_hours_diff_edit.xs("lower",   axis=1, level="stat").loc[cadre]
        x_u = num_hcw_hours_diff_edit.xs("upper",   axis=1, level="stat").loc[cadre]

        yc = y_c.reindex(draws_order)
        yl = y_l.reindex(draws_order)
        yu = y_u.reindex(draws_order)

        for d in draws_order:
            xc, xm, xp = x_c[d], x_c[d] - x_l[d], x_u[d] - x_c[d]
            yc_d, ym, yp = yc[d], yc[d] - yl[d], yu[d] - yc[d]
            colour = scenario_colours.get(d, "grey") if scenario_colours else "C0"

            ax.errorbar(
                xc, yc_d,
                xerr=[[xm], [xp]], yerr=[[ym], [yp]],
                fmt='o', capsize=3,
                color=colour, ecolor=colour, elinewidth=1, alpha=0.9,
            )

        ax.set_title(cadre)
        ax.set_xlabel("Hours (difference)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.axhline(0, color="grey", linewidth=1, alpha=0.6)
        ax.axvline(0, color="grey", linewidth=1, alpha=0.6)

    # Proxy legend (robust, independent of errorbar handles)
    handles = [
        mlines.Line2D([], [], color=scenario_colours.get(d, "grey"),
                      marker='o', linestyle='None', markersize=8, label=d)
        for d in draws_order
    ]

    # reserve space on the right so the legend doesn’t overlap
    fig.tight_layout(rect=(0, 0, 0.7, 1))  # 80% width for axes, 20% for legend

    fig.legend(handles=handles, title="Scenario",
               loc="center left", bbox_to_anchor=(0.7, 0.5),
               bbox_transform=fig.transFigure, frameon=False)
    plt.savefig(results_folder / f'num_hcw_hours_diff_{ylabel}_{target_period()}.png')
    plt.show()


# todo update this with the new dfs
num_hcw_hours_diff = compute_summary_statistics(
    find_difference_relative_to_comparison_series_dataframe(
        hours_all_years_cadre,
        comparison='Status Quo'
    ), central_measure='mean'
)


num_hcw_hours_diff.to_csv(results_folder / f'num_hcw_hours_diff_{target_period()}.csv')

draw_order = hours_all_years_cadre.columns.get_level_values("draw").unique()

# Reorder the index
num_hcw_hours_diff = num_hcw_hours_diff.reindex(draw_order, axis=1, level="draw")

# remove Status Quo columns
num_hcw_hours_diff_edit = num_hcw_hours_diff.drop(columns="Status Quo", level="draw")



plot_cadre_hours_vs_outcomes(
    num_hcw_hours_diff_edit=num_hcw_hours_diff_edit,
    epi_df=infect_ci,
    scenario_colours=scenario_colours,
    cadres=("Clinical", "Nursing_and_Midwifery", "Pharmacy"),
    figsize=(10, 11),
    ylabel="New HIV infections (difference)",
    exclude_scenarios=["Program Scale-up"]
)



# for deaths vc HCW time
epi_df = aids_deaths_ci.stack().to_frame().T
epi_df.columns.names = ["draw", "stat"]
# optional: name the single row (to be explicit)
epi_df.index = ["n_aids_deaths"]

plot_cadre_hours_vs_outcomes(
    num_hcw_hours_diff_edit=num_hcw_hours_diff_edit,
    epi_df=epi_df,
    scenario_colours=scenario_colours,
    cadres=("Clinical", "Nursing_and_Midwifery", "Pharmacy"),
    figsize=(10, 11),
    ylabel="AIDS deaths",
    exclude_scenarios=["Program Scale-up"]
)




# epi outputs vs cost differences

def plot_cost_vs_outcomes(
    num_hcw_hours_diff_edit,
    epi_df,
    scenario_colours: dict | None = None,
    figsize=(10, 6),
    ylabel=None,
    exclude_scenarios: list[str] | None = None,
):
    """
    Scatter: Costs(x) vs epidemiological outcome (y) with asymmetric CI whiskers.
    """

    # --- y (epi) (assume single row) ---
    y_c = epi_df.xs("central", axis=1, level="stat").iloc[0]
    y_l = epi_df.xs("lower",   axis=1, level="stat").iloc[0]
    y_u = epi_df.xs("upper",   axis=1, level="stat").iloc[0]

    # --- x (hours) (assume single row) ---
    x_c = num_hcw_hours_diff_edit.xs("central", axis=1, level="stat").iloc[0]
    x_l = num_hcw_hours_diff_edit.xs("lower",   axis=1, level="stat").iloc[0]
    x_u = num_hcw_hours_diff_edit.xs("upper",   axis=1, level="stat").iloc[0]

    # Use scenario/draw order from x_c index
    draws_order = list(x_c.index)

    # Drop any excluded scenarios
    if exclude_scenarios:
        draws_order = [d for d in draws_order if d not in exclude_scenarios]

    # Align y to the same order
    yc = y_c.reindex(draws_order)
    yl = y_l.reindex(draws_order)
    yu = y_u.reindex(draws_order)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for d in draws_order:
        xc = x_c[d]
        xm = x_c[d] - x_l[d]
        xp = x_u[d] - x_c[d]

        yc_d = yc[d]
        ym = yc[d] - yl[d]
        yp = yu[d] - yc[d]

        colour = scenario_colours.get(d, "grey") if scenario_colours else "C0"

        ax.errorbar(
            xc, yc_d,
            xerr=[[xm], [xp]], yerr=[[ym], [yp]],
            fmt="o", capsize=3,
            color=colour, ecolor=colour, elinewidth=1, alpha=0.9,
        )

    ax.set_title(ylabel)
    ax.set_xlabel("Cost (difference)")
    ax.set_ylabel(ylabel if ylabel is not None else "")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axhline(0, color="grey", linewidth=1, alpha=0.6)
    ax.axvline(0, color="grey", linewidth=1, alpha=0.6)

    # Proxy legend (robust, independent of errorbar handles)
    if scenario_colours:
        handles = [
            mlines.Line2D(
                [], [], color=scenario_colours.get(d, "grey"),
                marker="o", linestyle="None", markersize=8, label=d
            )
            for d in draws_order
        ]
        # reserve space on the right so the legend doesn’t overlap
        fig.tight_layout(rect=(0, 0, 0.7, 1))
        fig.legend(
            handles=handles, title="Scenario",
            loc="center left", bbox_to_anchor=(0.7, 0.5),
            bbox_transform=fig.transFigure, frameon=False
        )
    else:
        fig.tight_layout()

    # NOTE: results_folder/target_period() must exist in your outer scope (as in your original)
    plt.savefig(results_folder / f'cost_diff_{ylabel}_{target_period()}.png')
    plt.show()




plot_cost_vs_outcomes(
    num_hcw_hours_diff_edit=cost_diff_from_statusquo,
    epi_df=infect_ci,
    scenario_colours=scenario_colours,
    figsize=(10, 11),
    ylabel="New HIV infections (difference)",
    exclude_scenarios=["Program Scale-up"]
)


plot_cost_vs_outcomes(
    num_hcw_hours_diff_edit=cost_diff_from_statusquo,
    epi_df=epi_df,
    scenario_colours=scenario_colours,
    figsize=(10, 11),
    ylabel="New HIV infections (difference)",
    exclude_scenarios=["Program Scale-up"]
)


####################################################################################
#%% get unit costs
####################################################################################


resourcefilepath = Path("./resources")

# Period relevant for costing
relevant_period_for_costing = [i.year for i in TARGET_PERIOD]
list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0],  relevant_period_for_costing[1] + 1))
list_of_years_for_plot = list(range(2024, 2051))
number_of_years_costed = relevant_period_for_costing[1] - 2024 + 1

# Costing parameters
discount_rate = 0.03

# todo costing scripts throwing errors
# todo perhaps my logs files out of date compared with costing scripts
# todo get consumables item codes + quantities and assign costs


# Estimate standard input costs of scenario
# -----------------------------------------------------------------------------------------------------------------------
# Standard 3% discount rate
input_costs = estimate_input_cost_of_scenarios(results_folder, resourcefilepath,
                                               _years=list_of_relevant_years_for_costing,
                                               cost_only_used_staff=False,
                                               _discount_rate=discount_rate,
                                               summarize=True)






# Undiscounted costs
input_costs_undiscounted = estimate_input_cost_of_scenarios(results_folder, resourcefilepath,
                                               _years=list_of_relevant_years_for_costing, cost_only_used_staff=True,
                                               _discount_rate=0, summarize=False)



