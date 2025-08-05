

import datetime
from pathlib import Path

# import lacroix
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from typing import Tuple

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

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")
# outputspath = Path("./outputs")

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("hiv_program_simplification", outputspath)[-1]


# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

#             module_param                value
# draw
# 0     Hiv:type_of_scaleup                 none
# 1     Hiv:type_of_scaleup      reduce_HIV_test
# 2     Hiv:type_of_scaleup            remove_VL
# 3     Hiv:type_of_scaleup  replace_VL_with_TDF
# 4     Hiv:type_of_scaleup      remove_prep_fsw
# 5     Hiv:type_of_scaleup     remove_prep_agyw
# 6     Hiv:type_of_scaleup           remove_IPT
# 7     Hiv:type_of_scaleup           target_IPT
# 8     Hiv:type_of_scaleup          remove_vmmc
# 9     Hiv:type_of_scaleup        increase_6MMD
# 10    Hiv:type_of_scaleup           remove_all


TARGET_PERIOD = (Date(2025, 1, 1), Date(2045, 1, 1))


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
colours = sns.color_palette("mako", n_colors=11)
scenario_colours = dict(zip(param_names, colours))





def plot_central_with_ci(
    df,
    variable: str,
    title: str = None,
    ylabel: str = None,
    figsize=(10, 6),
    colour_map: dict = None
):
    central = df.xs("central", axis=1, level="stat").loc[variable]
    lower = df.xs("lower", axis=1, level="stat").loc[variable]
    upper = df.xs("upper", axis=1, level="stat").loc[variable]
    yerr = [central - lower, upper - central]

    colours = [colour_map.get(s, "grey") for s in central.index] if colour_map else None

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(central.index, central.values, yerr=yerr, capsize=5, color=colours)

    ax.set_ylabel(ylabel if ylabel else variable.replace("_", " ").title())
    ax.set_title(title if title else f"{variable.replace('_', ' ').title()} by Scenario")
    ax.set_xticks(range(len(central.index)))
    ax.set_xticklabels(central.index, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axhline(0, color="grey", linewidth=1)

    plt.tight_layout()
    plt.show()



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


plot_central_with_ci(
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


plot_central_with_ci(
    py_fsw_prep,
    variable="PY_PREP_ORAL_FSW",
    title=f"Number of person-years on PrEP for FSW {target_period()}",
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


plot_central_with_ci(
    py_agyw_prep,
    variable="PY_PREP_ORAL_AGYW",
    title=f"Number of person-years on PrEP for AGYW {target_period()}",
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

plot_central_with_ci(
    num_men_circ,
    variable="N_NewVMMC",
    title=f"Number of VMMC performed {target_period()}",
    ylabel="Number new VMMC",
    colour_map=scenario_colours
)






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

plot_central_with_ci(
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
        custom_generate_series=get_num_inf
    ), central_measure="mean", use_standard_error=True
).pipe(set_param_names_as_column_index_level_0)

plot_central_with_ci(
    num_new_infections,
    variable="n_new_infections_adult_1549",
    title=f"Number new HIV infections {target_period()}",
    ylabel="Number HIV infections",
    colour_map=scenario_colours
)






# 6MMD

mmd_M = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="arv_dispensing_intervals",
        column="group=adult_male|interval=6+",
        index="date",
        do_scaling=False
)

mmd_F = extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="arv_dispensing_intervals",
        column="group=adult_female|interval=6+",
        index="date",
        do_scaling=False
)




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

plot_central_with_ci(
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


def plot_with_ci(
    df,
    title,
    ylabel,
    figsize=(12, 6),
    percent_df=None
):
    """
    Plot a bar chart with central values and confidence intervals using a custom colour scheme.
    Optionally annotate bars with percentage differences, positioned above the upper error bar.
    """
    central = df["central"]
    lower = df["lower"]
    upper = df["upper"]
    yerr = [central - lower, upper - central]

    # Colour mapping
    bar_colours = (
        [scenario_colours[sc] for sc in central.index]
        if scenario_colours else None
    )

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(central.index, central.values, yerr=yerr, capsize=5, color=bar_colours)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(central.index)))
    ax.set_xticklabels(central.index, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axhline(0, color="grey", linewidth=1)

    # Annotate bars with percentage change above the upper CI
    if percent_df is not None:
        for label, bar in zip(central.index, bars):
            pc = percent_df.loc[label, "central"]
            bar_x = bar.get_x() + bar.get_width() / 2
            upper_bound = upper[label]
            offset = 0.05 * (upper.max() - lower.min())  # 5% vertical offset
            y_pos = upper_bound + offset
            ax.text(bar_x, y_pos, f"{pc:.1f}%", ha='center', va='bottom', fontsize=9)

    # Adjust y-limits to avoid clipping
    y_max = (upper + (0.1 * (upper.max() - lower.min()))).max()
    y_min = (lower - (0.1 * (upper.max() - lower.min()))).min()
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()



plot_with_ci(
    df=dalys_diff_from_statusquo,
    title=f"Difference in DALYs from Status Quo {target_period()}",
    ylabel="Difference from Status Quo",
    percent_df=pc_dalys_diff_from_statusquo
)


def num_dalys_by_cause(_df):
    """Return total number of DALYS (Stacked) (total by age-group within the TARGET_PERIOD)"""
    return _df \
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
        .drop(columns=['date', 'sex', 'age_range', 'year']) \
        .sum()

# extract dalys by cause with mean and upper/lower intervals
# With 'collapse_columns', if number of draws is 1, then collapse columns multi-index:

daly_by_cause = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True,
    ),
    only_mean=False,
    collapse_columns=False,
).pipe(set_param_names_as_column_index_level_0)
daly_by_cause.to_excel(results_folder / "daly_by_cause.xlsx")


dalys_labelled_diff_from_statusquo = summarize(
    find_difference_relative_to_comparison_series_dataframe(
        daly_by_cause,
        comparison='Status Quo'
    ),
    only_mean=True
)

dalys_labelled_diff_from_statusquo.to_excel(results_folder / "dalys_labelled_diff_from_statusquo.xlsx")



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
            pc = percent_df.loc[scenario, "central"]
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
    plt.show()



plot_stacked_dalys_by_cause(
    df_by_cause=diff_num_dalys_by_label_vs_statusquo,
    percent_df=pc_dalys_diff_from_statusquo,
    param_names=param_names,
    title="Difference in DALYs Compared to Status Quo (by Cause)",
    ylabel="Difference in DALYs"
)




# HS use

def summarise_appointments(df: pd.DataFrame) -> pd.Series:
    """
    Extract and sum all appointment types during the TARGET_PERIOD from the HSI_Event log.

    Returns a Series indexed by appointment type.
    """
    df["date"] = pd.to_datetime(df["date"])
    mask = df["date"].between(*TARGET_PERIOD)
    filtered = df.loc[mask, "Number_By_Appt_Type_Code"]

    # Expand list of counts per row into a DataFrame
    expanded = pd.DataFrame(filtered.tolist())

    # Sum across all rows (time points)
    summed = expanded.sum()

    # Return as Series with integer index
    summed.index.name = "AppointmentTypeCode"
    return summed

appt_counts = extract_results(
    results_folder=results_folder,
    module="tlo.methods.healthsystem.summary",
    key="HSI_Event",
    custom_generate_series=summarise_appointments,
    do_scaling=True  # to scale to national population
).pipe(set_param_names_as_column_index_level_0)

appt_counts_by_draw = appt_counts.groupby(level="draw", axis=1).mean()


# diff_appt_counts_vs_statusquo = summarize(
#     find_difference_relative_to_comparison_series_dataframe(
#         appt_counts,
#         comparison='Status Quo',
#     ),
#     only_mean=True
# )
appt_diff_from_statusquo = compute_summary_statistics(
    find_difference_relative_to_comparison_series_dataframe(
        appt_counts,
        comparison="Status Quo"
    )
)
appt_diff_from_statusquo.to_excel(results_folder / "appt_diff_from_statusquo.xlsx")


pc_diff_appt_counts_vs_statusquo = 100.0 * compute_summary_statistics(
    pd.DataFrame(
        find_difference_relative_to_comparison_series_dataframe(
            appt_counts,
            comparison='Status Quo',
        scaled=True)
    ), only_central=True
)






def plot_grouped_appt_counts(appt_counts_by_draw, title="Percentage change in appt numbers", figsize=(14, 6)):
    """
    Plot grouped bar chart showing appointment counts per type, grouped by draw.
    Adds vertical separators between appointment type groups.
    """

    index = appt_counts_by_draw.index.astype(str)  # appointment types as str
    columns = appt_counts_by_draw.columns          # draw numbers
    n_types = len(index)
    n_draws = len(columns)

    bar_width = 0.8 / n_draws
    x = np.arange(n_types)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars
    for i, draw in enumerate(columns):
        offset = (i - n_draws / 2) * bar_width + bar_width / 2
        ax.bar(x + offset, appt_counts_by_draw[draw], width=bar_width, label=f"{draw}")

    # Add vertical lines between groups
    for i in range(1, n_types):
        ax.axvline(x=i - 0.5, color="grey", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(index, rotation=45, ha="right")
    ax.set_ylabel("Percentage change")
    ax.set_title(title)
    ax.legend(title="Scenario")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


plot_grouped_appt_counts(pc_diff_appt_counts_vs_statusquo)
