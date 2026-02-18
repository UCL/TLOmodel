"""Produce outputs for Impact of Improved Consumables Availability Paper
"""
import datetime
import os
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from scripts.costing.cost_estimation import (
    clean_consumable_name,
    create_summary_treemap_by_cost_subgroup,
    do_line_plot_of_cost,
    do_stacked_bar_plot_of_cost_by_category,
    estimate_input_cost_of_scenarios,
    summarize_cost_data
)
from tlo import Date
from tlo.analysis.utils import (
    compute_summary_statistics,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    create_pickles_locally,
    summarize,
)

# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# Create folders to store results
resourcefilepath = Path("./resources")
outputfilepath = Path('./outputs/')
figurespath = Path('./outputs/consumables_impact_analysis/manuscript')
if not os.path.exists(figurespath):
    os.makedirs(figurespath)
path_for_consumable_resourcefiles = resourcefilepath / "healthsystem/consumables"

# Load result files
# ------------------------------------------------------------------------------------------------------------------
results_folder = get_scenario_outputs('consumables_impact-2026-02-14T203020Z.py', outputfilepath)[0] # Dec 2025 runs
suspended_results_folder = get_scenario_outputs('consumables_impact-2026-02-13T183325Z.py', outputfilepath)[0]
#create_pickles_locally(scenario_output_dir = "./outputs/consumables_impact-2026-02-14T203020Z") # from .log files
scaling_factor =  load_pickled_dataframes(
            suspended_results_folder, draw = 0, run = 0, name = 'tlo.methods.demography'
            )['tlo.methods.demography']['scaling_factor']['scaling_factor'].values[0]

# Check can read results from draw=0, run=0
log = load_pickled_dataframes(results_folder, 0, 0)  # look at one log (so can decide what to extract)
params = extract_params(results_folder)
info = get_scenario_info(results_folder)

# Declare default parameters for cost analysis
# ------------------------------------------------------------------------------------------------------------------
# Period relevant for costing
TARGET_PERIOD = (Date(2025, 1, 1), Date(2029, 12, 31))  # TODO change to 2040
relevant_period_for_costing = [i.year for i in TARGET_PERIOD]
list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0], relevant_period_for_costing[1] + 1))
list_of_years_for_plot = list(range(2025, 2030))
number_of_years_costed = relevant_period_for_costing[1] - 2025 + 1

discount_rate_health = 0
chosen_metric = 'mean'
chosen_cet = 65

# Scenarios
cons_scenarios = {
    0:  "Baseline availability – Default health system",
    1:  "Baseline availability – Perfect health system",

    2:  "Non-therapeutic consumables (NTC) – Default health system",
    3:  "Non-therapeutic consumables (NTC) – Perfect health system",

    4:  "NTC + Vital medicines (VM) – Default health system",
    5:  "NTC + Vital medicines (VM) – Perfect health system",

    6:  "NTC + VM + Pharmacist-managed stocks – Default health system",
    7:  "NTC + VM + Pharmacist-managed stocks – Perfect health system",

    8:  "75th percentile facility – Default health system",
    9:  "75th percentile facility – Perfect health system",

    10: "90th percentile facility – Default health system",
    11: "90th percentile facility – Perfect health system",

    12: "Best facility – Default health system",
    13: "Best facility – Perfect health system",

    14: "District pooling – Default health system",
    15: "District pooling – Perfect health system",

    16: "Neighbourhood pooling – Default health system",
    17: "Neighbourhood pooling – Perfect health system",

    18: "Pairwise exchange (Large radius) – Default health system",
    19: "Pairwise exchange (Large radius) – Perfect health system",

    20: "Pairwise exchange (Small radius) – Default health system",
    21: "Pairwise exchange (Small radius) – Perfect health system",

    22: "Perfect availability – Default health system",
    23: "Perfect availability – Perfect health system",
}

main_analysis_subset = [
    k for k, v in cons_scenarios.items()
    if "Default health system" in v
]

cons_scenarios_main = {
    k: v.replace(" – Default health system", "")
    for k, v in cons_scenarios.items()
    if k in main_analysis_subset
}

# Dict to assign DALY causes to disease groups
disease_groups = {
    # --- HIV / TB / Malaria ---
    "HIV/AIDS": [
        "AIDS",
    ],
    "Malaria": [
        "Malaria",
    ],

    # --- MNCH ---
    "RMNCH": [
        "Maternal Disorders",
        "Neonatal Disorders",
        "Congenital birth defects",
        "Childhood Diarrhoea",
        "Childhood Undernutrition",
        "Lower respiratory infections",
        "Measles",
    ],

    # --- NCDs ---
    "Cardiometabolic": [
        "Heart Disease",
        "Stroke",
        "Diabetes",
        "Kidney Disease",
    ],
    "Cancer": [
        "Cancer (Bladder)",
        "Cancer (Breast)",
        "Cancer (Cervix)",
        "Cancer (Oesophagus)",
        "Cancer (Prostate)",
        "Cancer (Other)",
    ],
    "Mental & Neurological": [
        "Depression / Self-harm",
        "Epilepsy",
        "Lower Back Pain",
    ],
    "Other": [
        "COPD",
        "Schistosomiasis",
        "Other",
    ],

    # --- Injuries ---
    "Injuries": [
        "Transport Injuries",
    ],
}
# Dict to assign colours to disease groups
disease_colors = {
    "HIV/AIDS": "#e41a1c",
    "Malaria": "#377eb8",
    "RMNCH": "#4daf4a",
    "Cardiometabolic": "#984ea3",
    "Cancer": "#ff7f00",
    "Mental & Neurological": "#a65628",
    "Injuries": "#f781bf",
    "Other": "#999999",
}
# Dict to recategorize above TREATMENT_IDs into disease groups (as used to classify DALYs averted)
service_to_group = {
    # --- HIV / TB / Malaria ---
    "Hiv*": "HIV/AIDS",
    "Tb*": "HIV/AIDS",  # TB grouped with HIV/AIDS in DALYs
    "Malaria*": "Malaria",

    # --- RMNCH ---
    "AntenatalCare*": "RMNCH",
    "DeliveryCare*": "RMNCH",
    "PostnatalCare*": "RMNCH",
    "Contraception*": "RMNCH",
    "Diarrhoea*": "RMNCH",
    "Undernutrition*": "RMNCH",
    "Alri*": "RMNCH",
    "Measles*": "RMNCH",
    "Epi*": "RMNCH",

    # --- Cardiometabolic ---
    "CardioMetabolicDisorders*": "Cardiometabolic",

    # --- Cancer ---
    "BladderCancer*": "Cancer",
    "BreastCancer*": "Cancer",
    "CervicalCancer*": "Cancer",
    "OesophagealCancer*": "Cancer",
    "ProstateCancer*": "Cancer",
    "OtherAdultCancer*": "Cancer",

    # --- Mental & Neurological ---
    "Depression*": "Mental & Neurological",
    "Epilepsy*": "Mental & Neurological",

    # --- Other ---
    "Copd*": "Other",
    "Inpatient*": "Other",
    "Schisto*": "Other",

    # --- Injuries ---
    "Rti*": "Injuries",
}


# Function to get incremental values
def find_difference_relative_to_comparison(
    df: pd.DataFrame,
    comparison: str,
    scaled: bool = False,
    drop_comparison: bool = True,
):
    """
    Compute difference relative to a comparison draw
    for a DataFrame with MultiIndex columns (draw, run).
    """

    # Ensure draw is first level
    if df.columns.names[0] != "draw":
        df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)

    # Extract comparison values
    comp_df = df.xs(comparison, level="draw", axis=1)

    # Broadcast subtraction across draws
    if scaled:
        result = (df - comp_df) / comp_df
    else:
        result = df - comp_df

    if drop_comparison:
        result = result.drop(columns=comparison, level="draw")

    return result

# ----------------------------
# Define utility functions
# ----------------------------

# Define a function to create bar plots
def do_standard_bar_plot_with_ci(_df: pd.DataFrame, set_colors=None, annotations=None,
                                 xticklabels_wrapped=False,
                                 put_labels_in_legend=True, scenarios_dict = None,
                                 offset=1e6):
    """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
     extent of the error bar."""

    substitute_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    yerr = np.array([
        (_df[chosen_metric] - _df['lower']).values,
        (_df['upper'] - _df[chosen_metric]).values,
    ])

    xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}

    if set_colors is not None:
        # dict mapping -> use index keys; list/tuple/Series -> use as-is
        if isinstance(set_colors, dict):
            colors = [set_colors.get(k, 'grey') for k in _df.index]
            # Optional debug:
            # missing = [k for k in _df.index if k not in set_colors]
            # if missing: print("No color for:", missing)
        else:
            colors = list(set_colors)
    else:
        cmap = sns.color_palette('Spectral', as_cmap=True)
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))  # noqa: E731
        colors = list(map(cmap, rescale(np.array(list(xticks.keys()))))) if put_labels_in_legend else None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        xticks.keys(),
        _df[chosen_metric].values,
        yerr=yerr,
        ecolor='black',
        color=colors,
        capsize=10,
        label=xticks.values()
    )

    if annotations:
        for xpos, (ypos, text) in zip(xticks.keys(), zip(_df['upper'].values.flatten(), annotations)):
            annotation_y = ypos + offset

            ax.text(
                xpos,
                annotation_y,
                '\n'.join(text.split(' ', 1)),
                horizontalalignment='center',
                verticalalignment='bottom',  # Aligns text at the bottom of the annotation position
                fontsize='x-small',
                rotation='horizontal'
            )

    ax.set_xticks(list(xticks.keys()))

    if put_labels_in_legend:
        # Update xticks label with substitute labels
        # Insert legend with updated labels that shows correspondence between substitute label and original label
        # Use all_manuscript_scenarios for the legend
        xtick_legend = [f'{letter}: {scenarios_dict.get(label, label)}' for letter, label in
                        zip(substitute_labels, xticks.values())]
        xtick_values = [letter for letter, label in zip(substitute_labels, xticks.values())]

        h, legs = ax.get_legend_handles_labels()
        ax.legend(h, xtick_legend, loc='center left', fontsize='small', bbox_to_anchor=(1, 0.5))
        ax.set_xticklabels(xtick_values)
    else:
        # Use scenarios_dict if provided, otherwise fall back to original labels
        if scenarios_dict is not None:
            labels = [scenarios_dict.get(label, label) for label in xticks.values()]
        else:
            labels = list(xticks.values())

        if not xticklabels_wrapped:
            ax.set_xticklabels(labels, rotation=90)
        else:
            wrapped_labs = ["\n".join(textwrap.wrap(_lab, 20)) for _lab in labels]
            ax.set_xticklabels(wrapped_labs, rotation=90)

    # Extend ylim to accommodate data labels
    ymin, ymax = ax.get_ylim()
    extension = 0.1 * (ymax - ymin)  # 10% of range
    ax.set_ylim(ymin - extension, ymax + extension)  # Set new y-axis limits with the extended range

    ax.grid(axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # fig.tight_layout()
    fig.tight_layout(pad=2.0)

    if put_labels_in_legend:
        # Leave space on right for legend
        plt.subplots_adjust(left=0.15, right=0.5, top=0.88)
    else:
        # Use full width of figure
        plt.subplots_adjust(left=0.15, right=0.95, top=0.88)

    return fig, ax

def plot_stacked_mean_with_total_ci(
    summary_df,
    colors: dict,
    scenario_labels: dict | None = None,
    ylabel: str = "",
    xlabel: str = "Scenario",
    title: str | None = None,
    figsize=(12, 6),
    legend_outside: bool = True,
    xticklabels_wrapped: bool = False,
    wrap_width: int = 20,
):
    """
    Plot stacked bars using mean values by disease group, with a confidence
    interval for the total (lower/upper summed across groups).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Index: disease_group
        Columns: MultiIndex (draw, stat) where stat ∈ {'lower','mean','upper'}

    colors : dict
        Mapping {disease_group: color}

    scenario_labels : dict, optional
        Mapping {draw: label} for x-axis

    ylabel : str
        Y-axis label

    xlabel : str
        X-axis label

    title : str, optional
        Figure title

    figsize : tuple
        Figure size

    legend_outside : bool
        Whether to place legend outside the plot
    """

    # ---- Extract mean values (for stacking) ----
    mean_df = summary_df.xs("mean", level="stat", axis=1)

    # ---- Extract totals for CI ----
    total_mean = mean_df.sum(axis=0)
    total_lower = (
        summary_df.xs("lower", level="stat", axis=1)
        .sum(axis=0)
    )
    total_upper = (
        summary_df.xs("upper", level="stat", axis=1)
        .sum(axis=0)
    )

    # ---- X axis ----
    draws = mean_df.columns
    x = np.arange(len(draws))

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=figsize)

    bottom = np.zeros(len(draws))

    for group in mean_df.index:
        vals = mean_df.loc[group].values
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=colors.get(group, "grey"),
            label=group
        )
        bottom += vals

    # ---- CI for total only ----
    yerr = np.vstack([
        total_mean - total_lower,
        total_upper - total_mean
    ])

    ax.errorbar(
        x,
        total_mean,
        yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=1.5,
        capsize=4,
        zorder=5
    )

    # ---- Formatting ----
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title, pad=10)

    # ---- X tick labels ----
    rotation = 90
    if scenario_labels:
        labels = [scenario_labels.get(d, d) for d in draws]
    else:
        labels = list(draws)

    if xticklabels_wrapped:
        labels = [
            "\n".join(textwrap.wrap(str(l), wrap_width))
            for l in labels
        ]
        ha = "center"
    else:
        ha = "right"

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)

    if legend_outside:
        ax.legend(
            title="Disease group",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False
        )
    else:
        ax.legend(frameon=False)

    fig.tight_layout()
    return fig, ax


def plot_percentage_change_with_ci(
    summary_df,
    colors: dict,
    scenario_labels: dict | None = None,
    ylabel: str = "Percentage change relative to baseline",
    xlabel: str = "Scenario",
    title: str | None = None,
    figsize=(12, 6),
    xticklabels_wrapped: bool = False,
    wrap_width: int = 20,
    markers: list | None = None,
):
    """
    Dot + 95% CI plot for percentage change outcomes (non-additive).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Index: disease_group
        Columns: MultiIndex (draw, stat) where stat ∈ {'mean','lower','upper'}

    colors : dict
        Mapping {disease_group: color}

    scenario_labels : dict, optional
        Mapping {draw: label}

    markers : list, optional
        Custom marker list per disease group
    """

    # ---- Extract statistics ----
    mean_df = summary_df.xs("mean", level="stat", axis=1)
    lower_df = summary_df.xs("lower", level="stat", axis=1)
    upper_df = summary_df.xs("upper", level="stat", axis=1)

    draws = mean_df.columns
    x = np.arange(len(draws))

    disease_groups = mean_df.index.tolist()

    # Default markers if not provided
    if markers is None:
        markers = ["o", "s", "D", "^", "P", "X", "v", "*", "<", ">"]
    marker_map = {
        dg: markers[i % len(markers)]
        for i, dg in enumerate(disease_groups)
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Slight horizontal jitter to prevent overlap
    jitter_strength = 0.15
    offsets = np.linspace(
        -jitter_strength, jitter_strength, len(disease_groups)
    )

    # ---- Plot each disease group ----
    for i, group in enumerate(disease_groups):

        y = mean_df.loc[group].values
        yerr = np.vstack([
            y - lower_df.loc[group].values,
            upper_df.loc[group].values - y
        ])

        ax.errorbar(
            x + offsets[i],
            y,
            yerr=yerr,
            fmt=marker_map[group],
            color=colors.get(group, "black"),
            markersize=6,
            capsize=3,
            linestyle="none",
            alpha=0.85,
            label=group
        )

    # Reference line at zero
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    # ---- X tick labels ----
    if scenario_labels:
        labels = [scenario_labels.get(d, d) for d in draws]
    else:
        labels = list(draws)

    if xticklabels_wrapped:
        labels = [
            "\n".join(textwrap.wrap(str(l), wrap_width))
            for l in labels
        ]
        ha = "center"
    else:
        ha = "right"

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha=ha)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if title:
        ax.set_title(title, pad=12)

    ax.grid(axis="y", alpha=0.3)

    ax.legend(
        title="Disease group",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    fig.tight_layout()
    return fig, ax


def plot_change_in_cons_unavailability_by_scenario(
    nat_mean,
    nat_lower,
    nat_upper,
    scenario_labels =cons_scenarios_main,
    figsize=(14, 6),
    wrap_width=20,
):

    fig, ax = plt.subplots(figsize=figsize)

    draws = nat_mean.index
    x = np.arange(len(draws))

    # Map draw → scenario name
    labels = [
        "\n".join(textwrap.wrap(scenario_labels.get(d, str(d)), wrap_width))
        for d in draws
    ]

    # CI
    yerr = np.vstack([
        nat_mean - nat_lower,
        nat_upper - nat_mean
    ])

    ax.errorbar(
        x,
        nat_mean,
        yerr=yerr,
        fmt="o",
        capsize=4,
        color="black"
    )

    ax.axhline(0, linestyle="--", color="black", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylabel(
        "Change in consumable unavailability \n across consumables (percentage points)"
    )
    ax.set_xlabel("Scenario")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

def plot_change_in_cons_unavailability_by_program(
    delta_mean,
    figsize=(14, 6),
    wrap_width=20,
):
    """
    Plot distribution of programme-specific change in consumable
    unavailability across scenarios.

    Parameters
    ----------
    delta_mean : pd.DataFrame
        Index = disease_group
        Columns = draw (scenarios)
        Values = change in percentage points
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Transpose so each box represents a programme
    data_to_plot = delta_mean.T

    sns.boxplot(
        data=data_to_plot,
        showfliers=False,
        ax=ax
    )

    # Wrap programme names
    wrapped_labels = [
        "\n".join(textwrap.wrap(str(label), wrap_width))
        for label in delta_mean.index
    ]

    ax.set_xticklabels(
        wrapped_labels,
        rotation=45,
        ha="right"
    )

    ax.axhline(0, linestyle="--", color="black", linewidth=1)

    ax.set_ylabel(
        "Change in consumable unavailability \n across scenarios (percentage points)"
    )
    ax.set_xlabel("Disease programme")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()


def plot_heatmap_delta(delta_mean, scenario_labels=None, figsize=(12, 6), baseline_draw = 0):

    df = delta_mean.copy()
    df = df.drop(columns=baseline_draw)

    if scenario_labels:
        df = df.rename(columns=scenario_labels)

    plt.figure(figsize=figsize)

    sns.heatmap(
        df,
        cmap="RdBu_r",          # blue = reduction (good), red = increase (bad)
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Change in % unavailable (vs baseline)"}
    )

    plt.xlabel("Scenario")
    plt.ylabel("Disease group")
    plt.title("Change in consumable unavailability relative to baseline")

    plt.tight_layout()


def aggregate_by_disease_group(df: pd.DataFrame,
                               disease_groups: dict) -> pd.DataFrame:
    """
    Aggregate disease-level rows into disease-group rows.

    Parameters
    ----------
    df : pd.DataFrame
        Index: disease names
        Columns: MultiIndex (draw, run)
        Values: DALYs

    disease_groups : dict
        Mapping {group_name: [list of disease names]}

    Returns
    -------
    pd.DataFrame
        Index: disease groups
        Columns: same as df (draw, run)
        Values: summed DALYs per group
    """
    grouped_rows = {}

    for group_name, diseases in disease_groups.items():
        # Select diseases that actually exist in the index
        present = [d for d in diseases if d in df.index]

        if not present:
            continue

        grouped_rows[group_name] = df.loc[present].sum(axis=0)

    return pd.DataFrame.from_dict(grouped_rows, orient="index")

def summarize_aggregated_results_for_figure(
    df,
    main_analysis_subset,
    chosen_metric="mean"
):
    """
    Prepare a draw/run DataFrame for plotting:
    - drop redundant column level if present
    - restrict to main_analysis_subset
    - summarize across runs
    """

    # If columns have extra level (e.g. ('mean', run)), drop first level
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels > 1:
        df = df.copy()
        df.columns = df.columns.droplevel(0)

    # Restrict to subset of draws
    df = df[df.index.isin(main_analysis_subset)]

    # Summarize across runs
    summarized = summarize_cost_data(df, _metric=chosen_metric)

    return summarized

def summarize_disaggregated_results_for_figure(
    df_grouped,
    main_analysis_subset,
    chosen_metric="mean"
):
    """
    Summarize grouped (e.g., disease_group) results
    from wide draw/run format into MultiIndex columns (draw, stat).
    """

    # Ensure long format: index = (group, draw, run)
    df_long = df_grouped.stack(level=["draw", "run"])
    df_long.index.names = ["group", "draw", "run"]

    summaries = {}

    for group in df_long.index.get_level_values("group").unique():
        ser = df_long.xs(group, level="group")
        df_wide = ser.unstack("run")
        summaries[group] = summarize_cost_data(df_wide, _metric=chosen_metric)

    result = pd.concat(summaries, names=["group"])

    # Reformat columns to (draw, stat)
    result = result.unstack()
    result.columns = result.columns.swaplevel("stat", "draw")

    # Restrict draws
    result = result.loc[
        :,
        result.columns.get_level_values("draw").isin(main_analysis_subset)
    ]

    return result

def set_param_names_as_column_index_level_0(_df):
    """Set the columns index (level 0) as the param_names."""
    ordered_param_names_no_prefix = {i: x for i, x in enumerate(cons_scenarios)}
    names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
    assert len(names_of_cols_level0) == len(_df.columns.levels[0])
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df

# Functions to extract results
def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
    Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
    results from runs that crashed mid-way through the simulation.
    """
    years_needed = relevant_period_for_costing
    assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
    _df = _df.loc[_df.year.between(*years_needed)].drop(columns=['date', 'sex', 'age_range']).groupby(
        'year').sum().sum(axis=1)

    # Initial year and discount rate
    initial_year = min(_df.index.unique())

    # Calculate the discounted values
    discounted_values = _df / (1 + discount_rate_health) ** (_df.index - initial_year)

    return pd.Series(discounted_values.sum())

def get_num_dalys_by_disease(_df):
    """
    Return discounted total DALYs by disease over the TARGET_PERIOD.
    Output: Series indexed by disease name.
    """
    years_needed = relevant_period_for_costing
    assert set(_df.year.unique()).issuperset(years_needed), \
        "Some years are not recorded."

    # Keep only years of interest
    _df = _df.loc[_df.year.between(*years_needed)]

    # Drop non-disease columns
    disease_cols = _df.columns.difference(
        ['date', 'sex', 'age_range', 'year']
    )

    # Sum by year × disease
    by_year_disease = (
        _df[['year'] + list(disease_cols)]
        .groupby('year')
        .sum()
    )

    # Discounting
    initial_year = by_year_disease.index.min()
    discount_factors = (1 + discount_rate_health) ** (
        by_year_disease.index - initial_year
    )

    discounted = by_year_disease.div(discount_factors, axis=0)

    # Sum over time → total DALYs by disease
    return discounted.sum()

def get_num_treatments_total(_df):
    """Return the number of treatments in total of all treatments (total within the TARGET_PERIOD)"""
    _df = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD), 'TREATMENT_ID'].apply(pd.Series).sum()
    _df.index = _df.index.map(lambda x: x.split('_')[0] + "*")
    _df = _df.groupby(level=0).sum().sum()
    return pd.Series(_df)

def get_num_treatments_by_disease_group(_df):
    """Return the number of treatments by short treatment id (total within the TARGET_PERIOD)"""
    _df = _df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD), 'TREATMENT_ID'].apply(pd.Series).sum()
    _df.index = _df.index.map(lambda x: x.split('_')[0] + "*")
    _df = _df.rename(index=disease_groups)
    _df = _df.groupby(level=0).sum()
    return _df

def get_monetary_value_of_incremental_health(_num_dalys_averted, _chosen_value_of_life_year):
    monetary_value_of_incremental_health = (_num_dalys_averted * _chosen_value_of_life_year).clip(lower=0.0)
    return monetary_value_of_incremental_health

def get_percentage_unavailable_by_program(_df):
    """
    Compute percentage of times consumables were unavailable
    by disease program within TARGET_PERIOD.
    """

    # Restrict to target period
    _df = _df.loc[
        pd.to_datetime(_df.date).between(*TARGET_PERIOD),
        ['Item_Available', 'Item_NotAvailable']
    ]

    # ---- Sum dictionaries across rows ----
    available = (
        _df['Item_Available']
        .apply(pd.Series)
        .sum()
    )

    not_available = (
        _df['Item_NotAvailable']
        .apply(pd.Series)
        .sum()
    )

    # Align indices
    total = available.add(not_available, fill_value=0)

    # % unavailable per item
    pct_unavailable = (
        not_available / total.replace(0, np.nan)
    )

    # Map items to program
    pct_unavailable.index = pct_unavailable.index.astype(str)
    pct_unavailable = pct_unavailable.rename(index=item_to_program_map)

    # Aggregate to program level
    pct_unavailable = (
        pct_unavailable
        .groupby(level=0)
        .mean()
    )

    return pct_unavailable

def compute_delta_unavailability_from_baseline(summary_df, baseline_draw=0):
    """
    Convert absolute % unavailable to change relative to baseline.
    Negative = improvement.
    """
    mean_df = summary_df.xs("mean", level="stat", axis=1)
    baseline = mean_df[baseline_draw]
    delta_mean = mean_df.subtract(baseline, axis=0)

    return delta_mean

def compute_national_unavailability_summary(summary_df, baseline_draw=0):

    mean_df = summary_df.xs("mean", level="stat", axis=1)
    lower_df = summary_df.xs("lower", level="stat", axis=1)
    upper_df = summary_df.xs("upper", level="stat", axis=1)

    baseline = mean_df[baseline_draw]

    delta_mean = mean_df.subtract(baseline, axis=0)
    delta_lower = lower_df.subtract(baseline, axis=0)
    delta_upper = upper_df.subtract(baseline, axis=0)

    national_mean = delta_mean.mean(axis=0)
    national_lower = delta_lower.mean(axis=0)
    national_upper = delta_upper.mean(axis=0)

    return delta_mean, national_mean, national_lower, national_upper

# ----------------------------
# 1) DALYs averted - Total
# ----------------------------
num_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys,
        do_scaling=True,
        suspended_results_folder = suspended_results_folder,
    )

num_dalys_averted = ((-1.0 *
                     pd.DataFrame(
                         find_difference_relative_to_comparison(
                             num_dalys,
                             comparison=0)  # sets the comparator to 0 which is the Actual scenario
                     ).T.unstack(level='run')))

num_dalys_averted_summarized = summarize_aggregated_results_for_figure(
    num_dalys_averted,
    main_analysis_subset,
    chosen_metric
)

num_dalys_averted_percent = ((-1.0 *
                     pd.DataFrame(
                         find_difference_relative_to_comparison(
                             num_dalys,
                             comparison=0, scaled = True)  # sets the comparator to 0 which is the Actual scenario
                     ).T.unstack(level='run')))

num_dalys_averted_percent_summarized = summarize_aggregated_results_for_figure(
    num_dalys_averted_percent,
    main_analysis_subset,
    chosen_metric
)

# Plot DALYs
name_of_plot = f'Incremental DALYs averted compared to baseline {relevant_period_for_costing[0]}-{relevant_period_for_costing[1]}'
fig, ax = do_standard_bar_plot_with_ci(
    (num_dalys_averted_summarized / 1e6).clip(0.0),
    annotations=[
        f"{row[chosen_metric]*100:.1f}% "
        f"({row['lower']*100:.1f}–{row['upper']*100:.1f}%)"
        for _, row in num_dalys_averted_percent_summarized.iterrows()
    ],
    xticklabels_wrapped=True,
    put_labels_in_legend=False,
    offset=0.05,
    scenarios_dict=cons_scenarios_main
)
#ax.set_title(name_of_plot)
ax.set_ylabel('DALYs \n(Millions)')
ax.set_ylim(bottom=0)
fig.savefig(figurespath / 'dalys_averted_total.png', dpi=600)
plt.close(fig)

# --------------------------------------------
# 2) DALYs averted by disease group
# --------------------------------------------
num_dalys_by_disease = extract_results(
    results_folder,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=get_num_dalys_by_disease,
    do_scaling=True,
    suspended_results_folder = suspended_results_folder,
)

num_dalys_by_group = aggregate_by_disease_group(
    num_dalys_by_disease,
    disease_groups
)

# Get absolute DALYs averted
num_dalys_by_group.columns.names = ["draw", "run"]

num_dalys_averted_by_group =  (-1.0 *
                     pd.DataFrame(
                         find_difference_relative_to_comparison(
                             num_dalys_by_group,
                             comparison=0,
                             scaled=True,
                             drop_comparison=True)  # sets the comparator to 0 which is the Actual scenario
                     ))

num_dalys_averted_by_group_summarized = summarize_disaggregated_results_for_figure(
    num_dalys_averted_by_group,
    main_analysis_subset,
    chosen_metric
)

# Plot DALYs averted by disease group
fig, ax = plot_percentage_change_with_ci(
    summary_df=num_dalys_averted_by_group_summarized,
    colors=disease_colors,
    scenario_labels=cons_scenarios_main,
    ylabel="DALYs averted \n (% relative to Baseline)",
    title="",
    xticklabels_wrapped=True,
)
fig.savefig(figurespath / "dalys_averted_by_disease_group.png",
            dpi=300, bbox_inches="tight")

# --------------------------------------------
# 3) Total number of services delivered
# --------------------------------------------
num_treatments_total = extract_results(
    results_folder,
    module='tlo.methods.healthsystem.summary',
    key='HSI_Event_non_blank_appt_footprint',
    custom_generate_series=get_num_treatments_total,
    do_scaling=True,
    suspended_results_folder = suspended_results_folder,
).pipe(set_param_names_as_column_index_level_0)

num_incremental_treatments = (
                     pd.DataFrame(
                         find_difference_relative_to_comparison(
                             num_treatments_total,
                             comparison=0,
                             scaled = True)  # sets the comparator to 0 which is the Actual scenario
                     ).T.unstack(level='run'))

num_incremental_treatments_summarized = summarize_aggregated_results_for_figure(
    num_incremental_treatments,
    main_analysis_subset,
    chosen_metric
)

# Plot DALYs
name_of_plot = f'Incremental services delivered compared to baseline {relevant_period_for_costing[0]}-{relevant_period_for_costing[1]}'
yaxis_scaling_factor = 1/100
fig, ax = do_standard_bar_plot_with_ci(
    (num_incremental_treatments_summarized / yaxis_scaling_factor).clip(0.0),
    annotations=[
        f"{row[chosen_metric] / yaxis_scaling_factor:.2f} ({row['lower'] / yaxis_scaling_factor :.2f}- {row['upper'] / yaxis_scaling_factor:.2f})"
        for _, row in num_incremental_treatments_summarized.iterrows()
    ],
    xticklabels_wrapped=True,
    put_labels_in_legend=False,
    offset=0.05,
    scenarios_dict=cons_scenarios_main
)
#ax.set_title(name_of_plot)
ax.set_ylabel('Additional services delivered \n(% relative to Baseline)')
ax.set_ylim(bottom=0)
fig.savefig(figurespath / 'incremental_services_delivered_total.png', dpi=600)
plt.close(fig)

# ----------------------------------------------------------
# 4) Total number of services delivered by disease group
# ----------------------------------------------------------
num_treatments_by_disease_group = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event_non_blank_appt_footprint',
        custom_generate_series=get_num_treatments_by_disease_group,
        do_scaling=True,
    suspended_results_folder = suspended_results_folder,
    ).pipe(set_param_names_as_column_index_level_0)

# Map services to DALY groups
num_treatments_by_disease_group['disease_group'] = (
    num_treatments_by_disease_group.index.map(service_to_group))

# Check if any services are unmapped
assert (num_treatments_by_disease_group['disease_group'].isna().sum() == 0)

# Aggregate
num_treatments_by_disease_group = (num_treatments_by_disease_group
                           .set_index("disease_group", append=True)
                           .groupby(level="disease_group")
                           .sum())


num_incremental_treatments_by_disease_group  = pd.DataFrame(
                         find_difference_relative_to_comparison(
                             num_treatments_by_disease_group,
                             comparison=0,
                             scaled = True)  # sets the comparator to 0 which is the Actual scenario
                     )

num_incremental_treatments_by_disease_group_summarized = summarize_disaggregated_results_for_figure(
    num_incremental_treatments_by_disease_group,
    main_analysis_subset,
    chosen_metric
)

fig, ax = plot_percentage_change_with_ci(
    summary_df=num_incremental_treatments_by_disease_group_summarized,
    colors=disease_colors,
    scenario_labels=cons_scenarios_main,
    ylabel="Additional services delivered \n (% relative to Baseline)",
    title="",
    xticklabels_wrapped=True,
)
fig.savefig(figurespath / "incremental_services_delivered_by_disease_group.png",
            dpi=300, bbox_inches="tight")

# ----------------------------------------------------------
# 5) Maximum ability to pay
# ----------------------------------------------------------
max_ability_to_pay = (get_monetary_value_of_incremental_health(num_dalys_averted,
                                                               _chosen_value_of_life_year=chosen_cet)).clip(lower=0.0)
max_ability_to_pay = max_ability_to_pay[
    max_ability_to_pay.index.get_level_values('draw').isin(list(cons_scenarios_main.keys()))]
max_ability_to_pay_summarized = summarize_cost_data(
    max_ability_to_pay, _metric=chosen_metric)

def reformat_with_draw_as_index_and_stat_as_column(_df):
    df = _df.copy()
    df.index = df.index.set_names(["stat", "draw"])
    formatted = df.unstack("stat")
    formatted.columns = formatted.columns.droplevel(0)
    return formatted

max_ability_to_pay_summarized = reformat_with_draw_as_index_and_stat_as_column(
    max_ability_to_pay_summarized)

# Plot Maximum ability to pay
name_of_plot = f'Maximum ability to pay at CET, {relevant_period_for_costing[0]}-{relevant_period_for_costing[1]}'
fig, ax = do_standard_bar_plot_with_ci(
    (max_ability_to_pay_summarized / 1e6),
    annotations=[
        f"{row[chosen_metric] / 1e6 :.2f} ({row['lower'] / 1e6 :.2f}- \n {row['upper']/ 1e6:.2f})"
        for _, row in max_ability_to_pay_summarized.iterrows()
    ],
    xticklabels_wrapped=True,
    put_labels_in_legend=False,
    offset=3, scenarios_dict = cons_scenarios_main
)
ax.set_title(name_of_plot)
ax.set_ylabel('Maximum ability to pay \n(USD, millions)')
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(figurespath / 'max_ability_to_pay.png', dpi = 300, bbox_inches = "tight")
plt.close(fig)

# ----------------------------------------------------------
# 6) Consumable unavailability
# ----------------------------------------------------------
item_to_program_df = pd.read_csv(
    resourcefilepath / 'healthsystem' / 'consumables' / 'ResourceFile_Consumables_Item_Designations.csv'
)[['Item_Code', 'item_category']]

item_to_program_map = dict(
    zip(
        item_to_program_df['Item_Code'].astype(str),
        item_to_program_df['item_category']
    )
)

# Plot the proportion of instances that a consumable was not available when requested
pct_unavailable_by_program = extract_results(
    results_folder,
    module='tlo.methods.healthsystem.summary',
    key='Consumables',
    custom_generate_series=get_percentage_unavailable_by_program,
    do_scaling=False,
    suspended_results_folder = suspended_results_folder,
)

pct_unavailable_by_program_summarized = summarize_disaggregated_results_for_figure(
    pct_unavailable_by_program,
    main_analysis_subset,
    chosen_metric
)

fig, ax = plot_percentage_change_with_ci(
    summary_df=pct_unavailable_by_program_summarized,
    colors=disease_colors,
    scenario_labels=cons_scenarios_main,
    ylabel="% instances of consumables being unavailable",
    title="",
    xticklabels_wrapped=True,
)
fig.savefig(figurespath / "pct_unavailable_by_program.png",
            dpi=300, bbox_inches="tight")

delta_mean = compute_delta_unavailability_from_baseline(pct_unavailable_by_program_summarized)

plot_heatmap_delta(delta_mean, scenario_labels=cons_scenarios_main, baseline_draw = 0)
plt.savefig(figurespath / "pct_change_in_availability_by_scenario_and_program_heatmap.png",
            dpi=300, bbox_inches="tight")

delta_mean, nat_mean, nat_lower, nat_upper = compute_national_unavailability_summary(
    pct_unavailable_by_program_summarized
)

plot_change_in_cons_unavailability_by_program(
    delta_mean,
)
plt.savefig(figurespath / "change_in_cons_unavailability_by_program.png",
            dpi=300, bbox_inches="tight")

plot_change_in_cons_unavailability_by_scenario(
    nat_mean,
    nat_lower,
    nat_upper,
    scenario_labels=cons_scenarios_main
)
plt.savefig(figurespath / "change_in_cons_unavailability_by_scenario.png",
            dpi=300, bbox_inches="tight")
