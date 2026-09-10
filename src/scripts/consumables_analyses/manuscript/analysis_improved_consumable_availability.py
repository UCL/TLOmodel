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
    summarize_cost_data,
    estimate_projected_health_spending,
    load_unit_cost_assumptions
)
from tlo import Date
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    create_pickles_locally,
)

# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# Create folders to store results
resourcefilepath = Path("./resources")
outputfilepath = Path('./outputs/')
scenariooutputs_filepath =  Path('./outputs/sakshi.mohan@york.ac.uk/')
figurespath = Path('./outputs/consumables_impact_analysis/manuscript')
if not os.path.exists(figurespath):
    os.makedirs(figurespath)
path_for_consumable_resourcefiles = resourcefilepath / "healthsystem/consumables"

# Load result files
# ------------------------------------------------------------------------------------------------------------------
results_folder = get_scenario_outputs('consumables_impact-2026-08-29T161006Z.py', scenariooutputs_filepath)[0] # Dec 2025 runs
suspended_results_folder = get_scenario_outputs('consumables_impact-2026-08-28T153202Z.py', scenariooutputs_filepath)[0]
#create_pickles_locally(scenario_output_dir = "./outputs/consumables_impact-2026-08-21T204253Z") # from .log files
scaling_factor =  load_pickled_dataframes(
            suspended_results_folder, draw = 0, run = 0, name = 'tlo.methods.demography'
            )['tlo.methods.demography']['scaling_factor']['scaling_factor'].values[0]
#scaling_factor = 145.39609000000002

# Check can read results from draw=0, run=0
log = load_pickled_dataframes(results_folder, 0, 0)  # look at one log (so can decide what to extract)
params = extract_params(results_folder)
info = get_scenario_info(results_folder)

# Declare default parameters for cost analysis
# ------------------------------------------------------------------------------------------------------------------
# Period relevant for costing
TARGET_PERIOD = (Date(2026, 1, 1), Date(2040, 12, 31))  # TODO change to 2025 to 2040
relevant_period_for_costing = [i.year for i in TARGET_PERIOD]
list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0], relevant_period_for_costing[1] + 1))
list_of_years_for_plot = list(range(2026, 2041))  # TODO change to 2025 onwards
number_of_years_costed = relevant_period_for_costing[1] - 2026 + 1  # TODO change to 2025 onwards

discount_rate_health = 0
chosen_metric = 'mean'
chosen_cet = 65

# Scenarios
cons_scenarios = {
    0:  "Baseline availability – Default health system",
    13:  "Baseline availability – Perfect health system",

    1:  "Non-therapeutic consumables (NTC) – Default health system",
    14:  "Non-therapeutic consumables (NTC) – Perfect health system",

    2:  "NTC + Vital medicines (VM) – Default health system",
    15:  "NTC + Vital medicines (VM) – Perfect health system",

    3:  "NTC + VM + Pharmacist-managed stocks – Default health system",
    16:  "NTC + VM + Pharmacist-managed stocks – Perfect health system",

    4:  "75th percentile facility – Default health system",
    17:  "75th percentile facility – Perfect health system",

    5: "90th percentile facility – Default health system",
    18: "90th percentile facility – Perfect health system",

    6: "Best facility – Default health system",
    19: "Best facility – Perfect health system",

    7: "District pooling – Default health system",
    20: "District pooling – Perfect health system",

    8: "Neighbourhood pooling – Default health system",
    21: "Neighbourhood pooling – Perfect health system",

    9: "Pairwise exchange (Large radius) – Default health system",
    22: "Pairwise exchange (Large radius) – Perfect health system",

    10: "Pairwise exchange (Small radius) – Default health system",
    23: "Pairwise exchange (Small radius) – Perfect health system",

    11: "National pooling – Default health system",
    24: "National pooling – Perfect health system",

    12: "Perfect availability – Default health system",
    25: "Perfect availability – Perfect health system",
}

# Order in which scenarios should appear in every figure (applies uniformly across the manuscript).
# Pairwise exchange kept Small-then-Large, matching the convention used throughout the redistribution
# analysis in generating_consumable_scenarios/create_consumable_redistribution_scenarios.py.
SCENARIO_ORDER = [
    "Baseline availability",
    "Non-therapeutic consumables (NTC)",
    "NTC + Vital medicines (VM)",
    "NTC + VM + Pharmacist-managed stocks",
    "75th percentile facility",
    "90th percentile facility",
    "Best facility",
    "Neighbourhood pooling",
    "District pooling",
    "National pooling",
    "Pairwise exchange (Small radius)",
    "Pairwise exchange (Large radius)",
    "Perfect availability",
]

def _ordered_subset(suffix: str) -> list:
    """Draw numbers of scenarios ending in `suffix`, ordered to match SCENARIO_ORDER."""
    label_to_draw = {v.replace(suffix, ""): k for k, v in cons_scenarios.items() if suffix in v}
    return [label_to_draw[name] for name in SCENARIO_ORDER]

main_analysis_subset = _ordered_subset(" – Default health system")
perfect_analysis_subset = _ordered_subset(" – Perfect health system")

cons_scenarios_main = {
    k: cons_scenarios[k].replace(" – Default health system", "") for k in main_analysis_subset
}

cons_scenarios_perfect = {
    k: cons_scenarios[k].replace(" – Perfect health system", "") for k in perfect_analysis_subset
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
    "TB (non-AIDS)": [
        "TB (non-AIDS)",
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


def plot_change_in_cons_availability_by_scenario(
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
    ax.set_xticklabels(labels, rotation=90, ha="right")

    ax.set_ylabel(
        "Change in consumable availability \n across consumables (percentage points)"
    )
    ax.set_xlabel("Scenario")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

def plot_change_in_cons_availability_by_program(
    delta_mean,
    figsize=(14, 6),
    wrap_width=20,
):
    """
    Plot distribution of programme-specific change in consumable
    availability across scenarios.

    Parameters
    ----------
    delta_mean : pd.DataFrame
        Index = disease_group
        Columns = draw (scenarios)
        Values = change in percentage points (positive = improvement)
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Transpose so each box represents a programme
    data_to_plot = delta_mean.T

    # Clean item category names
    clean_category_names = {
        'cancer': 'Cancer',
        'cardiometabolicdisorders': 'Cardiometabolic Disorders',
        'contraception': 'Contraception',
        'general': 'General',
        'hiv': 'HIV',
        'malaria': 'Malaria',
        'ncds': 'Non-communicable Diseases',
        'neonatal_health': 'Neonatal Health',
        'other_childhood_illnesses': 'Other Childhood Illnesses',
        'reproductive_health': 'Reproductive Health',
        'road_traffic_injuries': 'Road Traffic Injuries',
        'tb': 'Tuberculosis',
        'undernutrition': 'Undernutrition',
        'epi': 'Expanded programme on immunization'
    }
    delta_mean.index = delta_mean.index.map(clean_category_names)

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
        rotation=90,
        ha="right"
    )

    ax.axhline(0, linestyle="--", color="black", linewidth=1)

    ax.set_ylabel(
        "Change in consumable availability \n across scenarios (percentage points)"
    )
    ax.set_xlabel("Disease programme")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

def plot_dalys_averted_vs_change_in_availability(
    change_in_availability: pd.Series,
    dalys_averted: pd.Series,
    scenario_labels: dict,
    title: str | None = None,
    figsize=(9, 7),
    label_fontsize=6,
):
    """
    Scatter plot with one dot per scenario: average change in consumable availability across
    consumables (x-axis) vs. DALYs averted (y-axis, %), with small-font scenario name labels.

    Parameters
    ----------
    change_in_availability : pd.Series
        Index = draw, values = average change in consumable availability vs. baseline
        (percentage points; positive = improvement).
    dalys_averted : pd.Series
        Index = draw, values = DALYs averted vs. baseline (%).
    scenario_labels : dict
        Mapping {draw: scenario name}.
    title : str, optional
        Figure title (e.g. the consumables programme, when plotting one scatter per programme).
    """
    fig, ax = plt.subplots(figsize=figsize)

    draws = change_in_availability.index

    for d in draws:
        x = change_in_availability.loc[d]
        y = dalys_averted.loc[d]
        ax.scatter(x, y, s=45, color="#377eb8", zorder=3)
        ax.annotate(
            scenario_labels.get(d, str(d)),
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=label_fontsize,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xlabel("Average change in consumable availability\nacross facilities and consumables (percentage points)")
    ax.set_ylabel("DALYs averted (%, relative to baseline)")

    if title:
        ax.set_title(title, fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig, ax


def plot_heatmap_delta(delta_mean,
                       scenario_labels=None,
                       figsize=(12, 6),
                       baseline_draw=0,
                       wrap_xticks=True,
                       wrap_width=20,
                       legend_label = None):

    df = delta_mean.copy()

    # Clean item category names
    clean_category_names = {
        'cancer': 'Cancer',
        'cardiometabolicdisorders': 'Cardiometabolic Disorders',
        'contraception': 'Contraception',
        'general': 'General',
        'hiv': 'HIV',
        'malaria': 'Malaria',
        'ncds': 'Non-communicable Diseases',
        'neonatal_health': 'Neonatal Health',
        'other_childhood_illnesses': 'Other Childhood Illnesses',
        'reproductive_health': 'Reproductive Health',
        'road_traffic_injuries': 'Road Traffic Injuries',
        'tb': 'Tuberculosis',
        'undernutrition': 'Undernutrition',
        'epi': 'Expanded programme on immunization'
    }
    df.index = df.index.map(clean_category_names)

    # Drop baseline
    df = df.drop(columns=baseline_draw)

    # Rename scenarios if provided
    if scenario_labels:
        df = df.rename(columns=scenario_labels)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df,
        cmap="RdBu",          # blue = reduction (good), red = increase (bad)
        center=0,
        linewidths=0.5,
        cbar_kws={"label": legend_label},
        ax=ax
    )

    # Wrap x tick labels
    if wrap_xticks:
        wrapped_labels = [
            "\n".join(textwrap.wrap(label.get_text(), wrap_width))
            for label in ax.get_xticklabels()
        ]
        ax.set_xticklabels(wrapped_labels, rotation=90, ha="center")
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Disease group")
    ax.set_title("Change in consumable availability relative to baseline")

    plt.tight_layout()
    return fig, ax


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

    # Restrict to subset of draws, preserving the given order (not the original draw-number order).
    # Some inputs (e.g. num_dalys_averted) have already had the comparator draw dropped by
    # find_difference_relative_to_comparison -- only keep requested draws that are actually present.
    present = [d for d in main_analysis_subset if d in df.index]
    df = df.loc[present]

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

    # Restrict draws, preserving the given order (not the original draw-number order).
    # `list(...)` normalizes both a plain list and a dict (whose keys are used, in insertion order).
    # Only keep requested draws that are actually present (e.g. the comparator draw may have
    # already been dropped upstream by find_difference_relative_to_comparison).
    existing_draws = set(result.columns.get_level_values("draw"))
    present = [d for d in list(main_analysis_subset) if d in existing_draws]
    result = result.reindex(columns=present, level="draw")

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
    monetary_value_of_incremental_health = (_num_dalys_averted * _chosen_value_of_life_year)
    return monetary_value_of_incremental_health

def make_pct_available_by_group_fn(item_to_group_map: dict):
    """
    Build a `custom_generate_series` function (for `extract_results`) that computes the
    percentage of times consumables WERE available within TARGET_PERIOD, aggregated per
    `item_to_group_map` (item_code -> group name). Items not present in `item_to_group_map`
    are excluded, so this can be used both for the full item-category "programs" and for a
    coarser regrouping (e.g. collapsed to DALY disease groups) by passing a different map.

    Availability (not unavailability) is computed directly here -- since available_prop is
    already a positive figure throughout the rest of the pipeline, tracking "% available"
    end-to-end avoids the confusion of mixing it with a mirrored "% unavailable" framing.

    Note: takes the mapping as a parameter (rather than closing over a module-level global)
    deliberately -- item_to_program_map used to only ever be assigned *inside*
    generate_all_consumable_figures, which is a different scope to this module-level
    function, so referencing it as a free variable raised NameError the first time this
    was actually called.
    """
    def _pct_available_by_group(_df):
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

        # % available per item (positive framing: higher = better)
        pct_available = (
            available / total.replace(0, np.nan)
        )

        # Map items to group (dropping items with no mapping) and aggregate
        pct_available.index = pct_available.index.astype(str)
        pct_available = pct_available[pct_available.index.isin(item_to_group_map)]
        pct_available = pct_available.rename(index=item_to_group_map)
        pct_available = (
            pct_available
            .groupby(level=0)
            .mean()
        )

        return pct_available

    return _pct_available_by_group

def compute_delta_availability_from_baseline(summary_df, comparator_draw=0):
    """
    Convert absolute % available to change relative to baseline.
    Positive = improvement.
    """
    mean_df = summary_df.xs("mean", level="stat", axis=1)
    baseline = mean_df[comparator_draw]
    delta_mean = mean_df.subtract(baseline, axis=0)

    return delta_mean

def compute_national_availability_summary(summary_df, comparator_draw=0):
    """
    National (mean-across-programmes/groups) change in % available relative to baseline.
    Positive = improvement.
    """

    mean_df = summary_df.xs("mean", level="stat", axis=1)
    lower_df = summary_df.xs("lower", level="stat", axis=1)
    upper_df = summary_df.xs("upper", level="stat", axis=1)

    baseline = mean_df[comparator_draw]

    delta_mean = mean_df.subtract(baseline, axis=0)
    delta_lower = lower_df.subtract(baseline, axis=0)
    delta_upper = upper_df.subtract(baseline, axis=0)

    national_mean = delta_mean.mean(axis=0)
    national_lower = delta_lower.mean(axis=0)
    national_upper = delta_upper.mean(axis=0)

    return delta_mean, national_mean, national_lower, national_upper

def reformat_with_draw_as_index_and_stat_as_column(_df):
    df = _df.copy()
    df.index = df.index.set_names(["stat", "draw"])
    formatted = df.unstack("stat")
    formatted.columns = formatted.columns.droplevel(0)
    return formatted

def get_manuscript_ready_table_of_projected_health_spending(_relevant_period_for_costing):
    def get_total_population_by_year(_df):
        years_needed = _relevant_period_for_costing  # Malaria scale-up period years
        _df['year'] = pd.to_datetime(_df['date']).dt.year

        # Validate that all necessary years are in the DataFrame
        if not set(years_needed).issubset(_df['year'].unique()):
            raise ValueError("Some years are not recorded in the dataset.")

        # Filter for relevant years and return the total population as a Series
        return \
        _df.loc[_df['year'].between(min(years_needed), max(years_needed)), ['year', 'total']].set_index('year')[
            'total']

    # Get total population by year
    total_population_by_year = extract_results(
        results_folder,
        module='tlo.methods.demography',
        key='population',
        custom_generate_series=get_total_population_by_year,
        do_scaling=True,
        suspended_results_folder=suspended_results_folder,
    ).unstack().reset_index().rename(columns={0: 'population'})
    total_population_summary = total_population_by_year[total_population_by_year.draw == 0].groupby("year")[
        "population"].agg(
        population="median"
    ).reset_index()
    unit_costs = load_unit_cost_assumptions(resourcefilepath)
    health_spending_per_capita = unit_costs["health_spending_projections"]
    health_spending_per_capita = health_spending_per_capita[health_spending_per_capita.year.isin(
        list(range(_relevant_period_for_costing[0], _relevant_period_for_costing[1] + 1)))]
    health_spending_per_capita = health_spending_per_capita[['year', 'total_mean']].apply(
        pd.to_numeric, errors='coerce')
    health_spending_per_capita_table = health_spending_per_capita.merge(total_population_summary, on="year",
                                                                        how="left", validate="1:1")
    health_spending_per_capita_table["total_health_spending"] = health_spending_per_capita_table['total_mean'] * \
                                                                health_spending_per_capita_table['population']
    return health_spending_per_capita_table

def generate_all_consumable_figures(
    scenario_dict: dict,
    results_folder: Path,
    suspended_results_folder: Path,
    comparator_draw: int,
    figurespath: Path,
):
    """
    Generate all consumables impact figures for a given scenario set.

    Parameters
    ----------
    scenario_dict : dict
        Mapping {draw: scenario_name}

    results_folder : Path
        Folder containing model results

    suspended_results_folder : Path
        Folder containing suspended results (if applicable)

    figurespath : Path
        Output folder for figures
    """
    figurespath.mkdir(parents=True, exist_ok=True)
    scenario_subset = list(scenario_dict.keys())

    # --------------------------------------------------
    # 1) TOTAL DALYs AVERTED
    # --------------------------------------------------
    num_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys,
        do_scaling=True,
        suspended_results_folder=suspended_results_folder,
    )

    # Absolute
    num_dalys_averted = (
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_dalys,
                comparison=comparator_draw,
                scaled=False
            )
        ).T.unstack(level='run')
    )

    num_dalys_averted_summarized = summarize_aggregated_results_for_figure(
        num_dalys_averted,
        scenario_subset,
        chosen_metric
    )

    # Percentage
    num_dalys_averted_percent = (
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_dalys,
                comparison=comparator_draw,
                scaled=True
            )
        ).T.unstack(level='run')
    )

    num_dalys_averted_percent_summarized = summarize_aggregated_results_for_figure(
        num_dalys_averted_percent,
        scenario_subset,
        chosen_metric
    )

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
        scenarios_dict=scenario_dict
    )
    ax.set_ylabel('DALYs (Millions)')
    ax.set_ylim(bottom=0)
    fig.savefig(figurespath / 'dalys_averted_total.png', dpi=600)
    plt.close(fig)


    # --------------------------------------------------
    # 2) DALYs BY DISEASE GROUP
    # --------------------------------------------------
    num_dalys_by_disease = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys_by_disease,
        do_scaling=True,
        suspended_results_folder=suspended_results_folder,
    )

    num_dalys_by_group = aggregate_by_disease_group(
        num_dalys_by_disease,
        disease_groups
    )

    num_dalys_by_group.columns.names = ["draw", "run"]

    num_dalys_averted_by_group = (
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_dalys_by_group,
                comparison=comparator_draw,
                scaled=True,
                drop_comparison=True
            )
        )
    )

    num_dalys_averted_by_group_summarized = summarize_disaggregated_results_for_figure(
        num_dalys_averted_by_group,
        scenario_subset,
        chosen_metric
    )

    fig, ax = plot_percentage_change_with_ci(
        summary_df=num_dalys_averted_by_group_summarized,
        colors=disease_colors,
        scenario_labels=scenario_dict,
        ylabel="DALYs averted (% relative to baseline)",
        xticklabels_wrapped=True,
    )
    fig.savefig(figurespath / "dalys_averted_by_disease_group.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


    # --------------------------------------------------
    # 3) TOTAL SERVICES DELIVERED
    # --------------------------------------------------
    num_treatments_total = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event_non_blank_appt_footprint',
        custom_generate_series=get_num_treatments_total,
        do_scaling=True,
        suspended_results_folder=suspended_results_folder,
    ).pipe(set_param_names_as_column_index_level_0)

    num_incremental_treatments = (
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_treatments_total,
                comparison = comparator_draw,
                scaled=True
            )
        ).T.unstack(level='run')
    )

    num_incremental_treatments_summarized = summarize_aggregated_results_for_figure(
        num_incremental_treatments,
        scenario_subset,
        chosen_metric
    )

    fig, ax = do_standard_bar_plot_with_ci(
        num_incremental_treatments_summarized.clip(0.0),
        annotations=[
            f"{row[chosen_metric]*100:.1f}% "
            f"({row['lower']*100:.1f}–{row['upper']*100:.1f}%)"
            for _, row in num_incremental_treatments_summarized.iterrows()
        ],
        xticklabels_wrapped=True,
        put_labels_in_legend=False,
        offset=0.05,
        scenarios_dict=scenario_dict
    )
    ax.set_ylabel('Additional services delivered (% relative to baseline)')
    ax.set_ylim(bottom=0)
    fig.savefig(figurespath / 'incremental_services_delivered_total.png', dpi=600)
    plt.close(fig)


    # --------------------------------------------------
    # 4) SERVICES BY DISEASE GROUP
    # --------------------------------------------------
    num_treatments_by_disease_group = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='HSI_Event_non_blank_appt_footprint',
        custom_generate_series=get_num_treatments_by_disease_group,
        do_scaling=True,
        suspended_results_folder=suspended_results_folder,
    ).pipe(set_param_names_as_column_index_level_0)

    num_treatments_by_disease_group['disease_group'] = (
        num_treatments_by_disease_group.index.map(service_to_group)
    )

    num_treatments_by_disease_group = (
        num_treatments_by_disease_group
        .set_index("disease_group", append=True)
        .groupby(level="disease_group")
        .sum()
    )

    num_incremental_by_group = pd.DataFrame(
        find_difference_relative_to_comparison(
            num_treatments_by_disease_group,
            comparison=comparator_draw,
            scaled=True
        )
    )

    summarized = summarize_disaggregated_results_for_figure(
        num_incremental_by_group,
        scenario_subset,
        chosen_metric
    )

    fig, ax = plot_percentage_change_with_ci(
        summary_df=summarized,
        colors=disease_colors,
        scenario_labels=scenario_dict,
        ylabel="Additional services (% relative to baseline)",
        xticklabels_wrapped=True,
    )
    fig.savefig(figurespath / "incremental_services_delivered_by_disease_group.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


    # --------------------------------------------------
    # 5) MAXIMUM ABILITY TO PAY
    # --------------------------------------------------
    max_ability_to_pay = (
        get_monetary_value_of_incremental_health(
            num_dalys_averted,
            _chosen_value_of_life_year=chosen_cet
        )
    )

    max_ability_to_pay = max_ability_to_pay[
        max_ability_to_pay.index.get_level_values('draw').isin(scenario_subset)
    ]

    max_ability_to_pay_summarized = summarize_cost_data(
        max_ability_to_pay,
        _metric=chosen_metric
    ).clip(lower=0.0)

    max_ability_to_pay_summarized = reformat_with_draw_as_index_and_stat_as_column(
        max_ability_to_pay_summarized
    )

    projected_health_spending = estimate_projected_health_spending(resourcefilepath,
                                                                   results_folder,
                                                                   _years=list_of_relevant_years_for_costing,
                                                                   _discount_rate=0,
                                                                   _summarize=True,
                                                                   _metric=chosen_metric,
                                                                   suspended_results_folder=suspended_results_folder,)
    projected_health_spending_baseline = \
        projected_health_spending[projected_health_spending.index.get_level_values(0) == 0][chosen_metric][0]

    # Extract projected health spending table for appendix
    health_spending_per_capita_table = get_manuscript_ready_table_of_projected_health_spending(
        _relevant_period_for_costing=relevant_period_for_costing)
    health_spending_per_capita_table.to_csv(figurespath / 'projected_health_spending.csv', index=False)

    max_ability_to_pay_billions = max_ability_to_pay_summarized / 1e9
    max_ability_to_pay_pct_of_spending = 100 * max_ability_to_pay_summarized / projected_health_spending_baseline

    fig, ax = do_standard_bar_plot_with_ci(
        max_ability_to_pay_billions,
        xticklabels_wrapped=True,
        put_labels_in_legend=False,
        scenarios_dict=scenario_dict
    )
    ax.set_ylabel('Maximum ability to pay (USD billions)')
    ax.set_ylim(bottom=0)

    # Two-part data label per bar: absolute value (black, USD billions) above, and that same
    # value expressed as a % of projected health spending (blue) stacked directly beneath it.
    # Anchored via va='bottom'/va='top' at a shared y0 so the two blocks sit flush against each
    # other regardless of rendered font metrics.
    label_offset = 0.3  # billions; gap between the bar's upper CI and the label block
    label_gap = 0.04     # billions; gap between the black (value) and blue (%) blocks

    for xpos, draw in zip(ax.get_xticks(), max_ability_to_pay_billions.index):
        row = max_ability_to_pay_billions.loc[draw]
        pct_row = max_ability_to_pay_pct_of_spending.loc[draw]

        value_text = f"{row[chosen_metric]:.2f}\n[{row['lower']:.2f}–{row['upper']:.2f}]"
        pct_text = f"{pct_row[chosen_metric]:.1f}%\n[{pct_row['lower']:.1f}%–{pct_row['upper']:.1f}%]"

        y0 = row['upper'] + label_offset
        ax.text(xpos, y0 + label_gap, value_text, ha='center', va='bottom',
                fontsize='x-small', color='black')
        ax.text(xpos, y0, pct_text, ha='center', va='top',
                fontsize='x-small', color='tab:blue')

    # Extend the y-limit so the (now taller, 4-line) label blocks aren't clipped.
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    fig.savefig(figurespath / 'max_ability_to_pay.png',
                dpi=300, bbox_inches="tight")
    plt.close(fig)


    # --------------------------------------------------
    # 6) CONSUMABLE AVAILABILITY
    # --------------------------------------------------
    item_to_program_df = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'consumables' / 'ResourceFile_Consumables_Item_Designations.csv'
    )[['Item_Code', 'item_category']]

    item_to_program_map = dict(
        zip(
            item_to_program_df['Item_Code'].astype(str),
            item_to_program_df['item_category']
        )
    )

    # Plot the proportion of instances that a consumable was available when requested
    pct_available_by_program = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Consumables',
        custom_generate_series=make_pct_available_by_group_fn(item_to_program_map),
        do_scaling=False,
        suspended_results_folder=suspended_results_folder,
    )

    pct_available_by_program_summarized = summarize_disaggregated_results_for_figure(
        pct_available_by_program,
        scenario_dict,
        chosen_metric
    )

    fig, ax = plot_percentage_change_with_ci(
        summary_df=pct_available_by_program_summarized,
        colors=disease_colors,
        scenario_labels=scenario_dict,
        ylabel="% instances of consumables being available",
        title="",
        xticklabels_wrapped=True,
    )
    fig.savefig(figurespath / "pct_available_by_program.png",
                dpi=300, bbox_inches="tight")

    # ---- Same consumable-availability data, regrouped to match the DALYs disease groups ----
    # (used in section 7 below, to plot each disease group's own change in consumable
    # availability against its own change in DALYs averted). Multiple consumables "programs"
    # collapse into RMNCH; programs with no corresponding DALYs disease group (general, ncds,
    # tb, undernutrition) are dropped rather than guessed at.
    CONS_PROGRAM_TO_DALY_GROUP = {
        'cancer': 'Cancer',
        'cardiometabolicdisorders': 'Cardiometabolic',
        'hiv': 'HIV/AIDS',
        'road_traffic_injuries': 'Injuries',
        'malaria': 'Malaria',
        'reproductive_health': 'RMNCH',
        'epi': 'RMNCH',
        'neonatal_health': 'RMNCH',
        'other_childhood_illnesses': 'RMNCH',
        'contraception': 'RMNCH',
    }
    item_to_daly_group_map = {
        item: CONS_PROGRAM_TO_DALY_GROUP[prog]
        for item, prog in item_to_program_map.items()
        if prog in CONS_PROGRAM_TO_DALY_GROUP
    }

    pct_available_by_daly_group = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Consumables',
        custom_generate_series=make_pct_available_by_group_fn(item_to_daly_group_map),
        do_scaling=False,
        suspended_results_folder=suspended_results_folder,
    )

    pct_available_by_daly_group_summarized = summarize_disaggregated_results_for_figure(
        pct_available_by_daly_group,
        scenario_dict,
        chosen_metric
    )
    delta_mean_available_by_daly_group = compute_delta_availability_from_baseline(
        pct_available_by_daly_group_summarized,
        comparator_draw=comparator_draw
    )

    delta_mean_available, nat_mean_available, nat_lower_available, nat_upper_available = compute_national_availability_summary(
        pct_available_by_program_summarized,
        comparator_draw = comparator_draw
    )

    plot_heatmap_delta(delta_mean_available, scenario_labels=scenario_dict, baseline_draw=comparator_draw,
                       legend_label = "Change in % available (vs baseline)")
    plt.savefig(figurespath / "pct_change_in_availability_by_scenario_and_program_heatmap.png",
                dpi=300, bbox_inches="tight")

    plot_change_in_cons_availability_by_program(
        delta_mean_available,
    )
    plt.savefig(figurespath / "change_in_cons_availability_by_program.png",
                dpi=300, bbox_inches="tight")

    plot_change_in_cons_availability_by_scenario(
        nat_mean_available,
        nat_lower_available,
        nat_upper_available,
        scenario_labels=scenario_dict
    )
    plt.savefig(figurespath / "change_in_cons_availability_by_scenario.png",
                dpi=300, bbox_inches="tight")

    # --------------------------------------------------
    # 7) DALYS AVERTED (%) vs. CHANGE IN CONSUMABLE AVAILABILITY (one dot per scenario)
    # --------------------------------------------------
    # num_dalys_averted_percent_summarized excludes the comparator draw (dropped upstream by
    # find_difference_relative_to_comparison, since "DALYs averted vs. itself" is trivially 0);
    # fill it back in as 0 so the comparator still shows up as the (0, 0) reference point.
    dalys_averted_pct = (
        (num_dalys_averted_percent_summarized[chosen_metric] * 100)
        .reindex(scenario_subset)
        .fillna(0.0)
    )

    change_in_availability_overall = (nat_mean_available).reindex(scenario_subset)

    fig, ax = plot_dalys_averted_vs_change_in_availability(
        change_in_availability=change_in_availability_overall,
        dalys_averted=dalys_averted_pct,
        scenario_labels=scenario_dict,
    )
    fig.savefig(figurespath / "dalys_averted_vs_change_in_cons_availability.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Same scatter, repeated once per DALYs disease group: BOTH axes now restricted to that
    # group -- x-axis is that group's own change in consumable availability (from
    # delta_mean_available_by_daly_group, computed above), y-axis is that group's own change in
    # DALYs averted (from num_dalys_averted_by_group_summarized, section 2), so each panel shows
    # a within-group relationship rather than mixing a group-specific x against an overall y.
    by_program_figurespath = figurespath / "dalys_averted_vs_change_in_cons_availability_by_program"
    by_program_figurespath.mkdir(parents=True, exist_ok=True)

    for daly_group in delta_mean_available_by_daly_group.index:
        change_in_availability_group = (
            delta_mean_available_by_daly_group.loc[daly_group].reindex(scenario_subset)
        )
        dalys_averted_pct_group = (
            (num_dalys_averted_by_group_summarized.loc[daly_group].xs("mean", level="stat") * 100)
            .reindex(scenario_subset)
            .fillna(0.0)
        )

        fig, ax = plot_dalys_averted_vs_change_in_availability(
            change_in_availability=change_in_availability_group,
            dalys_averted=dalys_averted_pct_group,
            scenario_labels=scenario_dict,
            title=daly_group,
        )
        ax.set_ylabel(f"DALYs averted within {daly_group} (%, relative to baseline)")
        safe_name = daly_group.replace("/", "_").replace(" ", "_")
        fig.savefig(
            by_program_figurespath / f"dalys_averted_vs_change_in_cons_availability_{safe_name}.png",
            dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    print("✓ All figures generated.")

generate_all_consumable_figures(
    scenario_dict=cons_scenarios_main,
    results_folder=results_folder,
    suspended_results_folder=suspended_results_folder,
    figurespath=figurespath / "main",
    comparator_draw=0,
)

generate_all_consumable_figures(
    scenario_dict=cons_scenarios_perfect,
    results_folder=results_folder,
    suspended_results_folder=suspended_results_folder,
    figurespath=figurespath / "perfect",
    comparator_draw=13,
)
