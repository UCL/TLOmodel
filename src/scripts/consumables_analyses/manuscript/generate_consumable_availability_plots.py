"""
Box (or violin) plot of the DISTRIBUTION of consumable availability -- not just the mean --
across the scenarios used in the manuscript.

Motivation
----------
The averaged program x scenario heatmap (descriptive_analysis.py) collapses every
facility-item-month to a single mean per cell, which hides *how* a scenario achieves that
mean: whether it shifts every facility's availability by roughly the same amount (as a
benchmark facility scenario built from a reference distribution would), or whether it leaves
most facilities untouched and pushes only the deficit-holding ones up (as redistribution,
which is targeted and supply-conserving, should). Two scenarios with a similar average can
have very different underlying distributions.

Read-only: loads the compiled availability resource file directly. Does not import or modify
any of the redistribution pipeline scripts.
"""
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.data_file_processing.healthsystem.consumables.generating_consumable_scenarios.redistribution_utils import (
    _add_custom_legend,
)

resourcefilepath = Path("./resources")
consumable_resourcefilepath = resourcefilepath / "healthsystem/consumables"
outputfilepath = Path("./outputs/consumables_impact_analysis")

# Scenarios used in the manuscript, and the order they should appear in on the x-axis
# (benchmark facility scenarios first, then redistribution scenarios: national, district,
# cluster, 60-min, 30-min).
SCENARIO_NAMES = {
    'available_prop': 'Actual',
    'available_prop_scenario1': 'Non-therapeutic consumables (NTC)',
    'available_prop_scenario2': 'NTC + Vital medicines (VM)',
    'available_prop_scenario3': 'NTC + VM + Pharmacist-managed stocks',
    'available_prop_scenario6': '75th percentile\n  facility',
    'available_prop_scenario7': '90th percentile \n facility',
    'available_prop_scenario8': 'Best \n facility',
    'available_prop_scenario17': 'Neighbourhood pooling',
    'available_prop_scenario16': 'District pooling',
    'available_prop_scenario20': 'National pooling',
    'available_prop_scenario19': 'Pairwise exchange (Small radius)',
    'available_prop_scenario18': 'Pairwise exchange (large radius)',
}
SCENARIO_ORDER = list(SCENARIO_NAMES.values())

# Default figure convention going forward: x-axis category labels are wrapped to a fixed
# character width and rotated 90 degrees, rather than angled -- keeps long scenario/category
# names fully legible without overlapping regardless of how many categories are on the axis.
DEFAULT_XTICK_WRAP_WIDTH = 20


def wrap_and_rotate_xticklabels(ax, width: int = DEFAULT_XTICK_WRAP_WIDTH, fontsize: int = 9):
    """Wrap each x-tick label to `width` characters and rotate 90 degrees. Call this AFTER
    the axis's category order is finalised (e.g. after the last plotting call on `ax`)."""
    labels = [textwrap.fill(t.get_text(), width) for t in ax.get_xticklabels()]
    ax.set_xticks(ax.get_xticks())  # fix tick positions before relabelling (avoids a matplotlib warning)
    ax.set_xticklabels(labels, rotation=90, ha='center', va='top', fontsize=fontsize)


def load_availability_long(
    include_levels=("1a",),
    resource_filename: str = "ResourceFile_Consumables_availability_small_original.csv",
) -> pd.DataFrame:
    """
    Load the compiled availability resource file and reshape the manuscript's scenario columns
    to long format: one row per (Facility_ID, item_code, month, scenario, availability).
    """
    df = pd.read_csv(consumable_resourcefilepath / resource_filename)

    mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
    df = df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']], on='Facility_ID', how='left')
    if include_levels:
        df = df[df['Facility_Level'].isin(include_levels)]

    keep_cols = [c for c in SCENARIO_NAMES if c in df.columns]
    long_df = df.melt(
        id_vars=['Facility_ID', 'District', 'Facility_Level', 'item_code', 'month'],
        value_vars=keep_cols,
        var_name='scenario_col', value_name='availability',
    )
    long_df['scenario'] = long_df['scenario_col'].map(SCENARIO_NAMES)
    long_df['scenario'] = pd.Categorical(long_df['scenario'], categories=SCENARIO_ORDER, ordered=True)
    return long_df.dropna(subset=['availability'])


def plot_availability_distribution_by_scenario(
    long_df: pd.DataFrame,
    figures_path: Path = outputfilepath,
    figname: str = "availability_distribution_by_scenario.png",
    kind: str = "box",  # "box" (default; robust at scale) or "violin"
    title: str = "Distribution of consumable availability by scenario",
):
    """
    Box (default) or violin plot of the availability LEVEL (not the change in availability)
    per scenario, across every facility-item-month in `long_df` -- shows the full shape and
    spread, not just the mean shown in the heatmap. Mean (diamond) and median (circle) are
    overlaid, matching the visual convention used elsewhere in the appendix.
    """
    order = [c for c in SCENARIO_ORDER if c in long_df['scenario'].cat.categories
             and (long_df['scenario'] == c).any()]
    mean_df = long_df.groupby('scenario', observed=True)['availability'].mean().reindex(order).reset_index()
    median_df = long_df.groupby('scenario', observed=True)['availability'].median().reindex(order).reset_index()

    fig, ax = plt.subplots(figsize=(13, 6))

    if kind == "violin":
        sns.violinplot(data=long_df, x='scenario', y='availability', order=order, cut=0,
                       density_norm='width', inner=None, linewidth=0.8, color="#4C72B0",
                       alpha=0.6, ax=ax)
        sns.boxplot(data=long_df, x='scenario', y='availability', order=order, width=0.04,
                   showcaps=True, showfliers=False,
                   boxprops={"facecolor": "grey", "edgecolor": "black", "linewidth": 1},
                   whiskerprops={"linewidth": 1}, medianprops={"linewidth": 0}, ax=ax)
    else:
        sns.boxplot(data=long_df, x='scenario', y='availability', order=order, width=0.55,
                   showcaps=True, showfliers=True, fliersize=1.2,
                   boxprops={"facecolor": "#4C72B0", "edgecolor": "black", "linewidth": 1, "alpha": 0.75},
                   whiskerprops={"linewidth": 1}, medianprops={"linewidth": 1.4, "color": "black"}, ax=ax)

    sns.scatterplot(data=mean_df, x='scenario', y='availability', color="#b2182b", marker="D",
                    s=60, zorder=10, ax=ax)
    sns.scatterplot(data=median_df, x='scenario', y='availability', color="#b2182b", marker="o",
                    s=45, zorder=11, ax=ax)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("")
    ax.set_ylabel("Proportion of days on which consumable is available")
    wrap_and_rotate_xticklabels(ax)
    ax.set_title(title, fontsize=11)
    _add_custom_legend(legend_location="lower right")

    fig.tight_layout()
    figures_path.mkdir(parents=True, exist_ok=True)
    outpath = figures_path / figname
    fig.savefig(outpath, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {outpath}")
    return fig


if __name__ == "__main__":
    for level in ("1a", "1b"):
        long_df = load_availability_long(include_levels=(level,))
        plot_availability_distribution_by_scenario(
            long_df,
            figname=f"availability_distribution_by_scenario_{level}.png",
            title="",
        )
