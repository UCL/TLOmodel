
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LogNorm

import pandas as pd
# import lacroix
import matplotlib.colors as colors
import numpy as np
import seaborn as sns
from collections import defaultdict
import textwrap
from typing import Tuple, Union

from scipy.stats import norm

from scripts.calibration_analyses.analysis_scripts.analysis_hsi_events_by_date import TARGET_PERIOD
from scripts.schistosomiasis.plot_scenario_runs import treatment_episodes
from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    format_gbd,
    make_age_grp_types,
    parse_log_file,
    compare_number_of_deaths,
    extract_params,
    compute_summary_statistics,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    compute_summary_statistics,
    make_age_grp_lookup,
    make_age_grp_types,
    unflatten_flattened_multi_index_in_logging,
)

from scripts.costing.cost_estimation import (estimate_input_cost_of_scenarios,
                                             summarize_cost_data,
                                             do_stacked_bar_plot_of_cost_by_category,
                                             do_line_plot_of_cost,
                                             create_summary_treemap_by_cost_subgroup,
                                             estimate_projected_health_spending)

resourcefilepath = Path("./resources")

output_folder = Path("./outputs/t.mangal@imperial.ac.uk")

results_folder = get_scenario_outputs("schisto_scenarios-2025.py", output_folder)[-1]

# Name of species that being considered:
species = ('mansoni', 'haematobium')

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

#################################################################################################
# multi-panel plot
#################################################################################################


#################
# PANEL 1
#################

# national-level person-years by age - no MDA = baseline scenario
# read in data by district and age
num_py = pd.read_excel(
    results_folder / "national_py_infected_by_age_year_mean_over_runs.xlsx",
    sheet_name=0,
    index_col=[0, 1]   # first two columns → row MultiIndex
)

# select draw
num_py_no_mda = num_py["Continue WASH, no MDA"]
num_py_mda_sac = num_py["Continue WASH, MDA SAC"]





def plot_stacked_py_with_stacked_overlay(
    s_base,
    s_overlay,
    year_min=2024,
    year_max=2050,
    title=None,
):
    wide_b = s_base.unstack("age_group").sort_index().loc[year_min:year_max]
    wide_o = s_overlay.unstack("age_group").sort_index().loc[year_min:year_max]

    infant_b = wide_b.get("Infant", 0.0)
    psac_b   = wide_b.get("PSAC", 0.0)
    sac_b    = wide_b.get("SAC", 0.0)
    adults_b = wide_b.get("Adults", 0.0)

    infant_o = wide_o.get("Infant", 0.0)
    psac_o   = wide_o.get("PSAC", 0.0)
    sac_o    = wide_o.get("SAC", 0.0)
    adults_o = wide_o.get("Adults", 0.0)

    base_1 = infant_b + psac_b
    base_2 = base_1 + sac_b
    base_3 = base_2 + adults_b

    # OVERLAY MUST BE CUMULATIVE (stack boundaries)
    ov_1 = infant_o + psac_o
    ov_2 = ov_1 + sac_o
    ov_3 = ov_2 + adults_o

    x = wide_b.index.to_numpy()

    # Your colours (areas)
    colours = {
        "Infant+PSAC": "#74c7a5",
        "SAC":         "#378ebb",
        "Adults":      "#9e0142",
    }

    # Slightly darker line colours (same hue family) so they read on top
    line_colours = {
        "Infant+PSAC": "#4aa88c",
        "SAC":         "#1f5f8b",
        "Adults":      "#7f002f",
    }

    fig, ax = plt.subplots(figsize=(6, 4))

    # Stacked areas use the COMPONENTS (not cumulative)
    ax.stackplot(
        x,
        base_1.to_numpy(),                 # Infant+PSAC component
        sac_b.to_numpy(),                  # SAC component
        adults_b.to_numpy(),               # Adults component
        colors=[colours["Infant+PSAC"], colours["SAC"], colours["Adults"]],
        labels=["Infant+PSAC", "SAC", "Adults"],
        alpha=0.7,
    )

    # Dashed stacked boundaries (cumulative)
    ax.plot(x, ov_1.to_numpy(), linestyle="--", linewidth=2, color=line_colours["Infant+PSAC"])
    ax.plot(x, ov_2.to_numpy(), linestyle="--", linewidth=2, color=line_colours["SAC"])
    ax.plot(x, ov_3.to_numpy(), linestyle="--", linewidth=2, color=line_colours["Adults"])

    ax.set_xlim(year_min, year_max)
    ax.set_xlabel("Year")
    ax.set_ylabel("Person-years infected")

    # ---- title & styling
    if title is None:
        title = s_base.name if getattr(s_base, "name", None) else \
            "National person-years infected by age"
    ax.set_title(title)

    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.5)
    ax.grid(axis="x", visible=False)

    plt.tight_layout()
    plt.savefig(results_folder / "stacked_py_with_stacked_overlay.png", dpi=300)
    plt.show()


plot_stacked_py_with_stacked_overlay(
    s_base=num_py_no_mda,
    s_overlay=num_py_mda_sac,
    year_min=2024,
    year_max=2050,
    title="Number person-years infected (national)"
)





def make_boxes_lines_key(colours, line_colours, figsize=(2, 1)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # tighter x positions
    x_box = 0.38
    x_line = 0.62

    # tighter y positions (top to bottom)
    y_rows = [0.72, 0.42, 0.12]
    age_groups = ["Infant+PSAC", "SAC", "Adults"]

    for y, age in zip(y_rows, age_groups):
        # shaded box (smaller)
        ax.add_patch(
            Rectangle(
                (x_box - 0.045, y - 0.045),
                0.09, 0.09,
                transform=ax.transAxes,
                facecolor=colours[age],
                edgecolor="none",
                alpha=0.7,
            )
        )

        # dashed line (shorter)
        ax.plot(
            [x_line - 0.06, x_line + 0.06],
            [y, y],
            transform=ax.transAxes,
            linestyle="--",
            linewidth=1.8,
            color=line_colours[age],
        )

    plt.tight_layout(pad=0)
    return fig, ax


colours = {
    "Infant+PSAC": "#74c7a5",
    "SAC":         "#378ebb",
    "Adults":      "#9e0142",
}

line_colours = {
    "Infant+PSAC": "#4aa88c",
    "SAC":         "#1f5f8b",
    "Adults":      "#7f002f",
}

fig, ax = make_boxes_lines_key(colours, line_colours)
plt.savefig(results_folder / "stacked_py_with_stacked_overlay_legend.png", dpi=300)
plt.show()



#################
# PANEL 2
#################

# adult reservoir vs incremental benefit of adult treatment_episodes

# reservoir
adult_reservoir= pd.read_excel(
    results_folder / "num_py_infected_by_district_and_age_2024_HML.xlsx",
    header=[0, 1],      # two header rows: draw / stat
    index_col=[0, 1]    # two index cols: district / age_group
)

adult_reservoir = (
    adult_reservoir.loc[pd.IndexSlice[:, "Adults"], "Continue WASH, MDA All"]
)
mean_adult_reservoir = adult_reservoir.mean(axis=1)
se_adult_reservoir = adult_reservoir.sem(axis=1)


# adult PY averted
py_averted = pd.read_excel(
    results_folder / "num_py_averted_by_district_age_2024-2050_HML_FULL.xlsx",
    header=[0, 1],      # two header rows: draw / run
    index_col=[0, 1]    # two index cols: district / age_group
)
# name the index levels
py_averted.index = py_averted.index.set_names(["district", "age_group"])

# 1) Adults only (rows remain indexed by district, age_group)
adults = py_averted.loc[pd.IndexSlice[:, "Adults"], :]

# 2) Get the two scenarios as (rows × runs) DataFrames
mda_all = adults.xs("Continue WASH, MDA All", level=1, axis=1)         # columns: run
psac_sac = adults.xs("Continue WASH, MDA PSAC+SAC", level=1, axis=1)   # columns: run

# 3) Run-by-run incremental benefit (rows × runs)
inc_by_run = mda_all - psac_sac

# 4) Mean and SE across runs (per row: district)
inc_mean = inc_by_run.mean(axis=1)
inc_se   = inc_by_run.sem(axis=1)

# (optional) combine for export
incremental_benefit_adults = pd.DataFrame({"mean": inc_mean, "se": inc_se})


# scatterplot
# Align everything on district index
x, y = mean_adult_reservoir.align(
    incremental_benefit_adults["mean"], join="inner"
)
x_se = se_adult_reservoir.loc[x.index]
y_se = incremental_benefit_adults.loc[y.index, "se"]



fig, ax = plt.subplots(figsize=(6, 5))

point_colour = "#378ebb"

ax.errorbar(
    x,
    y,
    xerr=1.96 * x_se,
    yerr=1.96 * y_se,
    fmt="o",
    markersize=5,
    color=point_colour,
    ecolor=point_colour,
    elinewidth=1,
    capsize=2,
    alpha=0.85,
)


threshold = 75_000

for idx in x.index:
    if x.loc[idx] > threshold:
        district = idx[0]  # level 0 = district
        ax.annotate(
            district,
            (x.loc[idx], y.loc[idx]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

ax.set_xlabel("Adult reservoir (person-years infected, 2023)")
ax.set_ylabel(
    "Incremental benefit, (person-years averted)"
)
ax.set_title("Incremental benefit of community MDA")
# Reference lines
ax.axhline(0, color="grey", linewidth=1)
ax.axvline(0, color="grey", linewidth=1)

ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.5)
ax.grid(axis="x", visible=False)

plt.tight_layout()
plt.savefig(results_folder / "adult_reservoir_vs_benefit.png", dpi=600)

plt.show()




#################
# PANEL 3
#################

# Cumulative DALYs averted by district
dalys_averted = pd.read_excel(
    results_folder / "total_dalys_averted_district_compared_noMDA2024-2050.xlsx",
    header=[0, 1],      # two header rows: draw / run
    index_col=[0]    # two index cols: district / age_group
)
dalys_averted.index = dalys_averted.index.set_names(["district"])

mask = dalys_averted.columns.get_level_values(0).str.contains("Continue", case=False)
dalys_averted_continue = dalys_averted.loc[:, mask]

dalys_averted_mean = dalys_averted_continue.groupby(axis=1, level="draw").mean()



baseline_dalys = pd.read_excel(
    results_folder / "schisto_dalys_by_year_run_district2024-2050.xlsx",
    header=[0, 1],      # two header rows: draw / run
    index_col=[0, 1]    # two index cols: district / age_group
)
baseline_dalys.index = baseline_dalys.index.set_names(["year", "district"])

mask = baseline_dalys.columns.get_level_values(0).str.contains("Continue", case=False)
baseline_dalys_continue = baseline_dalys.loc[:, mask]

baseline_dalys_continue2022 = baseline_dalys_continue.loc[2023]
baseline_dalys_continue2022_mean = baseline_dalys_continue2022.groupby(axis=1, level="draw").mean()

# align index safely
baseline_dalys_continue2022_mean, dalys_averted_mean = baseline_dalys_continue2022_mean.align(dalys_averted_mean, join="inner")
norm = dalys_averted_mean / baseline_dalys_continue2022_mean



norm = dalys_averted_mean


# Order columns as requested (adjust strings if yours differ exactly)
draw_order = [
    "Continue WASH, MDA SAC",
    "Continue WASH, MDA PSAC+SAC",
    "Continue WASH, MDA All",
]


# Reorder + keep only those draws (will raise if a name is missing)
hm = norm.loc[:, draw_order].copy()

exclude = []  # stochastic noise creates anomaly - Inf in SAC, All, v high value PSAC

hm = hm.drop(index=exclude, errors="ignore")


# mask undefined (baseline=0) and flip sign so "more averted" is larger
hm = (-hm).replace([np.inf, -np.inf], np.nan)

hm = hm.loc[~hm.isna().all(axis=1)]

vmax = np.nanpercentile(hm.values, 95)
vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0

# remove those with NaN - already eliminated by baseline
masked = np.ma.masked_invalid(hm.values)

# Colormap: sequential for magnitude; grey for undefined (baseline=0)
cmap = plt.cm.viridis.copy()
cmap.set_bad("lightgrey")

fig, ax = plt.subplots(figsize=(6.5, 10))

# im = ax.imshow(
#     masked,
#     aspect="auto",
#     cmap=cmap,
#     vmin=0,
#     vmax=vmax,
#     interpolation="nearest",
# )

norm_scale = LogNorm(
    vmin=np.nanmin(hm.values[hm.values > 0]),
    vmax=np.nanmax(hm.values)
)

im = ax.imshow(
    masked,
    aspect="auto",
    cmap=cmap,
    norm=norm_scale
)

# Axis labels/ticks
ax.set_xticks(range(len(draw_order)))
ax.set_xticklabels(["MDA SAC", "MDA PSAC+SAC", "MDA All"])
ax.set_yticks(range(hm.shape[0]))
ax.set_yticklabels(hm.index)

ax.set_xlabel("MDA strategy")
ax.set_ylabel("District")

# Subtle grid lines to help track rows/columns
ax.set_xticks(np.arange(-.5, len(draw_order), 1), minor=True)
ax.set_yticks(np.arange(-.5, hm.shape[0], 1), minor=True)
ax.grid(which="minor", linewidth=0.3)
ax.tick_params(which="minor", bottom=False, left=False)

# Colourbar with scientific notation (single exponent)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Normalised DALYs averted (2025–2050) relative to baseline (grey = baseline 0)")

fmt = ScalarFormatter(useMathText=True)
fmt.set_scientific(True)
fmt.set_powerlimits((0, 0))  # force a single ×10^k factor
cbar.formatter = fmt
cbar.update_ticks()

plt.tight_layout()
plt.show()


