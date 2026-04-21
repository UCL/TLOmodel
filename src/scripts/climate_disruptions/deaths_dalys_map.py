import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tlo import Date
from tlo.analysis.utils import extract_results, summarize

from plot_configurations import (FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND,
                                 FS_PANEL, FS_SUPTITLE, SCENARIO_COLOURS,
                                 apply_style)

# Configuration
MIN_YEAR = 2025
MAX_YEAR = 2041
SPACING_OF_YEARS = 1
PREFIX_ON_FILENAME = "1"
SCALING_FACTOR = 145.39
VMIN = -15
VMAX = 15

DISTRICT_COLOURS = [
    "red", "blue", "green", "orange", "purple", "brown", "pink", "gray",
    "olive", "cyan", "magenta", "yellow", "black", "navy", "maroon", "teal",
    "lime", "aqua", "fuchsia", "silver", "gold", "indigo", "violet", "crimson",
    "coral", "salmon", "khaki", "plum", "orchid", "tan", "wheat", "azure",
]

# --- Select analysis mode (set exactly one to True) ---
CLIMATE_SENSITIVITY_ANALYSIS = False
PARAMETER_SENSITIVITY_ANALYSIS = False
MAIN_TEXT = True

if CLIMATE_SENSITIVITY_ANALYSIS:
    SCENARIO_NAMES = [
        "Baseline",
        "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean",
        "SSP 2.45 High", "SSP 2.45 Low", "SSP 2.45 Mean",
        "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean",
    ]
    SUFFIX = "climate_SA"
    SCENARIOS_OF_INTEREST = list(range(len(SCENARIO_NAMES)))

if PARAMETER_SENSITIVITY_ANALYSIS:
    SCENARIO_NAMES = list(range(0, 9))
    SUFFIX = "parameter_SA"
    SCENARIOS_OF_INTEREST = list(SCENARIO_NAMES)

if MAIN_TEXT:
    SCENARIO_NAMES = ["No Disruptions", "Default", "Worst case"]
    SUFFIX = "main_text"
    SCENARIOS_OF_INTEREST = [0, 1, 2]


# Helper functions
def get_num_dalys_by_district(_df, period):
    """Sum DALYs across all causes, returning totals indexed by district."""
    _df["date"] = pd.to_datetime(_df["date"])
    filtered = _df.loc[_df["date"].between(*period)]
    filtered = filtered.drop(columns=["date", "year"])
    return filtered.groupby("district_of_residence").sum().sum(axis=1)


def get_monthly_dalys_total(_df, period):
    """Return total DALYs (all districts summed) indexed by date."""
    _df["date"] = pd.to_datetime(_df["date"])
    filtered = _df.loc[_df["date"].between(*period)].copy()
    filtered = filtered.drop(columns=["year"], errors="ignore")
    numeric_cols = filtered.select_dtypes(include="number").columns
    filtered["total_dalys"] = filtered[numeric_cols].sum(axis=1)
    return filtered.groupby("date")["total_dalys"].sum()


def get_monthly_dalys_by_district(_df, period):
    """Return DALYs per (date, district) as a MultiIndex Series."""
    _df["date"] = pd.to_datetime(_df["date"])
    filtered = _df.loc[_df["date"].between(*period)].copy()
    filtered = filtered.drop(columns=["year"], errors="ignore")
    numeric_cols = [c for c in filtered.select_dtypes(include="number").columns if c != "date"]
    filtered["total_dalys"] = filtered[numeric_cols].sum(axis=1)
    return filtered.groupby(["date", "district_of_residence"])["total_dalys"].sum()


def extract_and_summarize(results_folder, custom_fn, draw, only_mean=False):
    """Wrapper around extract_results + summarize for the healthburden DALY log."""
    return summarize(
        extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_by_district_stacked_by_age_and_time",
            custom_generate_series=custom_fn,
            do_scaling=False,
        ),
        only_mean=only_mean,
        collapse_columns=True,
    )[draw]


# Main analysis function
def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce plots and CSVs describing climate scenario effects on DALYs."""

    apply_style()  # apply shared style once at the start

    def graph_path(stub, draw):
        return output_folder / f"{PREFIX_ON_FILENAME}_{stub}_{draw}.png"

    full_period = (Date(MIN_YEAR, 1, 1), Date(MAX_YEAR, 12, 31))
    active_scenario_names = [SCENARIO_NAMES[i] for i in SCENARIOS_OF_INTEREST]

    # Diagnostic: individual run totals
    raw = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_by_district_stacked_by_age_and_time",
        custom_generate_series=lambda df: get_num_dalys_by_district(df, full_period),
        do_scaling=False,
    )
    for draw in SCENARIOS_OF_INTEREST:
        runs = raw.xs(draw, axis=1, level='draw')
        totals = runs.sum(axis=0)
        print(f"Draw {draw} ({SCENARIO_NAMES[draw]}): run totals = {totals.values.round(0)}")

    # 1. Per-year extraction: district-level DALY totals
    all_scenarios_dalys = {}
    all_scenarios_dalys_upper = {}
    all_scenarios_dalys_lower = {}
    all_scenarios_dalys_p1k = {}
    all_scenarios_dalys_upper_p1k = {}
    all_scenarios_dalys_lower_p1k = {}

    for draw in SCENARIOS_OF_INTEREST:
        scenario_name = SCENARIO_NAMES[draw]
        print(f"Processing draw {draw}: {scenario_name}")

        yearly_mean, yearly_lower, yearly_upper = {}, {}, {}

        for year in range(MIN_YEAR, MAX_YEAR, SPACING_OF_YEARS):
            period = (Date(year, 1, 1), Date(year, 12, 31))
            result = extract_and_summarize(
                results_folder,
                custom_fn=lambda df, p=period: get_num_dalys_by_district(df, p),
                draw=draw,
            )
            yearly_mean[year] = result["mean"] * SCALING_FACTOR
            yearly_lower[year] = result["lower"] * SCALING_FACTOR
            yearly_upper[year] = result["upper"] * SCALING_FACTOR

        df_mean = pd.DataFrame(yearly_mean)
        df_lower = pd.DataFrame(yearly_lower)
        df_upper = pd.DataFrame(yearly_upper)

        # --- Line plot: per-district DALYs across years ---
        fig, ax = plt.subplots(figsize=(15, 10))
        for i, district in enumerate(df_mean.index):
            ax.plot(df_mean.columns, df_mean.loc[district], marker="o",
                    label=district, color=DISTRICT_COLOURS[i])
        ax.set_xlabel("Year")
        ax.set_ylabel("DALYs")
        ax.legend(title="District", bbox_to_anchor=(1.0, 1), loc="upper left")
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(graph_path("Trend_DALYs_by_district_All_Years_Scatter", draw))
        plt.close(fig)

        # --- Stacked bar: district contribution over years ---
        fig, ax = plt.subplots(figsize=(15, 10))
        df_mean.T.plot.bar(
            stacked=True, ax=ax,
            color=[DISTRICT_COLOURS[i] for i in range(len(df_mean.index))],
        )
        ax.axhline(0, color="black")
        ax.set_title(f"DALYs by District Over Time — {scenario_name}")
        ax.set_ylabel("Number of DALYs")
        ax.set_xlabel("Year")
        ax.legend(title="District", bbox_to_anchor=(1.05, 1), loc="upper left",
                  fontsize=FS_LEGEND - 3, ncol=2)
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(graph_path("Trend_DALYs_by_district_All_Years_Stacked", draw))
        plt.close(fig)

        all_scenarios_dalys[scenario_name] = df_mean.sum(axis=1)
        all_scenarios_dalys_upper[scenario_name] = df_upper.sum(axis=1)
        all_scenarios_dalys_lower[scenario_name] = df_lower.sum(axis=1)
        all_scenarios_dalys_p1k[scenario_name] = df_mean.mean(axis=1)
        all_scenarios_dalys_upper_p1k[scenario_name] = df_upper.mean(axis=1)
        all_scenarios_dalys_lower_p1k[scenario_name] = df_lower.mean(axis=1)

    df_dalys = pd.DataFrame(all_scenarios_dalys)
    df_dalys_upper = pd.DataFrame(all_scenarios_dalys_upper)
    df_dalys_lower = pd.DataFrame(all_scenarios_dalys_lower)
    df_dalys_p1k = pd.DataFrame(all_scenarios_dalys_p1k)
    df_dalys_upper_p1k = pd.DataFrame(all_scenarios_dalys_upper_p1k)
    df_dalys_lower_p1k = pd.DataFrame(all_scenarios_dalys_lower_p1k)

    # -------------------------------------------------------------------------
    # 2. Stacked bar: district contribution to total DALYs per scenario
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    df_dalys.T.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("DALYs")
    ax.legend(title="District", bbox_to_anchor=(1.05, 1), loc="upper left",
              fontsize=FS_LEGEND - 3)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_folder / "stacked_dalys_by_district_scenario.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. Combined figure: (A) Dot plot with 95% CI  |  (B) Bar chart by scenario
    # -------------------------------------------------------------------------
    district_order = df_dalys[active_scenario_names[0]].sort_values(ascending=True).index
    n_districts = len(district_order)
    n_scenarios = len(active_scenario_names)

    dalys_totals = df_dalys.sum(axis=0)
    dalys_upper_total = df_dalys_upper.sum(axis=0)
    dalys_lower_total = df_dalys_lower.sum(axis=0)

    n_cols = 2
    fig = plt.figure(figsize=(14, max(8, n_districts * 0.35)))
    gs = fig.add_gridspec(1, n_cols, wspace=0.35)

    # --- Panel A: Dot plot ---
    ax_dot = fig.add_subplot(gs[0, 0])
    offsets = [(i - (n_scenarios - 1) / 2) * 0.25 for i in range(n_scenarios)]

    for s_idx, (scen_name, offset) in enumerate(zip(active_scenario_names, offsets)):
        colour = SCENARIO_COLOURS[s_idx]
        means = df_dalys[scen_name].loc[district_order]
        uppers = df_dalys_upper[scen_name].loc[district_order]
        lowers = df_dalys_lower[scen_name].loc[district_order]
        y_positions = [i + offset for i in range(n_districts)]
        ax_dot.errorbar(
            x=means, y=y_positions,
            xerr=[means - lowers, uppers - means],
            fmt="o", color=colour, ecolor=colour,
            elinewidth=1.2, capsize=3, markersize=5, alpha=0.85,
            label=scen_name,
        )

    ax_dot.set_yticks(range(n_districts))
    ax_dot.set_yticklabels(district_order, fontsize=FS_TICK)
    ax_dot.set_xlabel(f"Total DALYs ({MIN_YEAR}–{MAX_YEAR - 1})", fontsize=FS_LABEL, fontweight="bold")
    ax_dot.tick_params(axis="x", labelsize=FS_TICK)
    ax_dot.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax_dot.legend(title="Scenario", loc="lower right", fontsize=FS_LEGEND, ncol=n_scenarios)
    ax_dot.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    ax_dot.set_title("(A)", fontsize=FS_PANEL, fontweight="bold", loc="left")

    # --- Panel B: Bar chart ---
    baseline_total = dalys_totals["No Disruptions"]
    diff_means = dalys_totals - baseline_total
    diff_upper = dalys_upper_total - baseline_total
    diff_lower = dalys_lower_total - baseline_total

    plot_mask = dalys_totals.index != "No Disruptions"
    diff_means = diff_means[plot_mask]
    diff_upper = diff_upper[plot_mask]
    diff_lower = diff_lower[plot_mask]
    diff_colours = [SCENARIO_COLOURS[i] for i, name in enumerate(active_scenario_names)
                    if name != "No Disruptions"]
    diff_yerr = [np.abs(diff_means - diff_lower), np.abs(diff_upper - diff_means)]

    print("Difference in total DALYs vs No Disruptions:")
    print(diff_means)

    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.bar(
        range(len(diff_means)), diff_means,
        yerr=diff_yerr, capsize=5,
        color=diff_colours,
        alpha=0.8, error_kw={"elinewidth": 2, "capthick": 2},
    )
    ax_bar.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_bar.set_xlabel("Scenario", fontsize=FS_LABEL, labelpad=10, fontweight="bold")
    ax_bar.set_ylabel('Excess DALYs\nvs. "No Disruptions"', fontsize=FS_LABEL, labelpad=10, fontweight="bold")
    ax_bar.set_xticks(range(len(diff_means)))
    ax_bar.set_xticklabels(diff_means.index, rotation=0, ha="right", fontsize=FS_TICK)
    ax_bar.tick_params(axis="both", which="major", labelsize=FS_TICK)
    ax_bar.grid(False)
    ax_bar.set_title("(B)", fontsize=FS_PANEL, fontweight="bold", loc="left")

    fig.savefig(
        output_folder / "dalys_dotplot_barchart_combined.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 4. Monthly time series — aggregate total across all districts
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    for s_idx, draw in enumerate(SCENARIOS_OF_INTEREST):
        scen_name = SCENARIO_NAMES[draw]
        colour = SCENARIO_COLOURS[s_idx]
        ts = extract_and_summarize(
            results_folder,
            custom_fn=lambda df, p=full_period: get_monthly_dalys_total(df, p),
            draw=draw,
        )
        dates = pd.to_datetime(ts["mean"].index)
        ax.plot(dates, ts["mean"].values, color=colour, linewidth=1.8, label=scen_name)
        ax.fill_between(dates, ts["lower"].values, ts["upper"].values, color=colour, alpha=0.2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Total DALYs")
    ax.set_title("Monthly Total DALYs by Scenario (95% CI)")
    ax.legend(title="Scenario", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_folder / "dalys_monthly_timeseries_total.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 5. Monthly time series — small multiples by district
    # -------------------------------------------------------------------------
    district_ts_data = {}
    for draw in SCENARIOS_OF_INTEREST:
        scen_name = SCENARIO_NAMES[draw]
        district_ts_data[scen_name] = extract_and_summarize(
            results_folder,
            custom_fn=lambda df, p=full_period: get_monthly_dalys_by_district(df, p),
            draw=draw,
        )

    all_districts = sorted(
        district_ts_data[active_scenario_names[0]]["mean"].index.get_level_values(
            "district_of_residence").unique()
    )

    n_cols = 4
    n_rows = -(-len(all_districts) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             sharex=True, sharey=False)
    axes_flat = axes.flatten()

    for d_idx, district in enumerate(all_districts):
        ax = axes_flat[d_idx]
        for s_idx, draw in enumerate(SCENARIOS_OF_INTEREST):
            scen_name = SCENARIO_NAMES[draw]
            colour = SCENARIO_COLOURS[s_idx]
            ts = district_ts_data[scen_name]
            try:
                mean_vals = ts["mean"].xs(district, level="district_of_residence")
                lower_vals = ts["lower"].xs(district, level="district_of_residence")
                upper_vals = ts["upper"].xs(district, level="district_of_residence")
                dates = pd.to_datetime(mean_vals.index)
                ax.plot(dates, mean_vals.values, color=colour, linewidth=1.2,
                        label=scen_name if d_idx == 0 else "_nolegend_")
                ax.fill_between(dates, lower_vals.values, upper_vals.values,
                                color=colour, alpha=0.2)
            except KeyError:
                pass
        ax.set_title(district)
        ax.tick_params(axis="x", labelrotation=0)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    for ax in axes_flat[len(all_districts):]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Scenario", loc="lower right",
               bbox_to_anchor=(1.0, 0.01))
    fig.suptitle("Monthly DALYs by District with 95% CI", fontsize=FS_SUPTITLE,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_folder / "dalys_monthly_timeseries_by_district.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 6. Save summary CSV
    # -------------------------------------------------------------------------
    df_dalys_p1k.to_csv(output_folder / "dalys_by_district_scenario.csv", index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path("./resources"),
    )
