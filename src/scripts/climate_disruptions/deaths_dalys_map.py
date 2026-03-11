import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from tlo import Date
from tlo.analysis.utils import extract_results, summarize

# Configuration
MIN_YEAR = 2025
MAX_YEAR = 2040
SPACING_OF_YEARS = 1
PREFIX_ON_FILENAME = "1"

VMIN = -8
VMAX = 8

SCENARIO_COLOURS = ["#0081a7", "#00afb9", "#FEB95F", "#fed9b7", "#f07167"] * 4

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
    SCENARIO_NAMES = ["No disruptions", "Baseline", "Worst case"]
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

    def graph_path(stub, draw):
        return output_folder / f"{PREFIX_ON_FILENAME}_{stub}_{draw}.png"

    full_period = (Date(MIN_YEAR, 1, 1), Date(MAX_YEAR, 12, 31))
    active_scenario_names = [SCENARIO_NAMES[i] for i in SCENARIOS_OF_INTEREST]

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
            yearly_mean[year] = result["mean"]
            yearly_lower[year] = result["lower"]
            yearly_upper[year] = result["upper"]

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
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
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
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(title="District", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(graph_path("Trend_DALYs_by_district_All_Years_Stacked", draw))
        plt.close(fig)

        # Store cross-scenario summaries (sum = total over period, mean = annual avg)
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

    # 2. Bar chart: total DALYs by scenario with 95% CI error bars
    dalys_means = df_dalys.mean(axis=0)
    dalys_upper_agg = df_dalys_upper.mean(axis=0)
    dalys_lower_agg = df_dalys_lower.mean(axis=0)
    dalys_yerr = [dalys_means - dalys_lower_agg, dalys_upper_agg - dalys_means]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.bar(
        range(len(dalys_means)), dalys_means,
        yerr=dalys_yerr, capsize=5,
        color=SCENARIO_COLOURS[: len(active_scenario_names)],
        alpha=0.8, error_kw={"elinewidth": 2, "capthick": 2},
    )
    ax.set_title("DALYs by Scenario", fontsize=14, fontweight="bold")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("DALYs")
    ax.set_xticks(range(len(dalys_means)))
    ax.set_xticklabels(dalys_means.index, rotation=45, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_folder / "dalys_total_all_scenarios.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Stacked bar: district contribution to total DALYs per scenario

    fig, ax = plt.subplots(figsize=(12, 8))
    df_dalys.T.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("DALYs")
    ax.legend(title="District", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_folder / "stacked_dalys_by_district_scenario.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. Dot plot: per-district DALYs with 95% CI, colour-coded by scenario
    district_order = df_dalys[active_scenario_names[0]].sort_values(ascending=True).index
    n_districts = len(district_order)
    n_scenarios = len(active_scenario_names)
    offsets = [(i - (n_scenarios - 1) / 2) * 0.25 for i in range(n_scenarios)]

    fig, ax = plt.subplots(figsize=(12, max(8, n_districts * 0.35)))
    for s_idx, (scen_name, offset) in enumerate(zip(active_scenario_names, offsets)):
        colour = SCENARIO_COLOURS[s_idx]
        means = df_dalys[scen_name].loc[district_order]
        uppers = df_dalys_upper[scen_name].loc[district_order]
        lowers = df_dalys_lower[scen_name].loc[district_order]
        y_positions = [i + offset for i in range(n_districts)]
        ax.errorbar(
            x=means, y=y_positions,
            xerr=[means - lowers, uppers - means],
            fmt="o", color=colour, ecolor=colour,
            elinewidth=1.2, capsize=3, markersize=5, alpha=0.85,
            label=scen_name,
        )
    ax.set_yticks(range(n_districts))
    ax.set_yticklabels(district_order, fontsize=8)
    ax.set_xlabel(f"Total DALYs ({MIN_YEAR}–{MAX_YEAR - 1})", fontsize=11)
    ax.set_title("DALYs by District with 95% CI", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Scenario", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_folder / "dalys_by_district_CI_dotplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 5. Monthly time series — aggregate total across all districts
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

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Total DALYs", fontsize=11)
    ax.set_title("Monthly Total DALYs by Scenario (95% CI)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Scenario", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_folder / "dalys_monthly_timeseries_total.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 6. Monthly time series — small multiples by district
    district_ts_data = {}
    for draw in SCENARIOS_OF_INTEREST:
        scen_name = SCENARIO_NAMES[draw]
        district_ts_data[scen_name] = extract_and_summarize(
            results_folder,
            custom_fn=lambda df, p=full_period: get_monthly_dalys_by_district(df, p),
            draw=draw,
        )

    all_districts = sorted(
        district_ts_data[active_scenario_names[0]]["mean"]
        .index.get_level_values("district_of_residence").unique()
    )

    n_cols = 4
    n_rows = -(-len(all_districts) // n_cols)  # ceiling division
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
                pass  # district absent from this draw
        ax.set_title(district, fontsize=8, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelrotation=45, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    for ax in axes_flat[len(all_districts):]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Scenario", loc="lower right",
               bbox_to_anchor=(1.0, 0.01), fontsize=9)
    fig.suptitle("Monthly DALYs by District with 95% CI", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_folder / "dalys_monthly_timeseries_by_district.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 7. Maps: percentage difference in DALYs vs reference scenario
    malawi_admin2 = gpd.read_file(
        "/Users/rem76/PycharmProjects/TLOmodel/resources/mapping/"
        "ResourceFile_mwi_admbnda_adm2_nso_20181016.shp"
    )
    water_bodies = gpd.read_file(
        "/Users/rem76/Desktop/Climate_Change_Health/Data/"
        "Water_Supply_Control-Rivers-shp/Water_Supply_Control-Rivers.shp"
    )
    for old, new in [
        ("Blantyre City", "Blantyre"), ("Mzuzu City", "Mzuzu"),
        ("Lilongwe City", "Lilongwe"), ("Zomba City", "Zomba"),
    ]:
        malawi_admin2["ADM2_EN"] = malawi_admin2["ADM2_EN"].replace(old, new)

    non_baseline = SCENARIOS_OF_INTEREST[1:]
    n_maps = len(non_baseline)
    fig, axes = plt.subplots(1, n_maps, figsize=(10 * n_maps, 18))
    if n_maps == 1:
        axes = [axes]

    for ax_idx, scenario in enumerate(non_baseline):
        pct_diff = (
                       (df_dalys_p1k.iloc[:, scenario] - df_dalys_p1k.iloc[:, 0])
                       / df_dalys_p1k.iloc[:, 0]
                   ) * 100
        malawi_admin2["DALY_Rate"] = malawi_admin2["ADM2_EN"].map(pct_diff)
        malawi_admin2.plot(
            column="DALY_Rate", ax=axes[ax_idx], legend=True,
            cmap="PiYG", edgecolor="black", vmin=VMIN, vmax=VMAX,
        )
        axes[ax_idx].set_title(
            f"{SCENARIO_NAMES[scenario]} vs {SCENARIO_NAMES[SCENARIOS_OF_INTEREST[0]]}\n"
            f"% Difference in DALYs: {MIN_YEAR}–{MAX_YEAR - 1}",
            fontsize=28,
        )
        axes[ax_idx].axis("off")
        water_bodies.plot(ax=axes[ax_idx], facecolor="#7BDFF2", alpha=0.6,
                          edgecolor="#999999", linewidth=0.5, hatch="xxx")
        water_bodies.plot(ax=axes[ax_idx], facecolor="#7BDFF2",
                          edgecolor="black", linewidth=1)

    fig.tight_layout()
    fig.savefig(output_folder / "dalys_maps_all_scenarios_difference.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 8. Save summary CSV
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
