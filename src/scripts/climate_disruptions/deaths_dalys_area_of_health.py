import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    get_color_cause_of_death_or_daly_label,
    make_age_grp_lookup,
    summarize,
)

PREFIX_ON_FILENAME = "1"

SCENARIO_COLOURS_PALETTE = [
    "#823038",  # Default / No Disruptions
    "#00566f",  # SSP 1.26 High
    "#0081a7",  # SSP 1.26 Low
    "#5ab4c6",  # SSP 1.26 Mean
    "#5b3f8c",  # SSP 2.45 High
    "#8e7cc3",  # SSP 2.45 Low
    "#c7b7ec",  # SSP 2.45 Mean
    "#c65a52",  # SSP 5.85 High
    "#f07167",  # SSP 5.85 Low
    "#f59e96",  # SSP 5.85 Mean
]

MAIN_TEXT_COLOURS = {
    "No Disruptions": "#0081a7",
    "Default": "#FEB95F",
    "Worst Case": "#f07167",
}


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Produce standard set of plots describing the effect of each climate scenario.
    - We estimate the epidemiological impact as the EXTRA DALYs that would occur under climate disruption.
    """

    # ─────────────────────────────────────────────────────────────────────────────
    #  ANALYSIS SWITCHES  — set exactly one to True
    # ─────────────────────────────────────────────────────────────────────────────
    min_year = 2025
    max_year = 2041
    spacing_of_years = 1

    main_text = True
    climate_sensitivity_analysis = False
    parameter_sensitivity_analysis = False

    if climate_sensitivity_analysis:
        scenario_names = [
            "Default",
            "SSP 1.26 High", "SSP 1.26 Low", "SSP 1.26 Mean",
            "SSP 2.45 High", "SSP 2.45 Low", "SSP 2.45 Mean",
            "SSP 5.85 High", "SSP 5.85 Low", "SSP 5.85 Mean",
        ]
        scenarios_of_interest = range(len(scenario_names))
        suffix = "climate_SA"

    if parameter_sensitivity_analysis:
        num_draws = 50
        scenario_names = [f"Draw_{i}" for i in range(num_draws)]
        scenarios_of_interest = range(num_draws)
        suffix = "parameter_SA"

    if main_text:
        scenario_names = ["No Disruptions", "Default", "Worst Case"]
        scenarios_of_interest = [0, 1, 2]
        suffix = "main_text"

    # ─────────────────────────────────────────────────────────────────────────────

    cmap = cm.get_cmap("tab20", len(scenarios_of_interest))

    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    _, age_grp_lookup = make_age_grp_lookup()

    def get_num_dalys_by_cause_label(_df):
        """Return total number of DALYS (Stacked) by label (total by age-group within the TARGET_PERIOD)"""
        return (
            _df.loc[_df.year.between(*[i.year for i in TARGET_PERIOD])]
            .drop(columns=["date", "sex", "age_range", "year"])
            .sum()
        )

    def get_population_for_year(_df):
        """Returns the population in the year of interest"""
        _df["date"] = pd.to_datetime(_df["date"])
        filtered_df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
        numeric_df = filtered_df.drop(columns=["female", "male"], errors="ignore")
        return numeric_df.sum(numeric_only=True)

    target_year_sequence = range(min_year, max_year, spacing_of_years)

    all_draws_dalys_mean = []
    all_draws_dalys_lower = []
    all_draws_dalys_upper = []
    all_draws_dalys_mean_1000 = []
    all_draws_dalys_lower_1000 = []
    all_draws_dalys_upper_1000 = []

    for draw in scenarios_of_interest:
        print(f"Processing draw {draw}...")

        all_years_data_dalys_mean = {}
        all_years_data_dalys_upper = {}
        all_years_data_dalys_lower = {}
        all_years_data_population_mean = {}
        all_years_data_population_lower = {}
        all_years_data_population_upper = {}

        for target_year in target_year_sequence:
            TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))

            result_data_dalys = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.healthburden",
                    key="dalys_stacked_by_age_and_time",
                    custom_generate_series=get_num_dalys_by_cause_label,
                    do_scaling=False,
                ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            all_years_data_dalys_mean[target_year] = result_data_dalys["mean"]
            all_years_data_dalys_lower[target_year] = result_data_dalys["lower"]
            all_years_data_dalys_upper[target_year] = result_data_dalys["upper"]

            result_data_population = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.demography",
                    key="population",
                    custom_generate_series=get_population_for_year,
                    do_scaling=False,
                ),
                only_mean=True,
                collapse_columns=True,
            )[draw]
            all_years_data_population_mean[target_year] = result_data_population["mean"]
            all_years_data_population_lower[target_year] = result_data_population["lower"]
            all_years_data_population_upper[target_year] = result_data_population["upper"]

        df_all_years_DALYS_mean = pd.DataFrame(all_years_data_dalys_mean)
        df_all_years_DALYS_lower = pd.DataFrame(all_years_data_dalys_lower)
        df_all_years_DALYS_upper = pd.DataFrame(all_years_data_dalys_upper)

        df_all_years_data_population_mean = pd.DataFrame(all_years_data_population_mean)
        df_all_years_data_population_lower = pd.DataFrame(all_years_data_population_lower)
        df_all_years_data_population_upper = pd.DataFrame(all_years_data_population_upper)

        df_daly_per_1000_mean = df_all_years_DALYS_mean.div(df_all_years_data_population_mean.iloc[0, 0], axis=0) * 1000
        df_daly_per_1000_lower = df_all_years_DALYS_lower.div(df_all_years_data_population_lower.iloc[0, 0],
                                                              axis=0) * 1000
        df_daly_per_1000_upper = df_all_years_DALYS_upper.div(df_all_years_data_population_upper.iloc[0, 0],
                                                              axis=0) * 1000

        all_draws_dalys_mean.append(pd.Series(df_all_years_DALYS_mean.sum(axis=1), name=f"Draw {draw}"))
        all_draws_dalys_lower.append(pd.Series(df_all_years_DALYS_lower.sum(axis=1), name=f"Draw {draw}"))
        all_draws_dalys_upper.append(pd.Series(df_all_years_DALYS_upper.sum(axis=1), name=f"Draw {draw}"))

        all_draws_dalys_mean_1000.append(pd.Series(df_daly_per_1000_mean.mean(axis=1), name=f"Draw {draw}"))
        all_draws_dalys_lower_1000.append(pd.Series(df_daly_per_1000_lower.mean(axis=1), name=f"Draw {draw}"))
        all_draws_dalys_upper_1000.append(pd.Series(df_daly_per_1000_upper.mean(axis=1), name=f"Draw {draw}"))

    df_dalys_all_draws_mean = pd.concat(all_draws_dalys_mean, axis=1)
    df_dalys_all_draws_lower = pd.concat(all_draws_dalys_lower, axis=1)
    df_dalys_all_draws_upper = pd.concat(all_draws_dalys_upper, axis=1)

    df_dalys_all_draws_mean_1000 = pd.concat(all_draws_dalys_mean_1000, axis=1)
    df_dalys_all_draws_lower_1000 = pd.concat(all_draws_dalys_lower_1000, axis=1)
    df_dalys_all_draws_upper_1000 = pd.concat(all_draws_dalys_upper_1000, axis=1)

    # Rename columns from "Draw N" → scenario name for legibility in CSVs and plots
    col_map = {f"Draw {draw}": scenario_names[draw] for draw in scenarios_of_interest}
    df_dalys_all_draws_mean = df_dalys_all_draws_mean.rename(columns=col_map)
    df_dalys_all_draws_lower = df_dalys_all_draws_lower.rename(columns=col_map)
    df_dalys_all_draws_upper = df_dalys_all_draws_upper.rename(columns=col_map)
    df_dalys_all_draws_mean_1000 = df_dalys_all_draws_mean_1000.rename(columns=col_map)
    df_dalys_all_draws_lower_1000 = df_dalys_all_draws_lower_1000.rename(columns=col_map)
    df_dalys_all_draws_upper_1000 = df_dalys_all_draws_upper_1000.rename(columns=col_map)

    df_dalys_all_draws_mean.to_csv(output_folder / f"dalys_by_cause_all_draws_{suffix}.csv")
    df_dalys_all_draws_mean_1000.to_csv(output_folder / f"dalys_per_1000_by_cause_all_draws_{suffix}.csv")

    named_scenarios = [scenario_names[d] for d in scenarios_of_interest]
    n_scenarios = len(named_scenarios)
    causes = df_dalys_all_draws_mean_1000.index.tolist()
    n_causes = len(causes)
    y_base = np.arange(n_causes)

    if main_text:
        scenario_colours = [MAIN_TEXT_COLOURS.get(s, "#888888") for s in named_scenarios]
    else:
        scenario_colours = [SCENARIO_COLOURS_PALETTE[i % len(SCENARIO_COLOURS_PALETTE)]
                            for i in range(n_scenarios)]

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT A: Grouped horizontal bar — all scenarios side by side, by cause
    #          Activates for main_text and climate_sensitivity_analysis
    # ─────────────────────────────────────────────────────────────────────────────

    if main_text or climate_sensitivity_analysis:
        bar_height = 0.7 / n_scenarios

        fig_grp, ax_grp = plt.subplots(figsize=(13, max(8, n_causes * 0.55 + 2)))

        for i, scen in enumerate(named_scenarios):
            means = df_dalys_all_draws_mean_1000[scen].values
            lowers = df_dalys_all_draws_lower_1000[scen].values
            uppers = df_dalys_all_draws_upper_1000[scen].values
            xerr = np.array([
                np.clip(means - lowers, 0, None),
                np.clip(uppers - means, 0, None),
            ])
            offset = (i - (n_scenarios - 1) / 2) * bar_height

            ax_grp.barh(
                y_base + offset, means,
                height=bar_height, color=scenario_colours[i], alpha=0.85,
                label=scen,
            )
            ax_grp.errorbar(
                means, y_base + offset,
                xerr=xerr,
                fmt="none", color="black", lw=0.8, capsize=2, alpha=0.6,
            )

        ax_grp.set_yticks(y_base)
        ax_grp.set_yticklabels([c.replace("*", "") for c in causes], fontsize=10)
        ax_grp.invert_yaxis()
        ax_grp.set_xlabel("DALYs per 1,000 population", fontsize=12, fontweight="bold")
        ax_grp.set_title(
            f"DALYs per 1,000 by cause and scenario ({min_year}–{max_year - 1})\n"
            "error bars = 95% CI",
            fontsize=13, fontweight="bold",
        )
        ax_grp.legend(title="Scenario", fontsize=10, framealpha=0.9)
        ax_grp.grid(axis="x", alpha=0.3)
        ax_grp.set_xlim(left=0)
        fig_grp.tight_layout()
        fig_grp.savefig(
            output_folder / f"dalys_per_1000_grouped_by_scenario_{suffix}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig_grp)

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT B: Difference dot plot — all scenarios vs reference, by cause
    #          Activates for main_text and climate_sensitivity_analysis
    # ─────────────────────────────────────────────────────────────────────────────

    if main_text or climate_sensitivity_analysis:
        ref_scen = named_scenarios[0]  # first scenario = reference (No Disruptions / Default)
        non_ref = [s for s in named_scenarios if s != ref_scen]
        offsets_diff = {s: (i - (len(non_ref) - 1) / 2) * 0.25
                        for i, s in enumerate(non_ref)}

        fig_diff, ax_diff = plt.subplots(figsize=(12, max(6, n_causes * 0.5 + 2)))

        for scen in non_ref:
            colour = (
                MAIN_TEXT_COLOURS.get(scen, "#888888") if main_text
                else SCENARIO_COLOURS_PALETTE[named_scenarios.index(scen) % len(SCENARIO_COLOURS_PALETTE)]
            )
            offset = offsets_diff[scen]

            ref_means = df_dalys_all_draws_mean_1000[ref_scen]
            cmp_means = df_dalys_all_draws_mean_1000[scen]
            cmp_lowers = df_dalys_all_draws_lower_1000[scen]
            cmp_uppers = df_dalys_all_draws_upper_1000[scen]
            diffs = cmp_means - ref_means

            xerr = np.array([
                np.clip(cmp_means.values - cmp_lowers.values, 0, None),
                np.clip(cmp_uppers.values - cmp_means.values, 0, None),
            ])

            ax_diff.errorbar(
                x=diffs.values,
                y=y_base + offset,
                xerr=xerr,
                fmt="o", color=colour, label=scen,
                capsize=3, markersize=5, linewidth=1.2,
            )

        ax_diff.axvline(0, color="black", linewidth=1, linestyle="--",
                        label=f"{ref_scen} (reference)")
        ax_diff.set_yticks(y_base)
        ax_diff.set_yticklabels([c.replace("*", "") for c in causes], fontsize=10)
        ax_diff.invert_yaxis()
        ax_diff.set_xlabel(
            f"Difference in DALYs per 1,000 vs {ref_scen}", fontsize=12, fontweight="bold"
        )
        ax_diff.set_title(
            f"Excess DALYs per 1,000 relative to {ref_scen} ({min_year}–{max_year - 1})\n"
            "error bars = 95% CI",
            fontsize=13, fontweight="bold",
        )
        ax_diff.legend(title="Scenario", fontsize=10, framealpha=0.9)
        ax_diff.grid(axis="x", alpha=0.3)
        fig_diff.tight_layout()
        fig_diff.savefig(
            output_folder / f"{PREFIX_ON_FILENAME}_DALYs_DiffPlot_{suffix}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig_diff)

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOTS 1–5: Distribution / uncertainty plots  (most useful for parameter_SA)
    # ─────────────────────────────────────────────────────────────────────────────

    # 1. Total DALYs — box plot by cause
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    box_data, box_labels, box_colors = [], [], []
    for condition in df_dalys_all_draws_mean.index:
        box_data.append(df_dalys_all_draws_mean.loc[condition].values)
        box_labels.append(condition)
        box_colors.append(get_color_cause_of_death_or_daly_label(condition))
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=True)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color);
        patch.set_alpha(0.7)
    ax.set_title(
        f"Distribution of Total DALYs Across {n_scenarios} Draws ({min_year}–{max_year})"
    )
    ax.set_ylabel("Total DALYs");
    ax.set_xlabel("Cause")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_folder / f"total_dalys_distribution_by_cause_{suffix}.png", dpi=300)
    plt.close(fig)

    # 2. DALYs per 1,000 — box plot by cause
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    box_data, box_labels, box_colors = [], [], []
    for condition in df_dalys_all_draws_mean_1000.index:
        box_data.append(df_dalys_all_draws_mean_1000.loc[condition].values)
        box_labels.append(condition)
        box_colors.append(get_color_cause_of_death_or_daly_label(condition))
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=True)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color);
        patch.set_alpha(0.7)
    ax.set_title(
        f"Distribution of DALYs per 1,000 Across {n_scenarios} Draws (Mean {min_year}–{max_year})"
    )
    ax.set_ylabel("DALYs per 1,000");
    ax.set_xlabel("Cause")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_folder / f"dalys_per_1000_distribution_by_cause_{suffix}.png", dpi=300)
    plt.close(fig)

    # 3. Overall total DALYs — histogram + box
    total_dalys_all_draws = df_dalys_all_draws_mean.sum(axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.hist(total_dalys_all_draws.values, bins=max(5, n_scenarios // 2),
             color="steelblue", alpha=0.7, edgecolor="black")
    ax1.axvline(total_dalys_all_draws.mean(), color="red", linestyle="--", linewidth=2,
                label=f"Mean: {total_dalys_all_draws.mean():.0f}")
    ax1.axvline(total_dalys_all_draws.median(), color="orange", linestyle="--", linewidth=2,
                label=f"Median: {total_dalys_all_draws.median():.0f}")
    ax1.set_title(f"Distribution of Total DALYs Across {n_scenarios} Draws")
    ax1.set_xlabel("Total DALYs");
    ax1.set_ylabel("Frequency")
    ax1.legend();
    ax1.grid(axis="y", alpha=0.3)
    ax2.boxplot([total_dalys_all_draws.values], labels=["All Draws"], patch_artist=True)
    ax2.set_title("Total DALYs Summary");
    ax2.set_ylabel("Total DALYs")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_folder / f"total_dalys_overall_distribution_{suffix}.png", dpi=300)
    plt.close(fig)

    # 4. Mean DALYs per 1,000 by cause (horizontal bar, mean across all draws/scenarios)
    mean_dalys_by_cause = df_dalys_all_draws_mean_1000.mean(axis=1).sort_values(ascending=False)
    colors_sorted = [get_color_cause_of_death_or_daly_label(lbl) for lbl in mean_dalys_by_cause.index]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.barh(range(len(mean_dalys_by_cause)), mean_dalys_by_cause.values, color=colors_sorted)
    ax.set_yticks(range(len(mean_dalys_by_cause)))
    ax.set_yticklabels(mean_dalys_by_cause.index)
    ax.set_xlabel("Mean DALYs per 1,000 (across all draws)")
    ax.set_title(f"Mean DALYs per 1,000 by Cause (average across {n_scenarios} draws)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_folder / f"mean_dalys_per_1000_by_cause_{suffix}.png", dpi=300)
    plt.close(fig)

    # 5. Coefficient of variation
    cv_by_cause = (
                      df_dalys_all_draws_mean_1000.std(axis=1) / df_dalys_all_draws_mean_1000.mean(axis=1)
                  ) * 100
    cv_by_cause = cv_by_cause.sort_values(ascending=False)
    colors_cv = [get_color_cause_of_death_or_daly_label(lbl) for lbl in cv_by_cause.index]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.barh(range(len(cv_by_cause)), cv_by_cause.values, color=colors_cv, alpha=0.7)
    ax.set_yticks(range(len(cv_by_cause)))
    ax.set_yticklabels(cv_by_cause.index)
    ax.set_xlabel("Coefficient of Variation (%)")
    ax.set_title(f"Uncertainty by Cause (CV across {n_scenarios} draws)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_folder / f"cv_by_cause_{suffix}.png", dpi=300)
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────────
    #  CSV: Summary statistics
    # ─────────────────────────────────────────────────────────────────────────────

    for label, df in [
        ("dalys", df_dalys_all_draws_mean),
        ("dalys_per_1000", df_dalys_all_draws_mean_1000),
    ]:
        pd.DataFrame({
            "Mean": df.mean(axis=1), "Median": df.median(axis=1),
            "Std": df.std(axis=1), "Min": df.min(axis=1),
            "Max": df.max(axis=1), "Q25": df.quantile(0.25, axis=1),
            "Q75": df.quantile(0.75, axis=1),
        }).to_csv(output_folder / f"summary_statistics_{label}_{suffix}.csv")

    # ─────────────────────────────────────────────────────────────────────────────
    #  CONSOLE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────────

    print(f"\nSummary figures saved to {output_folder}")
    print(f"Total DALYs — Mean: {total_dalys_all_draws.mean():.0f}  "
          f"Median: {total_dalys_all_draws.median():.0f}  "
          f"Range: {total_dalys_all_draws.min():.0f}–{total_dalys_all_draws.max():.0f}  "
          f"Std: {total_dalys_all_draws.std():.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path("./resources"),
    )
