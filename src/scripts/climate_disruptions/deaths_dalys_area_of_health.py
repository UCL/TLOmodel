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

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
    "figure.dpi": 150,
    "lines.linewidth": 1.6,
})

SCENARIO_COLOURS_PALETTE = [
    "#823038", "#00566f", "#0081a7", "#5ab4c6",
    "#5b3f8c", "#8e7cc3", "#c7b7ec",
    "#c65a52", "#f07167", "#f59e96",
]

MAIN_TEXT_COLOURS = {
    "No Disruptions": "#0081a7",
    "Default": "#FEB95F",
    "Worst Case": "#f07167",
}

# ─────────────────────────────────────────────────────────────────────────────
#  DISTRICT → REGION LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

DISTRICT_REGION = {
    "Chitipa": "North", "Karonga": "North", "Rumphi": "North",
    "Mzimba": "North", "Likoma": "North",
    "Kasungu": "Centre", "Nkhata Bay": "Centre", "Nkhotakota": "Centre",
    "Ntchisi": "Centre", "Dowa": "Centre", "Salima": "Centre",
    "Lilongwe": "Centre", "Mchinji": "Centre", "Dedza": "Centre",
    "Ntcheu": "Centre",
    "Mangochi": "South", "Machinga": "South", "Zomba": "South",
    "Chiradzulu": "South", "Blantyre": "South", "Mwanza": "South",
    "Thyolo": "South", "Mulanje": "South", "Phalombe": "South",
    "Chikwawa": "South", "Nsanje": "South", "Balaka": "South",
    "Neno": "South",
}

REGION_ORDER = ["North", "Centre", "South"]
REGION_COLOURS = {"North": "#5b3f8c", "Centre": "#0081a7", "South": "#c65a52"}


def _sorted_districts(districts):
    def _key(d):
        region = DISTRICT_REGION.get(d, "Unknown")
        rank = REGION_ORDER.index(region) if region in REGION_ORDER else 99
        return (rank, d)

    return sorted(districts, key=_key)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):

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

    cmap = cm.get_cmap("tab20", len(scenarios_of_interest))

    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))

    _, age_grp_lookup = make_age_grp_lookup()

    # ─────────────────────────────────────────────────────────────────────────────
    #  GENERATOR FUNCTIONS
    # ─────────────────────────────────────────────────────────────────────────────

    def get_num_dalys_by_cause_label(_df):
        return (
            _df.loc[_df.year.between(*[i.year for i in TARGET_PERIOD])]
            .drop(columns=["date", "sex", "age_range", "year"])
            .sum()
        )

    def get_population_for_year(_df):
        _df["date"] = pd.to_datetime(_df["date"])
        filtered_df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
        numeric_df = filtered_df.drop(columns=["female", "male"], errors="ignore")
        return numeric_df.sum(numeric_only=True)

    def get_monthly_yld_by_district(_df):
        """YLD only — no YLL. Source: monthly_dalys_by_district."""
        mask = _df["year"].between(min_year, max_year - 1)
        sub = _df.loc[mask][["year", "month", "district_of_residence", "dalys"]]
        return sub.set_index(["year", "month", "district_of_residence"])["dalys"]

    def get_dalys_by_district(_df):
        mask = _df["year"].between(*[i.year for i in TARGET_PERIOD])
        sub = _df.loc[mask].drop(columns=["date", "year"], errors="ignore")
        cause_cols = [c for c in sub.columns if c != "district_of_residence"]
        return sub.groupby("district_of_residence")[cause_cols].sum().sum(axis=1)

    def get_dalys_by_cause_and_district(_df):
        mask = _df["year"].between(*[i.year for i in TARGET_PERIOD])
        sub = _df.loc[mask].drop(columns=["date", "year"], errors="ignore")
        cause_cols = [c for c in sub.columns if c != "district_of_residence"]
        return sub.groupby("district_of_residence")[cause_cols].sum().stack()

    # ─────────────────────────────────────────────────────────────────────────────
    #  CAUSE-LEVEL DALY EXTRACTION (year-by-year for per-1,000 normalisation)
    # ─────────────────────────────────────────────────────────────────────────────

    target_year_sequence = range(min_year, max_year, spacing_of_years)

    all_draws_dalys_mean, all_draws_dalys_lower, all_draws_dalys_upper = [], [], []
    all_draws_dalys_mean_1000, all_draws_dalys_lower_1000, all_draws_dalys_upper_1000 = [], [], []

    for draw in scenarios_of_interest:
        print(f"Processing draw {draw}...")

        all_years_dalys_mean, all_years_dalys_lower, all_years_dalys_upper = {}, {}, {}
        all_years_pop_mean, all_years_pop_lower, all_years_pop_upper = {}, {}, {}

        for target_year in target_year_sequence:
            TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))

            r_dalys = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.healthburden",
                    key="dalys_stacked_by_age_and_time",
                    custom_generate_series=get_num_dalys_by_cause_label,
                    do_scaling=False,
                ),
                only_mean=True, collapse_columns=True,
            )[draw]
            all_years_dalys_mean[target_year] = r_dalys["mean"]
            all_years_dalys_lower[target_year] = r_dalys["lower"]
            all_years_dalys_upper[target_year] = r_dalys["upper"]

            r_pop = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.demography",
                    key="population",
                    custom_generate_series=get_population_for_year,
                    do_scaling=False,
                ),
                only_mean=True, collapse_columns=True,
            )[draw]
            all_years_pop_mean[target_year] = r_pop["mean"]
            all_years_pop_lower[target_year] = r_pop["lower"]
            all_years_pop_upper[target_year] = r_pop["upper"]

        df_dalys_mean = pd.DataFrame(all_years_dalys_mean)
        df_dalys_lower = pd.DataFrame(all_years_dalys_lower)
        df_dalys_upper = pd.DataFrame(all_years_dalys_upper)
        df_pop_mean = pd.DataFrame(all_years_pop_mean)
        df_pop_lower = pd.DataFrame(all_years_pop_lower)
        df_pop_upper = pd.DataFrame(all_years_pop_upper)

        df_per1000_mean = df_dalys_mean.div(df_pop_mean.iloc[0, 0], axis=0) * 1000
        df_per1000_lower = df_dalys_lower.div(df_pop_lower.iloc[0, 0], axis=0) * 1000
        df_per1000_upper = df_dalys_upper.div(df_pop_upper.iloc[0, 0], axis=0) * 1000

        all_draws_dalys_mean.append(pd.Series(df_dalys_mean.sum(axis=1), name=f"Draw {draw}"))
        all_draws_dalys_lower.append(pd.Series(df_dalys_lower.sum(axis=1), name=f"Draw {draw}"))
        all_draws_dalys_upper.append(pd.Series(df_dalys_upper.sum(axis=1), name=f"Draw {draw}"))
        all_draws_dalys_mean_1000.append(pd.Series(df_per1000_mean.mean(axis=1), name=f"Draw {draw}"))
        all_draws_dalys_lower_1000.append(pd.Series(df_per1000_lower.mean(axis=1), name=f"Draw {draw}"))
        all_draws_dalys_upper_1000.append(pd.Series(df_per1000_upper.mean(axis=1), name=f"Draw {draw}"))

    df_dalys_all_draws_mean = pd.concat(all_draws_dalys_mean, axis=1)
    df_dalys_all_draws_lower = pd.concat(all_draws_dalys_lower, axis=1)
    df_dalys_all_draws_upper = pd.concat(all_draws_dalys_upper, axis=1)
    df_dalys_all_draws_mean_1000 = pd.concat(all_draws_dalys_mean_1000, axis=1)
    df_dalys_all_draws_lower_1000 = pd.concat(all_draws_dalys_lower_1000, axis=1)
    df_dalys_all_draws_upper_1000 = pd.concat(all_draws_dalys_upper_1000, axis=1)

    col_map = {f"Draw {draw}": scenario_names[draw] for draw in scenarios_of_interest}
    for df in [df_dalys_all_draws_mean, df_dalys_all_draws_lower, df_dalys_all_draws_upper,
               df_dalys_all_draws_mean_1000, df_dalys_all_draws_lower_1000, df_dalys_all_draws_upper_1000]:
        df.rename(columns=col_map, inplace=True)

    df_dalys_all_draws_mean.to_csv(output_folder / f"dalys_by_cause_all_draws_{suffix}.csv")
    df_dalys_all_draws_mean_1000.to_csv(output_folder / f"dalys_per_1000_by_cause_all_draws_{suffix}.csv")

    named_scenarios = [scenario_names[d] for d in scenarios_of_interest]
    n_scenarios = len(named_scenarios)
    causes = df_dalys_all_draws_mean_1000.index.tolist()
    n_causes = len(causes)
    y_base = np.arange(n_causes)
    ref_scen = named_scenarios[0]
    non_ref = [s for s in named_scenarios if s != ref_scen]

    scenario_colours = (
        [MAIN_TEXT_COLOURS.get(s, "#888888") for s in named_scenarios] if main_text
        else [SCENARIO_COLOURS_PALETTE[i % len(SCENARIO_COLOURS_PALETTE)] for i in range(n_scenarios)]
    )

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT A: DALYs per 1,000 by cause — grouped horizontal bar
    # ─────────────────────────────────────────────────────────────────────────────

    if main_text or climate_sensitivity_analysis:
        bar_height = 0.7 / n_scenarios
        fig_a, ax_a = plt.subplots(figsize=(14, max(9, n_causes * 0.6 + 2)))

        for i, scen in enumerate(named_scenarios):
            means = df_dalys_all_draws_mean_1000[scen].values
            lowers = df_dalys_all_draws_lower_1000[scen].values
            uppers = df_dalys_all_draws_upper_1000[scen].values
            xerr = np.array([np.clip(means - lowers, 0, None),
                             np.clip(uppers - means, 0, None)])
            offset = (i - (n_scenarios - 1) / 2) * bar_height

            ax_a.barh(y_base + offset, means, height=bar_height,
                      color=scenario_colours[i], alpha=0.85, label=scen)
            ax_a.errorbar(means, y_base + offset, xerr=xerr,
                          fmt="none", color="black", lw=1.0, capsize=3, alpha=0.6)

        ax_a.set_yticks(y_base)
        ax_a.set_yticklabels([c.replace("*", "") for c in causes])
        ax_a.invert_yaxis()
        ax_a.set_xlabel("DALYs per 1,000 population")
        ax_a.set_title(f"DALYs per 1,000 by cause ({min_year}–{max_year - 1})")
        ax_a.legend(title="Scenario", framealpha=0.9)
        ax_a.grid(axis="x", alpha=0.3)
        ax_a.set_xlim(left=0)
        fig_a.tight_layout()
        fig_a.savefig(output_folder / f"dalys_per_1000_grouped_by_scenario_{suffix}.png",
                      dpi=300, bbox_inches="tight")
        plt.close(fig_a)

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT B: Excess DALYs per 1,000 by cause — dot plot
    # ─────────────────────────────────────────────────────────────────────────────

    if main_text or climate_sensitivity_analysis:
        offsets_diff = {s: (i - (len(non_ref) - 1) / 2) * 0.25
                        for i, s in enumerate(non_ref)}

        fig_b, ax_b = plt.subplots(figsize=(13, max(7, n_causes * 0.55 + 2)))

        for scen in non_ref:
            colour = (MAIN_TEXT_COLOURS.get(scen, "#888888") if main_text
                      else SCENARIO_COLOURS_PALETTE[named_scenarios.index(scen) % len(SCENARIO_COLOURS_PALETTE)])
            offset = offsets_diff[scen]
            ref_means = df_dalys_all_draws_mean_1000[ref_scen]
            cmp_means = df_dalys_all_draws_mean_1000[scen]
            cmp_lowers = df_dalys_all_draws_lower_1000[scen]
            cmp_uppers = df_dalys_all_draws_upper_1000[scen]
            diffs = cmp_means - ref_means
            xerr = np.array([np.clip(cmp_means.values - cmp_lowers.values, 0, None),
                             np.clip(cmp_uppers.values - cmp_means.values, 0, None)])

            diff_lower = diffs - xerr[0]
            diff_upper = diffs + xerr[1]
            significant = ~((diff_lower <= 0) & (diff_upper >= 0))

            ax_b.errorbar(diffs.values, y_base + offset, xerr=xerr,
                          fmt="none", color=colour, capsize=3, linewidth=1.4, alpha=0.7)
            ax_b.scatter(diffs.values[significant], (y_base + offset)[significant],
                         color=colour, s=50, zorder=3, label=f"{scen} (sig.)")
            ax_b.scatter(diffs.values[~significant], (y_base + offset)[~significant],
                         facecolors="none", edgecolors=colour, s=50, zorder=3,
                         label=f"{scen} (n.s.)")

        ax_b.axvline(0, color="black", linewidth=1.2, linestyle="--", label=ref_scen)
        ax_b.set_yticks(y_base)
        ax_b.set_yticklabels([c.replace("*", "") for c in causes])
        ax_b.invert_yaxis()
        ax_b.set_xlabel(f"Excess DALYs per 1,000 vs {ref_scen}")
        ax_b.set_title(f"Excess DALYs per 1,000 by cause ({min_year}–{max_year - 1})")
        ax_b.legend(title="Scenario", framealpha=0.9)
        ax_b.grid(axis="x", alpha=0.3)
        fig_b.tight_layout()
        fig_b.savefig(output_folder / f"{PREFIX_ON_FILENAME}_DALYs_DiffPlot_{suffix}.png",
                      dpi=300, bbox_inches="tight")
        plt.close(fig_b)

    # ─────────────────────────────────────────────────────────────────────────────
    #  DISTRICT EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────────

    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year - 1, 12, 31))

    dist_mean, dist_lower, dist_upper = {}, {}, {}

    for draw in scenarios_of_interest:
        scen = scenario_names[draw]
        print(f"Extracting district DALYs for draw {draw} ({scen})...")
        s = summarize(
            extract_results(results_folder,
                            module="tlo.methods.healthburden",
                            key="dalys_by_district_stacked_by_age_and_time",
                            custom_generate_series=get_dalys_by_district,
                            do_scaling=False),
            only_mean=True, collapse_columns=True,
        )[draw]
        dist_mean[scen] = s["mean"]
        dist_lower[scen] = s["lower"]
        dist_upper[scen] = s["upper"]

    all_districts = _sorted_districts(list({d for s in dist_mean.values() for d in s.index}))
    df_dist_mean = pd.DataFrame(dist_mean).reindex(all_districts, fill_value=0)
    df_dist_lower = pd.DataFrame(dist_lower).reindex(all_districts, fill_value=0)
    df_dist_upper = pd.DataFrame(dist_upper).reindex(all_districts, fill_value=0)
    df_dist_mean.to_csv(output_folder / f"dalys_by_district_{suffix}.csv")

    n_districts = len(all_districts)
    y_dist = np.arange(n_districts)
    bar_height_d = 0.7 / n_scenarios

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT C: Total DALYs by district — grouped horizontal bar
    # ─────────────────────────────────────────────────────────────────────────────

    if main_text or climate_sensitivity_analysis:
        fig_c, ax_c = plt.subplots(figsize=(14, max(11, n_districts * 0.5 + 2)))

        for i, scen in enumerate(named_scenarios):
            means = df_dist_mean[scen].values
            lowers = df_dist_lower[scen].values
            uppers = df_dist_upper[scen].values
            xerr = np.array([np.clip(means - lowers, 0, None),
                             np.clip(uppers - means, 0, None)])
            offset = (i - (n_scenarios - 1) / 2) * bar_height_d

            ax_c.barh(y_dist + offset, means, height=bar_height_d,
                      color=scenario_colours[i], alpha=0.85, label=scen)
            ax_c.errorbar(means, y_dist + offset, xerr=xerr,
                          fmt="none", color="black", lw=1.0, capsize=3, alpha=0.6)

        ax_c.set_yticks(y_dist)
        tick_labels_c = ax_c.set_yticklabels(all_districts)
        for lbl, d in zip(tick_labels_c, all_districts):
            lbl.set_color(REGION_COLOURS.get(DISTRICT_REGION.get(d, ""), "black"))
        ax_c.invert_yaxis()
        ax_c.set_xlabel("Total DALYs")
        ax_c.set_title(f"Total DALYs by district ({min_year}–{max_year - 1})")
        ax_c.legend(title="Scenario", framealpha=0.9)
        ax_c.grid(axis="x", alpha=0.3)
        ax_c.set_xlim(left=0)
        fig_c.tight_layout()
        fig_c.savefig(output_folder / f"{PREFIX_ON_FILENAME}_DALYs_by_district_{suffix}.png",
                      dpi=300, bbox_inches="tight")
        plt.close(fig_c)

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT D: Excess DALYs by district vs reference — dot plot
    # ─────────────────────────────────────────────────────────────────────────────

    if main_text or climate_sensitivity_analysis:
        offsets_dist_diff = {s: (i - (len(non_ref) - 1) / 2) * 0.25
                             for i, s in enumerate(non_ref)}

        fig_d, ax_d = plt.subplots(figsize=(13, max(9, n_districts * 0.5 + 2)))

        for scen in non_ref:
            colour = (MAIN_TEXT_COLOURS.get(scen, "#888888") if main_text
                      else SCENARIO_COLOURS_PALETTE[named_scenarios.index(scen) % len(SCENARIO_COLOURS_PALETTE)])
            diffs = df_dist_mean[scen] - df_dist_mean[ref_scen]
            xerr = np.array([np.clip(df_dist_mean[scen].values - df_dist_lower[scen].values, 0, None),
                             np.clip(df_dist_upper[scen].values - df_dist_mean[scen].values, 0, None)])
            offset = offsets_dist_diff[scen]

            diff_lower = diffs.values - xerr[0]
            diff_upper = diffs.values + xerr[1]
            significant = ~((diff_lower <= 0) & (diff_upper >= 0))

            ax_d.errorbar(diffs.values, y_dist + offset, xerr=xerr,
                          fmt="none", color=colour, capsize=3, linewidth=1.4, alpha=0.7)
            ax_d.scatter(diffs.values[significant], (y_dist + offset)[significant],
                         color=colour, s=50, zorder=3, label=f"{scen} (sig.)")
            ax_d.scatter(diffs.values[~significant], (y_dist + offset)[~significant],
                         facecolors="none", edgecolors=colour, s=50, zorder=3,
                         label=f"{scen} (n.s.)")

        ax_d.axvline(0, color="black", linewidth=1.2, linestyle="--", label=ref_scen)
        ax_d.set_yticks(y_dist)
        tick_labels_d = ax_d.set_yticklabels(all_districts)
        for lbl, d in zip(tick_labels_d, all_districts):
            lbl.set_color(REGION_COLOURS.get(DISTRICT_REGION.get(d, ""), "black"))
        ax_d.invert_yaxis()
        ax_d.set_xlabel(f"Excess DALYs vs {ref_scen}")
        ax_d.set_title(f"Excess DALYs by district ({min_year}–{max_year - 1})")
        ax_d.legend(title="Scenario", framealpha=0.9)
        ax_d.grid(axis="x", alpha=0.3)
        fig_d.tight_layout()
        fig_d.savefig(output_folder / f"{PREFIX_ON_FILENAME}_DALYs_DiffPlot_district_{suffix}.png",
                      dpi=300, bbox_inches="tight")
        plt.close(fig_d)

    # ─────────────────────────────────────────────────────────────────────────────
    #  CAUSE × DISTRICT EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────────

    cd_mean, cd_lower, cd_upper = {}, {}, {}

    for draw in scenarios_of_interest:
        scen = scenario_names[draw]
        print(f"Extracting cause × district DALYs for draw {draw} ({scen})...")
        s = summarize(
            extract_results(results_folder,
                            module="tlo.methods.healthburden",
                            key="dalys_by_district_stacked_by_age_and_time",
                            custom_generate_series=get_dalys_by_cause_and_district,
                            do_scaling=False),
            only_mean=True, collapse_columns=True,
        )[draw]
        cd_mean[scen] = s["mean"]
        cd_lower[scen] = s["lower"]
        cd_upper[scen] = s["upper"]

    all_causes_cd = sorted({idx[1] for s in cd_mean.values() for idx in s.index})

    def _unstack_cd(series_dict):
        return {
            scen: s.unstack(level=1).reindex(index=all_districts,
                                             columns=all_causes_cd, fill_value=0.0)
            for scen, s in series_dict.items()
        }

    cd_mean_df = _unstack_cd(cd_mean)
    cd_lower_df = _unstack_cd(cd_lower)
    cd_upper_df = _unstack_cd(cd_upper)

    cd_mean_df[ref_scen].to_csv(
        output_folder / f"dalys_by_cause_and_district_{ref_scen.replace(' ', '_')}_{suffix}.csv"
    )

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT E: DALYs by cause × district — heatmap per scenario
    # ─────────────────────────────────────────────────────────────────────────────

    if main_text or climate_sensitivity_analysis:
        n_causes_cd = len(all_causes_cd)
        mat_ref = cd_mean_df[ref_scen].values
        vmax_cd = max(cd_mean_df[s].values.max() for s in named_scenarios)

        for scen in named_scenarios:
            mat = cd_mean_df[scen].values
            fig_e, ax_e = plt.subplots(
                figsize=(max(11, n_causes_cd * 1.0 + 2), max(9, n_districts * 0.4 + 2))
            )
            im_e = ax_e.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax_cd)
            cbar_e = fig_e.colorbar(im_e, ax=ax_e, fraction=0.02, pad=0.02)
            cbar_e.set_label("Total DALYs")

            ax_e.set_xticks(range(n_causes_cd))
            ax_e.set_xticklabels(all_causes_cd, fontsize=9, rotation=40, ha="right")
            ax_e.set_yticks(range(n_districts))
            tick_labels_e = ax_e.set_yticklabels(all_districts, fontsize=10)
            for lbl, d in zip(tick_labels_e, all_districts):
                lbl.set_color(REGION_COLOURS.get(DISTRICT_REGION.get(d, ""), "black"))

            ax_e.set_title(f"DALYs by cause × district — {scen} ({min_year}–{max_year - 1})")
            fig_e.tight_layout()
            fig_e.savefig(
                output_folder / f"{PREFIX_ON_FILENAME}_DALYs_cause_district_{scen.replace(' ', '_')}_{suffix}.png",
                dpi=300, bbox_inches="tight")
            plt.close(fig_e)

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT F: Excess DALYs by cause × district — heatmap per non-ref scenario
    # ─────────────────────────────────────────────────────────────────────────────

    if main_text or climate_sensitivity_analysis:
        from matplotlib.colors import TwoSlopeNorm
        from matplotlib.patches import Rectangle

        for scen in non_ref:
            excess = cd_mean_df[scen].values - mat_ref
            excess_lower = cd_lower_df[scen].values - cd_upper_df[ref_scen].values
            excess_upper = cd_upper_df[scen].values - cd_lower_df[ref_scen].values
            significant = (excess_lower > 0) | (excess_upper < 0)

            abs_max = max(np.abs(excess).max(), 1)
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

            fig_f, ax_f = plt.subplots(
                figsize=(max(11, n_causes_cd * 1.0 + 2), max(9, n_districts * 0.4 + 2))
            )
            im_f = ax_f.imshow(excess, aspect="auto", cmap="RdBu_r", norm=norm)
            cbar_f = fig_f.colorbar(im_f, ax=ax_f, fraction=0.02, pad=0.02)
            cbar_f.set_label(f"Difference in DALYs vs {ref_scen}")

            ax_f.set_xticks(range(n_causes_cd))
            ax_f.set_xticklabels(all_causes_cd, fontsize=9, rotation=40, ha="right")
            ax_f.set_yticks(range(n_districts))
            tick_labels_f = ax_f.set_yticklabels(all_districts, fontsize=10)
            for lbl, d in zip(tick_labels_f, all_districts):
                lbl.set_color(REGION_COLOURS.get(DISTRICT_REGION.get(d, ""), "black"))

            for ri in range(n_districts):
                for ci in range(n_causes_cd):
                    val = excess[ri, ci]
                    if not significant[ri, ci]:
                        ax_f.add_patch(Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                                 facecolor="white", alpha=0.6, zorder=2))
                    else:
                        ax_f.add_patch(Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                                 facecolor="none", edgecolor="black",
                                                 linewidth=1.8, zorder=4))
                    text_colour = "white" if abs(val) > 0.6 * abs_max and significant[ri, ci] else "black"
                    ax_f.text(ci, ri, f"{val:+.0f}", ha="center", va="center",
                              fontsize=7, color=text_colour, zorder=3)

            ax_f.set_title(
                f"Difference in DALYs — {scen} vs {ref_scen} ({min_year}–{max_year - 1})\n")
            fig_f.tight_layout()
            fig_f.savefig(
                output_folder / f"{PREFIX_ON_FILENAME}_DALYs_cause_district_excess_{scen.replace(' ', '_')}_{suffix}.png",
                dpi=300, bbox_inches="tight")
            plt.close(fig_f)

    # ─────────────────────────────────────────────────────────────────────────────
    #  MONTHLY YLD EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────────

    mo_mean, mo_lower, mo_upper = {}, {}, {}

    for draw in scenarios_of_interest:
        scen = scenario_names[draw]
        print(f"Extracting monthly YLD by district for draw {draw} ({scen})...")
        s = summarize(
            extract_results(results_folder,
                            module="tlo.methods.healthburden",
                            key="monthly_dalys_by_district",
                            custom_generate_series=get_monthly_yld_by_district,
                            do_scaling=False),
            only_mean=True, collapse_columns=True,
        )[draw]
        mo_mean[scen] = s["mean"]
        mo_lower[scen] = s["lower"]
        mo_upper[scen] = s["upper"]

    def _pivot_monthly(s):
        df = s.reset_index()
        df.columns = ["year", "month", "district", "yld"]
        df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
        return df.pivot_table(index="date", columns="district", values="yld", aggfunc="sum")

    mo_mean_df = {scen: _pivot_monthly(s) for scen, s in mo_mean.items()}
    mo_lower_df = {scen: _pivot_monthly(s) for scen, s in mo_lower.items()}
    mo_upper_df = {scen: _pivot_monthly(s) for scen, s in mo_upper.items()}

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT G: Excess monthly YLD by district — line plot, panels by region
    # ─────────────────────────────────────────────────────────────────────────────

    if main_text or climate_sensitivity_analysis:
        import matplotlib.dates as mdates

        for scen in non_ref:
            ref_df = mo_mean_df[ref_scen]
            cmp_df = mo_mean_df[scen]

            common_dates = ref_df.index.intersection(cmp_df.index)
            common_districts = [d for d in all_districts
                                if d in ref_df.columns and d in cmp_df.columns]

            excess_df = cmp_df.loc[common_dates, common_districts] - ref_df.loc[common_dates, common_districts]
            excess_lower_df = mo_lower_df[scen].loc[common_dates, common_districts] - mo_upper_df[ref_scen].loc[
                common_dates, common_districts]
            excess_upper_df = mo_upper_df[scen].loc[common_dates, common_districts] - mo_lower_df[ref_scen].loc[
                common_dates, common_districts]

            districts_by_region = {
                region: [d for d in common_districts if DISTRICT_REGION.get(d) == region]
                for region in REGION_ORDER
            }

            fig_g, axes_g = plt.subplots(3, 1, figsize=(15, 14), sharex=True, sharey=False)

            for ax_g, region in zip(axes_g, REGION_ORDER):
                region_districts = districts_by_region[region]
                cmap_r = plt.cm.get_cmap(
                    "Purples" if region == "North" else
                    "Blues" if region == "Centre" else "Reds"
                )
                colours_r = [cmap_r(0.4 + 0.5 * i / max(len(region_districts) - 1, 1))
                             for i in range(len(region_districts))]

                for district, col in zip(region_districts, colours_r):
                    if district not in excess_df.columns:
                        continue
                    y = excess_df[district]
                    y_lo = excess_lower_df[district]
                    y_hi = excess_upper_df[district]

                    ax_g.plot(common_dates, y, color=col, linewidth=1.6, label=district)
                    ax_g.fill_between(common_dates, y_lo, y_hi, color=col, alpha=0.15)

                    sig_mask = (y_lo > 0) | (y_hi < 0)
                    if sig_mask.any():
                        ax_g.scatter(common_dates[sig_mask], y[sig_mask],
                                     marker="*", color=col, s=70, zorder=5)

                ax_g.axhline(0, color="black", linewidth=1.0, linestyle="--")
                ax_g.set_ylabel("Excess monthly YLD")
                ax_g.set_title(region, color=REGION_COLOURS[region])
                ax_g.legend(fontsize=9, ncol=2, framealpha=0.7, loc="upper left")
                ax_g.grid(axis="y", alpha=0.3)

            axes_g[-1].set_xlabel("Date")
            axes_g[-1].xaxis.set_major_locator(mdates.YearLocator())
            axes_g[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.setp(axes_g[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

            fig_g.suptitle(
                f"Excess monthly YLD — {scen} vs {ref_scen}  |  ★ = sig.  |  YLD only",
                fontsize=15, fontweight="bold",
            )
            fig_g.tight_layout()
            fig_g.savefig(
                output_folder / f"{PREFIX_ON_FILENAME}_monthly_YLD_excess_{scen.replace(' ', '_')}_{suffix}.png",
                dpi=300, bbox_inches="tight")
            plt.close(fig_g)

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOTS 1–5: Distribution / uncertainty (parameter SA only)
    # ─────────────────────────────────────────────────────────────────────────────

    if parameter_sensitivity_analysis:

        # 1. Total DALYs — box plot by cause
        fig, ax = plt.subplots(figsize=(16, 10))
        box_data, box_labels, box_colors = [], [], []
        for condition in df_dalys_all_draws_mean.index:
            box_data.append(df_dalys_all_draws_mean.loc[condition].values)
            box_labels.append(condition)
            box_colors.append(get_color_cause_of_death_or_daly_label(condition))
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=True)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color);
            patch.set_alpha(0.7)
        ax.set_title(f"Total DALYs by cause ({min_year}–{max_year})")
        ax.set_ylabel("Total DALYs");
        ax.set_xlabel("Cause")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3);
        fig.tight_layout()
        fig.savefig(output_folder / f"total_dalys_distribution_by_cause_{suffix}.png", dpi=300)
        plt.close(fig)

        # 2. DALYs per 1,000 — box plot by cause
        fig, ax = plt.subplots(figsize=(16, 10))
        box_data, box_labels, box_colors = [], [], []
        for condition in df_dalys_all_draws_mean_1000.index:
            box_data.append(df_dalys_all_draws_mean_1000.loc[condition].values)
            box_labels.append(condition)
            box_colors.append(get_color_cause_of_death_or_daly_label(condition))
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=True)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color);
            patch.set_alpha(0.7)
        ax.set_title(f"DALYs per 1,000 by cause ({min_year}–{max_year})")
        ax.set_ylabel("DALYs per 1,000");
        ax.set_xlabel("Cause")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3);
        fig.tight_layout()
        fig.savefig(output_folder / f"dalys_per_1000_distribution_by_cause_{suffix}.png", dpi=300)
        plt.close(fig)

        # 3. Total DALYs — histogram + box
        total_dalys_all_draws = df_dalys_all_draws_mean.sum(axis=0)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.hist(total_dalys_all_draws.values, bins=max(5, n_scenarios // 2),
                 color="steelblue", alpha=0.7, edgecolor="black")
        ax1.axvline(total_dalys_all_draws.mean(), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {total_dalys_all_draws.mean():.0f}")
        ax1.axvline(total_dalys_all_draws.median(), color="orange", linestyle="--", linewidth=2,
                    label=f"Median: {total_dalys_all_draws.median():.0f}")
        ax1.set_title("Total DALYs distribution");
        ax1.set_xlabel("Total DALYs");
        ax1.set_ylabel("Frequency")
        ax1.legend();
        ax1.grid(axis="y", alpha=0.3)
        ax2.boxplot([total_dalys_all_draws.values], labels=["All draws"], patch_artist=True)
        ax2.set_title("Total DALYs summary");
        ax2.set_ylabel("Total DALYs")
        ax2.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_folder / f"total_dalys_overall_distribution_{suffix}.png", dpi=300)
        plt.close(fig)

        # 4. Mean DALYs per 1,000 by cause
        mean_dalys_by_cause = df_dalys_all_draws_mean_1000.mean(axis=1).sort_values(ascending=False)
        colors_sorted = [get_color_cause_of_death_or_daly_label(lbl) for lbl in mean_dalys_by_cause.index]
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.barh(range(len(mean_dalys_by_cause)), mean_dalys_by_cause.values, color=colors_sorted)
        ax.set_yticks(range(len(mean_dalys_by_cause)))
        ax.set_yticklabels(mean_dalys_by_cause.index)
        ax.set_xlabel("Mean DALYs per 1,000")
        ax.set_title("Mean DALYs per 1,000 by cause")
        ax.grid(axis="x", alpha=0.3);
        fig.tight_layout()
        fig.savefig(output_folder / f"mean_dalys_per_1000_by_cause_{suffix}.png", dpi=300)
        plt.close(fig)

        # 5. Coefficient of variation
        cv_by_cause = (df_dalys_all_draws_mean_1000.std(axis=1) /
                       df_dalys_all_draws_mean_1000.mean(axis=1)) * 100
        cv_by_cause = cv_by_cause.sort_values(ascending=False)
        colors_cv = [get_color_cause_of_death_or_daly_label(lbl) for lbl in cv_by_cause.index]
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.barh(range(len(cv_by_cause)), cv_by_cause.values, color=colors_cv, alpha=0.7)
        ax.set_yticks(range(len(cv_by_cause)))
        ax.set_yticklabels(cv_by_cause.index)
        ax.set_xlabel("Coefficient of Variation (%)")
        ax.set_title("Uncertainty by cause (CV)")
        ax.grid(axis="x", alpha=0.3);
        fig.tight_layout()
        fig.savefig(output_folder / f"cv_by_cause_{suffix}.png", dpi=300)
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────────
    #  CSV OUTPUT
    # ─────────────────────────────────────────────────────────────────────────────

    for label, df in [("dalys", df_dalys_all_draws_mean),
                      ("dalys_per_1000", df_dalys_all_draws_mean_1000)]:
        df.to_csv(output_folder / f"summary_statistics_{label}_{suffix}.csv")

    # ─────────────────────────────────────────────────────────────────────────────
    #  CONSOLE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────────

    print(f"\nSummary figures saved to {output_folder}")
    for scen in named_scenarios:
        print(f"  Total DALYs {scen}: {df_dalys_all_draws_mean[scen].sum():,.0f}")
    print(f"\nTop 5 districts by DALYs ({ref_scen}):")
    print(df_dist_mean[ref_scen].sort_values(ascending=False).head(5).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()
    apply(results_folder=args.results_folder,
          output_folder=args.results_folder,
          resourcefilepath=Path("./resources"))
