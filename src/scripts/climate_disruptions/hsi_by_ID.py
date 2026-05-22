"""
Standalone script — Combined HSI volume + disruption figure.

Layout: 1 row × 2 panels
  Panel A (left):  Top N HSI types by total volume, grouped bars per scenario
  Panel B (right): Top N HSI types by disruption rate, chosen scenario

CSV outputs:
  hsi_volume_by_type_{suffix}.csv
  hsi_disruption_by_type_{suffix}.csv

Usage:
    python plot_hsi_volume_disruption_combined.py <results_folder>
    python plot_hsi_volume_disruption_combined.py <results_folder> \\
        --output_folder <out> --resources <res>
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date
from tlo.analysis.utils import extract_results

# ─────────────────────────────────────────────────────────────────────────────
#  STYLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FS_TICK = 13
FS_LABEL = 15
FS_TITLE = 16
FS_LEGEND = 12
FS_PANEL = 17
FS_SUPTITLE = 14

COLOUR_DELAYED = "#E67E22"
COLOUR_CANCELLED = "#D4AC0D"

# One colour per scenario (matches order of scenarios_of_interest)
SCENARIO_COLOURS = [
    "#5B8DB8",  # No Disruptions  — steel blue
    "#E8C882",  # Default         — gold/tan
    "#E8968A",  # Worst Case      — salmon
    "#82C882",  # extras for climate / supply-demand modes
    "#C882C8",
    "#82C8C8",
    "#C8A882",
    "#8282C8",
]

MAX_CHARS = 15


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ym(index):
    return index.astype(str).str.split(":", n=1).str[0]


def _parse_hsi_type(index):
    return index.astype(str).str.split(":", n=2).str[2]


def _wet_season_mask(df):
    months = _parse_ym(df.index).str.split("-").str[1].astype(int)
    return months.isin([11, 12, 1, 2, 3, 4])


def _hsi_volume_by_type(total_df, CI_LOWER, CI_UPPER):
    """Sum HSIs by type across all months/facilities; return mean and CI across runs."""
    hsi = _parse_hsi_type(total_df.index)
    by_type = total_df.groupby(hsi).sum()
    by_type = by_type[by_type.index.astype(str) != "nan"]
    return (
        by_type.mean(axis=1),
        by_type.quantile(CI_LOWER, axis=1),
        by_type.quantile(CI_UPPER, axis=1),
    )


def _hsi_type_stats(total_df, delayed_df, cancelled_df, CI_LOWER, CI_UPPER):
    hsi = _parse_hsi_type(total_df.index)
    total_by_type = total_df.groupby(hsi).sum()
    delayed_by_type = delayed_df.reindex(total_df.index, fill_value=0).groupby(hsi).sum()
    cancelled_by_type = cancelled_df.reindex(total_df.index, fill_value=0).groupby(hsi).sum()
    denom = total_by_type + delayed_by_type + cancelled_by_type
    delayed_rate = delayed_by_type.div(denom).where(denom > 0, 0.0)
    cancelled_rate = cancelled_by_type.div(denom).where(denom > 0, 0.0)
    return (
        delayed_rate.mean(axis=1), delayed_rate.quantile(CI_LOWER, axis=1),
        delayed_rate.quantile(CI_UPPER, axis=1),
        cancelled_rate.mean(axis=1), cancelled_rate.quantile(CI_LOWER, axis=1),
        cancelled_rate.quantile(CI_UPPER, axis=1),
        total_by_type.mean(axis=1),
    )


def _wrap_labels(labels, max_chars=MAX_CHARS):
    return [textwrap.fill(str(lbl), width=max_chars) for lbl in labels]


def _make_hsi_counts_by_real_facility_monthly(target_period):
    def _fn(_df):
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*target_period)]
        if _df.empty or "counts" not in _df.columns:
            return pd.Series(dtype=float)
        totals = {}
        for _, row in _df.iterrows():
            ym = row["date"].strftime("%Y-%m")
            counts_dict = row["counts"] if isinstance(row["counts"], dict) else {}
            for key, val in counts_dict.items():
                parts = str(key).split(":", 1)
                real_fac = parts[0]
                hsi_type = parts[1] if len(parts) > 1 else "unknown"
                composite = f"{ym}:{real_fac}:{hsi_type}"
                totals[composite] = totals.get(composite, 0) + val
        return pd.Series(totals, dtype=float)

    return _fn


def _make_disrupted_by_real_facility_monthly(target_period):
    def _fn(_df):
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*target_period)]
        if _df.empty or "RealFacility_ID" not in _df.columns:
            return pd.Series(dtype=float)
        _df = _df[_df["RealFacility_ID"].notna() & (_df["RealFacility_ID"] != "unknown")].copy()
        _df["hsi_type"] = (
            _df["TREATMENT_ID"].fillna("unknown").astype(str)
            if "TREATMENT_ID" in _df.columns else "unknown"
        )
        _df["composite"] = (
            _df["date"].dt.strftime("%Y-%m") + ":"
            + _df["RealFacility_ID"].astype(str) + ":"
            + _df["hsi_type"]
        )
        return _df["composite"].value_counts().astype(float)

    return _fn


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path):
    # ── configuration ────────────────────────────────────────────────────────
    min_year = 2025
    max_year = 2041
    spacing_of_years = 1

    main_text = True
    parameter_uncertainty_analysis = False
    mode_2 = False
    climate_analysis = False
    prop_supply_demand = False
    wet_season = True

    top_n_volume = 10  # HSI types shown in Panel A
    top_n_disruption = 10  # HSI types shown in Panel B

    # Which scenario to display in Panel B — must match a scenario_names entry.
    # If it doesn't match, the script falls back to the first non-"No Disruptions" scenario.
    disruption_panel_scenario = "Default"

    SCALING_FACTOR = 145.39
    CI_LOWER = 0.025
    CI_UPPER = 0.975

    if parameter_uncertainty_analysis:
        scenario_names = list(range(200))
        scenarios_of_interest = scenario_names
        suffix = "parameter_UA_mode_2" if mode_2 else "parameter_UA_mode_1"
    if main_text:
        scenario_names = ["No Disruptions", "Default", "Worst Case"]
        scenarios_of_interest = [0, 1, 2]
        suffix = "main_text_mode_2" if mode_2 else "main_text_mode_1"
    if climate_analysis:
        scenario_names = [
            "SSP126 Low Baseline",
            "SSP126 Low Worst",
            "SSP585 Low Baseline",
            "SSP585 Low Worst",
            "SSP585 High Baseline",
            "SSP585 High Worst",
            "SSP126 High Baseline",
            "SSP126 High Worst",
        ]
        scenarios_of_interest = list(range(8))
        suffix = "climate_scenarios"
    if prop_supply_demand:
        scenario_names = [
            "Default Supply 0.1",
            "Default Supply 0.5",
            "Default Supply 0.9",
            "Worst Case Supply 0.1",
            "Worst Case Supply 0.5",
            "Worst Case Supply 0.9",
        ]
        scenarios_of_interest = list(range(6))
        suffix = "prop_supply_demand"
        disruption_panel_scenario = "Default Supply 0.5"
    if wet_season:
        suffix += "_wet_season"

    period_label = "wet season: Nov–Apr" if wet_season else f"{min_year}–{max_year - 1}"

    target_year_sequence = range(min_year, max_year, spacing_of_years)

    # ── pre-load raw results ─────────────────────────────────────────────────
    print("Loading raw results …")
    raw_total = {}
    raw_delayed = {}
    raw_cancelled = {}

    for yr in target_year_sequence:
        print(f"  {yr}")
        period = (Date(yr, 1, 1), Date(yr, 12, 31))
        raw_total[yr] = extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="hsi_event_counts_by_facility_monthly",
            custom_generate_series=_make_hsi_counts_by_real_facility_monthly(period),
            do_scaling=False,
        )
        raw_delayed[yr] = extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="Weather_delayed_HSI_Event_full_info",
            custom_generate_series=_make_disrupted_by_real_facility_monthly(period),
            do_scaling=False,
        )
        raw_cancelled[yr] = extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="Weather_cancelled_HSI_Event_full_info",
            custom_generate_series=_make_disrupted_by_real_facility_monthly(period),
            do_scaling=False,
        )

    def _concat_years(dfs):
        return pd.concat(dfs).groupby(level=0).sum()

    # ── per-draw processing ──────────────────────────────────────────────────
    print("Processing draws …")

    all_draws_volume_mean = []
    all_draws_volume_lower = []
    all_draws_volume_upper = []

    all_draws_hsi_delayed_mean = []
    all_draws_hsi_delayed_lower = []
    all_draws_hsi_delayed_upper = []
    all_draws_hsi_cancelled_mean = []
    all_draws_hsi_cancelled_lower = []
    all_draws_hsi_cancelled_upper = []
    all_draws_hsi_total = []

    for draw in scenarios_of_interest:
        scen = scenario_names[draw]
        print(f"  draw {draw} ({scen})")

        total_all = _concat_years(
            [raw_total[yr][draw].fillna(0) for yr in target_year_sequence]
        ) * SCALING_FACTOR
        valid_fac = total_all.index.astype(str).str.split(":", n=2).str[1] != "nan"
        total_all = total_all[valid_fac]
        # wet season filter applied before everything else
        if wet_season:
            total_all = total_all[_wet_season_mask(total_all)]

        vm, vl, vu = _hsi_volume_by_type(total_all, CI_LOWER, CI_UPPER)
        all_draws_volume_mean.append(vm)
        all_draws_volume_lower.append(vl)
        all_draws_volume_upper.append(vu)

        if scen == "No Disruptions":
            for lst in [
                all_draws_hsi_delayed_mean, all_draws_hsi_delayed_lower,
                all_draws_hsi_delayed_upper, all_draws_hsi_cancelled_mean,
                all_draws_hsi_cancelled_lower, all_draws_hsi_cancelled_upper,
                all_draws_hsi_total,
            ]:
                lst.append(pd.Series(dtype=float))
            continue

        delayed_all = _concat_years(
            [raw_delayed[yr][draw].fillna(0) for yr in target_year_sequence]
        ) * SCALING_FACTOR
        cancelled_all = _concat_years(
            [raw_cancelled[yr][draw].fillna(0) for yr in target_year_sequence]
        ) * SCALING_FACTOR

        # reindex to match (possibly wet-season-filtered) total_all
        delayed_all = delayed_all.reindex(total_all.index, fill_value=0)
        cancelled_all = cancelled_all.reindex(total_all.index, fill_value=0)

        hd_m, hd_l, hd_u, hc_m, hc_l, hc_u, htot = _hsi_type_stats(
            total_all, delayed_all, cancelled_all, CI_LOWER, CI_UPPER
        )
        all_draws_hsi_delayed_mean.append(hd_m)
        all_draws_hsi_delayed_lower.append(hd_l)
        all_draws_hsi_delayed_upper.append(hd_u)
        all_draws_hsi_cancelled_mean.append(hc_m)
        all_draws_hsi_cancelled_lower.append(hc_l)
        all_draws_hsi_cancelled_upper.append(hc_u)
        all_draws_hsi_total.append(htot)

    # ── select HSI types for each panel ──────────────────────────────────────

    # Panel A reference: No Disruptions volume (or first scenario if absent)
    ref_idx = next(
        (i for i, d in enumerate(scenarios_of_interest)
         if scenario_names[d] == "No Disruptions"),
        0,
    )
    ref_volume = all_draws_volume_mean[ref_idx]
    ref_volume = ref_volume[ref_volume.index.astype(str) != "nan"]
    # sorted descending; reversed so highest ends up at top of the horizontal chart
    top_volume_types = (
        ref_volume[ref_volume > 0]
        .sort_values(ascending=False)
        .head(top_n_volume)
        .index.tolist()
    )
    top_volume_types_plot = list(reversed(top_volume_types))  # lowest at y=0, highest at y=N-1

    # Panel B: find index of chosen scenario
    panel_b_idx = next(
        (i for i, d in enumerate(scenarios_of_interest)
         if scenario_names[d] == disruption_panel_scenario),
        None,
    )
    if panel_b_idx is None:
        panel_b_idx = next(
            (i for i, d in enumerate(scenarios_of_interest)
             if scenario_names[d] != "No Disruptions"),
            0,
        )
        disruption_panel_scenario = scenario_names[scenarios_of_interest[panel_b_idx]]
        print(f"  Panel B: '{disruption_panel_scenario}' not found — "
              f"falling back to '{disruption_panel_scenario}'")

    hd_m_b = all_draws_hsi_delayed_mean[panel_b_idx]
    hd_l_b = all_draws_hsi_delayed_lower[panel_b_idx]
    hd_u_b = all_draws_hsi_delayed_upper[panel_b_idx]
    hc_m_b = all_draws_hsi_cancelled_mean[panel_b_idx]
    hc_l_b = all_draws_hsi_cancelled_lower[panel_b_idx]
    hc_u_b = all_draws_hsi_cancelled_upper[panel_b_idx]

    total_rate_b = (hd_m_b + hc_m_b).copy()
    total_rate_b = total_rate_b[total_rate_b.index.astype(str) != "nan"]
    top_disrupted_types = (
        total_rate_b[total_rate_b > 0]
        .sort_values(ascending=False)
        .head(top_n_disruption)
        .index.tolist()
    )
    top_disrupted_types_plot = list(reversed(top_disrupted_types))

    # ── CSV outputs ───────────────────────────────────────────────────────────
    print("Writing CSVs …")

    vol_rows = []
    for scen_idx, draw in enumerate(scenarios_of_interest):
        scen = scenario_names[draw]
        vm = all_draws_volume_mean[scen_idx]
        vl = all_draws_volume_lower[scen_idx]
        vu = all_draws_volume_upper[scen_idx]
        for hsi in vm.index:
            vol_rows.append({
                "scenario": scen,
                "hsi_type": hsi,
                "volume_mean": round(vm.get(hsi, 0), 1),
                "volume_lower": round(vl.get(hsi, 0), 1),
                "volume_upper": round(vu.get(hsi, 0), 1),
            })
    pd.DataFrame(vol_rows).sort_values(
        ["scenario", "volume_mean"], ascending=[True, False]
    ).to_csv(output_folder / f"hsi_volume_by_type_{suffix}.csv", index=False)

    dis_rows = []
    for scen_idx, draw in enumerate(scenarios_of_interest):
        scen = scenario_names[draw]
        if scen == "No Disruptions":
            continue
        hd_m = all_draws_hsi_delayed_mean[scen_idx]
        hd_l = all_draws_hsi_delayed_lower[scen_idx]
        hd_u = all_draws_hsi_delayed_upper[scen_idx]
        hc_m = all_draws_hsi_cancelled_mean[scen_idx]
        hc_l = all_draws_hsi_cancelled_lower[scen_idx]
        hc_u = all_draws_hsi_cancelled_upper[scen_idx]
        htot = all_draws_hsi_total[scen_idx]
        all_hsi = hd_m.index.union(hc_m.index)
        all_hsi = all_hsi[all_hsi.astype(str) != "nan"]
        for hsi in all_hsi:
            dis_rows.append({
                "scenario": scen,
                "hsi_type": hsi,
                "mean_total_count": round(htot.get(hsi, 0), 2),
                "delayed_rate_mean": round(hd_m.get(hsi, 0), 6),
                "delayed_rate_lower": round(hd_l.get(hsi, 0), 6),
                "delayed_rate_upper": round(hd_u.get(hsi, 0), 6),
                "cancelled_rate_mean": round(hc_m.get(hsi, 0), 6),
                "cancelled_rate_lower": round(hc_l.get(hsi, 0), 6),
                "cancelled_rate_upper": round(hc_u.get(hsi, 0), 6),
                "total_disruption_rate_mean": round(hd_m.get(hsi, 0) + hc_m.get(hsi, 0), 6),
                "total_disruption_rate_lower": round(hd_l.get(hsi, 0) + hc_l.get(hsi, 0), 6),
                "total_disruption_rate_upper": round(hd_u.get(hsi, 0) + hc_u.get(hsi, 0), 6),
            })
    pd.DataFrame(dis_rows).sort_values(
        ["scenario", "total_disruption_rate_mean"], ascending=[True, False]
    ).to_csv(output_folder / f"hsi_disruption_by_type_{suffix}.csv", index=False)

    # ── PLOT ─────────────────────────────────────────────────────────────────
    print("Plotting …")

    n_scen = len(scenarios_of_interest)
    bar_h_a = min(0.7, 0.75 / n_scen)  # per-scenario bar height in Panel A
    bar_h_b = 0.55  # bar height in Panel B

    n_a = len(top_volume_types_plot)
    n_b = len(top_disrupted_types_plot)
    fig_h = max(9, max(n_a * n_scen * bar_h_a * 1.8, n_b * bar_h_b * 2.2))

    fig, (ax_b, ax_c, ax_a) = plt.subplots(  # actually want a at end for flow
        1, 3,
        figsize=(33, fig_h),
        gridspec_kw={"width_ratios": [1.15, 1, 1]},
    )

    # ── Panel A: total HSI volume by type ─────────────────────────────────────
    ref_volume_full = all_draws_volume_mean[ref_idx]  # No Disruptions mean

    y_centers = np.arange(n_a, dtype=float)

    # Collect non-baseline scenarios only
    comparison_draws = [
        (scen_idx, draw)
        for scen_idx, draw in enumerate(scenarios_of_interest)
        if scenario_names[draw] != "No Disruptions"
    ]
    n_comp = len(comparison_draws)
    bar_h_a = min(0.7, 0.75 / n_comp)

    for plot_i, (scen_idx, draw) in enumerate(comparison_draws):
        col = SCENARIO_COLOURS[(scen_idx) % len(SCENARIO_COLOURS)]
        vm = all_draws_volume_mean[scen_idx]
        vl = all_draws_volume_lower[scen_idx]
        vu = all_draws_volume_upper[scen_idx]

        ref = ref_volume_full.reindex(top_volume_types_plot).replace(0, np.nan)

        pct_mean = ((vm.reindex(top_volume_types_plot) - ref) / ref * 100).fillna(0).values
        pct_lower = ((vl.reindex(top_volume_types_plot) - ref) / ref * 100).fillna(0).values
        pct_upper = ((vu.reindex(top_volume_types_plot) - ref) / ref * 100).fillna(0).values

        offset = (plot_i - (n_comp - 1) / 2.0) * bar_h_a
        y_pos = y_centers + offset

        ax_a.barh(
            y_pos, pct_mean,
            height=bar_h_a * 0.9,
            color=col, alpha=0.85,
            label=scenario_names[draw],
        )
        ax_a.errorbar(
            pct_mean, y_pos,
            xerr=[pct_mean - pct_lower, pct_upper - pct_mean],
            fmt="none", color="black", lw=0.8, capsize=2, alpha=0.5,
        )

    ax_a.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax_a.set_yticks(y_centers)
    ax_a.set_yticklabels(_wrap_labels(top_volume_types_plot), fontsize=FS_TICK)
    ax_a.set_xlabel(
        '% difference in total HSIs vs "No Disruptions"',
        fontsize=FS_LABEL, fontweight="bold",
    )
    ax_a.set_title(
        f"(C) Change in HSI volume by treatment type\nvs baseline ({period_label})",
        fontsize=FS_TITLE, fontweight="bold",
    )
    plt.setp(ax_a.xaxis.get_majorticklabels(), fontsize=FS_TICK)
    ax_a.legend(
        title="Scenario", fontsize=FS_LEGEND, title_fontsize=FS_LEGEND,
        loc="lower right", framealpha=0.9,
    )
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(axis="x", color="lightgrey", linewidth=0.5, zorder=0)

    # ── Panel B: disruption rates for chosen scenario ─────────────────────────
    hd_m_b_plot = hd_m_b.reindex(top_disrupted_types_plot, fill_value=0)
    hd_l_b_plot = hd_l_b.reindex(top_disrupted_types_plot, fill_value=0)
    hd_u_b_plot = hd_u_b.reindex(top_disrupted_types_plot, fill_value=0)
    hc_m_b_plot = hc_m_b.reindex(top_disrupted_types_plot, fill_value=0)
    hc_l_b_plot = hc_l_b.reindex(top_disrupted_types_plot, fill_value=0)
    hc_u_b_plot = hc_u_b.reindex(top_disrupted_types_plot, fill_value=0)

    y_b = np.arange(n_b, dtype=float)
    delayed_vals = hd_m_b_plot.values * 100
    cancelled_vals = hc_m_b_plot.values * 100
    total_vals = delayed_vals + cancelled_vals
    total_lo = (hd_l_b_plot + hc_l_b_plot).values * 100
    total_hi = (hd_u_b_plot + hc_u_b_plot).values * 100

    ax_b.barh(y_b, delayed_vals, height=bar_h_b,
              color=COLOUR_DELAYED, alpha=0.75, label="Delayed")
    ax_b.barh(y_b, cancelled_vals, height=bar_h_b, left=delayed_vals,
              color=COLOUR_CANCELLED, alpha=0.75, label="Cancelled")
    ax_b.errorbar(
        total_vals, y_b,
        xerr=[total_vals - total_lo, total_hi - total_vals],
        fmt="none", color="black", lw=1.0, capsize=2, alpha=0.6,
    )

    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(_wrap_labels(top_disrupted_types_plot), fontsize=FS_TICK)
    ax_b.set_xlabel("% of HSIs disrupted", fontsize=FS_LABEL, fontweight="bold")
    ax_b.set_title(
        f"(A) Weather-disrupted HSIs by type\n"
        f"{disruption_panel_scenario} — top {top_n_disruption} ({period_label})",
        fontsize=FS_TITLE, fontweight="bold",
    )
    plt.setp(ax_b.xaxis.get_majorticklabels(), fontsize=FS_TICK)
    ax_b.set_xlim(left=0)
    ax_b.legend(fontsize=FS_LEGEND, loc="lower right", framealpha=0.9)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(axis="x", color="lightgrey", linewidth=0.5, zorder=0)

    # PANEL C: Least disrupted

    # ── Panel C: least disrupted ──────────────────────────────────────────────
    bottom_disrupted_types = (
        total_rate_b[total_rate_b > 0]
        .sort_values(ascending=True)
        .head(top_n_disruption)
        .index.tolist()
    )
    bottom_disrupted_types_plot = list(reversed(bottom_disrupted_types))
    n_c = len(bottom_disrupted_types_plot)

    hd_m_c_plot = hd_m_b.reindex(bottom_disrupted_types_plot, fill_value=0)
    hd_l_c_plot = hd_l_b.reindex(bottom_disrupted_types_plot, fill_value=0)
    hd_u_c_plot = hd_u_b.reindex(bottom_disrupted_types_plot, fill_value=0)
    hc_m_c_plot = hc_m_b.reindex(bottom_disrupted_types_plot, fill_value=0)
    hc_l_c_plot = hc_l_b.reindex(bottom_disrupted_types_plot, fill_value=0)
    hc_u_c_plot = hc_u_b.reindex(bottom_disrupted_types_plot, fill_value=0)

    y_c = np.arange(n_c, dtype=float)
    delayed_vals_c = hd_m_c_plot.values * 100
    cancelled_vals_c = hc_m_c_plot.values * 100
    total_vals_c = delayed_vals_c + cancelled_vals_c
    total_lo_c = (hd_l_c_plot + hc_l_c_plot).values * 100
    total_hi_c = (hd_u_c_plot + hc_u_c_plot).values * 100

    ax_c.barh(y_c, delayed_vals_c, height=bar_h_b,
              color=COLOUR_DELAYED, alpha=0.75, label="Delayed")
    ax_c.barh(y_c, cancelled_vals_c, height=bar_h_b, left=delayed_vals_c,
              color=COLOUR_CANCELLED, alpha=0.75, label="Cancelled")
    ax_c.errorbar(
        total_vals_c, y_c,
        xerr=[total_vals_c - total_lo_c, total_hi_c - total_vals_c],
        fmt="none", color="black", lw=1.0, capsize=2, alpha=0.6,
    )

    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels(_wrap_labels(bottom_disrupted_types_plot), fontsize=FS_TICK)
    ax_c.set_xlabel("% of HSIs disrupted", fontsize=FS_LABEL, fontweight="bold")
    ax_c.set_title(
        f"(B) Least weather-disrupted HSIs by type\n"
        f"{disruption_panel_scenario} — bottom {top_n_disruption} ({period_label})",
        fontsize=FS_TITLE, fontweight="bold",
    )
    plt.setp(ax_c.xaxis.get_majorticklabels(), fontsize=FS_TICK)
    ax_c.set_xlim(left=0)
    ax_c.legend(fontsize=FS_LEGEND, loc="lower right", framealpha=0.9)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.grid(axis="x", color="lightgrey", linewidth=0.5, zorder=0)

    fig.tight_layout()
    png_path = output_folder / f"hsi_volume_and_disruption_by_type_{suffix}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {png_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    parser.add_argument("--output_folder", type=Path, default=None)
    parser.add_argument("--resources", type=Path, default=Path("./resources"))
    args = parser.parse_args()
    apply(
        results_folder=args.results_folder,
        output_folder=args.output_folder or args.results_folder,
        resourcefilepath=args.resources,
    )
