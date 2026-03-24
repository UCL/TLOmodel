"""
comparison_actual_vs_expected_disruption_realfacility.py

Key fix: rates are computed per-run (disrupted_run / total_run) before averaging
across runs. This ensures rates are always bounded [0,1] and that numerator/denominator
are paired within the same run. Lower/upper bounds are derived from quantiles of
per-run rates rather than from count quantiles.

Composite key format: YYYY-MM:RealFacility_ID:HSI_type
This allows disruption rates to be sliced by month, facility, or HSI type.

IMPORTANT: For monthly/annual/facility plots the raw counts are first collapsed
to YYYY-MM:RealFacility_ID granularity (summing across HSI types) before rates
are computed. This preserves identical behaviour to the original script.
The HSI-type plots use the full three-component key.
"""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date
from tlo.analysis.utils import extract_results


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path):
    # ─────────────────────────────────────────────────────────────────────────────
    #  CONFIG
    # ─────────────────────────────────────────────────────────────────────────────

    min_year = 2025
    max_year = 2041
    spacing_of_years = 1

    main_text = True
    parameter_uncertainty_analysis = False

    top_n_hsi = 15  # HSI types to show in bar charts

    if parameter_uncertainty_analysis:
        scenario_names = list(range(0, 50))
        scenarios_of_interest = scenario_names
        suffix = "parameter_UA"
    if main_text:
        scenario_names = ["No disruptions", "Baseline", "Worst Case"]
        scenarios_of_interest = [0, 1, 2]
        suffix = "main_text"

    # ─────────────────────────────────────────────────────────────────────────────
    #  EXTRACTION HELPERS
    # ─────────────────────────────────────────────────────────────────────────────

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
            if "TREATMENT_ID" in _df.columns:
                _df["hsi_type"] = _df["TREATMENT_ID"].fillna("unknown").astype(str)
            else:
                _df["hsi_type"] = "unknown"
            _df["composite"] = (
                _df["date"].dt.strftime("%Y-%m") + ":"
                + _df["RealFacility_ID"].astype(str) + ":"
                + _df["hsi_type"]
            )
            return _df["composite"].value_counts().astype(float)
        return _fn

    def _extract_df(results_folder, draw, log_key, fn):
            raw = extract_results(
                results_folder,
                module="tlo.methods.healthsystem.summary",
                key=log_key,
                custom_generate_series=fn,
                do_scaling=False,
            )
            return raw[draw].fillna(0)

    # ── Key parsing helpers ───────────────────────────────────────────────────────

    def _parse_ym(index):
        return index.astype(str).str.split(":", n=1).str[0]

    def _parse_facility(index):
        return index.astype(str).str.split(":", n=2).str[1]

    def _parse_hsi_type(index):
        return index.astype(str).str.split(":", n=2).str[2]

    def _collapse_hsi_types(df):
        if df.empty:
            return df
        key_2 = _parse_ym(df.index) + ":" + _parse_facility(df.index)
        return df.groupby(key_2).sum()

    def _align_and_rate(total_df, disrupted_df):
        idx = total_df.index.union(disrupted_df.index)
        t = total_df.reindex(idx, fill_value=0)
        d = disrupted_df.reindex(idx, fill_value=0)
        return d.div(t).where(t > 0, 0.0)

    def _monthly_stats(rate_df):
        rate_df = rate_df.copy()
        ym = _parse_ym(rate_df.index)
        monthly = rate_df.groupby(ym).mean().sort_index()
        return (
            monthly.mean(axis=1),
            monthly.quantile(0.025, axis=1),
            monthly.quantile(0.975, axis=1),
        )

    def _annual_stats(rate_df):
        rate_df = rate_df.copy()
        yr = _parse_ym(rate_df.index).str[:4]
        annual = rate_df.groupby(yr).mean().sort_index()
        return (
            annual.mean(axis=1),
            annual.quantile(0.025, axis=1),
            annual.quantile(0.975, axis=1),
        )

    def _facility_stats(rate_df, total_df):
        rate_df = rate_df.copy()
        total_df = total_df.reindex(rate_df.index, fill_value=0)
        fac = _parse_facility(rate_df.index)
        fac_rate = rate_df.groupby(fac).mean().mean(axis=1)
        fac_total = total_df.groupby(fac).sum().mean(axis=1)
        return fac_rate, fac_total

    def _hsi_type_stats(rate_df, total_df):
        rate_df = rate_df.copy()
        total_df = total_df.reindex(rate_df.index, fill_value=0)
        hsi = _parse_hsi_type(rate_df.index)
        by_type = rate_df.groupby(hsi).mean()
        by_type_total = total_df.groupby(hsi).sum()
        return (
            by_type.mean(axis=1),
            by_type.quantile(0.025, axis=1),
            by_type.quantile(0.975, axis=1),
            by_type_total.mean(axis=1),
        )

    # ─────────────────────────────────────────────────────────────────────────────
    #  MAIN EXTRACTION LOOP
    # ─────────────────────────────────────────────────────────────────────────────

    target_year_sequence = range(min_year, max_year, spacing_of_years)
    tlo_facilities = set()

    all_draws_monthly_delayed_mean = []
    all_draws_monthly_cancelled_mean = []
    all_draws_monthly_delayed_lower = []
    all_draws_monthly_delayed_upper = []
    all_draws_monthly_cancelled_lower = []
    all_draws_monthly_cancelled_upper = []

    all_draws_annual_delayed_mean = []
    all_draws_annual_cancelled_mean = []
    all_draws_annual_delayed_lower = []
    all_draws_annual_delayed_upper = []
    all_draws_annual_cancelled_lower = []
    all_draws_annual_cancelled_upper = []

    all_draws_hsi_delayed_mean = []
    all_draws_hsi_delayed_lower = []
    all_draws_hsi_delayed_upper = []
    all_draws_hsi_cancelled_mean = []
    all_draws_hsi_cancelled_lower = []
    all_draws_hsi_cancelled_upper = []
    all_draws_hsi_total = []

    all_draws_total_df = {}
    all_draws_delayed_df = {}
    all_draws_cancelled_df = {}

    for draw in scenarios_of_interest:

        if scenario_names[draw] == "No disruptions":
            all_months = pd.date_range(
                start=f"{min_year}-01-01", end=f"{max_year - 1}-12-01", freq="MS"
            ).strftime("%Y-%m")
            all_years_str = [str(y) for y in target_year_sequence]
            zeros = pd.Series(0.0, index=all_months)
            zeros_yr = pd.Series(0.0, index=all_years_str)
            for lst in [all_draws_monthly_delayed_mean, all_draws_monthly_cancelled_mean,
                        all_draws_monthly_delayed_lower, all_draws_monthly_delayed_upper,
                        all_draws_monthly_cancelled_lower, all_draws_monthly_cancelled_upper]:
                lst.append(zeros)
            for lst in [all_draws_annual_delayed_mean, all_draws_annual_cancelled_mean,
                        all_draws_annual_delayed_lower, all_draws_annual_delayed_upper,
                        all_draws_annual_cancelled_lower, all_draws_annual_cancelled_upper]:
                lst.append(zeros_yr)
            empty = pd.Series(dtype=float)
            for lst in [all_draws_hsi_delayed_mean, all_draws_hsi_delayed_lower,
                        all_draws_hsi_delayed_upper, all_draws_hsi_cancelled_mean,
                        all_draws_hsi_cancelled_lower, all_draws_hsi_cancelled_upper,
                        all_draws_hsi_total]:
                lst.append(empty)
            continue

        all_years_total_dfs = []
        all_years_delayed_dfs = []
        all_years_cancelled_dfs = []

        for target_year in target_year_sequence:
            TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))

            fn_total = _make_hsi_counts_by_real_facility_monthly(TARGET_PERIOD)
            fn_disrupted = _make_disrupted_by_real_facility_monthly(TARGET_PERIOD)

            total_df = _extract_df(results_folder, draw, "hsi_event_counts_by_facility_monthly", fn_total)
            delayed_df = _extract_df(results_folder, draw, "Weather_delayed_HSI_Event_full_info", fn_disrupted)
            cancelled_df = _extract_df(results_folder, draw, "Weather_cancelled_HSI_Event_full_info", fn_disrupted)

            all_years_total_dfs.append(total_df)
            all_years_delayed_dfs.append(delayed_df)
            all_years_cancelled_dfs.append(cancelled_df)

        def _concat_years(dfs):
            combined = pd.concat(dfs)
            return combined.groupby(level=0).sum()

        total_all = _concat_years(all_years_total_dfs)
        delayed_all = _concat_years(all_years_delayed_dfs)
        cancelled_all = _concat_years(all_years_cancelled_dfs)

        tlo_facilities.update(_parse_facility(total_all.index).dropna())

        all_draws_total_df[draw] = total_all
        all_draws_delayed_df[draw] = delayed_all
        all_draws_cancelled_df[draw] = cancelled_all

        # Collapse to YYYY-MM:RealFacility_ID before computing rates for
        # monthly/annual/facility plots (preserves original behaviour)
        total_2 = _collapse_hsi_types(total_all)
        delayed_2 = _collapse_hsi_types(delayed_all)
        cancelled_2 = _collapse_hsi_types(cancelled_all)

        delayed_rate_2 = _align_and_rate(total_2, delayed_2)
        cancelled_rate_2 = _align_and_rate(total_2, cancelled_2)

        dm, dl, du = _monthly_stats(delayed_rate_2)
        cm, cl, cu = _monthly_stats(cancelled_rate_2)
        all_draws_monthly_delayed_mean.append(dm)
        all_draws_monthly_delayed_lower.append(dl)
        all_draws_monthly_delayed_upper.append(du)
        all_draws_monthly_cancelled_mean.append(cm)
        all_draws_monthly_cancelled_lower.append(cl)
        all_draws_monthly_cancelled_upper.append(cu)

        dam, dal, dau = _annual_stats(delayed_rate_2)
        cam, cal, cau = _annual_stats(cancelled_rate_2)
        all_draws_annual_delayed_mean.append(dam)
        all_draws_annual_delayed_lower.append(dal)
        all_draws_annual_delayed_upper.append(dau)
        all_draws_annual_cancelled_mean.append(cam)
        all_draws_annual_cancelled_lower.append(cal)
        all_draws_annual_cancelled_upper.append(cau)

        # HSI-type stats use full three-component index
        delayed_rate_3 = _align_and_rate(total_all, delayed_all)
        cancelled_rate_3 = _align_and_rate(total_all, cancelled_all)

        hd_m, hd_l, hd_u, hd_tot = _hsi_type_stats(delayed_rate_3, total_all)
        hc_m, hc_l, hc_u, _ = _hsi_type_stats(cancelled_rate_3, total_all)
        all_draws_hsi_delayed_mean.append(hd_m)
        all_draws_hsi_delayed_lower.append(hd_l)
        all_draws_hsi_delayed_upper.append(hd_u)
        all_draws_hsi_cancelled_mean.append(hc_m)
        all_draws_hsi_cancelled_lower.append(hc_l)
        all_draws_hsi_cancelled_upper.append(hc_u)
        all_draws_hsi_total.append(hd_tot)

    # ─────────────────────────────────────────────────────────────────────────────
    #  RESOURCE FILE DISRUPTION OVERLAY
    # ─────────────────────────────────────────────────────────────────────────────

    disruptions_df = pd.read_csv(
        resourcefilepath / "climate_change_impacts"
        / "ResourceFile_Precipitation_Disruptions_ssp245_mean.csv"
    )
    disruptions_df = disruptions_df[disruptions_df["RealFacility_ID"].isin(tlo_facilities)]
    disruptions_df = disruptions_df[
        (disruptions_df["year"] >= min_year) & (disruptions_df["year"] <= max_year - 1)
    ]

    avg_df_annual = (
        disruptions_df.groupby("year")["mean_all_service"]
        .mean().reset_index().sort_values("year")
    )
    avg_df_annual["Date"] = pd.to_datetime(avg_df_annual["year"].astype(str) + "-01-01")
    rf_disruption_annual = avg_df_annual["mean_all_service"].values * 100
    rf_dates_annual = avg_df_annual["Date"].values

    avg_df_monthly = (
        disruptions_df.groupby(["year", "month"])["mean_all_service"]
        .mean().reset_index().sort_values(["year", "month"])
    )
    avg_df_monthly["date"] = pd.to_datetime(
        avg_df_monthly["year"].astype(str) + "-" +
        avg_df_monthly["month"].astype(str).str.zfill(2) + "-01"
    )
    rf_disruption_monthly = avg_df_monthly["mean_all_service"].values * 100
    rf_dates_monthly = avg_df_monthly["date"].values

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT A: MONTHLY time series
    # ─────────────────────────────────────────────────────────────────────────────


    n_plots = len(scenarios_of_interest) - (1 if "No disruptions" in scenario_names else 0)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    COLOUR_DELAYED = "#CC7000"
    COLOUR_CANCELLED = "#A8102E"
    COLOUR_TOTAL = "#1E3A8A"
    COLOUR_RF = "#75AE70"

    def _series_to_dates_pct(s):
        if s.empty:
            return pd.DatetimeIndex([]), np.array([])
        dates = pd.to_datetime(s.index.astype(str) + "-01")
        return dates, s.values * 100

    plot_idx = 0
    for idx, draw in enumerate(scenarios_of_interest):
        if scenario_names[draw] == "No disruptions":
            continue
        ax = axes_flat[plot_idx]
        plot_idx += 1

        d_dates, d_vals = _series_to_dates_pct(all_draws_monthly_delayed_mean[idx])
        c_dates, c_vals = _series_to_dates_pct(all_draws_monthly_cancelled_mean[idx])
        t_vals = d_vals + c_vals if len(d_vals) == len(c_vals) else np.array([])

        if len(d_dates):
            ax.plot(d_dates, d_vals, color=COLOUR_DELAYED, lw=1.5, alpha=0.6, label="Delayed (TLO)")
        if len(c_dates):
            ax.plot(c_dates, c_vals, color=COLOUR_CANCELLED, lw=1.5, alpha=0.6, label="Cancelled (TLO)")
        if len(d_dates):
            ax.plot(d_dates, t_vals, color=COLOUR_TOTAL, lw=3, alpha=0.9, label="Total disrupted (TLO)")
        if len(rf_disruption_monthly):
            ax.plot(rf_dates_monthly, rf_disruption_monthly,
                    color=COLOUR_RF, lw=2.5, ls="--", alpha=0.9, label="DHIS2 ANC Data")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Month", fontsize=11, fontweight="bold")
        ax.set_ylabel("% HSIs disrupted", fontsize=11, fontweight="bold")
        ax.set_title(scenario_names[draw], fontsize=13, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=pd.Timestamp("2025-01-01"))
        if plot_idx == 1:
            ax.legend(fontsize=9, framealpha=0.95, edgecolor="gray", fancybox=True)

    for j in range(plot_idx, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    out_monthly = output_folder / f"comparison_disruption_monthly_{suffix}.png"
    fig.savefig(out_monthly, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT B: ANNUAL time series
    # ─────────────────────────────────────────────────────────────────────────────

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    SCENARIO_COLOURS = [
        "#823038", "#00566f", "#c65a52",
        "#5b3f8c", "#8e7cc3", "#c7b7ec",
        "#0081a7", "#5ab4c6", "#f07167", "#f59e96",
    ]

    for idx, draw in enumerate(scenarios_of_interest):
        col = SCENARIO_COLOURS[idx % len(SCENARIO_COLOURS)]
        d_m = all_draws_annual_delayed_mean[idx]
        c_m = all_draws_annual_cancelled_mean[idx]
        d_lo = all_draws_annual_delayed_lower[idx]
        d_hi = all_draws_annual_delayed_upper[idx]
        c_lo = all_draws_annual_cancelled_lower[idx]
        c_hi = all_draws_annual_cancelled_upper[idx]

        if d_m.empty and c_m.empty:
            continue

        years = pd.to_datetime([f"{y}-01-01" for y in d_m.index])
        total = (d_m + c_m.reindex(d_m.index, fill_value=0)).values * 100
        total_lo = (d_lo + c_lo.reindex(d_lo.index, fill_value=0)).values * 100
        total_hi = (d_hi + c_hi.reindex(d_hi.index, fill_value=0)).values * 100

        ax2.fill_between(years, total_lo, total_hi, color=col, alpha=0.15, linewidth=0)
        ax2.plot(years, total, color=col, lw=2.5, label=scenario_names[draw])
        ax2.plot(years, d_m.values * 100, color=col, lw=1, ls="--", alpha=0.5)
        ax2.plot(years, c_m.reindex(d_m.index, fill_value=0).values * 100,
                 color=col, lw=1, ls=":", alpha=0.5)

    if len(rf_disruption_annual):
        ax2.plot(rf_dates_annual, rf_disruption_annual,
                 color=COLOUR_RF, lw=2.5, ls="--", alpha=0.9, label="ResourceFile disruption rate")

    style_handles = [
        mlines.Line2D([], [], color="grey", lw=2.5, ls="-", label="Total disrupted"),
        mlines.Line2D([], [], color="grey", lw=1, ls="--", alpha=0.7, label="Delayed"),
        mlines.Line2D([], [], color="grey", lw=1, ls=":", alpha=0.7, label="Cancelled"),
    ]
    ax2.legend(handles=style_handles, loc="upper right", fontsize=9,
               framealpha=0.85, title="Line style", title_fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax2.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax2.set_ylabel("% HSIs disrupted", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"Annual mean per-facility disruption rate by scenario ({min_year}–{max_year - 1})",
        fontsize=12, fontweight="bold",
    )
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=pd.Timestamp("2025-01-01"))
    ax2.legend(fontsize=9, frameon=True, framealpha=0.9)
    fig2.tight_layout()
    out_annual = output_folder / f"comparison_disruption_annual_{suffix}.png"
    fig2.savefig(out_annual, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # ─────────────────────────────────────────────────────────────────────────────
    #  SHARED FIGURE SETUP for Plots C and C2
    # ─────────────────────────────────────────────────────────────────────────────

    n_scen_hsi = sum(
        1 for idx, draw in enumerate(scenarios_of_interest)
        if scenario_names[draw] != "No disruptions"
        and not all_draws_hsi_delayed_mean[idx].empty
    )
    n_cols_hsi = min(3, n_scen_hsi)
    n_rows_hsi = (n_scen_hsi + n_cols_hsi - 1) // n_cols_hsi
    bar_height = max(0.3, min(0.7, 12 / max(top_n_hsi, 1)))
    fig_h = max(6, top_n_hsi * bar_height * 1.4)

    def _draw_hsi_bar_panel(ax, hd_m, hd_l, hd_u, hc_m, hc_l, hc_u,
                            hsi_types, title):
        """Draw a single stacked bar panel for a given ordered list of HSI types."""
        hd_m = hd_m.reindex(hsi_types, fill_value=0)
        hd_l = hd_l.reindex(hsi_types, fill_value=0)
        hd_u = hd_u.reindex(hsi_types, fill_value=0)
        hc_m = hc_m.reindex(hsi_types, fill_value=0)
        hc_l = hc_l.reindex(hsi_types, fill_value=0)
        hc_u = hc_u.reindex(hsi_types, fill_value=0)

        y_pos = np.arange(len(hsi_types))
        total_m = (hd_m + hc_m).values * 100
        total_l = (hd_l + hc_l).values * 100
        total_u = (hd_u + hc_u).values * 100
        delayed_m = hd_m.values * 100
        canceld_m = hc_m.values * 100

        ax.barh(y_pos, delayed_m, height=bar_height,
                color=COLOUR_DELAYED, alpha=0.75, label="Delayed")
        ax.barh(y_pos, canceld_m, height=bar_height,
                left=delayed_m, color=COLOUR_CANCELLED, alpha=0.75, label="Cancelled")
        ax.errorbar(
            total_m, y_pos,
            xerr=[total_m - total_l, total_u - total_m],
            fmt="none", color="black", lw=1.0, capsize=2, alpha=0.6,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(hsi_types, fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel("% of HSIs disrupted", fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlim(left=0)
        ax.legend(fontsize=8, loc="lower right")

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT C: TOP 15 MOST DISRUPTED HSI TYPES — per-draw ranking
    # ─────────────────────────────────────────────────────────────────────────────

    fig3, axes3 = plt.subplots(n_rows_hsi, n_cols_hsi,
                               figsize=(9 * n_cols_hsi, fig_h), squeeze=False)
    axes3_flat = axes3.flatten()

    plot_idx = 0
    for idx, draw in enumerate(scenarios_of_interest):
        if scenario_names[draw] == "No disruptions":
            continue
        hd_m = all_draws_hsi_delayed_mean[idx]
        hd_l = all_draws_hsi_delayed_lower[idx]
        hd_u = all_draws_hsi_delayed_upper[idx]
        hc_m = all_draws_hsi_cancelled_mean[idx]
        hc_l = all_draws_hsi_cancelled_lower[idx]
        hc_u = all_draws_hsi_cancelled_upper[idx]

        if hd_m.empty and hc_m.empty:
            continue

        # Rank by total disruption rate for THIS draw only
        draw_total_rate = (hd_m + hc_m).copy()
        draw_total_rate = draw_total_rate[draw_total_rate.index.astype(str) != "nan"]
        draw_total_rate = draw_total_rate[draw_total_rate > 0].sort_values(ascending=False)
        top_types = draw_total_rate.head(top_n_hsi).index.tolist()

        ax = axes3_flat[plot_idx]
        plot_idx += 1
        _draw_hsi_bar_panel(ax, hd_m, hd_l, hd_u, hc_m, hc_l, hc_u,
                            top_types, scenario_names[draw])

    for j in range(plot_idx, len(axes3_flat)):
        axes3_flat[j].set_visible(False)

    fig3.suptitle(
        f"Top {top_n_hsi} most disrupted HSI types ({min_year}–{max_year - 1})\n"
        "Ranked by mean total disruption rate within each scenario; whiskers = 95% CI",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig3.tight_layout()
    out_hsi = output_folder / f"comparison_disruption_by_hsi_type_{suffix}.png"
    fig3.savefig(out_hsi, dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT C2: TOP 15 MOST CANCELLED HSI TYPES — per-draw ranking
    # ─────────────────────────────────────────────────────────────────────────────

    fig5, axes5 = plt.subplots(n_rows_hsi, n_cols_hsi,
                               figsize=(9 * n_cols_hsi, fig_h), squeeze=False)
    axes5_flat = axes5.flatten()

    plot_idx = 0
    for idx, draw in enumerate(scenarios_of_interest):
        if scenario_names[draw] == "No disruptions":
            continue
        hd_m = all_draws_hsi_delayed_mean[idx]
        hd_l = all_draws_hsi_delayed_lower[idx]
        hd_u = all_draws_hsi_delayed_upper[idx]
        hc_m = all_draws_hsi_cancelled_mean[idx]
        hc_l = all_draws_hsi_cancelled_lower[idx]
        hc_u = all_draws_hsi_cancelled_upper[idx]

        if hd_m.empty and hc_m.empty:
            continue

        # Rank by CANCELLED rate for THIS draw only
        draw_cancelled_rate = hc_m.copy()
        draw_cancelled_rate = draw_cancelled_rate[draw_cancelled_rate.index.astype(str) != "nan"]
        draw_cancelled_rate = draw_cancelled_rate[draw_cancelled_rate > 0].sort_values(ascending=False)
        top_cancelled = draw_cancelled_rate.head(top_n_hsi).index.tolist()

        ax = axes5_flat[plot_idx]
        plot_idx += 1
        _draw_hsi_bar_panel(ax, hd_m, hd_l, hd_u, hc_m, hc_l, hc_u,
                            top_cancelled, scenario_names[draw])

    for j in range(plot_idx, len(axes5_flat)):
        axes5_flat[j].set_visible(False)

    fig5.suptitle(
        f"Top {top_n_hsi} most cancelled HSI types ({min_year}–{max_year - 1})\n"
        "Ranked by mean cancelled rate within each scenario; whiskers = 95% CI on total",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig5.tight_layout()
    out_cancelled = output_folder / f"comparison_disruption_top_cancelled_hsi_type_{suffix}.png"
    fig5.savefig(out_cancelled, dpi=300, bbox_inches="tight")
    plt.close(fig5)

    # ─────────────────────────────────────────────────────────────────────────────
    #  CSV OUTPUTS — monthly / annual / HSI type
    # ─────────────────────────────────────────────────────────────────────────────

    monthly_rows = []
    for idx, draw in enumerate(scenarios_of_interest):
        d_m = all_draws_monthly_delayed_mean[idx]
        c_m = all_draws_monthly_cancelled_mean[idx]
        for ym in d_m.index.union(c_m.index):
            monthly_rows.append({
                "Scenario": scenario_names[draw],
                "draw": draw,
                "year_month": ym,
                "delayed_rate": d_m.get(ym, 0),
                "cancelled_rate": c_m.get(ym, 0),
                "total_disruption_rate": d_m.get(ym, 0) + c_m.get(ym, 0),
            })
    pd.DataFrame(monthly_rows).to_csv(
        output_folder / f"monthly_disruption_rates_realfacilityid_{suffix}.csv", index=False
    )

    annual_rows = []
    for idx, draw in enumerate(scenarios_of_interest):
        d_m = all_draws_annual_delayed_mean[idx]
        c_m = all_draws_annual_cancelled_mean[idx]
        d_lo = all_draws_annual_delayed_lower[idx]
        d_hi = all_draws_annual_delayed_upper[idx]
        c_lo = all_draws_annual_cancelled_lower[idx]
        c_hi = all_draws_annual_cancelled_upper[idx]
        for yr in d_m.index.union(c_m.index):
            annual_rows.append({
                "Scenario": scenario_names[draw],
                "draw": draw,
                "year": yr,
                "delayed_rate_mean": d_m.get(yr, 0),
                "delayed_rate_lower": d_lo.get(yr, 0),
                "delayed_rate_upper": d_hi.get(yr, 0),
                "cancelled_rate_mean": c_m.get(yr, 0),
                "cancelled_rate_lower": c_lo.get(yr, 0),
                "cancelled_rate_upper": c_hi.get(yr, 0),
                "total_disruption_rate_mean": d_m.get(yr, 0) + c_m.get(yr, 0),
                "total_disruption_rate_lower": d_lo.get(yr, 0) + c_lo.get(yr, 0),
                "total_disruption_rate_upper": d_hi.get(yr, 0) + c_hi.get(yr, 0),
            })
    pd.DataFrame(annual_rows).to_csv(
        output_folder / f"annual_disruption_rates_realfacilityid_{suffix}.csv", index=False
    )

    hsi_rows = []
    for idx, draw in enumerate(scenarios_of_interest):
        if scenario_names[draw] == "No disruptions":
            continue
        hd_m = all_draws_hsi_delayed_mean[idx]
        hd_l = all_draws_hsi_delayed_lower[idx]
        hd_u = all_draws_hsi_delayed_upper[idx]
        hc_m = all_draws_hsi_cancelled_mean[idx]
        hc_l = all_draws_hsi_cancelled_lower[idx]
        hc_u = all_draws_hsi_cancelled_upper[idx]
        htot = all_draws_hsi_total[idx]
        for hsi in hd_m.index.union(hc_m.index):
            hsi_rows.append({
                "Scenario": scenario_names[draw],
                "draw": draw,
                "hsi_type": hsi,
                "mean_total_count": htot.get(hsi, 0),
                "delayed_rate_mean": hd_m.get(hsi, 0),
                "delayed_rate_lower": hd_l.get(hsi, 0),
                "delayed_rate_upper": hd_u.get(hsi, 0),
                "cancelled_rate_mean": hc_m.get(hsi, 0),
                "cancelled_rate_lower": hc_l.get(hsi, 0),
                "cancelled_rate_upper": hc_u.get(hsi, 0),
                "total_disruption_rate_mean": hd_m.get(hsi, 0) + hc_m.get(hsi, 0),
                "total_disruption_rate_lower": hd_l.get(hsi, 0) + hc_l.get(hsi, 0),
                "total_disruption_rate_upper": hd_u.get(hsi, 0) + hc_u.get(hsi, 0),
            })
    pd.DataFrame(hsi_rows).to_csv(
        output_folder / f"hsi_type_disruption_rates_{suffix}.csv", index=False
    )

    # ─────────────────────────────────────────────────────────────────────────────
    #  MAIN TEXT SUMMARY CSV
    #  Per scenario:
    #    1) total HSIs disrupted (delayed + cancelled), mean across runs
    #    2) total HSIs cancelled, mean across runs
    #    3) total HSIs delayed, mean across runs
    #    4) average monthly disruption rate across all facilities and months
    #    5) max monthly disruption rate across all facilities and months
    #    6) min monthly disruption rate across all facilities and months
    #    7) std monthly disruption rate across all facilities and months
    # ─────────────────────────────────────────────────────────────────────────────

    summary_rows = []
    for idx, draw in enumerate(scenarios_of_interest):
        scen = scenario_names[draw]

        if scen == "No disruptions":
            summary_rows.append({
                "Scenario": scen,
                "total_hsi_disrupted_mean": 0,
                "total_hsi_cancelled_mean": 0,
                "total_hsi_delayed_mean": 0,
                "monthly_disruption_rate_mean_%": 0,
                "monthly_disruption_rate_max_%": 0,
                "monthly_disruption_rate_min_%": 0,
                "monthly_disruption_rate_std_%": 0,
            })
            continue

        # ── absolute counts ───────────────────────────────────────────────────
        # Collapse to YYYY-MM:RealFacility_ID then sum across all rows per run,
        # then average across runs.
        total_2 = _collapse_hsi_types(all_draws_total_df[draw])
        delayed_2 = _collapse_hsi_types(all_draws_delayed_df[draw])
        cancelled_2 = _collapse_hsi_types(all_draws_cancelled_df[draw])

        # reindex disrupted onto total index so zeros are correctly included
        delayed_2_r = delayed_2.reindex(total_2.index, fill_value=0)
        cancelled_2_r = cancelled_2.reindex(total_2.index, fill_value=0)

        total_delayed_per_run = delayed_2_r.sum(axis=0)  # sum over facilities×months
        total_cancelled_per_run = cancelled_2_r.sum(axis=0)
        total_disrupted_per_run = total_delayed_per_run + total_cancelled_per_run

        mean_delayed = total_delayed_per_run.mean()
        mean_cancelled = total_cancelled_per_run.mean()
        mean_disrupted = total_disrupted_per_run.mean()

        # ── monthly disruption rate distribution ──────────────────────────────
        # all_draws_monthly_*_mean[idx] is a Series indexed by YYYY-MM,
        # already averaged across facilities per month per run, then averaged
        # across runs.  We want stats across the months dimension.
        monthly_total_rate = (
                                 all_draws_monthly_delayed_mean[idx] +
                                 all_draws_monthly_cancelled_mean[idx]
                             ) * 100  # convert to %

        summary_rows.append({
            "Scenario": scen,
            "total_hsi_disrupted_mean": round(mean_disrupted, 1),
            "total_hsi_cancelled_mean": round(mean_cancelled, 1),
            "total_hsi_delayed_mean": round(mean_delayed, 1),
            "monthly_disruption_rate_mean_%": round(monthly_total_rate.mean(), 4),
            "monthly_disruption_rate_max_%": round(monthly_total_rate.max(), 4),
            "monthly_disruption_rate_min_%": round(monthly_total_rate.min(), 4),
            "monthly_disruption_rate_std_%": round(monthly_total_rate.std(), 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    out_summary = output_folder / f"main_text_summary_{suffix}.csv"
    summary_df.to_csv(out_summary, index=False)

    # ─────────────────────────────────────────────────────────────────────────────
    #  PER-FACILITY COMPARISON: TLO vs ResourceFile
    # ─────────────────────────────────────────────────────────────────────────────

    rf_facility = (
        disruptions_df.groupby("RealFacility_ID")["mean_all_service"]
        .mean().rename("rf_rate")
    )

    n_scen = len(scenarios_of_interest) - (1 if "No disruptions" in scenario_names else 0)
    fig4, axes4 = plt.subplots(1, n_scen, figsize=(7 * n_scen, 6), squeeze=False)
    axes4_flat = axes4.flatten()
    plot_idx = 0
    merged_all = rf_facility.to_frame()

    for idx, draw in enumerate(scenarios_of_interest):
        if scenario_names[draw] == "No disruptions":
            continue

        total_2 = _collapse_hsi_types(all_draws_total_df[draw])
        delayed_2 = _collapse_hsi_types(all_draws_delayed_df[draw])
        cancelled_2 = _collapse_hsi_types(all_draws_cancelled_df[draw])

        delayed_rate_2 = _align_and_rate(total_2, delayed_2)
        cancelled_rate_2 = _align_and_rate(total_2, cancelled_2)
        total_rate_2 = delayed_rate_2.add(
            cancelled_rate_2.reindex(delayed_rate_2.index, fill_value=0), fill_value=0
        )

        tlo_rate, tlo_total = _facility_stats(total_rate_2, total_2)
        tlo_rate.name = f"tlo_rate_{scenario_names[draw]}"
        tlo_total.name = f"tlo_total_{scenario_names[draw]}"
        merged_all = merged_all.join(tlo_rate, how="outer")
        merged_all = merged_all.join(tlo_total, how="outer")

        ax = axes4_flat[plot_idx]
        plot_idx += 1
        merged = pd.concat([tlo_rate.rename("tlo_rate"), rf_facility], axis=1).dropna()
        merged = merged[merged.index != "nan"]

        ax.scatter(merged["rf_rate"] * 100, merged["tlo_rate"] * 100,
                   alpha=0.5, s=20, color=SCENARIO_COLOURS[idx % len(SCENARIO_COLOURS)])
        max_val = max(merged["rf_rate"].max(), merged["tlo_rate"].max()) * 100
        ax.plot([0, max_val], [0, max_val], "k--", lw=1, alpha=0.5, label="1:1 line")
        ax.set_xlabel("ResourceFile disruption rate (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("TLO disruption rate (%)", fontsize=11, fontweight="bold")
        ax.set_title(scenario_names[draw], fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    fig4.suptitle("Per-facility disruption rate: TLO vs ResourceFile",
                  fontsize=12, fontweight="bold")
    fig4.tight_layout()
    out_facility = output_folder / f"comparison_disruption_per_facility_{suffix}.png"
    fig4.savefig(out_facility, dpi=300, bbox_inches="tight")
    plt.close(fig4)

    merged_all.to_csv(output_folder / f"per_facility_disruption_rates_{suffix}.csv")

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT D: COMBINED FIGURE — 2x2
    #  (A) Baseline   most disrupted   (B) Worst Case most disrupted
    #  (C) Baseline   most cancelled   (D) Worst Case most cancelled
    # ─────────────────────────────────────────────────────────────────────────────

    non_zero_draws = [
        (idx, draw) for idx, draw in enumerate(scenarios_of_interest)
        if scenario_names[draw] != "No disruptions"
           and not all_draws_hsi_delayed_mean[idx].empty
    ]

    fig6, axes6 = plt.subplots(2, len(non_zero_draws),
                               figsize=(9 * len(non_zero_draws), fig_h * 2),
                               squeeze=False)

    panel_labels = iter("ABCDEFGH")

    for col, (idx, draw) in enumerate(non_zero_draws):
        hd_m = all_draws_hsi_delayed_mean[idx]
        hd_l = all_draws_hsi_delayed_lower[idx]
        hd_u = all_draws_hsi_delayed_upper[idx]
        hc_m = all_draws_hsi_cancelled_mean[idx]
        hc_l = all_draws_hsi_cancelled_lower[idx]
        hc_u = all_draws_hsi_cancelled_upper[idx]

        # ── Row 0: most disrupted (delayed + cancelled) ───────────────────────
        draw_total_rate = (hd_m + hc_m).copy()
        draw_total_rate = draw_total_rate[draw_total_rate.index.astype(str) != "nan"]
        draw_total_rate = draw_total_rate[draw_total_rate > 0].sort_values(ascending=False)
        top_disrupted = draw_total_rate.head(top_n_hsi).index.tolist()

        ax_top = axes6[0, col]
        _draw_hsi_bar_panel(ax_top, hd_m, hd_l, hd_u, hc_m, hc_l, hc_u,
                            top_disrupted, scenario_names[draw])
        label = next(panel_labels)
        ax_top.set_title(f"({label}) {scenario_names[draw]} — most disrupted",
                         fontsize=13, fontweight="bold")

        # ── Row 1: most cancelled ─────────────────────────────────────────────
        draw_cancelled_rate = hc_m.copy()
        draw_cancelled_rate = draw_cancelled_rate[draw_cancelled_rate.index.astype(str) != "nan"]
        draw_cancelled_rate = draw_cancelled_rate[draw_cancelled_rate > 0].sort_values(ascending=False)
        top_cancelled = draw_cancelled_rate.head(top_n_hsi).index.tolist()

        ax_bot = axes6[1, col]
        _draw_hsi_bar_panel(ax_bot, hd_m, hd_l, hd_u, hc_m, hc_l, hc_u,
                            top_cancelled, scenario_names[draw])
        label = next(panel_labels)
        ax_bot.set_title(f"({label}) {scenario_names[draw]} — most cancelled",
                         fontsize=13, fontweight="bold")

    fig6.suptitle(
        f"Disruption by HSI type: top {top_n_hsi} most disrupted (rows A–B) "
        f"and most cancelled (rows C–D)\n{min_year}–{max_year - 1}  |  "
        "whiskers = 95% CI on total disruption rate",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig6.tight_layout()
    pd.DataFrame(top_cancelled).to_csv(results_folder / "top_cancelled.csv", index=False)
    pd.DataFrame(top_disrupted).to_csv(results_folder / "top_disrupted.csv", index=False)
    out_combined = output_folder / f"comparison_disruption_hsi_combined_{suffix}.png"
    fig6.savefig(out_combined, dpi=300, bbox_inches="tight")
    plt.close(fig6)
    # ─────────────────────────────────────────────────────────────────────────────
    #  PRCC SUMMARY OUTPUT
    # ─────────────────────────────────────────────────────────────────────────────

    prcc_rows = []
    for idx, draw in enumerate(scenarios_of_interest):
        if scenario_names[draw] == "No disruptions":
            continue
        d_m = all_draws_annual_delayed_mean[idx]
        c_m = all_draws_annual_cancelled_mean[idx]
        prcc_rows.append({
            "draw": draw,
            "prop_delayed": d_m.mean(),
            "prop_cancelled": c_m.mean(),
        })

    pd.DataFrame(prcc_rows).to_csv(
        output_folder / "prcc_disruption_summary.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path("./resources"),
    )
