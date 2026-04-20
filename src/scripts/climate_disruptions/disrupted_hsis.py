"""
comparison_actual_vs_expected_disruption_realfacility.py
"""

import argparse
import string
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

from tlo import Date
from tlo.analysis.utils import extract_results

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL FONT SIZE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FS_TICK = 13
FS_LABEL = 15
FS_TITLE = 16
FS_LEGEND = 12
FS_PANEL = 17
FS_SUPTITLE = 14


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path):
    min_year = 2025
    max_year = 2041
    spacing_of_years = 1

    main_text = False
    parameter_uncertainty_analysis = False
    mode_2 = False
    climate_analysis = True
    prop_supply_demand = False
    top_n_hsi = 10

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

    facilities_df = pd.read_csv(
        resourcefilepath / "climate_change_impacts" / "facilities_with_lat_long_region.csv",
        low_memory=False,
    )[["Fname", "Dist", "Zonename", "Ftype"]].drop_duplicates(subset="Fname")
    dist_fixes = {"Blanytyre": "Blantyre", "Nkhatabay": "Nkhata Bay",
                  "Mzimba North": "Mzimba", "Mzimba South": "Mzimba"}
    facilities_df["Dist"] = facilities_df["Dist"].replace(dist_fixes)
    fac_to_district = facilities_df.set_index("Fname")["Dist"]
    fac_to_ftype = facilities_df.set_index("Fname")["Ftype"]
    fac_to_zone = facilities_df.set_index("Fname")["Zonename"]

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

    def _make_disrupted_persons_by_district(target_period, fac_to_dist):
        def _fn(_df):
            _df["date"] = pd.to_datetime(_df["date"])
            _df = _df.loc[_df["date"].between(*target_period)]
            if _df.empty or "Person_ID" not in _df.columns or "RealFacility_ID" not in _df.columns:
                return pd.Series(dtype=float)
            _df = _df[_df["RealFacility_ID"].notna() & (_df["RealFacility_ID"] != "unknown")].copy()
            _df["district"] = _df["RealFacility_ID"].map(fac_to_dist)
            _df = _df.dropna(subset=["district"])
            return _df.groupby("district")["Person_ID"].nunique().astype(float)
        return _fn

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

    def _align_and_rate(total_df, disrupted_df, delayed_df=None, cancelled_df=None):
        idx = total_df.index.union(disrupted_df.index)
        t = total_df.reindex(idx, fill_value=0)
        d = disrupted_df.reindex(idx, fill_value=0)
        if delayed_df is not None and cancelled_df is not None:
            t = t + delayed_df.reindex(idx, fill_value=0) + cancelled_df.reindex(idx, fill_value=0)
        return d.div(t).where(t > 0, 0.0).clip(upper=1.0)

    def _monthly_stats(rate_df):
        rate_df = rate_df.copy()
        monthly = rate_df.groupby(_parse_ym(rate_df.index)).mean().sort_index()
        return monthly.mean(axis=1), monthly.quantile(CI_LOWER, axis=1), monthly.quantile(CI_UPPER, axis=1)

    def _annual_stats(rate_df):
        rate_df = rate_df.copy()
        annual = rate_df.groupby(_parse_ym(rate_df.index).str[:4]).mean().sort_index()
        return annual.mean(axis=1), annual.quantile(CI_LOWER, axis=1), annual.quantile(CI_UPPER, axis=1)

    def _facility_stats(rate_df, total_df):
        rate_df = rate_df.copy()
        total_df = total_df.reindex(rate_df.index, fill_value=0)
        fac = _parse_facility(rate_df.index)
        return rate_df.groupby(fac).mean().mean(axis=1), total_df.groupby(fac).sum().mean(axis=1)

    def _hsi_type_stats(total_df, delayed_df, cancelled_df):
        hsi = _parse_hsi_type(total_df.index)
        total_by_type = total_df.groupby(hsi).sum()
        delayed_by_type = delayed_df.reindex(total_df.index, fill_value=0).groupby(hsi).sum()
        cancelled_by_type = cancelled_df.reindex(total_df.index, fill_value=0).groupby(hsi).sum()
        denom = total_by_type + delayed_by_type + cancelled_by_type
        delayed_rate = delayed_by_type.div(denom).where(denom > 0, 0.0)
        cancelled_rate = cancelled_by_type.div(denom).where(denom > 0, 0.0)
        return (delayed_rate.mean(axis=1), delayed_rate.quantile(CI_LOWER, axis=1),
                delayed_rate.quantile(CI_UPPER, axis=1), cancelled_rate.mean(axis=1),
                cancelled_rate.quantile(CI_LOWER, axis=1), cancelled_rate.quantile(CI_UPPER, axis=1),
                total_by_type.mean(axis=1))

    target_year_sequence = range(min_year, max_year, spacing_of_years)
    tlo_facilities = set()

    # ─────────────────────────────────────────────────────────────────────────
    #  PRE-LOAD ALL RAW RESULTS OUTSIDE THE DRAW LOOP
    #  Each extract_results call reads every draw/run on disk once.
    #  We then slice cheaply by draw index inside the loop below.
    # ─────────────────────────────────────────────────────────────────────────

    print("Pre-loading raw results for all years …")
    raw_total = {}
    raw_delayed = {}
    raw_cancelled = {}
    raw_delayed_persons = {}
    raw_cancelled_persons = {}

    for target_year in target_year_sequence:
        print(f"  year {target_year}")
        TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))

        raw_total[target_year] = extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="hsi_event_counts_by_facility_monthly",
            custom_generate_series=_make_hsi_counts_by_real_facility_monthly(TARGET_PERIOD),
            do_scaling=False,
        )
        raw_delayed[target_year] = extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="Weather_delayed_HSI_Event_full_info",
            custom_generate_series=_make_disrupted_by_real_facility_monthly(TARGET_PERIOD),
            do_scaling=False,
        )
        raw_cancelled[target_year] = extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="Weather_cancelled_HSI_Event_full_info",
            custom_generate_series=_make_disrupted_by_real_facility_monthly(TARGET_PERIOD),
            do_scaling=False,
        )
        raw_delayed_persons[target_year] = extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="Weather_delayed_HSI_Event_full_info",
            custom_generate_series=_make_disrupted_persons_by_district(TARGET_PERIOD, fac_to_district),
            do_scaling=False,
        )
        raw_cancelled_persons[target_year] = extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="Weather_cancelled_HSI_Event_full_info",
            custom_generate_series=_make_disrupted_persons_by_district(TARGET_PERIOD, fac_to_district),
            do_scaling=False,
        )

    print("Pre-loading complete. Processing draws …")

    # ─────────────────────────────────────────────────────────────────────────
    #  PER-DRAW PROCESSING  (slice from pre-loaded raw results — no I/O here)
    # ─────────────────────────────────────────────────────────────────────────

    all_draws_monthly_delayed_mean = [];
    all_draws_monthly_cancelled_mean = []
    all_draws_monthly_delayed_lower = [];
    all_draws_monthly_delayed_upper = []
    all_draws_monthly_cancelled_lower = [];
    all_draws_monthly_cancelled_upper = []
    all_draws_annual_delayed_mean = [];
    all_draws_annual_cancelled_mean = []
    all_draws_annual_delayed_lower = [];
    all_draws_annual_delayed_upper = []
    all_draws_annual_cancelled_lower = [];
    all_draws_annual_cancelled_upper = []
    all_draws_hsi_delayed_mean = [];
    all_draws_hsi_delayed_lower = [];
    all_draws_hsi_delayed_upper = []
    all_draws_hsi_cancelled_mean = [];
    all_draws_hsi_cancelled_lower = [];
    all_draws_hsi_cancelled_upper = []
    all_draws_hsi_total = []
    all_draws_total_df = {};
    all_draws_delayed_df = {};
    all_draws_cancelled_df = {}
    all_draws_disrupted_persons_by_district = {}

    def _concat_years(dfs):
        return pd.concat(dfs).groupby(level=0).sum()

    for draw in scenarios_of_interest:
        print(draw)

        if scenario_names[draw] == "No Disruptions":
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
            for lst in [all_draws_hsi_delayed_mean, all_draws_hsi_delayed_lower, all_draws_hsi_delayed_upper,
                        all_draws_hsi_cancelled_mean, all_draws_hsi_cancelled_lower,
                        all_draws_hsi_cancelled_upper, all_draws_hsi_total]:
                lst.append(empty)

            # Slice total from pre-loaded data
            _nd_total_dfs = [
                raw_total[yr][draw].fillna(0)
                for yr in target_year_sequence
            ]
            all_draws_total_df[draw] = _concat_years(_nd_total_dfs) * SCALING_FACTOR
            all_draws_delayed_df[draw] = pd.DataFrame(
                0.0,
                index=all_draws_total_df[draw].index,
                columns=all_draws_total_df[draw].columns,
            )
            all_draws_cancelled_df[draw] = pd.DataFrame(
                0.0,
                index=all_draws_total_df[draw].index,
                columns=all_draws_total_df[draw].columns,
            )
            tlo_facilities.update(_parse_facility(all_draws_total_df[draw].index).dropna())
            all_draws_disrupted_persons_by_district[draw] = None
            continue

        # ── Non-zero scenarios: slice each year from pre-loaded dicts ────────
        all_years_total_dfs = []
        all_years_delayed_dfs = []
        all_years_cancelled_dfs = []
        all_years_delayed_persons_dfs = []
        all_years_cancelled_persons_dfs = []

        for target_year in target_year_sequence:
            all_years_total_dfs.append(raw_total[target_year][draw].fillna(0))
            all_years_delayed_dfs.append(raw_delayed[target_year][draw].fillna(0))
            all_years_cancelled_dfs.append(raw_cancelled[target_year][draw].fillna(0))
            all_years_delayed_persons_dfs.append(raw_delayed_persons[target_year][draw].fillna(0))
            all_years_cancelled_persons_dfs.append(raw_cancelled_persons[target_year][draw].fillna(0))

        total_all = _concat_years(all_years_total_dfs) * SCALING_FACTOR
        delayed_all = _concat_years(all_years_delayed_dfs) * SCALING_FACTOR
        cancelled_all = _concat_years(all_years_cancelled_dfs) * SCALING_FACTOR
        disrupted_persons_all = (
            _concat_years(all_years_delayed_persons_dfs)
            .add(_concat_years(all_years_cancelled_persons_dfs), fill_value=0)
            * SCALING_FACTOR
        )

        all_draws_disrupted_persons_by_district[draw] = disrupted_persons_all
        tlo_facilities.update(_parse_facility(total_all.index).dropna())
        all_draws_total_df[draw] = total_all
        all_draws_delayed_df[draw] = delayed_all
        all_draws_cancelled_df[draw] = cancelled_all

        total_2 = _collapse_hsi_types(total_all)
        delayed_2 = _collapse_hsi_types(delayed_all)
        cancelled_2 = _collapse_hsi_types(cancelled_all)
        delayed_rate_2 = _align_and_rate(total_2, delayed_2, delayed_2, cancelled_2)
        cancelled_rate_2 = _align_and_rate(total_2, cancelled_2, delayed_2, cancelled_2)

        dm, dl, du = _monthly_stats(delayed_rate_2)
        cm, cl, cu = _monthly_stats(cancelled_rate_2)
        all_draws_monthly_delayed_mean.append(dm);
        all_draws_monthly_delayed_lower.append(dl)
        all_draws_monthly_delayed_upper.append(du);
        all_draws_monthly_cancelled_mean.append(cm)
        all_draws_monthly_cancelled_lower.append(cl);
        all_draws_monthly_cancelled_upper.append(cu)

        dam, dal, dau = _annual_stats(delayed_rate_2)
        cam, cal, cau = _annual_stats(cancelled_rate_2)
        all_draws_annual_delayed_mean.append(dam);
        all_draws_annual_delayed_lower.append(dal)
        all_draws_annual_delayed_upper.append(dau);
        all_draws_annual_cancelled_mean.append(cam)
        all_draws_annual_cancelled_lower.append(cal);
        all_draws_annual_cancelled_upper.append(cau)

        hd_m, hd_l, hd_u, hc_m, hc_l, hc_u, hd_tot = _hsi_type_stats(total_all, delayed_all, cancelled_all)
        all_draws_hsi_delayed_mean.append(hd_m);
        all_draws_hsi_delayed_lower.append(hd_l)
        all_draws_hsi_delayed_upper.append(hd_u);
        all_draws_hsi_cancelled_mean.append(hc_m)
        all_draws_hsi_cancelled_lower.append(hc_l);
        all_draws_hsi_cancelled_upper.append(hc_u)
        all_draws_hsi_total.append(hd_tot)

    # ─────────────────────────────────────────────────────────────────────────────
    #  RESOURCE FILE DISRUPTION OVERLAY
    # ─────────────────────────────────────────────────────────────────────────────

    def _load_rf(ssp_tag):
        df = pd.read_csv(
            resourcefilepath / "climate_change_impacts"
            / f"ResourceFile_Precipitation_Disruptions_{ssp_tag}_mean.csv"
        )
        df = df[df["RealFacility_ID"].isin(tlo_facilities)]
        df = df[(df["year"] >= min_year) & (df["year"] <= max_year - 1)]
        ann = df.groupby("year")["mean_all_service"].mean().reset_index().sort_values("year")
        ann["Date"] = pd.to_datetime(ann["year"].astype(str) + "-01-01")
        mo = df.groupby(["year", "month"])["mean_all_service"].mean().reset_index().sort_values(["year", "month"])
        mo["date"] = pd.to_datetime(
            mo["year"].astype(str) + "-" + mo["month"].astype(str).str.zfill(2) + "-01"
        )
        fac = df.groupby("RealFacility_ID")["mean_all_service"].mean().rename("rf_rate")
        return (ann["mean_all_service"].values * 100, ann["Date"].values,
                mo["mean_all_service"].values * 100, mo["date"].values, fac)

    rf_126_ann, rf_126_dates_ann, rf_126_mo, rf_126_dates_mo, rf_126_fac = _load_rf("ssp126")
    rf_585_ann, rf_585_dates_ann, rf_585_mo, rf_585_dates_mo, rf_585_fac = _load_rf("ssp585")
    rf_245_ann, rf_245_dates_ann, rf_245_mo, rf_245_dates_mo, rf_245_fac = _load_rf("ssp245")

    def _get_rf(scenario_name, monthly=True):
        ssp = ("ssp126" if "126" in scenario_name else "ssp585") if climate_analysis else "ssp245"
        lookup_mo = {"ssp126": (rf_126_dates_mo, rf_126_mo),
                     "ssp585": (rf_585_dates_mo, rf_585_mo),
                     "ssp245": (rf_245_dates_mo, rf_245_mo)}
        lookup_ann = {"ssp126": (rf_126_dates_ann, rf_126_ann),
                      "ssp585": (rf_585_dates_ann, rf_585_ann),
                      "ssp245": (rf_245_dates_ann, rf_245_ann)}
        return lookup_mo[ssp] if monthly else lookup_ann[ssp]

    def _get_rf_facility(scenario_name):
        if climate_analysis:
            return rf_126_fac if "126" in scenario_name else rf_585_fac
        return rf_245_fac

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT A: MONTHLY time series
    # ─────────────────────────────────────────────────────────────────────────────

    n_plots = len(scenarios_of_interest) - (1 if "No Disruptions" in scenario_names else 0)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 5 * n_rows),
        squeeze=False,
        sharey=True,
    )
    if climate_analysis:
        fig, axes = plt.subplots(2, 4, figsize=(7 * 4, 5 * 2), squeeze=False, sharey=True)

    axes_flat = axes.flatten()

    COLOUR_DELAYED = "#CC7000"
    COLOUR_CANCELLED = "#A8102E"
    COLOUR_TOTAL = "#7EBCE6"
    COLOUR_RF = "#75AE70"

    PANEL_LABELS = list(string.ascii_uppercase)

    def _series_to_dates_pct(s):
        if s.empty:
            return pd.DatetimeIndex([]), np.array([])
        return pd.to_datetime(s.index.astype(str) + "-01"), s.values * 100

    # ── Pre-compute global y-max across every non-zero scenario ─────────────
    global_ymax = 0.0
    for idx, draw in enumerate(scenarios_of_interest):
        if scenario_names[draw] == "No Disruptions":
            continue
        _, d_vals_pre = _series_to_dates_pct(all_draws_monthly_delayed_mean[idx])
        _, c_vals_pre = _series_to_dates_pct(all_draws_monthly_cancelled_mean[idx])
        if len(d_vals_pre) and len(c_vals_pre):
            global_ymax = max(global_ymax, (d_vals_pre + c_vals_pre).max())
        if not mode_2:
            _, _rf_vals_pre = _get_rf(scenario_names[draw], monthly=True)
            if len(_rf_vals_pre):
                global_ymax = max(global_ymax, _rf_vals_pre.max())
    global_ymax *= 1.05

    plot_idx = 0
    for idx, draw in enumerate(scenarios_of_interest):
        if scenario_names[draw] == "No Disruptions":
            continue
        ax = axes_flat[plot_idx]

        d_dates, d_vals = _series_to_dates_pct(all_draws_monthly_delayed_mean[idx])
        c_dates, c_vals = _series_to_dates_pct(all_draws_monthly_cancelled_mean[idx])
        t_vals = d_vals + c_vals if len(d_vals) == len(c_vals) else np.array([])

        if len(d_dates):
            ax.plot(d_dates, t_vals, color=COLOUR_TOTAL, lw=3, alpha=0.8, label="Total disrupted (TLO)")
        if len(d_dates):
            ax.plot(d_dates, d_vals, color=COLOUR_DELAYED, lw=1.5, alpha=0.6, label="Delayed (TLO)")
        if len(c_dates):
            ax.plot(c_dates, c_vals, color=COLOUR_CANCELLED, lw=1.5, alpha=0.6, label="Cancelled (TLO)")

        if not mode_2:
            _rf_dates_mo, _rf_vals_mo = _get_rf(scenario_names[draw], monthly=True)
            if len(_rf_vals_mo):
                ax.plot(_rf_dates_mo, _rf_vals_mo,
                        color=COLOUR_RF, lw=2.5, ls="--", alpha=0.9, label="DHIS2 ANC Data")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=FS_TICK)
        plt.setp(ax.yaxis.get_majorticklabels(), fontsize=FS_TICK)

        ax.set_xlabel("Year", fontsize=FS_LABEL, fontweight="bold")
        if plot_idx % n_cols == 0:
            ax.set_ylabel("% HSIs disrupted", fontsize=FS_LABEL, fontweight="bold")

        ax.set_title(scenario_names[draw], fontsize=FS_TITLE, fontweight="bold")
        ax.set_ylim(bottom=0, top=global_ymax)
        ax.set_xlim(left=pd.Timestamp("2025-01-01"), right=pd.Timestamp("2040-12-31"))
        ax.text(-0.07, 1.04, f"({PANEL_LABELS[plot_idx]})",
                transform=ax.transAxes, fontsize=FS_PANEL, fontweight="bold", va="bottom")

        if plot_idx == 0:
            ax.legend(fontsize=FS_LEGEND, framealpha=0.95, edgecolor="gray", fancybox=True)

        plot_idx += 1

    for j in range(plot_idx, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_folder / f"comparison_disruption_monthly_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT B: ANNUAL time series
    # ─────────────────────────────────────────────────────────────────────────────

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    SCENARIO_COLOURS = ["#ADB993", "#EDC7CF", "#6F8AB7"]
    for idx, draw in enumerate(scenarios_of_interest):
        col = SCENARIO_COLOURS[idx % len(SCENARIO_COLOURS)]
        d_m = all_draws_annual_delayed_mean[idx];
        c_m = all_draws_annual_cancelled_mean[idx]
        d_lo = all_draws_annual_delayed_lower[idx];
        d_hi = all_draws_annual_delayed_upper[idx]
        c_lo = all_draws_annual_cancelled_lower[idx];
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

    if not mode_2:
        seen_ssps = set()
        for _idx2, _draw2 in enumerate(scenarios_of_interest):
            _scen2 = scenario_names[_draw2]
            if _scen2 == "No Disruptions":
                continue
            _ssp2 = ("ssp126" if "126" in _scen2 else "ssp585") if climate_analysis else "ssp245"
            if _ssp2 not in seen_ssps:
                _rf_dates_ann, _rf_vals_ann = _get_rf(_scen2, monthly=False)
                _rf_col = COLOUR_RF if _ssp2 != "ssp585" else "#3d8c3d"
                _rf_label = f"ResourceFile {_ssp2.upper()}" if climate_analysis else "ResourceFile disruption rate"
                if len(_rf_vals_ann):
                    ax2.plot(_rf_dates_ann, _rf_vals_ann,
                             color=_rf_col, lw=2.5, ls="--", alpha=0.9, label=_rf_label)
                seen_ssps.add(_ssp2)

    style_handles = [
        mlines.Line2D([], [], color="grey", lw=2.5, ls="-", label="Total disrupted"),
        mlines.Line2D([], [], color="grey", lw=1, ls="--", alpha=0.7, label="Delayed"),
        mlines.Line2D([], [], color="grey", lw=1, ls=":", alpha=0.7, label="Cancelled"),
    ]
    ax2.legend(handles=style_handles, loc="upper right", fontsize=FS_LEGEND,
               framealpha=0.85, title="Line style", title_fontsize=FS_LEGEND)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=FS_TICK)
    plt.setp(ax2.yaxis.get_majorticklabels(), fontsize=FS_TICK)

    ax2.set_xlabel("Year", fontsize=FS_LABEL, fontweight="bold")
    ax2.set_ylabel("% HSIs disrupted", fontsize=FS_LABEL, fontweight="bold")
    ax2.set_title(
        f"Annual mean per-facility disruption rate by scenario ({min_year}–{max_year - 1})",
        fontsize=FS_TITLE, fontweight="bold",
    )
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=pd.Timestamp("2025-01-01"))
    ax2.legend(fontsize=FS_LEGEND, frameon=True, framealpha=0.9)

    fig2.tight_layout()
    fig2.savefig(output_folder / f"comparison_disruption_annual_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT B1: TOTAL HSI COUNT BAR CHART — prop_supply_demand mode
    # ─────────────────────────────────────────────────────────────────────────────

    if prop_supply_demand:
        supply_values = [0.1, 0.5, 0.9]
        default_draws = [0, 1, 2]
        worst_draws = [3, 4, 5]

        def _mean_ann_comp(idx, use_delayed):
            s = all_draws_annual_delayed_mean[idx] if use_delayed else all_draws_annual_cancelled_mean[idx]
            return s.mean() * 100

        def _ci_ann_comp(idx, use_delayed):
            lo = all_draws_annual_delayed_lower[idx] if use_delayed else all_draws_annual_cancelled_lower[idx]
            hi = all_draws_annual_delayed_upper[idx] if use_delayed else all_draws_annual_cancelled_upper[idx]
            mean_val = _mean_ann_comp(idx, use_delayed)
            return mean_val - lo.mean() * 100, hi.mean() * 100 - mean_val

        def _mean_annual_total(idx):
            d_m = all_draws_annual_delayed_mean[idx];
            c_m = all_draws_annual_cancelled_mean[idx]
            return (d_m + c_m.reindex(d_m.index, fill_value=0)).mean() * 100

        x = np.arange(len(supply_values));
        width = 0.35
        fig_b2, axes_b2 = plt.subplots(1, 2, figsize=(16, 5))
        ax_split = axes_b2[0]
        for draw_group, label, x_offset, col_d, col_c in [
            (default_draws, "Default", -width / 2, "#EDC7CF", "#CD657C"),
            (worst_draws, "Worst Case", +width / 2, "#6F8AB7", "#445D88"),
        ]:
            delayed_means = [_mean_ann_comp(i, True) for i in draw_group]
            cancelled_means = [_mean_ann_comp(i, False) for i in draw_group]
            total_means = [_mean_annual_total(i) for i in draw_group]
            ax_split.bar(x + x_offset, delayed_means, width, color=col_d, alpha=0.85,
                         label=f"{label} — Delayed")
            ax_split.bar(x + x_offset, cancelled_means, width, bottom=delayed_means,
                         color=col_c, alpha=0.85, label=f"{label} — Cancelled")
            ax_split.errorbar(x + x_offset, total_means,
                              yerr=np.array([_ci_ann_comp(i, True) for i in draw_group]).T,
                              fmt="none", color="black", lw=1.2, capsize=4)
        ax_split.set_xticks(x)
        ax_split.set_xticklabels([f"P({v})" for v in supply_values], fontsize=FS_TICK)
        plt.setp(ax_split.yaxis.get_majorticklabels(), fontsize=FS_TICK)
        ax_split.set_xlabel("Probability supply-side disrupted", fontsize=FS_LABEL, fontweight="bold")
        ax_split.set_ylabel("Mean annual % HSIs disrupted", fontsize=FS_LABEL, fontweight="bold")
        ax_split.legend(fontsize=FS_LEGEND, ncol=2)
        ax_split.set_ylim(bottom=0)
        axes_b2[1].set_visible(False)
        fig_b2.tight_layout()
        fig_b2.savefig(output_folder / f"supply_demand_split_and_volume_{suffix}.png",
                       dpi=300, bbox_inches="tight")
        plt.close(fig_b2)

    # ─────────────────────────────────────────────────────────────────────────────
    #  SHARED FIGURE SETUP for Plots C and C2
    # ─────────────────────────────────────────────────────────────────────────────

    n_scen_hsi = sum(1 for idx, draw in enumerate(scenarios_of_interest)
                     if scenario_names[draw] != "No Disruptions"
                     and not all_draws_hsi_delayed_mean[idx].empty)
    n_cols_hsi = min(3, n_scen_hsi)
    n_rows_hsi = (n_scen_hsi + n_cols_hsi - 1) // n_cols_hsi
    bar_height = max(0.3, min(0.7, 12 / max(top_n_hsi, 1)))
    fig_h = max(6, top_n_hsi * bar_height * 1.4)

    def _draw_hsi_bar_panel(ax, hd_m, hd_l, hd_u, hc_m, hc_l, hc_u, hsi_types, title, panel_label=None):
        hd_m = hd_m.reindex(hsi_types, fill_value=0);
        hd_l = hd_l.reindex(hsi_types, fill_value=0)
        hd_u = hd_u.reindex(hsi_types, fill_value=0);
        hc_m = hc_m.reindex(hsi_types, fill_value=0)
        hc_l = hc_l.reindex(hsi_types, fill_value=0);
        hc_u = hc_u.reindex(hsi_types, fill_value=0)
        y_pos = np.arange(len(hsi_types))
        total_m = (hd_m + hc_m).values * 100
        total_l = (hd_l + hc_l).values * 100
        total_u = (hd_u + hc_u).values * 100
        delayed_m = hd_m.values * 100
        canceld_m = hc_m.values * 100
        ax.barh(y_pos, delayed_m, height=bar_height, color=COLOUR_DELAYED, alpha=0.75, label="Delayed")
        ax.barh(y_pos, canceld_m, height=bar_height, left=delayed_m,
                color=COLOUR_CANCELLED, alpha=0.75, label="Cancelled")
        ax.errorbar(total_m, y_pos,
                    xerr=[total_m - total_l, total_u - total_m],
                    fmt="none", color="black", lw=1.0, capsize=2, alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(hsi_types, fontsize=FS_TICK)
        plt.setp(ax.xaxis.get_majorticklabels(), fontsize=FS_TICK)
        ax.invert_yaxis()
        ax.set_xlabel("% of HSIs disrupted", fontsize=FS_LABEL, fontweight="bold")
        title_str = f"({panel_label}) {title}" if panel_label else title
        ax.set_title(title_str, fontsize=FS_TITLE, fontweight="bold")
        ax.set_xlim(left=0)
        ax.legend(fontsize=FS_LEGEND, loc="lower right")

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT C and C2
    # ─────────────────────────────────────────────────────────────────────────────

    for fig_label, rate_key, fname_key in [
        ("most disrupted", "total", "by_hsi_type"),
        ("most cancelled", "cancelled", "top_cancelled_hsi_type"),
    ]:
        fig3, axes3 = plt.subplots(
            n_rows_hsi, n_cols_hsi,
            figsize=(9 * n_cols_hsi, fig_h),
            squeeze=False,
            sharex=True,
        )
        axes3_flat = axes3.flatten()
        plot_idx = 0

        for idx, draw in enumerate(scenarios_of_interest):
            if scenario_names[draw] == "No Disruptions":
                continue
            hd_m = all_draws_hsi_delayed_mean[idx];
            hd_l = all_draws_hsi_delayed_lower[idx]
            hd_u = all_draws_hsi_delayed_upper[idx];
            hc_m = all_draws_hsi_cancelled_mean[idx]
            hc_l = all_draws_hsi_cancelled_lower[idx];
            hc_u = all_draws_hsi_cancelled_upper[idx]
            if hd_m.empty and hc_m.empty:
                continue

            rate_s = (hd_m + hc_m).copy() if rate_key == "total" else hc_m.copy()
            rate_s = rate_s[rate_s.index.astype(str) != "nan"]
            rate_s = rate_s[rate_s > 0].sort_values(ascending=False)
            top_types = rate_s.head(top_n_hsi).index.tolist()

            ax = axes3_flat[plot_idx]
            _draw_hsi_bar_panel(ax, hd_m, hd_l, hd_u, hc_m, hc_l, hc_u, top_types,
                                scenario_names[draw], panel_label=PANEL_LABELS[plot_idx])
            plot_idx += 1

        for j in range(plot_idx, len(axes3_flat)):
            axes3_flat[j].set_visible(False)

        fig3.suptitle(
            f"Top {top_n_hsi} {fig_label} HSI types ({min_year}–{max_year - 1})\nwhiskers = 95% CI",
            fontsize=FS_SUPTITLE, fontweight="bold", y=1.01,
        )
        fig3.tight_layout()
        fig3.savefig(output_folder / f"comparison_disruption_{fname_key}_{suffix}.png",
                     dpi=300, bbox_inches="tight")
        plt.close(fig3)

    # ─────────────────────────────────────────────────────────────────────────────
    #  CSV OUTPUTS
    # ─────────────────────────────────────────────────────────────────────────────

    monthly_rows = []
    for idx, draw in enumerate(scenarios_of_interest):
        d_m = all_draws_monthly_delayed_mean[idx];
        d_lo = all_draws_monthly_delayed_lower[idx]
        d_hi = all_draws_monthly_delayed_upper[idx];
        c_m = all_draws_monthly_cancelled_mean[idx]
        c_lo = all_draws_monthly_cancelled_lower[idx];
        c_hi = all_draws_monthly_cancelled_upper[idx]
        for ym in d_m.index.union(c_m.index):
            monthly_rows.append({
                "Scenario": scenario_names[draw], "draw": draw, "year_month": ym,
                "delayed_rate_mean": d_m.get(ym, 0), "delayed_rate_lower": d_lo.get(ym, 0),
                "delayed_rate_upper": d_hi.get(ym, 0), "cancelled_rate_mean": c_m.get(ym, 0),
                "cancelled_rate_lower": c_lo.get(ym, 0), "cancelled_rate_upper": c_hi.get(ym, 0),
                "total_disruption_rate_mean": min(d_m.get(ym, 0) + c_m.get(ym, 0), 1.0),
                "total_disruption_rate_lower": min(d_lo.get(ym, 0) + c_lo.get(ym, 0), 1.0),
                "total_disruption_rate_upper": min(d_hi.get(ym, 0) + c_hi.get(ym, 0), 1.0),
            })
    pd.DataFrame(monthly_rows).to_csv(
        output_folder / f"monthly_disruption_rates_realfacilityid_{suffix}.csv", index=False)

    annual_rows = []
    for idx, draw in enumerate(scenarios_of_interest):
        d_m = all_draws_annual_delayed_mean[idx];
        d_lo = all_draws_annual_delayed_lower[idx]
        d_hi = all_draws_annual_delayed_upper[idx];
        c_m = all_draws_annual_cancelled_mean[idx]
        c_lo = all_draws_annual_cancelled_lower[idx];
        c_hi = all_draws_annual_cancelled_upper[idx]
        for yr in d_m.index.union(c_m.index):
            annual_rows.append({
                "Scenario": scenario_names[draw], "draw": draw, "year": yr,
                "delayed_rate_mean": d_m.get(yr, 0), "delayed_rate_lower": d_lo.get(yr, 0),
                "delayed_rate_upper": d_hi.get(yr, 0), "cancelled_rate_mean": c_m.get(yr, 0),
                "cancelled_rate_lower": c_lo.get(yr, 0), "cancelled_rate_upper": c_hi.get(yr, 0),
                "total_disruption_rate_mean": min(d_m.get(yr, 0) + c_m.get(yr, 0), 1.0),
                "total_disruption_rate_lower": min(d_lo.get(yr, 0) + c_lo.get(yr, 0), 1.0),
                "total_disruption_rate_upper": min(d_hi.get(yr, 0) + c_hi.get(yr, 0), 1.0),
            })
    pd.DataFrame(annual_rows).to_csv(
        output_folder / f"annual_disruption_rates_realfacilityid_{suffix}.csv", index=False)

    hsi_rows = []
    for idx, draw in enumerate(scenarios_of_interest):
        if scenario_names[draw] == "No Disruptions":
            continue
        hd_m = all_draws_hsi_delayed_mean[idx];
        hd_l = all_draws_hsi_delayed_lower[idx]
        hd_u = all_draws_hsi_delayed_upper[idx];
        hc_m = all_draws_hsi_cancelled_mean[idx]
        hc_l = all_draws_hsi_cancelled_lower[idx];
        hc_u = all_draws_hsi_cancelled_upper[idx]
        htot = all_draws_hsi_total[idx]
        for hsi in hd_m.index.union(hc_m.index):
            hsi_rows.append({
                "Scenario": scenario_names[draw], "draw": draw, "hsi_type": hsi,
                "mean_total_count": htot.get(hsi, 0),
                "delayed_rate_mean": hd_m.get(hsi, 0), "delayed_rate_lower": hd_l.get(hsi, 0),
                "delayed_rate_upper": hd_u.get(hsi, 0), "cancelled_rate_mean": hc_m.get(hsi, 0),
                "cancelled_rate_lower": hc_l.get(hsi, 0), "cancelled_rate_upper": hc_u.get(hsi, 0),
                "total_disruption_rate_mean": hd_m.get(hsi, 0) + hc_m.get(hsi, 0),
                "total_disruption_rate_lower": hd_l.get(hsi, 0) + hc_l.get(hsi, 0),
                "total_disruption_rate_upper": hd_u.get(hsi, 0) + hc_u.get(hsi, 0),
            })
    pd.DataFrame(hsi_rows).to_csv(output_folder / f"hsi_type_disruption_rates_{suffix}.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────────
    #  MAIN TEXT SUMMARY CSV
    # ─────────────────────────────────────────────────────────────────────────────

    summary_rows = []
    for idx, draw in enumerate(scenarios_of_interest):
        scen = scenario_names[draw]
        if scen == "No Disruptions":
            _nd = _collapse_hsi_types(all_draws_total_df[draw]).sum(axis=0)
            summary_rows.append({"Scenario": scen,
                                 "total_hsi_count_mean": round(_nd.mean(), 1),
                                 "total_hsi_count_lower": round(_nd.quantile(CI_LOWER), 1),
                                 "total_hsi_count_upper": round(_nd.quantile(CI_UPPER), 1),
                                 **{k: 0 for k in [
                                     "total_hsi_disrupted_mean", "total_hsi_disrupted_lower",
                                     "total_hsi_disrupted_upper",
                                     "total_hsi_cancelled_mean", "total_hsi_cancelled_lower",
                                     "total_hsi_cancelled_upper",
                                     "total_hsi_delayed_mean", "total_hsi_delayed_lower", "total_hsi_delayed_upper",
                                     "pct_disrupted_mean", "pct_disrupted_lower", "pct_disrupted_upper",
                                     "pct_cancelled_mean", "pct_cancelled_lower", "pct_cancelled_upper",
                                     "pct_delayed_mean", "pct_delayed_lower", "pct_delayed_upper",
                                     "annual_disruption_rate_mean_%", "annual_disruption_rate_lower_%",
                                     "annual_disruption_rate_upper_%", "annual_disruption_rate_max_%",
                                     "monthly_disruption_rate_mean_%", "monthly_disruption_rate_lower_%",
                                     "monthly_disruption_rate_upper_%", "monthly_disruption_rate_max_%",
                                     "monthly_disruption_rate_min_%", "monthly_disruption_rate_std_%",
                                 ]}})
            continue

        total_2 = _collapse_hsi_types(all_draws_total_df[draw])
        delayed_2 = _collapse_hsi_types(all_draws_delayed_df[draw]).reindex(total_2.index, fill_value=0)
        cancelled_2 = _collapse_hsi_types(all_draws_cancelled_df[draw]).reindex(total_2.index, fill_value=0)
        per_run_total = total_2.sum(axis=0)
        per_run_delayed = delayed_2.sum(axis=0)
        per_run_cancelled = cancelled_2.sum(axis=0)
        per_run_disrupted = per_run_delayed + per_run_cancelled
        denom = per_run_total + per_run_disrupted

        d_m_ann = all_draws_annual_delayed_mean[idx];
        c_m_ann = all_draws_annual_cancelled_mean[idx]
        d_lo_ann = all_draws_annual_delayed_lower[idx];
        d_hi_ann = all_draws_annual_delayed_upper[idx]
        c_lo_ann = all_draws_annual_cancelled_lower[idx];
        c_hi_ann = all_draws_annual_cancelled_upper[idx]
        annual_rate_mean = (d_m_ann + c_m_ann.reindex(d_m_ann.index, fill_value=0)) * 100
        annual_rate_lower = (d_lo_ann + c_lo_ann.reindex(d_lo_ann.index, fill_value=0)) * 100
        annual_rate_upper = (d_hi_ann + c_hi_ann.reindex(d_hi_ann.index, fill_value=0)) * 100

        d_m_mo = all_draws_monthly_delayed_mean[idx];
        c_m_mo = all_draws_monthly_cancelled_mean[idx]
        d_lo_mo = all_draws_monthly_delayed_lower[idx];
        d_hi_mo = all_draws_monthly_delayed_upper[idx]
        c_lo_mo = all_draws_monthly_cancelled_lower[idx];
        c_hi_mo = all_draws_monthly_cancelled_upper[idx]
        monthly_rate_mean = (d_m_mo + c_m_mo) * 100
        monthly_rate_lower = (d_lo_mo + c_lo_mo) * 100
        monthly_rate_upper = (d_hi_mo + c_hi_mo) * 100

        summary_rows.append({
            "Scenario": scen,
            "total_hsi_count_mean": round(per_run_total.mean(), 1),
            "total_hsi_count_lower": round(per_run_total.quantile(CI_LOWER), 1),
            "total_hsi_count_upper": round(per_run_total.quantile(CI_UPPER), 1),
            "total_hsi_disrupted_mean": round(per_run_disrupted.mean(), 1),
            "total_hsi_disrupted_lower": round(per_run_disrupted.quantile(CI_LOWER), 1),
            "total_hsi_disrupted_upper": round(per_run_disrupted.quantile(CI_UPPER), 1),
            "total_hsi_cancelled_mean": round(per_run_cancelled.mean(), 1),
            "total_hsi_cancelled_lower": round(per_run_cancelled.quantile(CI_LOWER), 1),
            "total_hsi_cancelled_upper": round(per_run_cancelled.quantile(CI_UPPER), 1),
            "total_hsi_delayed_mean": round(per_run_delayed.mean(), 1),
            "total_hsi_delayed_lower": round(per_run_delayed.quantile(CI_LOWER), 1),
            "total_hsi_delayed_upper": round(per_run_delayed.quantile(CI_UPPER), 1),
            "pct_disrupted_mean": round((per_run_disrupted / denom * 100).mean(), 4),
            "pct_disrupted_lower": round((per_run_disrupted / denom * 100).quantile(CI_LOWER), 4),
            "pct_disrupted_upper": round((per_run_disrupted / denom * 100).quantile(CI_UPPER), 4),
            "pct_cancelled_mean": round((per_run_cancelled / denom * 100).mean(), 4),
            "pct_cancelled_lower": round((per_run_cancelled / denom * 100).quantile(CI_LOWER), 4),
            "pct_cancelled_upper": round((per_run_cancelled / denom * 100).quantile(CI_UPPER), 4),
            "pct_delayed_mean": round((per_run_delayed / denom * 100).mean(), 4),
            "pct_delayed_lower": round((per_run_delayed / denom * 100).quantile(CI_LOWER), 4),
            "pct_delayed_upper": round((per_run_delayed / denom * 100).quantile(CI_UPPER), 4),
            "annual_disruption_rate_mean_%": round(annual_rate_mean.mean(), 4),
            "annual_disruption_rate_lower_%": round(annual_rate_lower.mean(), 4),
            "annual_disruption_rate_upper_%": round(annual_rate_upper.mean(), 4),
            "annual_disruption_rate_max_%": round(annual_rate_mean.max(), 4),
            "monthly_disruption_rate_mean_%": round(monthly_rate_mean.mean(), 4),
            "monthly_disruption_rate_lower_%": round(monthly_rate_lower.mean(), 4),
            "monthly_disruption_rate_upper_%": round(monthly_rate_upper.mean(), 4),
            "monthly_disruption_rate_max_%": round(monthly_rate_mean.max(), 4),
            "monthly_disruption_rate_min_%": round(monthly_rate_mean.min(), 4),
            "monthly_disruption_rate_std_%": round(monthly_rate_mean.std(), 4),
        })
    pd.DataFrame(summary_rows).to_csv(output_folder / f"main_text_summary_{suffix}.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────────
    #  DISRUPTED PERSONS BY DISTRICT CSV
    # ─────────────────────────────────────────────────────────────────────────────

    persons_rows = []
    for idx, draw in enumerate(scenarios_of_interest):
        scen = scenario_names[draw]
        dp = all_draws_disrupted_persons_by_district.get(draw)
        if dp is None or (hasattr(dp, "empty") and dp.empty):
            continue
        mean_by_district = dp.mean(axis=1)
        lower_by_district = dp.quantile(CI_LOWER, axis=1)
        upper_by_district = dp.quantile(CI_UPPER, axis=1)
        for district in mean_by_district.index:
            persons_rows.append({
                "Scenario": scen, "district": district,
                "disrupted_persons_mean": round(mean_by_district[district], 1),
                "disrupted_persons_lower": round(lower_by_district[district], 1),
                "disrupted_persons_upper": round(upper_by_district[district], 1),
            })
    pd.DataFrame(persons_rows).to_csv(
        output_folder / f"disrupted_persons_by_district_{suffix}.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────────
    #  PER-FACILITY COMPARISON: TLO vs SSP-matched ResourceFile
    # ─────────────────────────────────────────────────────────────────────────────

    n_scen = len(scenarios_of_interest) - (1 if "No Disruptions" in scenario_names else 0)
    fig4, axes4 = plt.subplots(1, n_scen, figsize=(7 * n_scen, 6), squeeze=False)
    axes4_flat = axes4.flatten()
    plot_idx = 0
    merged_all = pd.DataFrame()

    for idx, draw in enumerate(scenarios_of_interest):
        if scenario_names[draw] == "No Disruptions":
            continue
        rf_facility = _get_rf_facility(scenario_names[draw])
        total_2 = _collapse_hsi_types(all_draws_total_df[draw])
        delayed_2 = _collapse_hsi_types(all_draws_delayed_df[draw])
        cancelled_2 = _collapse_hsi_types(all_draws_cancelled_df[draw])
        delayed_rate_2 = _align_and_rate(total_2, delayed_2, delayed_2, cancelled_2)
        cancelled_rate_2 = _align_and_rate(total_2, cancelled_2, delayed_2, cancelled_2)
        total_rate_2 = delayed_rate_2.add(
            cancelled_rate_2.reindex(delayed_rate_2.index, fill_value=0), fill_value=0).clip(upper=1.0)
        tlo_rate, tlo_total = _facility_stats(total_rate_2, total_2)
        tlo_rate.name = f"tlo_rate_{scenario_names[draw]}"
        tlo_total.name = f"tlo_total_{scenario_names[draw]}"
        if merged_all.empty:
            merged_all = rf_facility.to_frame()
        merged_all = merged_all.join(tlo_rate, how="outer").join(tlo_total, how="outer")

        ax = axes4_flat[plot_idx]
        merged = pd.concat([tlo_rate.rename("tlo_rate"), rf_facility], axis=1).dropna()
        merged = merged[merged.index != "nan"]
        ax.scatter(merged["rf_rate"] * 100, merged["tlo_rate"] * 100,
                   alpha=0.5, s=20, color=SCENARIO_COLOURS[idx % len(SCENARIO_COLOURS)])
        max_val = max(merged["rf_rate"].max(), merged["tlo_rate"].max()) * 100
        ax.plot([0, max_val], [0, max_val], "k--", lw=1, alpha=0.5, label="1:1 line")
        plt.setp(ax.xaxis.get_majorticklabels(), fontsize=FS_TICK)
        plt.setp(ax.yaxis.get_majorticklabels(), fontsize=FS_TICK)
        ax.set_xlabel("ResourceFile disruption rate (%)", fontsize=FS_LABEL, fontweight="bold")
        ax.set_ylabel("TLO disruption rate (%)", fontsize=FS_LABEL, fontweight="bold")
        ax.set_title(f"({PANEL_LABELS[plot_idx]}) {scenario_names[draw]}", fontsize=FS_TITLE, fontweight="bold")
        ax.legend(fontsize=FS_LEGEND)
        plot_idx += 1

    fig4.suptitle("Per-facility disruption rate: TLO vs ResourceFile",
                  fontsize=FS_SUPTITLE, fontweight="bold")
    fig4.tight_layout()
    fig4.savefig(output_folder / f"comparison_disruption_per_facility_{suffix}.png",
                 dpi=300, bbox_inches="tight")
    plt.close(fig4)
    merged_all.to_csv(output_folder / f"per_facility_disruption_rates_{suffix}.csv")

    # ─────────────────────────────────────────────────────────────────────────────
    #  FACILITY-LEVEL SUMMARY CSV
    # ─────────────────────────────────────────────────────────────────────────────

    facility_summary_rows = []
    for idx, draw in enumerate(scenarios_of_interest):
        scen = scenario_names[draw]
        total_2 = _collapse_hsi_types(all_draws_total_df[draw])
        delayed_2 = _collapse_hsi_types(all_draws_delayed_df[draw])
        cancelled_2 = _collapse_hsi_types(all_draws_cancelled_df[draw])
        delayed_rate_2 = _align_and_rate(total_2, delayed_2, delayed_2, cancelled_2)
        cancelled_rate_2 = _align_and_rate(total_2, cancelled_2, delayed_2, cancelled_2)
        total_rate_2 = delayed_rate_2.add(
            cancelled_rate_2.reindex(delayed_rate_2.index, fill_value=0), fill_value=0).clip(upper=1.0)
        fac = _parse_facility(total_rate_2.index)
        per_fac_rate = total_rate_2.groupby(fac).mean()
        mean_across = per_fac_rate.mean(axis=0)
        max_per_run = total_rate_2.max(axis=0)
        facility_summary_rows.append({
            "Scenario": scen,
            "avg_facility_disruption_rate_mean_%": round(mean_across.mean() * 100, 4),
            "avg_facility_disruption_rate_lower_%": round(mean_across.quantile(CI_LOWER) * 100, 4),
            "avg_facility_disruption_rate_upper_%": round(mean_across.quantile(CI_UPPER) * 100, 4),
            "max_monthly_facility_rate_mean_%": round(max_per_run.mean() * 100, 4),
            "max_monthly_facility_rate_lower_%": round(max_per_run.quantile(CI_LOWER) * 100, 4),
            "max_monthly_facility_rate_upper_%": round(max_per_run.quantile(CI_UPPER) * 100, 4),
            "most_common_max_facility_month":
                total_rate_2.idxmax(axis=0).value_counts().index[0]
                if scen != "No Disruptions" else "N/A",
        })
    pd.DataFrame(facility_summary_rows).to_csv(
        output_folder / f"facility_level_disruption_summary_{suffix}.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────────
    #  MAP: Per-district HSI disruption rate
    # ─────────────────────────────────────────────────────────────────────────────

    malawi_admin2 = gpd.read_file(
        resourcefilepath / "mapping" / "ResourceFile_mwi_admbnda_adm2_nso_20181016.shp")
    for old, new in [("Blantyre City", "Blantyre"), ("Mzuzu City", "Mzuzu"),
                     ("Lilongwe City", "Lilongwe"), ("Zomba City", "Zomba")]:
        malawi_admin2["ADM2_EN"] = malawi_admin2["ADM2_EN"].replace(old, new)

    district_rates = {}
    for idx, draw in enumerate(scenarios_of_interest):
        scen = scenario_names[draw]
        if scen == "No Disruptions":
            continue
        total_2 = _collapse_hsi_types(all_draws_total_df[draw])
        delayed_2 = _collapse_hsi_types(all_draws_delayed_df[draw])
        cancelled_2 = _collapse_hsi_types(all_draws_cancelled_df[draw])
        delayed_rate_2 = _align_and_rate(total_2, delayed_2, delayed_2, cancelled_2)
        cancelled_rate_2 = _align_and_rate(total_2, cancelled_2, delayed_2, cancelled_2)
        total_rate_2 = delayed_rate_2.add(
            cancelled_rate_2.reindex(delayed_rate_2.index, fill_value=0), fill_value=0).clip(upper=1.0)
        fac = _parse_facility(total_rate_2.index)
        district = fac.map(fac_to_district)
        rate_mean = total_rate_2.mean(axis=1)
        volume = total_2.reindex(total_rate_2.index, fill_value=0).mean(axis=1)
        df_tmp = pd.DataFrame({
            "district": district.values,
            "rate": rate_mean.values,
            "volume": volume.values,
        }).dropna(subset=["district"])

        def _weighted_mean(g):
            return (g["rate"] * g["volume"]).sum() / g["volume"].sum() if g["volume"].sum() > 0 else 0

        district_rates[scen] = df_tmp.groupby("district").apply(_weighted_mean)

    district_rates_df = pd.DataFrame(district_rates) * 100

    if main_text:
        map_panels = [("Default", "Default"), ("Worst Case", "Worst Case")]
    elif prop_supply_demand:
        map_panels = [("Default Supply 0.5", "Default (supply 0.5)"),
                      ("Worst Case Supply 0.5", "Worst Case (supply 0.5)")]
    elif climate_analysis:
        map_panels = [("SSP126 Low Baseline", "SSP126 Low — Baseline"),
                      ("SSP585 High Worst", "SSP585 High — Worst Case")]
    else:
        map_panels = []
    map_panels = [(k, t) for k, t in map_panels if k in district_rates_df.columns]

    if map_panels:
        import matplotlib.gridspec as gridspec

        n_map_cols = len(map_panels)
        map_height = 8
        bar_height_in = 4
        fig_width = 6 * n_map_cols

        fig_map = plt.figure(figsize=(fig_width, bar_height_in + map_height))
        bar_frac = bar_height_in / (bar_height_in + map_height)
        map_frac = map_height / (bar_height_in + map_height)
        row_gap = 0.05

        gs = gridspec.GridSpec(
            1, n_map_cols, figure=fig_map,
            left=0.05, right=0.95, bottom=0.02, top=map_frac - row_gap, wspace=0.0,
        )
        axes_map = [fig_map.add_subplot(gs[0, i]) for i in range(n_map_cols)]

        bar_width_frac = 0.5
        bar_left = (1.0 - bar_width_frac) / 2.0
        bar_bottom = map_frac + row_gap
        bar_height_frac = bar_frac - row_gap - 0.05
        ax_bar = fig_map.add_axes([bar_left, bar_bottom, bar_width_frac, bar_height_frac])

        # ── Bar chart: difference in total HSIs vs No Disruptions ─────────────
        bar_scen_labels = []
        bar_means = []
        bar_lowers = []
        bar_uppers = []
        for idx, draw in enumerate(scenarios_of_interest):
            per_run_total = _collapse_hsi_types(all_draws_total_df[draw]).sum(axis=0)
            bar_scen_labels.append(scenario_names[draw])
            bar_means.append(per_run_total.mean())
            bar_lowers.append(per_run_total.quantile(CI_LOWER))
            bar_uppers.append(per_run_total.quantile(CI_UPPER))

        bar_means = np.array(bar_means)
        bar_lowers = np.array(bar_lowers)
        bar_uppers = np.array(bar_uppers)
        baseline_mean = bar_means[0]
        diff_means = bar_means - baseline_mean
        diff_lowers = bar_lowers - baseline_mean
        diff_uppers = bar_uppers - baseline_mean

        plot_labels = bar_scen_labels[1:]
        plot_means = diff_means[1:]
        plot_lowers = diff_lowers[1:]
        plot_uppers = diff_uppers[1:]
        plot_colours = [SCENARIO_COLOURS[i % len(SCENARIO_COLOURS)] for i in range(1, len(bar_scen_labels))]

        n_bars = len(plot_labels)
        x_pos = np.arange(n_bars)
        yerr_lo = plot_means - plot_lowers
        yerr_hi = plot_uppers - plot_means

        ax_bar.bar(
            x_pos, abs(plot_means),
            yerr=[abs(yerr_lo), abs(yerr_hi)],
            color=plot_colours, alpha=0.85,
            error_kw={"lw": 1.5, "capsize": 5, "capthick": 1.5, "ecolor": "black"},
            width=0.6,
        )
        ax_bar.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(plot_labels, fontsize=FS_TICK)
        ax_bar.set_ylabel('Defecit in HSIs\nvs. "No Disruptions"', fontsize=FS_LABEL, fontweight="bold")
        ax_bar.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
        ax_bar.yaxis.get_offset_text().set_fontsize(FS_TICK)
        plt.setp(ax_bar.yaxis.get_majorticklabels(), fontsize=FS_TICK)
        ax_bar.set_title("(A) ", fontsize=FS_TITLE, fontweight="bold", loc="left")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)

        # ── Map panels ──────────────────────────────────────────────────────────
        for i, (ax, (scen_key, title)) in enumerate(zip(axes_map, map_panels)):
            malawi_admin2["disruption_rate"] = malawi_admin2["ADM2_EN"].map(district_rates_df[scen_key])
            malawi_admin2.plot(
                column="disruption_rate", ax=ax, legend=True, cmap="Oranges",
                edgecolor="black", vmin=0, vmax=4.0,
                legend_kwds={"label": "% HSIs disrupted", "shrink": 0.8},
                missing_kwds={"color": "lightgrey", "label": "No data"},
            )
            cbar_ax = fig_map.axes[-1]
            cbar_ax.set_ylabel("% HSIs disrupted", fontsize=FS_LABEL, fontweight="bold")
            cbar_ax.tick_params(labelsize=FS_TICK)
            panel_letter = PANEL_LABELS[i + 1]
            ax.set_title(f"({panel_letter}) {title}", fontsize=FS_TITLE, fontweight="bold")
            ax.axis("off")

        fig_map.savefig(
            output_folder / f"map_hsi_disruption_rate_by_district_{suffix}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig_map)

    district_rates_df.to_csv(output_folder / f"district_hsi_disruption_percentage_{suffix}.csv")

    # ─────────────────────────────────────────────────────────────────────────────
    #  PLOT D: COMBINED FIGURE
    # ─────────────────────────────────────────────────────────────────────────────

    non_zero_draws = [(idx, draw) for idx, draw in enumerate(scenarios_of_interest)
                      if scenario_names[draw] != "No Disruptions"
                      and not all_draws_hsi_delayed_mean[idx].empty]

    n_col_d = len(non_zero_draws)
    fig6, axes6 = plt.subplots(
        2, n_col_d,
        figsize=(9 * n_col_d, fig_h * 2),
        squeeze=False,
        sharex="row",
    )
    panel_label_iter = iter(string.ascii_uppercase[:n_col_d * 2])

    for col, (idx, draw) in enumerate(non_zero_draws):
        hd_m = all_draws_hsi_delayed_mean[idx];
        hd_l = all_draws_hsi_delayed_lower[idx]
        hd_u = all_draws_hsi_delayed_upper[idx];
        hc_m = all_draws_hsi_cancelled_mean[idx]
        hc_l = all_draws_hsi_cancelled_lower[idx];
        hc_u = all_draws_hsi_cancelled_upper[idx]

        draw_total_rate = (hd_m + hc_m).copy()
        draw_total_rate = draw_total_rate[draw_total_rate.index.astype(str) != "nan"]
        top_disrupted = (draw_total_rate[draw_total_rate > 0]
                         .sort_values(ascending=False).head(top_n_hsi).index.tolist())
        lbl_top = next(panel_label_iter)
        _draw_hsi_bar_panel(axes6[0, col], hd_m, hd_l, hd_u, hc_m, hc_l, hc_u,
                            top_disrupted,
                            f"{scenario_names[draw]} — most disrupted",
                            panel_label=lbl_top)

        draw_cancelled_rate = hc_m.copy()
        draw_cancelled_rate = draw_cancelled_rate[draw_cancelled_rate.index.astype(str) != "nan"]
        top_cancelled = (draw_cancelled_rate[draw_cancelled_rate > 0]
                         .sort_values(ascending=False).head(top_n_hsi).index.tolist())
        lbl_bot = next(panel_label_iter)
        _draw_hsi_bar_panel(axes6[1, col], hd_m, hd_l, hd_u, hc_m, hc_l, hc_u,
                            top_cancelled,
                            f"{scenario_names[draw]} — most cancelled",
                            panel_label=lbl_bot)

    fig6.suptitle(
        f"Disruption by HSI type: top {top_n_hsi} most disrupted (top row) and most cancelled (bottom row)\n"
        f"{min_year}–{max_year - 1}  |  whiskers = 95% CI",
        fontsize=FS_SUPTITLE, fontweight="bold", y=1.01,
    )
    fig6.tight_layout()
    pd.DataFrame(top_cancelled).to_csv(results_folder / "top_cancelled.csv", index=False)
    pd.DataFrame(top_disrupted).to_csv(results_folder / "top_disrupted.csv", index=False)
    fig6.savefig(output_folder / f"comparison_disruption_hsi_combined_{suffix}.png",
                 dpi=300, bbox_inches="tight")
    plt.close(fig6)

    # ─────────────────────────────────────────────────────────────────────────────
    #  PRCC + PER-RUN CSVs
    # ─────────────────────────────────────────────────────────────────────────────

    prcc_rows = []
    for idx, draw in enumerate(scenarios_of_interest):
        if scenario_names[draw] == "No Disruptions":
            continue
        prcc_rows.append({"draw": draw,
                          "prop_delayed": all_draws_annual_delayed_mean[idx].mean(),
                          "prop_cancelled": all_draws_annual_cancelled_mean[idx].mean()})
    pd.DataFrame(prcc_rows).to_csv(output_folder / "prcc_disruption_summary.csv", index=False)

    all_per_run = []
    for idx, draw in enumerate(scenarios_of_interest):
        scen = scenario_names[draw]
        if scen == "No Disruptions":
            continue
        total_2 = _collapse_hsi_types(all_draws_total_df[draw])
        delayed_2 = _collapse_hsi_types(all_draws_delayed_df[draw]).reindex(total_2.index, fill_value=0)
        cancelled_2 = _collapse_hsi_types(all_draws_cancelled_df[draw]).reindex(total_2.index, fill_value=0)
        per_run_delayed = delayed_2.sum(axis=0)
        per_run_cancelled = cancelled_2.sum(axis=0)
        per_run_disrupted = per_run_delayed + per_run_cancelled
        df = pd.DataFrame({
            "scenario": scen,
            "run": delayed_2.columns,
            "total_delayed": per_run_delayed.values,
            "total_cancelled": per_run_cancelled.values,
            "total_disrupted": per_run_disrupted.values,
        })
        ci_lo = f"summary_lower_{int(CI_LOWER * 1000)}permil"
        ci_hi = f"summary_upper_{int(CI_UPPER * 1000)}permil"
        summary_pr = pd.DataFrame([
            {"scenario": scen, "run": "summary_mean",
             "total_delayed": per_run_delayed.mean(), "total_cancelled": per_run_cancelled.mean(),
             "total_disrupted": per_run_disrupted.mean()},
            {"scenario": scen, "run": ci_lo,
             "total_delayed": per_run_delayed.quantile(CI_LOWER),
             "total_cancelled": per_run_cancelled.quantile(CI_LOWER),
             "total_disrupted": per_run_disrupted.quantile(CI_LOWER)},
            {"scenario": scen, "run": ci_hi,
             "total_delayed": per_run_delayed.quantile(CI_UPPER),
             "total_cancelled": per_run_cancelled.quantile(CI_UPPER),
             "total_disrupted": per_run_disrupted.quantile(CI_UPPER)},
        ])
        all_per_run.append(pd.concat([df, summary_pr], ignore_index=True))
    pd.concat(all_per_run, ignore_index=True).to_csv(
        output_folder / f"per_run_hsi_counts_all_scenarios_{suffix}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()
    apply(results_folder=args.results_folder, output_folder=args.results_folder,
          resourcefilepath=Path("./resources"))
