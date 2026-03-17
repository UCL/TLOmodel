"""
comparison_actual_vs_expected_disruption_realfacility.py

Key fix: rates are computed per-run (disrupted_run / total_run) before averaging
across runs. This ensures rates are always bounded [0,1] and that numerator/denominator
are paired within the same run. Lower/upper bounds are derived from quantiles of
per-run rates rather than from count quantiles.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date
from tlo.analysis.utils import extract_results

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

results_folder = Path(
    "/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/"
    "climate_scenario_runs_lhs_param_scan-2026-03-10T163913Z"
)
output_folder = results_folder

min_year = 2025
max_year = 2031
spacing_of_years = 1

main_text = False
parameter_uncertainty_analysis = True

if parameter_uncertainty_analysis:
    scenario_names = list(range(0, 20))
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
                real_fac = str(key).split(":")[0]
                composite = f"{ym}:{real_fac}"
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
        _df["composite"] = (
            _df["date"].dt.strftime("%Y-%m") + ":" + _df["RealFacility_ID"].astype(str)
        )
        return _df["composite"].value_counts().astype(float)
    return _fn


def _extract_df(results_folder, draw, log_key, fn):
    """Returns per-run DataFrame (index: composite keys, columns: run indices).
    Does NOT average across runs."""
    try:
        raw = extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key=log_key,
            custom_generate_series=fn,
            do_scaling=False,
        )
        return raw[draw].fillna(0)
    except Exception as e:
        print(f"  Warning: extraction failed for {log_key} draw {draw}: {e}")
        return pd.DataFrame(dtype=float)


def _align_and_rate(total_df, disrupted_df):
    """Align indices and compute per-run rate = disrupted / total (0 where total=0).
    Returns a DataFrame of rates, same shape as inputs after alignment."""
    idx = total_df.index.union(disrupted_df.index)
    t = total_df.reindex(idx, fill_value=0)
    d = disrupted_df.reindex(idx, fill_value=0)
    return d.div(t).where(t > 0, 0.0)


def _monthly_stats(rate_df):
    """Group per-run rates by YYYY-MM, average across facilities per month per run,
    then return mean/lower/upper across runs."""
    rate_df = rate_df.copy()
    rate_df.index = rate_df.index.astype(str)
    ym = rate_df.index.str.split(":").str[0]
    monthly = rate_df.groupby(ym).mean().sort_index()
    return (
        monthly.mean(axis=1),
        monthly.quantile(0.025, axis=1),
        monthly.quantile(0.975, axis=1),
    )


def _annual_stats(rate_df):
    """Group per-run rates by year, average across facilities+months per year per run,
    then return mean/lower/upper across runs."""
    rate_df = rate_df.copy()
    rate_df.index = rate_df.index.astype(str)
    yr = rate_df.index.str.split(":").str[0].str[:4]
    annual = rate_df.groupby(yr).mean().sort_index()
    return (
        annual.mean(axis=1),
        annual.quantile(0.025, axis=1),
        annual.quantile(0.975, axis=1),
    )


def _facility_stats(rate_df, total_df):
    """Group per-run rates by facility, average across months per facility per run,
    then return mean across runs. Also returns mean total count per facility."""
    rate_df = rate_df.copy()
    total_df = total_df.reindex(rate_df.index, fill_value=0)
    rate_df.index = rate_df.index.astype(str)
    total_df.index = total_df.index.astype(str)
    fac = rate_df.index.str.split(":").str[1]
    fac_rate = rate_df.groupby(fac).mean().mean(axis=1)
    fac_total = total_df.groupby(fac).sum().mean(axis=1)
    return fac_rate, fac_total


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

all_draws_total_df = {}
all_draws_delayed_df = {}
all_draws_cancelled_df = {}

for draw in scenarios_of_interest:
    print(f"\n=== Draw {draw} ({scenario_names[draw]}) ===")

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
        continue

    all_years_total_dfs = []
    all_years_delayed_dfs = []
    all_years_cancelled_dfs = []

    for target_year in target_year_sequence:
        TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))
        print(f"  Year {target_year}...", end=" ", flush=True)

        fn_total = _make_hsi_counts_by_real_facility_monthly(TARGET_PERIOD)
        fn_disrupted = _make_disrupted_by_real_facility_monthly(TARGET_PERIOD)

        total_df = _extract_df(results_folder, draw, "hsi_event_counts_by_facility_monthly", fn_total)
        delayed_df = _extract_df(results_folder, draw, "Weather_delayed_HSI_Event_full_info", fn_disrupted)
        cancelled_df = _extract_df(results_folder, draw, "Weather_cancelled_HSI_Event_full_info", fn_disrupted)

        all_years_total_dfs.append(total_df)
        all_years_delayed_dfs.append(delayed_df)
        all_years_cancelled_dfs.append(cancelled_df)

        print(f"total={total_df.sum().mean():.0f}  "
              f"delayed={delayed_df.sum().mean():.0f}  "
              f"cancelled={cancelled_df.sum().mean():.0f}")


    def _concat_years(dfs):
        combined = pd.concat(dfs)
        return combined.groupby(level=0).sum()


    total_all = _concat_years(all_years_total_dfs)
    delayed_all = _concat_years(all_years_delayed_dfs)
    cancelled_all = _concat_years(all_years_cancelled_dfs)

    tlo_facilities.update(
        total_all.index.astype(str).str.split(":").str[1].dropna()
    )

    all_draws_total_df[draw] = total_all
    all_draws_delayed_df[draw] = delayed_all
    all_draws_cancelled_df[draw] = cancelled_all

    delayed_rate_df = _align_and_rate(total_all, delayed_all)
    cancelled_rate_df = _align_and_rate(total_all, cancelled_all)

    dm, dl, du = _monthly_stats(delayed_rate_df)
    cm, cl, cu = _monthly_stats(cancelled_rate_df)
    all_draws_monthly_delayed_mean.append(dm)
    all_draws_monthly_delayed_lower.append(dl)
    all_draws_monthly_delayed_upper.append(du)
    all_draws_monthly_cancelled_mean.append(cm)
    all_draws_monthly_cancelled_lower.append(cl)
    all_draws_monthly_cancelled_upper.append(cu)

    dam, dal, dau = _annual_stats(delayed_rate_df)
    cam, cal, cau = _annual_stats(cancelled_rate_df)
    all_draws_annual_delayed_mean.append(dam)
    all_draws_annual_delayed_lower.append(dal)
    all_draws_annual_delayed_upper.append(dau)
    all_draws_annual_cancelled_mean.append(cam)
    all_draws_annual_cancelled_lower.append(cal)
    all_draws_annual_cancelled_upper.append(cau)

    print(f"  >> Mean annual delayed rate:   {dam.mean():.4f}")
    print(f"  >> Mean annual cancelled rate: {cam.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
#  RESOURCE FILE DISRUPTION OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

disruptions_df = pd.read_csv(
    "/Users/rem76/PycharmProjects/TLOmodel/resources/climate_change_impacts/"
    "ResourceFile_Precipitation_Disruptions_ssp245_mean.csv"
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

print("\nPlotting monthly time series...")

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
    ax.legend(fontsize=9, framealpha=0.95, edgecolor="gray", fancybox=True)

for j in range(plot_idx, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle(
    f"Monthly mean per-facility disruption rate: TLO vs ResourceFile ({min_year}–{max_year - 1})\n",
    fontsize=12, fontweight="bold", y=1.01,
)
fig.tight_layout()
out_monthly = output_folder / f"comparison_disruption_monthly_{suffix}.png"
fig.savefig(out_monthly, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out_monthly}")

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
print(f"  Saved {out_annual}")

# ─────────────────────────────────────────────────────────────────────────────
#  CSV OUTPUTS
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

# ─────────────────────────────────────────────────────────────────────────────
#  PER-FACILITY COMPARISON: TLO vs ResourceFile
# ─────────────────────────────────────────────────────────────────────────────

rf_facility = (
    disruptions_df.groupby("RealFacility_ID")["mean_all_service"]
    .mean()
    .rename("rf_rate")
)

n_scen = len(scenarios_of_interest) - (1 if "No disruptions" in scenario_names else 0)
fig3, axes3 = plt.subplots(1, n_scen, figsize=(7 * n_scen, 6), squeeze=False)
axes3_flat = axes3.flatten()

plot_idx = 0
merged_all = rf_facility.to_frame()

for idx, draw in enumerate(scenarios_of_interest):
    if scenario_names[draw] == "No disruptions":
        continue

    delayed_rate_df = _align_and_rate(all_draws_total_df[draw], all_draws_delayed_df[draw])
    cancelled_rate_df = _align_and_rate(all_draws_total_df[draw], all_draws_cancelled_df[draw])
    total_rate_df = delayed_rate_df.add(
        cancelled_rate_df.reindex(delayed_rate_df.index, fill_value=0), fill_value=0
    )

    tlo_rate, tlo_total = _facility_stats(total_rate_df, all_draws_total_df[draw])
    tlo_rate.name = f"tlo_rate_{scenario_names[draw]}"
    tlo_total.name = f"tlo_total_{scenario_names[draw]}"
    merged_all = merged_all.join(tlo_rate, how="outer")
    merged_all = merged_all.join(tlo_total, how="outer")

    ax = axes3_flat[plot_idx]
    plot_idx += 1
    merged = pd.concat([tlo_rate.rename("tlo_rate"), rf_facility], axis=1).dropna()
    merged = merged[merged.index != "nan"]

    ax.scatter(merged["rf_rate"] * 100, merged["tlo_rate"] * 100,
               alpha=0.5, s=20, color=SCENARIO_COLOURS[idx % len(SCENARIO_COLOURS)]
               )
    max_val = max(merged["rf_rate"].max(), merged["tlo_rate"].max()) * 100
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, alpha=0.5, label="1:1 line")
    ax.set_xlabel("ResourceFile disruption rate (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("TLO disruption rate (%)", fontsize=11, fontweight="bold")
    ax.set_title(scenario_names[draw], fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

fig3.suptitle("Per-facility disruption rate: TLO vs ResourceFile",
              fontsize=12, fontweight="bold")
fig3.tight_layout()
out_facility = output_folder / f"comparison_disruption_per_facility_{suffix}.png"
fig3.savefig(out_facility, dpi=300, bbox_inches="tight")
plt.close(fig3)
print(f"  Saved {out_facility}")

merged_all.to_csv(output_folder / f"per_facility_disruption_rates_{suffix}.csv")

# ─────────────────────────────────────────────────────────────────────────────
#  PER-FACILITY RATES FOR A SPECIFIC MONTH
# ─────────────────────────────────────────────────────────────────────────────

target_ym = "2031-12"
draw = 1  # Baseline

total_all_df = all_draws_total_df[draw]
delayed_all_df = all_draws_delayed_df[draw]
cancelled_all_df = all_draws_cancelled_df[draw]

mask = total_all_df.index.astype(str).str.startswith(target_ym + ":")

total_m = total_all_df[mask]
delayed_m = delayed_all_df.reindex(total_m.index, fill_value=0)
cancelled_m = cancelled_all_df.reindex(total_m.index, fill_value=0)

delayed_rate_m = _align_and_rate(total_m, delayed_m)
cancelled_rate_m = _align_and_rate(total_m, cancelled_m)

df_fac = pd.DataFrame({
    "facility": total_m.index.astype(str).str.split(":").str[1],
    "total": total_m.mean(axis=1).values,
    "delayed_rate": delayed_rate_m.mean(axis=1).values,
    "cancelled_rate": cancelled_rate_m.mean(axis=1).values,
})
df_fac["total_rate"] = df_fac["delayed_rate"] + df_fac["cancelled_rate"]
df_fac = df_fac[df_fac["facility"] != "nan"]
df_fac = df_fac.sort_values("total_rate", ascending=False).reset_index(drop=True)

out_month = output_folder / f"per_facility_rates_{target_ym}_baseline.csv"
df_fac.to_csv(out_month, index=False)
print(df_fac.to_string())
print(f"\nSaved to {out_month}")

# ─────────────────────────────────────────────────────────────────────────────
#  DIAGNOSTIC: single facility check
# ─────────────────────────────────────────────────────────────────────────────

fac = "Chilinda"
total_fac = all_draws_total_df[1]
total_fac = total_fac[total_fac.index.astype(str).str.contains(fac)]
print("\n=== DENOMINATOR (ran) — per run ===")
print(total_fac.sort_index().head(24))

delayed_fac = all_draws_delayed_df[1]
delayed_fac = delayed_fac[delayed_fac.index.astype(str).str.contains(fac)]
print("=== NUMERATOR (disrupted) — per run ===")
print(delayed_fac.sort_index().head(24))

print("=== RF RATE ===")
print(disruptions_df[disruptions_df["RealFacility_ID"] == fac]
      [["year", "month", "mean_all_service"]].sort_values(["year", "month"]).head(24))

print("\nDone.")
