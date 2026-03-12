"""
comparison_actual_vs_expected_disruption_realfacility.py

Remake of the notebook `comparison_actual_vs_expected_disruption.ipynb`
using RealFacility_ID (facility name strings) as the join key throughout,
instead of the integer Facility_ID.

Logic
-----
For each draw and year:
  denominator : total HSIs at each RealFacility_ID from the monthly facility log
                (key format in log: "RealFacility_ID:TREATMENT_ID")
  numerator   : weather-delayed / cancelled counts per RealFacility_ID
                (RealFacility_ID column in the full_info row-level logs)

  rate_i = disrupted_i / total_i  for each facility i
  monthly / annual mean = unweighted mean(rate_i)

The ANC overlay uses the ResourceFile disruptions keyed by RealFacility_ID,
so the merge is direct — no fuzzy facility-name matching required.
"""

from collections import defaultdict
from pathlib import Path
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date
from tlo.analysis.utils import extract_results, summarize

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  (edit these)
# ─────────────────────────────────────────────────────────────────────────────

results_folder = Path(
    "/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/"
    "baseline_run_with_pop-2026-03-05T101702Z"
)
output_folder = results_folder

min_year = 2025
max_year = 2041
spacing_of_years = 1

main_text = True
parameter_uncertainty_analysis = False

if parameter_uncertainty_analysis:
    scenario_names = list(range(0, 200))
    scenarios_of_interest = scenario_names
    suffix = "parameter_UA"
if main_text:
    scenario_names = ["No disruptions", "Baseline", "Worst Case"]
    scenarios_of_interest = [0, 1, 2]
    suffix = "main_text"

# ANC data paths
SSP = "ssp245"
MODEL = "mean"
SERVICE = "ANC"

ANC_PREDICTIONS_PATH = (
    f"/Users/rem76/Desktop/Climate_Change_Health/Data/"
    f"weather_predictions_with_X_{SSP}_{MODEL}_{SERVICE}.csv"
)
DISRUPTIONS_RESOURCE_PATH = (
    "/Users/rem76/PycharmProjects/TLOmodel/resources/climate_change_impacts/"
    f"ResourceFile_Precipitation_Disruptions_{SSP}_{MODEL}.csv"
)

# ─────────────────────────────────────────────────────────────────────────────
#  FACILITY-LEVEL HELPER: map Facility_Type → TLO level
# ─────────────────────────────────────────────────────────────────────────────

_FACILITY_LEVELS_TYPES = {
    "level_0": ["Health Post", "Village Health Committee", "Community Health Station",
                "Village Clinic", "Mobile Clinic", "Outreach Clinic", "Dispensary"],
    "level_1a": ["Health Centre", "Rural Health Centre", "Urban Health Centre", "Clinic",
                 "Private Clinic", "Special Clinic", "Antenatal Clinic", "Maternity Clinic",
                 "Maternity Facility", "Medical Centre"],
    "level_1b": ["Community Hospital", "Rural Hospital", "CHAM Hospital",
                 "Rural/Community Hospital", "Mission Hospital"],
    "level_2": ["District Hospital", "District Health Office"],
    "level_3": ["Kamuzu Central Hospital", "Mzuzu Central Hospital", "Zomba Central Hospital",
                "Queen Elizabeth Central Hospital", "Central Hospital"],
    "level_4": ["Zomba Mental Hospital", "Mental Hospital", "Psychiatric Hospital"],
}
_FTYPE_TO_LEVEL = {ft: lv for lv, fts in _FACILITY_LEVELS_TYPES.items() for ft in fts}


def map_facility_to_level(ftype):
    if pd.isna(ftype):
        return None
    if ftype in _FTYPE_TO_LEVEL:
        return _FTYPE_TO_LEVEL[ftype]
    fc = re.sub(r"\s+", " ", ftype.lower().strip())
    if any(t in fc for t in ["central hospital", "referral hospital"]):
        return "level_3"
    if any(t in fc for t in ["mental hospital", "psychiatric"]):
        return "level_4"
    if any(t in fc for t in ["district hospital", "district health"]):
        return "level_2"
    if "hospital" in fc:
        return "level_1b"
    if any(t in fc for t in ["health centre", "health center"]):
        return "level_1a"
    if "clinic" in fc:
        return "level_0" if any(t in fc for t in ["village", "mobile", "outreach"]) else "level_1a"
    if any(t in fc for t in ["dispensary", "health post", "village", "mobile",
                             "outreach", "community health station"]):
        return "level_0"
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACTION HELPERS  (all keyed by RealFacility_ID)
# ─────────────────────────────────────────────────────────────────────────────

def _make_hsi_counts_by_real_facility_monthly(target_period):
    """
    Returns a custom_generate_series function that reads the
    hsi_event_counts_by_facility_monthly log and produces a Series indexed by
    'YYYY-MM:RealFacility_ID' with total HSI counts.

    The log stores one row per end-of-month flush; the 'counts' column is a dict
    whose keys are already in the form 'RealFacility_ID:TREATMENT_ID'.
    """

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
                # key = "RealFacility_ID:TREATMENT_ID"
                real_fac = str(key).split(":")[0]
                composite = f"{ym}:{real_fac}"
                totals[composite] = totals.get(composite, 0) + val
        return pd.Series(totals, dtype=float)

    return _fn


def _make_disrupted_by_real_facility_monthly(target_period):
    """
    Returns a custom_generate_series function for the
    Weather_delayed/cancelled_HSI_Event_full_info row-level logs.

    Each row is one disrupted event and already carries a RealFacility_ID column.
    Returns a Series indexed by 'YYYY-MM:RealFacility_ID'.
    """

    def _fn(_df):
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df.loc[_df["date"].between(*target_period)]
        if _df.empty or "RealFacility_ID" not in _df.columns:
            return pd.Series(dtype=float)
        _df = _df[_df["RealFacility_ID"].notna() & (_df["RealFacility_ID"] != "unknown")].copy()
        _df["composite"] = _df["date"].dt.strftime("%Y-%m") + ":" + _df["RealFacility_ID"].astype(str)
        return _df["composite"].value_counts().astype(float)

    return _fn


def _extract_mean_series(results_folder, draw, log_key, fn):
    """Run extract_results + average across runs for a given draw."""
    try:
        raw = extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key=log_key,
            custom_generate_series=fn,
            do_scaling=False,
        )
        return raw[draw].fillna(0).mean(axis=1)
    except Exception as e:
        print(f"  Warning: extraction failed for {log_key} draw {draw}: {e}")
        return pd.Series(dtype=float)


def compute_monthly_rates(total_s, disrupted_s):
    """
    Given two Series indexed by 'YYYY-MM:RealFacility_ID', compute for each month
    the unweighted mean per-facility disruption rate:
        rate_i = disrupted_i / total_i   (only where total_i > 0)
        monthly_mean_rate = mean(rate_i) across facilities in that month

    Returns a pd.Series indexed by 'YYYY-MM'.
    """
    df = pd.concat([total_s.rename("total"), disrupted_s.rename("disrupted")], axis=1).fillna(0)
    df.index = df.index.astype(str)
    df = df[df["total"] > 0]
    if df.empty:
        return pd.Series(dtype=float)
    df["ym"] = df.index.str.split(":").str[0]
    df["rate"] = df["disrupted"] / df["total"]
    return df.groupby("ym")["rate"].mean().sort_index()


def compute_annual_rates(total_s, disrupted_s):
    """
    Same as compute_monthly_rates but groups by year ('YYYY').
    """
    df = pd.concat([total_s.rename("total"), disrupted_s.rename("disrupted")], axis=1).fillna(0)
    df.index = df.index.astype(str)
    df = df[df["total"] > 0]
    if df.empty:
        return pd.Series(dtype=float)
    df["yr"] = df.index.str.split(":").str[0].str[:4]
    df["rate"] = df["disrupted"] / df["total"]
    return df.groupby("yr")["rate"].mean().sort_index()


# ─────────────────────────────────────────────────────────────────────────────
#  ANC DATA LOADER  (keyed by RealFacility_ID via disruptions ResourceFile)
# ─────────────────────────────────────────────────────────────────────────────

def load_anc_disruption_rate_monthly(min_year, max_year):
    """
    Load the ANC facility-level predictions and the disruption ResourceFile,
    merge on RealFacility_ID (the facility name string used throughout TLO),
    and return a monthly mean disruption-rate Series indexed by pd.Timestamp.

    disruption_rate_i = |Difference_in_Expectation_i| / Predicted_No_Weather_i
    monthly_mean      = unweighted mean across all facilities that month
    """
    try:
        facility_df = pd.read_csv(ANC_PREDICTIONS_PATH)
        disruptions_df = pd.read_csv(DISRUPTIONS_RESOURCE_PATH).rename(
            columns={"Unnamed: 0": "Index"}
        )

        # The disruptions file is keyed by RealFacility_ID (facility name string).
        # Merge directly — no integer-ID translation required.
        facility_df = facility_df.merge(
            disruptions_df[["RealFacility_ID", "year", "month", "disruption"]],
            left_on=["Facility_ID", "Year", "Month"],  # Facility_ID here is the name string
            right_on=["RealFacility_ID", "year", "month"],
            how="left",
        )
        # Zero out difference where no disruption was modelled
        facility_df.loc[facility_df["disruption"] == 0, "Difference_in_Expectation"] = 0
        facility_df = facility_df.drop(columns=["disruption", "RealFacility_ID",
                                                "year", "month"], errors="ignore")

        # Map facility type → TLO level for grouping
        facility_df["Level"] = facility_df["Facility_Type"].apply(map_facility_to_level)
        facility_df = facility_df.dropna(subset=["District", "Level"])

        # Filter to simulation period
        facility_df = facility_df[
            ((facility_df["Year"] == min_year) & (facility_df["Month"] >= 1)) |
            ((facility_df["Year"] > min_year) & (facility_df["Year"] < max_year)) |
            ((facility_df["Year"] == max_year) & (facility_df["Month"] <= 12))
            ]

        # Per-facility-month disruption rate
        facility_df["disruption_rate"] = (
            facility_df["Difference_in_Expectation"].abs()
            / facility_df["Predicted_No_Weather_Model"].replace(0, np.nan)
        )

        # Monthly mean across all facilities  (unweighted — matches TLO metric)
        avg = (
            facility_df.groupby(["Year", "Month"])["disruption_rate"]
            .mean()
            .reset_index()
            .sort_values(["Year", "Month"])
        )
        avg["Date"] = pd.to_datetime(
            avg["Year"].astype(str) + "-" + avg["Month"].astype(str).str.zfill(2) + "-01"
        )
        return avg.set_index("Date")["disruption_rate"]

    except Exception as e:
        print(f"  Warning: could not load ANC data: {e}")
        return pd.Series(dtype=float)


def load_anc_disruption_rate_annual(min_year, max_year):
    """Annual version of the ANC loader."""
    monthly = load_anc_disruption_rate_monthly(min_year, max_year)
    if monthly.empty:
        return pd.Series(dtype=float)
    return monthly.resample("YS").mean()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN EXTRACTION LOOP
# ─────────────────────────────────────────────────────────────────────────────
target_year_sequence = range(min_year, max_year, spacing_of_years)

all_draws_monthly_delayed_rate = []
all_draws_monthly_cancelled_rate = []
all_draws_annual_delayed_rate = []
all_draws_annual_cancelled_rate = []

# CI storage — explicit mean/lower/upper, matching pattern from DALY script
all_draws_annual_delayed_lower = []
all_draws_annual_delayed_upper = []
all_draws_annual_cancelled_lower = []
all_draws_annual_cancelled_upper = []

for draw in scenarios_of_interest:
    print(f"\n=== Draw {draw} ({scenario_names[draw]}) ===")

    if scenario_names[draw] == "No disruptions":
        all_months = pd.date_range(
            start=f"{min_year}-01-01", end=f"{max_year - 1}-12-01", freq="MS"
        ).strftime("%Y-%m")
        all_years_str = [str(y) for y in target_year_sequence]
        all_draws_monthly_delayed_rate.append(pd.Series(0.0, index=all_months))
        all_draws_monthly_cancelled_rate.append(pd.Series(0.0, index=all_months))
        all_draws_annual_delayed_rate.append(pd.Series(0.0, index=all_years_str))
        all_draws_annual_cancelled_rate.append(pd.Series(0.0, index=all_years_str))
        all_draws_annual_delayed_lower.append(pd.Series(0.0, index=all_years_str))
        all_draws_annual_delayed_upper.append(pd.Series(0.0, index=all_years_str))
        all_draws_annual_cancelled_lower.append(pd.Series(0.0, index=all_years_str))
        all_draws_annual_cancelled_upper.append(pd.Series(0.0, index=all_years_str))
        continue

    monthly_total_parts = []
    monthly_delayed_parts = []
    monthly_cancelled_parts = []

    # Explicit mean/lower/upper accumulators — same pattern as DALY script
    all_years_delayed_mean = {}
    all_years_delayed_lower = {}
    all_years_delayed_upper = {}
    all_years_cancelled_mean = {}
    all_years_cancelled_lower = {}
    all_years_cancelled_upper = {}
    all_years_total_mean = {}

    for target_year in target_year_sequence:
        TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))
        print(f"  Year {target_year}...", end=" ", flush=True)

        fn_total = _make_hsi_counts_by_real_facility_monthly(TARGET_PERIOD)
        fn_disrupted = _make_disrupted_by_real_facility_monthly(TARGET_PERIOD)

        # Total HSIs — mean only (denominator, no CI needed)
        total_s = _extract_mean_series(
            results_folder, draw, "hsi_event_counts_by_facility_monthly", fn_total
        )

        # Delayed — extract mean/lower/upper via summarize
        result_delayed = summarize(
            extract_results(
                results_folder,
                module="tlo.methods.healthsystem.summary",
                key="Weather_delayed_HSI_Event_full_info",
                custom_generate_series=fn_disrupted,
                do_scaling=False,
            ),
            only_mean=False,
            collapse_columns=True,
        )[draw]

        # Cancelled — extract mean/lower/upper via summarize
        result_cancelled = summarize(
            extract_results(
                results_folder,
                module="tlo.methods.healthsystem.summary",
                key="Weather_cancelled_HSI_Event_full_info",
                custom_generate_series=fn_disrupted,
                do_scaling=False,
            ),
            only_mean=False,
            collapse_columns=True,
        )[draw]

        all_years_total_mean[target_year] = total_s
        all_years_delayed_mean[target_year] = result_delayed["mean"].fillna(0)
        all_years_delayed_lower[target_year] = result_delayed["lower"].fillna(0)
        all_years_delayed_upper[target_year] = result_delayed["upper"].fillna(0)
        all_years_cancelled_mean[target_year] = result_cancelled["mean"].fillna(0)
        all_years_cancelled_lower[target_year] = result_cancelled["lower"].fillna(0)
        all_years_cancelled_upper[target_year] = result_cancelled["upper"].fillna(0)

        print(f"total={total_s.sum():.0f}  "
              f"delayed={result_delayed['mean'].sum():.0f}  "
              f"cancelled={result_cancelled['mean'].sum():.0f}")

    # Concatenate across years
    total_all = pd.concat(all_years_total_mean.values()).groupby(level=0).sum()
    delayed_mean_all = pd.concat(all_years_delayed_mean.values()).groupby(level=0).sum()
    delayed_lower_all = pd.concat(all_years_delayed_lower.values()).groupby(level=0).sum()
    delayed_upper_all = pd.concat(all_years_delayed_upper.values()).groupby(level=0).sum()
    cancelled_mean_all = pd.concat(all_years_cancelled_mean.values()).groupby(level=0).sum()
    cancelled_lower_all = pd.concat(all_years_cancelled_lower.values()).groupby(level=0).sum()
    cancelled_upper_all = pd.concat(all_years_cancelled_upper.values()).groupby(level=0).sum()

    # Compute rates — mean, lower, upper each get their own rate Series
    all_draws_annual_delayed_rate.append(compute_annual_rates(total_all, delayed_mean_all))
    all_draws_annual_delayed_lower.append(compute_annual_rates(total_all, delayed_lower_all))
    all_draws_annual_delayed_upper.append(compute_annual_rates(total_all, delayed_upper_all))

    all_draws_annual_cancelled_rate.append(compute_annual_rates(total_all, cancelled_mean_all))
    all_draws_annual_cancelled_lower.append(compute_annual_rates(total_all, cancelled_lower_all))
    all_draws_annual_cancelled_upper.append(compute_annual_rates(total_all, cancelled_upper_all))

    # Monthly mean (unchanged — CIs on annual plot only)
    all_draws_monthly_delayed_rate.append(compute_monthly_rates(total_all, delayed_mean_all))
    all_draws_monthly_cancelled_rate.append(compute_monthly_rates(total_all, cancelled_mean_all))

    print(f"  >> Mean annual delayed rate:   {all_draws_annual_delayed_rate[-1].mean():.4f}")
    print(f"  >> Mean annual cancelled rate: {all_draws_annual_cancelled_rate[-1].mean():.4f}")
# ─────────────────────────────────────────────────────────────────────────────
#  ANC REFERENCE DATA
# ─────────────────────────────────────────────────────────────────────────────

print("\nLoading ANC disruption data...")
anc_monthly = load_anc_disruption_rate_monthly(min_year, max_year - 1)
anc_annual = load_anc_disruption_rate_annual(min_year, max_year - 1)
print(f"  ANC monthly mean disruption rate: {anc_monthly.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT A: MONTHLY time series — one subplot per scenario
# ─────────────────────────────────────────────────────────────────────────────

print("\nPlotting monthly time series...")

n_plots = len(scenarios_of_interest)
n_cols = min(3, n_plots)
n_rows = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
axes_flat = axes.flatten()

COLOUR_DELAYED = "#CC7000"
COLOUR_CANCELLED = "#A8102E"
COLOUR_TOTAL = "#1E3A8A"
COLOUR_ANC = "#0A8C0F"


def _series_to_dates_pct(s):
    """Convert a 'YYYY-MM'-indexed Series to (DatetimeIndex, array_pct)."""
    if s.empty:
        return pd.DatetimeIndex([]), np.array([])
    dates = pd.to_datetime(s.index.astype(str) + "-01")
    return dates, s.values * 100


for idx, draw in enumerate(scenarios_of_interest):
    ax = axes_flat[idx]

    d_dates, d_vals = _series_to_dates_pct(all_draws_monthly_delayed_rate[idx])
    c_dates, c_vals = _series_to_dates_pct(all_draws_monthly_cancelled_rate[idx])

    t_vals = d_vals + c_vals if len(d_vals) == len(c_vals) else np.array([])
    t_dates = d_dates

    if len(d_dates):
        ax.plot(d_dates, d_vals, color=COLOUR_DELAYED, lw=1.5, alpha=0.6,
                label="Delayed (mean across facilities)")
    if len(c_dates):
        ax.plot(c_dates, c_vals, color=COLOUR_CANCELLED, lw=1.5, alpha=0.6,
                label="Cancelled (mean across facilities)")
    if len(t_dates):
        ax.plot(t_dates, t_vals, color=COLOUR_TOTAL, lw=3, alpha=0.9,
                label="Total disrupted (TLO)")

    # ANC overlay — same series for all non-disruption draws
    if not anc_monthly.empty and scenario_names[draw] != "No disruptions":
        ax.plot(
            anc_monthly.index, anc_monthly.values * 100,
            color=COLOUR_ANC, lw=2.5, ls="--",
            alpha=0.9, label="ANC disruptions (DHIS2 model, RealFacility_ID)",
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Month", fontsize=11, fontweight="bold")
    ax.set_ylabel("% disrupted (mean across facilities)", fontsize=11, fontweight="bold")
    ax.set_title(f"{scenario_names[draw]}", fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, ls=":", lw=0.5)
    ax.set_facecolor("#F8F9FA")
    ax.legend(fontsize=9, framealpha=0.95, edgecolor="gray", fancybox=True)

for j in range(n_plots, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle(
    f"Monthly mean per-facility disruption rate: TLO vs DHIS2 ANC  ({min_year}–{max_year - 1})\n"
    "Rate = disrupted HSIs / total HSIs, averaged across facilities (RealFacility_ID)",
    fontsize=12, fontweight="bold", y=1.01,
)
fig.tight_layout()
out_monthly = output_folder / f"comparison_disruption_monthly_{suffix}.png"
fig.savefig(out_monthly, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out_monthly}")

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT B: ANNUAL time series — all scenarios on one panel
# ─────────────────────────────────────────────────────────────────────────────

print("Plotting annual time series...")

fig2, ax2 = plt.subplots(figsize=(10, 5))

SCENARIO_COLOURS = [
    "#823038",  # Baseline
    "#00566f",  # Best case / SSP low
    "#c65a52",  # Worst case / SSP high
    "#5b3f8c", "#8e7cc3", "#c7b7ec",
    "#0081a7", "#5ab4c6", "#f07167", "#f59e96",
]
for idx, draw in enumerate(scenarios_of_interest):
    col = SCENARIO_COLOURS[idx % len(SCENARIO_COLOURS)]
    d_s = all_draws_annual_delayed_rate[idx]
    c_s = all_draws_annual_cancelled_rate[idx]
    print(f"idx={idx}  draw={draw}  name={scenario_names[draw]}  "
          f"total_mean={(d_s + c_s.reindex(d_s.index, fill_value=0)).values * 100}")
    d_lo = all_draws_annual_delayed_lower[idx]
    d_hi = all_draws_annual_delayed_upper[idx]
    c_lo = all_draws_annual_cancelled_lower[idx]
    c_hi = all_draws_annual_cancelled_upper[idx]

    if d_s.empty and c_s.empty:
        continue

    years = pd.to_datetime([f"{y}-01-01" for y in d_s.index])
    total = (d_s + c_s.reindex(d_s.index, fill_value=0)).values * 100
    total_lo = (d_lo + c_lo.reindex(d_lo.index, fill_value=0)).values * 100
    total_hi = (d_hi + c_hi.reindex(d_hi.index, fill_value=0)).values * 100

    ax2.fill_between(years, total_lo, total_hi, color=col, alpha=0.15, linewidth=0)
    ax2.plot(years, total, color=col, lw=2.5,
             label=f"{scenario_names[draw]} — total disrupted")
    ax2.plot(years, d_s.values * 100, color=col, lw=1, ls="--", alpha=0.5)
    ax2.plot(years, c_s.reindex(d_s.index, fill_value=0).values * 100,
             color=col, lw=1, ls=":", alpha=0.5)

# ANC annual overlay
if not anc_annual.empty:
    ax2.plot(
        anc_annual.index, anc_annual.values * 100,
        color=COLOUR_ANC, lw=2.5, ls="--",
        alpha=0.9, label="ANC disruptions (DHIS2, annual mean)",
    )

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax2.set_xlabel("Year", fontsize=12, fontweight="bold")
ax2.set_ylabel("% disrupted (mean across RealFacility_IDs)", fontsize=12, fontweight="bold")
ax2.set_title(
    f"Annual mean per-facility disruption rate by scenario ({min_year}–{max_year - 1})\n"
    "Solid = total; dashed = delayed; dotted = cancelled",
    fontsize=12, fontweight="bold",
)
ax2.set_ylim(bottom=0)
ax2.grid(True, alpha=0.3, ls=":")
ax2.legend(fontsize=9, frameon=True, framealpha=0.9)
fig2.tight_layout()
out_annual = output_folder / f"comparison_disruption_annual_{suffix}.png"
fig2.savefig(out_annual, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved {out_annual}")

# ─────────────────────────────────────────────────────────────────────────────
#  CSV OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

# Monthly rates per scenario
monthly_rows = []
for idx, draw in enumerate(scenarios_of_interest):
    d_s = all_draws_monthly_delayed_rate[idx]
    c_s = all_draws_monthly_cancelled_rate[idx]
    all_ym = d_s.index.union(c_s.index)
    for ym in all_ym:
        monthly_rows.append({
            "Scenario": scenario_names[draw],
            "draw": draw,
            "year_month": ym,
            "delayed_rate": d_s.get(ym, 0),
            "cancelled_rate": c_s.get(ym, 0),
            "total_disruption_rate": d_s.get(ym, 0) + c_s.get(ym, 0),
        })
pd.DataFrame(monthly_rows).to_csv(
    output_folder / f"monthly_disruption_rates_realfacilityid_{suffix}.csv", index=False
)

# Annual rates per scenario
annual_rows = []
for idx, draw in enumerate(scenarios_of_interest):
    d_s = all_draws_annual_delayed_rate[idx]
    c_s = all_draws_annual_cancelled_rate[idx]
    for yr in d_s.index.union(c_s.index):
        annual_rows.append({
            "Scenario": scenario_names[draw],
            "draw": draw,
            "year": yr,
            "delayed_rate": d_s.get(yr, 0),
            "cancelled_rate": c_s.get(yr, 0),
            "total_disruption_rate": d_s.get(yr, 0) + c_s.get(yr, 0),
        })
pd.DataFrame(annual_rows).to_csv(
    output_folder / f"annual_disruption_rates_realfacilityid_{suffix}.csv", index=False
)

# ANC reference rates
if not anc_monthly.empty:
    anc_monthly.rename("anc_disruption_rate").to_csv(
        output_folder / f"anc_disruption_rate_monthly_{SSP}_{MODEL}.csv", header=True
    )

print("\nDone.")
