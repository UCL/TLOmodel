\"""
validate_disruption_experiment.py

Validates the uniform p=0.1 disruption experiment using the same
extract_results/summarize pattern as the rest of the analysis codebase.
"""

import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from tlo import Date
from tlo.analysis.utils import extract_results, summarize

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_FOLDER = Path("/Users/rem76/Desktop/Climate_Change_Health/model_output/validation_run")
OUTPUT_DIR = Path("/Users/rem76/Desktop/Climate_Change_Health/validation")
OUTPUT_DIR.mkdir(exist_ok=True)

PROB_DISRUPTION = 0.1
ALPHA = 0.05
DRAW = 0  # which draw to inspect (0 if single run)

min_year = 2025
max_year = 2031
TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))


# ── Helper functions (same pattern as your existing analysis) ─────────────────

def get_total_cancelled(_df):
    """Count total weather-cancelled HSI events within TARGET_PERIOD."""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    return pd.Series(len(_df), name="total")


def get_total_delayed(_df):
    """Count total weather-delayed HSI events within TARGET_PERIOD."""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    return pd.Series(len(_df), name="total")


def get_total_hsi_ran(_df):
    """Sum all HSI event counts (did_run=True) within TARGET_PERIOD."""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    total = {}
    for d in _df["hsi_event_key_to_counts"]:
        if isinstance(d, dict):
            for k, v in d.items():
                total[k] = total.get(k, 0) + v
    return pd.Series(sum(total.values()), name="total")


def get_cancelled_by_facility(_df):
    """Count cancelled events per RealFacility_ID within TARGET_PERIOD."""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    return _df.groupby("RealFacility_ID").size().rename("n_cancelled")


def get_delayed_by_facility(_df):
    """Count delayed events per RealFacility_ID within TARGET_PERIOD."""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    return _df.groupby("RealFacility_ID").size().rename("n_delayed")


def get_cancelled_by_level(_df):
    """Count cancelled events per Facility_Level within TARGET_PERIOD."""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    return _df.groupby("Facility_Level").size().rename("n_cancelled")


def get_delayed_by_level(_df):
    """Count delayed events per Facility_Level within TARGET_PERIOD."""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    return _df.groupby("Facility_Level").size().rename("n_delayed")


def get_cancelled_by_month(_df):
    """Count cancelled events per month within TARGET_PERIOD."""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    _df["month"] = _df["date"].dt.to_period("M")
    return _df.groupby("month").size().rename("n_cancelled")


def get_delayed_by_month(_df):
    """Count delayed events per month within TARGET_PERIOD."""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    _df["month"] = _df["date"].dt.to_period("M")
    return _df.groupby("month").size().rename("n_delayed")


# ── 1. Extract totals ─────────────────────────────────────────────────────────
print("Extracting totals...")

n_cancelled = summarize(extract_results(
    RESULTS_FOLDER,
    module="tlo.methods.healthsystem.summary",
    key="Weather_cancelled_HSI_Event_full_info",
    custom_generate_series=get_total_cancelled,
    do_scaling=False,
), only_mean=True).iloc[DRAW, 0]

n_delayed = summarize(extract_results(
    RESULTS_FOLDER,
    module="tlo.methods.healthsystem.summary",
    key="Weather_delayed_HSI_Event_full_info",
    custom_generate_series=get_total_delayed,
    do_scaling=False,
), only_mean=True).iloc[DRAW, 0]

n_ran = summarize(extract_results(
    RESULTS_FOLDER,
    module="tlo.methods.healthsystem.summary",
    key="hsi_event_counts",
    custom_generate_series=get_total_hsi_ran,
    do_scaling=False,
), only_mean=True).iloc[DRAW, 0]

n_cancelled = int(n_cancelled)
n_delayed = int(n_delayed)
n_ran = int(n_ran)
n_disrupted = n_cancelled + n_delayed
n_total = n_ran + n_disrupted

print(f"  HSI ran:        {n_ran:>10,}")
print(f"  Cancelled:      {n_cancelled:>10,}")
print(f"  Delayed:        {n_delayed:>10,}")
print(f"  Total attempts: {n_total:>10,}")

# ── 2. Overall rate ───────────────────────────────────────────────────────────
observed_rate = n_disrupted / n_total if n_total > 0 else 0.0
ci_lo = stats.binom.ppf(0.025, n_total, PROB_DISRUPTION) / n_total
ci_hi = stats.binom.ppf(0.975, n_total, PROB_DISRUPTION) / n_total
within_ci = ci_lo <= observed_rate <= ci_hi

print(f"\n{'═' * 55}")
print(f"  Observed disruption rate:  {observed_rate:.4f}")
print(f"  Expected rate (p=0.1):     {PROB_DISRUPTION:.4f}")
print(f"  95% CI under H0:           [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Within CI:                 {'YES ✓' if within_ci else 'NO ✗'}")
print(f"{'═' * 55}\n")

# ── 3. By facility (→ by district, since FIC_{level}_{district}) ──────────────
print("Extracting per-facility counts...")

FACILITY_PATTERN = re.compile(r"^FIC_(level_\w+)_(.+)$")


def parse_fac(fac_id):
    if pd.isna(fac_id) or fac_id == "unknown":
        return None, None
    m = FACILITY_PATTERN.match(str(fac_id))
    return (m.group(1), m.group(2)) if m else (None, None)


cancelled_by_fac = summarize(extract_results(
    RESULTS_FOLDER,
    module="tlo.methods.healthsystem.summary",
    key="Weather_cancelled_HSI_Event_full_info",
    custom_generate_series=get_cancelled_by_facility,
    do_scaling=False,
), only_mean=True, collapse_columns=False)[DRAW]

delayed_by_fac = summarize(extract_results(
    RESULTS_FOLDER,
    module="tlo.methods.healthsystem.summary",
    key="Weather_delayed_HSI_Event_full_info",
    custom_generate_series=get_delayed_by_facility,
    do_scaling=False,
), only_mean=True, collapse_columns=False)[DRAW]

# Combine into a single per-facility DataFrame
fac_df = pd.DataFrame({
    "n_cancelled": cancelled_by_fac,
    "n_delayed": delayed_by_fac,
}).fillna(0).astype(int)
fac_df.index.name = "RealFacility_ID"
fac_df = fac_df.reset_index()

# Parse level and district from fictional facility name
parsed = fac_df["RealFacility_ID"].apply(parse_fac)
fac_df["level"] = [p[0] for p in parsed]
fac_df["district"] = [p[1] for p in parsed]
fac_df["n_disrupted"] = fac_df["n_cancelled"] + fac_df["n_delayed"]

# ── 4. Aggregate to district level ────────────────────────────────────────────
# Under fictional setup every district gets equal individuals at every level,
# so attempts per district ≈ n_total / n_districts
district_df = fac_df.groupby("district")["n_disrupted"].sum().reset_index()
n_districts = len(district_df)
attempts_per_district = n_total / n_districts

district_df["n_attempts"] = attempts_per_district
district_df["expected_disruptions"] = attempts_per_district * PROB_DISRUPTION
district_df["observed_rate"] = district_df["n_disrupted"] / district_df["n_attempts"]
district_df["abs_error"] = (district_df["observed_rate"] - PROB_DISRUPTION).abs()

district_df["binom_pvalue"] = district_df.apply(
    lambda r: 2 * min(
        stats.binom.cdf(int(r["n_disrupted"]), int(r["n_attempts"]), PROB_DISRUPTION),
        1 - stats.binom.cdf(int(r["n_disrupted"]) - 1, int(r["n_attempts"]), PROB_DISRUPTION),
    ), axis=1
)
district_df["significant"] = district_df["binom_pvalue"] < ALPHA

# ── 5. Aggregate to level ─────────────────────────────────────────────────────
level_df = fac_df.groupby("level")["n_disrupted"].sum().reset_index()
n_levels = level_df["level"].nunique()
level_df["n_attempts"] = n_total / n_levels
level_df["observed_rate"] = level_df["n_disrupted"] / level_df["n_attempts"]

# ── 6. Monthly convergence ────────────────────────────────────────────────────
print("Extracting monthly counts...")

monthly_cancelled = summarize(extract_results(
    RESULTS_FOLDER,
    module="tlo.methods.healthsystem.summary",
    key="Weather_cancelled_HSI_Event_full_info",
    custom_generate_series=get_cancelled_by_month,
    do_scaling=False,
), only_mean=True, collapse_columns=False)[DRAW]

monthly_delayed = summarize(extract_results(
    RESULTS_FOLDER,
    module="tlo.methods.healthsystem.summary",
    key="Weather_delayed_HSI_Event_full_info",
    custom_generate_series=get_delayed_by_month,
    do_scaling=False,
), only_mean=True, collapse_columns=False)[DRAW]

monthly = pd.DataFrame({
    "n_cancelled": monthly_cancelled,
    "n_delayed": monthly_delayed,
}).fillna(0)
monthly.index.name = "month"
monthly = monthly.reset_index()
monthly["n_disrupted"] = monthly["n_cancelled"] + monthly["n_delayed"]
monthly["n_attempts"] = n_total / len(monthly)
monthly["monthly_rate"] = monthly["n_disrupted"] / monthly["n_attempts"]
monthly["cum_disrupted"] = monthly["n_disrupted"].cumsum()
monthly["cum_attempts"] = monthly["n_attempts"].cumsum()
monthly["cum_rate"] = monthly["cum_disrupted"] / monthly["cum_attempts"]
monthly["ci_lo"] = monthly.apply(
    lambda r: stats.binom.ppf(0.025, int(r["cum_attempts"]), PROB_DISRUPTION) / r["cum_attempts"], axis=1
)
monthly["ci_hi"] = monthly.apply(
    lambda r: stats.binom.ppf(0.975, int(r["cum_attempts"]), PROB_DISRUPTION) / r["cum_attempts"], axis=1
)
monthly["month_str"] = monthly["month"].astype(str)

# ── 7. Save CSVs ──────────────────────────────────────────────────────────────
district_df.to_csv(OUTPUT_DIR / "district_observed_vs_expected.csv", index=False)
level_df.to_csv(OUTPUT_DIR / "level_observed_vs_expected.csv", index=False)
fac_df.to_csv(OUTPUT_DIR / "facility_observed_vs_expected.csv", index=False)
print("Saved CSVs.")

# ── 8. Plots ──────────────────────────────────────────────────────────────────
LEVEL_COLOURS = {
    "level_0": "#a8d8ea", "level_1a": "#4caf93", "level_1b": "#f4a261",
    "level_2": "#e76f51", "level_3": "#c0392b", "level_4": "#8e44ad",
}

fig = plt.figure(figsize=(18, 14), facecolor="#f5f0eb")
fig.suptitle(
    f"Validation: Uniform p={PROB_DISRUPTION} Disruption Experiment\n"
    f"Total HSI attempts ≈ {n_total:,}   |   "
    f"Observed rate = {observed_rate:.4f}   |   "
    f"Expected = {PROB_DISRUPTION}   |   "
    f"Within 95% CI: {'YES ✓' if within_ci else 'NO ✗'}",
    fontsize=13, fontweight="bold", y=0.98, color="#111111",
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# A – observed vs expected scatter
ax1 = fig.add_subplot(gs[0, :2])
colors = ["#e74c3c" if s else "#2c7bb6" for s in district_df["significant"]]
ax1.scatter(district_df["expected_disruptions"], district_df["n_disrupted"],
            c=colors, s=60, edgecolors="#333", linewidths=0.5, alpha=0.85, zorder=3)
lim = max(district_df[["expected_disruptions", "n_disrupted"]].max()) * 1.08
ax1.plot([0, lim], [0, lim], "k--", lw=1.2, label="Perfect agreement")
ax1.set_xlabel("Expected disruptions")
ax1.set_ylabel("Observed disruptions")
ax1.set_title("Observed vs Expected per district  (red = p<0.05)", fontsize=11)
ax1.legend(fontsize=9)
ax1.set_facecolor("#eef3f7")
for _, row in district_df[district_df["significant"]].iterrows():
    ax1.annotate(row["district"],
                 xy=(row["expected_disruptions"], row["n_disrupted"]),
                 fontsize=6.5, color="#c0392b", xytext=(4, 4), textcoords="offset points")

# B – histogram of observed rates
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(district_df["observed_rate"], bins=20,
         color="#4caf93", edgecolor="#333", linewidth=0.5, alpha=0.85)
ax2.axvline(PROB_DISRUPTION, color="#e74c3c", lw=2, linestyle="--", label="Expected")
ax2.axvline(district_df["observed_rate"].mean(), color="#2c7bb6",
            lw=1.5, linestyle=":", label=f"Mean ({district_df['observed_rate'].mean():.3f})")
ax2.set_xlabel("Observed disruption rate")
ax2.set_ylabel("Districts")
ax2.set_title("Distribution of per-district rates", fontsize=11)
ax2.legend(fontsize=8)
ax2.set_facecolor("#eef3f7")

# C – absolute error bar chart
ax3 = fig.add_subplot(gs[1, :2])
sdf = district_df.sort_values("abs_error", ascending=False)
ax3.bar(range(len(sdf)), sdf["abs_error"] * 100,
        color=["#e74c3c" if s else "#2c7bb6" for s in sdf["significant"]],
        edgecolor="#333", linewidth=0.3, alpha=0.85)
ax3.set_xticks(range(len(sdf)))
ax3.set_xticklabels(sdf["district"], rotation=75, ha="right", fontsize=6.5)
ax3.set_ylabel("|Observed rate − 0.1| (pp)")
ax3.set_title("Absolute error by district  (red = p<0.05)", fontsize=11)
ax3.set_facecolor("#eef3f7")

# D – per-level observed rates
ax4 = fig.add_subplot(gs[1, 2])
ax4.bar(range(len(level_df)),
        level_df["observed_rate"] * 100,
        color=[LEVEL_COLOURS.get(l, "#aaa") for l in level_df["level"]],
        edgecolor="#333", linewidth=0.5, alpha=0.9)
ax4.axhline(PROB_DISRUPTION * 100, color="#e74c3c", lw=2, linestyle="--", label="Expected (10%)")
ax4.set_xticks(range(len(level_df)))
ax4.set_xticklabels(level_df["level"], rotation=30, ha="right", fontsize=9)
ax4.set_ylabel("Observed disruption rate (%)")
ax4.set_title("Disruption rate by facility level", fontsize=11)
ax4.legend(fontsize=8)
ax4.set_facecolor("#eef3f7")

plt.savefig(OUTPUT_DIR / "disruption_validation.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved validation plot.")

# ── 9. Convergence plot ───────────────────────────────────────────────────────
if not monthly.empty:
    fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#f5f0eb")
    fig2.suptitle("Convergence of disruption rate over simulation time",
                  fontsize=13, fontweight="bold")
    for ax in (ax_a, ax_b):
        ax.set_facecolor("#eef3f7")

    ax_a.plot(monthly["month_str"], monthly["cum_rate"],
              color="#2c7bb6", lw=2, label="Cumulative rate")
    ax_a.fill_between(monthly["month_str"], monthly["ci_lo"], monthly["ci_hi"],
                      alpha=0.15, color="#2c7bb6", label="95% CI")
    ax_a.axhline(PROB_DISRUPTION, color="#e74c3c", lw=1.5, linestyle="--", label="Expected")
    ax_a.set_title("Cumulative disruption rate")
    ax_a.set_xlabel("Month")
    ax_a.legend(fontsize=8)
    ax_a.tick_params(axis="x", rotation=45, labelsize=7)

    ax_b.plot(monthly["month_str"], monthly["monthly_rate"],
              color="#4caf93", lw=1.5, label="Monthly rate")
    ax_b.axhline(PROB_DISRUPTION, color="#e74c3c", lw=1.5, linestyle="--", label="Expected")
    ax_b.set_title("Month-by-month disruption rate")
    ax_b.set_xlabel("Month")
    ax_b.legend(fontsize=8)
    ax_b.tick_params(axis="x", rotation=45, labelsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "disruption_convergence.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("Saved convergence plot.")

# ── 10. Summary ───────────────────────────────────────────────────────────────
flagged = district_df[district_df["significant"]].sort_values("binom_pvalue")
print(f"\nDistricts with significant deviation: {len(flagged)} / {n_districts}")
if not flagged.empty:
    print(flagged[["district", "n_attempts", "n_disrupted",
                   "expected_disruptions", "observed_rate", "binom_pvalue"]].to_string(index=False))
else:
    print("  None — all districts consistent with p=0.1 ✓")

print(f"""
{'═' * 55}
VALIDATION SUMMARY
{'═' * 55}
Total HSI attempts (est.):    {n_total:>10,}
  Ran:                        {n_ran:>10,}
  Cancelled:                  {n_cancelled:>10,}
  Delayed:                    {n_delayed:>10,}

Expected disruptions:         {n_total * PROB_DISRUPTION:>10,.1f}
Observed disruptions:         {n_disrupted:>10,}
Difference:                   {n_disrupted - n_total * PROB_DISRUPTION:>+10.1f}

Overall observed rate:        {observed_rate:>10.4f}
Expected rate:                {PROB_DISRUPTION:>10.4f}
Within 95% CI:                {'YES ✓' if within_ci else 'NO ✗':>10}

Per-district stats:
  Mean:  {district_df['observed_rate'].mean():.4f}
  Std:   {district_df['observed_rate'].std():.4f}
  Min:   {district_df['observed_rate'].min():.4f}
  Max:   {district_df['observed_rate'].max():.4f}

Districts significantly deviating: {district_df['significant'].sum()} / {n_districts}
{'═' * 55}
""")
