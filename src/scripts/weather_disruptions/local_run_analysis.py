"""
run_weather_disruptions_analysis.py

A single-run simulation script for the WeatherDisruptions module.
Usage:
    python run_weather_disruptions_analysis.py
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods.fullmodel import fullmodel

#  CONFIGURATION

seed = 0

log_config = {
    "filename": "weather_disruptions_analysis",
    "directory": "./outputs",
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.weather_disruptions": logging.INFO,
        "tlo.methods.weather_disruptions.summary": logging.INFO,
        "tlo.methods.healthsystem.summary": logging.INFO,
        "tlo.methods.demography": logging.CRITICAL,
        "tlo.methods.healthsystem": logging.CRITICAL,
    },
}

start_date = Date(2010, 1, 1)
end_date = Date(2030, 1, 1)
pop_size = 5_000

resourcefilepath = Path("./resources")

#  COLOURS

COLOUR_DELAYED = "#CC7000"
COLOUR_CANCELLED = "#A8102E"
COLOUR_TOTAL = "#1E3A8A"
COLOUR_RATE = "#5b3f8c"

#  SIMULATION SETUP

sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, resourcefilepath=resourcefilepath)

sim.register(
    *fullmodel(
        module_kwargs={
            "WeatherDisruptions": {
                "climate_ssp": "ssp245",
                "climate_model_ensemble_model": "mean",
                "year_effective_climate_disruptions": 2025,
                "services_affected_precip": "all",
                "scale_factor_prob_disruption": 10.0,
                "delay_in_seeking_care_weather": 28.0,
                "scale_factor_reseeking_healthcare_post_disruption": 1.0,
                "scale_factor_appointment_urgency": 1.0,
                "scale_factor_severity_disruption_and_delay": 1.0,
                "prop_supply_side_disruptions": 0.5,
            }
        }
    )
)

sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)

#  PARSE LOGS

log_df = parse_log_file(sim.log_filepath)

# Monthly summary from the WeatherDisruptions logger
monthly = log_df["tlo.methods.weather_disruptions.summary"]["weather_disruptions_monthly"].copy()
monthly["date"] = pd.to_datetime(monthly["date"])
monthly = monthly.sort_values("date").set_index("date")

# Individual event logs (one row per disrupted HSI)
delayed_events = log_df["tlo.methods.weather_disruptions.summary"].get(
    "Weather_delayed_HSI_Event_full_info", pd.DataFrame()
)
cancelled_events = log_df["tlo.methods.weather_disruptions.summary"].get(
    "Weather_cancelled_HSI_Event_full_info", pd.DataFrame()
)

#  DERIVED QUANTITIES

monthly["total_disrupted"] = monthly["cancelled"] + monthly["delayed"]
monthly["disruption_rate"] = (
    monthly["total_disrupted"] / (monthly["hsi_total"] + monthly["total_disrupted"])
).clip(upper=1.0)

cumulative_cancelled = monthly["cancelled"].cumsum()
cumulative_delayed = monthly["delayed"].cumsum()
cumulative_total = monthly["total_disrupted"].cumsum()

# Annual rollup
annual = monthly.resample("YE").agg(
    hsi_total=("hsi_total", "sum"),
    cancelled=("cancelled", "sum"),
    delayed=("delayed", "sum"),
    total_disrupted=("total_disrupted", "sum"),
)
annual["disruption_rate"] = (
    annual["total_disrupted"] / (annual["hsi_total"] + annual["total_disrupted"])
).clip(upper=1.0)

# Per-treatment-type breakdown from event logs (top N most common)
TOP_N = 15


def _count_by_treatment(df):
    if df.empty or "TREATMENT_ID" not in df.columns:
        return pd.Series(dtype=float)
    return df["TREATMENT_ID"].value_counts()


cancelled_by_type = _count_by_treatment(cancelled_events)
delayed_by_type = _count_by_treatment(delayed_events)

all_types = cancelled_by_type.index.union(delayed_by_type.index)
by_type = pd.DataFrame({
    "cancelled": cancelled_by_type.reindex(all_types, fill_value=0),
    "delayed": delayed_by_type.reindex(all_types, fill_value=0),
})
by_type["total"] = by_type["cancelled"] + by_type["delayed"]
top_types = by_type.nlargest(TOP_N, "total")

#  PLOT 1: Monthly counts — cancelled, delayed, total

fig1, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(monthly.index, monthly["delayed"], color=COLOUR_DELAYED, lw=1.5, alpha=0.8, label="Delayed")
ax1.plot(monthly.index, monthly["cancelled"], color=COLOUR_CANCELLED, lw=1.5, alpha=0.8, label="Cancelled")
ax1.plot(monthly.index, monthly["total_disrupted"], color=COLOUR_TOTAL, lw=2.5, label="Total disrupted")

ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax1.set_xlabel("Month", fontsize=11, fontweight="bold")
ax1.set_ylabel("Number of HSIs disrupted", fontsize=11, fontweight="bold")
ax1.set_title(
    f"Monthly weather-disrupted HSIs — SSP2-4.5 mean (n={pop_size:,})",
    fontsize=12, fontweight="bold",
)
ax1.set_ylim(bottom=0)
ax1.legend(fontsize=10, framealpha=0.9)
fig1.tight_layout()
fig1.savefig("./outputs/weather_disruptions_monthly_counts.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig1)

#  PLOT 2: Monthly disruption rate (%)

fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(
    monthly.index, monthly["disruption_rate"] * 100,
    color=COLOUR_RATE, lw=2, label="Disruption rate (%)",
)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax2.set_xlabel("Month", fontsize=11, fontweight="bold")
ax2.set_ylabel("% HSIs disrupted", fontsize=11, fontweight="bold", color=COLOUR_RATE)
ax2.set_ylim(bottom=0)
ax2.set_title(
    "Monthly disruption rate and total HSI volume",
    fontsize=12, fontweight="bold",
)

lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_2, labels_2, fontsize=10, framealpha=0.9)

fig2.tight_layout()
fig2.savefig("./outputs/weather_disruptions_monthly_rate.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig2)

#  PLOT 3: Annual disruption counts and rate

fig3, ax3a = plt.subplots(figsize=(10, 5))
ax3b = ax3a.twinx()

years = annual.index.year
x = np.arange(len(years))
width = 0.35

ax3a.bar(x - width / 2, annual["delayed"], width, color=COLOUR_DELAYED, alpha=0.85, label="Delayed")
ax3a.bar(x + width / 2, annual["cancelled"], width, color=COLOUR_CANCELLED, alpha=0.85, label="Cancelled")
ax3b.plot(x, annual["disruption_rate"] * 100, "o--", color=COLOUR_RATE, lw=2, ms=6, label="Disruption rate (%)")

ax3a.set_xticks(x)
ax3a.set_xticklabels(years, rotation=45, ha="right")
ax3a.set_xlabel("Year", fontsize=11, fontweight="bold")
ax3a.set_ylabel("Number of HSIs disrupted", fontsize=11, fontweight="bold")
ax3b.set_ylabel("% HSIs disrupted", fontsize=11, fontweight="bold", color=COLOUR_RATE)
ax3a.set_title(
    "Annual weather-disrupted HSIs and disruption rate",
    fontsize=12, fontweight="bold",
)

lines_a, labels_a = ax3a.get_legend_handles_labels()
lines_b, labels_b = ax3b.get_legend_handles_labels()
ax3a.legend(lines_a + lines_b, labels_a + labels_b, fontsize=10, framealpha=0.9)

fig3.tight_layout()
fig3.savefig("./outputs/weather_disruptions_annual.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig3)

#  PLOT 4: Top N HSI types disrupted — horizontal bar chart

if not top_types.empty:
    fig4, ax4 = plt.subplots(figsize=(10, max(5, TOP_N * 0.5)))

    y_pos = np.arange(len(top_types))
    ax4.barh(
        y_pos, top_types["delayed"].values,
        color=COLOUR_DELAYED, alpha=0.85, label="Delayed",
    )
    ax4.barh(
        y_pos, top_types["cancelled"].values,
        left=top_types["delayed"].values,
        color=COLOUR_CANCELLED, alpha=0.85, label="Cancelled",
    )

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_types.index, fontsize=9)
    ax4.invert_yaxis()
    ax4.set_xlabel("Number of HSI events disrupted", fontsize=11, fontweight="bold")
    ax4.set_title(
        f"Top {TOP_N} HSI types by weather disruption count ({start_date.year}–{end_date.year - 1})",
        fontsize=12, fontweight="bold",
    )
    ax4.legend(fontsize=10, framealpha=0.9)

    fig4.tight_layout()
    fig4.savefig("./outputs/weather_disruptions_by_hsi_type.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig4)

# ─────────────────────────────────────────────────────────────────────────────
#  PRINTED SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

total_cancelled = int(monthly["cancelled"].sum())
total_delayed = int(monthly["delayed"].sum())
total_disrupted = total_cancelled + total_delayed
total_hsi_seen = int(monthly["hsi_total"].sum())
mean_monthly_rate = monthly["disruption_rate"].mean() * 100
peak_month = monthly["disruption_rate"].idxmax()
peak_rate = monthly["disruption_rate"].max() * 100

print("\n" + "=" * 60)
print("  WEATHER DISRUPTIONS — SIMULATION SUMMARY")
print("=" * 60)
print("  Period            : {start_date} → {end_date}")
print("  Population size   : {pop_size:,}")
print("  SSP / model       : ssp245 / mean")
print("-" * 60)
print("Total HSIs seen   : {total_hsi_seen:,}")
print("  Total disrupted   : {total_disrupted:,}")
print("    Cancelled       : {total_cancelled:,}  ({total_cancelled / total_disrupted * 100:.1f}%)")
print("    Delayed         : {total_delayed:,}  ({total_delayed / total_disrupted * 100:.1f}%)")
print("  Mean monthly rate : {mean_monthly_rate:.2f}%")
print("  Peak month        : {peak_month.strftime('%Y-%m')}  ({peak_rate:.2f}%)")
print("=" * 60)
