import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import extract_results, summarize

min_year = 2025
max_year = 2027

scenarios_of_interest = range(4)
scenario_names = list(scenarios_of_interest)
suffix = "mode_2_test"
scenario_colours = [
    "#823038",  # Baseline
    "#00566f",
    "#0081a7",
    "#5ab4c6",
]

TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 12, 31))


# ---------------------------------------------------------------------------
# Unified extraction helpers
# ---------------------------------------------------------------------------

def sum_treatment_id_counts(_df):
    """Sum TREATMENT_ID dict values across all rows within the target period.
    Works for: HSI_Event, Never_ran_HSI_Event, HSI_Event_non_blank_appt_footprint,
               Weather_cancelled_HSI_Event, Weather_delayed_HSI_Event.
    """
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    total = 0
    for d in _df["TREATMENT_ID"]:
        total += sum(d.values())
    return pd.Series(total, name="total")


def get_mean_frac_capacity_used(_df):
    """Extract mean fraction of capacity used overall (averaged across clinics)
    from the annual Capacity summary log.
    """
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    # average_Frac_Time_Used_Overall is a dict keyed by clinic name
    values = []
    for d in _df["average_Frac_Time_Used_Overall"]:
        if isinstance(d, dict) and len(d) > 0:
            values.append(np.mean(list(d.values())))
    return pd.Series(np.mean(values) if values else np.nan, name="frac_capacity")


def get_population_mean(_df):
    _df["date"] = pd.to_datetime(_df["date"])
    filtered = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    numeric = filtered.drop(columns=["female", "male"], errors="ignore")
    return pd.Series(numeric.sum(numeric_only=True).mean(), name="population")


# ---------------------------------------------------------------------------
# Extraction wrapper
# ---------------------------------------------------------------------------

def extract_draw(results_folder, module, key, func, draw):
    return summarize(
        extract_results(
            results_folder,
            module=module,
            key=key,
            custom_generate_series=func,
            do_scaling=False,
        ),
        only_mean=False,
        collapse_columns=True,
    )[draw]


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    records = []

    for draw in scenarios_of_interest:
        print(f"Processing draw {draw}...")
        is_baseline = (draw == 0)

        treatments = extract_draw(results_folder, "tlo.methods.healthsystem.summary", "HSI_Event",
                                  sum_treatment_id_counts, draw)
        never_ran = extract_draw(results_folder, "tlo.methods.healthsystem.summary", "Never_ran_HSI_Event",
                                 sum_treatment_id_counts, draw)
        non_blank = extract_draw(results_folder, "tlo.methods.healthsystem.summary",
                                 "HSI_Event_non_blank_appt_footprint", sum_treatment_id_counts, draw)
        capacity = extract_draw(results_folder, "tlo.methods.healthsystem.summary", "Capacity",
                                get_mean_frac_capacity_used, draw)
        population = extract_draw(results_folder, "tlo.methods.demography", "population", get_population_mean, draw)

        if is_baseline:
            delayed = {"mean": pd.Series([0]), "lower": pd.Series([0]), "upper": pd.Series([0])}
            cancelled = {"mean": pd.Series([0]), "lower": pd.Series([0]), "upper": pd.Series([0])}
        else:
            delayed = extract_draw(results_folder, "tlo.methods.healthsystem.summary", "Weather_delayed_HSI_Event",
                                   sum_treatment_id_counts, draw)
            cancelled = extract_draw(results_folder, "tlo.methods.healthsystem.summary", "Weather_cancelled_HSI_Event",
                                     sum_treatment_id_counts, draw)

        def v(d, stat="mean"):
            val = d[stat].values[0] if hasattr(d[stat], "values") else d[stat]
            return float(val)

        # Accounting check:
        # treatments + never_ran + weather_cancelled ~ total resolved HSIs
        # (weather_delayed are re-queued so excluded from this sum)
        total_resolved = v(treatments) + v(never_ran) + v(cancelled)

        records.append({
            "draw": draw,
            "treatments_mean": v(treatments),
            "treatments_lower": v(treatments, "lower"),
            "treatments_upper": v(treatments, "upper"),
            "never_ran_mean": v(never_ran),
            "never_ran_lower": v(never_ran, "lower"),
            "never_ran_upper": v(never_ran, "upper"),
            "non_blank_mean": v(non_blank),
            "non_blank_lower": v(non_blank, "lower"),
            "non_blank_upper": v(non_blank, "upper"),
            "weather_delayed_mean": v(delayed),
            "weather_delayed_lower": v(delayed, "lower"),
            "weather_delayed_upper": v(delayed, "upper"),
            "weather_cancelled_mean": v(cancelled),
            "weather_cancelled_lower": v(cancelled, "lower"),
            "weather_cancelled_upper": v(cancelled, "upper"),
            "frac_capacity_used_mean": v(capacity),
            "frac_capacity_used_lower": v(capacity, "lower"),
            "frac_capacity_used_upper": v(capacity, "upper"),
            "population_mean": v(population),
            # accounting columns
            "total_resolved": total_resolved,
            "blank_footprint_hsis": v(treatments) - v(non_blank),  # ran but no footprint
            "disruption_rate_pct": (v(delayed) + v(cancelled)) / (
                    v(treatments) + v(cancelled)) * 100 if not is_baseline else 0.0,
        })

    summary_df = pd.DataFrame(records).set_index("draw")
    summary_df.to_csv(output_folder / "summary_all_draws.csv")
    print(summary_df[[
        "treatments_mean", "never_ran_mean", "non_blank_mean",
        "weather_delayed_mean", "weather_cancelled_mean",
        "total_resolved", "frac_capacity_used_mean", "disruption_rate_pct"
    ]].to_string())

    print("\n=== MODE 2 SANITY CHECKS ===")

    # 1. No weather events in no-disruption or supply-side scenarios
    assert summary_df.loc[0, "weather_delayed_mean"] == 0, "Draw 0 should have no delayed events"
    assert summary_df.loc[0, "weather_cancelled_mean"] == 0, "Draw 0 should have no cancelled events"

    # 2. Mixed should be between demand-side and supply-side on capacity
    assert summary_df.loc[1, "frac_capacity_used_mean"] <= summary_df.loc[2, "frac_capacity_used_mean"] <= \
           summary_df.loc[3, "frac_capacity_used_mean"], \
        "Capacity utilisation should increase with prop_supply_side"

    print("All checks passed.")
    # ---------------------------------------------------------------------------
    # Differences from baseline
    # ---------------------------------------------------------------------------
    baseline = summary_df.loc[0]
    diff_df = pd.DataFrame({
        "draw": summary_df.index,
        "treatments_diff": summary_df["treatments_mean"] - baseline["treatments_mean"],
        "treatments_pct_change": (summary_df["treatments_mean"] - baseline["treatments_mean"]) / baseline[
            "treatments_mean"] * 100,
        "never_ran_diff": summary_df["never_ran_mean"] - baseline["never_ran_mean"],
        "never_ran_pct_change": (summary_df["never_ran_mean"] - baseline["never_ran_mean"]) / baseline[
            "never_ran_mean"] * 100,
        "non_blank_diff": summary_df["non_blank_mean"] - baseline["non_blank_mean"],
        "frac_capacity_diff": summary_df["frac_capacity_used_mean"] - baseline["frac_capacity_used_mean"],
    })
    diff_df.to_csv(output_folder / "differences_from_baseline.csv", index=False)

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    draws = summary_df.index.tolist()
    x = np.arange(len(draws))
    colours = scenario_colours[:len(draws)]

    def err(col):
        return np.array([
            summary_df[f"{col}_mean"] - summary_df[f"{col}_lower"],
            summary_df[f"{col}_upper"] - summary_df[f"{col}_mean"],
        ])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel A: total treatments
    axes[0].bar(x, summary_df["treatments_mean"], color=colours, yerr=err("treatments"), capsize=6)
    axes[0].set_title(f"Total HSIs ({min_year}–{max_year})")
    axes[0].set_xticks(x);
    axes[0].set_xticklabels([f"Draw {d}" for d in draws])
    axes[0].set_ylabel("Total HSIs");
    axes[0].grid(False)
    axes[0].text(-0.0, 1.05, "(A)", transform=axes[0].transAxes, fontsize=14, va="top", ha="right")

    # Panel B: weather disruptions (excluding baseline)
    x_w = np.arange(len(draws) - 1)
    w = 0.35 / 2
    axes[1].bar(x_w - w, summary_df["weather_delayed_mean"].iloc[1:], w, label="Delayed", color="#FEB95F",
                yerr=err("weather_delayed")[:, 1:], capsize=6)
    axes[1].bar(x_w + w, summary_df["weather_cancelled_mean"].iloc[1:], w, label="Cancelled", color="#f07167",
                yerr=err("weather_cancelled")[:, 1:], capsize=6)
    axes[1].set_title(f"Weather-Disrupted HSIs ({min_year}–{max_year})")
    axes[1].set_xticks(x_w);
    axes[1].set_xticklabels([f"Draw {d}" for d in draws[1:]])
    axes[1].set_ylabel("HSIs");
    axes[1].grid(False)
    axes[1].legend(frameon=False)
    axes[1].text(-0.0, 1.05, "(B)", transform=axes[1].transAxes, fontsize=14, va="top", ha="right")

    # Panel C: capacity utilisation
    axes[2].bar(x, summary_df["frac_capacity_used_mean"], color=colours, yerr=err("frac_capacity_used"), capsize=6)
    axes[2].set_title(f"Mean Fraction of Capacity Used ({min_year}–{max_year})")
    axes[2].set_xticks(x);
    axes[2].set_xticklabels([f"Draw {d}" for d in draws])
    axes[2].set_ylabel("Fraction of capacity used");
    axes[2].grid(False)
    axes[2].text(-0.0, 1.05, "(C)", transform=axes[2].transAxes, fontsize=14, va="top", ha="right")

    fig.tight_layout()
    fig.savefig(output_folder / f"hsi_summary_{suffix}.png", dpi=300)
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()
    apply(results_folder=args.results_folder, output_folder=args.results_folder, resourcefilepath=Path("./resources"))
