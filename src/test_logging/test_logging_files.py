"""
check_hsi_counts.py
-------------------
Quick sanity-check script for a single-draw TLO run with no climate disruptions.
Prints total HSIs, non-blank-footprint HSIs, and the blank-footprint fraction.

Usage:
    python check_hsi_counts.py <results_folder>
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from tlo import Date
from tlo.analysis.utils import extract_results, summarize

# ── Time window ────────────────────────────────────────────────────────────────
MIN_YEAR = 2025
MAX_YEAR = 2027
TARGET_PERIOD = (Date(MIN_YEAR, 1, 1), Date(MAX_YEAR, 12, 31))

DRAW = 0  # only one draw in this run


# ── Extraction helpers ─────────────────────────────────────────────────────────

def _filter_to_period(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.loc[df["date"].between(*TARGET_PERIOD)]


def sum_treatment_id_counts(df: pd.DataFrame) -> pd.Series:
    """Sum TREATMENT_ID dict values across all rows in the target period."""
    df = _filter_to_period(df)
    total = sum(sum(d.values()) for d in df["TREATMENT_ID"])
    return pd.Series([total], name="total")


def count_rows(df: pd.DataFrame) -> pd.Series:
    """Simple row count within the target period (fallback for flat logs)."""
    return pd.Series([len(_filter_to_period(df))], name="count")


# ── Core extraction ────────────────────────────────────────────────────────────

def _extract(results_folder: Path, module: str, key: str, func) -> dict:
    """Return {mean, lower, upper} for a single draw."""
    raw = summarize(
        extract_results(
            results_folder,
            module=module,
            key=key,
            custom_generate_series=func,
            do_scaling=False,
        ),
        only_mean=False,
        collapse_columns=True,
    )

    def _scalar(series_or_val):
        if hasattr(series_or_val, "values"):
            return float(series_or_val.values[0])
        return float(series_or_val)

    return {
        "mean": _scalar(raw["mean"]),
        "lower": _scalar(raw["lower"]),
        "upper": _scalar(raw["upper"]),
    }


def _fmt(d: dict) -> str:
    return f"{d['mean']:>14,.0f}  [{d['lower']:,.0f} – {d['upper']:,.0f}]"


# ── Main ───────────────────────────────────────────────────────────────────────

def apply(results_folder: Path) -> None:
    print(f"\nResults folder : {results_folder}")
    print(f"Target period  : {MIN_YEAR}–{MAX_YEAR}")
    print(f"Draw           : {DRAW}\n")

    # --- HSI counts -----------------------------------------------------------
    print("Extracting HSI_Event (all ran HSIs)...")
    treatments = _extract(
        results_folder,
        "tlo.methods.healthsystem.summary", "HSI_Event",
        sum_treatment_id_counts,
    )

    print("Extracting HSI_Event_non_blank_appt_footprint...")
    non_blank = _extract(
        results_folder,
        "tlo.methods.healthsystem.summary", "HSI_Event_non_blank_appt_footprint",
        sum_treatment_id_counts,
    )

    print("Extracting Never_ran_HSI_Event...")
    never_ran = _extract(
        results_folder,
        "tlo.methods.healthsystem.summary", "Never_ran_HSI_Event",
        sum_treatment_id_counts,
    )

    # --- Derived quantities ---------------------------------------------------
    blank_hsis = treatments["mean"] - non_blank["mean"]
    blank_pct = blank_hsis / treatments["mean"] * 100 if treatments["mean"] else float("nan")
    never_ran_pct = never_ran["mean"] / (treatments["mean"] + never_ran["mean"]) * 100 \
        if (treatments["mean"] + never_ran["mean"]) else float("nan")

    # --- Weather disruption check (expect zero) --------------------------------
    print("Checking for weather-disrupted HSIs (should be absent / zero)...")
    try:
        delayed = _extract(
            results_folder,
            "tlo.methods.healthsystem.summary", "Weather_delayed_HSI_Event",
            sum_treatment_id_counts,
        )
    except Exception:
        delayed = {"mean": 0.0, "lower": 0.0, "upper": 0.0}

    try:
        cancelled = _extract(
            results_folder,
            "tlo.methods.healthsystem.summary", "Weather_cancelled_HSI_Event",
            sum_treatment_id_counts,
        )
    except Exception:
        cancelled = {"mean": 0.0, "lower": 0.0, "upper": 0.0}

    # --- Report ---------------------------------------------------------------
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"{'METRIC':<38} {'MEAN':>10}  [lower – upper]")
    print(sep)
    print(f"{'Total HSIs ran':<38} {_fmt(treatments)}")
    print(f"{'  of which: non-blank footprint':<38} {_fmt(non_blank)}")
    print(f"{'  of which: blank footprint':<38} {blank_hsis:>14,.0f}  ({blank_pct:.1f}%)")
    print(f"{'Never-ran HSIs':<38} {_fmt(never_ran)}")
    print(f"{'Weather-delayed HSIs':<38} {_fmt(delayed)}")
    print(f"{'Weather-cancelled HSIs':<38} {_fmt(cancelled)}")
    print(sep)
    print(f"\nDerived rates")
    print(f"  Blank-footprint share of ran HSIs : {blank_pct:.2f}%")
    print(f"  Never-ran share of all attempted  : {never_ran_pct:.2f}%")
    print(
        f"  Weather disruption rate           : {(delayed['mean'] + cancelled['mean']) / max(treatments['mean'], 1) * 100:.4f}%")


def _check(label: str, condition: bool) -> None:
    status = "✓ PASS" if condition else "✗ FAIL"
    print(f"  {status}  {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check HSI counts for a no-disruption TLO run.")
    parser.add_argument("results_folder", type=Path, help="Path to the scenario results folder")
    args = parser.parse_args()
    apply(results_folder=args.results_folder)
