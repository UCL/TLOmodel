
import argparse
from pathlib import Path

import pandas as pd

from tlo import Date
from tlo.analysis.utils import extract_results, summarize

# ── Time window ────────────────────────────────────────────────────────────────
MIN_YEAR = 2025
MAX_YEAR = 2026
TARGET_PERIOD = (Date(MIN_YEAR, 1, 1), Date(MAX_YEAR, 12, 31))

scenario_names = [
    "Status Quo",
    "HTM Scale-up",
    "Worsening Lifestyle Factors",
    "Improving Lifestyle Factors",
    "Maximal Healthcare Provision",
]


# ── Extraction helpers ─────────────────────────────────────────────────────────

def sum_treatment_id_counts(df: pd.DataFrame) -> pd.Series:
    """Sum TREATMENT_ID dict values across all rows in the target period."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.loc[df["date"].between(*TARGET_PERIOD)]
    total = sum(sum(d.values()) for d in df["TREATMENT_ID"])
    return pd.Series([total], name="total")


# ── Main ───────────────────────────────────────────────────────────────────────

def apply(results_folder: Path) -> None:
    all_draws_treatments = []
    all_draws_non_blank = []
    all_draws_never_ran = []
    all_draws_delayed = []
    all_draws_cancelled = []

    for draw in range(len(scenario_names)):
        print(f"  Extracting draw {draw} ({scenario_names[draw]})...")

        treatments = summarize(
            extract_results(
                results_folder,
                module="tlo.methods.healthsystem.summary",
                key="HSI_Event",
                custom_generate_series=sum_treatment_id_counts,
                do_scaling=False,
            ),
            only_mean=False, collapse_columns=True,
        )[draw]

        non_blank = summarize(
            extract_results(
                results_folder,
                module="tlo.methods.healthsystem.summary",
                key="HSI_Event_non_blank_appt_footprint",
                custom_generate_series=sum_treatment_id_counts,
                do_scaling=False,
            ),
            only_mean=False, collapse_columns=True,
        )[draw]

        never_ran = summarize(
            extract_results(
                results_folder,
                module="tlo.methods.healthsystem.summary",
                key="Never_ran_HSI_Event",
                custom_generate_series=sum_treatment_id_counts,
                do_scaling=False,
            ),
            only_mean=False, collapse_columns=True,
        )[draw]

        try:
            delayed = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.healthsystem.summary",
                    key="Weather_delayed_HSI_Event",
                    custom_generate_series=sum_treatment_id_counts,
                    do_scaling=False,
                ),
                only_mean=False, collapse_columns=True,
            )[draw]
        except Exception:
            delayed = {"mean": 0.0, "lower": 0.0, "upper": 0.0}

        try:
            cancelled = summarize(
                extract_results(
                    results_folder,
                    module="tlo.methods.healthsystem.summary",
                    key="Weather_cancelled_HSI_Event",
                    custom_generate_series=sum_treatment_id_counts,
                    do_scaling=False,
                ),
                only_mean=False, collapse_columns=True,
            )[draw]
        except Exception:
            cancelled = {"mean": 0.0, "lower": 0.0, "upper": 0.0}

        all_draws_treatments.append(treatments)
        all_draws_non_blank.append(non_blank)
        all_draws_never_ran.append(never_ran)
        all_draws_delayed.append(delayed)
        all_draws_cancelled.append(cancelled)

    # ── Summary table ──────────────────────────────────────────────────────────
    sep = "─" * 100
    col_w = 22

    print(f"\n{sep}")
    print(f"{'METRIC':<38}" + "".join(f"{n:>{col_w}}" for n in scenario_names))
    print(sep)

    def _row(label, data_list):
        print(f"{label:<38}" + "".join(f"{d['mean']:>{col_w},.0f}" for d in data_list))

    def _pct_row(label, numerators, denominators):
        vals = []
        for n, d in zip(numerators, denominators):
            pct = n["mean"] / d["mean"] * 100 if d["mean"] else float("nan")
            vals.append(f"{pct:>{col_w}.2f}%")
        print(f"{label:<38}" + "".join(vals))

    blank_row = [{"mean": t["mean"] - nb["mean"]} for t, nb in zip(all_draws_treatments, all_draws_non_blank)]
    total_attempted = [{"mean": t["mean"] + nr["mean"]} for t, nr in zip(all_draws_treatments, all_draws_never_ran)]
    disrupted = [{"mean": dl["mean"] + cn["mean"]} for dl, cn in zip(all_draws_delayed, all_draws_cancelled)]

    _row("Total HSIs ran", all_draws_treatments)
    _row("  Non-blank footprint HSIs", all_draws_non_blank)
    _row("  Blank footprint HSIs", blank_row)
    _row("Never-ran HSIs", all_draws_never_ran)
    _row("Weather-delayed HSIs", all_draws_delayed)
    _row("Weather-cancelled HSIs", all_draws_cancelled)

    print(sep)
    _pct_row("Blank-footprint share of ran HSIs", blank_row, all_draws_treatments)
    _pct_row("Never-ran share of all attempted", all_draws_never_ran, total_attempted)
    _pct_row("Weather disruption rate", disrupted, all_draws_treatments)
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sanity-check HSI counts across all scenario draws."
    )
    parser.add_argument("results_folder", type=Path,
                        help="Path to the scenario results folder")
    args = parser.parse_args()
    apply(results_folder=args.results_folder)
