import argparse
from pathlib import Path

import pandas as pd

from tlo import Date
from tlo.analysis.utils import extract_results, summarize

# ── Time window ────────────────────────────────────────────────────────────────
MIN_YEAR = 2026
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


def extract_treatment_id_counts(df: pd.DataFrame) -> pd.Series:
    """Return per-TREATMENT_ID counts across the target period as a flat Series."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.loc[df["date"].between(*TARGET_PERIOD)]
    totals = {}
    for d in df["TREATMENT_ID"]:
        for treatment_id, count in d.items():
            totals[treatment_id] = totals.get(treatment_id, 0) + count
    return pd.Series(totals)


# ── Diagnostic: blank footprint breakdown ──────────────────────────────────────

def blank_footprint_breakdown(results_folder: Path, draw: int = 0) -> None:
    """Print a ranked breakdown of blank-footprint HSIs by TREATMENT_ID for a single draw."""

    print(f"\n{'─' * 80}")
    print(f"BLANK FOOTPRINT BREAKDOWN BY TREATMENT_ID  (draw {draw}: {scenario_names[draw]})")
    print(f"{'─' * 80}")

    all_hsi = extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="HSI_Event",
        custom_generate_series=extract_treatment_id_counts,
        do_scaling=False,
    ).mean(axis=1)

    non_blank_hsi = extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="HSI_Event_non_blank_appt_footprint",
        custom_generate_series=extract_treatment_id_counts,
        do_scaling=False,
    ).mean(axis=1)

    # Align on index (some treatment IDs may be absent from non_blank)
    combined = pd.DataFrame({
        "total": all_hsi,
        "non_blank": non_blank_hsi,
    }).fillna(0)

    combined["blank"] = combined["total"] - combined["non_blank"]
    combined["blank_pct"] = combined["blank"] / combined["total"] * 100

    # Only rows where there are some blank footprints
    has_blanks = combined.loc[combined["blank"] > 0].copy()
    has_blanks = has_blanks.sort_values("blank", ascending=False)

    if has_blanks.empty:
        print("No blank-footprint HSIs found.")
        return

    total_blank = has_blanks["blank"].sum()
    has_blanks["share_of_all_blanks"] = has_blanks["blank"] / total_blank * 100

    col_w = 14
    print(
        f"\n{'TREATMENT_ID':<55} {'Total':>{col_w}} {'Blank':>{col_w}} {'Blank%':>{col_w}} {'Share of blanks':>{col_w}}")
    print("─" * (55 + col_w * 4 + 4))

    for treatment_id, row in has_blanks.iterrows():
        print(
            f"{str(treatment_id):<55}"
            f"{row['total']:>{col_w},.0f}"
            f"{row['blank']:>{col_w},.0f}"
            f"{row['blank_pct']:>{col_w}.1f}%"
            f"{row['share_of_all_blanks']:>{col_w}.1f}%"
        )

    print("─" * (55 + col_w * 4 + 4))
    print(f"\nTotal blank-footprint HSIs : {total_blank:,.0f}")
    print(f"Unique TREATMENT_IDs with any blanks: {len(has_blanks)}")

    # Highlight treatment IDs that are ~100% blank - most suspicious
    always_blank = has_blanks.loc[has_blanks["blank_pct"] >= 99.9]
    if not always_blank.empty:
        print(f"\n⚠  TREATMENT_IDs that are ~100% blank ({len(always_blank)} found):")
        for tid in always_blank.index:
            print(f"   - {tid}")


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
        )

        non_blank = summarize(
            extract_results(
                results_folder,
                module="tlo.methods.healthsystem.summary",
                key="HSI_Event_non_blank_appt_footprint",
                custom_generate_series=sum_treatment_id_counts,
                do_scaling=False,
            ),
            only_mean=False, collapse_columns=True,
        )

        never_ran = summarize(
            extract_results(
                results_folder,
                module="tlo.methods.healthsystem.summary",
                key="Never_ran_HSI_Event",
                custom_generate_series=sum_treatment_id_counts,
                do_scaling=False,
            ),
            only_mean=False, collapse_columns=True,
        )

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
            )
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
            )
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

    def _scalar(d):
        """Extract a scalar from a value that may be a Series or a float."""
        v = d["mean"]
        if hasattr(v, "iloc"):
            return float(v.iloc[0])
        return float(v)

    def _row(label, data_list):
        print(f"{label:<38}" + "".join(f"{_scalar(d):>{col_w},.0f}" for d in data_list))

    def _pct_row(label, numerators, denominators):
        vals = []
        for n, d in zip(numerators, denominators):
            denom = _scalar(d)
            pct = _scalar(n) / denom * 100 if denom else float("nan")
            vals.append(f"{pct:>{col_w}.2f}%")
        print(f"{label:<38}" + "".join(vals))

    blank_row = [{"mean": _scalar(t) - _scalar(nb)} for t, nb in zip(all_draws_treatments, all_draws_non_blank)]
    total_attempted = [{"mean": _scalar(t) + _scalar(nr)} for t, nr in zip(all_draws_treatments, all_draws_never_ran)]
    disrupted = [{"mean": _scalar(dl) + _scalar(cn)} for dl, cn in zip(all_draws_delayed, all_draws_cancelled)]

    print(f"\n{sep}")
    print(f"{'METRIC':<38}" + "".join(f"{n:>{col_w}}" for n in scenario_names))
    print(sep)

    _row("Total HSIs ran", all_draws_treatments)
    _row("  of which: non-blank footprint", all_draws_non_blank)
    _row("  of which: blank footprint", blank_row)
    _row("Never-ran HSIs", all_draws_never_ran)
    _row("Weather-delayed HSIs", all_draws_delayed)
    _row("Weather-cancelled HSIs", all_draws_cancelled)

    print(sep)
    print("Derived rates")
    _pct_row("  Blank-footprint share of ran HSIs :", blank_row, all_draws_treatments)
    _pct_row("  Never-ran share of all attempted  :", all_draws_never_ran, total_attempted)
    _pct_row("  Weather disruption rate           :", disrupted, all_draws_treatments)
    print(sep)

    # ── Diagnostic: blank footprint breakdown by TREATMENT_ID (draw 0) ─────────
    blank_footprint_breakdown(results_folder, draw=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sanity-check HSI counts across all scenario draws."
    )
    parser.add_argument("results_folder", type=Path,
                        help="Path to the scenario results folder")
    args = parser.parse_args()
    apply(results_folder=args.results_folder)
