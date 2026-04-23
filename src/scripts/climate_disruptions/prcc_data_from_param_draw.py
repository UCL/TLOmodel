"""
prcc_disruption_precompute.py

Lightweight parameter-scan processor.
Streams through all 200 draws one at a time, writing prcc_disruption_summary.csv
incrementally and checkpointing each draw so a killed run can resume.

Does NOT produce any plots or accumulate DataFrames in RAM.

Usage:
    python prcc_disruption_precompute.py <results_folder>

Outputs (all written to results_folder):
    prcc_disruption_summary.csv          — one row per draw, ready for PRCC script
    draw_checkpoints/draw_<N>.pkl        — lightweight per-draw checkpoints
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd

from tlo import Date
from tlo.analysis.utils import extract_results

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  — edit these to match your run
# ─────────────────────────────────────────────────────────────────────────────

N_DRAWS = 199
MIN_YEAR = 2025
MAX_YEAR = 2041  # exclusive, so runs 2025–2040 inclusive
SCALING_FACTOR = 145.39
CI_LOWER = 0.025
CI_UPPER = 0.975


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ym(index):
    return index.astype(str).str.split(":", n=1).str[0]


def _parse_facility(index):
    return index.astype(str).str.split(":", n=2).str[1]


def _collapse_hsi_types(df):
    if df.empty:
        return df
    key_2 = _parse_ym(df.index) + ":" + _parse_facility(df.index)
    return df.groupby(key_2).sum()


def _align_and_rate(total_df, disrupted_df, delayed_df, cancelled_df):
    idx = total_df.index.union(disrupted_df.index)
    t = total_df.reindex(idx, fill_value=0)
    d = disrupted_df.reindex(idx, fill_value=0)
    t = t + delayed_df.reindex(idx, fill_value=0) + cancelled_df.reindex(idx, fill_value=0)
    return d.div(t).where(t > 0, 0.0).clip(upper=1.0)


def _annual_stats(rate_df):
    annual = rate_df.groupby(_parse_ym(rate_df.index).str[:4]).mean().sort_index()
    return annual.mean(axis=1), annual.quantile(CI_LOWER, axis=1), annual.quantile(CI_UPPER, axis=1)


def _concat_years(dfs):
    return pd.concat(dfs).groupby(level=0).sum()


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


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def apply(results_folder: Path, output_folder: Path):
    scenarios_of_interest = list(range(N_DRAWS))
    target_year_sequence = range(MIN_YEAR, MAX_YEAR)

    ckpt_dir = output_folder.parent / f"{output_folder.name}_draw_checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    prcc_out = output_folder / "prcc_disruption_summary.csv"

    # ── Identify draws that still need processing ─────────────────────────────
    draws_todo = [d for d in scenarios_of_interest
                  if not (ckpt_dir / f"draw_{d}.pkl").exists()]

    if not draws_todo:
        print("All draws already checkpointed — nothing to do.")
        print(f"PRCC CSV: {prcc_out}")
        return

    print(f"{len(draws_todo)} draws to process "
          f"({N_DRAWS - len(draws_todo)} already cached).")

    # ─────────────────────────────────────────────────────────────────────────
    #  PRE-LOAD: one year × one log key at a time.
    #  Slice immediately per draw, del the raw full DataFrame.
    #  Only load years/keys needed by draws_todo.
    # ─────────────────────────────────────────────────────────────────────────

    per_draw_total = {d: [] for d in draws_todo}
    per_draw_delayed = {d: [] for d in draws_todo}
    per_draw_cancelled = {d: [] for d in draws_todo}

    print("Pre-loading raw results …")
    for target_year in target_year_sequence:
        print(f"  year {target_year}")
        TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))

        raw = extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="hsi_event_counts_by_facility_monthly",
            custom_generate_series=_make_hsi_counts_by_real_facility_monthly(TARGET_PERIOD),
            do_scaling=False,
        )
        for d in draws_todo:
            per_draw_total[d].append(raw[d].fillna(0))
        del raw

        raw = extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="Weather_delayed_HSI_Event_full_info",
            custom_generate_series=_make_disrupted_by_real_facility_monthly(TARGET_PERIOD),
            do_scaling=False,
        )
        for d in draws_todo:
            per_draw_delayed[d].append(raw[d].fillna(0))
        del raw

        raw = extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="Weather_cancelled_HSI_Event_full_info",
            custom_generate_series=_make_disrupted_by_real_facility_monthly(TARGET_PERIOD),
            do_scaling=False,
        )
        for d in draws_todo:
            per_draw_cancelled[d].append(raw[d].fillna(0))
        del raw

    print("Pre-loading complete. Processing draws …")

    # ─────────────────────────────────────────────────────────────────────────
    #  PER-DRAW: compute stats, write PRCC row, checkpoint, free RAM
    # ─────────────────────────────────────────────────────────────────────────

    for draw in draws_todo:
        print(draw)

        total_all = _concat_years(per_draw_total[draw]) * SCALING_FACTOR
        delayed_all = _concat_years(per_draw_delayed[draw]) * SCALING_FACTOR
        cancelled_all = _concat_years(per_draw_cancelled[draw]) * SCALING_FACTOR

        # Collapse ym:facility:hsi → ym:facility  (much smaller)
        total_2 = _collapse_hsi_types(total_all)
        delayed_2 = _collapse_hsi_types(delayed_all)
        cancelled_2 = _collapse_hsi_types(cancelled_all)
        del total_all, delayed_all, cancelled_all

        delayed_rate_2 = _align_and_rate(total_2, delayed_2, delayed_2, cancelled_2)
        cancelled_rate_2 = _align_and_rate(total_2, cancelled_2, delayed_2, cancelled_2)
        del total_2, delayed_2, cancelled_2

        dam, dal, dau = _annual_stats(delayed_rate_2)
        cam, cal, cau = _annual_stats(cancelled_rate_2)
        del delayed_rate_2, cancelled_rate_2

        # ── Write PRCC row immediately ────────────────────────────────────────
        pd.DataFrame([{
            "draw": draw,
            "prop_delayed": dam.mean(),
            "prop_cancelled": cam.mean(),
        }]).to_csv(prcc_out, mode="a", header=not prcc_out.exists(), index=False)

        # ── Checkpoint (tiny — just the annual Series) ────────────────────────
        ckpt_path = ckpt_dir / f"draw_{draw}.pkl"
        with open(ckpt_path, "wb") as f:
            pickle.dump({
                "draw": draw,
                "annual_delayed_mean": dam, "annual_delayed_lower": dal,
                "annual_delayed_upper": dau, "annual_cancelled_mean": cam,
                "annual_cancelled_lower": cal, "annual_cancelled_upper": cau,
            }, f, protocol=4)

        # ── Free this draw's pre-loaded slices ────────────────────────────────
        del per_draw_total[draw], per_draw_delayed[draw], per_draw_cancelled[draw]
        del dam, dal, dau, cam, cal, cau

    print(f"\nDone. PRCC CSV written to:\n  {prcc_out}")
    print(f"Checkpoints in:\n  {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute PRCC disruption summary for parameter scan draws."
    )
    parser.add_argument("results_folder", type=Path)
    parser.add_argument("--output_folder", type=Path, default=None,
                        help="Where to write outputs (default: same as results_folder)")
    args = parser.parse_args()
    out = args.output_folder or args.results_folder
    apply(results_folder=args.results_folder, output_folder=out)
