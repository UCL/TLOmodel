"""
Standalone diagnostics for a SMAC/TLO optimisation run - readable at ANY
point, mid-run or after completion, since it only ever reads
history_log.jsonl from disk. Never touches the live SMAC process, Azure,
or any other running state.

Implements the four checks for confirming a run is actually optimising,
not just running without crashing:

  1. Best-so-far, feasibility-respecting     -> check_best_so_far()
  2. Per-config mean DALYs vs proposal order -> check_dalys_trend()
  3. Feasible fraction vs proposal order     -> check_feasibility_trend()
  4. Intensified vs single-shot configs      -> check_intensification_effect()

These checks are always relevant - not just for a one-off small test -
worth re-running periodically against a live run's history_log.jsonl, or
at the end of a full-scale run, to catch a degraded search early.

Usage:
    python evaluate_pipeline_run.py [path-to-history_log.jsonl]
or import run_all_checks() / any individual check function elsewhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from convergence_monitoring import config_key, get_best_feasible_dalys


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_history_log(filepath: str = "history_log.jsonl") -> list[dict]:
    """
    Reads history_log.jsonl from disk, in file order - a proxy for
    completion order (not necessarily identical to proposal order under
    concurrency, since N_CONCURRENT trials can complete out of the order
    they were proposed in, but close enough for trend-checking).

    Safe to call while the pipeline is still running: history_log.jsonl
    is append-only, and this only reads whatever's been flushed so far.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - has the pipeline logged any results yet?")

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"{path} exists but is empty - no results logged yet.")
    return records


def _config_frame(records: list[dict]) -> pd.DataFrame:
    """
    Groups raw per-(config, seed) log records by config - NOT raw rows,
    since an intensified config's repeated seeds would otherwise look
    like multiple independent proposals and distort every trend check
    below. One row per distinct config, in the order each was first
    seen in the log (proposal_index), with:
      - mean_dalys, mean violations, feasible (all violations <= 0)
      - n_seeds (how many times this config was evaluated)

    config_key() and get_best_feasible_dalys() are imported directly
    from convergence_monitoring.py - the same functions the live
    pipeline itself uses - so these checks validate against the
    pipeline's own definitions, not separately reinvented ones. Both
    already accept plain dicts (not just ConfigSpace Configuration
    objects), which is what history_log.jsonl's JSON-deserialised
    config_object entries actually are.
    """
    df = pd.DataFrame(records)
    df["cfg_key"] = df["config_object"].apply(config_key)
    df["order"] = range(len(df))  # file order = completion-order proxy

    grouped = df.groupby("cfg_key").agg(
        mean_dalys=("dalys", "mean"),
        cost_violation=("cost_violation", "mean"),
        hr_violation=("hr_violation", "mean"),
        stock_violation=("stock_violation", "mean"),
        n_seeds=("seed", "count"),
        first_order=("order", "min"),
    ).reset_index()

    grouped["feasible"] = (
        (grouped["cost_violation"] <= 0)
        & (grouped["hr_violation"] <= 0)
        & (grouped["stock_violation"] <= 0)
    )
    grouped = grouped.sort_values("first_order").reset_index(drop=True)
    grouped["proposal_index"] = range(len(grouped))
    return grouped


# --------------------------------------------------------------------------
# Check 1: best-so-far (plumbing / feasibility-gating check)
# --------------------------------------------------------------------------

def check_best_so_far(records: list[dict]) -> pd.Series:
    """
    Best-feasible-DALYs-so-far, recomputed incrementally as if each
    record were told to SMAC one at a time, in file order.

    By construction this can only stay flat or improve - it's a running
    minimum, so it will look "improving" even under pure random search.
    This confirms the plumbing works end-to-end (submission ->
    postprocessing -> ConstrainedEI -> selection) and that feasibility
    gating is never violated - it does NOT by itself confirm the search
    is smarter than chance. See check_dalys_trend()/
    check_feasibility_trend() for that.
    """
    running_best = []
    partial_history: list[dict] = []
    for rec in records:
        partial_history.append(rec)
        running_best.append(get_best_feasible_dalys(partial_history))
    return pd.Series(running_best, name="best_feasible_dalys_so_far")


# --------------------------------------------------------------------------
# Check 2: is the search actually getting better, not just the floor
# --------------------------------------------------------------------------

def check_dalys_trend(records: list[dict], window: int = 10) -> pd.DataFrame:
    """
    Per-config mean DALYs vs proposal order, with a rolling median -
    checks whether the SPREAD/MEDIAN of proposed configs is trending
    down over the course of the run, not just the running minimum
    (which check_best_so_far already covers, and which improves even
    under random search).
    """
    cfg_df = _config_frame(records)
    cfg_df["rolling_median_dalys"] = cfg_df["mean_dalys"].rolling(window, min_periods=1).median()
    return cfg_df[["proposal_index", "cfg_key", "mean_dalys", "n_seeds", "feasible", "rolling_median_dalys"]]


# --------------------------------------------------------------------------
# Check 3: is the constraint-learning actually steering the search
# --------------------------------------------------------------------------

def check_feasibility_trend(records: list[dict], window: int = 10) -> pd.DataFrame:
    """
    Feasible fraction vs proposal order (per config, not per trial). If
    ConstrainedEI's P(feasible) term is actually steering proposals, the
    rolling feasible fraction should trend up over the run rather than
    staying flat throughout.
    """
    cfg_df = _config_frame(records)
    cfg_df["rolling_feasible_fraction"] = (
        cfg_df["feasible"].astype(float).rolling(window, min_periods=1).mean()
    )
    return cfg_df[["proposal_index", "cfg_key", "feasible", "rolling_feasible_fraction"]]


# --------------------------------------------------------------------------
# Check 4: is the intensifier actually discriminating
# --------------------------------------------------------------------------

def check_intensification_effect(records: list[dict], intensified_threshold: int = 3) -> dict:
    """
    Are heavily-intensified configs (n_seeds >= intensified_threshold)
    disproportionately GOOD ones, compared to single-shot configs
    (n_seeds == 1)? If the intensifier is discriminating correctly -
    spending extra seed-confirmations on genuinely promising configs -
    intensified configs' mean DALYs (among feasible configs) should
    skew better than single-shot ones. If the two groups look
    statistically indistinguishable, the intensifier isn't
    discriminating - it's just resampling arbitrarily.
    """
    cfg_df = _config_frame(records)
    feasible_df = cfg_df[cfg_df["feasible"]]

    intensified = feasible_df[feasible_df["n_seeds"] >= intensified_threshold]
    single_shot = feasible_df[feasible_df["n_seeds"] == 1]

    intensified_mean = float(intensified["mean_dalys"].mean()) if len(intensified) else None
    single_shot_mean = float(single_shot["mean_dalys"].mean()) if len(single_shot) else None

    return {
        "n_intensified_configs": len(intensified),
        "n_single_shot_configs": len(single_shot),
        "intensified_mean_dalys": intensified_mean,
        "single_shot_mean_dalys": single_shot_mean,
        "intensified_better": (
            intensified_mean < single_shot_mean
            if intensified_mean is not None and single_shot_mean is not None else None
        ),
    }


# --------------------------------------------------------------------------
# Run everything, print a readable summary
# --------------------------------------------------------------------------

def run_all_checks(
    filepath: str = "history_log.jsonl",
    window: int = 10,
    intensified_threshold: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Runs all four checks against history_log.jsonl and returns a dict of
    results (including the full per-config DataFrames, for further
    inspection/plotting). If verbose (default), also prints a summary
    using the same [tag] convention already established elsewhere in
    this pipeline (optimisation_pipeline.py's [submitted]/[polling]/
    [convergence] markers), so this output is greppable/consistent with
    everything else the pipeline already prints - here tagged [evaluate].

    Safe to call at ANY point in a run - this is purely a read of
    whatever's already on disk.
    """
    records = load_history_log(filepath)
    cfg_df = _config_frame(records)
    n_trials = len(records)
    n_configs = len(cfg_df)

    best_so_far = check_best_so_far(records)
    dalys_trend = check_dalys_trend(records, window=window)
    feas_trend = check_feasibility_trend(records, window=window)
    intens = check_intensification_effect(records, intensified_threshold=intensified_threshold)

    results = {
        "n_trials": n_trials,
        "n_distinct_configs": n_configs,
        "best_feasible_dalys_final": float(best_so_far.iloc[-1]) if best_so_far.iloc[-1] is not None else None,
        "best_so_far": best_so_far,
        "dalys_trend": dalys_trend,
        "feasibility_trend": feas_trend,
        "intensification_effect": intens,
    }

    if verbose:
        print(f"[evaluate] {n_trials} trial(s) logged, {n_configs} distinct config(s)")
        print(f"[evaluate] best feasible DALYs so far: {results['best_feasible_dalys_final']}")

        if n_configs < 2:
            print("[evaluate] fewer than 2 distinct configs so far - too early for trend checks.")
            return results

        half = max(1, n_configs // 2)

        early_dalys = dalys_trend["mean_dalys"].iloc[:half].median()
        late_dalys = dalys_trend["mean_dalys"].iloc[half:].median()
        print(f"[evaluate] median DALYs, first half of configs proposed:  {early_dalys:.4f}")
        print(f"[evaluate] median DALYs, second half of configs proposed: {late_dalys:.4f}")
        print("[evaluate]   -> trending down (good sign)" if late_dalys < early_dalys
              else "[evaluate]   -> NOT trending down - worth investigating")

        early_feas = feas_trend["feasible"].iloc[:half].mean()
        late_feas = feas_trend["feasible"].iloc[half:].mean()
        print(f"[evaluate] feasible fraction, first half of configs proposed:  {early_feas:.1%}")
        print(f"[evaluate] feasible fraction, second half of configs proposed: {late_feas:.1%}")
        print("[evaluate]   -> trending up or stable (good sign)" if late_feas >= early_feas
              else "[evaluate]   -> trending DOWN - worth investigating")

        print(f"[evaluate] intensified (n_seeds>={intensified_threshold}) feasible configs: "
              f"{intens['n_intensified_configs']}, mean DALYs = {intens['intensified_mean_dalys']}")
        print(f"[evaluate] single-shot (n_seeds==1) feasible configs: "
              f"{intens['n_single_shot_configs']}, mean DALYs = {intens['single_shot_mean_dalys']}")
        if intens["intensified_better"] is True:
            print("[evaluate]   -> intensified configs ARE better on average (good sign)")
        elif intens["intensified_better"] is False:
            print("[evaluate]   -> intensified configs are NOT better on average - worth investigating")
        else:
            print("[evaluate]   -> not enough data in one or both groups yet to compare")

    return results


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "history_log.jsonl"
    run_all_checks(log_path)
