"""
Standalone diagnostics for a SMAC/TLO optimisation run - readable at ANY
point, mid-run or after completion, since it only ever reads
history_log.jsonl from disk. Never touches the live SMAC process, Azure,
or any other running state.

Implements the four checks for confirming a run is actually optimising,
not just running without crashing:

  1. Best-so-far, feasibility-respecting     -> check_best_so_far()
  2. Per-config median DALYs vs proposal order -> check_dalys_trend()
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

from convergence_monitoring import config_key, get_best_feasible_dalys, get_best_dalys_regardless_of_feasibility
from optimisation_parameters import BASELINE_SUMMARY_FILE


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
      - median_dalys, median of every violation column, feasible (all
        violations <= 0)
      - n_seeds (how many times this config was evaluated)

    Violation column names are discovered DYNAMICALLY (any column
    CONTAINING "_violation" - substring, not suffix, since real names
    are period-bucketed, e.g. "hiv_hrh_violation_p1", which never ends
    in the literal text "_violation" - an earlier version of this line
    used .endswith("_violation"), which never matched any real
    constraint name at all) rather than hardcoded - this file cannot
    import optimisation_pipeline.py to get its current constraint list
    (that module runs the live pipeline at import time), so it stays
    correct automatically regardless of how many constraints exist or
    what they're named (currently the period-bucketed
    hiv_hrh_violation_p1..N / hiv_consumable_violation_p1..N plus
    hr_violation/stock_violation).

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

    violation_cols = [c for c in df.columns if "_violation" in c]

    agg_kwargs = {"median_dalys": ("dalys", "median")}
    for col in violation_cols:
        agg_kwargs[col] = (col, "median")
    agg_kwargs["n_seeds"] = ("seed", "count")
    agg_kwargs["first_order"] = ("order", "min")

    grouped = df.groupby("cfg_key").agg(**agg_kwargs).reset_index()

    grouped["feasible"] = True
    for col in violation_cols:
        grouped["feasible"] &= (grouped[col] <= 0)

    grouped = grouped.sort_values("first_order").reset_index(drop=True)
    grouped["proposal_index"] = range(len(grouped))
    return grouped


# --------------------------------------------------------------------------
# Check 1: best-so-far (plumbing / feasibility-gating check)
# --------------------------------------------------------------------------

def check_best_so_far(records: list[dict]) -> pd.DataFrame:
    """
    Best-DALYs-so-far, recomputed incrementally as if each record were
    told to SMAC one at a time, in file order - TWO parallel running
    minimums, "best_feasible_dalys_so_far" (feasibility-gated, via
    get_best_feasible_dalys()) and "best_dalys_so_far_all" (regardless
    of feasibility, via get_best_dalys_regardless_of_feasibility()). An
    earlier version of this function only tracked the feasible one,
    which stays None for as long as nothing has been found feasible
    yet - harmless once something is, but genuinely uninformative
    before that (e.g. early in a high-dimensional or
    tightly-constrained run, where that's expected, not a sign of a
    problem). The "all" column stays useful throughout, even then.

    By construction each column can only stay flat or improve - it's a
    running minimum, so it will look "improving" even under pure random
    search. This confirms the plumbing works end-to-end (submission ->
    postprocessing -> ConstrainedEI -> selection) and that feasibility
    gating is never violated - it does NOT by itself confirm the search
    is smarter than chance. See check_dalys_trend()/
    check_feasibility_trend() for that.
    """
    running_best_feasible = []
    running_best_all = []
    partial_history: list[dict] = []
    for rec in records:
        partial_history.append(rec)
        running_best_feasible.append(get_best_feasible_dalys(partial_history))
        running_best_all.append(get_best_dalys_regardless_of_feasibility(partial_history))
    return pd.DataFrame({
        "best_feasible_dalys_so_far": running_best_feasible,
        "best_dalys_so_far_all": running_best_all,
    })


# --------------------------------------------------------------------------
# Check 2: is the search actually getting better, not just the floor
# --------------------------------------------------------------------------

def check_dalys_trend(records: list[dict], window: int = 10) -> pd.DataFrame:
    """
    Per-config median DALYs vs proposal order, with a rolling median -
    checks whether the SPREAD/MEDIAN of proposed configs is trending
    down over the course of the run, not just the running minimum
    (which check_best_so_far already covers, and which improves even
    under random search).
    """
    cfg_df = _config_frame(records)
    cfg_df["rolling_median_dalys"] = cfg_df["median_dalys"].rolling(window, min_periods=1).median()
    return cfg_df[["proposal_index", "cfg_key", "median_dalys", "n_seeds", "feasible", "rolling_median_dalys"]]


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
    (n_seeds == 1)?

    Computes this TWICE - once restricted to FEASIBLE configs only
    (the "*_feasible" keys - the answer that actually matters once
    something is feasible: is the intensifier spending extra
    seed-confirmations on genuinely promising, USABLE configs), and
    once across ALL configs regardless of feasibility (the "*_all"
    keys). An earlier version of this function computed ONLY the
    feasible-restricted version - harmless once the search has found
    some feasible configs, but produces every count as 0 and every
    DALYs figure as None early in a run (or in a high-dimensional,
    tightly-constrained search) where nothing has been found feasible
    yet, even though the intensified/single-shot split itself is
    already meaningful across ALL proposed configs. The "*_all"
    numbers stay useful throughout; the "*_feasible" ones are the ones
    to trust once they're non-trivial.
    """
    cfg_df = _config_frame(records)

    def _stats(df: pd.DataFrame) -> dict:
        intensified = df[df["n_seeds"] >= intensified_threshold]
        single_shot = df[df["n_seeds"] == 1]
        intensified_median = float(intensified["median_dalys"].median()) if len(intensified) else None
        single_shot_median = float(single_shot["median_dalys"].median()) if len(single_shot) else None
        return {
            "n_intensified": len(intensified),
            "n_single_shot": len(single_shot),
            "intensified_median_dalys": intensified_median,
            "single_shot_median_dalys": single_shot_median,
            "intensified_better": (
                intensified_median < single_shot_median
                if intensified_median is not None and single_shot_median is not None else None
            ),
        }

    feasible_stats = _stats(cfg_df[cfg_df["feasible"]])
    all_stats = _stats(cfg_df)

    return {
        "n_intensified_configs_feasible": feasible_stats["n_intensified"],
        "n_single_shot_configs_feasible": feasible_stats["n_single_shot"],
        "intensified_median_dalys_feasible": feasible_stats["intensified_median_dalys"],
        "single_shot_median_dalys_feasible": feasible_stats["single_shot_median_dalys"],
        "intensified_better_feasible": feasible_stats["intensified_better"],
        "n_intensified_configs_all": all_stats["n_intensified"],
        "n_single_shot_configs_all": all_stats["n_single_shot"],
        "intensified_median_dalys_all": all_stats["intensified_median_dalys"],
        "single_shot_median_dalys_all": all_stats["single_shot_median_dalys"],
        "intensified_better_all": all_stats["intensified_better"],
    }


# --------------------------------------------------------------------------
# Check 5 (plot): HIV DALYs by completion order, new vs intensified,
# feasible vs infeasible, against the baseline
# --------------------------------------------------------------------------

def load_baseline_summary(filepath: str = BASELINE_SUMMARY_FILE) -> dict | None:
    """
    Reads the baseline run's summary (median DALYs, n_runs), written by
    postprocess_output.compute_and_save_baseline_budgets() the last time
    SUBMIT_BASELINE_RUN was True. Returns None if the file doesn't exist
    yet (no baseline run has ever been submitted) - callers should treat
    that as "no baseline to compare against" rather than an error, since
    SUBMIT_BASELINE_RUN is entirely optional.
    """
    path = Path(filepath)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_dalys_by_completion_order(
    records: list[dict],
    baseline_summary_path: str = BASELINE_SUMMARY_FILE,
):
    """
    One point per COMPLETED TRIAL (i.e. one point per raw row in
    history_log.jsonl - NOT grouped by config the way _config_frame()'s
    other checks are, since intensified/new status is itself something
    this plot needs to show per trial, not collapse away).

    x = completion order (file order - same completion-order proxy
        load_history_log()'s own docstring already documents).
    y = that trial's own "dalys" field - already HIV-specific, already
        summed over CONFIG_YEAR_START_DATE-YEAR_END_DATE via
        postprocess_output.py's own TARGET_PERIOD, so no separate
        re-computation is needed here; this is the exact same value
        every other check in this file already reads directly.

    COLOUR: cornflowerblue if this is the FIRST trial (in completion order) to
    evaluate this exact config; mediumblue if an EARLIER trial already
    evaluated it (i.e. this one is SMAC's intensifier confirming an
    already-proposed config with an additional seed). Determined via
    config_key() - the same hashable config identity used everywhere
    else in this pipeline, imported directly from convergence_monitoring.py
    rather than reimplemented.

    MARKER: circle if FEASIBLE (every *_violation column <= 0 for this
    trial), x if INFEASIBLE (any violation > 0) - same feasibility
    convention _config_frame()/get_best_feasible_dalys() already use.

    DASHED LINE: the baseline run's own median DALYs (same metric, same
    period), read from baseline_summary_path via load_baseline_summary()
    - silently omitted (not an error) if no baseline summary is found.

    Returns the matplotlib Figure - caller decides whether to show it
    inline (e.g. in a notebook) or fig.savefig(...) it, rather than this
    function making that choice itself. Requires matplotlib - imported
    HERE, not at module level, so the other four checks in this file
    (which don't need it at all) keep working even if matplotlib isn't
    installed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    violation_cols = [c for c in records[0] if "_violation" in c] if records else []

    # One (xs, ys) list per (new/intensify, feasible/infeasible) bucket -
    # matplotlib's scatter() takes one marker/colour per call, not per
    # point, so points are grouped into (at most) four calls rather than
    # plotted one at a time.
    style = {
        ("new", "feasible"):         dict(color="cornflowerblue",  marker="o", label="New config (feasible)"),
        ("new", "infeasible"):       dict(color="cornflowerblue",  marker="x", label="New config (infeasible)"),
        ("intensify", "feasible"):   dict(color="mediumblue", marker="o", label="Intensified (feasible)"),
        ("intensify", "infeasible"): dict(color="mediumblue", marker="x", label="Intensified (infeasible)"),
    }
    points: dict[tuple, tuple[list, list]] = {bucket: ([], []) for bucket in style}

    seen_configs = set()
    for i, rec in enumerate(records):
        key = config_key(rec["config_object"])
        is_new = key not in seen_configs
        seen_configs.add(key)

        feasible = all(rec[c] <= 0 for c in violation_cols)

        bucket = ("new" if is_new else "intensify", "feasible" if feasible else "infeasible")
        xs, ys = points[bucket]
        xs.append(i)
        ys.append(rec["dalys"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for bucket, (xs, ys) in points.items():
        if xs:
            ax.scatter(xs, ys, **style[bucket])

    # x-axis is a plain completion-order index (0, 1, 2, ...) - force
    # integer-only tick labels, since matplotlib's default locator can
    # otherwise place fractional ticks (0.5, 1.5, ...) when there are
    # few points, which don't correspond to any real run.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    baseline = load_baseline_summary(baseline_summary_path)
    if baseline is not None:
        ax.axhline(
            baseline["median_dalys"], linestyle="--", color="black",
            label=f"Baseline median DALYs ({baseline['n_runs']} runs)",
        )

    ax.set_xlabel("Run (completion order)")
    ax.set_ylabel("HIV DALYs")
    ax.set_title("HIV DALYs by completion order")
    ax.legend()
    return fig


# --------------------------------------------------------------------------
# Run everything, print a readable summary
# --------------------------------------------------------------------------

def run_all_checks(
    filepath: str = "history_log.jsonl",
    window: int = 10,
    intensified_threshold: int = 3,
    verbose: bool = True,
    save_plot: bool = True,
    plot_path: str = "dalys_by_completion_order.png",
) -> dict:
    """
    Runs all four checks against history_log.jsonl and returns a dict of
    results (including the full per-config DataFrames, for further
    inspection/plotting). If verbose (default), also prints a summary
    using the same [tag] convention already established elsewhere in
    this pipeline (optimisation_pipeline.py's [submitted]/[polling]/
    [convergence] markers), so this output is greppable/consistent with
    everything else the pipeline already prints - here tagged [evaluate].

    If save_plot (default), also generates plot_dalys_by_completion_order()
    and saves it to plot_path - an earlier version of this file defined
    that function but never actually called it from here (or anywhere),
    so running the evaluation never produced the plot at all unless it
    was called directly, separately, by name. Wrapped in try/except:
    matplotlib is imported lazily, inside the plotting function itself,
    specifically so a missing matplotlib install doesn't block the
    other four (matplotlib-free) checks above - if it's missing, or the
    plot fails for any other reason, this prints a warning and returns
    results["plot_path"] = None rather than crashing the whole
    evaluation over what's meant to be an additional, optional output.

    Safe to call at ANY point in a run - this is purely a read of
    whatever's already on disk.
    """
    records = load_history_log(filepath)
    cfg_df = _config_frame(records)
    n_trials = len(records)
    n_configs = len(cfg_df)

    best_so_far = check_best_so_far(records)  # now a DataFrame with two
        # columns (best_feasible_dalys_so_far, best_dalys_so_far_all) -
        # an earlier version returned a single Series, feasible-only.
    dalys_trend = check_dalys_trend(records, window=window)
    feas_trend = check_feasibility_trend(records, window=window)
    intens = check_intensification_effect(records, intensified_threshold=intensified_threshold)
        # now returns both "*_feasible" and "*_all" keys - an earlier
        # version only computed the feasible-restricted stats, which
        # were silently all-zero/None for as long as nothing had been
        # found feasible yet.

    final_feasible = best_so_far["best_feasible_dalys_so_far"].iloc[-1] if len(best_so_far) else None
    final_all = best_so_far["best_dalys_so_far_all"].iloc[-1] if len(best_so_far) else None

    results = {
        "n_trials": n_trials,
        "n_distinct_configs": n_configs,
        "best_feasible_dalys_final": float(final_feasible) if final_feasible is not None else None,
        "best_dalys_final_all": float(final_all) if final_all is not None else None,
        "best_so_far": best_so_far,
        "dalys_trend": dalys_trend,
        "feasibility_trend": feas_trend,
        "intensification_effect": intens,
        "plot_path": None,
    }

    if save_plot:
        try:
            fig = plot_dalys_by_completion_order(records)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            results["plot_path"] = plot_path
            if verbose:
                print(f"[evaluate] DALYs-by-completion-order plot saved to {plot_path}")
        except Exception as exc:
            # Deliberately never lets a plotting failure (e.g. matplotlib
            # not installed) take down the rest of the evaluation - the
            # four checks above are the primary output; the plot is an
            # additional, optional one.
            print(f"[evaluate] WARNING: could not generate/save the DALYs plot ({exc!r}) - "
                  f"skipping it, other checks unaffected.")

    if verbose:
        print(f"[evaluate] {n_trials} trial(s) logged, {n_configs} distinct config(s)")
        print(f"[evaluate] best feasible DALYs so far: {results['best_feasible_dalys_final']}")
        print(f"[evaluate] best DALYs so far (regardless of feasibility): {results['best_dalys_final_all']}")

        if n_configs < 2:
            print("[evaluate] fewer than 2 distinct configs so far - too early for trend checks.")
            return results

        half = max(1, n_configs // 2)

        early_dalys = dalys_trend["median_dalys"].iloc[:half].median()
        late_dalys = dalys_trend["median_dalys"].iloc[half:].median()
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

        print(f"[evaluate] intensified (n_seeds>={intensified_threshold}) configs, ALL: "
              f"{intens['n_intensified_configs_all']}, median DALYs = {intens['intensified_median_dalys_all']}")
        print(f"[evaluate] single-shot (n_seeds==1) configs, ALL: "
              f"{intens['n_single_shot_configs_all']}, median DALYs = {intens['single_shot_median_dalys_all']}")
        if intens["intensified_better_all"] is True:
            print("[evaluate]   -> intensified configs ARE better on average, among ALL configs (good sign)")
        elif intens["intensified_better_all"] is False:
            print("[evaluate]   -> intensified configs are NOT better on average, among ALL configs - worth investigating")
        else:
            print("[evaluate]   -> not enough data in one or both groups yet to compare (ALL configs)")

        print(f"[evaluate] intensified (n_seeds>={intensified_threshold}) FEASIBLE configs: "
              f"{intens['n_intensified_configs_feasible']}, median DALYs = {intens['intensified_median_dalys_feasible']}")
        print(f"[evaluate] single-shot (n_seeds==1) FEASIBLE configs: "
              f"{intens['n_single_shot_configs_feasible']}, median DALYs = {intens['single_shot_median_dalys_feasible']}")
        if intens["intensified_better_feasible"] is True:
            print("[evaluate]   -> intensified configs ARE better on average, among FEASIBLE configs (good sign)")
        elif intens["intensified_better_feasible"] is False:
            print("[evaluate]   -> intensified configs are NOT better on average, among FEASIBLE configs - worth investigating")
        else:
            print("[evaluate]   -> not enough FEASIBLE data in one or both groups yet to compare")

    return results


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "history_log.jsonl"
    run_all_checks(log_path)
