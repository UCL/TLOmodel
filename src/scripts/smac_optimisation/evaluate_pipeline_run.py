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
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from convergence_monitoring import config_key, get_best_feasible_dalys, get_best_dalys_regardless_of_feasibility
from optimisation_parameters import BASELINE_SUMMARY_FILE, VALID_PRIOR_RUN_COMMITS
from postprocess_output import TARGET_PERIOD  # the SAME authoritative period
    # every "dalys" value in history_log.jsonl is already computed over -
    # imported directly (matching initialise.py's own convention) rather
    # than reconstructed from CONFIG_YEAR_START_DATE/YEAR_END_DATE here,
    # which would risk the exact kind of silent desync initialise.py's
    # own comment on this warns about.


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _get_current_commit_for_filtering() -> str | None:
    """
    Lightweight, independent commit lookup for this file's own
    commit-filtering purposes - deliberately NOT
    optimisation_pipeline._get_commit() (which requires the repo to be
    clean AND pushed, via is_file_clean() - appropriate for submission-
    time reproducibility guarantees, not for a read-only diagnostic
    someone might run against a repo with uncommitted local changes).
    Also deliberately does NOT import optimisation_pipeline.py at all -
    that module's own top-level code submits jobs and runs the live
    ask-tell loop at import time, so importing it here (just to reuse
    one function) would accidentally start a second, unwanted
    optimisation run - directly contradicting this file's own "never
    touches the live process" design.

    search_parent_directories=True is REQUIRED here, not optional -
    GitPython's Repo() constructor defaults to False, meaning it only
    checks the EXACT given path for a .git directory, with no upward
    search through parent directories at all. Without this, running
    this file from anywhere other than the repo root itself (a very
    real possibility - this is a standalone diagnostic script, plausibly
    run from src/scripts/smac_optimisation/, where it actually lives)
    raises InvalidGitRepositoryError - silently caught below, returning
    None, which callers correctly treat as "can't filter" - but this
    meant commit filtering was effectively NEVER running for anyone not
    invoking this from the exact repo root, with only an easy-to-miss
    printed warning as the only sign anything was wrong. Confirmed as
    the actual cause of a real "filtering doesn't seem to be excluding
    anything" report.

    Returns None (not raises) if this isn't run from inside a git repo
    at all (even searching upward), or git itself isn't available -
    callers should treat that as "commit filtering isn't possible right
    now" and either skip it or warn, not crash the whole evaluation
    over it.
    """
    try:
        from git import Repo
        return Repo(".", search_parent_directories=True).head.commit.hexsha
    except Exception:
        return None


def load_history_log(
    filepath: str = "history_log.jsonl",
    filter_by_commit: bool = True,
) -> list[dict]:
    """
    Reads history_log.jsonl from disk, in file order - a proxy for
    completion order (not necessarily identical to proposal order under
    concurrency, since N_CONCURRENT trials can complete out of the order
    they were proposed in, but close enough for trend-checking).

    Safe to call while the pipeline is still running: history_log.jsonl
    is append-only, and this only reads whatever's been flushed so far.

    filter_by_commit=True (default): keeps only records whose own
    "commit" field is EITHER the current commit or listed in
    VALID_PRIOR_RUN_COMMITS (optimisation_parameters.py) - the SAME
    criteria optimisation_pipeline.recover_from_job_log() applies when
    deciding which completed jobs to recover into the live run, applied
    here too so the evaluation/plotting checks in this file are
    consistent with what the live pipeline itself would actually trust.
    An earlier version of this function read every record unconditionally,
    with no commit awareness at all - meaning a run submitted under an
    untrusted, unrelated commit (different smac_scenario.py,
    incomparable results) could silently appear in the plot/evaluation
    even though the live pipeline would never have recovered it as a
    prior run. Records missing a "commit" field entirely are treated
    the same as an unrecognised commit (excluded) - matching
    recover_from_job_log()'s own "don't silently trust unknown
    provenance" behaviour, not a special case.

    If the current commit can't be determined (not run from inside a
    git repo, or git unavailable), filtering is skipped entirely, with
    a printed warning - every record is kept rather than raising,
    since this file is meant to be usable defensively/diagnostically
    even outside a fully-set-up environment.
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

    if filter_by_commit:
        current_commit = _get_current_commit_for_filtering()
        if current_commit is None:
            print(
                "[evaluate] WARNING: could not determine the current git commit "
                "(not run from inside a git repo, or git unavailable) - skipping "
                "commit filtering, keeping all records as-is."
            )
        else:
            candidate_commits = {current_commit, *VALID_PRIOR_RUN_COMMITS}
            n_before = len(records)
            records = [r for r in records if r.get("commit") in candidate_commits]
            n_excluded = n_before - len(records)
            if n_excluded:
                print(
                    f"[evaluate] excluded {n_excluded} of {n_before} record(s) - submitted "
                    f"under a commit that's neither the current commit ({current_commit[:12]}) "
                    f"nor listed in VALID_PRIOR_RUN_COMMITS."
                )
            if not records:
                raise ValueError(
                    f"{path} has {n_before} record(s), but none match the current commit "
                    f"or VALID_PRIOR_RUN_COMMITS - nothing left to evaluate. Pass "
                    f"filter_by_commit=False to bypass this (e.g. to inspect old results "
                    f"regardless of commit provenance)."
                )

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

def _submission_datetime_from_job_id(job_id) -> datetime | None:
    """
    Extracts the submission timestamp embedded directly in a real
    trial's own job_id - submit_azure_job()'s own construction is
    "<filename>-<timestamp>-<uuid>", where timestamp is
    datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ") - genuinely the
    moment of submission, not completion, and (being zero-padded,
    most-significant-first) also lexicographically sortable as-is, were
    that ever preferable to parsing.

    Returns None if job_id is None/falsy, or doesn't contain a
    substring matching that exact pattern - notably true for
    PRIOR_RUNS entries warm-started via seed_history_with_prior_runs()
    (job_id=run.get("job_id"), which is None unless the prior run dict
    happened to include one) - callers should treat None as "no
    reliable submission-order position available for this record",
    not assume it means something else.

    Uses re.search (not a fixed-position slice), so this also correctly
    finds the timestamp regardless of where in the string it sits - the
    baseline job's own id inserts "-baseline-" before its timestamp
    rather than right after the filename, and checkpoint jobs use an
    entirely different scheme (checkpoint_seeds.checkpoint_job_id())
    with no timestamp at all, correctly returning None for those too.
    """
    if not job_id:
        return None
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{6}Z)", job_id)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%dT%H%M%SZ")
    except ValueError:
        return None


def print_config_group_details(records: list[dict], only_multi: bool = True) -> None:
    """
    Diagnostic: prints every group of records sharing the same
    config_key() (i.e. everything the rest of this file, and the live
    pipeline's own final selection, treats as "the same config") -
    each group showing its member index (completion order), seed, and
    the FULL config_object dict, so you can directly eyeball whether
    grouped records genuinely have identical hyperparameter values, or
    whether something's wrong with the grouping itself.

    only_multi=True (default): only prints groups with 2+ members - the
    ones actually worth checking (a group of 1 has nothing to compare
    against). Set False to also see every single-member group.

    This exists specifically because "two points are shown with the
    same colour/connected by a line in plot_dalys_by_completion_order()"
    is a MECHANICAL CONSEQUENCE of config_key() computing the same key
    for them - confirming that visually doesn't tell you whether the
    grouping is CORRECT (genuinely the same config, e.g. SMAC's own
    intensifier asking for the same challenger twice, concurrently,
    before either result is known) or a BUG (config_key(), or the
    underlying config_object data itself, incorrectly treating
    different configs as identical) - only looking at the actual
    parameter values settles that.
    """
    grouped: dict[tuple, list[int]] = {}
    for i, rec in enumerate(records):
        grouped.setdefault(config_key(rec["config_object"]), []).append(i)

    for key, indices in grouped.items():
        if only_multi and len(indices) < 2:
            continue
        print(f"=== config_key group ({len(indices)} member(s)) ===")
        for i in indices:
            rec = records[i]
            print(f"  [{i}] seed={rec.get('seed')} dalys={rec.get('dalys')}")
            print(f"      config_object={rec['config_object']}")
        print()


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
    color_by: str = "intensification",
    order_by: str = "completion",
):
    """
    One point per COMPLETED TRIAL (i.e. one point per raw row in
    history_log.jsonl - NOT grouped by config the way _config_frame()'s
    other checks are, since intensified/new status is itself something
    this plot needs to show per trial, not collapse away).

    x = run order - "completion" (default, file order, the same
        completion-order proxy load_history_log()'s own docstring
        already documents) or "submission" (the order jobs were
        actually SUBMITTED to Azure, not the order they happened to
        finish in - meaningfully different under N_CONCURRENT, where a
        fast job submitted later can complete before a slow job
        submitted earlier). "submission" order is recovered from the
        real timestamp embedded directly in each trial's own job_id
        (see _submission_datetime_from_job_id()) - records with no
        parseable timestamp (PRIOR_RUNS entries warm-started without a
        recorded job_id, most commonly) are EXCLUDED from the plot in
        this mode, with a printed warning naming how many - there's no
        honest position to place them at.
    y = that trial's own "dalys" field - already HIV-specific, already
        summed over CONFIG_YEAR_START_DATE-YEAR_END_DATE via
        postprocess_output.py's own TARGET_PERIOD, so no separate
        re-computation is needed here; this is the exact same value
        every other check in this file already reads directly.

    color_by="intensification" (default, unchanged from before):
        COLOUR distinguishes new (cornflowerblue) vs intensified
        (mediumblue) trials.

    color_by="config": each DISTINCT config gets its OWN colour (drawn
        from a qualitative colormap, cycling if there are more than 20
        distinct configs), so every occurrence of the same config -
        including its intensified repeats - shares a colour. A thin
        line also connects consecutive occurrences of the SAME config,
        in that config's own colour, so which earlier point a given
        intensification is confirming is directly traceable by eye,
        not just inferable from closely-matching colours - the
        motivating use case for this mode. No per-config legend (would
        be unreadable with more than a handful of distinct configs) -
        the colour/line pairing carries the grouping, not a legend key.

    Note: "new"/"intensified" (color_by="intensification") and each
    config's own point-connecting order (color_by="config") are BOTH
    determined by iterating `records` in whatever order this function
    is left in after the order_by step below - so under
    order_by="submission", "new" correctly means "first SUBMITTED",
    and a config's connecting line correctly traces submission order
    too, not completion order silently leaking back in.

    Either mode: MARKER is circle if FEASIBLE (every *_violation column
    <= 0 for this trial), x if INFEASIBLE (any violation > 0) - same
    feasibility convention _config_frame()/get_best_feasible_dalys()
    already use.

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

    if color_by not in ("intensification", "config"):
        raise ValueError(f"color_by must be 'intensification' or 'config', got {color_by!r}")
    if order_by not in ("completion", "submission"):
        raise ValueError(f"order_by must be 'completion' or 'submission', got {order_by!r}")

    if order_by == "submission":
        timestamped = [(rec, _submission_datetime_from_job_id(rec.get("job_id"))) for rec in records]
        n_missing = sum(1 for _, ts in timestamped if ts is None)
        if n_missing:
            print(
                f"[evaluate] WARNING: {n_missing} of {len(records)} record(s) have no parseable "
                f"submission timestamp in their job_id (commonly warm-started PRIOR_RUNS entries) "
                f"- excluded from this plot under order_by='submission'."
            )
        records = [rec for rec, ts in sorted(
            ((rec, ts) for rec, ts in timestamped if ts is not None),
            key=lambda pair: pair[1],
        )]

    violation_cols = [c for c in records[0] if "_violation" in c] if records else []

    fig, ax = plt.subplots(figsize=(10, 6))

    if color_by == "intensification":
        # One (xs, ys) list per (new/intensify, feasible/infeasible)
        # bucket - matplotlib's scatter() takes one marker/colour per
        # call, not per point, so points are grouped into (at most)
        # four calls rather than plotted one at a time.
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

        for bucket, (xs, ys) in points.items():
            if xs:
                ax.scatter(xs, ys, **style[bucket])

    else:  # color_by == "config"
        # Assign each distinct config a colour, in the order it FIRST
        # appears (so replotting the same history is stable), and
        # collect every (x, y) occurrence of that config, IN completion
        # order, so a line can be drawn connecting them.
        #
        # tab20 is structured as 10 hue-PAIRS (slots 0-1 are dark/light
        # blue, 2-3 dark/light orange, etc.) - an earlier version of
        # this indexed straight through (0, 1, 2, 3, ...), which meant
        # the FIRST TWO distinct configs ever seen always landed on the
        # SAME hue-pair (e.g. dark blue + light blue) purely by
        # construction, regardless of whether those two configs are
        # actually similar - easily misread as "these configs look
        # alike" when it's really just colormap structure. Indexing
        # through all 10 dark (even) slots FIRST, then all 10 light
        # (odd) slots, means the first 10 distinct configs get 10
        # genuinely different hues before any hue is ever reused.
        cmap = plt.colormaps["tab20"]
        color_order = list(range(0, cmap.N, 2)) + list(range(1, cmap.N, 2))
        config_colors: dict[tuple, tuple] = {}
        config_points: dict[tuple, tuple[list, list]] = {}

        for i, rec in enumerate(records):
            key = config_key(rec["config_object"])
            if key not in config_colors:
                config_colors[key] = cmap(color_order[len(config_colors) % cmap.N])
                config_points[key] = ([], [])
            config_points[key][0].append(i)
            config_points[key][1].append(rec["dalys"])

        for key, (xs, ys) in config_points.items():
            color = config_colors[key]
            # Connecting line FIRST (drawn underneath), thin and
            # semi-transparent so it reads as a grouping cue rather
            # than competing visually with the points themselves. Only
            # draws anything for configs with 2+ occurrences - a
            # single-point config has nothing to connect.
            if len(xs) > 1:
                ax.plot(xs, ys, color=color, linewidth=1, alpha=0.5, zorder=1)

        for i, rec in enumerate(records):
            key = config_key(rec["config_object"])
            feasible = all(rec[c] <= 0 for c in violation_cols)
            ax.scatter(
                i, rec["dalys"], color=config_colors[key],
                marker="o" if feasible else "x", zorder=2,
            )

    # x-axis is a plain completion-order index (0, 1, 2, ...) - force
    # integer-only tick labels, since matplotlib's default locator can
    # otherwise place fractional ticks (0.5, 1.5, ...) when there are
    # few points, which don't correspond to any real run.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    baseline = load_baseline_summary(baseline_summary_path)
    legend_handles = []
    if baseline is not None:
        baseline_line = ax.axhline(
            baseline["median_dalys"], linestyle="--", color="black",
            label=f"Baseline median DALYs ({baseline['n_runs']} runs)",
        )
        legend_handles.append(baseline_line)

    if color_by == "config":
        # color_by="config" deliberately has NO per-config legend entry
        # (unreadable with more than a handful of distinct configs - see
        # this function's own docstring) - but marker SHAPE still needs
        # explaining somewhere, since nothing else on the plot says what
        # a circle vs an x means. Generic grey proxy markers, not tied
        # to any specific config's own colour.
        from matplotlib.lines import Line2D
        legend_handles += [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="grey",
                   markeredgecolor="grey", label="Feasible"),
            Line2D([0], [0], marker="x", color="none", markeredgecolor="grey",
                   label="Infeasible"),
        ]

    ax.set_xlabel(f"Run ({order_by} order)")
    ax.set_ylabel(f"HIV DALYs ({TARGET_PERIOD[0].year}\u2013{TARGET_PERIOD[1].year})")
    ax.set_title(f"HIV DALYs by {order_by} order")
    if color_by == "intensification":
        # Plain ax.legend() with no handles= arg: auto-detects every
        # labelled artist, which is the four scatter buckets (each with
        # its own label=) PLUS the baseline line if present - explicitly
        # passing handles=legend_handles here would DROP the four
        # scatter labels, keeping only whatever was appended to
        # legend_handles (the config-mode-only marker proxies never
        # apply here, and legend_handles otherwise holds just the
        # baseline line) - a real bug an earlier version of this
        # function had.
        ax.legend()
    else:
        # color_by="config": the scatter calls have no label= at all
        # (avoiding a per-config legend entry), so auto-detection would
        # show nothing useful here - pass the explicit marker-shape
        # proxies (+ baseline line, if present) instead.
        ax.legend(handles=legend_handles)
    return fig


# --------------------------------------------------------------------------
# Check 6: is between-config spread real signal, or just seed noise
# --------------------------------------------------------------------------

def check_between_vs_within_config_variance(records: list[dict]) -> dict:
    """
    One-way ANOVA on raw per-trial DALYs, grouped by config: is the
    variation IN DALYS BETWEEN different configs meaningfully larger
    than the variation WITHIN a single fixed config across its own
    different seeds (pure simulation stochasticity)? If not, apparent
    differences between configs' DALYs can't yet be trusted as real -
    they may simply be within the noise floor, which is a genuinely
    different situation from "the model isn't sensitive to these
    parameters" (both would produce similar-looking DALYs across
    configs, but only one is a search problem worth acting on - this
    check alone can't distinguish the two, only confirm or rule out
    that noise ALONE could plausibly explain what's observed).

    Uses MEAN, not median, deliberately unlike this pipeline's own
    convention elsewhere (median_dalys, get_best_feasible_dalys, etc.)
    - variance/ANOVA are defined in terms of the mean; a "median-based
    ANOVA" isn't the standard, well-understood statistical tool this
    check is meant to be, so it intentionally breaks from the project's
    usual median convention here specifically.

    Requires at least one config with 2+ observed seeds (otherwise
    every group is a singleton, and within-group variance - the
    denominator - is undefined) - returns a dict with "insufficient_data":
    True and no F-statistic if that's not yet the case, rather than
    raising or dividing by zero.

    Returns a dict with n_configs, n_trials, n_multi_seed_configs (how
    many configs actually contributed to the within-group estimate),
    ms_between/ms_within (the two mean-square terms an F-ratio is built
    from), f_statistic, p_value (from scipy.stats.f's survival
    function), and a plain-English "interpretation" string.
    """
    from scipy.stats import f as f_dist

    grouped: dict[tuple, list[float]] = {}
    for rec in records:
        grouped.setdefault(config_key(rec["config_object"]), []).append(rec["dalys"])

    n_configs = len(grouped)
    n_trials = len(records)
    multi_seed_groups = {k: v for k, v in grouped.items() if len(v) >= 2}
    n_multi_seed_configs = len(multi_seed_groups)

    if n_multi_seed_configs == 0 or n_configs < 2:
        return {
            "insufficient_data": True,
            "n_configs": n_configs,
            "n_trials": n_trials,
            "n_multi_seed_configs": n_multi_seed_configs,
            "interpretation": (
                "Not enough data yet: need at least one config evaluated with 2+ seeds "
                "AND at least 2 distinct configs overall, to estimate both within- and "
                "between-config variance."
            ),
        }

    grand_mean = sum(v for vals in grouped.values() for v in vals) / n_trials

    ss_between = sum(len(vals) * (sum(vals) / len(vals) - grand_mean) ** 2 for vals in grouped.values())
    ss_within = sum(
        sum((v - sum(vals) / len(vals)) ** 2 for v in vals)
        for vals in multi_seed_groups.values()
    )

    df_between = n_configs - 1
    df_within = sum(len(vals) - 1 for vals in multi_seed_groups.values())

    if df_within == 0 or ss_within == 0:
        return {
            "insufficient_data": True,
            "n_configs": n_configs,
            "n_trials": n_trials,
            "n_multi_seed_configs": n_multi_seed_configs,
            "interpretation": "Within-group variance is degenerate (zero degrees of freedom or zero spread) - can't compute a meaningful F-ratio yet.",
        }

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_statistic = ms_between / ms_within
    p_value = float(f_dist.sf(f_statistic, df_between, df_within))

    if p_value < 0.05:
        interpretation = (
            f"Between-config spread (p={p_value:.4f}) is larger than seed noise alone would "
            f"plausibly explain - apparent DALYs differences between configs likely reflect "
            f"real signal, not just noise."
        )
    else:
        interpretation = (
            f"Between-config spread (p={p_value:.4f}) is NOT clearly distinguishable from "
            f"seed noise yet - apparent differences between configs' DALYs may just be within "
            f"the noise floor. More seeds per config (higher MAX_CONFIG_CALLS), a larger "
            f"pop_size, or more distinct configs evaluated would all help resolve this either way."
        )

    return {
        "insufficient_data": False,
        "n_configs": n_configs,
        "n_trials": n_trials,
        "n_multi_seed_configs": n_multi_seed_configs,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "f_statistic": f_statistic,
        "p_value": p_value,
        "interpretation": interpretation,
    }


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
    plot_color_by: str = "intensification",
    plot_order_by: str = "completion",
    filter_by_commit: bool = True,
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
    was called directly, separately, by name. plot_color_by/plot_order_by
    are passed straight through to that function's own color_by/order_by
    parameters - "intensification" (default) or "config" for the
    former, "completion" (default) or "submission" for the latter - see
    that function's own docstring for what each mode shows. Wrapped in
    try/except: matplotlib is imported
    lazily, inside the plotting function itself, specifically so a
    missing matplotlib install doesn't block the other four
    (matplotlib-free) checks above - if it's missing, or the plot fails
    for any other reason, this prints a warning and returns
    results["plot_path"] = None rather than crashing the whole
    evaluation over what's meant to be an additional, optional output.

    Safe to call at ANY point in a run - this is purely a read of
    whatever's already on disk.
    """
    records = load_history_log(filepath, filter_by_commit=filter_by_commit)
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
    variance_check = check_between_vs_within_config_variance(records)

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
        "variance_check": variance_check,
        "plot_path": None,
    }

    if save_plot:
        try:
            fig = plot_dalys_by_completion_order(records, color_by=plot_color_by, order_by=plot_order_by)
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

        if variance_check["insufficient_data"]:
            print(f"[evaluate] between vs within-config DALYs variance: {variance_check['interpretation']}")
        else:
            print(
                f"[evaluate] between vs within-config DALYs variance: F={variance_check['f_statistic']:.3f}, "
                f"p={variance_check['p_value']:.4f} ({variance_check['n_multi_seed_configs']} multi-seed config(s) "
                f"contributing to the within-group estimate)"
            )
            print(f"[evaluate]   -> {variance_check['interpretation']}")

    return results


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "history_log.jsonl"
    run_all_checks(log_path, plot_color_by="config", plot_order_by="submission")
