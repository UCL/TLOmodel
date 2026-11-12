"""
All post-processing of a single completed TLO simulation run lives here,
separate from ask_tell_azure_example.py's submission/polling/ask-tell
orchestration. Import postprocess_run() from this file wherever a
downloaded run's outputs need turning into what SMAC/ConstrainedEI
need: dalys, HIV-HRH cost by year, HIV-consumable cost by year. These
two costs are checked against year-specific BUDGETS (the allowed $ for
each category, loaded from a limits file) in initialise.py /
optimisation_pipeline.py - "cost" here always means the actual $ a
simulated run spent; "budget" always means the $ allowed - both in the
same unit, for the same category, never a physical hours/units measure.

Keeping this separate means the postprocessing pipeline can be edited,
tested, or run standalone against a downloaded output directory without
touching (or re-reading) the orchestration logic at all.

DALYs and cost are HIV-SPECIFIC, not all-cause/all-programme, both
adapted from src/scripts/hiv/program_simplification/process_outputs.py
(the tara_hiv_program_simplification branch). HIV-HRH cost and
HIV-consumable cost are each returned BY YEAR (not summed), so
optimisation_pipeline.py can bucket them into user-defined periods and
enforce them as two separate constraints - see _get_hiv_dalys(),
_get_hiv_hrh_cost_by_year(), and _get_hiv_consumable_cost_by_year()
below for exactly what each captures and where it was adapted from.
"""

import ast
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from tlo.analysis.utils import load_pickled_dataframes
from scripts.costing.cost_estimation import load_unit_cost_assumptions
from optimisation_parameters import (
    YEAR_END_DATE, CONFIG_YEAR_START_DATE, COST_LIMITS_FILE, BASELINE_SUMMARY_FILE,
    BASELINE_BUDGET_COMPARISON_FILE, INPUTS_DIR, POP_SIZE,
    HISTORY_LOG_FILE, MAX_CONFIG_CALLS, CONVERGENCE_WINDOW, EI_XI,
)

# Malawi's real population, used as the numerator of the scaling-factor
# fallback below (_get_scaling_factor()) when the run's own log doesn't
# report one - i.e. the fallback assumes a "true population / POP_SIZE"
# scaling factor of MALAWI_REAL_POPULATION / POP_SIZE, rather than
# silently leaving the run unscaled (scale factor 1.0).
MALAWI_REAL_POPULATION = 14_539_609
from smac_scenario_baseline import BASELINE_RUNS_PER_DRAW


# TARGET_PERIOD: adjust to match the evaluation window your simulation
# actually covers - this determines which years' DALYs/costs get
# extracted. optimisation_pipeline.py's PERIOD_BOUNDARIES (the
# user-definable constraint periods) are checked at import time against
# this range - see the consistency check there - so a mismatch between
# the two is caught explicitly rather than silently producing missing
# or zero-filled years.
#
# DELIBERATELY INDEPENDENT of TARGET_PERIOD in
# process_outputs_for_smac_input.py (which uses 2025-2050, the real
# evaluation window for full analysis runs) - this file's TARGET_PERIOD
# is set for whatever simulation window is actually being run through
# THIS pipeline (e.g. a shorter window during small-scale testing) and
# is not meant to be kept in sync with that file. If you change the
# simulated start/end dates in smac_scenario.py, update this value to
# match THIS pipeline's own run, not the other script's.
TARGET_PERIOD = (pd.Timestamp(CONFIG_YEAR_START_DATE, 1, 1), pd.Timestamp(YEAR_END_DATE, 1, 1))


def _extract_missing_path_from_error(exc: Exception) -> Path | None:
    """
    Parses "No such file or directory: '<path>'" out of an exception's
    string representation. Fragile in general, but the specific error
    format is consistent - confirmed against a real observed traceback
    (dill's _create_filehandle raising exactly this pattern, wrapped in
    an UnpicklingError). Returns None if the exception doesn't match
    this pattern at all, so callers can tell "this is the known
    resumed-trial issue" apart from "this is something else entirely".
    """
    match = re.search(r"No such file or directory: '([^']+)'", str(exc))
    if match is None:
        return None
    return Path(match.group(1))


def _load_pickled_dataframes_with_resume_workaround(job_root: Path, draw: int, run: int) -> dict:
    """
    Wraps load_pickled_dataframes() with a WORKAROUND for a confirmed
    failure mode when postprocessing RESUMED (suspend/resume) trials.
    This is a workaround for genuine TLO/dill behaviour, not a bug in
    our own pipeline code, and not something fixable here beyond
    catching and recovering from it.

    ROOT CAUSE, confirmed directly from TLO's own source:
    - tlo/simulation.py: Simulation._configure_logging() sets
      self.output_file = logging.set_output_file(log_path) - a genuine
      instance attribute.
    - tlo/logging/core.py: set_output_file() returns a real Python
      standard-library logging.FileHandler (opens a live file stream
      immediately on construction, by default).
    - Simulation.save_to_pickle() does dill.dump(self, ...) - the ENTIRE
      object, including this live FileHandler bound to whatever path
      was active at checkpoint-generation time.
    - dill, once imported anywhere in a process, registers itself with
      Python's global copyreg pickle-dispatch table for file-like
      types - a PROCESS-WIDE effect, not scoped to calls made via
      dill.load() specifically. So even load_pickled_dataframes's own
      plain pickle.load() delegates reconstruction of any embedded
      FileHandler to dill's logic, which tries to reopen the ORIGINAL
      path - the checkpoint job's own working directory, on a
      DIFFERENT Azure task, long gone by the time we load the resumed
      trial's own output locally.

    WORKAROUND: if load_pickled_dataframes fails with exactly this
    pattern, extract the missing path from the exception, create an
    EMPTY dummy file there (plus parent directories), and retry ONCE.
    This assumes the missing file is a stale FileHandler needing SOME
    path to reopen, not something whose actual CONTENT is needed to
    reconstruct the DataFrame we actually care about - consistent with
    the confirmed mechanism above, since the DALYs/cost data lives
    entirely separately from this stale logging handle. If that
    assumption is wrong, the retry fails again with a different, more
    informative error, rather than silently producing incorrect data -
    this does NOT swallow errors it doesn't recognise, or retry more
    than once.
    """
    try:
        return load_pickled_dataframes(job_root, draw=draw, run=run)
    except (FileNotFoundError, pickle.UnpicklingError) as exc:
        missing_path = _extract_missing_path_from_error(exc)
        if missing_path is None:
            raise  # not the pattern we know how to work around - fail loudly

        print(
            f"[postprocess] WORKAROUND: creating dummy file for a stale resumed-trial "
            f"reference at {missing_path} - see _load_pickled_dataframes_with_resume_"
            f"workaround()'s docstring for why this is a workaround, not a confirmed fix."
        )
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing_path.touch()

        return load_pickled_dataframes(job_root, draw=draw, run=run)


def _get_scaling_factor(log: dict) -> float:
    """
    Returns the ratio of true (real-world) population to this run's own
    SIMULATED population (POP_SIZE) - TLO's own standard mechanism for
    rescaling simulated-population-scale outputs (raw counts, sums) up
    to real-population magnitudes. Read directly from the log, matching
    tlo.analysis.utils.extract_results()'s own get_multiplier() and
    compare_number_of_deaths()'s own scaling-factor lookup:
    log["tlo.methods.population"]["scaling_factor"]["scaling_factor"].values[0]

    Defensive fallback if the key is genuinely absent: rather than
    leaving the run unscaled (raw simulated-population-scale numbers,
    which is silently wrong for every downstream use - see below),
    this assumes the standard TLO population-scaling relationship,
    MALAWI_REAL_POPULATION / POP_SIZE, i.e. exactly what the missing
    "tlo.methods.population" scaling_factor entry would itself have
    reported for a run at this same POP_SIZE. This is an approximation
    (the real logger's own value can differ slightly run to run,
    depending on the simulation's actual init population draw), not a
    confirmed substitute - kept as a fallback rather than raising,
    since a missing key here most likely means the population module
    simply isn't logging at INFO level in this particular scenario, not
    that scaling should silently be skipped.

    CONFIRMED against the reference script (process_outputs.py,
    tara_hiv_program_simplification) directly: every absolute
    count/total it extracts (DALYs - both get_num_dalys() and
    num_dalys_by_cause(), the function THIS project's own _get_hiv_dalys
    is explicitly adapted from - deaths, HIV test counts, person-years
    on PrEP, VMMC counts) is extracted with do_scaling=True. The one
    counterexample shown (art_coverage_adult) is a RATE/fraction, not a
    count, and correctly omits scaling - multiplying a fraction by a
    population scaling factor wouldn't make sense. HIV-HRH cost and
    HIV-consumable cost are both absolute $ totals (additive quantities,
    like DALYs and deaths), not rates - genuinely need the same
    treatment for consistency, though the reference script's own cost
    functions weren't directly confirmed (further down that same file,
    not reachable when this was checked - github.com's raw file URL
    returned robots-disallowed, and no accessible mirror was found).
    """
    try:
        return float(log["tlo.methods.population"]["scaling_factor"]["scaling_factor"].values[0])
    except (KeyError, IndexError):
        # LOUD on purpose (previously silent) - this fallback means
        # dalys/cost for THIS run come back scaled by an ASSUMED factor
        # rather than the run's own actually-logged one, which is worth
        # knowing about (not merely "unavailable") whenever the
        # population logger is missing but healthburden/other loggers
        # this run depends on are still present - the same class of
        # missing-logger problem that _get_hiv_dalys() (below) raises
        # loudly on, and gets excluded as CRASHED, for. Without this
        # print, such a run would instead be silently recorded as
        # SUCCESSFUL with an assumed-not-measured dalys/cost scale,
        # feeding straight into ConstrainedEI's surrogate and the
        # budget-feasibility check with no visible trace anywhere. Kept
        # as a fallback rather than raising (matching
        # compare_number_of_deaths()'s own convention of not raising on
        # a missing scaling_factor - see this function's docstring), but
        # now at least visible in the run's own logs so a suspiciously-
        # scaled result can be traced back to this cause instead of
        # being invisible.
        fallback = MALAWI_REAL_POPULATION / POP_SIZE
        print(
            "[scaling_factor] WARNING: 'tlo.methods.population' scaling_factor "
            f"missing from this run's log - falling back to an ASSUMED "
            f"MALAWI_REAL_POPULATION / POP_SIZE = {fallback:,.2f} rather than the "
            f"run's own actually-logged value. dalys/cost for this run may be "
            f"wrong if this run's own true scaling factor genuinely differed "
            f"from that assumption."
        )
        return fallback


def postprocess_run(run_dir: Path) -> dict:
    """
    Turns a single completed run's raw output directory into what SMAC/
    ConstrainedEI need: dalys (single total), plus HIV-HRH cost and
    HIV-consumable cost EACH BY YEAR (not summed into one flat total) -
    optimisation_pipeline.py buckets these into user-defined periods and
    enforces them as two SEPARATE constraint families. Uses
    tlo.analysis.utils.load_pickled_dataframes (via the resume-workaround
    wrapper above) rather than reading raw pickle files directly - this
    is the officially supported way to load a single run's log.

    ALL THREE returned quantities are rescaled by this run's own
    _get_scaling_factor(log) (true population ÷ simulated POP_SIZE) -
    an earlier version of this function returned these figures at raw,
    simulated-population scale, unscaled, for every path (real trials,
    resumed trials, AND the baseline all funnel through this same
    function - there was never a separate code path for any of them,
    so a bug here would have affected all three identically, not
    selectively). See _get_scaling_factor()'s own docstring for the
    confirmed source of this convention. Applying the multiplication
    HERE, once, rather than inside each of the three extraction
    functions individually, since all three need the identical
    treatment and this avoids three separate (and potentially
    inconsistent) lookups of the same value from the same log.

    run_dir is expected to be .../<job_id>/0/<run_number> - i.e. what
    ask_tell_azure_example.py's download_run_outputs() returns entries
    of when iterating draw_dir.
    """
    job_root = run_dir.parent.parent   # .../job_id/0/<run_number> -> .../job_id
    run_number = int(run_dir.name)
    log = _load_pickled_dataframes_with_resume_workaround(job_root, draw=0, run=run_number)

    scaling_factor = _get_scaling_factor(log)

    dalys = _get_hiv_dalys(log) * scaling_factor
    hiv_hrh_cost_by_year = {
        year: cost * scaling_factor for year, cost in _get_hiv_hrh_cost_by_year(log).items()
    }
    hiv_consumable_cost_by_year = {
        year: cost * scaling_factor for year, cost in _get_hiv_consumable_cost_by_year(log).items()
    }
    return {
        "dalys": dalys,
        "hiv_hrh_cost_by_year": hiv_hrh_cost_by_year,
        "hiv_consumable_cost_by_year": hiv_consumable_cost_by_year,
    }


def _report_budget_comparison(per_run_results: list, hrh_stats: dict, cons_stats: dict,
                              first_year: int, last_year: int) -> None:
    """
    Prints and saves (BASELINE_BUDGET_COMPARISON_FILE) how the adopted
    budget (highest yearly cost across baseline runs) compares with the
    upper 95% CI of the mean: per year and category, and in total. Also
    reports how many baseline runs would have exceeded the CI budget in
    at least one year of each period, which is what the CI budget would
    have meant for the baseline's own feasibility.
    """
    n_runs = len(per_run_results)
    rows = []
    summary_lines = []
    for category, stats, key in (
        ("hiv_hrh", hrh_stats, "hiv_hrh_cost_by_year"),
        ("hiv_consumable", cons_stats, "hiv_consumable_cost_by_year"),
    ):
        years = [y for y in range(first_year, last_year + 1) if y in stats]
        for y in years:
            st = stats[y]
            rows.append({
                "category": category, "year": y, "mean": st["mean"], "max_budget": st["max"],
                "upper95_ci": st["upper95"],
                "max_over_ci_minus_1": (st["max"] / st["upper95"] - 1.0) if st["upper95"] > 0 else float("nan"),
            })
        if not years:
            continue
        # Totals over the years considered: summed yearly budgets vs the most
        # expensive single run (the summed maxima come from different runs).
        run_totals = np.array([sum(r[key].get(y, 0.0) for y in years) for r in per_run_results])
        sum_max = sum(stats[y]["max"] for y in years)
        sum_ci = sum(stats[y]["upper95"] for y in years)
        summary_lines.append(
            f"[baseline] {category}: summed yearly max budget = {sum_max:,.0f}; most expensive single "
            f"run total = {run_totals.max():,.0f} (summed max is {sum_max / run_totals.max() - 1:+.1%} vs it); "
            f"summed upper-95%-CI budget = {sum_ci:,.0f} (max budget is {sum_max / sum_ci - 1:+.1%} vs CI)."
        )
        # Baseline runs that would have breached the CI budget, per year.
        n_breach_any = sum(
            any(r[key].get(y, 0.0) > stats[y]["upper95"] for y in years) for r in per_run_results
        )
        summary_lines.append(
            f"[baseline] {category}: {n_breach_any}/{n_runs} baseline runs exceed the upper-95%-CI budget "
            f"in at least one year (0/{n_runs} under the max budget, by construction)."
        )
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(BASELINE_BUDGET_COMPARISON_FILE, index=False)
    for line in summary_lines:
        print(line)
    print(f"[baseline] per-year comparison saved to {BASELINE_BUDGET_COMPARISON_FILE}.")


def compute_and_save_baseline_budgets(baseline_draw_dir: Path, first_year: int, last_year: int) -> None:
    """
    Computes the upper 95% CI (across all runs found in baseline_draw_dir)
    of HIV-HRH cost and HIV-consumable cost, YEAR BY YEAR, over
    [first_year, last_year] inclusive, and OVERWRITES COST_LIMITS_FILE
    with the result - this is what optimisation_pipeline.py's real
    trials then get checked against as their budget constraint. Call
    this ONCE, before initialise.py is ever imported (that file reads
    COST_LIMITS_FILE at import time) - see
    optimisation_pipeline.submit_baseline_job() for the submission this
    is meant to be paired with.

    baseline_draw_dir is expected to be draw 0 of a MULTI-RUN (10 runs,
    differently seeded - see smac_scenario_baseline.py's own
    runs_per_draw=10) baseline submission of a FULL, non-suspend/resume
    scenario - i.e. exactly what download_run_outputs() returns for
    such a job. Each run subdirectory gets postprocessed via the SAME
    postprocess_run() every real trial's own output goes through, so
    these cost figures are computed identically to how a real trial's
    own costs are computed - just aggregated across 10 independent runs
    instead of 1.

    BUDGET = HIGHEST COST ACROSS THE BASELINE RUNS, per year and category:
    for each year, the maximum of that year's cost over all runs found in
    baseline_draw_dir. If the status-quo programme is taken to be always
    feasible, then in any given year the budget must allow the most
    expensive outcome the baseline can produce, so by construction every
    baseline run is feasible on every constraint. (An upper 95% CI of the
    MEAN cost, mean + 1.96 * std / sqrt(n), was the previous choice; it is
    far tighter than a single run's spread, so single baseline runs could
    exceed it. It is still computed here, for comparison only.)

    Because the budget is a maximum taken separately for each year, the
    maxima come from different runs, so the summed budget over years can
    exceed the total cost of ANY single baseline run. A comparison table
    (per year and category: mean, max, upper-95%-CI, and the max-vs-CI gap;
    and per category: the summed yearly budget against the most expensive
    single run and against the summed CI budget) is printed and saved to
    BASELINE_BUDGET_COMPARISON_FILE, together with how many baseline runs
    would have violated each period under the CI budget.

    hiv_dalys is written as 0.0 for every year in COST_LIMITS_FILE - loaded
    by initialise.py but not currently used as a constraint (DALYs
    remains the optimisation OBJECTIVE, not a budget) - see
    initialise.py's own comment on this. The baseline's ACTUAL DALYs
    (mean across the 10 runs, over TARGET_PERIOD - the same period and
    metric every real trial's own "dalys" field already is) gets
    persisted separately, to BASELINE_SUMMARY_FILE - see below.

    Years in [first_year, last_year] outside what the baseline run
    actually simulated (TARGET_PERIOD, above) get a budget of 0.0 too -
    confirmed SAFE, not a division-by-zero risk: initialise.py's own
    bucket_peak_budget_slack() only ever looks up a year's budget
    for years that ALSO appear as keys in a real trial's own
    cost_by_year dict, which is itself restricted to TARGET_PERIOD - so
    a year outside that range is never actually divided into, on either
    side (matches initialise.py's own documented "extra period-years
    simply never accumulate any data, which is harmless" behaviour for
    PERIOD_BOUNDARIES extending past TARGET_PERIOD).
    """
    run_dirs = sorted(baseline_draw_dir.iterdir(), key=lambda p: int(p.name))
    if not run_dirs:
        raise RuntimeError(f"{baseline_draw_dir} has no run subdirectories - nothing to compute CIs from.")
    if len(run_dirs) != BASELINE_RUNS_PER_DRAW:
        print(
            f"[warning] baseline run at {baseline_draw_dir} has {len(run_dirs)} run "
            f"subdirector{'y' if len(run_dirs) == 1 else 'ies'}, not the expected "
            f"{BASELINE_RUNS_PER_DRAW} (smac_scenario_baseline.py's own runs_per_draw). "
            f"Proceeding anyway, but the resulting budgets are derived from fewer "
            f"independent seeds than intended - CIs will be wider/less reliable than "
            f"expected. This usually means some of the baseline job's tasks hadn't "
            f"actually finished (or had failed) when this was called - confirm "
            f"azure_job_is_finished()/azure_task_succeeded() in optimisation_pipeline.py "
            f"are checking ALL of the job's tasks, not just the first."
        )

    per_run_results = [postprocess_run(run_dir) for run_dir in run_dirs]
    n_runs = len(per_run_results)

    def yearly_stats(key: str) -> dict[int, dict]:
        """Per year: mean, max (the budget), upper 95% CI of the mean (comparison only)."""
        all_years = set()
        for r in per_run_results:
            all_years.update(r[key].keys())

        out = {}
        for year in all_years:
            values = np.array([r[key].get(year, 0.0) for r in per_run_results], dtype=float)
            mean = values.mean()
            # ddof=1: SAMPLE std (n-1 denominator), as appropriate for a CI of the mean.
            std = values.std(ddof=1) if n_runs > 1 else 0.0
            out[int(year)] = {
                "mean": float(mean),
                "max": float(values.max()),
                "upper95": float(mean + 1.96 * std / np.sqrt(n_runs)),
                "values": values,
            }
        return out

    hrh_stats = yearly_stats("hiv_hrh_cost_by_year")
    cons_stats = yearly_stats("hiv_consumable_cost_by_year")
    hrh_upper = {y: v["max"] for y, v in hrh_stats.items()}
    cons_upper = {y: v["max"] for y, v in cons_stats.items()}

    _report_budget_comparison(per_run_results, hrh_stats, cons_stats, first_year, last_year)

    rows = [
        {
            "year": year,
            "hiv_dalys": 0.0,  # loaded but unused as a constraint - see docstring
            "hiv_hrh_budget": hrh_upper.get(year, 0.0),
            "hiv_consumable_budget": cons_upper.get(year, 0.0),
        }
        for year in range(first_year, last_year + 1)
    ]

    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(COST_LIMITS_FILE, index=False)
    print(f"[baseline] {COST_LIMITS_FILE} updated (budget = highest yearly cost across runs): {n_runs} run(s), years {first_year}-{last_year}.")

    # Persisted separately from the CSV above - this is the baseline's
    # own OBJECTIVE-side result (DALYs), not a budget/constraint value,
    # and has nowhere else to live once this function returns. Read by
    # evaluate_pipeline_run.py's plotting to draw the baseline
    # comparison line, independently, from disk. MEDIAN, not mean, for
    # consistency with the rest of this pipeline's own DALYs/cost
    # aggregation - unlike upper_95_ci_by_year() above (the budget/
    # constraint values themselves), which deliberately stays
    # mean-centered, since a 95% confidence interval is conventionally
    # defined around the sample mean, not the median - this is a
    # genuinely different computation from a plain aggregate DALYs
    # figure, not another instance of the same mean-vs-median choice.
    median_dalys = float(np.median([r["dalys"] for r in per_run_results]))
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_SUMMARY_FILE, "w") as f:
        json.dump({"median_dalys": median_dalys, "n_runs": n_runs}, f, indent=2)
    print(f"[baseline] {BASELINE_SUMMARY_FILE} updated: median DALYs = {median_dalys:.4f} ({n_runs} runs).")


def report_ei_xi_diagnostic(history_file: Path = HISTORY_LOG_FILE) -> None:
    """
    Prints a short diagnostic on whether EI_XI should be changed before the
    next resume, from the results logged in `history_file`.

    EI_XI (xi) is in DALYs: a candidate only earns expected improvement for
    beating the best feasible predicted DALYs by MORE than xi. xi = 0
    gives plain EI (SMAC's default); a positive xi shifts the search from
    exploiting towards exploring.

    The yardstick is the seed-to-seed noise of a config's DALYs, which is
    what limits how small a real improvement can be resolved:
      noise_sd = pooled within-config standard deviation (configs with >= 2 seeds)
      noise_se = noise_sd / sqrt(MAX_CONFIG_CALLS)   (error of a fully-confirmed config)
    and it compares that with the improvement of the best feasible DALYs
    over the last CONVERGENCE_WINDOW results. Heuristic, not a test:
      - improvement > noise_se: the search is still finding resolvable
        gains, keep xi at 0.
      - improvement <= noise_se: gains are within the noise; a xi of about
        noise_se stops the search chasing differences it cannot resolve.
    """
    if not Path(history_file).exists():
        print(f"[EI_XI] no history log at {history_file} - nothing to diagnose.")
        return
    entries = [json.loads(line) for line in open(history_file) if line.strip()]
    keys = sorted({k for e in entries for k in e if "_violation" in k})
    if len(entries) < 2 * CONVERGENCE_WINDOW:
        print(f"[EI_XI] only {len(entries)} results (< {2 * CONVERGENCE_WINDOW}) - too early to judge; keep EI_XI = {EI_XI}.")
        return

    def cfg_key(e):
        return json.dumps(e["config_object"], sort_keys=True)

    # within-config noise
    by_cfg: dict[str, list[float]] = {}
    for e in entries:
        by_cfg.setdefault(cfg_key(e), []).append(float(e["dalys"]))
    variances = [np.var(v, ddof=1) for v in by_cfg.values() if len(v) >= 2]
    if not variances:
        print("[EI_XI] no config has >= 2 seeds yet, so the noise cannot be estimated; keep EI_XI as is.")
        return
    noise_se = float(np.sqrt(np.mean(variances))) / np.sqrt(MAX_CONFIG_CALLS)

    # running best feasible (by config median), in trial order
    seen: dict[str, list[dict]] = {}
    best_over_time: list[float | None] = []
    for e in entries:
        seen.setdefault(cfg_key(e), []).append(e)
        best = None
        for rows in seen.values():
            if all(np.median([r[k] for r in rows if k in r]) <= 0 for k in keys):
                d = float(np.median([r["dalys"] for r in rows]))
                best = d if best is None or d < best else best
        best_over_time.append(best)
    now, before = best_over_time[-1], best_over_time[-1 - CONVERGENCE_WINDOW]
    if now is None:
        print("[EI_XI] no feasible config yet; EI_XI is not the limiting factor. Keep as is.")
        return
    improvement = (before - now) if before is not None else float("inf")
    print(
        f"[EI_XI] seed noise of a {MAX_CONFIG_CALLS}-seed config (SE) = {noise_se:.4g} DALYs; "
        f"best-feasible improvement over the last {CONVERGENCE_WINDOW} results = {improvement:.4g}."
    )
    if improvement > noise_se:
        print(f"[EI_XI] gains are larger than the noise: keep EI_XI = {EI_XI} (or lower).")
    else:
        print(f"[EI_XI] gains are within the noise: consider raising EI_XI to about {noise_se:.4g} before resuming.")


def _get_hiv_dalys(log: dict) -> float:
    """
    HIV/AIDS-specific DALYs (stacked), summed over TARGET_PERIOD, from
    one run's log. Adapted from num_dalys_by_cause() in
    process_outputs.py: same module/key/year-completeness guard as
    _get_total_dalys() below, but does NOT collapse across cause columns
    before summing - instead selects the 'AIDS' cause specifically.

    'AIDS' is the cause label TLOmodel's healthburden module uses for
    HIV-related DALYs - confirmed by the {'AIDS': 'HIV/AIDS', '': 'Other'}
    mapping in get_total_num_dalys_by_label() in the same reference
    script, which lumps every OTHER cause into 'Other', implying 'AIDS'
    is the sole HIV-specific cause label in this model's DALY breakdown.
    """
    df = log["tlo.methods.healthburden"]["dalys_stacked"]
    years_needed = [d.year for d in TARGET_PERIOD]
    assert set(df.year.unique()).issuperset(years_needed), \
        "Some years are not recorded - run may have crashed early."
    by_cause = (
        df.loc[df.year.between(*years_needed)]
          .drop(columns=['date', 'sex', 'age_range', 'year'])
          .sum()
    )
    return float(by_cause.get('AIDS', 0.0))


def _get_total_dalys(log: dict) -> float:
    """
    ALL-CAUSE DALYs (stacked), summed over TARGET_PERIOD - kept for
    reference/diagnostics. postprocess_run() uses _get_hiv_dalys() above
    instead, since the optimisation target is HIV-specific DALYs, not
    every cause the simulation tracks.
    """
    df = log["tlo.methods.healthburden"]["dalys_stacked"]
    years_needed = [d.year for d in TARGET_PERIOD]
    assert set(df.year.unique()).issuperset(years_needed), \
        "Some years are not recorded - run may have crashed early."
    return float(
        df.loc[df.year.between(*years_needed)]
          .drop(columns=['date', 'sex', 'age_range', 'year'])
          .sum().sum()
    )


# --------------------------------------------------------------------------
# HIV-specific cost pipelines, adapted from process_outputs.py's own
# pipeline (summarise_appointments -> keep_selected_appt_types ->
# add_mapped_treatments_as_appt_types -> appt_counts_to_hcw_minutes ->
# hrh_time_costs_by_facility). See _get_hiv_hrh_cost_by_year()'s
# docstring for the important caveat on what this does and doesn't
# capture.
#
# Note process_outputs.py imports estimate_input_cost_of_scenarios from
# scripts.costing.cost_estimation, but never actually CALLS it anywhere
# in the file - the functions adapted below are the real, working
# HIV-cost implementation there, not that import.
# --------------------------------------------------------------------------

def _parse_dict_cell(x) -> dict:
    """
    Cells in HSI/appt-footprint logs are stored as dicts when read
    straight from a live log, but sometimes come back as their string
    repr after a pickle round-trip - normalise both to a plain dict.
    Identical to _parse_dict_cell() in process_outputs.py.
    """
    if isinstance(x, dict):
        return x
    if pd.isna(x):
        return {}
    if isinstance(x, str):
        return ast.literal_eval(x)
    return {}


# Explicit allow-list of appointment types to KEEP - copied verbatim
# from process_outputs.py's KEEP_APPT_TYPES. This is the authoritative
# list; do not add/remove entries without checking that file first.
KEEP_APPT_TYPES = [
    "VCTPositive",
    "VCTNegative",
    "NewAdult",
    "Peds",
    "EstNonCom",
    "MaleCirc",
]

# Maps a TREATMENT_ID onto a synthetic appointment-type row (appt_type,
# facility_level, multiplier), copied verbatim from process_outputs.py's
# TREATMENT_TO_APPT_SPEC. These TREATMENT_IDs (PrEP dispensing, self-test
# counselling, palliative/inpatient care) use generic appt-type codes
# ALSO used by non-HIV treatments, so KEEP_APPT_TYPES filtering alone
# can't isolate their HIV-attributable share - this injects that share
# back in via the source TREATMENT_ID instead, which IS HIV-specific.
TREATMENT_TO_APPT_SPEC = {
    "PharmDispensing": ("Hiv_Prevention_Prep", "1a", 1.0),
    "ConWithDCSA": ("Hiv_Test_Selftest", "0", 1.0),
    "IPAdmission": ("Hiv_PalliativeCare", "2", 2.0),
    "InpatientDays": ("Hiv_PalliativeCare", "2", 17.0),
}

# Static reference data, loaded ONCE at import time (not per-run, since
# it never changes between runs) - same convention as TARGET_PERIOD
# above. Paths are relative to the TLOmodel repo root, same assumption
# as the rest of this pipeline.
_HCW_TIME_TABLE = pd.read_csv(
    "resources/healthsystem/human_resources/definitions/ResourceFile_Appt_Time_Table.csv"
)
_HCW_TIME_TABLE["Facility_Level"] = _HCW_TIME_TABLE["Facility_Level"].astype(str).str.strip()
_HCW_TIME_TABLE["Appt_Type_Code"] = _HCW_TIME_TABLE["Appt_Type_Code"].astype(str).str.strip()
_HCW_TIME_TABLE["Officer_Category"] = _HCW_TIME_TABLE["Officer_Category"].astype(str).str.strip()

# Facility-level-SPECIFIC minutes-per-appointment map, indexed by
# (Facility_Level, Appt_Type_Code) - NOT a blanket single-facility-level
# assumption. Matches appt_counts_to_hcw_minutes()'s map_table exactly.
_APPT_MINUTES_BY_FACILITY_MAP = _HCW_TIME_TABLE.pivot_table(
    index=["Facility_Level", "Appt_Type_Code"], columns="Officer_Category",
    values="Time_Taken_Mins", aggfunc="mean",
)

_HCW_HOURLY_COSTS = pd.read_csv("resources/ResourceFile_HIV/hrh_costs.csv")
_HCW_HOURLY_COSTS["Facility_Level"] = _HCW_HOURLY_COSTS["Facility_Level"].astype(str).str.strip()
_HCW_HOURLY_COSTS["Officer_Category"] = _HCW_HOURLY_COSTS["Officer_Category"].astype(str).str.strip()

# Hourly wage lookup by (Facility_Level, Officer_Category) - matches
# hrh_time_costs_by_facility()'s cost_lookup exactly (averaging any
# duplicate rows for the same facility_level/cadre pair).
_HCW_HOURLY_COST_LOOKUP = (
    _HCW_HOURLY_COSTS.groupby(["Facility_Level", "Officer_Category"])["Total_hourly_cost"].mean()
)


def _get_hiv_treatment_counts_by_year(log: dict) -> dict:
    """
    HIV-prefixed TREATMENT_ID counts, keyed by (year, treatment_id), from
    one run's log, restricted to TARGET_PERIOD. Adapted from
    make_series_treatment_counts_by_year() in process_outputs.py.

    Year is genuinely preserved here (earlier versions of this function
    collapsed it despite the "_by_year" name) - needed so HIV-HRH and
    HIV-consumable cost can each be computed per year, for the
    period-bucketed constraints built on top of them in
    optimisation_pipeline.py (see PERIOD_BOUNDARIES there).
    """
    df = log["tlo.methods.healthsystem.summary"]["HSI_Event"]
    years_needed = [d.year for d in TARGET_PERIOD]
    dates = pd.to_datetime(df["date"])
    df = df.loc[dates.dt.year.between(*years_needed)]
    years = pd.to_datetime(df["date"]).dt.year

    counts: Counter = Counter()
    for year, cell in zip(years, df["TREATMENT_ID"]):
        for treatment_id, count in _parse_dict_cell(cell).items():
            if treatment_id.startswith("Hiv"):
                counts[(int(year), treatment_id)] += count
    return dict(counts)


def _get_hiv_appt_counts_by_year_and_facility(log: dict) -> dict:
    """
    HIV-relevant appointment counts by (year, facility_level, appt_type),
    from one run's log, restricted to TARGET_PERIOD. Adapted from
    process_outputs.py's summarise_appointments() ->
    keep_selected_appt_types() -> add_mapped_treatments_as_appt_types()
    pipeline. Year preserved throughout - see
    _get_hiv_treatment_counts_by_year()'s docstring for why.
    """
    df = log["tlo.methods.healthsystem.summary"]["HSI_Event_non_blank_appt_footprint"]
    years_needed = [d.year for d in TARGET_PERIOD]
    dates = pd.to_datetime(df["date"])
    df = df.loc[dates.dt.year.between(*years_needed)]
    years = pd.to_datetime(df["date"]).dt.year

    counts: Counter = Counter()
    for year, cell in zip(years, df["Number_By_Appt_Type_Code_And_Level"]):
        nested = _parse_dict_cell(cell)  # {facility_level: {appt_type: count}}
        if not isinstance(nested, dict):
            continue
        for facility_level, inner in nested.items():
            if isinstance(inner, dict):
                for appt_type, count in inner.items():
                    if appt_type in KEEP_APPT_TYPES:
                        counts[(int(year), str(facility_level), appt_type)] += count

    # inject treatment-derived synthetic appointment rows, per year -
    # see TREATMENT_TO_APPT_SPEC's comment above for why this is needed
    hiv_treatment_counts = _get_hiv_treatment_counts_by_year(log)
    for appt_type, (trt_id, facility_level, mult) in TREATMENT_TO_APPT_SPEC.items():
        for (year, this_trt_id), trt_count in hiv_treatment_counts.items():
            if this_trt_id == trt_id:
                key = (year, str(facility_level), appt_type)
                counts[key] = counts.get(key, 0.0) + trt_count * mult

    return dict(counts)


def _get_hiv_appt_minutes_by_year_facility_and_cadre(log: dict) -> dict:
    """
    HIV-specific appointment MINUTES by (year, facility_level, cadre),
    from one run's log. Adapted from appt_counts_to_hcw_minutes() in
    process_outputs.py - uses the facility-level-SPECIFIC (Facility_Level,
    Appt_Type_Code) -> minutes mapping, year preserved throughout.
    """
    appt_counts = _get_hiv_appt_counts_by_year_and_facility(log)

    minutes: Counter = Counter()
    for (year, facility_level, appt_type), count in appt_counts.items():
        key = (facility_level, appt_type)
        if key not in _APPT_MINUTES_BY_FACILITY_MAP.index:
            continue
        row = _APPT_MINUTES_BY_FACILITY_MAP.loc[key]
        for cadre, mins_per_appt in row.items():
            if pd.notna(mins_per_appt):
                minutes[(year, facility_level, cadre)] += count * mins_per_appt
    return dict(minutes)


def _get_hiv_hrh_cost_by_year(log: dict) -> dict:
    """
    HIV-HRH cost (staff time x hourly wage), BY YEAR, from one run's log.
    Adapted from hrh_time_costs_by_facility() in process_outputs.py:
    hours by (facility_level, cadre) x the facility/cadre-specific
    hourly wage from hrh_costs.csv, summed WITHIN each year (not across
    the whole TARGET_PERIOD) - so optimisation_pipeline.py can apply
    year-specific limits and bucket into user-defined periods.

    Named "HIV-HRH cost" (not just "HRH cost") since this captures only
    the HIV-attributable share of health-worker time, per
    KEEP_APPT_TYPES/TREATMENT_TO_APPT_SPEC above - not total
    health-system HRH cost.
    """
    minutes_by_year_fac_cadre = _get_hiv_appt_minutes_by_year_facility_and_cadre(log)

    cost_by_year: Counter = Counter()
    for (year, facility_level, cadre), minutes in minutes_by_year_fac_cadre.items():
        hourly_cost = _HCW_HOURLY_COST_LOOKUP.get((facility_level, cadre))
        if hourly_cost is None or pd.isna(hourly_cost):
            continue
        cost_by_year[year] += (minutes / 60.0) * hourly_cost
    return dict(cost_by_year)


# --------------------------------------------------------------------------
# HIV-consumable cost, adapted from process_outputs.py's
# get_counts_of_items_requested() -> apply_unit_costs() pipeline, plus
# the two flat-rate additions it applies OUTSIDE that pipeline (self-tests
# and TDF urine tests - see cons_costed_hiv's construction there:
# item-code-filtered consumables + selftests_costs + tdf_costs).
# --------------------------------------------------------------------------

# HIV-specific consumable item codes, hardcoded from the resolved values
# shown directly in process_outputs.py's own comment block - originally
# resolved there via get_item_code_from_item_name()/
# get_item_codes_from_package_name() against
# ResourceFile_Consumables_Items_and_Packages.csv. Hardcoding the
# already-resolved codes here avoids an extra CSV load + name-lookup
# per run for values that never change.
HIV_ITEM_CODES = {
    196: "HIV test",
    190: "Viral load",
    197: "VMMC",
    1191: "Adult PrEP",
    198: "Infant PrEP",
    2671: "First-line ART regimen: adult",
    204: "First-line ART regimen: adult: cotrimoxazole",
    2672: "First line ART regimen: older child",
    2673: "First line ART regimen: young child",
    202: "First line ART regimen: young child: cotrimoxazole",
}
_HIV_ITEM_CODE_STRS = set(map(str, HIV_ITEM_CODES.keys()))

# Per-unit consumable costs, loaded ONCE via TLOmodel's own costing
# module. Unlike estimate_input_cost_of_scenarios (imported but never
# called in process_outputs.py), load_unit_cost_assumptions IS actually
# called there - a genuine, demonstrated adaptation, not a guess at an
# unused function's API.
_UNIT_COST_ASSUMPTIONS = load_unit_cost_assumptions(Path("resources"))
_CONS_UNIT_COST_BY_ITEM_CODE = dict(zip(
    _UNIT_COST_ASSUMPTIONS["consumables"]["Item_Code"],
    _UNIT_COST_ASSUMPTIONS["consumables"]["Price_per_unit"],
))

# Flat per-test costs process_outputs.py applies OUTSIDE the normal
# item-code costing pipeline, for tests not tracked as a standard
# consumable dispensation. Kept as named constants (rather than magic
# numbers) so their provenance stays visible at the call site:
# self-test cost is a MIHPSA costing assumption; TDF test cost is a
# South Africa cost estimate (see the source citation left in
# process_outputs.py next to tdf_costs).
SELFTEST_UNIT_COST = 3.14
TDF_TEST_UNIT_COST = 6.86


def _get_hiv_item_counts_by_year(log: dict) -> dict:
    """
    HIV-relevant consumable item counts, keyed by (year, item_code), from
    one run's log, restricted to TARGET_PERIOD. Adapted from
    get_counts_of_items_requested() in process_outputs.py, filtered down
    to HIV_ITEM_CODES, year preserved.
    """
    df = log["tlo.methods.healthsystem.summary"]["Consumables"]
    years_needed = [d.year for d in TARGET_PERIOD]
    dates = pd.to_datetime(df["date"])
    df = df.loc[dates.dt.year.between(*years_needed)]
    years = pd.to_datetime(df["date"]).dt.year

    counts: Counter = Counter()
    for year, cell in zip(years, df["Item_Used"]):
        for item_code, count in _parse_dict_cell(cell).items():
            if str(item_code) in _HIV_ITEM_CODE_STRS:
                counts[(int(year), str(item_code))] += count
    return dict(counts)


def _get_num_tdf_tests_by_year(log: dict) -> dict:
    """
    Total TDF urine tests performed, keyed by year, from one run's log.
    Adapted from get_num_tdf() in process_outputs.py, year preserved.
    """
    df = log["tlo.methods.hiv"]["hiv_program_coverage"].copy()
    years_needed = [d.year for d in TARGET_PERIOD]
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df = df.loc[df["year"].between(*years_needed)]
    return {int(year): float(v) for year, v in df.groupby("year")["n_tdf_tests_performed"].sum().items()}


def _get_hiv_consumable_cost_by_year(log: dict) -> dict:
    """
    HIV-consumable cost, BY YEAR, from one run's log: item-code-filtered
    consumables (via unit costs) PLUS self-test and TDF-test flat-rate
    costs, summed WITHIN each year. Matches process_outputs.py's
    cons_costed_hiv construction, disaggregated by year instead of
    summed over the whole TARGET_PERIOD.
    """
    item_counts_by_year = _get_hiv_item_counts_by_year(log)
    hiv_treatment_counts_by_year = _get_hiv_treatment_counts_by_year(log)
    tdf_by_year = _get_num_tdf_tests_by_year(log)

    cost_by_year: Counter = Counter()

    for (year, item_code), count in item_counts_by_year.items():
        cost_by_year[year] += count * _CONS_UNIT_COST_BY_ITEM_CODE.get(int(item_code), 0.0)

    for (year, treatment_id), count in hiv_treatment_counts_by_year.items():
        if treatment_id == "Hiv_Test_Selftest":
            cost_by_year[year] += count * SELFTEST_UNIT_COST

    for year, count in tdf_by_year.items():
        cost_by_year[int(year)] += count * TDF_TEST_UNIT_COST

    return dict(cost_by_year)


def _get_hiv_total_cost_diagnostic(log: dict) -> float:
    """
    DIAGNOSTIC ONLY - HIV-HRH + HIV-consumable cost, summed across years
    and combined into one number. NOT used by postprocess_run() or any
    constraint: HIV-HRH cost and HIV-consumable cost are two SEPARATE
    constraint families (see optimisation_pipeline.py), each enforced
    per-period, not combined into a single total. Kept only for manual
    sanity-checking against process_outputs.py's own combined
    cons_hrh_costs_year_draw_run figure.
    """
    hrh = sum(_get_hiv_hrh_cost_by_year(log).values())
    cons = sum(_get_hiv_consumable_cost_by_year(log).values())
    return float(hrh + cons)


def _get_total_cost(log: dict) -> float:
    """
    NOT YET ADAPTED - a fully comprehensive input cost (overheads,
    equipment, programme management, etc. on top of what
    _get_hiv_hrh_cost_by_year()/_get_hiv_consumable_cost_by_year()
    already cover), via TLOmodel's official costing module
    (scripts.costing.cost_estimation: estimate_input_cost_of_scenarios,
    summarize_cost_data, etc.), which is never actually called in
    process_outputs.py despite being imported there - so there's no
    working reference call to adapt from in that file. Worth checking
    scripts/costing/cost_estimation.py directly if overhead/equipment
    costs need to be captured too; HIV-HRH + HIV-consumable cost is
    likely sufficient for most purposes and is what postprocess_run()
    actually uses.
    """
    raise NotImplementedError("Adapt from scripts/costing/cost_estimation.py - see docstring")


if __name__ == "__main__":
    # Quick standalone sanity check: point this at an already-downloaded
    # run directory and confirm postprocess_run() works without needing
    # to run the full ask-tell pipeline.
    import sys
    if len(sys.argv) != 2:
        print("Usage: python postprocess_output.py <path-to-run-dir>")
        sys.exit(1)
    result = postprocess_run(Path(sys.argv[1]))
    print(result)
