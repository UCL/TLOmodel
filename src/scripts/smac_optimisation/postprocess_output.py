"""
All post-processing of a single completed TLO simulation run lives here,
separate from ask_tell_azure_example.py's submission/polling/ask-tell
orchestration. Import postprocess_run() from this file wherever a
downloaded run's outputs need turning into the four numbers SMAC/
ConstrainedEI need (dalys, cost, hr_used, stock_used).

Keeping this separate means the postprocessing pipeline can be edited,
tested, or run standalone against a downloaded output directory without
touching (or re-reading) the orchestration logic at all.
"""

from pathlib import Path

import pandas as pd

from tlo.analysis.utils import load_pickled_dataframes


# TARGET_PERIOD: adjust to match the evaluation window your simulation
# actually covers - this determines which years' DALYs/costs get summed.
TARGET_PERIOD = (pd.Timestamp(2010, 1, 1), pd.Timestamp(2011, 1, 1))


def postprocess_run(run_dir: Path) -> dict:
    """
    Turns a single completed run's raw output directory into the four
    numbers SMAC/ConstrainedEI need. Adapted from the DALY-extraction
    pattern used in TLOmodel's own analysis scripts (see e.g.
    src/scripts/hiv/program_simplification/process_outputs.py), using
    tlo.analysis.utils.load_pickled_dataframes rather than reading raw
    pickle files directly - this is the officially supported way to
    load a single run's log.

    run_dir is expected to be .../<job_id>/0/<run_number> - i.e. what
    ask_tell_azure_example.py's download_run_outputs() returns entries
    of when iterating draw_dir.
    """
    job_root = run_dir.parent.parent   # .../job_id/0/<run_number> -> .../job_id
    run_number = int(run_dir.name)
    log = load_pickled_dataframes(job_root, draw=0, run=run_number)

    dalys = _get_total_dalys(log)
    cost = _get_total_cost(log)
    hr_used = _get_hr_capacity_used(log)
    stock_used = _get_consumables_used(log)

    return {"dalys": dalys, "cost": cost, "hr_used": hr_used, "stock_used": stock_used}


def _get_total_dalys(log: dict) -> float:
    """
    All-cause DALYs (stacked) summed over TARGET_PERIOD, from one run's log.
    Directly adapted from get_num_dalys() in process_outputs.py - same
    module/key, same year-completeness assertion (guards against
    silently using results from a run that crashed partway through),
    same drop-non-value-columns-then-sum-twice pattern (once across
    columns/causes, once across years).
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


def _get_total_cost(log: dict) -> float:
    """
    NOT YET ADAPTED - the reference script imports its costing logic from
    a separate module (scripts.costing.cost_estimation:
    estimate_input_cost_of_scenarios, summarize_cost_data, etc.), which
    operates at the batch/results_folder level (across many draws/runs
    at once) rather than on a single run's log the way DALYs above does.
    I could not see that module's actual function bodies/signatures to
    adapt it correctly for a single-run call here - worth checking
    scripts/costing/cost_estimation.py directly and adapting whichever
    function computes cost for one run (or a single-row extraction from
    its batch-level output) rather than guessing at its API.
    """
    raise NotImplementedError("Adapt from scripts/costing/cost_estimation.py - see docstring")


def _get_hr_capacity_used(log: dict) -> float:
    """
    NOT YET ADAPTED - no HR-capacity extraction appeared in the portion
    of process_outputs.py available to adapt from. Likely lives in
    log["tlo.methods.healthsystem"] under some capability-usage key -
    worth checking that module's logged keys directly (e.g. via
    log["tlo.methods.healthsystem"].keys()) to find the right one.
    """
    raise NotImplementedError("Find the right key under log['tlo.methods.healthsystem']")


def _get_consumables_used(log: dict) -> float:
    """
    NOT YET ADAPTED - same situation as _get_hr_capacity_used: no
    consumables-usage extraction appeared in the portion of
    process_outputs.py available to adapt from. Likely also under
    log["tlo.methods.healthsystem"], possibly a consumables-specific key.
    """
    raise NotImplementedError("Find the right key under log['tlo.methods.healthsystem']")


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
