"""Analyse the results of scenario to test impact of consumable availability."""

from pathlib import Path
from tlo.analysis.utils import (
    get_scenario_outputs,
    load_pickled_dataframes
)


if __name__ == "__main__":

    # Find results_folder associated with a given batch_file and get most recent
    results_folder = get_scenario_outputs(
        "scenario_impact_of_art_coverage.py", Path('./outputs')
    )[-1]

    # Load log (useful for checking what can be extracted)
    log_run0 = load_pickled_dataframes(results_folder, draw=0)
    log_run1 = load_pickled_dataframes(results_folder, draw=1)

    print(log_run0['tlo.methods.hivlite']['aids_cases'])
    print(log_run1['tlo.methods.hivlite']['aids_cases'])
