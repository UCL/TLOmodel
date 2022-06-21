from pathlib import Path

from scripts.calibration_analyses.analysis_scripts import (
    analysis_cause_of_death_and_disability_calibrations,
    analysis_demography_calibrations,
    analysis_hsi_descriptions,
    analysis_get_simulated_monthly_usage_by_appt_type,
    analysis_compare_appt_usage_real_and_simulation
)
from tlo.analysis.utils import get_scenario_outputs


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Run all the calibration analyses (demographic, epidemiological & descriptions of healthcare system usage) on
    the results of running `long_run_all_diseases.py`."""

    analysis_demography_calibrations.apply(
        results_folder=results_folder, output_folder=results_folder, resourcefilepath=rfp)

    analysis_cause_of_death_and_disability_calibrations.apply(
        results_folder=results_folder, output_folder=results_folder, resourcefilepath=rfp)

    analysis_hsi_descriptions.apply(
        results_folder=results_folder, output_folder=results_folder, resourcefilepath=rfp)

    analysis_get_simulated_monthly_usage_by_appt_type.apply(
        results_folder=results_folder, output_folder=results_folder)  # this much run before analysis_compare_

    analysis_compare_appt_usage_real_and_simulation.apply(
        results_folder=results_folder, output_folder=results_folder, resourcefilepath=rfp)


if __name__ == "__main__":
    outputspath = Path('./outputs/bshe@ic.ac.uk')  # tbh03
    rfp = Path('./resources')

    # Find results folder (most recent run generated using that scenario_filename)
    scenario_filename = 'long_run_all_diseases.py'
    results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

    # Test dataset:
    # results_folder = Path('/Users/tbh03/GitHub/TLOmodel/outputs/tbh03@ic.ac.uk/long_run_all_diseases-small')

    # If needed -- in the case that pickles were not created remotely during batch
    # create_pickles_locally(results_folder)

    # Run all the calibrations
    apply(results_folder=results_folder, output_folder=results_folder, resourcefilepath=rfp)
