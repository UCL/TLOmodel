from pathlib import Path

from scripts.calibration_analyses.analysis_scripts import (
    analysis_cause_of_death_and_disability_calibrations,
    analysis_compare_appt_usage_real_and_simulation,
    analysis_demography_calibrations,
    analysis_hsi_descriptions,
)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Run all the calibration analyses (demographic, epidemiological & descriptions of healthcare system usage) on
    the results of running `long_run_all_diseases.py`."""

    analysis_demography_calibrations.apply(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath)

    analysis_cause_of_death_and_disability_calibrations.apply(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath)

    analysis_hsi_descriptions.apply(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath)

    analysis_compare_appt_usage_real_and_simulation.apply(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath)


if __name__ == "__main__":
    # outputspath = Path('./outputs/bshe@ic.ac.uk')  # tbh03
    rfp = Path('./resources')

    # Find results folder (most recent run generated using that scenario_filename)
    # scenario_filename = 'long_run_all_diseases.py'
    # from tlo.analysis.utils import get_scenario_outputs
    # results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

    # Test dataset:
    results_folder = Path('/Users/tbh03/GitHub/TLOmodel/outputs/small_test_run')

    # If needed -- in the case that pickles were not created remotely during batch
    # create_pickles_locally(results_folder)

    # Run all the calibrations
    apply(results_folder=results_folder, output_folder=results_folder, resourcefilepath=rfp)
