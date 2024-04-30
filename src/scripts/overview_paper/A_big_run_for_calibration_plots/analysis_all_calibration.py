import argparse
from pathlib import Path

from scripts.calibration_analyses.analysis_scripts import (
    analysis_cause_of_death_and_disability_calibrations,
    analysis_compare_appt_usage_real_and_simulation,
    analysis_demography_calibrations,
    analysis_hsi_descriptions,
    plot_appt_use_by_hsi,
    plot_legends,
)
from scripts.healthsystem.org_chart_of_hsi import plot_org_chart_treatment_ids


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

    plot_appt_use_by_hsi.apply(
        results_folder=results_folder, output_folder=output_folder, resourcefilepath=resourcefilepath)

    # Plot the legends
    plot_legends.apply(
        results_folder=None, output_folder=output_folder, resourcefilepath=resourcefilepath)

    # Plot the org chart of HSI
    plot_org_chart_treatment_ids.apply(
        results_folder=None, output_folder=output_folder, resourcefilepath=resourcefilepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    # Needed the first time as pickles were not created on Azure side:
    # from tlo.analysis.utils import create_pickles_locally
    # create_pickles_locally(
    #     scenario_output_dir=args.results_folder,
    #     compressed_file_name_prefix=args.results_folder.name.split('-')[0],
    # )

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
