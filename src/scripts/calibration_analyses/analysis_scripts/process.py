import os
import sys
from pathlib import Path


def check_completed(scenario_outputs, number_of_runs):
    print(f'Looking in {scenario_outputs}')
    # check each scenario run
    for scenario_number in range(0, number_of_runs):
        scenario_output = scenario_outputs / str(scenario_number)
        # if output directory doesn't exist
        if not os.path.isdir(scenario_output):
            print(f'{scenario_output} doesnt exist')
            return None
        # if exit status file doesn't exist
        exit_status_file = f'{scenario_output}/exit_status.txt'
        if not os.path.isfile(exit_status_file):
            print(f'{exit_status_file} doesnt exist')
            return None
    # both the output directory and the exit status exists - all runs of scenario are finish
    # begin analysis here:
    return scenario_outputs


def process(output_dir, scenario_runs_dir):
    from importlib import import_module

    function = getattr(import_module('analysis_all_calibration'), 'apply')
    function(scenario_runs_dir, output_dir, Path('/home/azureuser/TLOmodel/resources'))

    # function = getattr(import_module('analysis_demography_calibrations'), 'apply')
    # function(scenario_runs_dir, output_dir, Path('/home/azureuser/TLOmodel/resources'))
    #
    # function = getattr(import_module('analysis_cause_of_death_and_disability_calibrations'), 'apply')
    # function(scenario_runs_dir, output_dir, Path('/home/azureuser/TLOmodel/resources'))
    #
    # function = getattr(import_module('analysis_hsi_descriptions'), 'apply')
    # function(scenario_runs_dir, output_dir, Path('/home/azureuser/TLOmodel/resources'))


if __name__ == '__main__':
    _output_dir_for_processed = Path(sys.argv[1])
    _run_directory_name = sys.argv[2]
    _number_of_runs = int(sys.argv[3])
    _output_dir_for_scenario_runs = _output_dir_for_processed.parent / _run_directory_name / '0'
    _scenario_output_dir = check_completed(_output_dir_for_scenario_runs, _number_of_runs)
    if _scenario_output_dir is not None:
        process(_output_dir_for_processed, _output_dir_for_processed.parent / _run_directory_name)
    else:
        # exit status 99 to signal we're not ready to process results, resubmit this task
        sys.exit(99)
