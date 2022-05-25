import os
import sys
from pathlib import Path

number_of_runs = 5
run_directory_name = "011_long_run_no_diseases_run"  # this is the name of the task that ran the scenario


def check_completed(scenario_outputs):
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
    print(f'Parse the run files from {output_dir}')
    for run in range(0, number_of_runs):
        run_dir = scenario_runs_dir / str(run)
        print(f'Collecting results from run directory {run_dir}')
    print(f'Save output to processed directory {output_dir}')


if __name__ == '__main__':
    output_dir_for_processed = Path(sys.argv[1])
    output_dir_for_scenario_runs = output_dir_for_processed.parent / run_directory_name
    scenario_output_dir = check_completed(output_dir_for_scenario_runs)
    if scenario_output_dir is not None:
        process(output_dir_for_processed, output_dir_for_scenario_runs)
    else:
        # exit status 99 to signal we're not ready to process results, resubmit this task
        sys.exit(99)
