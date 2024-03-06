import calibration_outputs_with_ci

# This script runs the files which output calibration plots for the maternal and perinatal health modules

# pass file name to provide outputs
for file in ['baseline_scenario-2023-11-20T133043Z']:
    calibration_outputs_with_ci.output_incidence_for_calibration(
       scenario_filename=f'{file}.py',
       pop_size='250K',
       outputspath='./outputs/sejjj49@ucl.ac.uk/',
       sim_years=list(range(2010, 2023)))
