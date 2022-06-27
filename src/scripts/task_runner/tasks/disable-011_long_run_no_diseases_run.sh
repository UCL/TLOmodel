# seq 0 4
tlo scenario-run --output-dir "${task_output_dir}" --draw 0 ${index} ./src/scripts/calibration_analyses/scenarios/long_run_no_diseases.py > "${task_output_dir}"/stdout.txt 2> "${task_output_dir}"/stderr.txt
