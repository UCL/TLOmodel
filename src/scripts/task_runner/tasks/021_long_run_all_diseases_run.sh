# seq 0 9
tlo scenario-run --output-dir "${task_output_dir}" --draw 0 ${index} ./src/scripts/calibration_analyses/scenarios/long_run_all_diseases.py > "${task_output_dir}"/stdout.txt 2> "${task_output_dir}"/stderr.txt
