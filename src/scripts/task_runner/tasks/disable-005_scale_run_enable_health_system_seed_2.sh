python src/scripts/profiling/scale_run.py --parse-log-file \
    --years 20 --months 0 --seed 711037118 --initial-population 50000  \
    --output-dir "${task_output_dir}" > /dev/null 2> "${task_output_dir}"/stderr.txt
