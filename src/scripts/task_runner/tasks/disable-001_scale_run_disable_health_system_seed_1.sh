python src/scripts/profiling/scale_run.py --parse-log-file \
    --disable-health-system --years 20 --months 0 --seed 164827310 --initial-population 50000  \
    --output-dir "${task_output_dir}" > /dev/null 2> "${task_output_dir}"/stderr.txt
