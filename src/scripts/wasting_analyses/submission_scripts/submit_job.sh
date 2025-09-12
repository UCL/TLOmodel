
#!/bin/bash

set -x

# Read parameters
SCENARIO=$1
TIME=$2
MEMORY=$3
DRAWS=$4
RUNS=$5

# Calculate number of jobs
JOBS=$((DRAWS * RUNS))

# Submit the job array with parameters
qsub -t 1-${JOBS} -N wasting_analysis__full_model_${SCENARIO} \
 -l h_rt=${TIME}:0:0 -l mem=${MEMORY} \
 -wd /home/sejjej5/Scratch/thanzi/TLOmodel-outputs \
 -m beas -M sejjej5@ucl.ac.uk \
 job_script.sh $SCENARIO $TIME $MEMORY $DRAWS $RUNS
