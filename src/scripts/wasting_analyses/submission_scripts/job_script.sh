#!/bin/bash -l

# Parameters passed from submit_job.sh
SCENARIO=$1
TIME=$2
MEMORY=$3
DRAWS=$4
RUNS=$5

# Determine the task number and create the task-specific directory
taskNumber="$SGE_TASK_ID"
thisRun=$(awk -v n="$taskNumber" "BEGIN { for (i=0; i<"$DRAWS"; i++) for (j=0; j<"$RUNS"; j++) if (++count==n) print i, j }")
drawNumber=$(( (taskNumber - 1) / RUNS ))
runNumber=$(( (taskNumber - 1) % RUNS ))

# Output directory setup
mainOutputDir="$HOME/Scratch/thanzi/TLOmodel-outputs/${JOB_NAME}-${JOB_ID}"
outputDir="$mainOutputDir/${drawNumber}/${runNumber}"
mkdir -p "$outputDir"

# Load and activate python environment
module load python3/3.11
source ~/thanzi/venv-tlo/bin/activate
cd ~/thanzi/TLOmodel

# *** Run the specified scenario with resource usage measurement
/usr/bin/time --verbose tlo scenario-run --draw $thisRun --output-dir "$outputDir" \
  src/scripts/wasting_analyses/scenarios/scenario_wasting_full_model_${SCENARIO}_totestsubmission.py

# Parse and compress logs
tlo parse-log "$outputDir"
gzip "$outputDir"/*.log

# Move and rename .o and .e files
mv "$HOME/Scratch/thanzi/TLOmodel-outputs/${JOB_NAME}.o${JOB_ID}.${taskNumber}" "$outputDir/${JOB_NAME}.o${JOB_ID}.$((taskNumber - 1))"
mv "$HOME/Scratch/thanzi/TLOmodel-outputs/${JOB_NAME}.e${JOB_ID}.${taskNumber}" "$outputDir/${JOB_NAME}.e${JOB_ID}.$((taskNumber - 1))"
