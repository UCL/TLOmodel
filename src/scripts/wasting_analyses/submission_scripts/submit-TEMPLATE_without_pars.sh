#!/bin/bash -l

############ JOB CONFIG

# *** Request for the most reasonable minimum you can, up to 72 hours. This specifies 24 hours
#$ -l h_rt=15:0:0

# *** Request for the most reasonable minimum of memory you can. This specifies 16 GB
#$ -l mem=16G

# *** Personal job name identifier
#$ -N wasting_analysis__full_model_XX

# *** Put in your username below
#$ -wd /home/sejjej5/Scratch/thanzi/TLOmodel-outputs

# *** Setup the job array: 1-(no. of draws * no. of runs) e.g. if 3 draws, 3 runs: 1-9
#$ -t 1-10

# *** Email notifications
#$ -m beas
#$ -M sejjej5@ucl.ac.uk

############ END OF JOB CONFIG

# *** Specify number of draws & runs
numberOfDraws=1
numberOfRuns=10

# Create the main output directory
mainOutputDir="$HOME/Scratch/thanzi/TLOmodel-outputs/${JOB_NAME}-${JOB_ID}"
mkdir -p "$mainOutputDir"

# Determine the task number and create the task-specific directory
taskNumber="$SGE_TASK_ID"
thisRun=$(awk -v n="$taskNumber" "BEGIN { for (i=0; i<"$numberOfDraws"; i++) for (j=0; j<"$numberOfRuns"; j++) if (++count==n) print i, j }")
drawNumber=$(echo $thisRun | awk "{print \$1}")
runNumber=$(echo $thisRun | awk "{print \$2}")
outputDir="$mainOutputDir/${drawNumber}/${runNumber}"
mkdir -p "$outputDir"

# Load and activate python environment
module load python3/3.11
source ~/thanzi/venv-tlo/bin/activate

cd ~/thanzi/TLOmodel

# *** Run the specified scenario with resource usage measurement
/usr/bin/time --verbose tlo scenario-run --draw $thisRun --output-dir "$outputDir" src/scripts/wasting_analyses/scenarios/scenario_wasting_full_model_XX.py
tlo parse-log "$outputDir"
gzip "$outputDir"/*.log

# Move and rename the .o and .e files to the task-specific directory
mv "$HOME/Scratch/thanzi/TLOmodel-outputs/${JOB_NAME}.o${JOB_ID}.${taskNumber}" "$outputDir/${JOB_NAME}.o${JOB_ID}.$((taskNumber - 1))"
mv "$HOME/Scratch/thanzi/TLOmodel-outputs/${JOB_NAME}.e${JOB_ID}.${taskNumber}" "$outputDir/${JOB_NAME}.e${JOB_ID}.$((taskNumber - 1))"
