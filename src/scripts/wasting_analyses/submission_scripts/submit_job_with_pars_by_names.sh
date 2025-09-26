
#!/bin/bash

set -x

# Default values (optional, if needed)
SCENARIO=""
TIME=""
MEMORY=""
DRAWS=""
RUNS=""

# Parse named arguments in NAME=value format
for ARG in "$@"; do
  case $ARG in
    SCENARIO=*)
      SCENARIO="${ARG#*=}"
      ;;
    TIME=*)
      TIME="${ARG#*=}"
      ;;
    MEMORY=*)
      MEMORY="${ARG#*=}"
      ;;
    DRAWS=*)
      DRAWS="${ARG#*=}"
      ;;
    RUNS=*)
      RUNS="${ARG#*=}"
      ;;
    *)
      echo "Unknown parameter: $ARG"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "$SCENARIO" || -z "$TIME" || -z "$MEMORY" || -z "$DRAWS" || -z "$RUNS" ]]; then
  echo "Usage: $0 SCENARIO=<scenario> TIME=<time> MEMORY=<memory> DRAWS=<draws> RUNS=<runs>"
  exit 1
fi

# Calculate number of jobs
JOBS=$((DRAWS * RUNS))

# Submit the job array with parameters
qsub -t 1-${JOBS} -N wasting_analysis__full_model_${SCENARIO} \
 -l h_rt=${TIME}:0:0 -l mem=${MEMORY} \
 -wd /home/sejjej5/Scratch/thanzi/TLOmodel-outputs \
 -m beas -M sejjej5@ucl.ac.uk \
 job_script.sh $SCENARIO $TIME $MEMORY $DRAWS $RUNS
