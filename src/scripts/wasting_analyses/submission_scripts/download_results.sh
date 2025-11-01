#!/bin/bash

# Check for two input arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_csv_file> <timestamp_suffix>"
    exit 1
fi

INPUT_FILE="$(realpath "$1")"
START_TIME="$2"

cd ~/Scratch/thanzi/TLOmodel-outputs || exit

# Output files with timestamp suffix
DOWNLOAD_FILE="download_commands_${START_TIME}.txt"
RUN_NAMES_FILE="run_names_${START_TIME}.csv"

# Create or clear output files
> "$DOWNLOAD_FILE"
> "$RUN_NAMES_FILE"

# Process each line of the CSV, skipping the header
tail -n +2 "$INPUT_FILE" | while IFS=, read -r JOBID SCENARIO_TYPE FSTIME; do
    SCENARIO_NAME="wasting_analysis__full_model_${SCENARIO_TYPE}"
    TIMESTAMP=$(date -d "$FSTIME" +"%Y-%m-%dT%H%M%SZ")

    orig_name=$(find . -maxdepth 1 -type d -name "${SCENARIO_NAME}*${JOBID}")
    name_w_timestamp="${SCENARIO_NAME}-${TIMESTAMP}-${JOBID}"

    mv "${orig_name}" "${name_w_timestamp}"

    base_name=$(echo "${name_w_timestamp}" | sed "s/-${JOBID}//")

    cp -r "${name_w_timestamp}" "${base_name}"
    zip -r "${base_name}.zip" "${base_name}"
    rm -r "${base_name}"

    mkdir "${name_w_timestamp}/oe_files_${JOBID}"
    mv *.o${JOBID}* "${name_w_timestamp}/oe_files_${JOBID}/"
    mv *.e${JOBID}* "${name_w_timestamp}/oe_files_${JOBID}/"

    echo "scp sejjej5@myriad.rc.ucl.ac.uk:~/Scratch/thanzi/TLOmodel-outputs/${SCENARIO_NAME}-${TIMESTAMP}.zip ~/PycharmProjects/TLOmodel/outputs/sejjej5@ucl.ac.uk/wasting/scenarios/${SCENARIO_TYPE}/" >> "$DOWNLOAD_FILE"
    echo "${SCENARIO_NAME}-${TIMESTAMP}" >> "$RUN_NAMES_FILE"
done
