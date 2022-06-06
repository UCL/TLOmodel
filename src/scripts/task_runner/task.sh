#!/usr/bin/env bash
set -e -u

default_index="-1"

path_this_script=$0
execute_path=$1
worktree_dir=$2
task_output_dir=$3
conda_env=$4
index="${5:-$default_index}"

username=$(whoami)

exec > >(tee -a "${task_output_dir}/task.txt") 2>&1

echo "    Start time: $(date)"
echo "    PWD: $(pwd)"
echo "    PID: $$"
echo "    Path to this script: $path_this_script"
echo "    Path to execute: $execute_path"
echo "    Worktree dir: $worktree_dir"
echo "    Task output dir: $task_output_dir"
echo "    Conda env: $conda_env"
echo "    Index for this run: $index"

cd "$worktree_dir"

eval "$(/home/${username}/miniconda3/condabin/conda shell.bash hook)"
conda activate "${conda_env}"

echo "    Python interpreter: $(which python)"
echo "    Starting script ${execute_path}"
set +e
. "$execute_path"
exit_status=$?
set -e

echo "${exit_status}" > "${task_output_dir}"/exit_status.txt

if [ "$exit_status" -eq 99 ]; then
    # resubmit task
    echo "    Resubmit time: $(date -d 5mins)"
    sleep 5m
    /usr/local/bin/ts -E \
        ./src/scripts/task_runner/task.sh "${execute_path}" "${worktree_dir}" "${task_output_dir}" "${conda_env}";
else
    echo "    Finished script ${execute_path}"
    echo "    Exit status: ${exit_status}"
    echo "    End time: $(date)"
    # ignore error if no log files found
    gzip "${task_output_dir}"/*.log || true
fi
