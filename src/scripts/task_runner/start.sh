#!/usr/bin/env bash
set -u -e

if [ -a .profile ]; then
   . ./.profile
fi

username=$(whoami)
wd_root="/mnt/tlodev2stg/tlo-dev-fs-2/task-runner"
output_root="${wd_root}/output"
worktrees_root="${wd_root}/worktrees"
conda_env_root="/mnt/task-runner/envs"

# construct the worktree and output directories for this commit
# commit_id=${1:0:8}
# commit_date=$(git show -s --date=format:'%Y-%m-%d_%H%M%S' --format=%cd "$1")
# commit_dir="${commit_date}_${commit_id}"
commit_dir=$(git show -s --date=format:'%Y-%m-%d_%H%M%S' --format=%cd_%h "$1")
worktree_dir="${worktrees_root}/${commit_dir}"
output_dir="${output_root}/${commit_dir}"

# if output directory already exists, then we've already run this commit - exit
[ -d "$output_dir" ] && echo "EXITING: commit $1 output directory already exists." && exit

# create directory for task output
mkdir "$output_dir"

exec > >(tee -a "${output_dir}/stdout.txt") 2>&1

echo "Commit hash: $1"
echo "Directory: ${commit_dir}";
echo "Commit working dir: ${worktree_dir}"
echo "Commit output dir: ${output_dir}"

# pull latest master branch (that's what were on), and create the worktree for the required commit
echo "Making worktree"
git pull
git worktree add "${worktree_dir}" "$1"

# need a new conda environment for this worktree
eval "$(/home/${username}/miniconda3/condabin/conda shell.bash hook)"
conda create -p "${conda_env_root}/${commit_dir}" --clone tlo
conda activate "${conda_env_root}/${commit_dir}"
pip uninstall -y tlo  # remove the existing tlo installation (we cloned the virtual environment)
pip install --use-feature=in-tree-build "${worktree_dir}"  # install tlo from the worktree

# working directory is the commit directory
cd "${worktree_dir}"

echo "Executing tasks from working directory: $(pwd)"

# loop over each task in the tasks folder
for file in $(find ./src/scripts/task_runner/tasks -name "[0-9]*" | sort -n)
do
    printf "\nTriggering task with %s.\n" "$file"
    fullpath=$(readlink -f "$file")

    # read in the task file and check for any special statements
    seq_command=""
    while read -r line
    do
        # if there is a line that starts with "# seq"
        if [[ "$line" ==  "# seq"* ]]
        then
            # capture the seq command for looping
            seq_command=${line:2}
        fi
    done < "$fullpath"

    basename_file=$(basename -- "$fullpath")
    execute_file="${basename_file%.*}"
    task_output_dir="$output_dir/$execute_file"

    mkdir "${task_output_dir}"

    # if seq command has not been given for this task
    if [ -z "$seq_command" ]
    then
        # run the task as normal - no looping
        /usr/local/bin/ts -E \
        ./src/scripts/task_runner/task.sh "$fullpath" "${worktree_dir}" "${task_output_dir}" "${conda_env_root}/${commit_dir}";
    else
        # otherwise, we've got a seq command to execute for looping
        for index in $(eval "$seq_command")
        do
            echo "Calling task with index $index"
            mkdir -p "${task_output_dir}/0/${index}"
            /usr/local/bin/ts -E \
            ./src/scripts/task_runner/task.sh "$fullpath" "${worktree_dir}" "${task_output_dir}/0/${index}" "${conda_env_root}/${commit_dir}" "$index";
        done
    fi
done

echo "Finished."
