commit_id="${1:-12345}"

root_dir="/mnt/tlodev2stg/tlo-dev-fs-2/task-runner"
conda env remove -p "/mnt/task-runner/envs/${commit_id}"
git worktree remove --force "${root_dir}/worktrees/${commit_id}"
rm -r "${root_dir}/output/${commit_id}"
