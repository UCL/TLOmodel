commit_id="${1:-12345}"

conda env remove -p /mnt/autoload/envs/${commit_id}
git worktree remove --force /mnt/autoload/worktrees/${commit_id}

rm -r /home/$(whoami)/automated_runs/outputs/${commit_id}
