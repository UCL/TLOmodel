#!/usr/bin/env bash
username=$(whoami)
tlo_dirname="/home/${username}/TLOmodel"
output_dir="/mnt/tlodev2stg/tlo-dev-fs-2/task-runner/output"

. /etc/profile

if [ -a .profile ]; then
   . ./.profile
fi

cd "${tlo_dirname}"

script_full_path=$(dirname "$0")

# get latest changes
git pull

# loop over commits and kick off those that haven't run
git log --date=format:'%Y-%m-%d_%H%M%S' --format="%cd_%h %H" | while read -r commit_key commit_hash
do
    commit_dir="${output_dir}/${commit_key}"
    if [ -d "${commit_dir}" ]
    then
        echo "${commit_dir} exists; exit"
        break
    else
        echo "Schedule commit ${commit_hash}"
        "${script_full_path}"/start.sh "${commit_hash}"
    fi
done

# create the HTML file for the output
eval "$(/home/${username}/miniconda3/condabin/conda shell.bash hook)"
conda activate tlo
cd "${output_dir}" || exit
python "${script_full_path}"/generate_html.py > index.html
