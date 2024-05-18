"""
From the top-level directory of the TLOmodel repository, run the following commands:

tlo scenario-run src/scripts/dev/suspend-restore/full-simple.py
tlo scenario-run src/scripts/dev/suspend-restore/suspend-simple.py

Then edit the restore-simple.py file to point to the suspended run directory and run the following command:

tlo scenario-run src/scripts/dev/suspend-restore/resume-simple.py

Once you have run the three scenarios, run this script to compare the dataframes in the full run directory with the
concatenated dataframes from the suspended and restored run directories.

python src/scripts/dev/suspend-restore/check_suspend_resume_df.py \
    outputs/full-simple-... outputs/suspend-simple... outputs/resume-simple...
"""
from pathlib import Path
import sys

import pandas as pd

full_run_path = Path(sys.argv[1])
suspend_run_path = Path(sys.argv[2])
resume_run_path = Path(sys.argv[3])

print("Full run path:", full_run_path)
print("Suspended run path:", suspend_run_path)
print("Resumed run path:", resume_run_path)

run_number = "0/0"

# for each .pickle file in the single run directory
for file in (full_run_path / run_number).glob("*.pickle"):
    print("=" * 80)
    print("Processing", file)

    # read the dictionary of dataframes from the .pickle file
    dfs_from_full = pd.read_pickle(file)

    # get the corresponding .pickle file in the suspended run directory
    if (suspend_run_path / run_number / file.name).exists():
        dfs_from_suspended = pd.read_pickle(suspend_run_path / run_number / file.name)
    else:
        print(f"File {file.name} does not exist in the suspended run directory")
        dfs_from_suspended = None

    # get the corresponding .pickle file in the restored run directory
    if (resume_run_path / run_number / file.name).exists():
        dfs_from_resumed = pd.read_pickle(resume_run_path / run_number / file.name)
    else:
        dfs_from_resumed = None

    # loop over each of the dataframes in the original dictionary
    for log_key, log_df in dfs_from_full.items():
        if log_key == "_metadata":
            continue

        print("Key:", log_key)

        if dfs_from_suspended is None and dfs_from_resumed is None:
            print(f"Key {log_key} does not exist in the suspended or restored run directory")
            continue

        if dfs_from_suspended is None:
            concatenated_value = dfs_from_resumed[log_key]
        elif dfs_from_resumed is None:
            concatenated_value = dfs_from_suspended[log_key]
        else:
            if log_key in dfs_from_suspended and log_key in dfs_from_resumed:
                try:
                    concatenated_value = pd.concat([dfs_from_suspended[log_key], dfs_from_resumed[log_key]])
                except (ValueError, KeyError) as e:
                    print("Error:", e)
                    continue
            elif log_key in dfs_from_suspended:
                concatenated_value = dfs_from_suspended[log_key]
            else:
                concatenated_value = dfs_from_resumed[log_key]

        # compare the single run dataframe with the concatenated dataframe
        # and print the differences
        try:
            # reindex value dataframe
            log_df = log_df.reset_index(drop=True)
            concatenated_value = concatenated_value.reset_index(drop=True)
            concatenated_value = concatenated_value.reindex(columns=log_df.columns)

            compared = log_df.compare(concatenated_value)
            if compared.empty:
                print("\tNo differences")
            else:
                print("\tDifferences:")
                print(compared.to_string())
        except ValueError as e:
            print("Error:", e)
            print("original", log_df)
            print("concatenated", concatenated_value)
            print("original columns", log_df.columns)
            print("concatenated columns", log_df.columns)
            print("-" * 80)
