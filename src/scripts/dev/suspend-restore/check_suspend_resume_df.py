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

import numpy as np
import pandas as pd

full_run_path = Path(sys.argv[1])
suspend_run_path = Path(sys.argv[2])
resume_run_path = Path(sys.argv[3])

print("Full run path:", full_run_path)
print("Suspended run path:", suspend_run_path)
print("Resumed run path:", resume_run_path)

run_number = "0/0"


def categorical_series_all_equal(series_1: pd.Series, series_2: pd.Series):
    return (
        (series_1.values == series_2.values)
        | (pd.isna(series_1) & pd.isna(series_2)).values
    ).all()


def robust_compare(
    dataframe_1: pd.DataFrame, dataframe_2: pd.DataFrame, indent=0
) -> None:

    def print_with_indent(val):
        print("\t" * indent + str(val))

    if dataframe_1.equals(dataframe_2):
        print_with_indent("No differences")
    elif len(dataframe_1.columns) != len(dataframe_2.columns):
        print_with_indent(
            f"Different number of columns: {len(dataframe_1.columns)} vs {len(dataframe_2.columns)}"
        )
    elif not (dataframe_1.columns == dataframe_2.columns).all():
        print_with_indent(
            f"Different columns:\n{dataframe_1.columns}\n{dataframe_2.columns}"
        )
    elif len(dataframe_1) != len(dataframe_2):
        print_with_indent(
            f"Different number of rows: {len(dataframe_1)} vs {len(dataframe_2)}"
        )
    else:
        all_matching = True
        for column in dataframe_1.columns:
            if (
                dataframe_1[column].equals(dataframe_2[column])
                or (dataframe_1[column] == dataframe_2[column]).all()
                or (
                    dataframe_1.dtypes[column] == "float"
                    and dataframe_2.dtypes[column] == "float"
                    and np.allclose(
                        dataframe_1[column].values,
                        dataframe_2[column].values,
                        equal_nan=True,
                    )
                )
                or (
                    (
                        dataframe_1.dtypes[column] == "category"
                        or dataframe_2.dtypes[column] == "category"
                    )
                    and categorical_series_all_equal(
                        dataframe_1[column], dataframe_2[column]
                    )
                )
            ):
                continue
            elif (
                dataframe_1.dtypes[column] == "object"
                and dataframe_2.dtypes[column] == "object"
            ):
                for row_index in dataframe_1.index:
                    entry_1 = dataframe_1.at[row_index, column]
                    entry_2 = dataframe_2.at[row_index, column]
                    if (
                        isinstance(entry_1, list)
                        and isinstance(entry_2, list)
                        and (
                            entry_1 == entry_2
                            or np.allclose(entry_1, entry_2, equal_nan=True)
                        )
                    ) or (
                        isinstance(entry_1, dict)
                        and isinstance(entry_2, dict)
                        and (
                            entry_1 == entry_2
                            or (
                                entry_1.keys() == entry_2.keys()
                                and np.allclose(
                                    list(entry_1.values()),
                                    list(entry_2.values()),
                                    equal_nan=True,
                                )
                            )
                        )
                    ):
                        continue
                    else:
                        print_with_indent(
                            f"Column {column} row {row_index}: {entry_1} vs {entry_2}"
                        )
                        all_matching = False
            else:
                comparison = dataframe_1[column].compare(dataframe_2[column])
                print_with_indent(f"Differences in column {column}")
                print_with_indent(comparison.to_string())
                all_matching = False
        if all_matching:
            print_with_indent("No differences")


# for each .pickle file in the single run directory
for file in (full_run_path / run_number).glob("*.pickle"):
    print("=" * 80)
    print("Processing", file.name)

    # read the dictionary of dataframes from the .pickle file
    dfs_from_full = pd.read_pickle(file)

    # get the corresponding .pickle file in the suspended run directory
    if (suspend_run_path / run_number / file.name).exists():
        dfs_from_suspended = pd.read_pickle(suspend_run_path / run_number / file.name)
    else:
        print(f"\tFile {file.name} does not exist in the suspended run directory")
        dfs_from_suspended = None

    # get the corresponding .pickle file in the restored run directory
    if (resume_run_path / run_number / file.name).exists():
        dfs_from_resumed = pd.read_pickle(resume_run_path / run_number / file.name)
    else:
        dfs_from_resumed = None

    # loop over each of the dataframes in the original dictionary
    for log_key, log_df in dfs_from_full.items():
        if log_key == "_metadata" or (
            file.name.startswith("tlo.simulation") and log_key == "info"
        ):
            continue

        print("\tKey:", log_key)

        if dfs_from_suspended is None and dfs_from_resumed is None:
            print(
                f"\tKey {log_key} does not exist in the suspended or restored run directory"
            )
            continue

        if dfs_from_suspended is None:
            concatenated_value = dfs_from_resumed[log_key]
        elif dfs_from_resumed is None:
            concatenated_value = dfs_from_suspended[log_key]
        else:
            if log_key in dfs_from_suspended and log_key in dfs_from_resumed:
                try:
                    concatenated_value = pd.concat(
                        [dfs_from_suspended[log_key], dfs_from_resumed[log_key]]
                    )
                except (ValueError, KeyError) as e:
                    print("\tError:", e)
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
            robust_compare(log_df, concatenated_value, indent=2)
        except ValueError as e:
            print("\tError:", e)
            print("\toriginal", log_df)
            print("\tconcatenated", concatenated_value)
            print("\toriginal columns", log_df.columns)
            print("\tconcatenated columns", log_df.columns)
            print("-" * 80)
