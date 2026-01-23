import argparse
from pathlib import Path

import pandas as pd
import os
import wandb
import ast
import torch
from tlo.analysis.utils import extract_individual_histories
import subprocess
import sys
import re

import json
from collections import Counter
from tlo import Date
from datetime import datetime

datetime_format = '%Y-%m-%d %H:%M:%S'


eval_env = {
        'datetime': datetime,  # Add the datetime class to the eval environment
        'pd': pd,              # Add pandas to handle Timestamp
        'Timestamp': pd.Timestamp,  # Specifically add Timestamp for eval
        'NaT': pd.NaT,
        'nan': float('nan'),       # Include NaN for eval (can also use pd.NA if preferred)
        'Counter': Counter
        }
        
        
def remove_diffs(dicts, diffs):
    """
    Removes keys from dicts that appear in diffs.
    Works recursively for nested dictionaries.
    """
    for key, diff_value in diffs.items():
        if isinstance(diff_value, dict):
            # Nested differences, recurse
            nested_dicts = [d.get(key, {}) for d in dicts if key in d]
            remove_diffs(nested_dicts, diff_value)
        else:
            # Remove the differing key from all dicts if it exists
            for d in dicts:
                if key in d:
                    d.pop(key)
                    
def collect_diffs(dicts):
    """
    Takes a list of dictionaries and returns a dictionary of keys with differing values.
    Works recursively for nested dictionaries.
    """
    from collections import defaultdict

    diffs = {}

    # Get all keys across all dictionaries
    all_keys = set().union(*(d.keys() for d in dicts))

    for key in all_keys:
    
        # Skip metadata keys
        if key in ('draw_name', 'draw_number'):
            continue
            
        # Collect all values for this key
        values = [d.get(key, None) for d in dicts]

        # Check if all values are dicts -> recurse
        if all(isinstance(v, dict) for v in values if v is not None):
            # Recursive call
            nested_diff = collect_diffs(values)
            if nested_diff:  # only add if there is a difference
                diffs[key] = nested_diff
        else:
            # Check if values differ
            unique_values = set(map(str, values))  # str to handle unhashable types
            if len(unique_values) > 1:
                if key != 'draw_name' and key != 'draw_number':
                    diffs[key] = values

    return diffs
        
def flatten_resource_access(data):
    result = {}
    for level, resources in data.items():
        for res_type, counter in resources.items():
            for item, count in counter.items():
                key = f"Level{level}_{res_type}_{item}"
                result[key] = count
    return result
        
def string_to_set_simple(s):
    s = s.strip()
    if s == "set()":
        return set()
    # extract numbers inside {}
    import re
    numbers = re.findall(r'\d+', s)
    return set(map(int, numbers))
    
def parse_counter_string(s: str) -> Counter:
    """
    Parse a string representation of a Counter, including nested Counters.
    Works for strings like:
        "Counter({'Over5OPD': 1})"
        "Counter({})"
    """
    s = s.strip()
    if not s.startswith("Counter(") or not s.endswith(")"):
        raise ValueError(f"String does not look like a Counter: {s}")
    
    # Extract inner dict
    inner = s[len("Counter("):-1].strip()
    
    # Empty Counter
    if inner == "{}":
        return Counter()
    
    # Safely evaluate inner dict
    try:
        data = ast.literal_eval(inner)
    except Exception as e:
        raise ValueError(f"Cannot parse Counter string: {s}") from e
    
    # Recursively convert nested Counters if needed
    result = Counter()
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = Counter(v)
        else:
            result[k] = v
    
    return result

    
def update_resource_access(info, resource_access):
    
    resource_types = ['footprint', 'ConsAccess', 'beds']

    if 'treatment_ID' in info:
        # First time accessing resources at this level
        if info['level'] not in resource_access:
            resource_access[info['level']] = {}
            resource_access[info['level']]['footprint'] = Counter()
            resource_access[info['level']]['ConsCall_Item_Used'] = Counter()
            resource_access[info['level']]['ConsCall_Item_Available'] = Counter()
            resource_access[info['level']]['equipment'] = Counter()
            
        # Add footprint (always in)
        resource_access[info['level']]['footprint'] += eval(info['footprint'],eval_env)
        
        # if any of the keys contain 'ConsCall' add them iteratively to the counter
        for key, value in info.items():
            if 'ConsCall' in key:
                conscall_dict = eval(value,eval_env)
                resource_access[info['level']]['ConsCall_Item_Used'] += eval(conscall_dict['Item_Used'],eval_env)
                resource_access[info['level']]['ConsCall_Item_Available'] += eval(conscall_dict['Item_Available'],eval_env)
            if 'equipment' in key:
                equipment_set = eval(value,eval_env)
                resource_access[info['level']]['equipment'].update(equipment_set)
            
    return resource_access
    
    
def convert_datetime(datetime_str):
    return datetime.strptime(datetime_str, datetime_format)


def print_filtered_df(df):
    """
    Prints rows of the DataFrame excluding event_name 'Initialise' and 'Birth'.
    """
    pd.set_option('display.max_colwidth', None)
    filtered = df  # [~df['event_name'].isin(['StartOfSimulation', 'Birth'])]

    dict_cols = ["Info"]
    max_items = 2
    # Step 2: Truncate dictionary columns for display
    if dict_cols is not None:
        for col in dict_cols:
            def truncate_dict(d):
                if isinstance(d, dict):
                    items = list(d.items())[:max_items]  # keep only first `max_items`
                    return dict(items)
                return d
            filtered[col] = filtered[col].apply(truncate_dict)
    print(filtered)

def check_repo_not_dirty(file):

    # Check if the file is tracked by git
    try:
        subprocess.check_output(["git", "ls-files", "--error-unmatch", str(file)])
    except subprocess.CalledProcessError:
        print(f"ERROR: {file.name} is NOT committed (untracked). Commit before proceeding.")
        sys.exit(1)

    # Check that no uncommitted changes exist
    diff = subprocess.run(["git", "diff", "--quiet", str(file)])
    if diff.returncode != 0:
        print(f"ERROR: {file.name} has uncommitted changes. Commit before proceeding.")
        sys.exit(1)

    return True
    
def retrieve_scenario_json_file(output_path):
    json_files = list(output_path.glob("*.json"))

    if len(json_files) == 0:
        print(f"ERROR: No JSON file found in '{output_path}'")
        sys.exit(1)

    if len(json_files) > 1:
        print(f"ERROR: Multiple JSON files found in '{output_path}': {[f.name for f in json_files]}")
        sys.exit(1)

    # Load the unique JSON file
    json_file = json_files[0]
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded JSON file: {json_file.name}")
    return data
    
def retrieve_analysis_script_commit_hash():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def postprocess_individual_histories(individual_histories, draws_parameters):
    """
    """
    list_of_df = []

    # Find differences between two draws; parameters that are common across draws can be added to metadata, differences will be added to dataset.
    # Find all common parameters across draws
    # Artificially inflate differences:
    draws_parameters[0]['parameters']['CervicalCancer']['different_param'] = 30
    draws_parameters[1]['parameters']['CervicalCancer']['different_param'] = 39
    draws_parameters[0]['parameters']['CervicalCancer']['same_param'] = 2
    draws_parameters[1]['parameters']['CervicalCancer']['same_param'] = 2
    draws_parameters[1]['parameters']['CervicalCancer']['different_param+2'] = 57
    # Collect differences
    differences = collect_diffs(draws_parameters)
    # Remove them from the original draws_parameters
    remove_diffs(draws_parameters,differences)
    print(draws_parameters)
    print(differences)

    # Iterate over draws
    for draw in range(len(draws_parameters)):
    
        data_for_draw = []
        
        # For each draw, group by individual
        for person_ID, group in individual_histories[draw].groupby('person_ID_in_draw'):
            print("At person_ID", person_ID)
            if group.iloc[0]['Info']['iht_track_history'] is False:
                continue
                
            # If proceeding, will collect following
            polling_event_found = False
            # The changing or adding of properties from the first_event will be stored in progression_properties
            progression_properties = {}
            running_date = None
        
            episode_start_date = None
            episode_end_date = None
            episode_start_properties = {}
            episode_end_properties = {}
            
            resource_access = {}

            # Iterate over each row in this group
            for idx, row in group.iterrows():
                
                info = row['Info']
                running_date = row['date']
                
                # Update running properties
                if len(progression_properties) == 0:
                    progression_properties = info
                else:
                    progression_properties.update(info)
                    
                # Check if anything was accessed:
                update_resource_access(info, resource_access)
                print(resource_access)
            
                if 'CervicalCancerMainPollingEvent' in row['event_name'] and progression_properties['ce_hpv_cc_status'] == 'cin1':
                    polling_event_found = True

                    # Retain a copy of Polling event
                    polling_event = row.copy()
                
                    # Make note of date of event and of the properties of individual at the time of this event
                    episode_start_date = row['date']
                    episode_start_properties = progression_properties
                    
                # Ensure episode started
                if (episode_start_date is not None):
                    # Check if episode has resolved
                    if progression_properties['is_alive'] is False or progression_properties['ce_hpv_cc_status'] == 'none':
                        episode_end_date = row['date']
                        episode_end_properties = progression_properties

            if episode_start_date is not None and episode_end_date is None:
                print("Episode began but was not completed for this individual")
                
            if polling_event_found:
                # To store for each individual:
                data = {}
                data['person_ID_in_draw'] = person_ID
                data['draw'] = draw
    
                if episode_end_date is not None and episode_start_date is not None:
                    data['duration_of_episode'] = (episode_end_date - episode_start_date).days
                else:
                    data['duration_of_episode'] = None
                
                if len(episode_end_properties)>0:
                    data['is_alive_after_ce'] = episode_end_properties['is_alive']
                else:
                    data['is_alive_after_ce'] = None
                
                resource_access_flatten = flatten_resource_access(resource_access)
                data.update(resource_access_flatten)
                print('data for individual', data)

                data_for_draw.append(data)
        
        df = pd.DataFrame(data_for_draw)
        
        # Now for this draw, attach draw parameter selection to individual as conditional variables
        # for k,v in draws_parameters.items()
        for key,value in differences['parameters'].items():
            print(key, value)
            for module_key, module_value in value.items():
                df[module_key] = module_value[draw]
        
        # Attend draw data
        list_of_df.append(df)
            
    # Concatenate this df to the overall dataset
    dataset = pd.concat(list_of_df, ignore_index=True, sort=False) # This will append data sample from next draws
    
    return dataset


def apply(results_folder: Path, output_folder: Path, log_to_wandb, resourcefilepath: Path = None):
  
    # Dictionary to collect all metadata relevant to this dataset
    metadata = {}

    # 1. Check that analysis file has been committed, and store path + commit
    file = Path(__file__).resolve()
    if log_to_wandb:
        proceed = check_repo_not_dirty(file)
        if proceed:
            print("Repo is clean and can proceed")
    
    # Get project root using git
    git_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )
    # Compute relative path
    metadata['analysis_script_path'] = str(file.relative_to(git_root))
    metadata['analysis_script_commit_hash'] = retrieve_analysis_script_commit_hash()
    
    # 2. Load json file from output to retrieve:
    # A) Scenario file path
    # B) Commit from which scenario was
    # C) Draw combinations
    # Note: A) and B) will be stored in wandb, C) will be attached to data itself
    scenario_json_data = retrieve_scenario_json_file(results_folder)
    metadata['scenario_script_path'] = scenario_json_data['scenario_script_path']
    metadata['scenario_script_commit_hash'] = scenario_json_data['commit']
    metadata['job_ID'] = os.path.basename(os.path.normpath(results_folder))
    draws_parameters = scenario_json_data['draws']
    
    # 3. Extract individual histories
    individual_histories = extract_individual_histories(results_folder)
    print(len(individual_histories))
    for d in range(len(individual_histories)):
        individual_histories[d].to_csv(f'individual_histories_draw{d}.csv')

    # 4 Postprocess them, i.e. only extract outcomes of interest and add draw parameters
    dataset = postprocess_individual_histories(individual_histories, draws_parameters)

    # Only parameters in draws_parameters are the ones common across all draws, so
    # can safely add info from one draw (0) to metadata
    metadata['parameters'] = draws_parameters[0]['parameters']
    
    # 5. Store in wandb dataset's metadata
    if log_to_wandb:

        if wandb.run is not None:
            wandb.finish()
    
        wandb.init(project="dataset-demo", name="test-run2")

        table = wandb.Table(dataframe=dataset)

        artifact = wandb.Artifact(
            "test_dataset",
            type="dataset",
            metadata=metadata
        )

        artifact.add(table, "data")
        wandb.log_artifact(artifact)
        wandb.finish()

if __name__ == "__main__":
    rfp = Path('resources')

    parser = argparse.ArgumentParser(
        description="Produce plots to show the impact each set of treatments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-path",
        help=(
            "Directory to write outputs to. If not specified (set to None) outputs "
            "will be written to value of --results-path argument."
        ),
        type=Path,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--resources-path",
        help="Directory containing resource files",
        type=Path,
        default=Path('resources'),
        required=False,
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        help=(
            "Directory containing results from running "
            "src/scripts/analysis_data_generation/scenario_track_individual_histories.py "
        ),
        default=None,
        required=False
    )
    parser.add_argument(
        "--log-to-wandb",
        action="store_true",
        help="Enable logging"
    )
    args = parser.parse_args()
    assert args.results_path is not None
    results_path = args.results_path

    output_path = results_path if args.output_path is None else args.output_path

    apply(
        results_folder=results_path,
        output_folder=output_path,
        resourcefilepath=args.resources_path,
        log_to_wandb=args.log_to_wandb
    )
