import argparse
from pathlib import Path

import pandas as pd
import os
import wandb
import torch
from tlo.analysis.utils import extract_individual_histories
import subprocess
import sys
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
        }
        
def convert_datetime(datetime_str):
    return datetime.strptime(datetime_str, datetime_format)

def log_on_wandb(dataset, metadata):

    # Start a run, with a type to label it and a project it can call home
    with wandb.init(project="dataset-example", job_type="generate-dataset") as run:

        raw_data = wandb.Artifact(
            "cervical-cancer",
            type="dataset",
            description="TLO-generated dataset for the cervical cancer module",
            metadata=metadata)

        # Store a new file in the artifact, and write something into its contents.
        with raw_data.new_file(name + ".pt", mode="wb") as file:
            x, y = data.tensors
            torch.save((x, y), file)

        # Save the artifact to W&B.
        run.log_artifact(raw_data)

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

def postprocess_individual_histories(individual_histories): #, draws_parameters):

    # Define initial properties of interest
    initial_properties_of_interest = []##TO-DO: fill this in]
    initial_ce_event_properties = set()
    num_runs = 50

    # Iterate over draws
    for draw in range(1):
    
        # For each draw, group by individual
        for person_ID, group in individual_histories[draw].groupby('person_ID_in_draw'):
        
            polling_event_found = False
            # The changing or adding of properties from the first_event will be stored in progression_properties
            progression_properties = {}
            running_date = None
        
            episode_start_date = None
            episode_end_date = None
            episode_start_properties = {}
            episode_end_properties = {}
            
            # Iterate over each row in this group
            for idx, row in group.iterrows():
                
                info = row['Info']
                running_date = row['date']

                if len(progression_properties) == 0:
                    progression_properties = info
                else:
                    progression_properties.update(info)
            
                if 'CervicalCancerMainPollingEvent' in row['event_name'] and progression_properties['ce_hpv_cc_status'] == 'cin1' and progression_properties['sex'] == 'F':
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
            print("Episode start ", episode_start_date)
            print("properties ", episode_start_properties)
            print("Episode end ", episode_end_date)
            print("properties ", episode_end_properties)
            

            print("Overall duration")





def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None, ):
  
    individual_histories = extract_individual_histories(results_folder)
    
    individual_histories[0].to_csv('individual_histories_draw0.csv')
    postprocess_individual_histories(individual_histories)
    
    file = Path(__file__).resolve()
    
    # 1. Check that analysis file has been committed, and store path + commit
    #proceed = check_repo_not_dirty()
    #if proceed:
    #    print("Repo is clean and can proceed")
    
    # Get project root using git
    git_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )
    # Compute relative path
    analysis_script_path = file.relative_to(git_root)
    analysis_script_commit_hash = retrieve_analysis_script_commit_hash()
    
    # 2. Load json file from output to retrieve:
    # A) Scenario file path
    # B) Commit from which scenario was
    # C) Draw combinations
    # Note: A) and B) will be stored in wandb, C) will be attached to data itself
    scenario_json_data = retrieve_scenario_json_file(results_folder)
    scenario_script_path = scenario_json_data['scenario_script_path']
    scenario_script_commit_hash = scenario_json_data['commit']
    draws_parameters = scenario_json_data['draws']
    
    # 3. Extract individual histories
    individual_histories = extract_individual_histories(results_folder)
    
    # 4 Postprocess them, i.e. only extract outcomes of interest and add draw parameters

    # 5. Store in wandb dataset's metadata
    # https://docs.wandb.ai/models/tutorials/artifacts
    

    # 3. Postprocess individual histories, such that for each disease episode, we extract the relevant information needed to train emulators, and ensure that the draw parameters are additionally stored
    # Retreive number of draws
    # for d in [draws]:
        # group individuals by ID
            
            # for each individual
            # evolving_status = {}
            # status_at_start_of_episode = None
            # status_at_concl_of_episode = None
            # average_dalys = 0
            # permanent_
            # average_disability = 0
            # total_dt_included = 0
            # dt_in_prev_disability = 0
            # prev_disability_incurred = 0
            
            # dalys_incurred
            # save initial status from first event (i.e. either birth or StartOfSimulation)
            
            # for events linked to individual except the first one:
                # if event considered is the first official one, freeze status prior event
                
                # in any case, update status
                # status.update(status_in_event)
                
                
        # If the first event  store entired status:
            # status = status_in_event
        # else, update propertiesthem which each subsequent event. Hence the 'status' of the individual evolves at
             # Only update properties which have been changed by event
        # each 'event'-step. This can be logged at a particularly relevant event.
        
        # if i ==
        # Have two key events: start of the disease episode, and conclusion. Must declare the two events to watch out for.
        # Note: there could be multiple events linked to the conclusion, e.g. death and resolution
        
        # Things to calculate on the fly between these two key events:
        # 1. The overall duration of the episode
        # 2. The average DALYs incurred + permanent DALYs incurred
        # 3. Running resource access
        # 4. Did individual die as a result of the episode?
        
        # Finally store:
        # Status of individual at the start
        # 1-4 above
        # Status of individual at the end


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
    args = parser.parse_args()
    assert args.results_path is not None
    results_path = args.results_path

    output_path = results_path if args.output_path is None else args.output_path

    apply(
        results_folder=results_path,
        output_folder=output_path,
        resourcefilepath=args.resources_path
    )


"""
                # Skip the initial_properties, or in other words only consider these if they are 'proper' events
                if row['event_name'] != 'StartOfSimulation' and row['event_name'] != 'Birth':

                    if 'CervicalCancerMainPollingEvent' in row['event_name']:
                        polling_event_found = True

                        # Retain a copy of Polling event
                        polling_event = row.copy()
  
                        # Capture properties statu
                        progression_properties = initial_properties.copy()
                        progression_properties.update(i)
                        progression_properties['event_start_date'] = row['date']
                        
                        # Update parameters of interest following Polling
                        key_first_event = {key: i[key] if key in i else value for key, value in first_event.items()}
                        
                        # Calculate age of individual at time of event
                        key_first_event['age_in_days_at_event'] = (row['date'] - convert_datetime(initial_properties['date_of_birth'])).days
                        

                        
                        # Initialise chain of Dalys incurred
                        if 'ce_disability' in i:
                            prev_disability_incurred = i['ce_disability']
                            prev_date = i['event_date']

                    else:
                        # Progress properties of individual, even if this event is a death
                        progression_properties.update(i)
                        progression_properties['event_date'] = row['date']

                        # If disability has changed as a result of this, recalculate and add previous to rolling average
                        if 'ce_disability' in i:

                            dt_in_prev_disability = (i['event_date'] - prev_date).days
                            #print("Detected change in disability", i['rt_disability'], "after dt=", dt_in_prev_disability)
                            #print("Adding the following to the average", prev_disability_incurred, " x ", dt_in_prev_disability )
                            average_disability += prev_disability_incurred*dt_in_prev_disability
                            total_dt_included += dt_in_prev_disability
                            # Update variables
                            prev_disability_incurred = i['rt_disability']
                            prev_date = i['event_date']

                    # Update running footprint
                    if 'appt_footprint' in i and i['appt_footprint'] != 'Counter()':
                        footprint = i['appt_footprint']
                        if 'Counter' in footprint:
                            footprint = footprint[len("Counter("):-1]
                        apply = eval(footprint, eval_env)
                        ind_Counter[i['level']].update(Counter(apply))
                    
                    # If the individual has died, ensure chain of event is interrupted here and update rolling average of DALYs
                    if 'is_alive' in i and i['is_alive'] is False:
                        if ((i['event_date'] - polling_event['rt_date_inj']).days) > total_dt_included:
                            dt_in_prev_disability = (i['event_date'] - prev_date).days
                            average_disability += prev_disability_incurred*dt_in_prev_disability
                            total_dt_included += dt_in_prev_disability
                        break
                       
            # check_if_beyond_time_range_considered(progression_properties)
            
            if polling_event_found:
                # Compute final properties of individual
                key_last_event['is_alive_after_ce'] = progression_properties['is_alive']
                key_last_event.update({'total_footprint': ind_Counter})

                if key_last_event['duration_days']!=total_dt_included:
                    print("The duration of event and total_dt_included don't match", key_last_event['duration_days'], total_dt_included)
                    exit(-1)
                
                properties = key_first_event | key_last_event
                    
                record.append(properties)
            
    df = pd.DataFrame(record)
    df.to_csv("new_raw_data_" + name_tag + ".csv", index=False)
"""
