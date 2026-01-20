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

# Files to merge:
# analysis_extract_data.py

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

def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None, ):
  
    individual_histories = extract_individual_histories(results_folder)
    for i in range(2):
        print(individual_histories[i].to_csv(f'individual_histories_draw{i}.csv'))
    exit(-1)
    
    
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
    
    print(analysis_script_path)
    print(analysis_script_commit_hash)
    
    # 2. Load json file from output to retrieve:
    # A) Scenario file path
    # B) Commit from which scenario was
    # C) Draw combinations
    # Note: A) and B) will be stored in wandb, C) will be attached to data itself
    scenario_json_data = retrieve_scenario_json_file(results_folder)
    scenario_script_path = scenario_json_data['scenario_script_path']
    scenario_script_commit_hash = scenario_json_data['commit']
    
    print(scenario_script_path)
    print(scenario_script_commit_hash)
    print(scenario_json_data)
    
    draws_parameters = scenario_json_data['draws']
    print(draws_parameters)
    
    # 3. Extract individual histories
    individual_histories = extract_individual_histories(results_folder)
    
    # 4 Postprocess them, i.e. only extract outcomes of interest and 

    # 4. Store in wandb dataset's metadata
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
