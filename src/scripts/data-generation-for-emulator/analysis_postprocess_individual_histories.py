import argparse
from pathlib import Path

import pandas as pd

from tlo.analysis.utils import extract_individual_histories


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
# danalysis_extract_data.py

def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None, ):
    
    # 1. Load json file from output to find relevant information about draws

    
    # 2. Extract individual histories
    individual_histories = extract_individual_histories(results_folder)

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
