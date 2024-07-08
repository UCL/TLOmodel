import argparse
import os
from pathlib import Path
import datetime

from tlo.analysis.probability_premature_death import (
   get_probability_of_dying_before_70)

if __name__ == "__main__":
    # Parse command line argument
    parser = argparse.ArgumentParser(description="Process results folder path.")
    parser.add_argument("results_folder", type=str, help="Path to the results folder")
    args = parser.parse_args()
    #
    results_folder = Path(args.results_folder)
    print(results_folder)
    p_dying_before_70_dataframe = get_probability_of_dying_before_70(results_folder = results_folder,
        target_period=(datetime.date(2010, 1, 1), datetime.date(2020, 12, 31)),
        summary=False

    )
print(p_dying_before_70_dataframe)

# write to csv
#p_dying_before_70_dataframe.to_csv(results_folder / "p_dying_before_70.csv")
