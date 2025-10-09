from tlo.analysis.utils import create_pickles_locally, parse_log_file
import glob


for draw in [0]:
    for run in range(5):
        log_pattern = f'/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/climate_scenario_runs-2025-09-26T144635Z/{draw}/{run}/climate_scenario_runs__2025-10-06*.log'

        # Get all matching log files
        log_files = glob.glob(log_pattern)

        for log_file in log_files:
            parse_log_file(log_file)
