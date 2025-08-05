from tlo.analysis.utils import create_pickles_locally, parse_log_file

# File paths


# Parse the log file

scenario_output_dir = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/climate_scenario_runs-2025-08-04T121950Z/'

# # get the pickled files if not generated at the batch run
create_pickles_locally(scenario_output_dir = scenario_output_dir, compressed_file_name_prefix='climate_scenario_runs_')
