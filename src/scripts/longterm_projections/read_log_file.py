import datetime
from pathlib import Path

from tlo.analysis.utils import (
    create_pickles_locally,
    extract_params,
    extract_results,
    get_scenario_info,
    load_pickled_dataframes,
    summarize,
    parse_log_file
)


# File paths
#input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/0/longterm_trends_all_diseases__2024-08-02T111712.log'
#input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/4/longterm_trends_all_diseases__2024-08-02T111658.log'
#input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/5/longterm_trends_all_diseases__2024-08-02T111706.log'
#input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/6/longterm_trends_all_diseases__2024-08-02T111657.log'
#input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/7/longterm_trends_all_diseases__2024-08-02T111646.log'
#input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/8/longterm_trends_all_diseases__2024-08-02T111652.log'
#input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/9/longterm_trends_all_diseases__2024-08-02T111628.log'
# = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/0'
#overall_results = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/0'

# Get the results folder from the scenario outputs
# Parse the log file
#results = parse_log_file(input_file)


scenario_output_dir = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/'

# # get the pickled files if not generated at the batch run
create_pickles_locally(scenario_output_dir = scenario_output_dir, compressed_file_name_prefix='longterm_trends_all_diseases__')
