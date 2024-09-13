from tlo.analysis.utils import create_pickles_locally, parse_log_file

# File paths

input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-02T093957Z/0/0/longterm_trends_all_diseases__2024-09-02T094225.log'
input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-02T093957Z/0/1/longterm_trends_all_diseases__2024-09-02T094225.log'
input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-02T093957Z/0/2/longterm_trends_all_diseases__2024-09-02T094225.log'
input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-02T093957Z/0/3/longterm_trends_all_diseases__2024-09-02T094225.log'
input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-02T093957Z/0/4/longterm_trends_all_diseases__2024-09-02T094225.log'
input_file = '/Users/rem76/Documents/longterm_trends_all_diseases__2024-09-04T172323.log'
#overall_results = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/0'

# Get the results folder from the scenario outputs
# Parse the log file
results = parse_log_file(input_file)


scenario_output_dir = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-02T093957Z'


# # get the pickled files if not generated at the batch run
create_pickles_locally(scenario_output_dir = scenario_output_dir, compressed_file_name_prefix='longterm_trends_all_diseases__')

