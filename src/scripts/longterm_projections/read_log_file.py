
from tlo.analysis.utils import create_pickles_locally, parse_log_file

# File paths

# input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-12T084811Z/0/1/longterm_trends_all_diseases-2024-09-12T084811Z.log'
# input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-12T084811Z/0/2/longterm_trends_all_diseases-2024-09-12T084811Z.log'
# input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-12T084811Z/0/3/longterm_trends_all_diseases-2024-09-12T084811Z.log'
# input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-12T084811Z/0/4/longterm_trends_all_diseases-2024-09-12T084811Z.log'
# input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-12T084811Z/0/5/longterm_trends_all_diseases-2024-09-12T084811Z.log'
# input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-12T084811Z/0/6/longterm_trends_all_diseases-2024-09-12T084811Z.log'
# input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-12T084811Z/0/7/longterm_trends_all_diseases-2024-09-12T084811Z.log'
# input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-12T084811Z/0/8/longterm_trends_all_diseases-2024-09-12T084811Z.log'
# input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-12T084811Z/0/9/longterm_trends_all_diseases-2024-09-12T084811Z.log'

# Parse the log file
#results = parse_log_file(input_file)


scenario_output_dir = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-12T084811Z/'

# # get the pickled files if not generated at the batch run
create_pickles_locally(scenario_output_dir = scenario_output_dir, compressed_file_name_prefix='longterm_trends_all_diseases__')

