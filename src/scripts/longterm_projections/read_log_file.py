import
from tlo.analysis.utils import (
    parse_log_file,

)

# File paths
#input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/4/longterm_trends_all_diseases__2024-08-02T111712.log'
#input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/3/longterm_trends_all_diseases__2024-08-02T111657.log'
#input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/4/longterm_trends_all_diseases__2024-08-02T111658.log'
#input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/5/longterm_trends_all_diseases__2024-08-02T111706.log'
#input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/6/longterm_trends_all_diseases__2024-08-02T111657.log'
#input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/7/longterm_trends_all_diseases__2024-08-02T111646.log'
#input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/8/longterm_trends_all_diseases__2024-08-02T111652.log'
#input_file = '/Users/rem76/PycharmProjects/TLOmodel/outputs/tbh03@ic.ac.uk/longterm_trends_all_diseases-2024-08-02T111316Z/0/9/longterm_trends_all_diseases__2024-08-02T111628.log'



# Get the results folder from the scenario outputs

# Parse the log file
results = parse_log_file(input_file)
print(results)
