# Check that all registered disease modules have the report_daly_values() function

### from healthburden model
#for module_name in self.recognised_modules_names:
#    assert getattr(self.sim.modules[module_name], 'report_daly_values', None) and \
#           callable(self.sim.modules[module_name].report_daly_values), 'A module that declares use of ' \
#                                                                       'HealthBurden module must have a ' \
#                                                                       'callable function "report_daly_values"'
# can i use the CAUSES_OF_DISABILITY?

# or e.g. for HIV,
# or e.g. for HIV, infections_by_2age_groups_and_sex with key "total_prev"
# tb tbPrevActive
# nothing for measels, only incidence

from pathlib import Path
from tlo.analysis.utils import (
    extract_results,
    summarize,
)
results_folder = Path("/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-07-22T131925Z")
total_num_dalys_by_wealth_and_label = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="infections_by_2age_groups_and_sex",
        column="total_prev",
        #custom_generate_series=get_total_num_dalys_by_wealth_and_label,
        do_scaling=True,
    ))
print(total_num_dalys_by_wealth_and_label)
