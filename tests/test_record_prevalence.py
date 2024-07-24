# Check that all registered disease modules have the report_daly_values() function

### from healthburden model
#for module_name in self.recognised_modules_names:
#    assert getattr(self.sim.modules[module_name], 'report_daly_values', None) and \
#           callable(self.sim.modules[module_name].report_daly_values), 'A module that declares use of ' \
#                                                                       'HealthBurden module must have a ' \
#                                                                       'callable function "report_daly_values"'
# can i use the CAUSES_OF_DISABILITY?

## for measels"
# logger.info(key="incidence",
#                     data=incidence_summary,
#                     description="summary of measles incidence per 1000 people per month")
# could multiply this by the average duration of disease to get the prevalence?


##schiso
#logger.info(
#    key=f'infection_status_{self.name}',
#    data=flatten_multi_index_series_into_dict_for_logging(data),
#    description='Counts of infection status with this species by age-group and district.'
#)
#'Counts of infection status with this species by age-group and district.'

##stunting
#        logger.info(
#        key='prevalence',
#            data=convert_keys_to_string(d_to_log),
#            description='Current number of children in each stunting category by single year of age.'
#        )
#
#

# Create a dictionary that will map the disease modules to the keys and columns used in the logger to record prevalence
module_prevalence_key = {'stunting':{'prevalence':["(0, 'HAZ<-3')", "(0, '-3<=HAZ<-2')", "(0, 'HAZ>=-2')",
                                                   "(1, 'HAZ<-3')", "(1, '-3<=HAZ<-2')", "(1, 'HAZ>=-2')",
                                                   "(2, 'HAZ<-3')", "(2, '-3<=HAZ<-2')", "(2, 'HAZ>=-2')",
                                                   "(3, 'HAZ<-3')", "(3, '-3<=HAZ<-2')", "(3, 'HAZ>=-2')",
                                                   "(4, 'HAZ<-3')", "(4, '-3<=HAZ<-2')", "(4, 'HAZ>=-2')"]}, # prevalence
                         'measels':['incidence'],
                         'hiv':{'infections_by_2age_groups_and_sex':['total_prev']}
                         'malaria':{'prev_district': list(map(str, range(32)))}, # district prevalences
                         'rti':{'Inj_category_incidence':['number_of_injuries']}, #number
                         'prostate_cancer':{'summary_stats': ['total_none', 'total_prostate_confined', 'total_local_ln', 'total_metastatic']}, # numbers
                         'other_adult_cancers':{'summary_stats': ['total_none', 'total_metastatic', 'total_site_confined', 'total_local_ln']}, #numebers
                         'tb': {'tb_prevalence':{'tbPrevActive','tbPrevActiveAdult', 'tbPrevActiveChild', 'tbPrevLatent', 'tbPrevLatentAdult', 'tbPrevLatentChild'}}
            }


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
