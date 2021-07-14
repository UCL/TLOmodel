
from tlo.analysis.utils import parse_log_file
from src.scripts.maternal_perinatal_analyses import graph_maker

antenatal_comps = ['spontaneous_abortion', 'induced_abortion', 'spontaneous_abortion_haemorrhage',
                   'induced_abortion_haemorrhage', 'spontaneous_abortion_sepsis', 'spontaneous_abortion_other_comp',
                   'induced_abortion_sepsis', 'induced_abortion_injury', 'induced_abortion_other_comp',
                   'complicated_induced_abortion',
                   'complicated_spontaneous_abortion', 'iron_deficiency', 'folate_deficiency', 'b12_deficiency',
                   'mild_anaemia', 'moderate_anaemia', 'severe_anaemia', 'gest_diab',
                   'mild_pre_eclamp', 'mild_gest_htn', 'severe_pre_eclamp', 'eclampsia', 'severe_gest_htn',
                   'placental_abruption', 'severe_antepartum_haemorrhage', 'mild_mod_antepartum_haemorrhage',
                   'clinical_chorioamnionitis', 'PROM', 'ectopic_unruptured', 'multiple_pregnancy', 'placenta_praevia',
                   'ectopic_ruptured', 'syphilis']

intrapartum_comps = ['placental_abruption', 'mild_mod_antepartum_haemorrhage', 'severe_antepartum_haemorrhage',
                     'sepsis', 'uterine_rupture', 'eclampsia', 'severe_gest_htn', 'severe_pre_eclamp',
                     'early_preterm_labour', 'late_preterm_labour', 'post_term_labour', 'obstructed_labour',
                     'pph_uterine_atony', 'pph_retained_placenta', 'pph_other', 'primary_postpartum_haemorrhage']

postnatal_comps = ['vesicovaginal_fistula', 'rectovaginal_fistula', 'sepsis', 'secondary_postpartum_haemorrhage',
                   'iron_deficiency', 'folate_deficiency', 'b12_deficiency', 'mild_anaemia', 'moderate_anaemia',
                   'severe_anaemia',  'mild_pre_eclamp', 'mild_gest_htn', 'severe_pre_eclamp', 'eclampsia',
                   'severe_gest_htn']

direct_causes = ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion',
                 'severe_gestational_hypertension', 'severe_pre_eclampsia', 'eclampsia', 'antenatal_sepsis',
                 'uterine_rupture', 'intrapartum_sepsis', 'postpartum_sepsis', 'postpartum_haemorrhage',
                 'secondary_postpartum_haemorrhage', 'antepartum_haemorrhage']

indirect_causes = ['AIDS', 'severe_malaria', 'Suicide', 'diabetes', 'chronic_kidney_disease', 'chronic_ischemic_hd']


logs_dict = dict()

new_parse_log_2010 = {2010: parse_log_file(
    filepath=f"./outputs/calibration_files/sa_10_with_baseline_effect_calibration_180__2021-07-13T180804.log")}
new_parse_log_2015 = {2015: parse_log_file(
    filepath=f"./outputs/calibration_files/sa_15_with_baseline_effect_calibration_181__2021-07-13T195247.log")}
logs_dict.update(new_parse_log_2010)
logs_dict.update(new_parse_log_2015)

master_dict_an_2010 = dict()
master_dict_an_2015 = dict()
master_dict_la_2010 = dict()
master_dict_la_2015 = dict()
master_dict_pn_2010 = dict()
master_dict_pn_2015 = dict()

def update_dicts(comps, dict_2010, dict_2015, module):
    for complication in comps:
        new_row = {complication: 0}
        dict_2010.update(new_row)
        graph_maker.get_incidence(logs_dict[2010], module, complication, dict_2010, specific_year=False, year=2010
                                  )
        dict_2015.update(new_row)
        graph_maker.get_incidence(logs_dict[2015], module, complication, dict_2015, specific_year=False, year=2015)

update_dicts(antenatal_comps, master_dict_an_2010, master_dict_an_2015, 'pregnancy_supervisor')
update_dicts(intrapartum_comps, master_dict_la_2010, master_dict_la_2015, 'labour')
update_dicts(postnatal_comps, master_dict_pn_2010, master_dict_pn_2015, 'postnatal_supervisor')

#graph_maker.get_parity_graphs(logs_dict[2010])

total_births_2010 = graph_maker.get_total_births(logs_dict[2010])
total_births_2015 = graph_maker.get_total_births(logs_dict[2015])

pregnancies_2011 = graph_maker.get_pregnancies_in_a_year(logs_dict[2010], 2010)
pregnancies_2016 = graph_maker.get_pregnancies_in_a_year(logs_dict[2015], 2015)

# SPONTANEOUS/INDUCED ABORTION...
graph_maker.get_generic_incidence_graph('spontaneous_abortion', master_dict_an_2010, master_dict_an_2015,
                                         10000, 10000, 189, 189, ['firebrick', 'lightcoral'])

#graph_maker.get_abortion_complication_graphs(master_dict_an_2010, master_dict_an_2015, pregnancies_2011,
 #                                            pregnancies_2016, ['firebrick', 'lightcoral'])

#graph_maker.get_anc_coverage_graph(logs_dict[2010], 2010)
#graph_maker.get_anc_coverage_graph(logs_dict[2015], 2015)
#graph_maker.get_facility_delivery_graph(logs_dict[2010], total_births_2010, 2010)
#graph_maker.get_facility_delivery_graph(logs_dict[2015], total_births_2015, 2015)
#graph_maker.get_pnc_coverage(logs_dict[2010], total_births_2010, 2010)
#graph_maker.get_pnc_coverage(logs_dict[2015], total_births_2015, 2015)



#graph_maker.get_total_anaemia_graph(logs_dict[2010], logs_dict[2015], ['red', 'pink'])

#graph_maker.get_generic_incidence_graph('mild_anaemia', master_dict_an_2010, master_dict_an_2015,
#                                        1000, 1000, 195, 225, ['firebrick', 'lightcoral'])
#graph_maker.get_generic_incidence_graph('moderate_anaemia', master_dict_an_2010, master_dict_an_2015,
#                                        1000, 1000, 178, 219, ['firebrick', 'lightcoral'])
#graph_maker.get_generic_incidence_graph('severe_anaemia', master_dict_an_2010, master_dict_an_2015,
#                                        1000, 1000, 15, 2, ['firebrick', 'lightcoral'])

#graph_maker.get_preterm_birth_graph(master_dict_la_2010, master_dict_la_2015, total_births_2010, total_births_2015,
#                                    ['plum', 'thistle'])

#graph_maker.get_htn_disorders_graph(master_dict_an_2010, master_dict_la_2010, master_dict_pn_2010, total_births_2010,
#                                    2010)
#graph_maker.get_htn_disorders_graph(master_dict_an_2015, master_dict_la_2015, master_dict_pn_2015, total_births_2015,
#                                    2015)


