
from tlo.analysis.utils import parse_log_file
from src.scripts.maternal_perinatal_analyses import graph_maker

antenatal_comps = ['spontaneous_abortion', 'induced_abortion', 'spontaneous_abortion_haemorrhage',
                   'induced_abortion_haemorrhage', 'spontaneous_abortion_sepsis',
                   'induced_abortion_sepsis', 'induced_abortion_injury',
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

files = ['cov_anc_ints_test_new_calibration_101__2021-06-22T171454']

for file in files:
    new_parse_log = {file: parse_log_file(filepath=f"./outputs/calibration_files/{file}.log")}
    logs_dict.update(new_parse_log)

master_dict_an = dict()
master_dict_la = dict()

for complication in antenatal_comps:
    new_row = {complication: 0}
    master_dict_an.update(new_row)

for complication in intrapartum_comps:
    new_row = {complication: 0}
    master_dict_la.update(new_row)

for complication in antenatal_comps:
    graph_maker.get_incidence(logs_dict, 'pregnancy_supervisor', complication, master_dict_an)
for complication in intrapartum_comps:
    graph_maker.get_incidence(logs_dict, 'labour', complication, master_dict_la)

graph_maker.get_coverage_of_anc_interventions(logs_dict['cov_anc_ints_test_new_calibration_101__2021-06-22T171454'])

#total_births_2010 = len(logs_dict['fd_2010_age_corr_calibration_67__2021-06-15T155327'][
#                         'tlo.methods.labour']['delivery_setting'])
#total_births_2015 = len(logs_dict['fd_2015_age_corr_calibration_73__2021-06-15T170334'][
#                         'tlo.methods.labour']['delivery_setting'])

# graph_maker.get_htn_disorders_graph(master_dict_an=master_dict_an, master_dict_la=master_dict_la)
# graph_maker.get_anc_coverage_graph(logs_dict)
#graph_maker.get_facility_delivery_graph(logs_dict['fd_2010_age_corr_calibration_67__2021-06-15T155327'],
#                                        total_births_2010, 2010)
#graph_maker.get_anc_coverage_graph(logs_dict['anc_check_new_10_calibration_83__2021-06-16T155440'], 2010)

# graph_maker.get_anc_coverage_graph(logs_dict['fd_2015_age_corr_calibration_71__2021-06-15T163959'],
#                                         total_births_2010, 2010)
# graph_maker.get_anc_coverage_graph(logs_dict['fd_2015_age_corr_calibration_71__2021-06-15T163959'],
#                                         total_births_2015, 2015)
#total_births_2010 = graph_maker.get_total_births(logs_dict['neonatal_check_calibration_94__2021-06-22T145220'])
#total_births_2015 = graph_maker.get_total_births(logs_dict['pnc_check_15_new_calibration_93__2021-06-22T114404'])

#graph_maker.get_pnc_coverage(logs_dict['neonatal_check_calibration_94__2021-06-22T145220'],
#                             total_births_2010, 2010)
#graph_maker.get_pnc_coverage(logs_dict['pnc_check_15_new_calibration_93__2021-06-22T114404'],
#                             total_births_2015, 2015)
