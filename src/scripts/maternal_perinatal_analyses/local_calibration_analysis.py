
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

files = ['ectopic_rupture_2010_calibration_111__2021-07-02T093216',
         'ectopic_rupture_2015_calibration_111__2021-07-02T093734']

for file in files:
    new_parse_log = {file: parse_log_file(filepath=f"./outputs/calibration_files/{file}.log")}
    logs_dict.update(new_parse_log)

master_dict_an_2010 = dict()
master_dict_an_2015 = dict()
master_dict_la_2010 = dict()
master_dict_la_2015 = dict()
master_dict_pn_2010 = dict()
master_dict_pn_2015 = dict()

for complication in antenatal_comps:
    new_row = {complication: 0}
    master_dict_an_2010.update(new_row)
    master_dict_an_2015.update(new_row)

for complication in intrapartum_comps:
    new_row = {complication: 0}
    master_dict_la_2010.update(new_row)
    master_dict_la_2015.update(new_row)

for complication in postnatal_comps:
    new_row = {complication: 0}
    master_dict_pn_2010.update(new_row)
    master_dict_pn_2015.update(new_row)

for complication in antenatal_comps:
    graph_maker.get_incidence(logs_dict['ectopic_rupture_2010_calibration_111__2021-07-02T093216'],
                              'pregnancy_supervisor', complication, master_dict_an_2010)
    graph_maker.get_incidence(logs_dict['ectopic_rupture_2015_calibration_111__2021-07-02T093734'],
                              'pregnancy_supervisor', complication, master_dict_an_2015)

#for complication in intrapartum_comps:
#    graph_maker.get_incidence(logs_dict['hpd_test_with_pn_levels_2015_calibration_111__2021-06-29T133857'],
#                              'labour', complication, master_dict_la_2010)
#    graph_maker.get_incidence(logs_dict['hpd_test_with_pn_levels_2010_calibration_111__2021-06-29T132104'],
#                              'labour', complication, master_dict_la_2015)

#for complication in postnatal_comps:
#    graph_maker.get_incidence(logs_dict['hpd_test_with_pn_levels_2015_calibration_111__2021-06-29T133857'],
#                              'postnatal_supervisor', complication, master_dict_pn_2010)
#    graph_maker.get_incidence(logs_dict['hpd_test_with_pn_levels_2010_calibration_111__2021-06-29T132104'],
#                              'labour', complication, master_dict_pn_2015)

#total_births_2010 = graph_maker.get_total_births(
#    logs_dict['hpd_test_with_pn_levels_2015_calibration_111__2021-06-29T133857'])
#total_births_2015 = graph_maker.get_total_births(
#    logs_dict['hpd_test_with_pn_levels_2010_calibration_111__2021-06-29T132104'])

graph_maker.get_generic_incidence_graph('induced_abortion', master_dict_an_2010, master_dict_an_2015, 10000, 86,
                                        159, ['orange', 'moccasin'])

graph_maker.get_generic_incidence_graph('complicated_induced_abortion', master_dict_an_2010, master_dict_an_2015,
                                        10000, 0,
                                        0, ['orange', 'moccasin'])
#graph_maker.get_generic_incidence_graph('spontaneous_abortion', master_dict_an_2010, master_dict_an_2015, 10000, 189,
#                                        189, ['orange', 'moccasin'])
