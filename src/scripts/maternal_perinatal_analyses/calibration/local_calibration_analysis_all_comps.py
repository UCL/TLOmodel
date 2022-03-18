
from scripts.maternal_perinatal_analyses.calibration import graph_maker_for_local_calibration
from tlo.analysis.utils import parse_log_file

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

neonatal_comps = ['congenital_heart_anomaly', 'limb_or_musculoskeletal_anomaly', 'urogenital_anomaly',
                  'digestive_anomaly', 'other_anomaly', 'early_onset_sepsis', 'mild_enceph', 'moderate_enceph',
                  'severe_enceph', 'respiratory_distress_syndrome', 'not_breathing_at_birth', 'macrosomia',
                  'low_birth_weight', 'small_for_gestational_age']


direct_causes = ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion',
                 'severe_gestational_hypertension', 'severe_pre_eclampsia', 'eclampsia', 'antenatal_sepsis',
                 'uterine_rupture', 'intrapartum_sepsis', 'postpartum_sepsis', 'postpartum_haemorrhage',
                 'secondary_postpartum_haemorrhage', 'antepartum_haemorrhage']

indirect_causes = ['AIDS', 'severe_malaria', 'Suicide', 'diabetes', 'chronic_kidney_disease', 'chronic_ischemic_hd']


logs_dict = dict()

new_parse_log_2010 = {2010: parse_log_file(
    log_filepath="./outputs/calibration_files/test_pph_traetment_calibration_111__2021-10-20T133832.log")}
new_parse_log_2015 = {2015: parse_log_file(
    log_filepath="./outputs/calibration_files/anc1_checker_15_calibration_2__2021-09-22T153811.log")}
logs_dict.update(new_parse_log_2010)
logs_dict.update(new_parse_log_2015)

master_dict_an_2010 = dict()
master_dict_an_2015 = dict()
master_dict_la_2010 = dict()
master_dict_la_2015 = dict()
master_dict_pn_2010 = dict()
master_dict_pn_2015 = dict()
master_dict_nb_2010 = dict()
master_dict_nb_2015 = dict()


def update_dicts(comps, dict_2010, dict_2015, module, age_group):
    for complication in comps:
        new_row = {complication: 0}
        dict_2010.update(new_row)
        graph_maker_for_local_calibration.get_incidence(logs_dict[2010], module, complication, dict_2010,
                                                        specific_year=False, year=2010,
                                                        age_group=age_group)
        dict_2015.update(new_row)
        graph_maker_for_local_calibration.get_incidence(logs_dict[2015], module, complication, dict_2015,
                                                        specific_year=False, year=2015,
                                                        age_group=age_group)


update_dicts(antenatal_comps, master_dict_an_2010, master_dict_an_2015, 'pregnancy_supervisor', 'maternal')
update_dicts(intrapartum_comps, master_dict_la_2010, master_dict_la_2015, 'labour', 'maternal')
update_dicts(postnatal_comps, master_dict_pn_2010, master_dict_pn_2015, 'postnatal_supervisor', 'maternal')
update_dicts(neonatal_comps, master_dict_nb_2010, master_dict_nb_2015, 'newborn_outcomes', 'newborn')


def get_crf(dict_incidence, death_label, year):
    comp_death = logs_dict[year]['tlo.methods.demography']['death']['cause'] == death_label
    cfr = (len(comp_death.loc[comp_death].index) / dict_incidence) * 100
    print(cfr)

# graph_maker.get_parity_graphs(logs_dict[2010])


total_births_2010 = graph_maker_for_local_calibration.get_total_births(logs_dict[2010])
total_births_2015 = graph_maker_for_local_calibration.get_total_births(logs_dict[2015])
graph_maker_for_local_calibration.output_distribution_of_ga_at_birth_for_logfile_year(logs_dict[2010])


pregnancies_2011 = graph_maker_for_local_calibration.get_pregnancies_in_a_year(logs_dict[2010], 2010)
pregnancies_2016 = graph_maker_for_local_calibration.get_pregnancies_in_a_year(logs_dict[2015], 2015)

# total_ended_pregnancies = graph_maker_for_local_calibration.get_completed_pregnancies_in_a_year(logs_dict[2010],
#                                                                                               master_dict_an_2010)

graph_maker_for_local_calibration.get_single_year_generic_incidence_graph('low_birth_weight', master_dict_nb_2010,
                                                                          total_births_2010, 12.5,
                                                                          ['firebrick', 'lightcoral'], 100)
