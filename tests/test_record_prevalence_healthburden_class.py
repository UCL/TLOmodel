import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import pickle
from tlo import Date, Simulation
from tlo.analysis.utils import create_pickles_locally, extract_results, parse_log_file
from tlo.methods.fullmodel import fullmodel

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
outputpath = Path("./outputs")

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 1000
seed = 42


def extract_mapper(key):
    return pd.Series(key.drop(columns={'date'}).loc[0]).to_dict()


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_run_with_healthburden_with_dummy_diseases(tmpdir, seed):
    """Check that everything runs in the simple cases of Mockitis and Chronic Syndrome and that outputs are as expected."""

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=seed, log_config={'filename': 'test_log', 'directory': outputpath})

    # Register the appropriate modules
    sim.register(*fullmodel(
        resourcefilepath=resourcefilepath,
        use_simplified_births=False,
    ))

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath)
    prevalence = output['tlo.methods.healthburden']['prevalence_of_diseases']

    # Test ALRI
    prevalence_ALRIL_function = prevalence['Alri'] # only records incidence in logs

    # Test Antenatal stillbirth
    prevalence_antenatal_stillbirth = prevalence['Antenatal stillbirth']
    #prevalence_antenatal_stillbirth_log = len(output['tlo.methods.pregnancy_supervisor']["antenatal_stillbirth"]["mother"])

    # Test Bladder cancer
    prevalence_bladder_cancer_function = prevalence['BladderCancer']
    prevalence_bladder_cancer_log = output['tlo.methods.bladder_cancer']["summary_stats"]["total_tis_t1"] + output['tlo.methods.bladder_cancer']["summary_stats"]["total_t2p"] + output['tlo.methods.bladder_cancer']["summary_stats"]["total_metastatic"]

    # Test Breast cancer
    prevalence_breast_cancer_function = prevalence['BreastCancer']
    prevalence_breast_cancer_log = output['tlo.methods.breast_cancer']["summary_stats"]["total_stage1"] + output['tlo.methods.breast_cancer']["summary_stats"]["total_stage2"] + output['tlo.methods.breast_cancer']["summary_stats"]["total_stage3"]  + output['tlo.methods.breast_cancer']["summary_stats"]["total_stage4"]

    # Test CardioMetabolicDisorders
    prevalence_diabetes_function = prevalence['diabetes']
    prevalence_diabetes_log = output['tlo.methods.cardio_metabolic_disorders']["diabetes_prevalence"]["prevalence"]

    prevalence_hypertension_function = prevalence['hypertension']
    prevalence_hypertension_log = output['tlo.methods.cardio_metabolic_disorders']["hypertension_prevalence"]["prevalence"]
    #assert prevalence_hypertension_function[0] == prevalence_hypertension_log[0]

    prevalence_chronic_kidney_disease_function = prevalence['chronic_kidney_disease']
    prevalence_chronic_kidney_disease_log = output['tlo.methods.cardio_metabolic_disorders']["chronic_kidney_disease_prevalence"]["prevalence"]
    #assert prevalence_chronic_kidney_disease_function[0] == prevalence_chronic_kidney_disease_log[0]

    prevalence_chronic_lower_back_function = prevalence['chronic_lower_back_pain']
    prevalence_chronic_lower_back_log = output['tlo.methods.cardio_metabolic_disorders']["chronic_lower_back_pain_prevalence"]["prevalence"]
    #assert prevalence_chronic_lower_back_function[0] == prevalence_chronic_lower_back_log[0]

    prevalence_chronic_ischemic_hd_function = prevalence['chronic_ischemic_hd']
    prevalence_chronic_ischemic_hd_log = output['tlo.methods.cardio_metabolic_disorders']["chronic_ischemic_hd_prevalence"]["prevalence"]
    #assert prevalence_chronic_ischemic_hd_function[0] == prevalence_chronic_ischemic_hd_log[0]


    # Test COPD
    prevalence_COPD_function = prevalence['Copd']
    #def sum_copd_prevalence(data):
    #    # Extract the columns dictionary
    #    COPD_prev_sum = 0
    #    columns = data.columns
    #    for col in columns:
    #       COPD_prev_sum += data[col]
    #    return COPD_prev_sum

    #prevalence_COPD_log = [sum_copd_prevalence(output['tlo.methods.copd']["copd_prevalence"])]

    #assert prevalence_COPD_function[0] == prevalence_COPD_log[0]

    # Test Depression
    prevalence_depression_function = prevalence['Depression']
    prevalence_depression_log = output['tlo.methods.depression']["summary_stats"]["prop_ge15_depr"]

    # Test Diarrhoea
    prevalence_diarrhoea = prevalence['Diarrhoea']

    # Test Epilepsy
    prevalence_epilepsy = prevalence['Epilepsy']

    # Test HIV
    prevalence_HIV_function = prevalence['Hiv']
    prevalence_HIV_log = output['tlo.methods.hiv']["summary_inc_and_prev_for_adults_and_children_and_fsw"]["total_plhiv"]/output['tlo.methods.hiv']["summary_inc_and_prev_for_adults_and_children_and_fsw"]["pop_total"]
    #assert prevalence_HIV_function[0] == prevalence_HIV_log[0]

    # Test Intrapartum stillbirths
    prevalence_intrapartum_stillbirths = prevalence['Intrapartum stillbirth']

    # Test Malaria
    prevalence_malaria_function = prevalence['Malaria']
    prevalence_malaria_log = output['tlo.methods.malaria']["prevalence"]["clinical_prev"]

    #assert prevalence_malaria_function[0] == prevalence_malaria_log[0]

    # Test Measles - regular log does not record, only incidence
    prevalence_measles = prevalence['Measles']

    # Test Oesophageal cancer
    prevalence_oesophageal_cancer_function = prevalence['OesophagealCancer']
    prevalence_oesophageal_cancer_log = output['tlo.methods.oesophagealcancer']["summary_stats"]["total_low_grade_dysplasia"] + output['tlo.methods.oesophagealcancer']["summary_stats"]["total_high_grade_dysplasia"] + output['tlo.methods.oesophagealcancer']["summary_stats"]["total_stage1"] + output['tlo.methods.oesophagealcancer']["summary_stats"]["total_stage2"] + output['tlo.methods.oesophagealcancer']["summary_stats"]["total_stage3"]  + output['tlo.methods.oesophagealcancer']["summary_stats"]["total_stage4"]

    # Test Other adult cancer
    prevalence_other_adult_cancer = prevalence['OtherAdultCancer']
    prevalence_other_adult_cancer_cancer_log = output['tlo.methods.other_adult_cancers']["summary_stats"]["total_site_confined"] + output['tlo.methods.other_adult_cancers']["summary_stats"]["total_local_ln"] + output['tlo.methods.other_adult_cancers']["summary_stats"]["total_metastatic"]

    # Test Prostate cancer
    prevalence_prostate_cancer = prevalence['ProstateCancer']
    prevalence_prostate_cancer_log = output['tlo.methods.prostate_cancer']["summary_stats"]["total_prostate_confined"] + output['tlo.methods.prostate_cancer']["summary_stats"]["total_local_ln"] + output['tlo.methods.prostate_cancer']["summary_stats"]["total_metastatic"]

    # Test RTI
    prevalence_RTI = prevalence['RTI']

    # Test Schisto
    prevalence_schisto = prevalence['Schisto']

    # Test TB
    prevalence_tb_function = prevalence['Tb']
    prevalence_tb_log = output['tlo.methods.tb']["tb_prevalence"]["tbPrevActive"] + \
                        output['tlo.methods.tb']["tb_prevalence"]["tbPrevLatent"]

    #assert prevalence_tb_function[0] == prevalence_tb_log[0]
    #assert prevalence_tb_function[-1] == prevalence_tb_log[0]


    # Test maternal deaths
    maternal_deaths_function = prevalence['maternal_deaths']
    death_df = output['tlo.methods.demography']['death']

    direct_deaths = len(death_df[death_df['cause'] == 'Maternal Disorders'])
    properties_deceased = output['tlo.methods.demography.detail']["properties_of_deceased_persons"]
    indirect_deaths_non_hiv = len(properties_deceased.loc[(properties_deceased['is_pregnant'] | properties_deceased['la_is_postpartum']) &
                              (properties_deceased['cause_of_death'].str.contains('Malaria|Suicide|ever_stroke|diabetes|'
                                                                 'chronic_ischemic_hd|ever_heart_attack|'
                                                                 'chronic_kidney_disease') |
                               (properties_deceased['cause_of_death'] == 'TB'))])
    hiv_pd = len(properties_deceased.loc[(properties_deceased['is_pregnant'] | properties_deceased['la_is_postpartum']) &
                              (properties_deceased['cause_of_death'].str.contains('AIDS_non_TB|AIDS_TB'))])

    # Multiply only numeric columns by 0.3 in place
    hiv_indirect_maternal_deaths = hiv_pd * 0.3

    maternal_deaths_log = direct_deaths + indirect_deaths_non_hiv + hiv_indirect_maternal_deaths
    assert maternal_deaths_function[0] == maternal_deaths_log

    # Test newborn deaths
    prevalence_newborn_deaths_function = prevalence['newborn_deaths']
    prevalence_newborn_deaths_log = len(properties_deceased[(properties_deceased['age_days'] < 29) & (properties_deceased['age_years'] == 0) & (properties_deceased['is_alive'] == False)])

    #assert prevalence_newborn_deaths_function.sum()  == prevalence_newborn_deaths_log.sum()
    #assert len(prevalence_newborn_deaths_log) < 1

    return prevalence_newborn_deaths_log, prevalence_newborn_deaths_function, prevalence, prevalence_tb_log


prevalence_newborn_deaths_log, prevalence_newborn_deaths_function, prevalence, prevalence_tb_log = test_run_with_healthburden_with_dummy_diseases(outputpath, seed)
print("prevalence_newborn_deaths_log",prevalence_newborn_deaths_log)
print("prevalence_newborn_deaths_function",prevalence_newborn_deaths_function)
print("prevalence",prevalence)
print("prevalence",prevalence['Tb'])
print("prevalence_tb_log",prevalence_tb_log)


