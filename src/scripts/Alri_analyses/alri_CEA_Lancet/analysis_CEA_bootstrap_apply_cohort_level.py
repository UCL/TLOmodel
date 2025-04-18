""" This script will output the values for the CEA for the Lancet Commission on Medical oxygen """

import random
from pathlib import Path
import os
from typing import List
import datetime
from math import e
from openpyxl import Workbook
from openpyxl import load_workbook
import scipy.stats as stats

# from tlo.util import random_date, sample_outcome
import numpy.random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from tlo.lm import LinearModel, LinearModelType, Predictor

from tlo import Date, Simulation, DAYS_IN_YEAR
from tlo.methods import (
    alri,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager, epi
)
from tlo.methods.alri import (
    AlriIncidentCase,
    AlriPollingEvent,
    AlriPropertiesOfOtherModules,
    HSI_Alri_Treatment,
    _make_hw_diagnosis_perfect,
    _make_perfect_conditions,
    _make_treatment_and_diagnosis_perfect,
    _reduce_hw_dx_sensitivity,
    _prioritise_oxygen_to_hospitals

)
from tlo.util import sample_outcome

MODEL_POPSIZE = 150_000
MIN_SAMPLE_OF_NEW_CASES = 200
NUM_REPS_FOR_EACH_CASE = 20

dx_accuracy = 'imperfect'

# False if main analysis, True if a sensitivity analysis (HW Dx Accuracy, or Prioritise oxygen in Hospitals)
sensitivity_analysis = False

sensitivity_analysis_hw_dx = True  # change to False if analysing Oxygen prioritisation


# Helper function for conversion between odds and probabilities
to_odds = lambda pr: pr / (1.0 - pr)  # noqa: E731tab e402c

to_prob = lambda odds: odds / (1.0 + odds)  # noqa: E731

# Date for saving the image for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


def get_sim(popsize):
    """Return a simulation (composed of only <5 years old) that has run for 0 days."""
    resourcefilepath = Path('./resources')
    start_date = Date(2024, 1, 1)
    sim = Simulation(start_date=start_date, seed=1)

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(
            resourcefilepath=resourcefilepath,
            force_any_symptom_to_lead_to_healthcareseeking=True,
        ),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  cons_availability='all',
                                  ),
        alri.Alri(resourcefilepath=resourcefilepath),
        # epi.Epi(resourcefilepath=resourcefilepath),
        AlriPropertiesOfOtherModules(),
    )
    sim.modules['Demography'].parameters['max_age_initial'] = 4
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=sim.start_date)
    return sim


# Create simulation (This is needed to generate a population with representative characteristics and to initialise the
# Alri module.)

# Alri module with default values (imperfect diagnosis/ imperfect treatment)
sim = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment = sim.modules['Alri']
hsi_with_imperfect_diagnosis_and_imperfect_treatment = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment, person_id=None)

# Alri module setting for sensitivity analysis - imperfect Hw Dx accuracy -30%
sim0 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_30 = sim0.modules['Alri']
_reduce_hw_dx_sensitivity(alri_module_with_imperfect_diagnosis_30)
hsi_with_imperfect_diagnosis_30 = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_30, person_id=None)

# Alri module setting for sensitivity analysis - prioritise oxygen at hospitals
sim3 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_prioritise_hosp = sim3.modules['Alri']
_prioritise_oxygen_to_hospitals(alri_module_with_imperfect_diagnosis_prioritise_hosp)
hsi_with_imperfect_diagnosis_prioritise_hosp = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_prioritise_hosp, person_id=None)


# Alri module with perfect diagnosis (and imperfect treatment)
sim1 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_perfect_diagnosis = sim1.modules['Alri']
_make_hw_diagnosis_perfect(alri_module_with_perfect_diagnosis)
_make_perfect_conditions(alri_module_with_perfect_diagnosis)
hsi_with_perfect_diagnosis = HSI_Alri_Treatment(module=alri_module_with_perfect_diagnosis, person_id=None)

# Alri module with perfect diagnosis and perfect treatment
sim2 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_perfect_treatment_and_diagnosis = sim2.modules['Alri']
_make_treatment_and_diagnosis_perfect(alri_module_with_perfect_treatment_and_diagnosis)
hsi_with_perfect_diagnosis_and_perfect_treatment = HSI_Alri_Treatment(
    module=alri_module_with_perfect_treatment_and_diagnosis, person_id=None)


def generate_case_mix() -> pd.DataFrame:
    """Generate table of all the cases that may be created"""

    def get_incident_case_mix() -> pd.DataFrame:
        """Return a representative mix of new incidence alri cases (age, sex of person and the pathogen)."""

        alri_polling_event = AlriPollingEvent(module=alri_module_with_perfect_diagnosis)

        # Get probabilities for each person of being infected with each pathogen
        probs_of_acquiring_pathogen = alri_polling_event.get_probs_of_acquiring_pathogen(
            interval_as_fraction_of_a_year=1.0)

        # Sample who is infected and with what pathogen & Repeat 10 times with replacement to generate larger numbers
        new_alri = []
        while len(new_alri) < MIN_SAMPLE_OF_NEW_CASES:
            new_alri.extend(
                [(k, v) for k, v in
                 sample_outcome(probs=probs_of_acquiring_pathogen,
                                rng=alri_module_with_perfect_diagnosis.rng).items()]
            )

        # Return dataframe in which person_id is replaced with age and sex (ignoring variation in vaccine /
        # under-nutrition).
        return pd.DataFrame(columns=['person_id', 'pathogen'], data=new_alri) \
            .merge(sim1.population.props[['age_exact_years', 'sex', 'hv_inf', 'hv_art',
                                          'va_hib_all_doses', 'va_pneumo_all_doses', 'un_clinical_acute_malnutrition']],
                   right_index=True, left_on=['person_id'], how='left') \
            .drop(columns=['person_id'])

    def char_of_incident_case(sex,
                              age_exact_years,
                              hv_inf, hv_art,
                              pathogen,
                              va_hib_all_doses,
                              va_pneumo_all_doses,
                              un_clinical_acute_malnutrition,
                              ) -> dict:
        """Return the characteristics that are determined by IncidentCase (over 1000 iterations), given an infection
        caused by the pathogen"""
        incident_case = AlriIncidentCase(module=alri_module_with_perfect_diagnosis,
                                         person_id=None, pathogen=None)

        samples = []
        for _ in range(NUM_REPS_FOR_EACH_CASE):
            nature_of_char = incident_case.determine_nature_of_the_case(
                age_exact_years=age_exact_years,
                sex=sex,
                pathogen=pathogen,
                va_hib_all_doses=va_hib_all_doses,
                va_pneumo_all_doses=va_pneumo_all_doses,
                un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
            )

            samples.append({**nature_of_char, **{'sex': sex,
                                                 'age_exact_years': age_exact_years,
                                                 'hv_inf': hv_inf, 'hv_art': hv_art,
                                                 'pathogen': pathogen,
                                                 'va_hib_all_doses': va_hib_all_doses,
                                                 'va_pneumo_all_doses': va_pneumo_all_doses,
                                                 'un_clinical_acute_malnutrition': un_clinical_acute_malnutrition,

                                                 }
                            })

        return samples

    incident_case_mix = get_incident_case_mix()
    overall_case_mix = []  # This will be the mix of cases, representing the characteristics of the case as well as the
    #                        pathogen and who they are incident to.
    # df1 = sim1.population.props[['age_exact_years', 'sex', 'hv_inf', 'hv_art',
    #                        'va_hib_all_doses', 'va_pneumo_all_doses', 'un_clinical_acute_malnutrition']]
    # pd.cut(df1['age_exact_years'], [0, 1, 2, 5]).value_counts()

    for x in incident_case_mix.itertuples():
        overall_case_mix.extend(
            char_of_incident_case(sex=x.sex, age_exact_years=x.age_exact_years, hv_inf=x.hv_inf, hv_art=x.hv_art,
                                  pathogen=x.pathogen,
                                  va_hib_all_doses=x.va_hib_all_doses, va_pneumo_all_doses=x.va_pneumo_all_doses,
                                  un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition)
        )

    return pd.DataFrame(overall_case_mix)

def resample_cohort(cohort):
    """Resample cohort with replacement"""
    # n_people = len(cohort)
    cohort = cohort.reset_index(drop=True)
    # Allow same person to be selected multiple times (replace=True), reset the index to an ordered index, do not keep original index as a column
    return cohort.sample(n=len(cohort), replace=True).reset_index(drop=True)



def configuration_to_use(treatment_perfect, hw_dx_perfect):
    """ Use the simulations based on arguments of perfect treatment, perfect hw dx, and scenario """

    # Decide which hsi configuration to use:
    if treatment_perfect:
        hsi = hsi_with_perfect_diagnosis_and_perfect_treatment
        alri_module = alri_module_with_perfect_treatment_and_diagnosis

    else:
        if not sensitivity_analysis:
            if hw_dx_perfect:
                hsi = hsi_with_perfect_diagnosis
                alri_module = alri_module_with_perfect_diagnosis
            else:
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment
        else:
            if sensitivity_analysis_hw_dx:
                hsi = hsi_with_imperfect_diagnosis_30
                alri_module = alri_module_with_imperfect_diagnosis_30
            else:
                hsi = hsi_with_imperfect_diagnosis_prioritise_hosp
                alri_module = alri_module_with_imperfect_diagnosis_prioritise_hosp

    return alri_module, hsi


def treatment_efficacy(
    age_exact_years,
    symptoms,
    oxygen_saturation,
    disease_type,
    complications,
    un_clinical_acute_malnutrition,
    hiv_infected_and_not_on_art,
    duration_in_days_of_alri,
    oximeter_available,
    treatment_perfect,
    hw_dx_perfect,
    facility_level,
    scenario,
):
    """Return the percentage by which the treatment reduce the risk of death"""

    # Decide which hsi configuration to use:

    config = configuration_to_use(treatment_perfect=treatment_perfect,
                                  hw_dx_perfect=hw_dx_perfect,
                                  )
    alri_module = config[0]
    hsi = config[1]

    # availability of oxygen
    oxygen_available = list()
    if scenario.startswith('existing_psa'):
        oxygen_available = alri_module.models.coverage_of_oxygen(scenario='existing_psa')
    elif scenario.startswith('planned_psa'):
        oxygen_available = alri_module.models.coverage_of_oxygen(scenario='planned_psa')
    elif scenario.startswith('all_district_psa'):
        oxygen_available = alri_module.models.coverage_of_oxygen(scenario='all_district_psa')
    elif scenario.startswith('baseline_ant'):
        oxygen_available = alri_module.models.coverage_of_oxygen(scenario='baseline_ant')

    oxygen_available_by_level = {'2': oxygen_available[0], '1b': oxygen_available[1], '1a': oxygen_available[2],
                                 '0': oxygen_available[2]}

    # Get Treatment classification
    classification_for_treatment_decision = hsi._get_disease_classification_for_treatment_decision(
        age_exact_years=age_exact_years, symptoms=symptoms, oxygen_saturation=oxygen_saturation,
        facility_level=facility_level, use_oximeter=oximeter_available,
        hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
        un_clinical_acute_malnutrition=un_clinical_acute_malnutrition)

    imci_symptom_based_classification = alri_module.get_imci_classification_based_on_symptoms(
        child_is_younger_than_2_months=(age_exact_years < 2.0 / 12.0),
        symptoms=symptoms, facility_level='2', hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
        un_clinical_acute_malnutrition=un_clinical_acute_malnutrition)

    # change facility_level if there was a referral
    referral_info = alri_module.referral_from_hc(
        classification_for_treatment_decision=classification_for_treatment_decision, facility_level=facility_level)

    needs_referral = referral_info[0]
    referred_up = referral_info[1]
    new_facility_level = referral_info[2]

    referral_status = 'needs referral, referred' if needs_referral and referred_up else \
        'needs referral, not referred' if needs_referral and not referred_up else \
            'no referral needed' if not needs_referral else None

    # Provision for pre-referral oxygen treatment
    if needs_referral and (oxygen_saturation == '<90%'):
        if oxygen_available_by_level[facility_level]:
            pre_referral_oxygen = 'provided'
        else:
            pre_referral_oxygen = 'not_provided'
    else:
        pre_referral_oxygen = 'not_applicable'

    # Get the treatment selected based on classification given
    ultimate_treatment = alri_module._ultimate_treatment_indicated_for_patient(
        classification_for_treatment_decision=classification_for_treatment_decision,
        age_exact_years=age_exact_years,
        facility_level=new_facility_level,
        oxygen_saturation=oxygen_saturation,
    )

    first_line_iv_failed = False

    # "Treatment Fails" is the probability that a death is averted (if one is schedule)
    treatment_fails = alri_module.models.treatment_fails(
        antibiotic_provided=ultimate_treatment['antibiotic_indicated'][0],
        oxygen_provided=ultimate_treatment['oxygen_indicated'] if oxygen_available_by_level[new_facility_level] and new_facility_level in ('2', '1b') else False,
        imci_symptom_based_classification=imci_symptom_based_classification,
        SpO2_level=oxygen_saturation,
        disease_type=disease_type,
        age_exact_years=age_exact_years,
        symptoms=symptoms,
        complications=complications,
        hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
        un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
        pre_referral_oxygen=pre_referral_oxygen,
        this_is_follow_up=False
    )
    second_line_treatment_fails = alri_module.models.treatment_fails(
        antibiotic_provided='2nd_line_IV_flucloxacillin_gentamicin',
        oxygen_provided=ultimate_treatment['oxygen_indicated'] if oxygen_available_by_level[new_facility_level] and new_facility_level in ('2', '1b') else False,
        imci_symptom_based_classification=imci_symptom_based_classification,
        SpO2_level=oxygen_saturation,
        disease_type=disease_type,
        age_exact_years=age_exact_years,
        symptoms=symptoms,
        complications=complications,
        hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
        un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
        pre_referral_oxygen=pre_referral_oxygen,
        this_is_follow_up=False
    )

    if ultimate_treatment['antibiotic_indicated'][0].startswith('1st_line_IV'):
        if treatment_fails:
            treatment_fails = second_line_treatment_fails
            first_line_iv_failed = True

    # follow-up oral TF --------------------------------------------------

    # do follow-up care for oral TF
    follow_up_care = alri_module.follow_up_treatment_failure(
        original_classification_given=classification_for_treatment_decision,
        symptoms=symptoms)

    sought_follow_up_care = follow_up_care[0]
    follow_up_classification = follow_up_care[1]

    eligible_for_follow_up_care_3_day_amox = all([ultimate_treatment['antibiotic_indicated'][0].endswith('_3days'),
                                                  treatment_fails, sought_follow_up_care, duration_in_days_of_alri > 3])
    eligible_for_follow_up_care_7_day_amox = all([ultimate_treatment['antibiotic_indicated'][0].endswith('_7days'),
                                                  treatment_fails, sought_follow_up_care, duration_in_days_of_alri > 7])
    eligible_for_follow_up_care_5_day_amox = all([ultimate_treatment['antibiotic_indicated'][0].endswith('_5days'),
                                                  treatment_fails, sought_follow_up_care, duration_in_days_of_alri > 5])
    eligible_for_follow_up_care_no_amox = all([ultimate_treatment['antibiotic_indicated'][0] == '',
                                               treatment_fails, sought_follow_up_care, duration_in_days_of_alri > 3])

    eligible_for_follow_up_care = any([eligible_for_follow_up_care_3_day_amox,
                                       eligible_for_follow_up_care_5_day_amox,
                                       eligible_for_follow_up_care_7_day_amox,
                                       eligible_for_follow_up_care_no_amox])

    # change facility_level if there was a referral
    referral_info_follow_up = alri_module.referral_from_hc(
        classification_for_treatment_decision=follow_up_classification,
        facility_level=new_facility_level)

    needs_referral_follow_up = referral_info_follow_up[0]
    referred_up_follow_up = referral_info_follow_up[1]
    new_facility_level_follow_up = referral_info_follow_up[2]

    referral_status_follow_up = 'needs referral, referred' if needs_referral_follow_up and referred_up_follow_up else \
        'needs referral, not referred' if needs_referral_follow_up and not referred_up_follow_up else \
            'no referral needed' if not needs_referral_follow_up else None

    # Provision for pre-referral oxygen treatment
    if needs_referral_follow_up and (oxygen_saturation == '<90%'):
        if oxygen_available_by_level[new_facility_level]:
            pre_referral_oxygen_follow_up = 'provided'
        else:
            pre_referral_oxygen_follow_up = 'not_provided'
    else:
        pre_referral_oxygen_follow_up = 'not_applicable'

    # Get the treatment selected based on classification given
    ultimate_treatment_follow_up = alri_module._ultimate_treatment_indicated_for_patient(
        classification_for_treatment_decision=follow_up_classification,
        age_exact_years=age_exact_years,
        facility_level=new_facility_level_follow_up,
        oxygen_saturation=oxygen_saturation,
    )
    first_line_iv_failed_follow_up = False

    # apply the TFs:
    # "Treatment Fails" is the probability that a death is averted (if one is schedule)
    treatment_fails_at_follow_up = alri_module.models.treatment_fails(
        antibiotic_provided=ultimate_treatment_follow_up['antibiotic_indicated'][0],
        oxygen_provided=ultimate_treatment_follow_up['oxygen_indicated'] if
        oxygen_available_by_level[new_facility_level_follow_up] and new_facility_level_follow_up in ('2', '1b')
        else False,
        imci_symptom_based_classification=imci_symptom_based_classification,
        SpO2_level=oxygen_saturation,
        disease_type=disease_type,
        age_exact_years=age_exact_years,
        symptoms=symptoms,
        complications=complications,
        hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
        un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
        pre_referral_oxygen=pre_referral_oxygen_follow_up,
        this_is_follow_up=True
    )
    second_line_treatment_fails_at_follow_up = alri_module.models.treatment_fails(
        antibiotic_provided='2nd_line_IV_flucloxacillin_gentamicin',
        oxygen_provided=ultimate_treatment_follow_up['oxygen_indicated'] if
        oxygen_available_by_level[new_facility_level_follow_up] and new_facility_level_follow_up in ('2', '1b')
        else False,
        imci_symptom_based_classification=imci_symptom_based_classification,
        SpO2_level=oxygen_saturation,
        disease_type=disease_type,
        age_exact_years=age_exact_years,
        symptoms=symptoms,
        complications=complications,
        hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
        un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
        pre_referral_oxygen=pre_referral_oxygen_follow_up,
        this_is_follow_up=True
    )

    if ultimate_treatment_follow_up['antibiotic_indicated'][0].startswith('1st_line_IV'):
        if treatment_fails_at_follow_up:
            treatment_fails_at_follow_up = second_line_treatment_fails_at_follow_up
            first_line_iv_failed_follow_up = True

    if not eligible_for_follow_up_care:
        return (treatment_fails,
                ultimate_treatment['oxygen_indicated'] if oxygen_available_by_level[new_facility_level] else False,
                new_facility_level,
                first_line_iv_failed,
                (referral_status, pre_referral_oxygen),
                eligible_for_follow_up_care,
                None, None, None, None, None)

    else:
        return (treatment_fails,
                ultimate_treatment['oxygen_indicated'] if oxygen_available_by_level[new_facility_level] else False,
                new_facility_level,
                first_line_iv_failed,
                (referral_status, pre_referral_oxygen),
                (eligible_for_follow_up_care, follow_up_classification),
                treatment_fails_at_follow_up,
                ultimate_treatment_follow_up['oxygen_indicated'] if oxygen_available_by_level[new_facility_level_follow_up] else False,
                new_facility_level_follow_up,
                (referral_status_follow_up, pre_referral_oxygen_follow_up),
                first_line_iv_failed_follow_up)


def generate_table(cohort_df, scenario, oximeter_available, implementation_level):
    """Return table providing a representative case mix of persons with Alri, the intrinsic risk of death and the
    efficacy of treatment under different conditions."""

    # Get Case Mix
    df = cohort_df
    seek_level = list()
    for x in df.itertuples():
        seek_level.append({
            'seek_level': alri_module_with_perfect_diagnosis.seek_care_level(
                symptoms=x.symptoms, age=x.age_exact_years
            )})
    df = df.join(pd.DataFrame(seek_level))

    hiv_not_on_art = list()
    for x in df.itertuples():
        hiv_not_on_art.append({
            'hiv_not_on_art': (x.hv_inf and x.hv_art != "on_VL_suppressed")})
    df = df.join(pd.DataFrame(hiv_not_on_art))

    # Consider risk of death for this person, intrinsically and under different conditions of treatments
    risk_of_death = list()

    hw_dx_perfect = True if dx_accuracy == 'perfect' else False

    for x in df.itertuples():
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10 = treatment_efficacy(
            # Information about the patient:
            age_exact_years=x.age_exact_years,
            symptoms=x.symptoms,
            oxygen_saturation=x.oxygen_saturation,
            disease_type=x.disease_type,
            complications=x.complications,
            hiv_infected_and_not_on_art=x.hiv_not_on_art,
            un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
            duration_in_days_of_alri=x.duration_in_days_of_alri,
            # Information about the care that can be provided:
            scenario=scenario,
            oximeter_available=True if oximeter_available and x.seek_level in implementation_level else False,
            treatment_perfect=False,
            hw_dx_perfect=hw_dx_perfect,
            facility_level=x.seek_level
        )

        # Get the total oxygen for ALRI consumption
        v0p, v1p, v2p, v3p, v4p, v5p, v6p, v7p, v8p, v9p, v10p = treatment_efficacy(
            # Information about the patient:
            age_exact_years=x.age_exact_years,
            symptoms=x.symptoms,
            oxygen_saturation=x.oxygen_saturation,
            disease_type=x.disease_type,
            complications=x.complications,
            hiv_infected_and_not_on_art=x.hiv_not_on_art,
            un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
            duration_in_days_of_alri=x.duration_in_days_of_alri,
            # Information about the care that can be provided:
            scenario='all_district_psa_with_po_level0',
            oximeter_available=True,
            treatment_perfect=False,
            hw_dx_perfect=True,
            facility_level=x.seek_level
        )

        risk_of_death.append({
            'prob_die_if_no_treatment': alri_module_with_perfect_diagnosis.models.prob_die_of_alri(
                age_exact_years=x.age_exact_years,
                sex=x.sex,
                bacterial_infection=x.pathogen,
                disease_type=x.disease_type,
                SpO2_level=x.oxygen_saturation,
                complications=x.complications,
                all_symptoms=x.symptoms,
                un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
            ),

            # Treatment Efficacy at ultimate facility level (for those referred)
            f'treatment_efficacy_scenario_{scenario}_{dx_accuracy}_hw_dx': v0,

            # Oxygen provided at ultimate facility level (for those referred)
            f'oxygen_provided_scenario_{scenario}_{dx_accuracy}_hw_dx': v1,

            # Referred facility level (either 1b or 2)
            f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx': v2,

            # First line IV failed
            f'first_line_iv_failed_scenario_{scenario}_{dx_accuracy}_hw_dx': v3,

            # needs referral and referred status
            f'referral_status_and_oxygen_scenario_{scenario}_{dx_accuracy}_hw_dx': v4,

            # Follow-up properties (ORAL TF) -------------------------------
            f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx': v5,

            f'follow_up_treatment_failure_scenario_{scenario}_{dx_accuracy}_hw_dx': v6,

            f'oxygen_provided_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx': v7,

            f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx': v8,

            f'referral_status_and_oxygen_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx': v9,

            # First line IV failed
            f'first_line_iv_failed_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx': v10,

            f'po_available_seek_level_{scenario}_hw_dx':
                True if oximeter_available and x.seek_level in implementation_level else False,

            # CLASSIFICATION BY PO AVAILABILITY PER SCENARIO
            f'classification_in_{scenario}_perfect_hw_dx':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level,
                    use_oximeter=True if oximeter_available and x.seek_level in implementation_level else False,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition),
            # imperfect HW Dx scenarios ---------------------------------------------------
            f'classification_in_{scenario}_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level,
                    use_oximeter=True if oximeter_available and x.seek_level in implementation_level else False,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition),

            # # # # # # * Classifications * # # # # # #
            # * CLASSIFICATION BY LEVEL 2 *
            'classification_for_treatment_decision_with_oximeter_perfect_accuracy_level2':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level='2', use_oximeter=True, hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition),

            'classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level='2', use_oximeter=False, hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition),

            # * CLASSIFICATION BY LEVEL 1 *
            'classification_for_treatment_decision_with_oximeter_perfect_accuracy_level1':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level='1a', use_oximeter=True, hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition),

            'classification_for_treatment_decision_without_oximeter_perfect_accuracy_level1':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level='1a', use_oximeter=False, hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition),

            # * CLASSIFICATION BY LEVEL THEY SOUGHT CARE *
            'classification_for_treatment_decision_with_oximeter_perfect_accuracy_sought_level':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True, hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition),

            'classification_for_treatment_decision_without_oximeter_perfect_accuracy_sought_level':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=False, hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition),

            # ---------------- PERFECT OXYEGN SCENARIO - GET TOTAL LITERS OF OXYGEN FOR ALRI ---------------
            # Oxygen provided at ultimate facility level (for those referred)
            f'oxygen_provided_scenario_all_district_psa_with_po_level0_perfect_hw_dx': v1p,

            # Referred facility level (either 1b or 2)
            f'final_facility_scenario_all_district_psa_with_po_level0_perfect_hw_dx': v2p,

            # needs referral and referred status
            f'referral_status_and_oxygen_scenario_all_district_psa_with_po_level0_perfect_hw_dx': v4p,

            # Follow-up properties (ORAL TF) -------------------------------
            f'eligible_for_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx': v5p,

            f'oxygen_provided_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx': v7p,

            f'final_facility_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx': v8p,

            f'referral_status_and_oxygen_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx': v9p,

        })

    return df.join(pd.DataFrame(risk_of_death))


if __name__ == "__main__":
    # # # 1. Run the simulation, create a cohort of ALRI
    df = generate_case_mix()

    # # # 2. Perform the bootstrap at the cohort level for each scenario
    all_scenarios_results = []  # store all the scenarios bootstrapping results
    n_bootstrap = 1000

    for i in range(n_bootstrap):
        print(f"\nStarting bootstrap iteration {i+1}")
        # Apply bootstrapping here! re-sample the cohort
        resampled_cohort = resample_cohort(cohort=df)

        # bootstrap iteration results for each scenario
        bootstrap_iteration_results = {}

        # Apply for each scenario
        for scenario in ['baseline_ant', 'baseline_ant_with_po_level2', 'baseline_ant_with_po_level1b',
                         # 'baseline_ant_with_po_level1a', 'baseline_ant_with_po_level0',
                         # 'existing_psa', 'existing_psa_with_po_level2', 'existing_psa_with_po_level1b',
                         # 'existing_psa_with_po_level1a', 'existing_psa_with_po_level0',
                         # 'planned_psa', 'planned_psa_with_po_level2', 'planned_psa_with_po_level1b',
                         # 'planned_psa_with_po_level1a', 'planned_psa_with_po_level0'
                         ]:

            # Pulse oximetry configuration
            oximeter_available = True if 'with_po' in scenario else False
            implementation_level = ('2') if 'with_po_level2' in scenario else ('2', '1b') if 'with_po_level1b' in scenario else \
                ('2', '1b', '1a') if 'with_po_level1a' in scenario else ('2', '1b', '1a', '0') if 'with_po_level0' in scenario else \
                    'none'

              # store each scenario bootstrapping results

            # # # 3. Continue to run the simulation steps with resampled cohort
            # Get the individual outcomes (for each resampled cohort)
            table = generate_table(cohort_df=resampled_cohort, scenario=scenario,
                                   oximeter_available=oximeter_available, implementation_level=implementation_level)

            # # # 4 Get the population metrics (for each resampled cohort)

            # YLD = I × DW × L
            daly_weight = table['classification_for_treatment_decision_with_oximeter_perfect_accuracy_level2'].apply(
                lambda x: 0.133 if 'danger_signs_pneumonia' in x else 0.051)
            duration_years = table['duration_in_days_of_alri'] / 365.25

            YLD = table.index.size * (daly_weight * duration_years).mean()   #1055.7898

            # YLL = N × L
            mean_age = table['age_exact_years'].mean()  # 1.47548
            iq1, iq2 = np.percentile(table['age_exact_years'], [25, 75])
            total_deaths = table.loc[(table[
                f'treatment_efficacy_scenario_{scenario}_{dx_accuracy}_hw_dx']) &
                                     (table[f'follow_up_treatment_failure_scenario_{scenario}_{dx_accuracy}_hw_dx'] != False),
                                     'will_die'].sum()

            # deaths_scenario = table['prob_die_if_no_treatment'] * (1.0 - table[
            #     f'treatment_efficacy_scenario_{scenario}_{dx_accuracy}_hw_dx'] / 100.0)
            # total_deaths = deaths_scenario.sum()

            YLL = total_deaths * (54.7 - mean_age)

            # DALYS
            DALYs = (YLD + YLL).sum()
            DALYs_discounted = ((total_deaths*(1-np.exp(-0.03*(54.7 - mean_age))))/0.03)

            def cea_df_by_scenario(scenario, dx_accuracy):

                # create the series for the CEA dataframe

                low_oxygen = (table["oxygen_saturation"])
                seek_facility = (table["seek_level"])
                classification_by_seek_level = table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx']
                final_facility = table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx']
                final_facility_follow_up = table[f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx']

                number_cases_by_seek_level = table.groupby(
                    by=[classification_by_seek_level, low_oxygen, final_facility]).size()

                final_facility = table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx']
                cases_per_facility = table.groupby(by=[final_facility]).size()

                deaths_per_facility = table.loc[
                    ((table[f'treatment_efficacy_scenario_{scenario}_{dx_accuracy}_hw_dx']) &
                     (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'] == False)) |
                    ((table[f'treatment_efficacy_scenario_{scenario}_{dx_accuracy}_hw_dx']) &
                     (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
                     (table[f'follow_up_treatment_failure_scenario_{scenario}_{dx_accuracy}_hw_dx'] == True)),
                    'will_die'].groupby(by=[seek_facility]).sum()

                need_oxygen_ds_with_SpO2lt90 = number_cases_by_seek_level.sum(level=[0, 1]).reindex(
                    pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%')]))

                full_oxygen_provided = table.loc[((final_facility == '2') | (final_facility == '1b')),
                                                 f'oxygen_provided_scenario_{scenario}_{dx_accuracy}_hw_dx'].sum()

                # add follow-up cases from failure oral treatment
                follow_up_full_oxygen_provided = table.loc[((final_facility == '2') | (final_facility == '1b')),
                                                           f'oxygen_provided_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].sum()

                total_full_oxygen_provided = full_oxygen_provided + follow_up_full_oxygen_provided

                # ------ add the cost of other health care factors -----------

                # ANTIBIOTICS COST --------------
                table['oral_amox_cost'] = table['age_exact_years'].apply(
                    lambda x: (1 * 2 * 5 * 0.02734) if x < 1 else (2 * 2 * 5 * 0.02734) if 1 <= x < 2 else
                    (3 * 2 * 5 * 0.02734))

                oral_antibiotics_cost = table.loc[
                    ((table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] == 'fast_breathing_pneumonia') |
                     (table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] == 'chest_indrawing_pneumonia')),
                    'oral_amox_cost'].sum()

                # IV antibiotics - 1st line - amp + gent , 2nd line - ceftri
                table['ampicillin_unit_cost'] = table['age_exact_years'].apply(
                    lambda x: ((1 / 2.5) * 0.25752) if x < 1 / 3 else
                    ((2 / 2.5) * 0.25752) if 1 / 3 <= x < 1 else
                    ((3 / 2.5) * 0.25752) if 1 <= x < 3 else
                    ((5 / 2.5) * 0.25752))  # ampicillin

                table['gentamycin_unit_cost'] = table['age_exact_years'].apply(
                    lambda x: ((1 / 2) * 0.15037) if x < 1 / 3 else
                    ((1.8 / 2) * 0.15037) if 1 / 3 <= x < 1 else
                    ((2.7 / 2) * 0.15037) if 1 <= x < 3 else
                    ((3.5 / 2) * 0.15037))

                table['ceftriaxone_unit_cost'] = table['age_exact_years'].apply(
                    lambda x: ((3 / 10) * 0.6801) if x < 1 / 3 else
                    ((6 / 10) * 0.6801) if 1 / 3 <= x < 1 else
                    ((10 / 10) * 0.6801) if 1 <= x < 3 else
                    ((14 / 10) * 0.6801))

                first_line_total_cost = table[[f'classification_in_{scenario}_{dx_accuracy}_hw_dx',
                                               f'first_line_iv_failed_scenario_{scenario}_{dx_accuracy}_hw_dx',
                                               'ampicillin_unit_cost',
                                               'gentamycin_unit_cost',
                                               f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx']].apply(
                    lambda x: (x[2] * 4 * 5) + (x[3] * 5) if ((x[0] == 'danger_signs_pneumonia') and (x[1] == False) and
                                                              (x[4] == '2' or x[4] == '1b'))
                    else (x[2] * 4 * 2) + (x[3] * 2) if ((x[0] == 'danger_signs_pneumonia') and x[1] == True) else 0,
                    axis=1).sum()

                second_line_total_cost = table[[f'classification_in_{scenario}_{dx_accuracy}_hw_dx',
                                                f'first_line_iv_failed_scenario_{scenario}_{dx_accuracy}_hw_dx',
                                                'ceftriaxone_unit_cost',
                                                f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx']].apply(
                    lambda x: (x[2] * 5) if ((x[0] == 'danger_signs_pneumonia') and (x[1] == True) and
                                             (x[3] == '2' or x[3] == '1b')) else 0, axis=1).sum()

                iv_antibiotic_cost = first_line_total_cost + second_line_total_cost

                # Oral antibiotic cost at follow-up care
                follow_up_oral_antibiotics_cost = table.loc[
                    (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
                    (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'chest_indrawing_pneumonia'),
                    'oral_amox_cost'].sum()

                # IV antibiotic cost at follow-up care
                first_line_total_cost_fu = table[[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx',
                                                  f'first_line_iv_failed_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx',
                                                  'ampicillin_unit_cost',
                                                  'gentamycin_unit_cost',
                                                  f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx']].apply(
                    lambda x: (x[2] * 4 * 5) + (x[3] * 5) if ((x[0] != False) and (x[0][1] == 'danger_signs_pneumonia') and
                                                              (x[1] == False) and (x[4] == '2' or x[4] == '1b'))
                    else (x[2] * 4 * 2) + (x[3] * 2) if ((x[0] != False) and (x[0][1] == 'danger_signs_pneumonia') and (x[1] == True)) else 0,
                    axis=1).sum()

                second_line_total_cost_fu = table[[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx',
                                                   f'first_line_iv_failed_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx',
                                                   'ceftriaxone_unit_cost',
                                                   f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx']].apply(
                    lambda x: (x[2] * 5) if ((x[0] != False) and (x[0][1] == 'danger_signs_pneumonia') and
                                             (x[1] == True) and (x[3] == '2' or x[3] == '1b')) else 0, axis=1).sum()

                follow_up_iv_antibiotics_cost = first_line_total_cost_fu + second_line_total_cost_fu

                # OUTPATIENT CONSULTATION COST ------------
                table['consultation_cost_seek_level'] = table['seek_level'].apply(
                    lambda x: 2.58 if x == '2' else 2.47 if x == '1b' else 2.17 if x == '1a' else 1.76
                )
                table['consultation_cost_final_facility'] = table[
                    f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx'].apply(
                    lambda x: 2.58 if x == '2' else 2.47 if x == '1b' else 2.17 if x == '1a' else 1.76
                )

                # consultation cost of all outpatient visits - any level and non-sev classification
                consultation_outpatient = table.loc[
                    (table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] != 'danger_signs_pneumonia'),
                    'consultation_cost_seek_level'].sum()

                # danger signs pneumonia initial consultation cost at lower levels before referral
                # consultation cost of ds pneumonia at lower levels of the HS before referral up
                consultation_for_referral = table.loc[
                    (table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] == 'danger_signs_pneumonia') &
                    (table['seek_level'] != table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx']),
                    'consultation_cost_seek_level'].sum()

                # for classified as ds pneumonia at level 1a, or 0 needing referral but not referred
                consultation_for_referral_not_referred = table.loc[
                    (table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] == 'danger_signs_pneumonia') &
                    ((table['seek_level'] == '1a') | (table['seek_level'] == '0')) &
                    (table['seek_level'] == table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx']),
                    'consultation_cost_seek_level'].sum()

                # follow-up consultation costs --------
                # consultation cost of all follow-up outpatient visits - any level and non-sev classification
                follow_up_consultation_outpatient = table.loc[
                    (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
                    (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'chest_indrawing_pneumonia'),
                    'consultation_cost_final_facility'].sum()

                # danger signs pneumonia initial consultation cost at lower levels before referral
                # follow-up consultation cost of ds pneumonia at lower levels of the HS before referral up
                follow_up_consultation_for_referral = table.loc[
                    (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
                    (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'danger_signs_pneumonia') &
                    (final_facility != final_facility_follow_up),
                    'consultation_cost_final_facility'].sum()

                # for classified as ds pneumonia at level 1a, or 0 needing referral but not referred
                follow_up_consultation_for_referral_not_referred = table.loc[
                    (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
                    (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'danger_signs_pneumonia') &
                    ((final_facility == '1a') | (final_facility == '0')) &
                    (final_facility == final_facility_follow_up), 'consultation_cost_final_facility'].sum()

                # INPATIENT BED/DAY COST ------------
                table['inpatient_bed_cost_per_day'] = table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx'].apply(
                    lambda x: 7.94 if x == '2' else 6.89 if x == '1b' else 0)

                table['inpatient_bed_cost'] = table[
                    ['inpatient_bed_cost_per_day',
                     f'first_line_iv_failed_scenario_{scenario}_{dx_accuracy}_hw_dx',
                     f'classification_in_{scenario}_{dx_accuracy}_hw_dx',
                     f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx']].apply(
                    lambda x: (x[0] * 5) if (x[1] == False and x[2] == 'danger_signs_pneumonia') else
                    (x[0] * 7) if (x[1] == True and (x[2] == 'danger_signs_pneumonia') and (x[3] == '2' or x[3] == '1b'))
                    else 0, axis=1)

                hospitalisation_cost = table.loc[
                    (table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] == 'danger_signs_pneumonia') &
                    ((final_facility == '2') | (final_facility == '1b')), 'inpatient_bed_cost'].sum()

                # follow-up inpatient bed days costs --------
                table['inpatient_bed_cost_per_day_fu'] = table[
                    f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].apply(
                    lambda x: 7.94 if x == '2' else 6.89 if x == '1b' else 0)

                table['inpatient_bed_cost_follow_up'] = table[
                    ['inpatient_bed_cost_per_day_fu',
                     f'first_line_iv_failed_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx',
                     f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx',
                     f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx']].apply(
                    lambda x: (x[0] * 5) if ((x[1] == False) and (x[2] != False) and x[2][1] == 'danger_signs_pneumonia') else
                    (x[0] * 7) if ((x[1] == True) and (x[2] != False) and (x[2][1] == 'danger_signs_pneumonia') and
                                   (x[3] == '2' or x[3] == '1b')) else 0, axis=1)

                follow_up_hospitalisation_cost = table.loc[
                    (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
                    (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'danger_signs_pneumonia') &
                    ((final_facility_follow_up == '2') | (final_facility_follow_up == '1b')),
                    'inpatient_bed_cost_follow_up'].sum()

                # PULSE OXIMETRY COST ----------------------
                po_cost_by_usage_rate = {'perfect': [0.167, 0.096, 0.063], 'imperfect': [0.166777, 0.095711, 0.063215]}

                table['PO_cost_seek_level'] = table['seek_level'].apply(
                    lambda x: po_cost_by_usage_rate[dx_accuracy][0] if x in ('2', '1b') else
                    po_cost_by_usage_rate[dx_accuracy][1] if x == '1a' else po_cost_by_usage_rate[dx_accuracy][2])
                table['PO_cost_final_level'] = table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx'].apply(
                    lambda x: po_cost_by_usage_rate[dx_accuracy][0] if x in ('2', '1b') else
                    po_cost_by_usage_rate[dx_accuracy][1] if x == '1a' else po_cost_by_usage_rate[dx_accuracy][2])
                table['PO_cost_final_level_follow_up'] = table[
                    f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].apply(
                    lambda x: po_cost_by_usage_rate[dx_accuracy][0] if x in ('2', '1b') else
                    po_cost_by_usage_rate[dx_accuracy][1] if x == '1a' else po_cost_by_usage_rate[dx_accuracy][2])

                # table['PO_cost_final_level'] = table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx'].apply(
                #     lambda x: 0.15897 if x in ('2', '1b') else 0.08667 if x == '1a' else 0.06025)
                # table['PO_cost_final_level_follow_up'] = table[
                #     f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].apply(
                #     lambda x: 0.15897 if x in ('2', '1b') else 0.08667 if x == '1a' else 0.06025 if x == '0' else 0)

                table['po_available_referral_level'] = table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx'].apply(
                    lambda x: True if oximeter_available and x[0] in implementation_level else False)
                table['po_available_referral_level_at_follow_up'] = final_facility_follow_up.apply(
                    lambda x: True if oximeter_available and x is not None and x in implementation_level else False)

                po_cost_sought_level = table.loc[(table[f'po_available_seek_level_{scenario}_hw_dx']) &
                                                 (seek_facility == final_facility), 'PO_cost_seek_level'].sum()

                po_cost_referred_level = table.loc[(table['po_available_referral_level']) &
                                                   (seek_facility != final_facility), 'PO_cost_final_level'].sum()

                po_cost_referred_level_at_follow_up = table.loc[
                    (table['po_available_referral_level_at_follow_up']) &
                    (final_facility != final_facility_follow_up), 'PO_cost_final_level_follow_up'].sum()

                # OXYGEN COST ----------------
                oxygen_scenario = 'existing_psa' if scenario.startswith('existing') else \
                    'planned_psa' if scenario.startswith('planned') else \
                        'all_district_psa' if scenario.startswith('all_district') else 'baseline_ant'

                # oxygen unit cost changes with PO vs without
                p = sim1.modules['Alri'].parameters
                unit_cost_by_po_level_implementation = {
                    'none': p[f'oxygen_unit_cost_by_po_implementation_{oxygen_scenario}_{dx_accuracy}_hw_dx'][0],
                    '2': p[f'oxygen_unit_cost_by_po_implementation_{oxygen_scenario}_{dx_accuracy}_hw_dx'][1],
                    ('2', '1b'): p[f'oxygen_unit_cost_by_po_implementation_{oxygen_scenario}_{dx_accuracy}_hw_dx'][2],
                    ('2', '1b', '1a'): p[f'oxygen_unit_cost_by_po_implementation_{oxygen_scenario}_{dx_accuracy}_hw_dx'][3],
                    ('2', '1b', '1a', '0'): p[f'oxygen_unit_cost_by_po_implementation_{oxygen_scenario}_{dx_accuracy}_hw_dx'][4]}

                # calculate the treatment cost of oxygen for this scenario in analysis
                table[f'oxygen_cost_full_treatment'] = table['age_exact_years'].apply(
                    lambda x: (unit_cost_by_po_level_implementation[implementation_level] * 1 * 60 * 24 * 3) if x < 1/6 else
                    (unit_cost_by_po_level_implementation[implementation_level] * 2 * 60 * 24 * 3))

                # stabilization oxygen cost
                table[f'oxygen_for_stabilization_cost'] = table['age_exact_years'].apply(
                    lambda x: (unit_cost_by_po_level_implementation[implementation_level] * 1 * 60 * 6) if x < 1/6 else
                    (unit_cost_by_po_level_implementation[implementation_level] * 2 * 60 * 6))

                oxygen_cost_full_treatment = table.loc[
                    ((final_facility == '2') | (final_facility == '1b')) &
                    (table[f'oxygen_provided_scenario_{scenario}_{dx_accuracy}_hw_dx']),
                    'oxygen_cost_full_treatment'].sum()

                # cost the stabilisation oxygen at level 1a regardless or their referral completion
                oxygen_cost_for_stabilisation = table.loc[
                    (table[f'referral_status_and_oxygen_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] != 'no referral needed') &
                    (table[f'referral_status_and_oxygen_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'provided'),
                    'oxygen_for_stabilization_cost'].sum()

                # oxygen cost at follow_up
                follow_up_oxygen_cost_full_treatment = table.loc[
                    (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
                    (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'danger_signs_pneumonia') &
                    ((final_facility_follow_up == '2') | (final_facility_follow_up == '1b')) &
                    (table[f'oxygen_provided_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx']),
                    'oxygen_cost_full_treatment'].sum()

                # cost the stabilisation oxygen at level 1a regardless or their referral completion
                follow_up_oxygen_cost_for_stabilisation = table.loc[
                    (table[f'referral_status_and_oxygen_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] != 'no referral needed') &
                    (table[f'referral_status_and_oxygen_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'provided'),
                    'oxygen_for_stabilization_cost'].sum()

                # get total oxygen liters ------------
                table['total_liters_full'] = table['age_exact_years'].apply(
                    lambda x: (1 * 60 * 24 * 3) if x < 1/6 else (2 * 60 * 24 * 3))
                total_liters_full_sum = table.loc[
                    ((final_facility == '2') | (final_facility == '1b')) &
                    (table[f'oxygen_provided_scenario_{scenario}_{dx_accuracy}_hw_dx']),
                    'total_liters_full'].sum()
                total_liters_full_sum_fup = table.loc[
                    (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
                    (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'danger_signs_pneumonia') &
                    ((final_facility_follow_up == '2') | (final_facility_follow_up == '1b')) &
                    (table[f'oxygen_provided_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx']),
                    'total_liters_full'].sum()

                table['total_liters_stab'] = table['age_exact_years'].apply(
                    lambda x: (1 * 60 * 6) if x < 1/6 else(2 * 60 * 6))
                total_liters_stab_sum = table.loc[
                    (table[f'referral_status_and_oxygen_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] != 'no referral needed') &
                    (table[f'referral_status_and_oxygen_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'provided'),
                    'total_liters_stab'].sum()
                total_liters_stab_sum_fup = table.loc[
                    (table[f'referral_status_and_oxygen_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] != 'no referral needed') &
                    (table[f'referral_status_and_oxygen_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'provided'),
                    'total_liters_stab'].sum()

                total_ox_liters_used = total_liters_full_sum + total_liters_full_sum_fup + total_liters_stab_sum + total_liters_stab_sum_fup
                # ----------------------------------

                # get total oxygen liters in perfect system ------------
                total_liters_full_sum_perfect = table.loc[
                    ((table[f'final_facility_scenario_all_district_psa_with_po_level0_perfect_hw_dx'] == '2') |
                     (table[f'final_facility_scenario_all_district_psa_with_po_level0_perfect_hw_dx'] == '1b')) &
                    (table[f'oxygen_provided_scenario_all_district_psa_with_po_level0_perfect_hw_dx']),
                    'total_liters_full'].sum()
                total_liters_full_sum_fup_perfect = table.loc[
                    (table[f'eligible_for_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx'].str[0] == True) &
                    (table[f'eligible_for_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx'].str[1] == 'danger_signs_pneumonia') &
                    ((table[f'final_facility_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx'] == '2') |
                     (table[f'final_facility_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx'] == '1b')) &
                    (table[f'oxygen_provided_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx']),
                    'total_liters_full'].sum()

                total_liters_stab_sum_perfect = table.loc[
                    (table[f'referral_status_and_oxygen_scenario_all_district_psa_with_po_level0_perfect_hw_dx'].str[0] != 'no referral needed') &
                    (table[f'referral_status_and_oxygen_scenario_all_district_psa_with_po_level0_perfect_hw_dx'].str[1] == 'provided'),
                    'total_liters_stab'].sum()
                total_liters_stab_sum_fup_perfect = table.loc[
                    (table[f'referral_status_and_oxygen_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx'].str[0] != 'no referral needed') &
                    (table[f'referral_status_and_oxygen_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx'].str[1] == 'provided'),
                    'total_liters_stab'].sum()

                total_ox_liters_used_perfect = total_liters_full_sum_perfect + total_liters_full_sum_fup_perfect + total_liters_stab_sum_perfect + total_liters_stab_sum_fup_perfect
                #-------------------------

                # Get the total costs
                total_costs = list()

                total_costs.append({
                    # Total IV antibiotic cost
                    f'sum_iv_ant_cost_{scenario}': iv_antibiotic_cost + follow_up_iv_antibiotics_cost,
                    # Total oral antibiotic cost
                    f'sum_oral_ant_cost_{scenario}': oral_antibiotics_cost + follow_up_oral_antibiotics_cost,
                    # Total outpatient consultation cost
                    f'sum_consultation_cost_{scenario}':
                        consultation_outpatient + consultation_for_referral +
                        consultation_for_referral_not_referred +
                        follow_up_consultation_outpatient + follow_up_consultation_for_referral +
                        follow_up_consultation_for_referral_not_referred,
                    # Total hospitalisation cost
                    f'sum_hospitalisation_cost_{scenario}':
                        hospitalisation_cost + follow_up_hospitalisation_cost,
                    # Total PO cost
                    f'sum_po_cost_{scenario}': po_cost_sought_level + po_cost_referred_level +
                                               po_cost_referred_level_at_follow_up,
                    # Total oxygen cost
                    f'sum_oxygen_cost_{scenario}':
                        oxygen_cost_full_treatment + oxygen_cost_for_stabilisation +
                        follow_up_oxygen_cost_full_treatment + follow_up_oxygen_cost_for_stabilisation
                })

                costs_df = pd.DataFrame(total_costs)

                return deaths_per_facility, costs_df, \
                       need_oxygen_ds_with_SpO2lt90, total_full_oxygen_provided, total_ox_liters_used, \
                       total_ox_liters_used_perfect


            # Do the Dataframe with summary output --------------------------------------

            # Baseline antibiotics
            cea_df = cea_df_by_scenario(scenario=scenario, dx_accuracy=dx_accuracy)
            deaths = cea_df[0].sum()
            costs = cea_df[1]
            total_cost = cea_df[1].sum(axis=1)
            oxygen_provided = cea_df[3]
            need_oxygen = cea_df[2]
            oxygen_liters_provided = cea_df[4]
            total_ox_liters_for_alri = cea_df[5]
            total_costs = costs[f'sum_oral_ant_cost_{scenario}'][0] + costs[f'sum_consultation_cost_{scenario}'][0] + \
                          costs[f'sum_iv_ant_cost_{scenario}'][0] + costs[f'sum_hospitalisation_cost_{scenario}'][0] + \
                          costs[f'sum_po_cost_{scenario}'][0] + \
                          ((3051794 / total_ox_liters_for_alri) if scenario.startswith('existing_psa') else
                           (5884936 / total_ox_liters_for_alri) if scenario.startswith('planned_psa') else
                           (8048553 / total_ox_liters_for_alri) if scenario.startswith('all_district_psa') else
                           0.0 if scenario.startswith('baseline_ant') else None)

            # get descriptives
            pop_size = len(table)
            age_lt1 = table['age_exact_years'].apply(lambda x: 1 if x < 1 else 0).sum()
            age_1to2 = table['age_exact_years'].apply(lambda x: 1 if 1 <= x < 2 else 0).sum()
            age_2to5 = table['age_exact_years'].apply(lambda x: 1 if 2 <= x < 5 else 0).sum()
            pathogen_distributon = table.groupby('pathogen').size()
            coinfection = table.groupby('bacterial_coinfection').size().sum()   # 8.5%?
            disease_type_prop = table.groupby('disease_type').size() # / table.groupby('disease_type').size().sum()
            mean_duration = table['duration_in_days_of_alri'].mean()
            sd_duration = table['duration_in_days_of_alri'].std()
            care_seeking_level = table.groupby('seek_level').size()
            hypoxaemia = table.groupby('oxygen_saturation').size()
            pc_complications = table['complications'].apply(lambda x: 1 if any(
                e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x) else 0).sum()
            pleural_effusion = table['complications'].apply(lambda x: 1 if 'pleural_effusion' in x else 0).sum()
            empyema = table['complications'].apply(lambda x: 1 if 'empyema' in x else 0).sum()
            lung_abscess = table['complications'].apply(lambda x: 1 if 'lung_abscess' in x else 0).sum()
            pneumothorax = table['complications'].apply(lambda x: 1 if 'pneumothorax' in x else 0).sum()
            bacteraemia = table['complications'].apply(lambda x: 1 if 'bacteraemia' in x else 0).sum()
            all_complications = table['complications'].apply(lambda x: 1 if any(
                e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax', 'bacteraemia', 'hypoxaemia'] for e in x) else 0).sum()

            malnutrition = table.groupby('un_clinical_acute_malnutrition').size()
            hiv_pos = table.groupby('hv_inf').size()
            hiv_pos_not_on_art = table.groupby('hiv_not_on_art').size()
            will_die = table.groupby('will_die').size()
            will_die_by_class = table['will_die'].groupby(table['classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2']).sum()
            will_die_by_ox_sat = table['will_die'].groupby(
                table['oxygen_saturation']).sum()  # <90% 0.30552, 90-92%: 0.13409

            # TF
            TF = table[f'treatment_efficacy_scenario_{scenario}_{dx_accuracy}_hw_dx'].groupby(by=[table[f'classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2'], table['disease_type'], table['oxygen_saturation']]).sum() / \
                 table.groupby(by=[table[f'classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2'], table['disease_type'], table['oxygen_saturation']]).size()


            will_die_with_treatment = table.loc[
                (((table[f'treatment_efficacy_scenario_{scenario}_{dx_accuracy}_hw_dx']) &
                  (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'] == False)) |
                 ((table[f'treatment_efficacy_scenario_{scenario}_{dx_accuracy}_hw_dx']) &
                  (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
                  (table[f'follow_up_treatment_failure_scenario_{scenario}_{dx_accuracy}_hw_dx'] == True))),
                'will_die'].groupby(by=[table[f'classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2'], table['oxygen_saturation']]).sum()

            will_die_with_treatment_prop = will_die_with_treatment / table.groupby(
                [table['classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2'],
                 table['oxygen_saturation']]).size()  # 15.566% prob die with treatment spo2<90%, 4.285% spo2 90-92% - baseline; perfect hw dx = <90%: 0.094527588, 90-92: 0.01928559

            classification_distribution = table.groupby(
                [table['classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2'],
                 table['oxygen_saturation']]).size() / table.groupby(
                [table['classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2'],
                 table['oxygen_saturation']]).size().sum()

            debug_point = 0

            #-----------------------------
            # # #  Get the final metrics for CI estimation
            data = {
                'alri_size': pop_size,
                'mean_age': mean_age,
                'deaths': deaths,
                'DALYs_discounted': DALYs_discounted,
                'cost of oral antibiotics': costs[f'sum_oral_ant_cost_{scenario}'][0],
                'cost of outpatient': costs[f'sum_consultation_cost_{scenario}'][0],
                'cost of IV antibiotics': costs[f'sum_iv_ant_cost_{scenario}'][0],
                'cost of inpatient bed': costs[f'sum_hospitalisation_cost_{scenario}'][0],
                'cost of PO': costs[f'sum_po_cost_{scenario}'][0],
                'cost of Oxygen': (3051794 / total_ox_liters_for_alri) if scenario.startswith('existing_psa') else
                (5884936 / total_ox_liters_for_alri) if scenario.startswith('planned_psa') else
                (8048553 / total_ox_liters_for_alri) if scenario.startswith('all_district_psa') else
                0.0 if scenario.startswith('baseline_ant') else None,
                'total_costs': total_costs,
                'need_oxygen': need_oxygen[0],
                'oxygen_provided': oxygen_provided,
                'oxygen_liters_provided': oxygen_liters_provided,
                'total_ox_liters_for_alri': total_ox_liters_for_alri,
                # 'age <1': age_lt1,
                # 'age 1-2': age_1to2,
                # 'age 2-5': age_2to5,
                # 'Enterobacteriaceae': pathogen_distributon['Enterobacteriaceae'],
                # 'H.influenzae_non_type_b': pathogen_distributon['H.influenzae_non_type_b'],
                # 'HMPV': pathogen_distributon['HMPV'],
                # 'Hib': pathogen_distributon['Hib'],
                # 'Influenza': pathogen_distributon['Influenza'],
                # 'P.jirovecii': pathogen_distributon['P.jirovecii'],
                # 'Parainfluenza': pathogen_distributon['Parainfluenza'],
                # 'RSV': pathogen_distributon['RSV'],
                # 'Rhinovirus': pathogen_distributon['Rhinovirus'],
                # 'Staph_aureus': pathogen_distributon['Staph_aureus'],
                # 'Strep_pneumoniae_PCV13': pathogen_distributon['Strep_pneumoniae_PCV13'],
                # 'Strep_pneumoniae_non_PCV13': pathogen_distributon['Strep_pneumoniae_non_PCV13'],
                # 'other_Strepto_Enterococci': pathogen_distributon['other_Strepto_Enterococci'],
                # 'other_bacterial_pathogens': pathogen_distributon['other_bacterial_pathogens'],
                # 'other_pathogens_NoS': pathogen_distributon['other_pathogens_NoS'],
                # 'other_viral_pathogens': pathogen_distributon['other_viral_pathogens'],
                # 'coinfection': coinfection,
                # 'disease_type_other_alri': disease_type_prop['other_alri'],
                # 'disease_type_pneumonia': disease_type_prop['pneumonia'],
                # 'mean_duration_in_days': mean_duration,
                # 'sd_duration': sd_duration,
                # 'care_seeking_level0': care_seeking_level['0'],
                # 'care_seeking_level1a': care_seeking_level['1a'],
                # 'care_seeking_level1b': care_seeking_level['1b'],
                # 'care_seeking_level2': care_seeking_level['2'],
                # 'hypoxaemia_90-92%': hypoxaemia['90-92%'],
                # 'hypoxaemia_<90%': hypoxaemia['<90%'],
                # 'hypoxaemia_>=93%': hypoxaemia['>=93%'],
                # 'any_pc_complications': pc_complications,
                # 'pleural_effusion': pleural_effusion,
                # 'empyema': empyema,
                # 'lung_abscess': lung_abscess,
                # 'pneumothorax': pneumothorax,
                # 'bacteraemia': bacteraemia,
                # 'all_complications': all_complications,
                # 'MAM': malnutrition['MAM'],
                # 'SAM': malnutrition['SAM'],
                # 'well_nourished': malnutrition['well'],
                # 'hiv_negative': hiv_pos[False],
                # 'hiv_positive': hiv_pos[True],
                # 'hiv_pos_not_on_art': hiv_pos_not_on_art[True],
                # 'will_die': will_die[True],
                # 'prob_will_die_no_treat': will_die[True] / will_die.sum(),
                # 'will_die_by_class_ci': will_die_by_class['chest_indrawing_pneumonia'],
                # 'will_die_by_class_cc': will_die_by_class['cough_or_cold'],
                # 'will_die_by_class_ds': will_die_by_class['danger_signs_pneumonia'],
                # 'will_die_by_class_fb': will_die_by_class['fast_breathing_pneumonia'],
                # 'will_die_by_ox_sat_90-92%': will_die_by_ox_sat['90-92%'],
                # 'will_die_by_ox_sat_<90%': will_die_by_ox_sat['<90%'],
                # 'will_die_by_ox_sat_>=93%': will_die_by_ox_sat['>=93%'],
                # 'prob_will_die_by_ox_sat_90-92%': will_die_by_ox_sat['90-92%'] / hypoxaemia['90-92%'],
                # 'prob_will_die_by_ox_sat_<90%': will_die_by_ox_sat['<90%'] / hypoxaemia['<90%'],
                # 'prob_will_die_by_ox_sat_>=93%': will_die_by_ox_sat['>=93%'] / hypoxaemia['>=93%'],
                # 'age IQR_25': iq1, 'age IQR_75': iq2,
            }

            # Store individual iteration results with scenario identifier
            bootstrap_iteration_results[scenario] = data
            print(f"Scenario {scenario} completed")

        # Append results from this iteration
        all_scenarios_results.append(bootstrap_iteration_results)
        print(f"Results appended, current length: {len(all_scenarios_results)}")

    debug_point = 0
    print(f"Final length of all_scenarios_results: {len(all_scenarios_results)}")

    len(all_scenarios_results)

    # store the results per scenario
    scenario_results = {}
    for scenario in all_scenarios_results[0].keys():
        # Initialize dictionary for this scenario
        scenario_results[scenario] = {}
        for metric in all_scenarios_results[0][scenario].keys():
            # print(run[scenario][metric] for run in all_scenarios_results)
            values = [run[scenario][metric] for run in all_scenarios_results]
            mean = np.mean(values)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            scenario_results[scenario][metric] = f"{mean:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})"

    # get the cost difference, DALYs and death averted, ICERs and INB
    def calculate_metric_difference(scenario_name, run_results, metric_key, reverse=False):
        """Calculate difference between scenario and baseline for a single run."""
        baseline_value = run_results['baseline_ant'][metric_key]
        scenario_value = run_results[scenario_name][metric_key]
        return baseline_value - scenario_value if reverse else scenario_value - baseline_value

    def calculate_economic_metrics(cost_diff, deaths_averted, dalys_averted, wtp=80):
        """Calculate ICERs and INB metrics for a single run."""
        icer_deaths = cost_diff / deaths_averted if deaths_averted != 0 else float('inf')
        icer_dalys = cost_diff / dalys_averted if dalys_averted != 0 else float('inf')
        inhb = dalys_averted - (cost_diff / wtp)
        inmb = (dalys_averted * wtp) - cost_diff

        return {
            'icer_deaths': icer_deaths,
            'icer_dalys': icer_dalys,
            'inhb': inhb,
            'inmb': inmb
        }

    def get_scenario_differences(scenario_results, wtp=80):
        """Calculate differences and economic metrics for all scenarios relative to baseline."""
        base_scenarios = ['baseline_ant', 'existing_psa', 'planned_psa']
        po_levels = ['level2', 'level1b', 'level1a', 'level0']

        scenarios = (
            [f"{base}_with_po_{level}"
             for base in base_scenarios
             for level in po_levels] +
            base_scenarios[1:]  # Add non-PO scenarios
        )

        results = {}
        # First calculate primary metrics for each run
        for scenario in scenarios:
            scenario_results_across_runs = {
                'cost_difference': [],
                'deaths_averted': [],
                'dalys_averted': []
            }

            for run_results in scenario_results:
                # Calculate primary metrics
                cost_diff = calculate_metric_difference(scenario, run_results, 'total_costs')
                deaths_averted = calculate_metric_difference(scenario, run_results, 'deaths', reverse=True)
                dalys_averted = calculate_metric_difference(scenario, run_results, 'DALYs_discounted', reverse=True)

                # Store primary metrics
                scenario_results_across_runs['cost_difference'].append(cost_diff)
                scenario_results_across_runs['deaths_averted'].append(deaths_averted)
                scenario_results_across_runs['dalys_averted'].append(dalys_averted)

                # Calculate and store economic metrics
                economic_metrics = calculate_economic_metrics(cost_diff, deaths_averted, dalys_averted, wtp)
                for metric_name, value in economic_metrics.items():
                    if metric_name not in scenario_results_across_runs:
                        scenario_results_across_runs[metric_name] = []
                    scenario_results_across_runs[metric_name].append(value)

            # Store all results for this scenario
            for metric, values in scenario_results_across_runs.items():
                results[f"{metric}_{scenario}"] = values

        return results

    def calculate_statistics(differences):
        """Calculate mean and 95% CI using percentiles."""
        mean = np.mean(differences)
        ci = np.percentile(differences, [2.5, 97.5])

        return {
            'mean': mean,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }

    def get_scenario_statistics(all_differences):
        """Calculate statistics for all metrics across all scenarios."""
        return {
            scenario: calculate_statistics(differences)
            for scenario, differences in all_differences.items()
        }

    # Usage
    all_differences = get_scenario_differences(scenario_results=all_scenarios_results)
    scenario_statistics = get_scenario_statistics(all_differences)

    # Print results by metric type
    metric_types = ['cost_difference', 'deaths_averted', 'dalys_averted', 'icer_deaths', 'icer_dalys', 'inhb', 'inmb']

    for metric_type in metric_types:
        print(f"\n=== {metric_type.upper()} ===")
        relevant_stats = {k: v for k, v in scenario_statistics.items() if k.startswith(metric_type)}
        for scenario, stats in relevant_stats.items():
            print(f"\n{scenario.replace(metric_type + '_', '')}:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  95% CI: ({stats['ci_lower']:.2f}, {stats['ci_upper']:.2f})")


    def get_frontier_points(scenario_statistics):
        """Get points for cost-effectiveness frontier."""
        # Extract mean costs and effects for each scenario
        # Add origin point (baseline)
        points = [{'scenario': 'baseline_ant', 'cost': 0, 'effect': 0}]

        # Extract mean costs and effects for each scenario
        for key in scenario_statistics:
            if key.startswith('cost_difference_'):
                scenario = key.replace('cost_difference_', '')
                cost_stats = scenario_statistics[f'cost_difference_{scenario}']
                effect_stats = scenario_statistics[f'dalys_averted_{scenario}']
                points.append({
                    'scenario': scenario,
                    'cost': cost_stats['mean'],
                    'effect': effect_stats['mean']
                })

        # Sort by effects
        points = sorted(points, key=lambda x: x['effect'])

        # Find frontier points (non-dominated strategies)
        # Start with baseline point (0,0)
        frontier = [points[0]]

        for i in range(1, len(points)):
            current_point = points[i]

            while len(frontier) >= 2:
                # Calculate slopes between last three points
                previous = frontier[-1]
                second_previous = frontier[-2]

                # Calculate ICERs
                slope_last = (current_point['cost'] - previous['cost']) / (current_point['effect'] - previous['effect'])
                slope_second_last = (previous['cost'] - second_previous['cost']) / (previous['effect'] - second_previous['effect'])

                # If current point creates a more efficient frontier, remove the last point
                if slope_last < slope_second_last:
                    frontier.pop()
                else:
                    break

            # Add point to frontier
            frontier.append(current_point)

        # Filter out points that are dominated (higher cost, same or lower effect)
        final_frontier = frontier
        # for i in range(1, len(frontier)):
        #     if (frontier[i]['cost'] / frontier[i]['effect']) > (frontier[i-1]['cost'] / frontier[i-1]['effect']):
        #         final_frontier.append(frontier[i])

        return final_frontier


    def plot_ce_plane(scenario_statistics):
        """
        Create cost-effectiveness plane plot with confidence intervals.

        Parameters:
        - all_differences: dictionary of differences from get_scenario_differences()
        - wtp: willingness to pay threshold (for reference line)
        """
        base_scenarios = ['baseline_ant', 'existing_psa', 'planned_psa']
        po_levels = ['level2', 'level1b', 'level1a', 'level0']

        # Get unique scenarios
        scenarios = [k.split('cost_difference_')[1] for k in scenario_statistics.keys()
                     if k.startswith('cost_difference_')]

        # Set up the plot
        plt.figure(figsize=(12, 8))

        # Find max y value (including CI upper bounds)
        max_cost = max(scenario_statistics[f'cost_difference_{s}']['ci_upper'] for s in scenarios)
        # Find max and min x values for DALYs
        max_dalys = max(scenario_statistics[f'dalys_averted_{s}']['ci_upper'] for s in scenarios)
        min_dalys = min(scenario_statistics[f'dalys_averted_{s}']['ci_lower'] for s in scenarios)

        # Plot reference lines
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        # Colors for different scenarios

        # Define base colors for each scenario
        BASE_COLOURS = {
            'baseline_ant': '#1f77b4',    # blue
            'existing_psa': '#ff7f0e',    # orange
            'planned_psa': '#2ca02c'      # green
        }

        # Use lightness adjustments
        def create_lightness_variants(base_colour):
            # Convert hex to rgb
            rgb = np.array(mcolors.hex2color(base_colour))

            return {
                'base': base_colour,
                'level2': mcolors.to_hex(np.minimum(rgb * 1.2, 1.0)),
                'level1b': mcolors.to_hex(np.minimum(rgb * 1.4, 1.0)),
                'level1a': mcolors.to_hex(np.minimum(rgb * 1.6, 1.0)),
                'level0': mcolors.to_hex(np.minimum(rgb * 1.8, 1.0)),
            }

        lightness_colours = {
            scenario: create_lightness_variants(colour)
            for scenario, colour in BASE_COLOURS.items()
        }

        colours = lightness_colours

        # # Create color mapping for base scenarios
        # base_colors = plt.cm.tab10(np.linspace(0, 1, len(base_scenarios)))
        # color_map = dict(zip(base_scenarios, base_colors))

        # Define markers by PO implementation level
        po_level_markers = {
            'base': 'o',  # circle
            'level2': 's',   # square
            'level1b': 'p',  # pentagon
            'level1a': '*',  # star
            'level0': '^'   # triangle
        }

        # Plot each scenario
        for scenario in scenarios:
            # Get pre-calculated statistics
            cost_stats = scenario_statistics[f'cost_difference_{scenario}']
            daly_stats = scenario_statistics[f'dalys_averted_{scenario}']

            # Determine base scenario and PO level
            base_scenario = scenario.split('_with_')[0] if '_with_' in scenario else scenario

            if '_with_' in scenario:
                # Determine PO level
                po_level = next((po for po in po_levels if po in scenario), 'base')
                colour = colours[base_scenario][po_level]
                marker = po_level_markers[po_level]
            else:
                # Base scenario without PO level
                colour = colours[base_scenario]['base']
                marker = po_level_markers['base']

            # Plot point (mean values)
            plt.scatter(daly_stats['mean'], cost_stats['mean'],
                        color=colour, marker=marker,
                        label=scenario)

            # For error bars, use same color with slight transparency
            if isinstance(colour, str):  # If hex color
                error_color = mcolors.to_rgba(colour, alpha=0.3)
            else:  # If already rgba
                error_color = (*colour[:3], 0.3)

            # Plot confidence intervals using pre-calculated CIs
            plt.errorbar(daly_stats['mean'], cost_stats['mean'],
                         yerr=[[cost_stats['mean'] - cost_stats['ci_lower']],
                               [cost_stats['ci_upper'] - cost_stats['mean']]],
                         xerr=[[daly_stats['mean'] - daly_stats['ci_lower']],
                               [daly_stats['ci_upper'] - daly_stats['mean']]],
                         color=error_color, capsize=5)

        # Get and plot frontier
        frontier_points = get_frontier_points(scenario_statistics)
        frontier_x = [p['effect'] for p in frontier_points]
        frontier_y = [p['cost'] for p in frontier_points]
        plt.plot(frontier_x, frontier_y, 'k--', alpha=0.3, label='Cost-effectiveness frontier')

        # Set y-axis limits to match max cost
        plt.ylim(-max_cost * 0.1, max_cost * 1.1)  # Add 10% padding

        # Customize plot
        plt.xlabel('DALYs Averted')
        plt.ylabel('Incremental Cost ($)')
        plt.title('Cost-Effectiveness Plane')

        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust layout to prevent legend cutoff
        plt.tight_layout()

        return plt.gcf()

    # Apply
    fig = plot_ce_plane(scenario_statistics)
    plt.show()
#
# # Optionally save the figure
# # fig.savefig('ce_plane.png', dpi=300, bbox_inches='tight')
