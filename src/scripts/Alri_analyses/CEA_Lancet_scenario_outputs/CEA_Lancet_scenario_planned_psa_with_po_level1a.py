""" This script will output the scenario outcomes for planned psa with pulse oximetry implemented at level 1a """

import random
from pathlib import Path
import os
from typing import List
import datetime
from math import e
from openpyxl import Workbook
from openpyxl import load_workbook
import scipy.stats as stats
import pickle

# from tlo.util import random_date, sample_outcome
import numpy.random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
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

dx_accuracy = 'perfect'
sa_name = ''

# False if main analysis, True if a sensitivity analysis (HW Dx Accuracy, or Prioritise oxygen in Hospitals)
sensitivity_analysis = False

sensitivity_analysis_hw_dx = True  # change to False if analysing Oxygen prioritisation

# Helper function for conversion between odds and probabilities
to_odds = lambda pr: pr / (1.0 - pr)  # noqa: E731tab e402c

to_prob = lambda odds: odds / (1.0 + odds)  # noqa: E731

# Date for saving the image for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# # # Set the scenario
scenario = 'planned_psa_with_po_level1a'

# Pulse oximetry configuration
oximeter_available = True if 'with_po' in scenario else False
implementation_level = ('2') if 'with_po_level2' in scenario else \
    ('2', '1b') if 'with_po_level1b' in scenario else ('2', '1b', '1a') if 'with_po_level1a' in scenario else \
        ('2', '1b', '1a', '0') if 'with_po_level0' in scenario else 'none'


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
alri_module_with_imperfect_diagnosis_50 = sim0.modules['Alri']
_reduce_hw_dx_sensitivity(alri_module_with_imperfect_diagnosis_50)
hsi_with_imperfect_diagnosis_50 = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_50, person_id=None)

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
                hsi = hsi_with_imperfect_diagnosis_50
                alri_module = alri_module_with_imperfect_diagnosis_50
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


def generate_table():
    """Return table providing a representative case mix of persons with Alri, the intrinsic risk of death and the
    efficacy of treatment under different conditions."""

    # Get Case Mix
    df = generate_case_mix()
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
    # generate care management outcomes for the case mix
    table = generate_table()

    # Get the population metrics

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
    # use only YLL for DALYs estimation
    DALYs_discounted = ((total_deaths*(1-np.exp(-0.03*(54.7 - mean_age))))/0.03)

    # # # # Add the costs to the dataframe for later analyses # # # #
    low_oxygen = (table["oxygen_saturation"])
    seek_facility = (table["seek_level"])
    classification_by_seek_level = table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx']
    final_facility = table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx']
    final_facility_follow_up = table[f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx']

    number_cases_by_seek_level = table.groupby(
        by=[classification_by_seek_level, low_oxygen, final_facility]).size()

    final_facility = table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx']

    full_oxygen_provided = table.loc[((final_facility == '2') | (final_facility == '1b')),
                                     f'oxygen_provided_scenario_{scenario}_{dx_accuracy}_hw_dx'].sum()
    # add follow-up cases from failure oral treatment
    follow_up_full_oxygen_provided = table.loc[((final_facility_follow_up == '2') | (final_facility_follow_up == '1b')),
                                               f'oxygen_provided_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].sum()

    need_oxygen_ds_with_SpO2lt90 = number_cases_by_seek_level.sum(level=[0, 1]).reindex(
        pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%')]))

    # ------ mortality outcome per individual -----------
    table['mortality_outcome'] = table.loc[
        ((table[f'treatment_efficacy_scenario_{scenario}_{dx_accuracy}_hw_dx']) &
         (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'] == False)) |
        ((table[f'treatment_efficacy_scenario_{scenario}_{dx_accuracy}_hw_dx']) &
         (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
         (table[f'follow_up_treatment_failure_scenario_{scenario}_{dx_accuracy}_hw_dx'] == True)),
        'will_die']
    table['mortality_outcome'] = table['mortality_outcome'].fillna(False)  # fill NaN with False

    table['DALYs_discounted'] = ((table['mortality_outcome']*(1-np.exp(-0.03*(54.7 - table['age_exact_years']))))/0.03)

    # ------ cost outcome per individual -----------

    # ANTIBIOTICS COST --------------
    # oral amoxicillin
    table['oral_amox_cost_by_age'] = table['age_exact_years'].apply(
        lambda x: (1 * 2 * 5 * 0.02734) if x < 1 else (2 * 2 * 5 * 0.02734) if 1 <= x < 2 else
        (3 * 2 * 5 * 0.02734))
    table['oral_amox_cost'] = table.loc[
        ((table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] == 'fast_breathing_pneumonia') |
         (table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] == 'chest_indrawing_pneumonia')),
        'oral_amox_cost_by_age']
    table['oral_amox_cost'] = table['oral_amox_cost'].fillna(0.0)

    # for classified as ds pneumonia at level 1a, or 0 needing referral but not referred - give oral amox
    table['non_referred_oral_amox_cost'] = table.loc[
        (table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] == 'danger_signs_pneumonia') &
        ((table['seek_level'] == '1a') | (table['seek_level'] == '0')) &
        (table['seek_level'] == table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx']),
        'oral_amox_cost_by_age']
    table['non_referred_oral_amox_cost'] = table['non_referred_oral_amox_cost'].fillna(0.0)

    # Oral antibiotic cost at follow-up care
    table['follow_up_oral_amox_cost'] = table.loc[
        (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
        (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'chest_indrawing_pneumonia'),
        'oral_amox_cost_by_age']
    table['follow_up_oral_amox_cost'] = table['follow_up_oral_amox_cost'].fillna(0.0)

    # for classified as ds pneumonia at level 1a, or 0 needing referral but not referred - give oral amox
    table['follow_up_non_referred_oral_amox_cost'] = table.loc[
        (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
        (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'danger_signs_pneumonia') &
        ((final_facility == '1a') | (final_facility == '0')) &
        (final_facility == final_facility_follow_up), 'oral_amox_cost_by_age']
    table['follow_up_non_referred_oral_amox_cost'] = table['follow_up_non_referred_oral_amox_cost'].fillna(0.0)

    table['all_oral_amox_cost'] = table['oral_amox_cost'] + table['non_referred_oral_amox_cost'] + \
                                  table['follow_up_oral_amox_cost'] + table['follow_up_non_referred_oral_amox_cost']
    # table = table.drop('oral_amox_cost_by_age', axis=1)
    # ----------------

    # IV antibiotics - 1st line - amp + gent , 2nd line - ceftriaxone
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

    # first line IV
    table['iv_first_line_cost'] = table[[f'classification_in_{scenario}_{dx_accuracy}_hw_dx',
                                         f'first_line_iv_failed_scenario_{scenario}_{dx_accuracy}_hw_dx',
                                         'ampicillin_unit_cost',
                                         'gentamycin_unit_cost',
                                         f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx']].apply(
        lambda x: (x[2] * 4 * 5) + (x[3] * 5) if ((x[0] == 'danger_signs_pneumonia') and (x[1] == False) and
                                                  (x[4] == '2' or x[4] == '1b'))
        else (x[2] * 4 * 2) + (x[3] * 2) if ((x[0] == 'danger_signs_pneumonia') and x[1] == True) else 0,
        axis=1)
    # second line IV
    table['iv_second_line_cost'] = table[[f'classification_in_{scenario}_{dx_accuracy}_hw_dx',
                                          f'first_line_iv_failed_scenario_{scenario}_{dx_accuracy}_hw_dx',
                                          'ceftriaxone_unit_cost',
                                          f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx']].apply(
        lambda x: (x[2] * 5) if ((x[0] == 'danger_signs_pneumonia') and (x[1] == True) and
                                 (x[3] == '2' or x[3] == '1b')) else 0, axis=1)
    # total IV antibitoic cost
    table['iv_antibiotics_cost'] = table['iv_first_line_cost'] + table['iv_second_line_cost']

    # IV antibiotic cost at follow-up care
    table['follow_up_iv_first_line_cost'] = table[
        [f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx',
         f'first_line_iv_failed_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx',
         'ampicillin_unit_cost', 'gentamycin_unit_cost',
         f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx']].apply(
        lambda x: (x[2] * 4 * 5) + (x[3] * 5) if ((x[0] != False) and (x[0][1] == 'danger_signs_pneumonia') and
                                                  (x[1] == False) and (x[4] == '2' or x[4] == '1b'))
        else (x[2] * 4 * 2) + (x[3] * 2) if ((x[0] != False) and (x[0][1] == 'danger_signs_pneumonia') and
                                             (x[1] == True)) else 0, axis=1)

    table['follow_up_iv_second_line_cost'] = table[
        [f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx',
         f'first_line_iv_failed_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx',
         'ceftriaxone_unit_cost',
         f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx']].apply(
        lambda x: (x[2] * 5) if ((x[0] != False) and (x[0][1] == 'danger_signs_pneumonia') and
                                 (x[1] == True) and (x[3] == '2' or x[3] == '1b')) else 0, axis=1)

    # IV antibitoic cost at follow-up
    table['follow_up_iv_antibiotics_cost'] = \
        table['follow_up_iv_first_line_cost'] + table['follow_up_iv_second_line_cost']

    # sum of IV antibiotics in initial visit and follow-up
    table['all_iv_antibiotics_cost'] = table['iv_antibiotics_cost'] + table['follow_up_iv_antibiotics_cost']
    # table = table.drop('ampicillin_unit_cost', axis=1)
    # table = table.drop('gentamycin_unit_cost', axis=1)
    # table = table.drop('ceftriaxone_unit_cost', axis=1)

    # OUTPATIENT CONSULTATION COST ------------
    table['consultation_cost_seek_level'] = table['seek_level'].apply(
        lambda x: 2.58 if x == '2' else 2.47 if x == '1b' else 2.17 if x == '1a' else 1.76
    )
    table['consultation_cost_final_facility'] = table[
        f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx'].apply(
        lambda x: 2.58 if x == '2' else 2.47 if x == '1b' else 2.17 if x == '1a' else 1.76
    )

    # consultation cost of all outpatient visits - any level and non-sev classification
    table['outpatient_consultation_cost'] = table.loc[
        (table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] != 'danger_signs_pneumonia'),
        'consultation_cost_seek_level']
    table['outpatient_consultation_cost'] = table['outpatient_consultation_cost'].fillna(0.0)

    # danger signs pneumonia initial consultation cost at lower levels before referral
    # consultation cost of ds pneumonia at lower levels of the HS before referral up
    table['consultation_for_referral_cost'] = table.loc[
        (table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] == 'danger_signs_pneumonia') &
        (table['seek_level'] != table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx']),
        'consultation_cost_seek_level']
    table['consultation_for_referral_cost'] = table['consultation_for_referral_cost'].fillna(0.0)

    # for classified as ds pneumonia at level 1a, or 0 needing referral but not referred
    table['consultation_for_referral_not_referred_cost'] = table.loc[
        (table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] == 'danger_signs_pneumonia') &
        ((table['seek_level'] == '1a') | (table['seek_level'] == '0')) &
        (table['seek_level'] == table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx']),
        'consultation_cost_seek_level']
    table['consultation_for_referral_not_referred_cost'] = \
        table['consultation_for_referral_not_referred_cost'].fillna(0.0)

    # follow-up consultation costs --------
    # consultation cost of all follow-up outpatient visits - any level and non-sev classification
    table['follow_up_outpatient_consultation_cost'] = table.loc[
        (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
        (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'chest_indrawing_pneumonia'),
        'consultation_cost_final_facility']
    table['follow_up_outpatient_consultation_cost'] = table['follow_up_outpatient_consultation_cost'].fillna(0.0)

    # danger signs pneumonia initial consultation cost at lower levels before referral
    # follow-up consultation cost of ds pneumonia at lower levels of the HS before referral up
    table['follow_up_consultation_for_referral_cost'] = table.loc[
        (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
        (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'danger_signs_pneumonia') &
        (final_facility != final_facility_follow_up),
        'consultation_cost_final_facility']
    table['follow_up_consultation_for_referral_cost'] = table['follow_up_consultation_for_referral_cost'].fillna(0.0)

    # for classified as ds pneumonia at level 1a, or 0 needing referral but not referred
    table['follow_up_consultation_for_referral_not_referred_cost'] = table.loc[
        (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
        (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'danger_signs_pneumonia') &
        ((final_facility == '1a') | (final_facility == '0')) &
        (final_facility == final_facility_follow_up), 'consultation_cost_final_facility']
    table['follow_up_consultation_for_referral_not_referred_cost'] = \
        table['follow_up_consultation_for_referral_not_referred_cost'].fillna(0.0)

    # total consultation costs - initial visit and follow-up
    table['all_outpatient_consultation_cost'] = table[
        ['outpatient_consultation_cost', 'consultation_for_referral_cost',
         'consultation_for_referral_not_referred_cost', 'follow_up_outpatient_consultation_cost',
         'follow_up_consultation_for_referral_cost',
         'follow_up_consultation_for_referral_not_referred_cost']].sum(axis=1)
    # table = table.drop(['consultation_cost_seek_level', 'consultation_cost_final_facility',
    #                     'outpatient_consultation_cost', 'consultation_for_referral_cost',
    #                     'consultation_for_referral_not_referred_cost', 'follow_up_outpatient_consultation_cost',
    #                     'follow_up_consultation_for_referral_cost',
    #                     'follow_up_consultation_for_referral_not_referred_cost'], axis=1)
    # ----------------

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

    table['all_inpatient_bed_cost'] = table[['inpatient_bed_cost', 'inpatient_bed_cost_follow_up']].sum(axis=1)
    # table = table.drop(['inpatient_bed_cost_per_day', 'inpatient_bed_cost_per_day_fu'], axis=1)
    # -----------------------

    # PULSE OXIMETRY COST ----------
    po_cost_by_usage_rate = {'perfect': [0.167, 0.096, 0.063], 'imperfect': [0.167, 0.096, 0.063]}

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

    # PO availability
    table['po_available_referral_level'] = table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx'].apply(
        lambda x: True if oximeter_available and x[0] in implementation_level else False)
    table['po_available_referral_level_at_follow_up'] = final_facility_follow_up.apply(
        lambda x: True if oximeter_available and x is not None and x in implementation_level else False)

    # PO cost by use
    table['po_cost_sought_level'] = table.loc[(table[f'po_available_seek_level_{scenario}_hw_dx']) &
                                              (seek_facility == final_facility), 'PO_cost_seek_level']
    table['po_cost_sought_level'] = table['po_cost_sought_level'].fillna(0.0)

    table['po_cost_referred_level'] = table.loc[(table['po_available_referral_level']) &
                                                (seek_facility != final_facility), 'PO_cost_final_level']
    table['po_cost_referred_level'] = table['po_cost_referred_level'].fillna(0.0)

    table['po_cost_referred_level_at_follow_up'] = table.loc[
        (table['po_available_referral_level_at_follow_up']) &
        (final_facility != final_facility_follow_up), 'PO_cost_final_level_follow_up']
    table['po_cost_referred_level_at_follow_up'] = table['po_cost_referred_level_at_follow_up'].fillna(0.0)


    # total PO costs
    table['all_po_cost'] = table[['po_cost_sought_level', 'po_cost_referred_level',
                                  'po_cost_referred_level_at_follow_up']].sum(axis=1)
    # table = table.drop(['PO_cost_seek_level', 'PO_cost_final_level', 'PO_cost_final_level_follow_up',
    #                     'po_cost_sought_level', 'po_cost_referred_level',
    #                     'po_cost_referred_level_at_follow_up'], axis=1)
    # ----------------

    # OXYGEN COST ----------------

    # get total oxygen liters ------------
    table['total_liters_full'] = table['age_exact_years'].apply(
        lambda x: (1 * 60 * 24 * 3) if x < 1/6 else (2 * 60 * 24 * 3))

    table['total_liters_3day_therapy'] = table.loc[
        ((final_facility == '2') | (final_facility == '1b')) &
        (table[f'oxygen_provided_scenario_{scenario}_{dx_accuracy}_hw_dx']), 'total_liters_full']
    table['follow_up_total_liters_3day_therapy'] = table.loc[
        (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
        (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'danger_signs_pneumonia') &
        ((final_facility_follow_up == '2') | (final_facility_follow_up == '1b')) &
        (table[f'oxygen_provided_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx']), 'total_liters_full']

    # stabilisation oxygen before referral
    table['total_liters_stab'] = table['age_exact_years'].apply(
        lambda x: (1 * 60 * 6) if x < 1/6 else(2 * 60 * 6))

    table['liters_for_stabilisation'] = table.loc[
        (table[f'referral_status_and_oxygen_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] != 'no referral needed') &
        (table[f'referral_status_and_oxygen_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'provided'),
        'total_liters_stab']
    table['follow_up_liters_for_stabilisation'] = table.loc[
        (table[f'referral_status_and_oxygen_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] != 'no referral needed') &
        (table[f'referral_status_and_oxygen_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'provided'),
        'total_liters_stab']

    # total of 3 day therapy, stabilisation at first visit and follow-up
    table['all_total_oxygen_liters_used'] = table[['total_liters_3day_therapy',
                                                   'follow_up_total_liters_3day_therapy', 'liters_for_stabilisation',
                                                   'follow_up_liters_for_stabilisation']].sum(axis=1)
    # table = table.drop('total_liters_full', 'total_liters_3day_therapy',
    #                    'follow_up_total_liters_3day_therapy', 'total_liters_stab', 'liters_for_stabilisation',
    #                    'follow_up_liters_for_stabilisation'], axis=1)
    # ----------------

    # get total oxygen liters in perfect system ------------
    table['total_liters_3day_therapy_perfect'] = table.loc[
        ((table[f'final_facility_scenario_all_district_psa_with_po_level0_perfect_hw_dx'] == '2') |
         (table[f'final_facility_scenario_all_district_psa_with_po_level0_perfect_hw_dx'] == '1b')) &
        (table[f'oxygen_provided_scenario_all_district_psa_with_po_level0_perfect_hw_dx']),
        'total_liters_full']
    table['follow_up_total_liters_3day_therapy_perfect'] = table.loc[
        (table[f'eligible_for_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx'].str[0] == True) &
        (table[f'eligible_for_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx'].str[1] == 'danger_signs_pneumonia') &
        ((table[f'final_facility_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx'] == '2') |
         (table[f'final_facility_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx'] == '1b')) &
        (table[f'oxygen_provided_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx']),
        'total_liters_full']

    # stabilisation oxygen
    table['liters_for_stabilisation_perfect'] = table.loc[
        (table[f'referral_status_and_oxygen_scenario_all_district_psa_with_po_level0_perfect_hw_dx'].str[0] != 'no referral needed') &
        (table[f'referral_status_and_oxygen_scenario_all_district_psa_with_po_level0_perfect_hw_dx'].str[1] == 'provided'),
        'total_liters_stab']
    table['follow_up_liters_for_stabilisation_perfect'] = table.loc[
        (table[f'referral_status_and_oxygen_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx'].str[0] != 'no referral needed') &
        (table[f'referral_status_and_oxygen_follow_up_scenario_all_district_psa_with_po_level0_perfect_hw_dx'].str[1] == 'provided'),
        'total_liters_stab']

    # total of 3 day therapy, stabilisation at first visit and follow-up
    table['all_total_oxygen_liters_used_perfect'] = table[['total_liters_3day_therapy_perfect',
                                                           'follow_up_total_liters_3day_therapy_perfect', 'liters_for_stabilisation_perfect',
                                                           'follow_up_liters_for_stabilisation_perfect']].sum(axis=1)
    # ---------------

    # OXYGEN COST ----------------
    # oxygen_scenario = 'existing_psa' if scenario.startswith('existing') else \
    #     'planned_psa' if scenario.startswith('planned') else \
    #         'all_district_psa' if scenario.startswith('all_district') else 'baseline_ant'
    total_ox_liters_for_alri = table['all_total_oxygen_liters_used_perfect'].sum()

    overall_oxygen_cost = ((3051794 * (total_ox_liters_for_alri/1401065280)) if scenario.startswith('existing_psa') else
                           (5884936 * (total_ox_liters_for_alri/1401065280)) if scenario.startswith('planned_psa') else
                           (8048553 * (total_ox_liters_for_alri/1401065280)) if scenario.startswith('all_district_psa') else
                           0.0 if scenario.startswith('baseline_ant') else None)
    # overall_oxygen_cost = ((3051794 / 2) if scenario.startswith('existing_psa') else
    #                        (5884936 / 2) if scenario.startswith('planned_psa') else
    #                        (8048553 / 2) if scenario.startswith('all_district_psa') else
    #                        0.0 if scenario.startswith('baseline_ant') else None)

    # oxygen unit cost changes with PO vs without
    oxygen_liter_unit_cost = (overall_oxygen_cost / table['all_total_oxygen_liters_used'].sum()) if \
        table['all_total_oxygen_liters_used'].sum() > 0.0 else 0.0

    # calculate the treatment cost of oxygen for this scenario in analysis
    table['oxygen_cost_3day_therapy_by_age'] = table['age_exact_years'].apply(
        lambda x: (oxygen_liter_unit_cost * 1 * 60 * 24 * 3) if x < 1/6 else
        (oxygen_liter_unit_cost * 2 * 60 * 24 * 3))
    # stabilization oxygen cost
    table['oxygen_cost_for_stabilisation_by_age'] = table['age_exact_years'].apply(
        lambda x: (oxygen_liter_unit_cost * 1 * 60 * 6) if x < 1/6 else
        (oxygen_liter_unit_cost * 2 * 60 * 6))

    # full treatment with oxygen (3 day)
    table[f'oxygen_cost_3day_therapy'] = table.loc[
        ((final_facility == '2') | (final_facility == '1b')) &
        (table[f'oxygen_provided_scenario_{scenario}_{dx_accuracy}_hw_dx']),
        'oxygen_cost_3day_therapy_by_age']
    table[f'oxygen_cost_3day_therapy'] = table[f'oxygen_cost_3day_therapy'].fillna(0.0)

    # cost the stabilisation oxygen at level 1a regardless or their referral completion
    table[f'oxygen_cost_for_stabilisation'] = table.loc[
        (table[f'referral_status_and_oxygen_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] != 'no referral needed') &
        (table[f'referral_status_and_oxygen_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'provided'),
        'oxygen_cost_for_stabilisation_by_age']
    table[f'oxygen_cost_for_stabilisation'] = table[f'oxygen_cost_for_stabilisation'].fillna(0.0)

    # oxygen cost at follow_up
    table['follow_up_oxygen_cost_3day_therapy'] = table.loc[
        (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
        (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'danger_signs_pneumonia') &
        ((final_facility_follow_up == '2') | (final_facility_follow_up == '1b')) &
        (table[f'oxygen_provided_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx']),
        'oxygen_cost_3day_therapy_by_age']
    table['follow_up_oxygen_cost_3day_therapy'] = table['follow_up_oxygen_cost_3day_therapy'].fillna(0.0)

    # cost the stabilisation oxygen at level 1a regardless or their referral completion
    table['follow_up_oxygen_cost_for_stabilisation'] = table.loc[
        (table[f'referral_status_and_oxygen_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] != 'no referral needed') &
        (table[f'referral_status_and_oxygen_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'provided'),
        'oxygen_cost_for_stabilisation_by_age']
    table['follow_up_oxygen_cost_for_stabilisation'] = table['follow_up_oxygen_cost_for_stabilisation'].fillna(0.0)

    # all oxygen costs
    table['all_oxygen_cost'] = \
        table['oxygen_cost_3day_therapy'] + table['oxygen_cost_for_stabilisation'] + \
        table['follow_up_oxygen_cost_3day_therapy'] + table['follow_up_oxygen_cost_for_stabilisation']
    # table = table.drop(['oxygen_cost_3day_therapy_by_age', 'oxygen_cost_for_stabilisation_by_age',
    #                     'oxygen_cost_3day_therapy', 'oxygen_cost_for_stabilisation',
    #                     'follow_up_oxygen_cost_3day_therapy', 'follow_up_oxygen_cost_for_stabilisation'],
    #                    axis=1)

    # total costs combined
    table['total_costs'] = table[['all_oral_amox_cost', 'all_iv_antibiotics_cost',
                                  'all_outpatient_consultation_cost', 'all_inpatient_bed_cost',
                                  'all_po_cost', 'all_oxygen_cost']].sum(axis=1)

    # --------------------------------------------------------------------------------

    # Save the table output
    with open(f'debug_output_{scenario}_{dx_accuracy}_{sa_name}.pkl', 'wb') as f:
        pickle.dump(table, f)

    # --------------------------------------------------------------------------------

    # # Later, load it back
    # with open(f'debug_output_{scenario}_{dx_accuracy}_{sa_name}.pkl', 'rb') as f:
    #     output_baseline_ant = pickle.load(f)
    #     table = output_baseline_ant

    # Get the total costs for the cohort
    total_costs = list()

    total_costs.append({
        # Total IV antibiotic cost
        f'sum_iv_ant_cost_{scenario}': table['all_iv_antibiotics_cost'].sum(),
        # Total oral antibiotic cost
        f'sum_oral_ant_cost_{scenario}': table['all_oral_amox_cost'].sum(),
        # Total outpatient consultation cost
        f'sum_consultation_cost_{scenario}': table['all_outpatient_consultation_cost'].sum(),
        # Total hospitalisation cost
        f'sum_hospitalisation_cost_{scenario}': table['all_inpatient_bed_cost'].sum(),
        # Total PO cost
        f'sum_po_cost_{scenario}': table['all_po_cost'].sum(),
        # Total oxygen cost
        f'sum_oxygen_cost_{scenario}': table['all_oxygen_cost'].sum()
    })

    costs_df = pd.DataFrame(total_costs)

    deaths = table['mortality_outcome'].sum()
    costs = costs_df
    total_cost = costs_df.sum(axis=1)
    oxygen_provided = full_oxygen_provided + follow_up_full_oxygen_provided
    need_oxygen = need_oxygen_ds_with_SpO2lt90
    oxygen_liters_provided = table['all_total_oxygen_liters_used'].sum()

    total_costs = costs[f'sum_oral_ant_cost_{scenario}'][0] + costs[f'sum_consultation_cost_{scenario}'][0] + \
                  costs[f'sum_iv_ant_cost_{scenario}'][0] + costs[f'sum_hospitalisation_cost_{scenario}'][0] + \
                  costs[f'sum_po_cost_{scenario}'][0] + \
                  ((3051794 * (total_ox_liters_for_alri/1401065280)) if scenario.startswith('existing_psa') else
                   (5884936 * (total_ox_liters_for_alri/1401065280)) if scenario.startswith('planned_psa') else
                   (8048553 * (total_ox_liters_for_alri/1401065280)) if scenario.startswith('all_district_psa') else
                   0.0 if scenario.startswith('baseline_ant') else 0.0)

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
        'cost of Oxygen': (3051794 * (total_ox_liters_for_alri/1401065280)) if scenario.startswith('existing_psa') else
        (5884936 * (total_ox_liters_for_alri/1401065280)) if scenario.startswith('planned_psa') else
        (8048553 * (total_ox_liters_for_alri/1401065280)) if scenario.startswith('all_district_psa') else
        0.0 if scenario.startswith('baseline_ant') else None,
        'total_costs': total_costs,
        'need_oxygen': need_oxygen[0],
        'oxygen_provided': oxygen_provided,
        'oxygen_liters_provided': oxygen_liters_provided,
        'total_ox_liters_for_alri': total_ox_liters_for_alri,
        'age <1': age_lt1,
        'age 1-2': age_1to2,
        'age 2-5': age_2to5,
        'Enterobacteriaceae': pathogen_distributon['Enterobacteriaceae'],
        'H.influenzae_non_type_b': pathogen_distributon['H.influenzae_non_type_b'],
        'HMPV': pathogen_distributon['HMPV'],
        'Hib': pathogen_distributon['Hib'],
        'Influenza': pathogen_distributon['Influenza'],
        'P.jirovecii': pathogen_distributon['P.jirovecii'],
        'Parainfluenza': pathogen_distributon['Parainfluenza'],
        'RSV': pathogen_distributon['RSV'],
        'Rhinovirus': pathogen_distributon['Rhinovirus'],
        'Staph_aureus': pathogen_distributon['Staph_aureus'],
        'Strep_pneumoniae_PCV13': pathogen_distributon['Strep_pneumoniae_PCV13'],
        'Strep_pneumoniae_non_PCV13': pathogen_distributon['Strep_pneumoniae_non_PCV13'],
        'other_Strepto_Enterococci': pathogen_distributon['other_Strepto_Enterococci'],
        'other_bacterial_pathogens': pathogen_distributon['other_bacterial_pathogens'],
        'other_pathogens_NoS': pathogen_distributon['other_pathogens_NoS'],
        'other_viral_pathogens': pathogen_distributon['other_viral_pathogens'],
        'coinfection': coinfection,
        'disease_type_other_alri': disease_type_prop['other_alri'],
        'disease_type_pneumonia': disease_type_prop['pneumonia'],
        'mean_duration_in_days': mean_duration,
        'sd_duration': sd_duration,
        'care_seeking_level0': care_seeking_level['0'],
        'care_seeking_level1a': care_seeking_level['1a'],
        'care_seeking_level1b': care_seeking_level['1b'],
        'care_seeking_level2': care_seeking_level['2'],
        'hypoxaemia_90-92%': hypoxaemia['90-92%'],
        'hypoxaemia_<90%': hypoxaemia['<90%'],
        'hypoxaemia_>=93%': hypoxaemia['>=93%'],
        'any_pc_complications': pc_complications,
        'pleural_effusion': pleural_effusion,
        'empyema': empyema,
        'lung_abscess': lung_abscess,
        'pneumothorax': pneumothorax,
        'bacteraemia': bacteraemia,
        'all_complications': all_complications,
        'MAM': malnutrition['MAM'],
        'SAM': malnutrition['SAM'],
        'well_nourished': malnutrition['well'],
        'hiv_negative': hiv_pos[False],
        'hiv_positive': hiv_pos[True],
        'hiv_pos_not_on_art': hiv_pos_not_on_art[True],
        'will_die': will_die[True],
        'prob_will_die_no_treat': will_die[True] / will_die.sum(),
        'will_die_by_class_ci': will_die_by_class['chest_indrawing_pneumonia'],
        'will_die_by_class_cc': will_die_by_class['cough_or_cold'],
        'will_die_by_class_ds': will_die_by_class['danger_signs_pneumonia'],
        'will_die_by_class_fb': will_die_by_class['fast_breathing_pneumonia'],
        'will_die_by_ox_sat_90-92%': will_die_by_ox_sat['90-92%'],
        'will_die_by_ox_sat_<90%': will_die_by_ox_sat['<90%'],
        'will_die_by_ox_sat_>=93%': will_die_by_ox_sat['>=93%'],
        'prob_will_die_by_ox_sat_90-92%': will_die_by_ox_sat['90-92%'] / hypoxaemia['90-92%'],
        'prob_will_die_by_ox_sat_<90%': will_die_by_ox_sat['<90%'] / hypoxaemia['<90%'],
        'prob_will_die_by_ox_sat_>=93%': will_die_by_ox_sat['>=93%'] / hypoxaemia['>=93%'],
        'age IQR_25': iq1, 'age IQR_75': iq2,
    }

    debug_point = 0
