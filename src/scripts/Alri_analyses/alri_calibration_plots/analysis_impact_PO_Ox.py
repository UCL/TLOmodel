"""This script will generate a table that describes a representative mix of all the IncidentCases that are created, and
 the associated diagnosis and risk of death for each under various conditions of treatments/non-treatment."""
import random
from pathlib import Path
from typing import List
import datetime
from math import e

# from tlo.util import random_date, sample_outcome
import numpy.random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# import matplotlib.colors as mcolors
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
    _set_current_policy, _set_new_policy

)
from tlo.util import sample_outcome


MODEL_POPSIZE = 15_000
MIN_SAMPLE_OF_NEW_CASES = 200
NUM_REPS_FOR_EACH_CASE = 20

_facility_level = '2'  # <-- assumes that the diagnosis/treatment occurs at this level
scenario = 'baseline_ant'
# scenario = 'baseline_ant_with_po_level2'
# scenario = 'baseline_ant_with_po_level1b'
# scenario = 'baseline_ant_with_po_level1a'
# scenario = 'baseline_ant_with_po_level0'
# scenario = 'existing_psa'
# scenario = 'existing_psa_with_po_level2'
# scenario = 'existing_psa_with_po_level1b'
# scenario = 'existing_psa_with_po_level1a'
# scenario = 'existing_psa_with_po_level0'
# scenario = 'planned_psa'
# scenario = 'planned_psa_with_po_level2'
# scenario = 'planned_psa_with_po_level1b'
# scenario = 'planned_psa_with_po_level1a'
# scenario = 'planned_psa_with_po_level0'
# scenario = 'all_district_psa'
# scenario = 'all_district_psa_with_po_level2'
# scenario = 'all_district_psa_with_po_level1b'
# scenario = 'all_district_psa_with_po_level1a'
# scenario = 'all_district_psa_with_po_level0'

dx_accuracy = 'imperfect'

# False if main analysis, True if a sensitivity analysis (HW Dx Accuracy, or Prioritise oxygen in Hospitals)
sensitivity_analysis = False

sensitivity_analysis_hw_dx = True  # change to False if analysing Oxygen prioritisation

# oximeter_available = True if 'with_po' in scenario else False

# implementation_level = ('2') if 'with_po_level2' in scenario else ('2', '1b') if 'with_po_level1b' in scenario else \
#     ('2', '1b', '1a') if 'with_po_level1a' in scenario else ('2', '1b', '1a', '0') if 'with_po_level0' in scenario else \
#         'none'

# Helper function for conversion between odds and probabilities
to_odds = lambda pr: pr / (1.0 - pr)  # noqa: E731
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
sim0 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment = sim0.modules['Alri']
hsi_with_imperfect_diagnosis_and_imperfect_treatment = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment, person_id=None)

# Alri module with perfect diagnosis and perfect treatment
sim2 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_perfect_treatment_and_diagnosis = sim2.modules['Alri']
_make_treatment_and_diagnosis_perfect(alri_module_with_perfect_treatment_and_diagnosis)

# current policy - perfect diagnosis (and imperfect treatment)
sim_cp = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_perfect_diagnosis_current_policy = sim_cp.modules['Alri']
_make_hw_diagnosis_perfect(alri_module_with_perfect_diagnosis_current_policy)
_set_current_policy(alri_module_with_perfect_diagnosis_current_policy)
hsi_with_perfect_diagnosis_current_policy = HSI_Alri_Treatment(module=alri_module_with_perfect_diagnosis_current_policy,
                                                               person_id=None)

# current policy - perfect diagnosis and perfect treatment
sim_cp_perfect = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_perfect_diagnosis_and_perfect_treatment_current_policy = sim_cp_perfect.modules['Alri']
_make_treatment_and_diagnosis_perfect(alri_module_with_perfect_diagnosis_and_perfect_treatment_current_policy)
_set_current_policy(alri_module_with_perfect_diagnosis_and_perfect_treatment_current_policy)
hsi_with_perfect_diagnosis_and_perfect_treatment_current_policy = HSI_Alri_Treatment(
    module=alri_module_with_perfect_diagnosis_and_perfect_treatment_current_policy, person_id=None)

# new policy - perfect diagnosis (and imperfect treatment)
sim_np = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_perfect_diagnosis_new_policy = sim_np.modules['Alri']
_make_hw_diagnosis_perfect(alri_module_with_perfect_diagnosis_new_policy)
_set_new_policy(alri_module_with_perfect_diagnosis_new_policy)
hsi_with_perfect_diagnosis_new_policy = HSI_Alri_Treatment(module=alri_module_with_perfect_diagnosis_new_policy,
                                                           person_id=None)

# new policy - imperfect diagnosis (and imperfect treatment)
sim_np = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_new_policy = sim_np.modules['Alri']
_set_new_policy(alri_module_with_imperfect_diagnosis_new_policy)
hsi_with_imperfect_diagnosis_new_policy = HSI_Alri_Treatment(module=alri_module_with_imperfect_diagnosis_new_policy,
                                                             person_id=None)

# current policy - perfect diagnosis and perfect treatment
sim_np_perfect = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_perfect_diagnosis_and_perfect_treatment_new_policy = sim_np_perfect.modules['Alri']
_make_treatment_and_diagnosis_perfect(alri_module_with_perfect_diagnosis_and_perfect_treatment_new_policy)
_set_new_policy(alri_module_with_perfect_diagnosis_and_perfect_treatment_new_policy)
hsi_with_perfect_diagnosis_and_perfect_treatment_new_policy = HSI_Alri_Treatment(
    module=alri_module_with_perfect_diagnosis_and_perfect_treatment_new_policy, person_id=None)

# current policy -----------------
sim0_cp_ant = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ant = sim0_cp_ant.modules['Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ant)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ant = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ant, person_id=None)

sim0_cp_po = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po = sim0_cp_po.modules['Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po, person_id=None)

sim0_cp_ox = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ox = sim0_cp_ox.modules['Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ox)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ox = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ox, person_id=None)

sim0_cp_po_ox = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po_ox = sim0_cp_po_ox.modules['Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po_ox)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po_ox = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po_ox, person_id=None)

# new policy -----------------
sim0_np_ant = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_ant = sim0_np_ant.modules['Alri']
_set_new_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_ant)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_ant = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_ant, person_id=None)

sim0_np_po = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_po = sim0_np_po.modules['Alri']
_set_new_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_po)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_po = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_po, person_id=None)

sim0_np_ox = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_ox = sim0_np_ox.modules['Alri']
_set_new_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_ox)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_ox = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_ox, person_id=None)

sim0_np_po_ox = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_po_ox = sim0_np_po_ox.modules['Alri']
_set_new_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_po_ox)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_po_ox = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_po_ox, person_id=None)


# Alri module with perfect diagnosis (and imperfect treatment)
sim1 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_perfect_diagnosis = sim1.modules['Alri']
_make_hw_diagnosis_perfect(alri_module_with_perfect_diagnosis)
hsi_with_perfect_diagnosis = HSI_Alri_Treatment(module=alri_module_with_perfect_diagnosis, person_id=None)


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



def configuration_to_use(treatment_perfect, hw_dx_perfect, new_policy, oxygen_available, oximeter_available):
    """ Use the simulations based on arguments of perfect treatment, perfect hw dx, and scenario """

    if treatment_perfect:
        if not new_policy:
            hsi = hsi_with_perfect_diagnosis_and_perfect_treatment_current_policy
            alri_module = alri_module_with_perfect_diagnosis_and_perfect_treatment_current_policy
        else:
            hsi = hsi_with_perfect_diagnosis_and_perfect_treatment_new_policy
            alri_module = alri_module_with_perfect_diagnosis_and_perfect_treatment_new_policy
    else:
        if hw_dx_perfect:
            if not new_policy:
                hsi = hsi_with_perfect_diagnosis_current_policy
                alri_module = alri_module_with_perfect_diagnosis_current_policy
            else:
                hsi = hsi_with_perfect_diagnosis_new_policy
                alri_module = alri_module_with_perfect_diagnosis_new_policy
        else:
            if not new_policy:
                if not oxygen_available and not oximeter_available:
                    hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ant
                    alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ant
                elif oxygen_available and oximeter_available:
                    hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po_ox
                    alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po_ox
                elif oxygen_available and not oximeter_available:
                    hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ox
                    alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ox
                elif oximeter_available and not oxygen_available:
                    hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po
                    alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po
                else:
                    raise ValueError('not using a sim above current pol')

            else:
                if not oxygen_available and not oximeter_available:
                    hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_ant
                    alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_ant
                elif oxygen_available and oximeter_available:
                    hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_po_ox
                    alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_po_ox
                elif oxygen_available and not oximeter_available:
                    hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_ox
                    alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_ox
                elif oximeter_available and not oxygen_available:
                    hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_po
                    alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_new_policy_po
                else:
                    raise ValueError('not using a sim above new pol')

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
    oxygen_available,
    treatment_perfect,
    hw_dx_perfect,
    facility_level,
    new_policy,
):
    """Return the percentage by which the treatment reduce the risk of death"""
    # Decide which hsi configuration to use:

    config = configuration_to_use(treatment_perfect=treatment_perfect,
                                  hw_dx_perfect=hw_dx_perfect,
                                  new_policy=new_policy,
                                  oxygen_available=oxygen_available, oximeter_available=oximeter_available
                                  )
    alri_module = config[0]
    hsi = config[1]

    # availability of oxygen
    oxygen_available_list = list()
    if oxygen_available:
        oxygen_available_list = alri_module.models.coverage_of_oxygen(scenario='all_district_psa')
    else:
        oxygen_available_list = alri_module.models.coverage_of_oxygen(scenario='baseline_ant')

    oxygen_available_by_level = {'2': oxygen_available_list[0],
                                 '1b': oxygen_available_list[1],
                                 '1a': oxygen_available_list[2],
                                 '0': oxygen_available_list[2]}

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

    # eligible_for_follow_up_care = all([not ultimate_treatment['antibiotic_indicated'][0].startswith('1st_line_IV'),
    #                                    treatment_fails, sought_follow_up_care, duration_in_days_of_alri > 5])

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
    new_policy = False

    for x in df.itertuples():
        # For Antibiotics only scenario
        v0_ant, v1_ant, v2_ant, v3_ant, v4_ant, v5_ant, v6_ant, v7_ant, v8_ant, v9_ant, v10_ant = treatment_efficacy(
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
            new_policy=new_policy,
            oximeter_available=False,
            oxygen_available=False,
            treatment_perfect=False,
            hw_dx_perfect=hw_dx_perfect,
            facility_level=x.seek_level
        )
        # for PO only scenario
        v0_po, v1_po, v2_po, v3_po, v4_po, v5_po, v6_po, v7_po, v8_po, v9_po, v10_po = treatment_efficacy(
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
            new_policy=new_policy,
            oximeter_available=True,
            oxygen_available=False,
            treatment_perfect=False,
            hw_dx_perfect=hw_dx_perfect,
            facility_level=x.seek_level
        )
        # for Oxygen only scenario
        v0_ox, v1_ox, v2_ox, v3_ox, v4_ox, v5_ox, v6_ox, v7_ox, v8_ox, v9_ox, v10_ox = treatment_efficacy(
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
            new_policy=new_policy,
            oximeter_available=False,
            oxygen_available=True,
            treatment_perfect=False,
            hw_dx_perfect=hw_dx_perfect,
            facility_level=x.seek_level
        )
        # for Oxygen only scenario
        v0_po_ox, v1_po_ox, v2_po_ox, v3_po_ox, v4_po_ox, v5_po_ox, v6_po_ox, v7_po_ox, v8_po_ox, v9_po_ox, v10_po_ox = treatment_efficacy(
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
            new_policy=new_policy,
            oximeter_available=True,
            oxygen_available=True,
            treatment_perfect=False,
            hw_dx_perfect=hw_dx_perfect,
            facility_level=x.seek_level
        )

        # Consider risk of death for this person, intrinsically and under different conditions of treatments
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

            # *** For antibiotics only ***
            # Treatment Efficacy at ultimate facility level (for those referred)
            f'treatment_efficacy_scenario_antibiotics_only_{dx_accuracy}_hw_dx': v0_ant,

            # # Oxygen provided at ultimate facility level (for those referred)
            # f'oxygen_provided_scenario_{scenario}_{dx_accuracy}_hw_dx': v1,
            # # Referred facility level (either 1b or 2)
            # f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx': v2,
            # # First line IV failed
            f'first_line_iv_failed_scenario_antibiotics_only_{dx_accuracy}_hw_dx': v3_ant,
            # # needs referral and referred status
            # f'referral_status_and_oxygen_scenario_{scenario}_{dx_accuracy}_hw_dx': v4,

            # Follow-up properties (ORAL TF) -------------------------------
            f'eligible_for_follow_up_scenario_antibiotics_only_{dx_accuracy}_hw_dx': v5_ant,

            f'follow_up_treatment_failure_scenario_antibiotics_only_{dx_accuracy}_hw_dx': v6_ant,

            # f'oxygen_provided_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx': v7,
            # f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx': v8,
            # f'referral_status_and_oxygen_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx': v9,

            # First line IV failed
            f'first_line_iv_failed_follow_up_scenario_antibiotics_only_{dx_accuracy}_hw_dx': v10_ant,

            # *** For PO only ***
            f'treatment_efficacy_scenario_PO_only_{dx_accuracy}_hw_dx': v0_po,
            f'first_line_iv_failed_scenario_PO_only_{dx_accuracy}_hw_dx': v3_po,
            f'eligible_for_follow_up_scenario_PO_only_{dx_accuracy}_hw_dx': v5_po,
            f'follow_up_treatment_failure_scenario_PO_only_{dx_accuracy}_hw_dx': v6_po,
            f'first_line_iv_failed_follow_up_scenario_PO_only_{dx_accuracy}_hw_dx': v10_po,

            # *** For Oxygen only ***
            f'treatment_efficacy_scenario_Oxygen_only_{dx_accuracy}_hw_dx': v0_ox,
            f'first_line_iv_failed_scenario_Oxygen_only_{dx_accuracy}_hw_dx': v3_ox,
            f'eligible_for_follow_up_scenario_Oxygen_only_{dx_accuracy}_hw_dx': v5_ox,
            f'follow_up_treatment_failure_scenario_Oxygen_only_{dx_accuracy}_hw_dx': v6_ox,
            f'first_line_iv_failed_follow_up_scenario_Oxygen_only_{dx_accuracy}_hw_dx': v10_ox,

            # *** For Oxygen only ***
            f'treatment_efficacy_scenario_PO_and_Ox_{dx_accuracy}_hw_dx': v0_po_ox,
            f'first_line_iv_failed_scenario_PO_and_Ox_{dx_accuracy}_hw_dx': v3_po_ox,
            f'eligible_for_follow_up_scenario_PO_and_Ox_{dx_accuracy}_hw_dx': v5_po_ox,
            f'follow_up_treatment_failure_scenario_PO_and_Ox_{dx_accuracy}_hw_dx': v6_po_ox,
            f'first_line_iv_failed_follow_up_scenario_PO_and_Ox_{dx_accuracy}_hw_dx': v10_po_ox,


            # # # # # # * Classifications * # # # # # #
            # * CLASSIFICATION BY LEVEL 2 *
            'classification_for_treatment_decision_with_oximeter_perfect_accuracy_level2':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level='2', use_oximeter=True, hiv_infected_and_not_on_art=False,
                    un_clinical_acute_malnutrition='well'),

            'classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level='2', use_oximeter=False, hiv_infected_and_not_on_art=False,
                    un_clinical_acute_malnutrition='well'),

            'classification_for_treatment_decision_with_oximeter_imperfect_accuracy_level2':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level='2', use_oximeter=True, hiv_infected_and_not_on_art=False,
                    un_clinical_acute_malnutrition='well'),

            'classification_for_treatment_decision_without_oximeter_imperfect_accuracy_level2':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level='2', use_oximeter=False, hiv_infected_and_not_on_art=False,
                    un_clinical_acute_malnutrition='well'),

            # * CLASSIFICATION BY LEVEL 1a *
            'classification_for_treatment_decision_with_oximeter_perfect_accuracy_level1a':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level='1a', use_oximeter=True, hiv_infected_and_not_on_art=False,
                    un_clinical_acute_malnutrition='well'),

            'classification_for_treatment_decision_without_oximeter_perfect_accuracy_level1a':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level='1a', use_oximeter=False, hiv_infected_and_not_on_art=False,
                    un_clinical_acute_malnutrition='well'),

            'classification_for_treatment_decision_with_oximeter_imperfect_accuracy_level1a':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level='1a', use_oximeter=True, hiv_infected_and_not_on_art=False,
                    un_clinical_acute_malnutrition='well'),

            'classification_for_treatment_decision_without_oximeter_imperfect_accuracy_level1a':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level='1a', use_oximeter=False, hiv_infected_and_not_on_art=False,
                    un_clinical_acute_malnutrition='well'),

        })
    return df.join(pd.DataFrame(risk_of_death))


if __name__ == "__main__":
    table = generate_table()
    table = table.assign(
        has_danger_signs=lambda df: df['symptoms'].apply(lambda x: 'danger_signs' in x),
        needs_oxygen=lambda df: df['oxygen_saturation'] == "<90%",
    )

    low_oxygen = (table["oxygen_saturation"])
    seek_facility = (table["seek_level"])

    table['final_death_ant_only'] = table.loc[
        ((table[f'treatment_efficacy_scenario_antibiotics_only_{dx_accuracy}_hw_dx']) &
         (table[f'eligible_for_follow_up_scenario_antibiotics_only_{dx_accuracy}_hw_dx'] == False)) |
        ((table[f'treatment_efficacy_scenario_antibiotics_only_{dx_accuracy}_hw_dx']) &
         (table[f'eligible_for_follow_up_scenario_antibiotics_only_{dx_accuracy}_hw_dx'].str[0] == True) &
         (table[f'follow_up_treatment_failure_scenario_antibiotics_only_{dx_accuracy}_hw_dx'] == True)),
        'will_die']

    table['final_death_po_only'] = table.loc[
        ((table[f'treatment_efficacy_scenario_PO_only_{dx_accuracy}_hw_dx']) &
         (table[f'eligible_for_follow_up_scenario_PO_only_{dx_accuracy}_hw_dx'] == False)) |
        ((table[f'treatment_efficacy_scenario_PO_only_{dx_accuracy}_hw_dx']) &
         (table[f'eligible_for_follow_up_scenario_PO_only_{dx_accuracy}_hw_dx'].str[0] == True) &
         (table[f'follow_up_treatment_failure_scenario_PO_only_{dx_accuracy}_hw_dx'] == True)),
        'will_die']

    table['final_death_ox_only'] = table.loc[
        ((table[f'treatment_efficacy_scenario_Oxygen_only_{dx_accuracy}_hw_dx']) &
         (table[f'eligible_for_follow_up_scenario_Oxygen_only_{dx_accuracy}_hw_dx'] == False)) |
        ((table[f'treatment_efficacy_scenario_Oxygen_only_{dx_accuracy}_hw_dx']) &
         (table[f'eligible_for_follow_up_scenario_Oxygen_only_{dx_accuracy}_hw_dx'].str[0] == True) &
         (table[f'follow_up_treatment_failure_scenario_Oxygen_only_{dx_accuracy}_hw_dx'] == True)),
        'will_die']  # .groupby(by=[low_oxygen]).sum()

    table['final_death_po_and_ox'] = table.loc[
        ((table[f'treatment_efficacy_scenario_PO_and_Ox_{dx_accuracy}_hw_dx']) &
         (table[f'eligible_for_follow_up_scenario_PO_and_Ox_{dx_accuracy}_hw_dx'] == False)) |
        ((table[f'treatment_efficacy_scenario_PO_and_Ox_{dx_accuracy}_hw_dx']) &
         (table[f'eligible_for_follow_up_scenario_PO_and_Ox_{dx_accuracy}_hw_dx'].str[0] == True) &
         (table[f'follow_up_treatment_failure_scenario_PO_and_Ox_{dx_accuracy}_hw_dx'] == True)),
        'will_die']


    def summarize_by(df: pd.DataFrame, by: List[str], columns: [List[str]]) -> pd.DataFrame:
        """Helper function returns dataframe that summarizes the dataframe provided using groupby, with by arguements,
        and provides columns as follows: [size-of-the-group, mean-of-column-1, mean-of-column-2, ...]"""
        return pd.DataFrame({'fraction': df.groupby(by=by).size()}).apply(lambda x: x / x.sum(), axis=0) \
            .join(df.groupby(by=by)[columns].mean())


    # Examine case mix
    case_mix_by_disease_and_pathogen = summarize_by(table,
                                                    by=['disease_type', 'pathogen'],
                                                    columns=['has_danger_signs', 'needs_oxygen',
                                                             'prob_die_if_no_treatment']
                                                    ).reset_index()
    print(f"{case_mix_by_disease_and_pathogen=}")

    # Average risk of death by disease type
    table.groupby(by=['disease_type'])['prob_die_if_no_treatment'].mean()

    # Examine danger_signs as a predictor of SpO2 < 90%
    print(pd.crosstab(
        pd.Series(table['oxygen_saturation'] == "<90%", name='SpO2<90%'),
        table['has_danger_signs'],
        normalize="index", margins=True
    ))

    # Look at diagnosis errors (assuming that the "truth" is
    # "classification_for_treatment_decision_with_oximeter_perfect_accuracy")
    truth = table["classification_for_treatment_decision_with_oximeter_perfect_accuracy_level2"]


    def cross_tab(truth: pd.Series, dx: pd.Series):
        """Return cross-tab between truth and dx and count number of incongruent rows."""
        print(f"% cases mis-classified is {100 * (truth != dx).mean()}%")
        return pd.crosstab(truth, dx, normalize='index')

    # Examine risk of death and treatment effectiveness by disease classification and oxygen saturation
    # And include estimate of % of deaths by scaling using "prob_die_if_no_treatment".
    # UNDER PERFECT HW DX
    risk_of_death = summarize_by(
        df=table,
        by=['oxygen_saturation', 'classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2'],
        columns=[
            'will_die',
            'final_death_ant_only',
            'final_death_po_only',
            'final_death_ox_only',
            'final_death_po_and_ox',
        ]
    ).assign(
        fraction_of_deaths=lambda df: (
            (df.fraction * df.will_die) / (df.fraction * df.will_die).sum()
        )
    )
    print(f"{risk_of_death=}")


    # Overall summary figure: Number of deaths in the cohort Deaths broken down by ( disease / oxygen_saturation) when
    # * No Treatment
    # * Treatment antibiotics only (no oxygen, no oximeter)
    # * Treatment with antibiotics and oxygen (no oximeter)
    # * Treatment with antibiotics and oxygen and use of oximeter
    # ... under assumptions of (i) normal treatment and normal dx accuracy; (ii) normal treatment and perfect dx
    # accuracy

    # disease_classification2 = table["classification_for_treatment_decision_with_oximeter_perfect_accuracy_level2"]
    disease_classification1a = table["classification_for_treatment_decision_with_oximeter_perfect_accuracy_level1a"]
    disease_classification2 = table['classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2']
    low_oxygen = (table["oxygen_saturation"] == "<90%").replace({True: '<90%', False: ">=90%"})
    # low_oxygen = (table["oxygen_saturation"])
    fraction = risk_of_death['fraction']
    number_cases = table.groupby(by=[disease_classification2, low_oxygen]).size()
    dx_accuracy = 'perfect'

    res = {
            "Antiobiotics only": (
                table['final_death_ant_only']
                ).groupby(by=[disease_classification2, low_oxygen]).sum(),
            "+ oxygen": (
                table['final_death_ox_only']).groupby(by=[disease_classification2, low_oxygen]).sum(),
            "+ oximeter": (
                table['final_death_po_only']).groupby(by=[disease_classification2, low_oxygen]).sum(),
            "+ oximeter & oxygen": (
                table['final_death_po_and_ox']).groupby(by=[disease_classification2, low_oxygen]).sum(),

    }

    results = (100_000 / len(table)) * pd.concat({k: pd.DataFrame(v) for k, v in res.items()}, axis=1)
    # results = pd.concat({k: pd.DataFrame(v) for k, v in res.items()}, axis=1)

    # # Plot graph by * classification differences *
    # # compare the classification differences
    # res1 = {
    #     "By hospital classification": {
    #         "No Treatment": table['prob_die_if_no_treatment'].groupby(by=[disease_classification2, low_oxygen]).sum(),
    #         "Antiobiotics only": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx_level1a'] / 100.0
    #              )).groupby(by=[disease_classification2, low_oxygen]).sum(),
    #         "+ oxygen": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx_level1a']
    #              / 100.0
    #              )).groupby(by=[disease_classification2, low_oxygen]).sum(),
    #         "+ oximeter": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx_level1a'] / 100.0
    #              )).groupby(by=[disease_classification2, low_oxygen]).sum(),
    #         "+ oximeter & oxygen": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_level1a'] / 100.0
    #              )).groupby(by=[disease_classification2, low_oxygen]).sum(),
    #     },
    #
    #     "By health centre classification": {
    #         "No Treatment": table['prob_die_if_no_treatment'].groupby(by=[disease_classification1a, low_oxygen]).sum(),
    #         "Antiobiotics only": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx_level1a'] / 100.0
    #              )).groupby(by=[disease_classification1a, low_oxygen]).sum(),
    #         "+ oxygen": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx_level1a']
    #              / 100.0
    #              )).groupby(by=[disease_classification1a, low_oxygen]).sum(),
    #         "+ oximeter": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx_level1a'] / 100.0
    #              )).groupby(by=[disease_classification1a, low_oxygen]).sum(),
    #         "+ oximeter & oxygen": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_level1a'] / 100.0
    #              )).groupby(by=[disease_classification1a, low_oxygen]).sum(),
    #     },
    # }
    #
    # results = (100_000 / len(table)) * pd.concat({k: pd.DataFrame(v) for k, v in res1.items()}, axis=1)

    # reorder the index:
    reorderlist = list()
    if results.index.size == 12:  # broken down by 3 SpO2 levels
        # rename the index - to shorten the labels
        results.index = pd.MultiIndex.from_tuples(
            [(x[0], f'SpO2_{x[1]}') if x[1] == '90-92%' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('cough/cold', f'SpO2{x[1]}') if x[0] == 'cough_or_cold' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('fb-pneumonia', f'SpO2{x[1]}') if x[0] == 'fast_breathing_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ci-pneumonia', f'SpO2{x[1]}') if x[0] == 'chest_indrawing_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ds-pneumonia', f'SpO2{x[1]}') if x[0] == 'danger_signs_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))

        reorderlist = [('cough/cold', 'SpO2>=93%'),
                       ('cough/cold', 'SpO2_90-92%'),
                       ('cough/cold', 'SpO2<90%'),
                       ('fb-pneumonia', 'SpO2>=93%'),
                       ('fb-pneumonia', 'SpO2_90-92%'),
                       ('fb-pneumonia', 'SpO2<90%'),
                       ('ci-pneumonia', 'SpO2>=93%'),
                       ('ci-pneumonia', 'SpO2_90-92%'),
                       ('ci-pneumonia', 'SpO2<90%'),
                       ('ds-pneumonia', 'SpO2>=93%'),
                       ('ds-pneumonia', 'SpO2_90-92%'),
                       ('ds-pneumonia', 'SpO2<90%'),
                       ]

    elif results.index.size == 5:
        # rename the index for danger_signs_pneumonia with SpO2<90% to any_severity,
        # only applicable to current policy (inpatient care to SpO2<90%) results
        idx_to_change = ('danger_signs_pneumonia', '<90%')
        results.index = pd.MultiIndex.from_tuples(
            [('any severity', f'SpO2{x[1]}') if x == idx_to_change else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_with_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('cough/cold', f'SpO2{x[1]}') if x[0] == 'cough_or_cold' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_with_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('fb-pneumonia', f'SpO2{x[1]}') if x[0] == 'fast_breathing_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_with_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ci-pneumonia', f'SpO2{x[1]}') if x[0] == 'chest_indrawing_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_with_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ds-pneumonia', f'SpO2{x[1]}') if x[0] == 'danger_signs_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_with_oximeter_perfect_accuracy", "oxygen_saturation"))

        reorderlist = [('cough/cold', 'SpO2>=90%'),
                       ('fb-pneumonia', 'SpO2>=90%'),
                       ('ci-pneumonia', 'SpO2>=90%'),
                       ('ds-pneumonia', 'SpO2>=90%'),
                       ('any severity', 'SpO2<90%')
                       ]

    elif results.index.size == 8:
        results.index = pd.MultiIndex.from_tuples(
            [(x[0], '>=90%') if x[1] == '90-92%' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('cough/cold', f'SpO2{x[1]}') if x[0] == 'cough_or_cold' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('fb-pneumonia', f'SpO2{x[1]}') if x[0] == 'fast_breathing_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ci-pneumonia', f'SpO2{x[1]}') if x[0] == 'chest_indrawing_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ds-pneumonia', f'SpO2{x[1]}') if x[0] == 'danger_signs_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))

        reorderlist = [('cough/cold', 'SpO2>=90%'),
                       ('cough/cold', 'SpO2<90%'),
                       ('fb-pneumonia', 'SpO2>=90%'),
                       ('fb-pneumonia', 'SpO2<90%'),
                       ('ci-pneumonia', 'SpO2>=90%'),
                       ('ci-pneumonia', 'SpO2<90%'),
                       ('ds-pneumonia', 'SpO2>=90%'),
                       ('ds-pneumonia', 'SpO2<90%')
                       ]
    else:
        raise ValueError(f'Index size not recognised {results.index.size}')

    results.reindex(reorderlist)
    results = results.iloc[pd.Categorical(results.index, reorderlist).argsort(ascending=False)]

    # index = results.index.tolist()  -- > copy this to match the multiindex

    assign_colors = dict()

    if results.index.size == 12:
        assign_colors = {('cough/_cold', 'SpO2>=93%'): '#fa8405',
                         ('cough/cold', 'SpO2_90-92%'): 'gold',
                         ('cough/cold', 'SpO2<90%'): 'navajowhite',
                         ('fb-pneumonia', 'SpO2>=93%'): 'navy',
                         ('fb-pneumonia', 'SpO2_90-92%'): 'blue',
                         ('fb-pneumonia', 'SpO2<90%'): 'deepskyblue',
                         ('ci-pneumonia', 'SpO2>=93%'): 'darkgreen',
                         ('ci-pneumonia', 'SpO2_90-92%'): 'seagreen',
                         ('ci-pneumonia', 'SpO2<90%'): 'lightgreen',
                         ('ds-pneumonia', 'SpO2>=93%'): 'darkred',
                         ('ds-pneumonia', 'SpO2_90-92%'): 'orangered',
                         ('ds-pneumonia', 'SpO2<90%'): 'darksalmon',
                         }
    elif results.index.size == 5:
        assign_colors = {('cough/cold', 'SpO2>=90%'): 'tab:orange',
                         ('fb-pneumonia', 'SpO2>=90%'): 'mediumblue',
                         ('ci-pneumonia', 'SpO2>=90%'): 'tab:green',
                         ('ds-pneumonia', 'SpO2>=90%'): 'firebrick',
                         ('any severity', 'SpO2<90%'): 'palevioletred'
                         }

    elif results.index.size == 8:
        assign_colors = {('cough/cold', 'SpO2>=90%'): 'tab:orange',
                         ('cough/cold', 'SpO2<90%'): 'navajowhite',
                         ('fb-pneumonia', 'SpO2>=90%'): 'mediumblue',
                         ('fb-pneumonia', 'SpO2<90%'): 'deepskyblue',
                         ('ci-pneumonia', 'SpO2>=90%'): 'tab:green',
                         ('ci-pneumonia', 'SpO2<90%'): 'lightgreen',
                         ('ds-pneumonia', 'SpO2>=90%'): 'firebrick',
                         ('ds-pneumonia', 'SpO2<90%'): 'darksalmon'
                         }

    else:
        raise ValueError(f'Index size not recognised {results.index.size}')

    # PLOTS!!
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    results.T.plot.bar(stacked=True, ax=ax, width=0.6, legend=False,
             color=results.index.map(assign_colors))
    plt.xticks(rotation=0, ha='center')

    # Create a dictionary to map old names to new names
    label_mapping = {
        results.columns.get_level_values(1)[0]: 'Antibiotics only',
        results.columns.get_level_values(1)[1]: '+ Oxygen',
        results.columns.get_level_values(1)[2]: '+ Pulse Oximetry',
        results.columns.get_level_values(1)[3]: '+ Pulse Oximetry \n & Oxygen'
    }
    new_labels = [label_mapping.get(label, label) for label in results.columns.get_level_values(1)]
    ax.set_xticklabels(new_labels)

    ax.set_ylabel('Deaths per 100,000 cases of ALRI')
    # ax.set_title(f"", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels),
              title='Case Type',
              loc='upper left',
              bbox_to_anchor=(1, 1),
              fontsize=7,
              title_fontsize=9)

    fig.suptitle('Current policy - enhance IMCI at rural hospitals', fontsize=12, fontweight='semibold')
    fig.show()
    # fig.savefig(Path('./outputs') / ('current policy' + datestamp + ".pdf"), format='pdf')
    plt.close(fig)

    # # # # Calculations # # # #
    # get the stats of impact between interventions
    # base comparator - natural history mortality
    natural_deaths = (table['prob_die_if_no_treatment']).groupby(by=[disease_classification2, low_oxygen]).sum()
    number_cases = table.groupby(by=[disease_classification2, low_oxygen]).size()
    cfr = natural_deaths / number_cases
    # overall_cfr = (cfr * fraction).sum()  # only works for 3 spo2 breakdowns
    # prop_death = natural_deaths / natural_deaths.sum()

    # base comparator - antibiotics only under perfect hw dx
    deaths_antibiotics_level2 = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_{dx_accuracy}_hw_dx_level2'] / 100.0)
                                 ).groupby(by=[disease_classification2, low_oxygen]).sum()

    # pulse oximeter only under perfect hw dx
    deaths_po_only_level2 = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_{dx_accuracy}_hw_dx_level2'] / 100.0)
                             ).groupby(by=[disease_classification2, low_oxygen]).sum()

    # oxygen only under perfect hw dx
    deaths_ox_only_level2 = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_{dx_accuracy}_hw_dx_level2'] / 100.0)
                             ).groupby(by=[disease_classification2, low_oxygen]).sum()

    # oxygen only under perfect hw dx
    deaths_po_and_ox_level2 = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_{dx_accuracy}_hw_dx_level2'] / 100.0)
                               ).groupby(by=[disease_classification2, low_oxygen]).sum()

    impact_antibiotics = 1 - (deaths_antibiotics_level2.mean() / natural_deaths.mean())
    impact_po = 1 - (deaths_po_only_level2.mean() / natural_deaths.mean())  #deaths_antibiotics_perfect_hw.mean()  # 0.887704
    impact_ox = 1 - (deaths_ox_only_level2.mean() / natural_deaths.mean()) #deaths_antibiotics_perfect_hw.mean()  # 0.76990223%
    impact_po_and_ox = 1 - (deaths_po_and_ox_level2.mean() / natural_deaths.mean())  # deaths_antibiotics_perfect_hw.mean()  # 0.671578

    cfr_antibiotics = deaths_antibiotics_level2 / number_cases
    cfr_po = deaths_po_only_level2 / number_cases
    cfr_ox = deaths_ox_only_level2 / number_cases
    cfr_po_and_ox = deaths_po_and_ox_level2 / number_cases


    # Repeat for FACILITY LEVEL 1A
    # base comparator - antibiotics only under normal hw dx
    deaths_antibiotics_level1a = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_{dx_accuracy}_hw_dx_level1a'] / 100.0)
                                  ).groupby(by=[disease_classification2, low_oxygen]).sum()

    # pulse oximeter only under perfect hw dx
    deaths_po_only_level1a = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_{dx_accuracy}_hw_dx_level1a'] / 100.0)
                              ).groupby(by=[disease_classification2, low_oxygen]).sum()

    # oxygen only under perfect hw dx
    deaths_ox_only_level1a = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_{dx_accuracy}_hw_dx_level1a'] / 100.0)
                              ).groupby(by=[disease_classification2, low_oxygen]).sum()

    # oxygen only under perfect hw dx
    deaths_po_and_ox_level1a = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_{dx_accuracy}_hw_dx_level1a'] / 100.0)
                                ).groupby(by=[disease_classification2, low_oxygen]).sum()

    # impact reduction on mortality
    impact_antibiotics_level1a = 1 - (deaths_antibiotics_level1a.mean() / natural_deaths.mean())
    impact_po_level1a = 1 - (deaths_po_only_level1a.mean() / natural_deaths.mean())  #deaths_antibiotics_perfect_hw.mean()  # 0.887704
    impact_ox_level1a = 1 - (deaths_ox_only_level1a.mean() / natural_deaths.mean()) #deaths_antibiotics_perfect_hw.mean()  # 0.76990223%
    impact_po_and_ox_level1a = 1 - (deaths_po_and_ox_level1a.mean() / natural_deaths.mean())  # deaths_antibiotics_perfect_hw.mean()  # 0.671578

    cfr_antibiotics_level1a = deaths_antibiotics_level1a / number_cases
    cfr_po_level1a = deaths_po_only_level1a / number_cases
    cfr_ox_level1a = deaths_ox_only_level1a / number_cases
    cfr_po_and_ox_level1a = deaths_po_and_ox_level1a / number_cases
