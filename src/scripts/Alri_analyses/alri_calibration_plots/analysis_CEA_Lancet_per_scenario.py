""" This script will output the values for the CEA for the Lancet Commission on Medical oxygen """

import random
from pathlib import Path
from typing import List
import datetime

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
    _make_treatment_and_diagnosis_perfect,
    _set_current_policy,
    _set_new_policy
)
from tlo.util import sample_outcome

MODEL_POPSIZE = 16_000
MIN_SAMPLE_OF_NEW_CASES = 200
NUM_REPS_FOR_EACH_CASE = 20

# scenario = 'baseline_ant'
# scenario = 'baseline_ant_with_po_level2'
# scenario = 'baseline_ant_with_po_level1b'
# scenario = 'baseline_ant_with_po_level1a'
# scenario = 'baseline_ant_with_po_level0'
# scenario = 'existing_psa'
# scenario = 'existing_psa_with_po_level2'
# scenario = 'existing_psa_with_po_level1b'
# scenario = 'existing_psa_with_po_level1a'
scenario = 'existing_psa_with_po_level0'
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

oximeter_available = True if 'with_po' in scenario else False

implementation_level = ('2') if 'with_po_level2' in scenario else ('2', '1b') if 'with_po_level1b' in scenario else \
    ('2', '1b', '1a') if 'with_po_level1a' in scenario else ('2', '1b', '1a', '0') if 'with_po_level0' in scenario else \
        'none'

# Helper function for conversion between odds and probabilities
to_odds = lambda pr: pr / (1.0 - pr)  # noqa: E731
to_prob = lambda odds: odds / (1.0 + odds)  # noqa: E731

# Date for saving the image for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


def get_sim(popsize):
    """Return a simulation (composed of only <5 years old) that has run for 0 days."""
    resourcefilepath = Path('./resources')
    start_date = Date(2010, 1, 1)
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
    sim.modules['Demography'].parameters['max_age_initial'] = 5
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

# Alri module with perfect diagnosis (and imperfect treatment)
sim1 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_perfect_diagnosis = sim1.modules['Alri']
_make_hw_diagnosis_perfect(alri_module_with_perfect_diagnosis)
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
        if hw_dx_perfect:
            hsi = hsi_with_perfect_diagnosis
            alri_module = alri_module_with_perfect_diagnosis
        else:
            hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment
            alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment

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
    if referred_up and (oxygen_saturation == '<90%'):
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

    # follow-up oral TF --------------------------------------------------
    # eligible_for_follow_up_care = all([not ultimate_treatment['antibiotic_indicated'][0].startswith('1st_line_IV'),
    #                                    treatment_fails, sought_follow_up_care, duration_in_days_of_alri > 5])

    # do follow-up care for oral TF
    follow_up_care = alri_module.follow_up_treatment_failure(
        original_classification_given=classification_for_treatment_decision)

    sought_follow_up_care = follow_up_care[0]
    follow_up_classification = follow_up_care[1]

    eligible_for_follow_up_care = all([not ultimate_treatment['antibiotic_indicated'][0].startswith('1st_line_IV'),
                                       treatment_fails, sought_follow_up_care, duration_in_days_of_alri > 5])

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
    if referred_up_follow_up and (oxygen_saturation == '<90%'):
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
    # apply the TFs:
    # "Treatment Fails" is the probability that a death is averted (if one is schedule)
    treatment_fails_at_follow_up = alri_module.models.treatment_fails(
        antibiotic_provided=ultimate_treatment_follow_up['antibiotic_indicated'][0],
        oxygen_provided=ultimate_treatment_follow_up['oxygen_indicated'] if oxygen_available_by_level[new_facility_level_follow_up]  and new_facility_level_follow_up in ('2', '1b') else False,
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

    if not eligible_for_follow_up_care:
        return (treatment_fails,
                ultimate_treatment['oxygen_indicated'] if oxygen_available_by_level[new_facility_level] else False,
                new_facility_level,
                (referral_status, pre_referral_oxygen),
                eligible_for_follow_up_care,
                None, None, None, None)

    else:
        return (treatment_fails,
                ultimate_treatment['oxygen_indicated'] if oxygen_available_by_level[new_facility_level] else False,
                new_facility_level,
                (referral_status, pre_referral_oxygen),
                (eligible_for_follow_up_care, follow_up_classification),
                treatment_fails_at_follow_up,
                ultimate_treatment_follow_up['oxygen_indicated'] if oxygen_available_by_level[new_facility_level_follow_up] else False,
                new_facility_level_follow_up,
                (referral_status_follow_up, pre_referral_oxygen_follow_up))


def generate_table():
    """Return table providing a representative case mix of persons with Alri, the intrinsic risk of death and the
    efficacy of treatment under different conditions."""

    # Get Case Mix
    df = generate_case_mix()
    seek_level = list()
    for x in df.itertuples():
        seek_level.append({
            'seek_level': alri_module_with_perfect_diagnosis.seek_care_level(
                symptoms=x.symptoms
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
        v0, v1, v2, v3, v4, v5, v6, v7, v8 = treatment_efficacy(
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

            # * PERFECT HW Diagnosis *
            # Treatment Efficacy at ultimate facility level (for those referred)
            f'treatment_efficacy_scenario_{scenario}_{dx_accuracy}_hw_dx': v0,

            # Oxygen provided at ultimate facility level (for those referred)
            f'oxygen_provided_scenario_{scenario}_{dx_accuracy}_hw_dx': v1,

            # Referred facility level (either 1b or 2)
            f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx': v2,

            # needs referral and referred status
            f'referral_status_and_oxygen_scenario_{scenario}_{dx_accuracy}_hw_dx': v3,

            # Follow-up properties (ORAL TF) -------------------------------
            f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx': v4,

            f'follow_up_treatment_failure_scenario_{scenario}_{dx_accuracy}_hw_dx': v5,

            f'oxygen_provided_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx': v6,

            f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx': v7,

            f'referral_status_and_oxygen_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx': v8,

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

        })

    return df.join(pd.DataFrame(risk_of_death))


if __name__ == "__main__":
    table = generate_table()

    # YLD = I × DW × L
    daly_weight = table['classification_for_treatment_decision_with_oximeter_perfect_accuracy_level2'].apply(
        lambda x: 0.133 if 'danger_signs_pneumonia' in x else 0.051)
    duration_years = table['duration_in_days_of_alri'] / 365.25

    YLD = table.index.size * (daly_weight * duration_years).mean()   # 67.719

    # YLL = N × L
    mean_age = table['age_exact_years'].mean()  # 1.5169246
    total_deaths = table.loc[(table[
        f'treatment_efficacy_scenario_{scenario}_{dx_accuracy}_hw_dx']) &
              (table[f'follow_up_treatment_failure_scenario_{scenario}_{dx_accuracy}_hw_dx'] != False),
              'will_die'].sum()

    # deaths_scenario = table['prob_die_if_no_treatment'] * (1.0 - table[
    #     f'treatment_efficacy_scenario_{scenario}_{dx_accuracy}_hw_dx'] / 100.0)
    # total_deaths = deaths_scenario.sum()

    YLL = total_deaths * (63 - mean_age)

    # DALYS
    DALYs = (YLD + YLL).sum()

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
            lambda x: (1 * 2 * 5 * 0.022448) if x < 1 else (2 * 2 * 5 * 0.022448) if 1 <= x < 2 else
            (3 * 2 * 5 * 0.022448))

        table['iv_antibiotics_cost'] = table['age_exact_years'].apply(
            lambda x: (((1 / 2.5) * 4 * 5 * 0.2114321) + ((1 / 2) * 5 * 0.2114321)) if x < 1 / 3 else
            (((2 / 2.5) * 4 * 5 * 0.2114321) + ((1.8 / 2) * 5 * 0.1464)) if 1 / 3 <= x < 1 else
            (((3 / 2.5) * 4 * 5 * 0.2114321) + ((2.7 / 2) * 5 * 0.1464)) if 1 <= x < 3 else
            (((5 / 2.5) * 4 * 5 * 0.2114321) + ((3.5 / 2) * 5 * 0.1464)))  # ampicillin + gentamicin

        oral_antibiotics_cost = table.loc[
            ((table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] == 'fast_breathing_pneumonia') |
            (table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] == 'chest_indrawing_pneumonia')),
            'oral_amox_cost'].sum()

        iv_antibiotic_cost = table.loc[
            (table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] == 'danger_signs_pneumonia'),
            'iv_antibiotics_cost'].sum()

        follow_up_oral_antibiotics_cost = table.loc[
            (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
            (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'chest_indrawing_pneumonia'),
            'oral_amox_cost'].sum()

        follow_up_iv_antibiotics_cost = table.loc[
            (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
            (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'danger_signs_pneumonia'),
            'iv_antibiotics_cost'].sum()

        # OUTPATIENT CONSULTATION COST ------------
        table['consultation_cost_seek_level'] = table['seek_level'].apply(
            lambda x: 2.45 if x == '2' else 2.35 if x == '1b' else 2.06 if x == '1a' else 1.67
        )
        table['consultation_cost_final_facility'] = table[
            f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx'].apply(
            lambda x: 2.45 if x == '2' else 2.35 if x == '1b' else 2.06 if x == '1a' else 1.67
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
        table['inpatient_bed_cost'] = table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx'].apply(
            lambda x: 6.81 * 5 if x == '2' else 6.53 * 5 if x == '1b' else 0
        )

        hospitalisation_cost = table.loc[
            (table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx'] == 'danger_signs_pneumonia') &
            ((table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx'] == '2') |
             (table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx'] == '1b')),  'inpatient_bed_cost'].sum()

        # follow-up inpatient bed days costs --------
        table['inpatient_bed_cost_follow_up'] = table[f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].apply(
            lambda x: 6.81 * 5 if x == '2' else 6.53 * 5 if x == '1b' else 0
        )

        follow_up_hospitalisation_cost = table.loc[
            (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[0] == True) &
            (table[f'eligible_for_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].str[1] == 'danger_signs_pneumonia') &
            ((final_facility_follow_up == '2') | (final_facility_follow_up == '1b')),
            'inpatient_bed_cost_follow_up'].sum()

        # PULSE OXIMETRY COST ----------------------
        table['PO_cost_seek_level'] = table['seek_level'].apply(
            lambda x: 0.15897 if x in ('2', '1b') else 0.08667 if x == '1a' else 0.06025)
        table['PO_cost_final_level'] = table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx'].apply(
            lambda x: 0.15897 if x in ('2', '1b') else 0.08667 if x == '1a' else 0.06025)
        table['PO_cost_final_level_follow_up'] = table[
            f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].apply(
            lambda x: 0.15897 if x in ('2', '1b') else 0.08667 if x == '1a' else 0.06025 if x == '0' else 0)

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

        return deaths_per_facility, costs_df, need_oxygen_ds_with_SpO2lt90, total_full_oxygen_provided


    # Do the Dataframe with summary output --------------------------------------

    # Baseline antibiotics
    cea_df = cea_df_by_scenario(scenario=scenario, dx_accuracy=dx_accuracy)
    deaths = cea_df[0].sum()
    costs = cea_df[1]
    total_cost = cea_df[1].sum(axis=1)
    oxygen_provided = cea_df[3]
    need_oxygen = cea_df[2]

    debug_point = 0

    # need_oxygen_ds_with_SpO2lt90 for baseline = 2765

    def scaled_baseline_tf(df, unscaled_intercept):
        """" scale the treatment failure of IV antibiotics for danger signs pneumonia """

        unscaled_lm = \
            LinearModel(
                LinearModelType.MULTIPLICATIVE,
                unscaled_intercept,
                Predictor('disease_type', external=True).when('pneumonia', 1.71),
                Predictor('complications', external=True).when(True, 2.31),
                Predictor('hiv_not_on_art', external=True).when(True, 1.8),
                Predictor('un_clinical_acute_malnutrition', external=True).when('MAM', 1.48),
                Predictor('un_clinical_acute_malnutrition', external=True).when('SAM', 2.02),
                Predictor('oxygen_saturation', external=True).when(True, 1.28),
                Predictor('symptoms', external=True).when(True, 1.55),
                Predictor('referral_hc', external=True).when(True, 1.66),
            )

        # make unscaled linear model
        unscaled_tf = unscaled_lm.predict(df,
                                          disease_type=df['disease_type'],
                                          complications=any(c in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for c in df.complications),
                                          hiv_not_on_art=df['hiv_not_on_art'],
                                          un_clinical_acute_malnutrition=df['un_clinical_acute_malnutrition'],
                                          oxygen_saturation=df.oxygen_saturation == '<90%',
                                          symptoms='danger_signs' in df.symptoms,
                                          referral_hc=any(l in ['0', '1a'] for l in df.seek_level) and
                                                      df.classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2 == 'danger_signs_pneumonia'
                                          )

        cases_combo = unscaled_tf.unique()
        total_each_combo = unscaled_tf.value_counts()
        total_unscale_tf = (cases_combo * total_each_combo).sum() / total_each_combo.sum()

        scaling_factor = unscaled_intercept / total_unscale_tf

        scaled_baseline_tf = unscaled_intercept * scaling_factor

        return scaled_baseline_tf


    df = table.drop(table[table.age_exact_years < 1 / 6].index)

    df_ds_pneumonia = df.drop(df.index[df[f"classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2"] != 'danger_signs_pneumonia'])

    get_base_tf = scaled_baseline_tf(df=df_ds_pneumonia, unscaled_intercept=0.17085)

    ds_pneumonia_total = len(df_ds_pneumonia)

    def scaled_baseline_oral_tf(df, unscaled_intercept):
        """" scale the treatment failure of IV antibiotics for danger signs pneumonia """

        def scaled_multiplicative_part():
            """ First scale for the multiplicative risk factors """

            unscaled_lm_multiplicative = \
                LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    unscaled_intercept,
                    Predictor('disease_type', external=True).when('pneumonia', 1.71),
                    Predictor('complications', external=True).when(True, 2.31),
                )

            # make unscaled linear model
            unscaled_tf_multiplicative = unscaled_lm_multiplicative.predict(
                df,
                disease_type=df['disease_type'],
                complications=any(c in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for c in df.complications),

            )

            cases_combo = unscaled_tf_multiplicative.unique()
            total_each_combo = unscaled_tf_multiplicative.value_counts()
            total_unscale_tf_multiplicative = (cases_combo * total_each_combo).sum() / total_each_combo.sum()

            scaling_factor_multiplicative = unscaled_intercept / total_unscale_tf_multiplicative

            scaled_baseline_tf_multiplicative = unscaled_intercept * scaling_factor_multiplicative

            return scaled_baseline_tf_multiplicative

        scaled_baseline_tf_multiplicative = scaled_multiplicative_part()

        unscaled_lm = \
            LinearModel(
                LinearModelType.LOGISTIC,
                scaled_baseline_tf_multiplicative,
                Predictor('un_clinical_acute_malnutrition', external=True).when('MAM', 1.88),
                Predictor('moderate_hypoxaemia', external=True).when(True, 1.73),
            )

        # make unscaled linear model
        unscaled_tf = unscaled_lm.predict(
            df,
            disease_type=df['disease_type'],
            complications=any(c in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for c in df.complications),
            un_clinical_acute_malnutrition=df['un_clinical_acute_malnutrition'],
            below90SpO2=df.oxygen_saturation == '<90%',
            moderate_hypoxaemia=df.oxygen_saturation == '90-92%'

        )

        cases_combo = unscaled_tf.unique()
        total_each_combo = unscaled_tf.value_counts()
        total_unscale_tf = (cases_combo * total_each_combo).sum() / total_each_combo.sum()

        scaling_factor = scaled_baseline_tf_multiplicative / total_unscale_tf

        scaled_baseline_tf = scaled_baseline_tf_multiplicative * scaling_factor

        return scaled_baseline_tf


    # TF for > 2 months
    df_fb_pneumonia = df.drop(df.index[df[f"classification_for_treatment_decision_with_oximeter_perfect_accuracy_sought_level"] != 'fast_breathing_pneumonia'])
    get_base_oral_tf_fb_pneum = scaled_baseline_oral_tf(df=df_fb_pneumonia, unscaled_intercept=0.101)

    df_ci_pneumonia = df.drop(df.index[df[f"classification_for_treatment_decision_with_oximeter_perfect_accuracy_sought_level"] != 'chest_indrawing_pneumonia'])
    get_base_oral_tf_ci_pneum = scaled_baseline_oral_tf(df=df_ci_pneumonia, unscaled_intercept=0.108)

    # TF for young infants < 2 months
    df2 = table.drop(table[table.age_exact_years >= 1 / 6].index)
    df_fb_pneumonia_infants = df2.drop(df2.index[df2[f"classification_for_treatment_decision_with_oximeter_perfect_accuracy_sought_level"] != 'fast_breathing_pneumonia'])
    get_base_oral_tf = scaled_baseline_oral_tf(df=df_fb_pneumonia_infants,  unscaled_intercept=0.054)



# 1.1 Import TLO model availability data
#------------------------------------------------------
resourcefilepath = Path('./resources')
tlo_availability_df = pd.read_csv(resourcefilepath / "healthsystem" / "consumables" / "ResourceFile_Consumables_availability_small.csv")
# Drop any scenario data previously included in the resourcefile
tlo_availability_df = tlo_availability_df[['Facility_ID', 'month', 'item_code', 'available_prop']]

# 1.1.1 Attach district, facility level, program to this dataset
#----------------------------------------------------------------
# Get TLO Facility_ID for each district and facility level
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
districts = set(pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')['District'])
fac_levels = {'0', '1a', '1b', '2', '3', '4'}
tlo_availability_df = tlo_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                                                on = ['Facility_ID'], how='left')

check = tlo_availability_df.groupby(['Facility_Level', 'item_code']).mean()
# amoxicillin 125 --- 0: 0.25758, 1a: 0.51568, 1b: 0.61787, 2: 0.85851, 3: 0.94966
# amoxicillin suspension 125mg/5ml - 70 --- 0: 0.12330, 1a: 0.24660, 1b: 0.31790, 2: 0.50819, 3: 0.77481

# gentamicin 106 --- 0: 0.44312, 1a: 0.88739, 1b: 0.86271, 2: 0.89192, 3: 0.96970
# ampicillin 86 --- 0: 0, 1a: 0, 1b: 0.71114, 2: 0.74023, 3: 0.95411
# benzylpen 99 ---- 0: 0.42456, 1a: 0.85039, 1b: 0.66246, 2: 0.91312, 3: 0.96970
# benzylpen 1g 2606 --- 0: 0.21851, 1a: 0.43703, 1b: 0.50399, 2: 0.31189, 3: 0.38913
# ceftriaxone 81 --- 0: 0.27986, 1a: 0.56095, 1b: 0.74610, 2: 0.89684, 3: 0.96621
# fluoxacillin 1831 --- 0: 0, 1a: 0, 1b: 0, 2: 0.32914, 3: 0.66691

# co-trimoxazole 203 ---- 0: 0.38546, 1a: 0.77091, 1b: 0.82631, 2: 0.73216, 3: 0.78459
# erythromycin 123 --- 0: 0.15414, 1a: 0.30827, 1b: 0.43940, 2: 0.59340, 3: 0.84973
# azithromycin --- 1a and 1b 0.5077 - assume 1a: 0.4577, 1b: 0.5577, half of 1a for 0: 0.22885, 2: 0.78655, 3: 0.900975

