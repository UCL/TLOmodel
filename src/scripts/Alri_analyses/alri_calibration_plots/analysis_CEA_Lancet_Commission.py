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

MODEL_POPSIZE = 15_000
MIN_SAMPLE_OF_NEW_CASES = 200
NUM_REPS_FOR_EACH_CASE = 20

facility_level_implementation = '2'  # <-- assumes that the diagnosis/treatment occurs at this level

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

# antibiotics -----------------
sim0_cp_ant = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ant = sim0_cp_ant.modules['Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ant)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ant = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ant, person_id=None)

# PO -----------------
sim0_cp_po = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po = sim0_cp_po.modules['Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po, person_id=None)

# Oxygen -----------------
sim0_cp_ox = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ox = sim0_cp_ox.modules['Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ox)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ox = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_ox, person_id=None)

# PO and Oxygen -----------------
sim0_cp_po_ox = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po_ox = sim0_cp_po_ox.modules['Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po_ox)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po_ox = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_current_policy_po_ox, person_id=None)


def generate_case_mix() -> pd.DataFrame:
    """Generate table of all the cases that may be created"""

    def get_incident_case_mix() -> pd.DataFrame:
        """Return a representative mix of new incidence alri cases (age, sex of person and the pathogen)."""

        alri_polling_event = AlriPollingEvent(module=alri_module_with_perfect_diagnosis_current_policy)

        # Get probabilities for each person of being infected with each pathogen
        probs_of_acquiring_pathogen = alri_polling_event.get_probs_of_acquiring_pathogen(
            interval_as_fraction_of_a_year=1.0)

        # Sample who is infected and with what pathogen & Repeat 10 times with replacement to generate larger numbers
        new_alri = []
        while len(new_alri) < MIN_SAMPLE_OF_NEW_CASES:
            new_alri.extend(
                [(k, v) for k, v in
                 sample_outcome(probs=probs_of_acquiring_pathogen, rng=alri_module_with_perfect_diagnosis_current_policy.rng).items()]
            )

        # Return dataframe in which person_id is replaced with age and sex (ignoring variation in vaccine /
        # under-nutrition).
        return pd.DataFrame(columns=['person_id', 'pathogen'], data=new_alri) \
            .merge(sim1.population.props[['age_exact_years', 'sex',
                                          'va_hib_all_doses', 'va_pneumo_all_doses', 'un_clinical_acute_malnutrition']],
                   right_index=True, left_on=['person_id'], how='left') \
            .drop(columns=['person_id'])

    def char_of_incident_case(sex,
                              age_exact_years,
                              pathogen,
                              va_hib_all_doses,
                              va_pneumo_all_doses,
                              un_clinical_acute_malnutrition,
                              ) -> dict:
        """Return the characteristics that are determined by IncidentCase (over 1000 iterations), given an infection
        caused by the pathogen"""
        incident_case = AlriIncidentCase(module=alri_module_with_perfect_diagnosis_current_policy,
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
            char_of_incident_case(sex=x.sex, age_exact_years=x.age_exact_years, pathogen=x.pathogen,
                                  va_hib_all_doses=x.va_hib_all_doses, va_pneumo_all_doses=x.va_pneumo_all_doses,
                                  un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition)
        )

    return pd.DataFrame(overall_case_mix)


def treatment_efficacy(
    age_exact_years,
    symptoms,
    oxygen_saturation,
    disease_type,
    complications,
    oximeter_available,
    oxygen_available,
    treatment_perfect,
    hw_dx_perfect,
    facility_level,
    new_policy,
    return_efficacy
):
    """Return the percentage by which the treatment reduce the risk of death"""

    # Decide which hsi configuration to use:

    # Decide which hsi configuration to use:
    if treatment_perfect:
        hsi = hsi_with_perfect_diagnosis_and_perfect_treatment_current_policy
        alri_module = alri_module_with_perfect_diagnosis_and_perfect_treatment_current_policy

    else:
        if hw_dx_perfect:
            hsi = hsi_with_perfect_diagnosis_current_policy
            alri_module = alri_module_with_perfect_diagnosis_current_policy
        else:
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
                raise ValueError('not using a sim above new pol')

    # Get Treatment classification
    classification_for_treatment_decision = hsi._get_disease_classification_for_treatment_decision(
        age_exact_years=age_exact_years, symptoms=symptoms, oxygen_saturation=oxygen_saturation,
        facility_level=facility_level, use_oximeter=oximeter_available, hiv_infected_and_not_on_art=False,
        un_clinical_acute_malnutrition='well')

    imci_symptom_based_classification = alri_module.get_imci_classification_based_on_symptoms(
        child_is_younger_than_2_months=(age_exact_years < 2.0 / 12.0),
        symptoms=symptoms, facility_level='2'
    )

    # Get the treatment selected based on classification given
    ultimate_treatment = alri_module._ultimate_treatment_indicated_for_patient(
        classification_for_treatment_decision=classification_for_treatment_decision,
        age_exact_years=age_exact_years,
        facility_level=facility_level,
        oxygen_saturation=oxygen_saturation,
    )

    # "Treatment Fails" is the probability that a death is averted (if one is schedule)
    treatment_fails = alri_module.models._prob_treatment_fails(
        antibiotic_provided=ultimate_treatment['antibiotic_indicated'][0],
        oxygen_provided=ultimate_treatment['oxygen_indicated'] if oxygen_available else False,
        imci_symptom_based_classification=imci_symptom_based_classification,
        SpO2_level=oxygen_saturation,
        disease_type=disease_type,
        age_exact_years=age_exact_years,
        symptoms=symptoms,
        complications=complications,
        hiv_infected_and_not_on_art=False,
        un_clinical_acute_malnutrition='well',
    )

    # for inpatients provide 2nd line IV antibiotic if 1st line failed
    if ultimate_treatment['antibiotic_indicated'][0].startswith('1st_line_IV'):
        treatment_fails = treatment_fails * (alri_module.models._prob_treatment_fails(
            antibiotic_provided='2nd_line_IV_flucloxacillin_gentamicin',
            oxygen_provided=ultimate_treatment['oxygen_indicated'] if oxygen_available else False,
            imci_symptom_based_classification=imci_symptom_based_classification,
            SpO2_level=oxygen_saturation,
            disease_type=disease_type,
            age_exact_years=age_exact_years,
            symptoms=symptoms,
            complications=complications,
            hiv_infected_and_not_on_art=False,
            un_clinical_acute_malnutrition='well',
        ))

    if return_efficacy:
        # Return percentage probability of treatment success
        return 100.0 * (1.0 - treatment_fails)

    else:
        return classification_for_treatment_decision

    # # Return percentage probability of treatment success
    # return 100.0 * (1.0 - treatment_fails)


def generate_table():
    """Return table providing a representative case mix of persons with Alri, the intrinsic risk of death and the
    efficacy of treatment under different conditions."""

    # Get Case Mix
    df = generate_case_mix()
    seek_level = list()
    for x in df.itertuples():
        seek_level.append({
            'seek_level': alri_module_with_perfect_diagnosis_current_policy.seek_care_level(
                symptoms=x.symptoms
            )})
    df = df.join(pd.DataFrame(seek_level))

    # Consider risk of death for this person, intrinsically and under different conditions of treatments
    risk_of_death = list()
    for x in df.itertuples():
        risk_of_death.append({
            'prob_die_if_no_treatment': alri_module_with_perfect_diagnosis_current_policy.models.prob_die_of_alri(
                age_exact_years=x.age_exact_years,
                sex=x.sex,
                bacterial_infection=x.pathogen,
                disease_type=x.disease_type,
                SpO2_level=x.oxygen_saturation,
                complications=x.complications,
                all_symptoms=x.symptoms,
                un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
            ),

            # All Perfect
            f'treatment_efficacy_if_perfect_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_current_pol':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=True,
                    oxygen_available=True,
                    treatment_perfect=True,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level,
                    new_policy=False,
                    return_efficacy=True
                ),

            # Treatment Efficacy with * PERFECT HW Diagnosis *
            f'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_current_pol':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=True,
                    oxygen_available=True,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level,
                    new_policy=False,
                    return_efficacy=True
                ),

            f'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx_current_pol':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=False,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level,
                    new_policy=False,
                    return_efficacy=True
                ),

            f'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx_current_pol':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=True,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level,
                    new_policy=False,
                    return_efficacy=True
                ),

            f'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx_current_pol':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=False,
                    oxygen_available=True,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level,
                    new_policy=False,
                    return_efficacy=True
                ),

            # Treatment Efficacy with * IMPERFECT HW Diangosis *
            f'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_imperfect_hw_dx_current_pol':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=True,
                    oxygen_available=True,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level,
                    new_policy=False,
                    return_efficacy=True
                ),

            f'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_imperfect_hw_dx_current_pol':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=False,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level,
                    new_policy=False,
                    return_efficacy=True
                ),

            f'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_imperfect_hw_dx_current_pol':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=True,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level,
                    new_policy=False,
                    return_efficacy=True
                ),

            f'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_imperfect_hw_dx_current_pol':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=False,
                    oxygen_available=True,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level,
                    new_policy=False,
                    return_efficacy=True
                ),


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

            # * CLASSIFICATION BY LEVEL THEY SOUGHT CARE *
            'classification_for_treatment_decision_with_oximeter_perfect_accuracy_sought_level':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True, hiv_infected_and_not_on_art=False,
                    un_clinical_acute_malnutrition='well'),

            'classification_for_treatment_decision_without_oximeter_perfect_accuracy_sought_level':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=False, hiv_infected_and_not_on_art=False,
                    un_clinical_acute_malnutrition='well'),
        })
    return df.join(pd.DataFrame(risk_of_death))


if __name__ == "__main__":
    table = generate_table()

    # YLD = I × DW × L
    daly_weight = table['classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2'].apply(
        lambda x: 0.133 if 'danger_signs_pneumonia' in x else 0.051)
    duration_years = table['duration_in_days_of_alri'] / 365.25

    YLD = table.index.size * (daly_weight * duration_years).mean()

    # YLL = N × L
    mean_age = table['age_exact_years'].mean()  # 1.5169246
    deaths = table['will_die'].sum()

    YLL = deaths * (63 - mean_age)

    # DALYS
    DALYs = YLD + YLL


    def summarize_by(df: pd.DataFrame, by: List[str], columns: [List[str]]) -> pd.DataFrame:
        """Helper function returns dataframe that summarizes the dataframe provided using groupby, with by arguements,
        and provides columns as follows: [size-of-the-group, mean-of-column-1, mean-of-column-2, ...]"""
        return pd.DataFrame({'fraction': df.groupby(by=by).size()}).apply(lambda x: x / x.sum(), axis=0) \
            .join(df.groupby(by=by)[columns].mean())

    # Average risk of death by disease type
    table.groupby(by=['disease_type'])['prob_die_if_no_treatment'].mean()

    # Look at diagnosis errors (assuming that the "truth" is
    # "classification_for_treatment_decision_with_oximeter_perfect_accuracy")
    truth = table["classification_for_treatment_decision_with_oximeter_perfect_accuracy_level2"]

    # Examine risk of death and treatment effectiveness by disease classification and oxygen saturation
    # And include estimate of % of deaths by scaling using "prob_die_if_no_treatment".
    # UNDER PERFECT HW DX
    risk_of_death = summarize_by(
        df=table,
        by=['oxygen_saturation', 'classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2'],
        columns=[
            'prob_die_if_no_treatment',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_current_pol',
            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx_current_pol',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx_current_pol',
            'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx_current_pol'
        ]
    ).assign(
        fraction_of_deaths=lambda df: (
            (df.fraction * df.prob_die_if_no_treatment) / (df.fraction * df.prob_die_if_no_treatment).sum()
        )
    )
    print(f"{risk_of_death=}")

    # # risk of deaths if no treatment
    # (risk_of_death.fraction * risk_of_death.prob_die_if_no_treatment).sum()  # 0.0342 -- 3.42%
    #
    # # risk of death with treatment with oximeter_and_oxygen_perfect_hw_dx
    # (risk_of_death.fraction * risk_of_death.prob_die_if_no_treatment * (
    #     (
    #         100 - risk_of_death.treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_current_pol) / 100)
    #  ).sum()  # 0.00604 -- 0.06% deaths

    # UNDER PERFECT HW DX
    risk_of_death_imperfect_dx = summarize_by(
        df=table,
        by=['oxygen_saturation', 'classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2'],
        columns=[
            'prob_die_if_no_treatment',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_imperfect_hw_dx_current_pol',
            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_imperfect_hw_dx_current_pol',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_imperfect_hw_dx_current_pol',
            'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_imperfect_hw_dx_current_pol'
        ]
    ).assign(
        fraction_of_deaths=lambda df: (
            (df.fraction * df.prob_die_if_no_treatment) / (df.fraction * df.prob_die_if_no_treatment).sum()
        )
    )
    print(f"{risk_of_death_imperfect_dx=}")

    # Overall summary figure: Number of deaths in the cohort Deaths broken down by ( disease / oxygen_saturation) when
    # * No Treatment
    # * Treatment antibiotics only (no oxygen, no oximeter)
    # * Treatment with antibiotics and oxygen (no oximeter)
    # * Treatment with antibiotics and oxygen and use of oximeter
    # ... under assumptions of (i) normal treatment and normal dx accuracy; (ii) normal treatment and perfect dx
    # accuracy

    disease_classification2 = table["classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2"]
    disease_classification1a = table["classification_for_treatment_decision_without_oximeter_perfect_accuracy_level1a"]
    classification_by_seek_level = table["classification_for_treatment_decision_without_oximeter_perfect_accuracy_sought_level"]

    low_oxygen = (table["oxygen_saturation"])
    number_cases = table.groupby(by=[disease_classification2, low_oxygen]).size()
    dx_accuracy = 'perfect'

    def cea_df_by_scenario(scenario, dx_accuracy):

        # create the series for the CEA dataframe
        low_oxygen = (table["oxygen_saturation"])
        classification_by_seek_level = table[f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"]
        facility_seeking = table['seek_level']
        number_cases_by_seek_level = table.groupby(by=[classification_by_seek_level, low_oxygen, facility_seeking]).size()
        cases_per_facility = table.groupby(by=[facility_seeking]).size()
        deaths_per_facility = (table['prob_die_if_no_treatment'] * (1.0 - table[
            f'treatment_efficacy_if_normal_treatment_{scenario}_{dx_accuracy}_hw_dx_current_pol'] /
                                                                    100.0)).groupby(by=[facility_seeking]).sum()

        # create the dataframe
        df_by_seek_level = pd.DataFrame([cases_per_facility, deaths_per_facility], index=['cases', 'deaths'])

        # number of cases needing inpatient care by severe signs (ds_pneumonia classification)
        n_inpatient_without_PO_level_0 = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '0'), ('danger_signs_pneumonia', '90-92%', '0'),
                                       ('danger_signs_pneumonia', '>=93%', '0')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()

        n_inpatient_without_PO_level_1a = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '1a'), ('danger_signs_pneumonia', '90-92%', '1a'),
                                       ('danger_signs_pneumonia', '>=93%', '1a')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()
        n_inpatient_without_PO_level_1b = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '1b'), ('danger_signs_pneumonia', '90-92%', '1b'),
                                       ('danger_signs_pneumonia', '>=93%', '1b')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()
        n_inpatient_without_PO_level_2 = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '2'), ('danger_signs_pneumonia', '90-92%', '2'),
                                       ('danger_signs_pneumonia', '>=93%', '2')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()

        # Add to dataframe - Need inpatient care based on severe signs
        df_by_seek_level = df_by_seek_level.append(
            pd.Series({'0': n_inpatient_without_PO_level_0, '1a': n_inpatient_without_PO_level_1a,
                       '1b': n_inpatient_without_PO_level_1b, '2': n_inpatient_without_PO_level_2},
                      name='Need inpatient care based on severe signs'))

        # number of cases needing inpatient care by severe signs + SpO2 level
        n_inpatient_ds_and_PO_level_0 = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '0'), ('danger_signs_pneumonia', '90-92%', '0'),
                                       ('danger_signs_pneumonia', '>=93%', '0'), ('chest_indrawing_pneumonia', '<90%', '0'),
                                       ('fast_breathing_pneumonia', '<90%', '0'), ('cough_or_cold', '<90%', '0')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()

        n_inpatient_ds_and_PO_level_1a = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '1a'), ('danger_signs_pneumonia', '90-92%', '1a'),
                                       ('danger_signs_pneumonia', '>=93%', '1a'),
                                       ('chest_indrawing_pneumonia', '<90%', '1a'),
                                       ('fast_breathing_pneumonia', '<90%', '1a'), ('cough_or_cold', '<90%', '1a')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()

        n_inpatient_ds_and_PO_level_1b = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '1b'), ('danger_signs_pneumonia', '90-92%', '1b'),
                                       ('danger_signs_pneumonia', '>=93%', '1b'),
                                       ('chest_indrawing_pneumonia', '<90%', '1b'),
                                       ('fast_breathing_pneumonia', '<90%', '1b'), ('cough_or_cold', '<90%', '1b')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()

        n_inpatient_ds_and_PO_level_2 = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '2'), ('danger_signs_pneumonia', '90-92%', '2'),
                                       ('danger_signs_pneumonia', '>=93%', '2'),
                                       ('chest_indrawing_pneumonia', '<90%', '2'),
                                       ('fast_breathing_pneumonia', '<90%', '2'), ('cough_or_cold', '<90%', '2')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()

        # update the dataframe - Need inpatient care based on severe signs and SpO2 <90%
        df_by_seek_level = df_by_seek_level.append(
            pd.Series({'0': n_inpatient_ds_and_PO_level_0, '1a': n_inpatient_ds_and_PO_level_1a,
                       '1b': n_inpatient_ds_and_PO_level_1b, '2': n_inpatient_ds_and_PO_level_2},
                      name='Need inpatient care based on severe signs and SpO2<90%'))

        # number of cases needing Oxygen based on severe pneumonia (ds) classification with SpO2 < 90% - Ox only scenario
        need_oxygen_ds_with_SpO2lt90_level_0 = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '0')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()
        need_oxygen_ds_with_SpO2lt90_level_1a = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '1a')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()
        need_oxygen_ds_with_SpO2lt90_level_1b = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '1b')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()
        need_oxygen_ds_with_SpO2lt90_level_2 = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '2')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()

        # update the dataframe - Need inpatient care based on severe signs and SpO2 <90%
        df_by_seek_level = df_by_seek_level.append(
            pd.Series({'0': need_oxygen_ds_with_SpO2lt90_level_0, '1a': need_oxygen_ds_with_SpO2lt90_level_1a,
                       '1b': need_oxygen_ds_with_SpO2lt90_level_1b, '2': need_oxygen_ds_with_SpO2lt90_level_2},
                      name='Need Oxygen - severe classification with SpO2<90%'))

        # number of cases needing Oxygen based on SpO2 < 90% - PO (+/- Ox) scenario
        need_oxygen_SpO2lt90_level_0 = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '0'),
                                       ('chest_indrawing_pneumonia', '<90%', '0'),
                                       ('fast_breathing_pneumonia', '<90%', '0'),
                                       ('cough_or_cold', '<90%', '0')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()
        need_oxygen_SpO2lt90_level_1a = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '1a'),
                                       ('chest_indrawing_pneumonia', '<90%', '1a'),
                                       ('fast_breathing_pneumonia', '<90%', '1a'),
                                       ('cough_or_cold', '<90%', '1a')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()
        need_oxygen_SpO2lt90_level_1b = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '1b'),
                                       ('chest_indrawing_pneumonia', '<90%', '1b'),
                                       ('fast_breathing_pneumonia', '<90%', '1b'),
                                       ('cough_or_cold', '<90%', '1b')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()
        need_oxygen_SpO2lt90_level_2 = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%', '2'),
                                       ('chest_indrawing_pneumonia', '<90%', '2'),
                                       ('fast_breathing_pneumonia', '<90%', '2'),
                                       ('cough_or_cold', '<90%', '2')],
                                      names=number_cases_by_seek_level.index.names), fill_value=0).sum()

        # update the dataframe - Need inpatient care based on SpO2 <90%
        df_by_seek_level = df_by_seek_level.append(
            pd.Series({'0': need_oxygen_SpO2lt90_level_0, '1a': need_oxygen_SpO2lt90_level_1a,
                       '1b': need_oxygen_SpO2lt90_level_1b, '2': need_oxygen_SpO2lt90_level_2},
                      name='Need Oxygen - any SpO2<90%'))

        # ------ add the cost of other health care factors -----------

        # ANTIBIOTICS COST --------
        oral_amox_by_age = table['age_exact_years'].apply(
            lambda x: (1 * 2 * 5 * 0.061) if x < 1 else (2 * 2 * 5 * 0.061) if 1 <= x < 2 else (3 * 2 * 5 * 0.061))

        iv_amp_by_age = table['age_exact_years'].apply(
            lambda x: (1/2.5 * 4 * 5 * 0.5212) if x < 1/3 else (2/2.5 * 4 * 5 * 0.5212) if 1/3 <= x < 1 else
            (3/2.5 * 4 * 5 * 0.5212) if 1 <= x < 3 else (5/2.5 * 4 * 5 * 0.5212))
        iv_gent_by_age = table['age_exact_years'].apply(
            lambda x: (1/2 * 4 * 5 * 0.2694) if x < 1/3 else (1.8/2 * 4 * 5 * 0.2694) if 1/3 <= x < 1 else
            (2.7/2 * 4 * 5 * 0.2694) if 1 <= x < 3 else (3.5/2 * 4 * 5 * 0.2694))

        iv_antibiotics = iv_amp_by_age + iv_gent_by_age

        df_oral_amox = oral_amox_by_age.to_frame(name='oral_amox')
        df_iv_ant = iv_antibiotics.to_frame(name='iv_antibiotics')

        new_table = table.join(df_oral_amox)
        new_table = new_table.join(df_iv_ant)

        # IV antibiotic cost without PO use
        sum_iv_ant_cost_without_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'iv_antibiotics'].sum()
        sum_iv_ant_cost_without_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'iv_antibiotics'].sum()
        sum_iv_ant_cost_without_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'iv_antibiotics'].sum()
        sum_iv_ant_cost_without_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'iv_antibiotics'].sum()
        # ------------------------
        # Oral antibiotic cost without PO use
        sum_oral_ant_cost_without_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'iv_antibiotics'].sum()
        sum_oral_ant_cost_without_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'iv_antibiotics'].sum()
        sum_oral_ant_cost_without_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'iv_antibiotics'].sum()
        sum_oral_ant_cost_without_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'iv_antibiotics'].sum()

        # introduce PO - increase IV antibiotic use ----
        sum_iv_ant_cost_with_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'iv_antibiotics'].sum()
        sum_iv_ant_cost_with_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'iv_antibiotics'].sum()
        sum_iv_ant_cost_with_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'iv_antibiotics'].sum()
        sum_iv_ant_cost_with_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'iv_antibiotics'].sum()
        # ------------------------
        # Oral antibiotic cost with PO use
        sum_oral_ant_cost_with_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'iv_antibiotics'].sum()
        sum_oral_ant_cost_with_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'iv_antibiotics'].sum()
        sum_oral_ant_cost_with_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'iv_antibiotics'].sum()
        sum_oral_ant_cost_with_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'iv_antibiotics'].sum()
        # --------------------------------------------------------------------------------------------

        # OUTPATIENT CONSULTATION COST ------------
        new_table['consultation_cost'] = new_table['seek_level'].apply(
            lambda x: 2.45 if x == '2' else 2.35 if x == '1b' else 2.06 if x == '1a' else 1.67
        )
        # INPATIENT BED/DAY COST ------------
        new_table['inpatient_bed_cost'] = new_table['seek_level'].apply(
            lambda x: 6.81*5 if x == '2' else 6.53*5 if x == '1b' else 6.53*5
        )

        # Outpatient consultation cost without PO
        sum_consultation_cost_without_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'consultation_cost'].sum()
        sum_consultation_cost_without_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'consultation_cost'].sum()
        sum_consultation_cost_without_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'consultation_cost'].sum()
        sum_consultation_cost_without_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'consultation_cost'].sum()

        # Outpatient consultation cost with PO
        sum_consultation_cost_with_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'consultation_cost'].sum()
        sum_consultation_cost_with_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'consultation_cost'].sum()
        sum_consultation_cost_with_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'consultation_cost'].sum()
        sum_consultation_cost_with_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'consultation_cost'].sum()

        # Inpatient hospitalisation cost without PO
        sum_hospitalisation_cost_without_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'inpatient_bed_cost'].sum()
        sum_hospitalisation_cost_without_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'inpatient_bed_cost'].sum()
        sum_hospitalisation_cost_without_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'inpatient_bed_cost'].sum()
        sum_hospitalisation_cost_without_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_without_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'inpatient_bed_cost'].sum()

        # Inpatient hospitalisation cost with PO
        sum_hospitalisation_cost_with_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'inpatient_bed_cost'].sum()
        sum_hospitalisation_cost_with_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'inpatient_bed_cost'].sum()
        sum_hospitalisation_cost_with_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'inpatient_bed_cost'].sum()
        sum_hospitalisation_cost_with_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_for_treatment_decision_with_oximeter_{dx_accuracy}_accuracy_sought_level"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'inpatient_bed_cost'].sum()

        # update the dataframe - without PO use
        if scenario in ('but_without_oximeter_or_oxygen', 'and_with_oxygen_but_without_oximeter'):
            # Oral amoxicillin cost without PO use
            df_by_seek_level = df_by_seek_level.append(
                pd.Series({'0': sum_oral_ant_cost_without_po_level_0, '1a': sum_oral_ant_cost_without_po_level_1a,
                           '1b': sum_oral_ant_cost_without_po_level_1b, '2': sum_oral_ant_cost_without_po_level_2},
                          name='Oral antibiotics w/out PO'))
            # IV antibiotics cost without PO use
            df_by_seek_level = df_by_seek_level.append(
                pd.Series({'0': sum_iv_ant_cost_without_po_level_0, '1a': sum_iv_ant_cost_without_po_level_1a,
                           '1b': sum_iv_ant_cost_without_po_level_1b, '2': sum_iv_ant_cost_without_po_level_2},
                          name='IV antibiotics w/out PO'))
            # consultation cost without PO use
            df_by_seek_level = df_by_seek_level.append(
                pd.Series({'0': sum_consultation_cost_without_po_level_0, '1a': sum_consultation_cost_without_po_level_1a,
                           '1b': sum_consultation_cost_without_po_level_1b, '2': sum_consultation_cost_without_po_level_2},
                          name='Consultations w/out PO'))
            # hospitalisation cost without PO use
            df_by_seek_level = df_by_seek_level.append(
                pd.Series({'0': sum_hospitalisation_cost_without_po_level_0, '1a': sum_hospitalisation_cost_without_po_level_1a,
                           '1b': sum_hospitalisation_cost_without_po_level_1b, '2': sum_hospitalisation_cost_without_po_level_2},
                          name='Inpatient bed w/out PO'))

        else:
            # update the dataframe - Oral amoxicillin cost with PO use
            df_by_seek_level = df_by_seek_level.append(
                pd.Series({'0': sum_oral_ant_cost_with_po_level_0, '1a': sum_oral_ant_cost_with_po_level_1a,
                           '1b': sum_oral_ant_cost_with_po_level_1b, '2': sum_oral_ant_cost_with_po_level_2},
                          name='Oral antibiotics w/ PO'))
            # update the dataframe - IV antibiotics cost with PO use
            df_by_seek_level = df_by_seek_level.append(
                pd.Series({'0': sum_iv_ant_cost_with_po_level_0, '1a': sum_iv_ant_cost_with_po_level_1a,
                           '1b': sum_iv_ant_cost_with_po_level_1b, '2': sum_iv_ant_cost_with_po_level_2},
                          name='IV antibiotics w/ PO'))
            # consultation cost with PO use
            df_by_seek_level = df_by_seek_level.append(
                pd.Series({'0': sum_consultation_cost_with_po_level_0, '1a': sum_consultation_cost_with_po_level_1a,
                           '1b': sum_consultation_cost_with_po_level_1b, '2': sum_consultation_cost_with_po_level_2},
                          name='Consultations w/out PO'))
            # hospitalisation cost with PO use
            df_by_seek_level = df_by_seek_level.append(
                pd.Series({'0': sum_hospitalisation_cost_with_po_level_0, '1a': sum_hospitalisation_cost_with_po_level_1a,
                           '1b': sum_hospitalisation_cost_with_po_level_1b, '2': sum_hospitalisation_cost_with_po_level_2},
                          name='Inpatient bed w/out PO'))

        return df_by_seek_level


    cea_df_ant = cea_df_by_scenario(scenario='but_without_oximeter_or_oxygen', dx_accuracy=dx_accuracy)
    cea_df_ox = cea_df_by_scenario(scenario='and_with_oxygen_but_without_oximeter', dx_accuracy=dx_accuracy)
    cea_df_po = cea_df_by_scenario(scenario='and_with_oximeter_but_without_oxygen', dx_accuracy=dx_accuracy)
    cea_df_po_ox = cea_df_by_scenario(scenario='and_with_oximeter_and_oxygen', dx_accuracy=dx_accuracy)

    # Plot graph
    low_oxygen = (table["oxygen_saturation"] == "<90%").replace({True: '<90%', False: ">=90%"})  # for Spo2<90% cutoff
    policy = 'current'
    res = {
        "Perfect HW IMCI Dx": {
            # "No Treatment": table['prob_die_if_no_treatment'].groupby(by=[disease_classification2, low_oxygen]).sum(),
            "Antiobiotics only": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx_{policy}_pol'] / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
            "+ oxygen": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx_{policy}_pol']
                 / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
            "+ oximeter": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx_{policy}_pol'] / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
            "+ oximeter & oxygen": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_{policy}_pol'] / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
        },

        "Normal HW IMCI Dx": {
            # "No Treatment": table['prob_die_if_no_treatment'].groupby(by=[disease_classification2, low_oxygen]).sum(),
            "Antiobiotics only": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_imperfect_hw_dx_{policy}_pol'] / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
            "+ oxygen": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_imperfect_hw_dx_{policy}_pol']
                 / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
            "+ oximeter": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_imperfect_hw_dx_{policy}_pol'] / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
            "+ oximeter & oxygen": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_imperfect_hw_dx_{policy}_pol'] / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
        },
    }

    results = (100_000 / len(table)) * pd.concat({k: pd.DataFrame(v) for k, v in res.items()}, axis=1)


    # reorder the index:
    reorderlist = list()
    if results.index.size == 12:  # broken down by 3 SpO2 levels
        # rename the index - to shorten the labels
        # results.index = pd.MultiIndex.from_tuples(
        #     [(x[0], f'SpO2_{x[1]}') if x[1] == '90-92%' else (x[0], x[1]) for x in results.index],
        #     names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('cough/cold', f'SpO2={x[1]}') if x == ('cough_or_cold', '90-92%') else (x[0], x[1]) for x in
             results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('cough/cold', f'SpO2{x[1]}') if x in (('cough_or_cold', '<90%'), ('cough_or_cold', '>=93%')) else (
                x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('fb-pneumonia', f'SpO2={x[1]}') if x == ('fast_breathing_pneumonia', '90-92%') else (x[0], x[1]) for x in
             results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('fb-pneumonia', f'SpO2{x[1]}') if x in (
                ('fast_breathing_pneumonia', '<90%'), ('fast_breathing_pneumonia', '>=93%')) else (x[0], x[1]) for x in
             results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ci-pneumonia', f'SpO2={x[1]}') if x == ('chest_indrawing_pneumonia', '90-92%') else (x[0], x[1]) for x in
             results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ci-pneumonia', f'SpO2{x[1]}') if x in (
                ('chest_indrawing_pneumonia', '<90%'), ('chest_indrawing_pneumonia', '>=93%')) else (x[0], x[1]) for x
             in
             results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ds-pneumonia', f'SpO2={x[1]}') if x == ('danger_signs_pneumonia', '90-92%') else (x[0], x[1]) for x in
             results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ds-pneumonia', f'SpO2{x[1]}') if x in (
                ('danger_signs_pneumonia', '<90%'), ('danger_signs_pneumonia', '>=93%')) else (x[0], x[1]) for x in
             results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))

        reorderlist = [('cough/cold', 'SpO2>=93%'),
                       ('cough/cold', 'SpO2=90-92%'),
                       ('cough/cold', 'SpO2<90%'),
                       ('fb-pneumonia', 'SpO2>=93%'),
                       ('fb-pneumonia', 'SpO2=90-92%'),
                       ('fb-pneumonia', 'SpO2<90%'),
                       ('ci-pneumonia', 'SpO2>=93%'),
                       ('ci-pneumonia', 'SpO2=90-92%'),
                       ('ci-pneumonia', 'SpO2<90%'),
                       ('ds-pneumonia', 'SpO2>=93%'),
                       ('ds-pneumonia', 'SpO2=90-92%'),
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
            [('fb-pneumonia', f'SpO2{x[1]}') if x[0] == 'fast_breathing_pneumonia' else (x[0], x[1]) for x in
             results.index],
            names=("classification_for_treatment_decision_with_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ci-pneumonia', f'SpO2{x[1]}') if x[0] == 'chest_indrawing_pneumonia' else (x[0], x[1]) for x in
             results.index],
            names=("classification_for_treatment_decision_with_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ds-pneumonia', f'SpO2{x[1]}') if x[0] == 'danger_signs_pneumonia' else (x[0], x[1]) for x in
             results.index],
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
            [('fb-pneumonia', f'SpO2{x[1]}') if x[0] == 'fast_breathing_pneumonia' else (x[0], x[1]) for x in
             results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ci-pneumonia', f'SpO2{x[1]}') if x[0] == 'chest_indrawing_pneumonia' else (x[0], x[1]) for x in
             results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ds-pneumonia', f'SpO2{x[1]}') if x[0] == 'danger_signs_pneumonia' else (x[0], x[1]) for x in
             results.index],
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
        assign_colors = {('cough/cold', 'SpO2>=93%'): '#fa8405',
                         ('cough/cold', 'SpO2=90-92%'): 'gold',
                         ('cough/cold', 'SpO2<90%'): 'navajowhite',
                         ('fb-pneumonia', 'SpO2>=93%'): 'navy',
                         ('fb-pneumonia', 'SpO2=90-92%'): 'royalblue',
                         ('fb-pneumonia', 'SpO2<90%'): 'deepskyblue',
                         ('ci-pneumonia', 'SpO2>=93%'): 'darkgreen',
                         ('ci-pneumonia', 'SpO2=90-92%'): 'seagreen',
                         ('ci-pneumonia', 'SpO2<90%'): 'mediumspringgreen',
                         ('ds-pneumonia', 'SpO2>=93%'): 'darkred',
                         ('ds-pneumonia', 'SpO2=90-92%'): 'orangered',
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
        assign_colors = {('cough/cold', 'SpO2>=93%'): '#fa8405',
                         ('cough/cold', 'SpO2=90-92%'): 'gold',
                         ('cough/cold', 'SpO2<90%'): 'navajowhite',
                         ('cough/cold', 'SpO2>=90%'): 'tab:orange',
                         ('fb-pneumonia', 'SpO2>=93%'): 'navy',
                         ('fb-pneumonia', 'SpO2=90-92%'): 'royalblue',
                         ('fb-pneumonia', 'SpO2<90%'): 'deepskyblue',
                         ('fb-pneumonia', 'SpO2>=90%'): 'mediumblue',
                         ('ci-pneumonia', 'SpO2>=93%'): 'darkgreen',
                         ('ci-pneumonia', 'SpO2=90-92%'): 'seagreen',
                         ('ci-pneumonia', 'SpO2<90%'): 'mediumspringgreen',
                         ('ci-pneumonia', 'SpO2>=90%'): 'tab:green',
                         ('ds-pneumonia', 'SpO2>=93%'): 'darkred',
                         ('ds-pneumonia', 'SpO2=90-92%'): 'orangered',
                         ('ds-pneumonia', 'SpO2<90%'): 'darksalmon',
                         ('ds-pneumonia', 'SpO2>=90%'): 'firebrick',
                         }

    else:
        raise ValueError(f'Index size not recognised {results.index.size}')

    # PLOTS!!!
    fig, axs = plt.subplots(ncols=2, nrows=1, sharey=True, constrained_layout=True)
    for i, ix in enumerate(results.columns.levels[0]):
        ax = axs[i]
        # results.loc[:, (ix, slice(None))].T.plot.bar(stacked=True, ax=ax, legend=False)
        results.loc[:, (ix, slice(None))].T.plot.bar(stacked=True, ax=ax, width=0.6, legend=False,
                                                     color=results.index.map(assign_colors))
        ax.set_xticklabels(results.loc[:, (ix, slice(None))].columns.levels[1])
        ax.set_ylabel('Deaths per 100,000 cases of ALRI')
        ax.set_title(f"{ix}", fontsize=10)
        ax.grid(axis='y')
    handles, labels = axs[-1].get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), title='Case Type', loc='upper left', bbox_to_anchor=(1, 1),
              fontsize=7)
    # fig.suptitle('Deaths Under Different Interventions Combinations', fontsize=14, fontweight='semibold')
    fig.suptitle('Under current policy', fontsize=12, fontweight='semibold')
    fig.show()
    # fig.savefig(Path('./outputs') / ('imperfect dx - hosp - current vs new policy' + datestamp + ".pdf"), format='pdf')
    plt.close(fig)

    # # # # Calculations # # # #
    # compare between policy
    # base comparators - current policy

    # oxygen only under *current* policy --- CFR SAME FOR EITHER POLICY
    deaths_ox_only_current_pol = (table['prob_die_if_no_treatment'] * (
        1.0 - table[
        f'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_{dx_accuracy}_hw_dx_current_pol'] / 100.0)
                                  ).groupby(by=[disease_classification2, low_oxygen]).sum()
    sum_deaths_ox_only_current_pol = deaths_ox_only_current_pol.sum()

    # antibiotics only under *current* policy --- CFR SAME FOR EITHER POLICY
    deaths_antibiotics_only_current_pol = (table['prob_die_if_no_treatment'] * (
        1.0 - table[
        f'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_{dx_accuracy}_hw_dx_current_pol'] / 100.0)
                                           ).groupby(by=[disease_classification2, low_oxygen]).sum()
    sum_deaths_antibiotics_only_current_pol = deaths_antibiotics_only_current_pol.sum()

    # pulse oximeter only under *current* policy
    deaths_po_only_current_pol = (table['prob_die_if_no_treatment'] * (
        1.0 - table[
        f'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_{dx_accuracy}_hw_dx_current_pol'] / 100.0)
                                  ).groupby(by=[disease_classification2, low_oxygen]).sum()
    sum_deaths_po_only_current_pol = deaths_po_only_current_pol.sum()

    # oxygen and pulse ox under *current* policy
    deaths_po_and_ox_current_pol = (table['prob_die_if_no_treatment'] * (
        1.0 - table[
        f'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_{dx_accuracy}_hw_dx_current_pol'] / 100.0)
                                    ).groupby(by=[disease_classification2, low_oxygen]).sum()
    sum_deaths_po_and_ox_current_pol = deaths_po_and_ox_current_pol.sum()

    # make table for chapter 5
    cfr_antibiotics = (deaths_antibiotics_only_current_pol / number_cases) * 100
    cfr_df = cfr_antibiotics.to_frame(name='antibiotics')
    cfr_df['oxygen_only'] = (deaths_ox_only_current_pol / number_cases) * 100
    cfr_df['po_only/current_pol'] = (deaths_po_only_current_pol / number_cases) * 100
    cfr_df['po_only/current_pol'] = (deaths_po_only_current_pol / number_cases) * 100
    cfr_df['po_and_ox/current_pol'] = (deaths_po_and_ox_current_pol / number_cases) * 100

    # no treatment cfr
    tot_deaths_no_treatment = table['prob_die_if_no_treatment'].groupby(
        by=[disease_classification2, low_oxygen]).sum().sum() / number_cases.sum()  # 0.11298


# -------------------------------------------------------------------------------------------------------------
# CEA -----

# CURRENT POLICY ---------------
total_n_inpatient_without_PO = number_cases.sum(level=[0, 1]).reindex(
    pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%'), ('danger_signs_pneumonia', '90-92%'),
                               ('danger_signs_pneumonia', '>=93%')],
                              names=number_cases.index.names), fill_value=0).sum()

total_eligible_for_Ox_without_PO = number_cases.sum(level=[0, 1]).reindex(
    pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%'), ('danger_signs_pneumonia', '90-92%'),
                               ('danger_signs_pneumonia', '>=93%')],
                              names=number_cases.index.names), fill_value=0).sum()

total_n_inpatient_with_PO = number_cases.sum(level=[0, 1]).reindex(
    pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%'), ('danger_signs_pneumonia', '90-92%'),
                               ('danger_signs_pneumonia', '>=93%'),
                               ('chest_indrawing_pneumonia', '<90%'), ('fast_breathing_pneumonia', '<90%'),
                               ('cough_or_cold', '<90%')],
                              names=number_cases.index.names), fill_value=0).sum()

total_eligible_for_Ox_with_PO = number_cases.sum(level=[0, 1]).reindex(
    pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%'),
                               ('chest_indrawing_pneumonia', '<90%'), ('fast_breathing_pneumonia', '<90%'),
                               ('cough_or_cold', '<90%')],
                              names=number_cases.index.names), fill_value=0).sum()


# CALCULATE DALYS AT THE INDIVIDUAL LEVEL ---------------------------------
def f_DALY(K, C=0.16243, r=None, beta=None, a_death=None, a_disability=None, YLL_L=None, D=None):
    """ CALCULATE DALYS FOR AN INDIVIDUAL - Calculates disability-adjusted life years for an individual
    Returns a list of Years of Life Lost (discounted), Years Lived in Disease, and total DALYs #

    :param K: Age weighting modulation factor (1=use age weighting, 0=no age weighting)
    :param C: contant (default = 0.16243)
    :param r: discount rate (between 0-1 --- use 0.03)
    :param beta: parameter of age weighting function (between 0-1 ---- use 0.04)
    :param a_death: Age of premature death due to disease (in years)
    :param a_disability: Age of disease onset (in years)
    :param YLL_L: Life expectancy at age of death (in years)
    :param D: Disability weight (between 0-1)
    :return:
    """

    # calculate YLD_L
    YLD_L = a_death - a_disability

    if r == 0 and K != 0:
        YLL = ((1 - K) * YLL_L) + (((K * C * np.exp(-beta * a_death)) / (beta ** 2)) * (np.exp(-beta * YLL_L) * ((-beta) * (YLL_L + a_death) - 1) - (-beta * a_death - 1)))
        YLL_discounted = YLL
        YLD = D * (((1 - K) * YLD_L) + (((K * C * np.exp(-beta * a_disability)) / (beta ** 2)) * (np.exp(-beta * YLD_L) * ((-beta) * (YLD_L + a_disability) - 1) - (-beta * a_disability - 1))))
        DALY_total = YLL_discounted + YLD

    elif r == 0 and K == 0:
        YLL = ((1 - K) / 0.00000001) * (1 - np.exp(-0.00000001 * YLL_L))
        YLD = D * (((1 - K) / 0.00000001) * (1 - np.exp(-0.00000001 * YLD_L)))
        s = a_death - a_disability
        YLL_discounted = YLL * np.exp(-(0.00000001 * s))
        DALY_total = YLL_discounted + YLD

    elif r != 0 and K == 0:
        YLL = ((1 - K) / r) * (1 - np.exp(-r * YLL_L))
        YLD = D * (((1 - K) / r) * (1 - np.exp(-r * YLD_L)))
        s = a_death - a_disability
        YLL_discounted = YLL * np.exp(-(r * s))
        DALY_total = YLL_discounted + YLD

    elif r != 0 and K != 0:
        YLL = ((1 - K) / r) * (1 - np.exp(-r * YLL_L)) + (((K * C * np.exp(r * a_death)) / ((r + beta) ** 2)) * (np.exp(-(r + beta) * (YLL_L + a_death)) * (-(r + beta) * (YLL_L + a_death) - 1) - np.exp(-(r + beta) * a_death) * (-(r + beta) * a_death - 1)))
        YLD = D * (((1 - K) / r) * (1 - np.exp(-r * YLD_L)) + (((K * C * np.exp(r * a_disability)) / ((r + beta) ** 2)) * (np.exp(-(r + beta) * (YLD_L + a_disability)) * (-(r + beta) * (YLD_L + a_disability) - 1) - np.exp(-(r + beta) * a_disability) * (-(r + beta) * a_disability - 1))))
        s = a_death - a_disability
        YLL_discounted = YLL * np.exp(-(r * s))
        DALY_total = YLL_discounted + YLD

    Amount = [YLL_discounted, YLD, DALY_total]
    return Amount
