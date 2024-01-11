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

# Baseline scenario with antibiotics only --------------------
sim_cp_baseline_ant = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant = sim_cp_baseline_ant.modules['Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant, person_id=None)

# Baseline scenario + PO at increasing facility levels
# 2
sim_cp_baseline_ant_po_2 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_2 = sim_cp_baseline_ant_po_2.modules[
    'Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_2)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_2 = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_2, person_id=None)
# 1b
sim_cp_baseline_ant_po_1b = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_1b = sim_cp_baseline_ant_po_1b.modules[
    'Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_1b)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_1b = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_1b, person_id=None)
# 1a
sim_cp_baseline_ant_po_1a = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_1a = sim_cp_baseline_ant_po_1a.modules[
    'Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_1a)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_1a = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_1a, person_id=None)
# 0
sim_cp_baseline_ant_po_0 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_0 = sim_cp_baseline_ant_po_0.modules[
    'Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_0)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_0 = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_0, person_id=None)


# Existing PSA scenario -----------------
sim_cp_existing_psa = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa = sim_cp_existing_psa.modules['Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa, person_id=None)

# Existing PSA + PO at increasing facility levels
# 2
sim_cp_existing_psa_po_2 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_2 = sim_cp_existing_psa_po_2.modules[
    'Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_2)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_2 = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_2, person_id=None)
# 1b
sim_cp_existing_psa_po_1b = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_1b = sim_cp_existing_psa_po_1b.modules[
    'Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_1b)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_1b = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_1b, person_id=None)
# 1a
sim_cp_existing_psa_po_1a = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_1a = sim_cp_existing_psa_po_1a.modules[
    'Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_1a)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_1a = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_1a, person_id=None)
# 0
sim_cp_existing_psa_po_0 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_0 = sim_cp_existing_psa_po_0.modules[
    'Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_0)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_0 = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_0, person_id=None)

# Planned PSA scenario ----------------
sim_cp_planned_psa = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa = sim_cp_planned_psa.modules['Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa, person_id=None)

# Planned PSA + PO at increasing facility levels
# 2
sim_cp_planned_psa_po_2 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_2 = sim_cp_planned_psa_po_2.modules[
    'Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_2)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_2 = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_2, person_id=None)
# 1b
sim_cp_planned_psa_po_1b = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_1b = sim_cp_planned_psa_po_1b.modules[
    'Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_1b)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_1b = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_1b, person_id=None)
# 1a
sim_cp_planned_psa_po_1a = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_1a = sim_cp_planned_psa_po_1a.modules[
    'Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_1a)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_1a = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_1a, person_id=None)
# 0
sim_cp_planned_psa_po_0 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_0 = sim_cp_planned_psa_po_0.modules[
    'Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_0)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_0 = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_0, person_id=None)

# All district PSA scenario -----------------
sim_cp_all_district_psa = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa = sim_cp_all_district_psa.modules[
    'Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa, person_id=None)

# All District PSA + PO at increasing facility levels
# 2
sim_cp_all_district_psa_po_2 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_2 = \
sim_cp_all_district_psa_po_2.modules['Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_2)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_2 = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_2, person_id=None)
# 1b
sim_cp_all_district_psa_po_1b = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_1b = \
sim_cp_all_district_psa_po_1b.modules['Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_1b)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_1b = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_1b, person_id=None)
# 1a
sim_cp_all_district_psa_po_1a = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_1a = \
sim_cp_all_district_psa_po_1a.modules['Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_1a)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_1a = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_1a, person_id=None)
# 0
sim_cp_all_district_psa_po_0 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_0 = \
sim_cp_all_district_psa_po_0.modules['Alri']
_set_current_policy(alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_0)
hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_0 = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_0, person_id=None)


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
                 sample_outcome(probs=probs_of_acquiring_pathogen,
                                rng=alri_module_with_perfect_diagnosis_current_policy.rng).items()]
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


def configuration_to_use(treatment_perfect, hw_dx_perfect, scenario):
    """ Use the simulations based on arguments of perfect treatment, perfect hw dx, and scenario """

    # Decide which hsi configuration to use:
    if treatment_perfect:
        hsi = hsi_with_perfect_diagnosis_and_perfect_treatment_current_policy
        alri_module = alri_module_with_perfect_diagnosis_and_perfect_treatment_current_policy

    else:
        if hw_dx_perfect:
            hsi = hsi_with_perfect_diagnosis_current_policy
            alri_module = alri_module_with_perfect_diagnosis_current_policy
        else:
            if scenario == 'existing_psa':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa
            elif scenario == 'existing_psa_with_po_level2':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_2
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_2
            elif scenario == 'existing_psa_with_po_level1b':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_1b
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_1b
            elif scenario == 'existing_psa_with_po_level1a':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_1a
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_1a
            elif scenario == 'existing_psa_with_po_level0':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_0
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_0

            elif scenario == 'planned_psa':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa
            elif scenario == 'planned_psa_with_po_level2':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_2
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_2
            elif scenario == 'planned_psa_with_po_level1b':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_1b
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_1b
            elif scenario == 'planned_psa_with_po_level1a':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_1a
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_1a
            elif scenario == 'planned_psa_with_po_level0':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_0
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_0

            elif scenario == 'all_district_psa':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa
            elif scenario == 'all_district_psa_with_po_level2':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_2
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_2
            elif scenario == 'all_district_psa_with_po_level1b':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_1b
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_1b
            elif scenario == 'all_district_psa_with_po_level1a':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_1a
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_1a
            elif scenario == 'all_district_psa_with_po_level0':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_0
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_0

            elif scenario == 'baseline_ant':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant
            elif scenario == 'baseline_ant_with_po_level2':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_2
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_2
            elif scenario == 'baseline_ant_with_po_level1b':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_1b
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_1b
            elif scenario == 'baseline_ant_with_po_level1a':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_1a
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_1a
            elif scenario == 'baseline_ant_with_po_level0':
                hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_0
                alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment_cp_baseline_ant_po_0
            else:
                raise ValueError('not using a sim above')

    return alri_module, hsi


def treatment_efficacy(
    age_exact_years,
    symptoms,
    oxygen_saturation,
    disease_type,
    complications,
    un_clinical_acute_malnutrition,
    hiv_infected_and_not_on_art,
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
                                  scenario=scenario)
    alri_module = config[0]
    hsi = config[1]

    # availability of oxygen
    oxygen_available = None
    if scenario.startswith('existing'):
        oxygen_available = alri_module.models.coverage_of_oxygen(scenario='existing')
    elif scenario.startswith('planned'):
        oxygen_available = alri_module.models.coverage_of_oxygen(scenario='planned')
    elif scenario.startswith('all_district'):
        oxygen_available = alri_module.models.coverage_of_oxygen(scenario='all_district')
    elif scenario.startswith('baseline_ant'):
        oxygen_available = [False, False, False]

    oxygen_available_by_level = {'2': oxygen_available[0], '1b': oxygen_available[1], '1a': oxygen_available[2],
                                 '0': oxygen_available[2]}

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

    if facility_level in ('1a', '0') and (oxygen_saturation == '<90%'):
        if oxygen_available_by_level['1a']:
            pre_referral_oxygen = 'provided'
        else:
            pre_referral_oxygen = 'not_provided'
    else:
        pre_referral_oxygen = 'not_applicable'

    # change facility_level if there was a referral
    needs_referral = classification_for_treatment_decision == 'danger_signs_pneumonia' and facility_level in ('1a', '0')
    facility_level = alri_module.referral_from_hc(needs_referral=needs_referral, facility_level=facility_level)  #todo: change this needs_referral to classification

    # "Treatment Fails" is the probability that a death is averted (if one is schedule)
    treatment_fails = alri_module.models._prob_treatment_fails(
        antibiotic_provided=ultimate_treatment['antibiotic_indicated'][0],
        oxygen_provided=ultimate_treatment['oxygen_indicated'] if oxygen_available_by_level[facility_level] else False,
        imci_symptom_based_classification=imci_symptom_based_classification,
        SpO2_level=oxygen_saturation,
        disease_type=disease_type,
        age_exact_years=age_exact_years,
        symptoms=symptoms,
        complications=complications,
        hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
        un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
        pre_referral_oxygen=pre_referral_oxygen
    )

    # for inpatients provide 2nd line IV antibiotic if 1st line failed
    if ultimate_treatment['antibiotic_indicated'][0].startswith('1st_line_IV'):
        treatment_fails = treatment_fails * (alri_module.models._prob_treatment_fails(
            antibiotic_provided='2nd_line_IV_flucloxacillin_gentamicin',
            oxygen_provided=ultimate_treatment['oxygen_indicated'] if oxygen_available_by_level[
                facility_level] else False,
            imci_symptom_based_classification=imci_symptom_based_classification,
            SpO2_level=oxygen_saturation,
            disease_type=disease_type,
            age_exact_years=age_exact_years,
            symptoms=symptoms,
            complications=complications,
            hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
            un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
            pre_referral_oxygen=pre_referral_oxygen
        ))

    # Return percentage probability of treatment success
    return 100.0 * (1.0 - treatment_fails), ultimate_treatment['oxygen_indicated'] if oxygen_available_by_level[facility_level] else False

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

    hiv_not_on_art = list()
    for x in df.itertuples():
        hiv_not_on_art.append({
            'hiv_not_on_art': (x.hv_inf and x.hv_art != "on_VL_suppressed")})
    df = df.join(pd.DataFrame(hiv_not_on_art))

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

            # Treatment Efficacy with * PERFECT HW Diagnosis *

            # Baseline Antibiotics only no PO
            f'treatment_efficacy_scenario_baseline_ant_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='baseline_ant',
                    oximeter_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'treatment_efficacy_scenario_baseline_ant_with_po_level2_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='baseline_ant_with_po_level2',
                    oximeter_available=True if x.seek_level == '2' else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'treatment_efficacy_scenario_baseline_ant_with_po_level1b_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='baseline_ant_with_po_level1b',
                    oximeter_available=True if x.seek_level in ('2', '1b') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'treatment_efficacy_scenario_baseline_ant_with_po_level1a_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='baseline_ant_with_po_level1a',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'treatment_efficacy_scenario_baseline_ant_with_po_level0_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='baseline_ant_with_po_level0',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a', '0') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],

            # Existing PSA, no PO
            f'treatment_efficacy_scenario_existing_psa_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='existing',
                    oximeter_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],

            f'oxygen_provided_scenario_existing_psa_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='existing',
                    oximeter_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[1],

            # Planned PSA, no PO
            f'treatment_efficacy_scenario_planned_psa_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='planned',
                    oximeter_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'oxygen_provided_scenario_planned_psa_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='planned',
                    oximeter_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[1],

            # All District PSA, no PO
            f'treatment_efficacy_scenario_all_district_psa_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='all_district',
                    oximeter_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'oxygen_provided_scenario_all_district_psa_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='all_district',
                    oximeter_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[1],

            # Existing PSA, with PO at level 2
            f'treatment_efficacy_scenario_existing_psa_with_po_level2_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='existing_psa_with_po_level2',
                    oximeter_available=True if x.seek_level == '2' else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'oxygen_provided_scenario_existing_psa_with_po_level2_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='existing_psa_with_po_level2',
                    oximeter_available=True if x.seek_level == '2' else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[1],
            # Existing PSA, with PO at level 2, 1b
            f'treatment_efficacy_scenario_existing_psa_with_po_level1b_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='existing_psa_with_po_level1b',
                    oximeter_available=True if x.seek_level in ('2', '1b') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'oxygen_provided_scenario_existing_psa_with_po_level1b_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='existing_psa_with_po_level1b',
                    oximeter_available=True if x.seek_level in ('2', '1b') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[1],
            # Existing PSA, with PO at level 2, 1b, 1a
            f'treatment_efficacy_scenario_existing_psa_with_po_level1a_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='existing_psa_with_po_level1a',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'oxygen_provided_scenario_existing_psa_with_po_level1a_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='existing_psa_with_po_level1a',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[1],
            # Existing PSA, with PO at level 2, 1b, 1a, 0
            f'treatment_efficacy_scenario_existing_psa_with_po_level0_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='existing_psa_with_po_level0',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a', '0') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'oxygen_provided_scenario_existing_psa_with_po_level0_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='existing_psa_with_po_level0',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a', '0') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[1],

            # Planned PSA, with PO at level 2
            f'treatment_efficacy_scenario_planned_psa_with_po_level2_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='planned_psa_with_po_level2',
                    oximeter_available=True if x.seek_level == '2' else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'oxygen_provided_scenario_planned_psa_with_po_level2_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='planned_psa_with_po_level2',
                    oximeter_available=True if x.seek_level == '2' else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[1],

            # Planned PSA, with PO at level 2, 1b
            f'treatment_efficacy_scenario_planned_psa_with_po_level1b_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='planned_psa_with_po_level1b',
                    oximeter_available=True if x.seek_level in ('2', '1b') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'oxygen_provided_scenario_planned_psa_with_po_level1b_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='planned_psa_with_po_level1b',
                    oximeter_available=True if x.seek_level in ('2', '1b') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[1],
            # Planned PSA, with PO at level 2, 1b, 1a
            f'treatment_efficacy_scenario_planned_psa_with_po_level1a_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='planned_psa_with_po_level1a',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'oxygen_provided_scenario_planned_psa_with_po_level1a_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='planned_psa_with_po_level1a',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[1],
            # Planned PSA, with PO at level 2, 1b, 1a, 0
            f'treatment_efficacy_scenario_planned_psa_with_po_level0_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='planned_psa_with_po_level0',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a', '0') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'oxygen_provided_scenario_planned_psa_with_po_level0_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='planned_psa_with_po_level0',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a', '0') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[1],

            # All Districts, with PO at level 2
            f'treatment_efficacy_scenario_all_district_psa_with_po_level2_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='all_district_psa_with_po_level2',
                    oximeter_available=True if x.seek_level == '2' else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'oxygen_provided_scenario_all_district_psa_with_po_level2_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='all_district_psa_with_po_level2',
                    oximeter_available=True if x.seek_level == '2' else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[1],
            # All Districts PSA, with PO at level 2, 1b
            f'treatment_efficacy_scenario_all_district_psa_with_po_level1b_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='all_district_psa_with_po_level1b',
                    oximeter_available=True if x.seek_level in ('2', '1b') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'oxygen_provided_scenario_all_district_psa_with_po_level1b_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='all_district_psa_with_po_level1b',
                    oximeter_available=True if x.seek_level in ('2', '1b') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[1],
            # All Districts PSA, with PO at level 2, 1b, 1a
            f'treatment_efficacy_scenario_all_district_psa_with_po_level1a_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='all_district_psa_with_po_level1a',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'oxygen_provided_scenario_all_district_psa_with_po_level1a_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='all_district_psa_with_po_level1a',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[1],
            # All Districts PSA, with PO at level 2, 1b, 1a, 0
            f'treatment_efficacy_scenario_all_district_psa_with_po_level0_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='all_district_psa_with_po_level0',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a', '0') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],
            f'oxygen_provided_scenario_all_district_psa_with_po_level0_perfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='all_district_psa_with_po_level0',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a', '0') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[1],

            # Treatment Efficacy with * IMPERFECT HW Diangosis *
            # Existing PSA, no PO
            f'treatment_efficacy_scenario_existing_psa_imperfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='existing_psa',
                    oximeter_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level
                )[0],

            # Planned PSA, no PO
            f'treatment_efficacy_scenario_planned_psa_imperfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='planned_psa',
                    oximeter_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level
                )[0],

            # All District PSA, no PO
            f'treatment_efficacy_scenario_all_district_psa_imperfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='all_district_psa',
                    oximeter_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level=x.seek_level
                )[0],

            # Existing PSA, with PO at level 2
            f'treatment_efficacy_scenario_existing_psa_with_po_level2_imperfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='existing_psa_with_po_level2',
                    oximeter_available=True if x.seek_level == '2' else False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level
                )[0],
            # Existing PSA, with PO at level 2, 1b
            f'treatment_efficacy_scenario_existing_psa_with_po_level1b_imperfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='existing_psa_with_po_level1b',
                    oximeter_available=True if x.seek_level in ('2', '1b') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level
                )[0],
            # Existing PSA, with PO at level 2, 1b, 1a
            f'treatment_efficacy_scenario_existing_psa_with_po_level1a_imperfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='existing_psa_with_po_level1a',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level
                )[0],
            # Existing PSA, with PO at level 2, 1b, 1a, 0
            f'treatment_efficacy_scenario_existing_psa_with_po_level0_imperfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='existing_psa_with_po_level0',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a', '0') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level
                )[0],

            # Planned PSA, with PO at level 2
            f'treatment_efficacy_scenario_planned_psa_with_po_level2_imperfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='planned_psa_with_po_level2',
                    oximeter_available=True if x.seek_level == '2' else False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level
                )[0],

            # Planned PSA, with PO at level 2, 1b
            f'treatment_efficacy_scenario_planned_psa_with_po_level1b_imperfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='planned_psa_with_po_level1b',
                    oximeter_available=True if x.seek_level in ('2', '1b') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level
                )[0],
            # Planned PSA, with PO at level 2, 1b, 1a
            f'treatment_efficacy_scenario_planned_psa_with_po_level1a_imperfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='planned_psa_with_po_level1a',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level
                )[0],
            # Planned PSA, with PO at level 2, 1b, 1a, 0
            f'treatment_efficacy_scenario_planned_psa_with_po_level0_imperfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='planned_psa_with_po_level0',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a', '0') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level
                )[0],

            # All Districts, with PO at level 2
            f'treatment_efficacy_scenario_all_district_psa_with_po_level2_imperfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='all_district_psa_with_po_level2',
                    oximeter_available=True if x.seek_level == '2' else False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level
                )[0],

            # All Districts PSA, with PO at level 2, 1b
            f'treatment_efficacy_scenario_all_district_psa_with_po_level1b_imperfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='all_district_psa_with_po_level1b',
                    oximeter_available=True if x.seek_level in ('2', '1b') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level
                )[0],

            # All Districts PSA, with PO at level 2, 1b, 1a
            f'treatment_efficacy_scenario_all_district_psa_with_po_level1a_imperfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='all_district_psa_with_po_level1a',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level
                )[0],

            # All Districts PSA, with PO at level 2, 1b, 1a, 0
            f'treatment_efficacy_scenario_all_district_psa_with_po_level0_imperfect_hw_dx':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
                    # Information about the care that can be provided:
                    scenario='all_district_psa_with_po_level0',
                    oximeter_available=True if x.seek_level in ('2', '1b', '1a', '0') else False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level=x.seek_level
                )[0],

            # CLASSIFICATION BY PO AVAILABILITY PER SCENARIO
            f'classification_in_scenarios_without_po_perfect_hw_dx':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=False, hiv_infected_and_not_on_art=False,
                    un_clinical_acute_malnutrition='well'),
            # imperfect HW Dx scenarios ---------------------------------------------------
            f'classification_in_existing_psa_without_po_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=False, hiv_infected_and_not_on_art=False,
                    un_clinical_acute_malnutrition='well'),
            f'classification_in_planned_psa_without_po_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=False, hiv_infected_and_not_on_art=False,
                    un_clinical_acute_malnutrition='well'),
            f'classification_in_all_district_psa_without_po_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=False, hiv_infected_and_not_on_art=False,
                    un_clinical_acute_malnutrition='well'),

            # scenarios with PO at facility levels ----------------------------------------------------------
            f'classification_in_scenarios_with_po_level2_perfect_hw_dx':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level == '2' else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),
            f'classification_in_scenarios_with_po_level1b_perfect_hw_dx':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level in ('2', '1b') else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),
            f'classification_in_scenarios_with_po_level1a_perfect_hw_dx':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level in ('2', '1b', '1a') else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),
            f'classification_in_scenarios_with_po_level0_perfect_hw_dx':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level in ('2', '1b', '1a', '0') else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),

            # Imperfect HW Dx - Existing PSA
            f'classification_in_existing_psa_with_po_level2_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_2._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level == '2' else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),
            f'classification_in_existing_psa_with_po_level1b_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_1b._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level in ('2', '1b') else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),
            f'classification_in_existing_psa_with_po_level1a_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_1a._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level in ('2', '1b', '1a') else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),
            f'classification_in_existing_psa_with_po_level0_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_existing_psa_po_0._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level in ('2', '1b', '1a', '0') else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),

            # Imperfect HW Dx - Planned PSA
            f'classification_in_planned_psa_with_po_level2_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_2._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level == '2' else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),
            f'classification_in_planned_psa_with_po_level1b_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_1b._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level in ('2', '1b') else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),
            f'classification_in_planned_psa_with_po_level1a_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_1a._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level in ('2', '1b', '1a') else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),
            f'classification_in_planned_psa_with_po_level0_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_planned_psa_po_0._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level in ('2', '1b', '1a', '0') else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),

            # Imperfect HW Dx - All District PSA
            f'classification_in_all_district_psa_with_po_level2_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_2._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level == '2' else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),
            f'classification_in_all_district_psa_with_po_level1b_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_1b._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level in ('2', '1b') else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),
            f'classification_in_all_district_psa_with_po_level1a_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_1a._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level in ('2', '1b', '1a') else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),
            f'classification_in_all_district_psa_with_po_level0_imperfect_hw_dx':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment_cp_all_district_psa_po_0._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=x.seek_level, use_oximeter=True if x.seek_level in ('2', '1b', '1a', '0') else False,
                    hiv_infected_and_not_on_art=False, un_clinical_acute_malnutrition='well'),

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


    def cea_df_by_scenario(scenario, dx_accuracy):

        # create the series for the CEA dataframe
        low_oxygen = (table["oxygen_saturation"])
        classification_by_seek_level = table[f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"]
        facility_seeking = table['seek_level']
        number_cases_by_seek_level = table.groupby(
            by=[classification_by_seek_level, low_oxygen, facility_seeking]).size()
        cases_per_facility = table.groupby(by=[facility_seeking]).size()
        deaths_per_facility = (table['prob_die_if_no_treatment'] * (1.0 - table[
            f'treatment_efficacy_scenario_{scenario}_{dx_accuracy}_hw_dx'] /
                                                                    100.0)).groupby(by=[facility_seeking]).sum()

        # create the dataframe
        df_by_seek_level = pd.DataFrame([cases_per_facility, deaths_per_facility], index=['cases', 'deaths'])

        # number of cases needing inpatient care by severe signs (ds_pneumonia classification)
        n_inpatient_without_PO_level_0 = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples(
                [('danger_signs_pneumonia', '<90%', '0'), ('danger_signs_pneumonia', '90-92%', '0'),
                 ('danger_signs_pneumonia', '>=93%', '0')],
                names=number_cases_by_seek_level.index.names), fill_value=0).sum()

        n_inpatient_without_PO_level_1a = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples(
                [('danger_signs_pneumonia', '<90%', '1a'), ('danger_signs_pneumonia', '90-92%', '1a'),
                 ('danger_signs_pneumonia', '>=93%', '1a')],
                names=number_cases_by_seek_level.index.names), fill_value=0).sum()
        n_inpatient_without_PO_level_1b = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples(
                [('danger_signs_pneumonia', '<90%', '1b'), ('danger_signs_pneumonia', '90-92%', '1b'),
                 ('danger_signs_pneumonia', '>=93%', '1b')],
                names=number_cases_by_seek_level.index.names), fill_value=0).sum()
        n_inpatient_without_PO_level_2 = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples(
                [('danger_signs_pneumonia', '<90%', '2'), ('danger_signs_pneumonia', '90-92%', '2'),
                 ('danger_signs_pneumonia', '>=93%', '2')],
                names=number_cases_by_seek_level.index.names), fill_value=0).sum()

        # Add to dataframe - Need inpatient care based on severe signs
        df_by_seek_level = df_by_seek_level.append(
            pd.Series({'0': n_inpatient_without_PO_level_0, '1a': n_inpatient_without_PO_level_1a,
                       '1b': n_inpatient_without_PO_level_1b, '2': n_inpatient_without_PO_level_2},
                      name='Need inpatient care based on severe signs'))

        # number of cases needing inpatient care by severe signs + SpO2 level
        n_inpatient_ds_and_PO_level_0 = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples(
                [('danger_signs_pneumonia', '<90%', '0'), ('danger_signs_pneumonia', '90-92%', '0'),
                 ('danger_signs_pneumonia', '>=93%', '0'), ('chest_indrawing_pneumonia', '<90%', '0'),
                 ('fast_breathing_pneumonia', '<90%', '0'), ('cough_or_cold', '<90%', '0')],
                names=number_cases_by_seek_level.index.names), fill_value=0).sum()

        n_inpatient_ds_and_PO_level_1a = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples(
                [('danger_signs_pneumonia', '<90%', '1a'), ('danger_signs_pneumonia', '90-92%', '1a'),
                 ('danger_signs_pneumonia', '>=93%', '1a'),
                 ('chest_indrawing_pneumonia', '<90%', '1a'),
                 ('fast_breathing_pneumonia', '<90%', '1a'), ('cough_or_cold', '<90%', '1a')],
                names=number_cases_by_seek_level.index.names), fill_value=0).sum()

        n_inpatient_ds_and_PO_level_1b = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples(
                [('danger_signs_pneumonia', '<90%', '1b'), ('danger_signs_pneumonia', '90-92%', '1b'),
                 ('danger_signs_pneumonia', '>=93%', '1b'),
                 ('chest_indrawing_pneumonia', '<90%', '1b'),
                 ('fast_breathing_pneumonia', '<90%', '1b'), ('cough_or_cold', '<90%', '1b')],
                names=number_cases_by_seek_level.index.names), fill_value=0).sum()

        n_inpatient_ds_and_PO_level_2 = number_cases_by_seek_level.sum(level=[0, 1, 2]).reindex(
            pd.MultiIndex.from_tuples(
                [('danger_signs_pneumonia', '<90%', '2'), ('danger_signs_pneumonia', '90-92%', '2'),
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
        new_table = table

        # ANTIBIOTICS COST --------
        new_table['oral_amox_cost'] = new_table['age_exact_years'].apply(
            lambda x: (1 * 2 * 5 * 0.061) if x < 1 else (2 * 2 * 5 * 0.061) if 1 <= x < 2 else (3 * 2 * 5 * 0.061))

        new_table['iv_antibiotics_cost'] = new_table['age_exact_years'].apply(
            lambda x: (((1 / 2.5) * 4 * 5 * 0.5212) + (1 / 2 * 4 * 5 * 0.2694)) if x < 1 / 3 else
            (((2 / 2.5) * 4 * 5 * 0.5212) + ((1.8 / 2) * 4 * 5 * 0.2694)) if 1 / 3 <= x < 1 else
            (((3 / 2.5) * 4 * 5 * 0.5212) + ((2.7 / 2) * 4 * 5 * 0.2694)) if 1 <= x < 3 else
            (((5 / 2.5) * 4 * 5 * 0.5212) + ((3.5 / 2) * 4 * 5 * 0.2694)))  # ampicillin + gentamicin

        # alternative check --------

        iv_amp_by_age = table['age_exact_years'].apply(
            lambda x: (1 / 2.5 * 4 * 5 * 0.5212) if x < 1 / 3 else (2 / 2.5 * 4 * 5 * 0.5212) if 1 / 3 <= x < 1 else
            (3 / 2.5 * 4 * 5 * 0.5212) if 1 <= x < 3 else (5 / 2.5 * 4 * 5 * 0.5212))
        iv_gent_by_age = table['age_exact_years'].apply(
            lambda x: (1 / 2 * 4 * 5 * 0.2694) if x < 1 / 3 else (1.8 / 2 * 4 * 5 * 0.2694) if 1 / 3 <= x < 1 else
            (2.7 / 2 * 4 * 5 * 0.2694) if 1 <= x < 3 else (3.5 / 2 * 4 * 5 * 0.2694))

        iv_antibiotics = iv_amp_by_age + iv_gent_by_age

        df_iv_ant = iv_antibiotics.to_frame(name='iv_antibiotics')

        new_table = new_table.join(df_iv_ant)

        # IV antibiotic cost without PO use
        sum_iv_ant_cost_without_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'iv_antibiotics_cost'].sum()
        sum_iv_ant_cost_without_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'iv_antibiotics_cost'].sum()
        sum_iv_ant_cost_without_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'iv_antibiotics_cost'].sum()
        sum_iv_ant_cost_without_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'iv_antibiotics_cost'].sum()
        # ------------------------
        # Oral antibiotic cost without PO use
        sum_oral_ant_cost_without_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'oral_amox_cost'].sum()
        sum_oral_ant_cost_without_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'oral_amox_cost'].sum()
        sum_oral_ant_cost_without_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'oral_amox_cost'].sum()
        sum_oral_ant_cost_without_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'oral_amox_cost'].sum()

        # introduce PO - increase IV antibiotic use ----
        sum_iv_ant_cost_with_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level0_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'iv_antibiotics_cost'].sum()
        sum_iv_ant_cost_with_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level1a_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'iv_antibiotics_cost'].sum()
        sum_iv_ant_cost_with_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level1b_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'iv_antibiotics_cost'].sum()
        sum_iv_ant_cost_with_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level2_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'iv_antibiotics_cost'].sum()
        # ------------------------
        # Oral antibiotic cost with PO use
        sum_oral_ant_cost_with_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level0_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'oral_amox_cost'].sum()
        sum_oral_ant_cost_with_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level1a_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'oral_amox_cost'].sum()
        sum_oral_ant_cost_with_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level1b_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'oral_amox_cost'].sum()
        sum_oral_ant_cost_with_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level2_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'oral_amox_cost'].sum()
        # --------------------------------------------------------------------------------------------

        # OUTPATIENT CONSULTATION COST ------------
        new_table['consultation_cost'] = new_table['seek_level'].apply(
            lambda x: 2.45 if x == '2' else 2.35 if x == '1b' else 2.06 if x == '1a' else 1.67
        )
        # INPATIENT BED/DAY COST ------------
        new_table['inpatient_bed_cost'] = new_table['seek_level'].apply(
            lambda x: 6.81 * 5 if x == '2' else 6.53 * 5 if x == '1b' else 6.53 * 5
        )

        # Outpatient consultation cost without PO
        sum_consultation_cost_without_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'consultation_cost'].sum()
        sum_consultation_cost_without_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'consultation_cost'].sum()
        sum_consultation_cost_without_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'consultation_cost'].sum()
        sum_consultation_cost_without_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'consultation_cost'].sum()

        # Outpatient consultation cost with PO
        sum_consultation_cost_with_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level0_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'consultation_cost'].sum()
        sum_consultation_cost_with_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level1a_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'consultation_cost'].sum()
        sum_consultation_cost_with_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level1b_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'consultation_cost'].sum()
        sum_consultation_cost_with_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level2_{dx_accuracy}_hw_dx"] !=
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'consultation_cost'].sum()

        # Inpatient hospitalisation cost without PO
        sum_hospitalisation_cost_without_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'inpatient_bed_cost'].sum()
        sum_hospitalisation_cost_without_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'inpatient_bed_cost'].sum()
        sum_hospitalisation_cost_without_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'inpatient_bed_cost'].sum()
        sum_hospitalisation_cost_without_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_without_po_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'inpatient_bed_cost'].sum()

        # Inpatient hospitalisation cost with PO
        sum_hospitalisation_cost_with_po_level_0 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level0_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '0'), 'inpatient_bed_cost'].sum()
        sum_hospitalisation_cost_with_po_level_1a = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level1a_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1a'), 'inpatient_bed_cost'].sum()
        sum_hospitalisation_cost_with_po_level_1b = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level1b_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '1b'), 'inpatient_bed_cost'].sum()
        sum_hospitalisation_cost_with_po_level_2 = new_table.loc[
            (new_table[
                 f"classification_in_scenarios_with_po_level2_{dx_accuracy}_hw_dx"] ==
             'danger_signs_pneumonia') & (new_table['seek_level'] == '2'), 'inpatient_bed_cost'].sum()

        # PULSE OXIMETRY COST ----------------------
        new_table['PO_cost'] = new_table['seek_level'].apply(
            lambda x: 0.04769 if x in ('2', '1b') else 0.02737 if x == '1a' else 0.020137
        )
        # total PO cost
        sum_po_cost_level_0 = new_table.loc[(new_table['seek_level'] == '0'), 'PO_cost'].sum()
        sum_po_cost_level_1a = new_table.loc[(new_table['seek_level'] == '1a'), 'PO_cost'].sum()
        sum_po_cost_level_1b = new_table.loc[(new_table['seek_level'] == '1b'), 'PO_cost'].sum()
        sum_po_cost_level_2 = new_table.loc[(new_table['seek_level'] == '2'), 'PO_cost'].sum()

        # OXYGEN COST --------------
        new_table['oxygen_cost_existing_psa'] = new_table['age_exact_years'].apply(
            lambda x: (0.004417303 * 1 * 60 * 24 * 3) if x < 1/6 else (0.004417303 * 2 * 60 * 24 * 3))
        new_table['oxygen_cost_planned_psa'] = new_table['age_exact_years'].apply(
            lambda x: (0.0043940 * 1 * 60 * 24 * 3) if x < 1/6 else (0.0043940 * 2 * 60 * 24 * 3))
        new_table['oxygen_cost_all_district_psa'] = new_table['age_exact_years'].apply(
            lambda x: (0.004861291 * 1 * 60 * 24 * 3) if x < 1/6 else (0.004861291 * 2 * 60 * 24 * 3))

        # Oxygen cost without PO use - Existing PSA
        sum_oxygen_cost_existing_psa_without_po_level_0 = new_table.loc[
            (new_table[f'oxygen_provided_scenario_existing_psa_{dx_accuracy}_hw_dx'])
            & (new_table['seek_level'] == '0'), 'oxygen_cost_existing_psa'].sum()
        sum_oxygen_cost_existing_psa_without_po_level_1a = new_table.loc[
           (new_table[f'oxygen_provided_scenario_existing_psa_{dx_accuracy}_hw_dx'])
           & (new_table['seek_level'] == '1a'), 'oxygen_cost_existing_psa'].sum()
        sum_oxygen_cost_existing_psa_without_po_level_1b = new_table.loc[
            (new_table[f'oxygen_provided_scenario_existing_psa_{dx_accuracy}_hw_dx']) &
           (new_table['seek_level'] == '1b'), 'oxygen_cost_existing_psa'].sum()
        sum_oxygen_cost_existing_psa_without_po_level_2 = new_table.loc[
            (new_table[f'oxygen_provided_scenario_existing_psa_{dx_accuracy}_hw_dx']) &
          (new_table['seek_level'] == '2'), 'oxygen_cost_existing_psa'].sum()

        # Oxygen cost without PO use - Planned PSA
        sum_oxygen_cost_planned_psa_without_po_level_0 = new_table.loc[
         (new_table[f'oxygen_provided_scenario_planned_psa_{dx_accuracy}_hw_dx']) &
         (new_table['seek_level'] == '0'), 'oxygen_cost_planned_psa'].sum()
        sum_oxygen_cost_planned_psa_without_po_level_1a = new_table.loc[
            (new_table[f'oxygen_provided_scenario_planned_psa_{dx_accuracy}_hw_dx']) &
            (new_table['seek_level'] == '1a'), 'oxygen_cost_planned_psa'].sum()
        sum_oxygen_cost_planned_psa_without_po_level_1b = new_table.loc[
            (new_table[f'oxygen_provided_scenario_planned_psa_{dx_accuracy}_hw_dx']) &
           (new_table['seek_level'] == '1b'), 'oxygen_cost_planned_psa'].sum()
        sum_oxygen_cost_planned_psa_without_po_level_2 = new_table.loc[
             (new_table[f'oxygen_provided_scenario_planned_psa_{dx_accuracy}_hw_dx']) &
            (new_table['seek_level'] == '2'), 'oxygen_cost_planned_psa'].sum()

        # Oxygen cost without PO use - All District PSA
        sum_oxygen_cost_all_district_psa_without_po_level_0 = new_table.loc[
            (new_table[f'oxygen_provided_scenario_all_district_psa_{dx_accuracy}_hw_dx']) &
             (new_table['seek_level'] == '0'), 'oxygen_cost_all_district_psa'].sum()
        sum_oxygen_cost_all_district_psa_without_po_level_1a = new_table.loc[
            (new_table[f'oxygen_provided_scenario_all_district_psa_{dx_accuracy}_hw_dx']) &
              (new_table['seek_level'] == '1a'), 'oxygen_cost_all_district_psa'].sum()
        sum_oxygen_cost_all_district_psa_without_po_level_1b = new_table.loc[
            (new_table[f'oxygen_provided_scenario_all_district_psa_{dx_accuracy}_hw_dx']) &
              (new_table['seek_level'] == '1b'), 'oxygen_cost_all_district_psa'].sum()
        sum_oxygen_cost_all_district_psa_without_po_level_2 = new_table.loc[
            (new_table[f'oxygen_provided_scenario_all_district_psa_{dx_accuracy}_hw_dx']) &
             (new_table['seek_level'] == '2'), 'oxygen_cost_all_district_psa'].sum()

        # ----------------------------------------------------------------------
        # Oxygen cost with PO use - Existing PSA
        sum_oxygen_cost_existing_psa_with_po_level_0 = new_table.loc[
               (new_table[f'oxygen_provided_scenario_existing_psa_with_po_level0_{dx_accuracy}_hw_dx']) &
               (new_table['seek_level'] == '0'), 'oxygen_cost_existing_psa'].sum()
        sum_oxygen_cost_existing_psa_with_po_level_1a = new_table.loc[
                (new_table[f'oxygen_provided_scenario_existing_psa_with_po_level1a_{dx_accuracy}_hw_dx']) &
                (new_table['seek_level'] == '1a'), 'oxygen_cost_existing_psa'].sum()
        sum_oxygen_cost_existing_psa_with_po_level_1b = new_table.loc[
                (new_table[f'oxygen_provided_scenario_existing_psa_with_po_level1b_{dx_accuracy}_hw_dx']) &
                (new_table['seek_level'] == '1b'), 'oxygen_cost_existing_psa'].sum()
        sum_oxygen_cost_existing_psa_with_po_level_2 = new_table.loc[
                (new_table[f'oxygen_provided_scenario_existing_psa_with_po_level2_{dx_accuracy}_hw_dx']) &
                (new_table['seek_level'] == '2'), 'oxygen_cost_existing_psa'].sum()

        # ----------------------------------------------------------------------
        # Oxygen cost with PO use - Planned PSA
        sum_oxygen_cost_planned_psa_with_po_level_0 = new_table.loc[
            (new_table[f'oxygen_provided_scenario_planned_psa_with_po_level0_{dx_accuracy}_hw_dx']) &
            (new_table['seek_level'] == '0'), 'oxygen_cost_planned_psa'].sum()
        sum_oxygen_cost_planned_psa_with_po_level_1a = new_table.loc[
            (new_table[f'oxygen_provided_scenario_planned_psa_with_po_level1a_{dx_accuracy}_hw_dx']) &
            (new_table['seek_level'] == '1a'), 'oxygen_cost_planned_psa'].sum()
        sum_oxygen_cost_planned_psa_with_po_level_1b = new_table.loc[
            (new_table[f'oxygen_provided_scenario_planned_psa_with_po_level1b_{dx_accuracy}_hw_dx']) &
            (new_table['seek_level'] == '1b'), 'oxygen_cost_planned_psa'].sum()
        sum_oxygen_cost_planned_psa_with_po_level_2 = new_table.loc[
            (new_table[f'oxygen_provided_scenario_planned_psa_with_po_level2_{dx_accuracy}_hw_dx']) &
            (new_table['seek_level'] == '2'), 'oxygen_cost_planned_psa'].sum()

        # ----------------------------------------------------------------------
        # Oxygen cost with PO use - Existing PSA
        sum_oxygen_cost_all_district_psa_with_po_level_0 = new_table.loc[
            (new_table[f'oxygen_provided_scenario_all_district_psa_with_po_level0_{dx_accuracy}_hw_dx']) &
            (new_table['seek_level'] == '0'), 'oxygen_cost_all_district_psa'].sum()
        sum_oxygen_cost_all_district_psa_with_po_level_1a = new_table.loc[
            (new_table[f'oxygen_provided_scenario_all_district_psa_with_po_level1a_{dx_accuracy}_hw_dx']) &
            (new_table['seek_level'] == '1a'), 'oxygen_cost_all_district_psa'].sum()
        sum_oxygen_cost_all_district_psa_with_po_level_1b = new_table.loc[
            (new_table[f'oxygen_provided_scenario_all_district_psa_with_po_level1b_{dx_accuracy}_hw_dx']) &
            (new_table['seek_level'] == '1b'), 'oxygen_cost_all_district_psa'].sum()
        sum_oxygen_cost_all_district_psa_with_po_level_2 = new_table.loc[
            (new_table[f'oxygen_provided_scenario_all_district_psa_with_po_level2_{dx_accuracy}_hw_dx']) &
            (new_table['seek_level'] == '2'), 'oxygen_cost_all_district_psa'].sum()

        # ----------------------------------------------------------------------
        # update the dataframe - without PO use
        if scenario in ('baseline_ant', 'existing_psa', 'planned_psa', 'all_district_psa'):
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
                pd.Series(
                    {'0': sum_consultation_cost_without_po_level_0, '1a': sum_consultation_cost_without_po_level_1a,
                     '1b': sum_consultation_cost_without_po_level_1b, '2': sum_consultation_cost_without_po_level_2},
                    name='Consultations w/out PO'))
            # hospitalisation cost without PO use
            df_by_seek_level = df_by_seek_level.append(
                pd.Series({'0': sum_hospitalisation_cost_without_po_level_0,
                           '1a': sum_hospitalisation_cost_without_po_level_1a,
                           '1b': sum_hospitalisation_cost_without_po_level_1b,
                           '2': sum_hospitalisation_cost_without_po_level_2},
                          name='Inpatient bed w/out PO'))
            # Oxygen cost without PO use -----
            if scenario == 'existing_psa':
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oxygen_cost_existing_psa_without_po_level_0,
                               '1a': sum_oxygen_cost_existing_psa_without_po_level_1a,
                               '1b': sum_oxygen_cost_existing_psa_without_po_level_1b,
                               '2': sum_oxygen_cost_existing_psa_without_po_level_2},
                              name='Oxygen cost in existing PSA, no PO'))
            if scenario == 'planned_psa':
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oxygen_cost_planned_psa_without_po_level_0,
                               '1a': sum_oxygen_cost_planned_psa_without_po_level_1a,
                               '1b': sum_oxygen_cost_planned_psa_without_po_level_1b,
                               '2': sum_oxygen_cost_planned_psa_without_po_level_2},
                              name='Oxygen cost in planned PSA, no PO'))
            if scenario == 'all_district_psa':
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oxygen_cost_all_district_psa_without_po_level_0,
                               '1a': sum_oxygen_cost_all_district_psa_without_po_level_1a,
                               '1b': sum_oxygen_cost_all_district_psa_without_po_level_1b,
                               '2': sum_oxygen_cost_all_district_psa_without_po_level_2},
                              name='Oxygen cost in all district PSA, no PO'))

        else:
            if scenario.endswith('with_po_level2'):
                # update the dataframe - Oral amoxicillin cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oral_ant_cost_without_po_level_0, '1a': sum_oral_ant_cost_without_po_level_1a,
                               '1b': sum_oral_ant_cost_without_po_level_1b, '2': sum_oral_ant_cost_with_po_level_2},
                              name='Oral antibiotics w/ PO at level 2'))
                # update the dataframe - IV antibiotics cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_iv_ant_cost_without_po_level_0, '1a': sum_iv_ant_cost_without_po_level_1a,
                               '1b': sum_iv_ant_cost_without_po_level_1b, '2': sum_iv_ant_cost_with_po_level_2},
                              name='IV antibiotics w/ PO at level 2'))
                # consultation cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_consultation_cost_without_po_level_0, '1a': sum_consultation_cost_without_po_level_1a,
                               '1b': sum_consultation_cost_without_po_level_1b, '2': sum_consultation_cost_with_po_level_2},
                              name='Consultations w/ PO at level 2'))
                # hospitalisation cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series(
                        {'0': sum_hospitalisation_cost_without_po_level_0, '1a': sum_hospitalisation_cost_without_po_level_1a,
                         '1b': sum_hospitalisation_cost_without_po_level_1b, '2': sum_hospitalisation_cost_with_po_level_2},
                        name='Inpatient bed w/ PO at level 2'))
                # PO use up to level 2
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series(
                        {'0': 0, '1a': 0,
                         '1b': 0, '2': sum_po_cost_level_2},
                        name='PO cost available at level 2'))

            elif scenario.endswith('with_po_level1b'):
                # update the dataframe - Oral amoxicillin cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oral_ant_cost_without_po_level_0, '1a': sum_oral_ant_cost_without_po_level_1a,
                               '1b': sum_oral_ant_cost_with_po_level_1b, '2': sum_oral_ant_cost_with_po_level_2},
                              name='Oral antibiotics w/ PO at level 1b'))
                # update the dataframe - IV antibiotics cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_iv_ant_cost_without_po_level_0, '1a': sum_iv_ant_cost_without_po_level_1a,
                               '1b': sum_iv_ant_cost_with_po_level_1b, '2': sum_iv_ant_cost_with_po_level_2},
                              name='IV antibiotics w/ PO at level 1b'))
                # consultation cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_consultation_cost_without_po_level_0, '1a': sum_consultation_cost_without_po_level_1a,
                               '1b': sum_consultation_cost_with_po_level_1b, '2': sum_consultation_cost_with_po_level_2},
                              name='Consultations w/ PO at level 1b'))
                # hospitalisation cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series(
                        {'0': sum_hospitalisation_cost_without_po_level_0, '1a': sum_hospitalisation_cost_without_po_level_1a,
                         '1b': sum_hospitalisation_cost_with_po_level_1b, '2': sum_hospitalisation_cost_with_po_level_2},
                        name='Inpatient bed w/ PO at level 1b'))
                # PO use up to level 1b
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series(
                        {'0': 0, '1a': 0,
                         '1b': sum_po_cost_level_1b, '2': sum_po_cost_level_2},
                        name='PO cost available at level 1b'))

            elif scenario.endswith('with_po_level1a'):
                # update the dataframe - Oral amoxicillin cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oral_ant_cost_without_po_level_0, '1a': sum_oral_ant_cost_with_po_level_1a,
                               '1b': sum_oral_ant_cost_with_po_level_1b, '2': sum_oral_ant_cost_with_po_level_2},
                              name='Oral antibiotics w/ PO at level 1a'))
                # update the dataframe - IV antibiotics cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_iv_ant_cost_without_po_level_0, '1a': sum_iv_ant_cost_with_po_level_1a,
                               '1b': sum_iv_ant_cost_with_po_level_1b, '2': sum_iv_ant_cost_with_po_level_2},
                              name='IV antibiotics w/ PO at level 1a'))
                # consultation cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_consultation_cost_without_po_level_0, '1a': sum_consultation_cost_with_po_level_1a,
                               '1b': sum_consultation_cost_with_po_level_1b, '2': sum_consultation_cost_with_po_level_2},
                              name='Consultations w/ PO at level 1a'))
                # hospitalisation cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series(
                        {'0': sum_hospitalisation_cost_without_po_level_0, '1a': sum_hospitalisation_cost_with_po_level_1a,
                         '1b': sum_hospitalisation_cost_with_po_level_1b, '2': sum_hospitalisation_cost_with_po_level_2},
                        name='Inpatient bed w/ PO at level 1a'))
                # PO use up to level 1a
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series(
                        {'0': 0, '1a': sum_po_cost_level_1a,
                         '1b': sum_po_cost_level_1b, '2': sum_po_cost_level_2},
                        name='PO cost available at level 1a'))

            elif scenario.endswith('with_po_level0'):
                # update the dataframe - Oral amoxicillin cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oral_ant_cost_with_po_level_0, '1a': sum_oral_ant_cost_with_po_level_1a,
                               '1b': sum_oral_ant_cost_with_po_level_1b, '2': sum_oral_ant_cost_with_po_level_2},
                              name='Oral antibiotics w/ PO at level 0'))
                # update the dataframe - IV antibiotics cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_iv_ant_cost_with_po_level_0, '1a': sum_iv_ant_cost_with_po_level_1a,
                               '1b': sum_iv_ant_cost_with_po_level_1b, '2': sum_iv_ant_cost_with_po_level_2},
                              name='IV antibiotics w/ PO at level 0'))
                # consultation cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_consultation_cost_with_po_level_0, '1a': sum_consultation_cost_with_po_level_1a,
                               '1b': sum_consultation_cost_with_po_level_1b, '2': sum_consultation_cost_with_po_level_2},
                              name='Consultations w/ PO at level 0'))
                # hospitalisation cost with PO use
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series(
                        {'0': sum_hospitalisation_cost_with_po_level_0, '1a': sum_hospitalisation_cost_with_po_level_1a,
                         '1b': sum_hospitalisation_cost_with_po_level_1b, '2': sum_hospitalisation_cost_with_po_level_2},
                        name='Inpatient bed w/ PO at level 0a'))
                # PO use at all levels
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series(
                        {'0': sum_po_cost_level_0, '1a': sum_po_cost_level_1a,
                         '1b': sum_po_cost_level_1b, '2': sum_po_cost_level_2},
                        name='PO cost available at level 0'))

            # Oxygen cost with PO use -------------
            # Existing PSA
            if scenario == 'existing_psa_with_po_level2':
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oxygen_cost_existing_psa_without_po_level_0,
                               '1a': sum_oxygen_cost_existing_psa_without_po_level_1a,
                               '1b': sum_oxygen_cost_existing_psa_without_po_level_1b,
                               '2': sum_oxygen_cost_existing_psa_with_po_level_2},
                              name='Oxygen cost in existing PSA, with PO at level 2'))
            if scenario == 'existing_psa_with_po_level1b':
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oxygen_cost_existing_psa_without_po_level_0,
                               '1a': sum_oxygen_cost_existing_psa_without_po_level_1a,
                               '1b': sum_oxygen_cost_existing_psa_with_po_level_1b,
                               '2': sum_oxygen_cost_existing_psa_with_po_level_2},
                              name='Oxygen cost in existing PSA, with PO at level 1b'))
            if scenario == 'existing_psa_with_po_level1a':
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oxygen_cost_existing_psa_without_po_level_0,
                               '1a': sum_oxygen_cost_existing_psa_with_po_level_1a,
                               '1b': sum_oxygen_cost_existing_psa_with_po_level_1b,
                               '2': sum_oxygen_cost_existing_psa_with_po_level_2},
                              name='Oxygen cost in existing PSA, with PO at level 1a'))
            if scenario == 'existing_psa_with_po_level0':
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oxygen_cost_existing_psa_with_po_level_0,
                               '1a': sum_oxygen_cost_existing_psa_with_po_level_1a,
                               '1b': sum_oxygen_cost_existing_psa_with_po_level_1b,
                               '2': sum_oxygen_cost_existing_psa_with_po_level_2},
                              name='Oxygen cost in existing PSA, with PO at level 0'))

            # Planned PSA
            if scenario == 'planned_psa_with_po_level2':
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oxygen_cost_planned_psa_without_po_level_0,
                               '1a': sum_oxygen_cost_planned_psa_without_po_level_1a,
                               '1b': sum_oxygen_cost_planned_psa_without_po_level_1b,
                               '2': sum_oxygen_cost_planned_psa_with_po_level_2},
                              name='Oxygen cost in planned PSA, with PO at level 2'))
            if scenario == 'planned_psa_with_po_level1b':
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oxygen_cost_planned_psa_without_po_level_0,
                               '1a': sum_oxygen_cost_planned_psa_without_po_level_1a,
                               '1b': sum_oxygen_cost_planned_psa_with_po_level_1b,
                               '2': sum_oxygen_cost_planned_psa_with_po_level_2},
                              name='Oxygen cost in planned PSA, with PO at level 1b'))
            if scenario == 'planned_psa_with_po_level1a':
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oxygen_cost_planned_psa_without_po_level_0,
                               '1a': sum_oxygen_cost_planned_psa_with_po_level_1a,
                               '1b': sum_oxygen_cost_planned_psa_with_po_level_1b,
                               '2': sum_oxygen_cost_planned_psa_with_po_level_2},
                              name='Oxygen cost in planned PSA, with PO at level 1a'))
            if scenario == 'planned_psa_with_po_level0':
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oxygen_cost_planned_psa_with_po_level_0,
                               '1a': sum_oxygen_cost_planned_psa_with_po_level_1a,
                               '1b': sum_oxygen_cost_planned_psa_with_po_level_1b,
                               '2': sum_oxygen_cost_planned_psa_with_po_level_2},
                              name='Oxygen cost in planned PSA, with PO at level 0'))

            # All district PSA
            if scenario == 'all_district_psa_with_po_level2':
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oxygen_cost_all_district_psa_without_po_level_0,
                               '1a': sum_oxygen_cost_all_district_psa_without_po_level_1a,
                               '1b': sum_oxygen_cost_all_district_psa_without_po_level_1b,
                               '2': sum_oxygen_cost_all_district_psa_with_po_level_2},
                              name='Oxygen cost in all_district PSA, with PO at level 2'))
            if scenario == 'all_district_psa_with_po_level1b':
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oxygen_cost_all_district_psa_without_po_level_0,
                               '1a': sum_oxygen_cost_all_district_psa_without_po_level_1a,
                               '1b': sum_oxygen_cost_all_district_psa_with_po_level_1b,
                               '2': sum_oxygen_cost_all_district_psa_with_po_level_2},
                              name='Oxygen cost in all_district PSA, with PO at level 1b'))
            if scenario == 'all_district_psa_with_po_level1a':
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oxygen_cost_all_district_psa_without_po_level_0,
                               '1a': sum_oxygen_cost_all_district_psa_with_po_level_1a,
                               '1b': sum_oxygen_cost_all_district_psa_with_po_level_1b,
                               '2': sum_oxygen_cost_all_district_psa_with_po_level_2},
                              name='Oxygen cost in all_district PSA, with PO at level 1a'))
            if scenario == 'all_district_psa_with_po_level0':
                df_by_seek_level = df_by_seek_level.append(
                    pd.Series({'0': sum_oxygen_cost_all_district_psa_with_po_level_0,
                               '1a': sum_oxygen_cost_all_district_psa_with_po_level_1a,
                               '1b': sum_oxygen_cost_all_district_psa_with_po_level_1b,
                               '2': sum_oxygen_cost_all_district_psa_with_po_level_2},
                              name='Oxygen cost in all_district PSA, with PO at level 0'))

        return df_by_seek_level


    # Do the Dataframe with summary output --------------------------------------
    dx_accuracy = 'perfect'

    # Baseline antibiotics
    cea_df_baseline_ant_without_po = cea_df_by_scenario(scenario='baseline_ant', dx_accuracy=dx_accuracy)

    # Oxygen scenarios
    cea_df_existing_psa_without_po = cea_df_by_scenario(scenario='existing_psa', dx_accuracy=dx_accuracy)
    cea_df_planned_psa_without_po = cea_df_by_scenario(scenario='planned_psa', dx_accuracy=dx_accuracy)
    cea_df_all_district_psa_without_po = cea_df_by_scenario(scenario='all_district_psa',
                                                            dx_accuracy=dx_accuracy)

    # Oxygen and PO - Existing PSA
    cea_df_existing_psa_with_po_level2 = cea_df_by_scenario(scenario='existing_psa_with_po_level2',
                                                            dx_accuracy=dx_accuracy)
    cea_df_existing_psa_with_po_level1b = cea_df_by_scenario(scenario='existing_psa_with_po_level1b',
                                                             dx_accuracy=dx_accuracy)
    cea_df_existing_psa_with_po_level1a = cea_df_by_scenario(scenario='existing_psa_with_po_level1a',
                                                             dx_accuracy=dx_accuracy)
    cea_df_existing_psa_with_po_level0 = cea_df_by_scenario(scenario='existing_psa_with_po_level0',
                                                            dx_accuracy=dx_accuracy)

    # Oxygen and PO - Planned PSA
    cea_df_planned_psa_with_po_level2 = cea_df_by_scenario(scenario='planned_psa_with_po_level2',
                                                           dx_accuracy=dx_accuracy)
    cea_df_planned_psa_with_po_level1b = cea_df_by_scenario(scenario='planned_psa_with_po_level1b',
                                                            dx_accuracy=dx_accuracy)
    cea_df_planned_psa_with_po_level1a = cea_df_by_scenario(scenario='planned_psa_with_po_level1a',
                                                            dx_accuracy=dx_accuracy)
    cea_df_planned_psa_with_po_level0 = cea_df_by_scenario(scenario='planned_psa_with_po_level0',
                                                           dx_accuracy=dx_accuracy)

    # Oxygen and PO - All district PSA
    cea_df_all_district_psa_with_po_level2 = cea_df_by_scenario(scenario='all_district_psa_with_po_level2',
                                                                dx_accuracy=dx_accuracy)
    cea_df_all_district_psa_with_po_level1b = cea_df_by_scenario(scenario='all_district_psa_with_po_level1b',
                                                                 dx_accuracy=dx_accuracy)
    cea_df_all_district_psa_with_po_level1a = cea_df_by_scenario(scenario='all_district_psa_with_po_level1a',
                                                                 dx_accuracy=dx_accuracy)
    cea_df_all_district_psa_with_po_level0 = cea_df_by_scenario(scenario='all_district_psa_with_po_level0',
                                                                dx_accuracy=dx_accuracy)

    # Plot graph
    low_oxygen = (table["oxygen_saturation"] == "<90%").replace({True: '<90%', False: ">=90%"})  # for Spo2<90% cutoff
    classification_level2 = table['classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2']
    sought_care_level = table['seek_level']
    dx_accuracy = 'perfect'
    res = {
        "Existing PSA": {
            "Existing PSA, no PO": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_scenario_existing_psa_{dx_accuracy}_hw_dx'] / 100.0
                 )).groupby(by=[sought_care_level, low_oxygen]).sum(),
            "Existing PSA, + PO level 2": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_scenario_existing_psa_with_po_level2_{dx_accuracy}_hw_dx']
                 / 100.0
                 )).groupby(by=[sought_care_level, low_oxygen]).sum(),
            "Existing PSA, + PO level 1b": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_scenario_existing_psa_with_po_level1b_{dx_accuracy}_hw_dx']
                 / 100.0
                 )).groupby(by=[sought_care_level, low_oxygen]).sum(),
            "Existing PSA, + PO level 1a": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_scenario_existing_psa_with_po_level1a_{dx_accuracy}_hw_dx']
                 / 100.0
                 )).groupby(by=[sought_care_level, low_oxygen]).sum(),
            "Existing PSA, + PO level 0": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_scenario_existing_psa_with_po_level0_{dx_accuracy}_hw_dx']
                 / 100.0
                 )).groupby(by=[sought_care_level, low_oxygen]).sum(),
        },
        "Planned PSA": {
            "Planned PSA, no PO": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_scenario_planned_psa_{dx_accuracy}_hw_dx'] / 100.0
                 )).groupby(by=[sought_care_level, low_oxygen]).sum(),
            "Planned PSA, + PO level 2": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_scenario_planned_psa_with_po_level2_{dx_accuracy}_hw_dx']
                 / 100.0
                 )).groupby(by=[sought_care_level, low_oxygen]).sum(),
            "Planned PSA, + PO level 1b": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_scenario_planned_psa_with_po_level1b_{dx_accuracy}_hw_dx']
                 / 100.0
                 )).groupby(by=[sought_care_level, low_oxygen]).sum(),
            "Planned PSA, + PO level 1a": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_scenario_planned_psa_with_po_level1a_{dx_accuracy}_hw_dx']
                 / 100.0
                 )).groupby(by=[sought_care_level, low_oxygen]).sum(),
            "Planned PSA, + PO level 0": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_scenario_planned_psa_with_po_level0_{dx_accuracy}_hw_dx']
                 / 100.0
                 )).groupby(by=[sought_care_level, low_oxygen]).sum(),
        },
        "All Districts PSA": {
            "All Districts PSA, no PO": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_scenario_all_district_psa_{dx_accuracy}_hw_dx'] / 100.0
                 )).groupby(by=[sought_care_level, low_oxygen]).sum(),
            "All Districts PSA, + PO level 2": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_scenario_all_district_psa_with_po_level2_{dx_accuracy}_hw_dx']
                 / 100.0
                 )).groupby(by=[sought_care_level, low_oxygen]).sum(),
            "All Districts PSA, + PO level 1b": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_scenario_all_district_psa_with_po_level1b_{dx_accuracy}_hw_dx']
                 / 100.0
                 )).groupby(by=[sought_care_level, low_oxygen]).sum(),
            "All Districts PSA, + PO level 1a": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_scenario_all_district_psa_with_po_level1a_{dx_accuracy}_hw_dx']
                 / 100.0
                 )).groupby(by=[sought_care_level, low_oxygen]).sum(),
            "All Districts PSA, + PO level 0": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_scenario_all_district_psa_with_po_level0_{dx_accuracy}_hw_dx']
                 / 100.0
                 )).groupby(by=[sought_care_level, low_oxygen]).sum(),
        },
    }

    deaths = pd.concat({k: pd.DataFrame(v) for k, v in res.items()}, axis=1)
    deaths_total = deaths.sum()

    #
    # # PLOTS!!!
    # fig, axs = plt.subplots(ncols=2, nrows=1, sharey=True, constrained_layout=True)
    # for i, ix in enumerate(results.columns.levels[0]):
    #     ax = axs[i]
    #     # results.loc[:, (ix, slice(None))].T.plot.bar(stacked=True, ax=ax, legend=False)
    #     results.loc[:, (ix, slice(None))].T.plot.bar(stacked=True, ax=ax, width=0.6, legend=False,
    #                                                  color=results.index.map(assign_colors))
    #     ax.set_xticklabels(results.loc[:, (ix, slice(None))].columns.levels[1])
    #     ax.set_ylabel('Deaths per 100,000 cases of ALRI')
    #     ax.set_title(f"{ix}", fontsize=10)
    #     ax.grid(axis='y')
    # handles, labels = axs[-1].get_legend_handles_labels()
    # ax.legend(reversed(handles), reversed(labels), title='Case Type', loc='upper left', bbox_to_anchor=(1, 1),
    #           fontsize=7)
    # # fig.suptitle('Deaths Under Different Interventions Combinations', fontsize=14, fontweight='semibold')
    # fig.suptitle('Under current policy', fontsize=12, fontweight='semibold')
    # fig.show()
    # # fig.savefig(Path('./outputs') / ('imperfect dx - hosp - current vs new policy' + datestamp + ".pdf"), format='pdf')
    # plt.close(fig)

    # # # # Calculations # # # #
    def deaths_by_scenario(scenario):
        """ Function to get the number of death by scenario"""

        # no PO
        deaths_existing_psa_without_po = (table['prob_die_if_no_treatment'] * (
            1.0 - table[
            f'treatment_efficacy_scenario_{scenario}_psa_{dx_accuracy}_hw_dx'] / 100.0)
                                          ).groupby(by=[classification_level2, low_oxygen]).sum()
        sum_deaths_scenario_without_po = deaths_existing_psa_without_po.sum()

        # PO level 2
        deaths_existing_psa_with_po_level2 = (table['prob_die_if_no_treatment'] * (
            1.0 - table[
            f'treatment_efficacy_scenario_{scenario}_psa_with_po_level2_{dx_accuracy}_hw_dx'] / 100.0)
                                              ).groupby(by=[classification_level2, low_oxygen]).sum()
        sum_deaths_scenario_with_po_level2 = deaths_existing_psa_with_po_level2.sum()

        # PO level 2, 1b
        deaths_existing_psa_with_po_level1b = (table['prob_die_if_no_treatment'] * (
            1.0 - table[
            f'treatment_efficacy_scenario_{scenario}_psa_with_po_level1b_{dx_accuracy}_hw_dx'] / 100.0)
                                               ).groupby(by=[classification_level2, low_oxygen]).sum()
        sum_deaths_scenario_with_po_level1b = deaths_existing_psa_with_po_level1b.sum()

        # PO level 2, 1b, 1a
        deaths_existing_psa_with_po_level1a = (table['prob_die_if_no_treatment'] * (
            1.0 - table[
            f'treatment_efficacy_scenario_{scenario}_psa_with_po_level1a_{dx_accuracy}_hw_dx'] / 100.0)
                                               ).groupby(by=[classification_level2, low_oxygen]).sum()
        sum_deaths_scenario_with_po_level1a = deaths_existing_psa_with_po_level1a.sum()

        # PO level 2, 1b, 1a, 0
        deaths_existing_psa_with_po_level0 = (table['prob_die_if_no_treatment'] * (
            1.0 - table[
            f'treatment_efficacy_scenario_{scenario}_psa_with_po_level0_{dx_accuracy}_hw_dx'] / 100.0)
                                              ).groupby(by=[classification_level2, low_oxygen]).sum()
        sum_deaths_scenario_with_po_level0 = deaths_existing_psa_with_po_level0.sum()

        return sum_deaths_scenario_without_po, \
               sum_deaths_scenario_with_po_level2, \
               sum_deaths_scenario_with_po_level1b, \
               sum_deaths_scenario_with_po_level1a, \
               sum_deaths_scenario_with_po_level0


    for scenario in ('existing', 'planned', 'all_district'):
        number_of_deaths_by_scenario = deaths_by_scenario(scenario)

