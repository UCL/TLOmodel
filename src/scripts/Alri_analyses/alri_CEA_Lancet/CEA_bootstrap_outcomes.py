""" This script will output the mean and CI from the bootstrap iterations of scenario outcomes """

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

dx_accuracy = 'imperfect'

# False if main analysis, True if a sensitivity analysis (HW Dx Accuracy, or Prioritise oxygen in Hospitals)
sensitivity_analysis = False

sensitivity_analysis_hw_dx = True  # change to False if analysing Oxygen prioritisation


# Helper function for conversion between odds and probabilities
to_odds = lambda pr: pr / (1.0 - pr)  # noqa: E731tab e402c

to_prob = lambda odds: odds / (1.0 + odds)  # noqa: E731

# Date for saving the image for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Store all the scenarios results
scenario_dfs = {}  # Initialize empty dictionary of dataframes
cohort_data_rows = []

scenarios = ['baseline_ant',
             'baseline_ant_with_po_level2', 'baseline_ant_with_po_level1b',
             'baseline_ant_with_po_level1a', 'baseline_ant_with_po_level0',
             'existing_psa', 'existing_psa_with_po_level2', 'existing_psa_with_po_level1b',
             'existing_psa_with_po_level1a', 'existing_psa_with_po_level0',
             'planned_psa', 'planned_psa_with_po_level2', 'planned_psa_with_po_level1b',
             'planned_psa_with_po_level1a', 'planned_psa_with_po_level0'
             ]

# Apply for each scenario
for scenario in scenarios:

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
        table['total_costs'] = table['all_oral_amox_cost'] + table['all_iv_antibiotics_cost'] + \
                               table['all_outpatient_consultation_cost'] + table['all_inpatient_bed_cost'] + \
                               table['all_po_cost'] + table['all_oxygen_cost']


        # --------------------------------------------------------------------------------

        # # #
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
        # Create DataFrame with headers
        df = pd.DataFrame(columns=['alri_size', 'mean_age', 'deaths', 'DALYs_discounted',
                                   'cost of oral antibiotics', 'cost of outpatient',
                                   'cost of IV antibiotics', 'cost of inpatient bed', 'cost of PO', 'cost of Oxygen',
                                   'total_costs', 'need_oxygen', 'oxygen_provided', 'oxygen_liters_provided',
                                   'total_ox_liters_for_alri',
                                   'age <1', 'age 1-2', 'age 2-5',
                                   'Enterobacteriaceae', 'H.influenzae_non_type_b', 'HMPV', 'Hib', 'Influenza',
                                   'P.jirovecii', 'Parainfluenza', 'RSV', 'Rhinovirus', 'Staph_aureus',
                                   'Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', 'other_Strepto_Enterococci',
                                   'other_bacterial_pathogens', 'other_pathogens_NoS', 'other_viral_pathogens',
                                   'coinfection',
                                   'disease_type_other_alri', 'disease_type_pneumonia',
                                   'mean_duration_in_days', 'sd_duration',
                                   'care_seeking_level0', 'care_seeking_level1a',
                                   'care_seeking_level1b', 'care_seeking_level2',
                                   'hypoxaemia_90-92%', 'hypoxaemia_<90%', 'hypoxaemia_>=93%',
                                   'any_pc_complications',
                                   'pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax',  'bacteraemia',
                                   'all_complications',
                                   'MAM', 'SAM', 'well_nourished',
                                   'hiv_negative', 'hiv_positive', 'hiv_pos_not_on_art',
                                   'will_die', 'prob_will_die_no_treat', 'will_die_by_class_ci',
                                   'will_die_by_class_cc',  'will_die_by_class_ds', 'will_die_by_class_fb',
                                   'will_die_by_ox_sat_90-92%', 'will_die_by_ox_sat_<90%', 'will_die_by_ox_sat_>=93%',
                                   'prob_will_die_by_ox_sat_90-92%', 'prob_will_die_by_ox_sat_<90%',
                                   'prob_will_die_by_ox_sat_>=93%', 'age IQR_25', 'age IQR_75'
                                   ])

        # Append results of the scenario in each loop
        # Append results of the scenario in each loop
        scenario_dfs[scenario] = pd.DataFrame(table)
        cohort_data_rows.append(data)

    print(len(scenario_dfs))


cohort_data_df = pd.DataFrame(cohort_data_rows)

# Save the table output
with open(f'debug_output_cohort_data{datestamp}.pkl', 'wb') as f:
    cohort_results = {
        'cohort_summary': cohort_data_df, 'cohort_table': scenario_dfs
    }
    pickle.dump(cohort_results, f)


# # Later, load it back
# with open(f'debug_output_cohort_data{datestamp}.pkl', 'rb') as f:
#     my_output = pickle.load(f)
#
#     cohort_table_baseline_ant = pd.DataFrame(my_output['cohort_table']['baseline_ant'])



def get_scenario_differences(scenario_dfs, scenario, wtp=80):
    """Calculate point estimate differences and economic metrics for all scenarios relative to baseline."""

    # deaths averted and incremental costs
    deaths_averted = (scenario_dfs['baseline_ant']['mortality_outcome'].astype(int) -
                      scenario_dfs[scenario]['mortality_outcome'].astype(int)).sum()
    dalys_averted = (scenario_dfs['baseline_ant']['DALYs_discounted'] -
                     scenario_dfs[scenario]['DALYs_discounted']).sum()

    incremental_cost = (scenario_dfs[scenario]['total_costs'] - scenario_dfs['baseline_ant']['total_costs']).sum()
    mortality_reduction = deaths_averted / scenario_dfs['baseline_ant']['mortality_outcome'].astype(int).sum()

    # economic metrics
    icer_deaths = incremental_cost / deaths_averted
    icer_dalys = incremental_cost / dalys_averted
    inhb = dalys_averted - (incremental_cost / wtp)
    inmb = (dalys_averted * wtp) - incremental_cost

    return {
        'deaths_averted': deaths_averted,
        'dalys_averted': dalys_averted,
        'incremental_cost': incremental_cost,
        'mortality_reduction': mortality_reduction,
        'icer_deaths': icer_deaths,
        'icer_dalys': icer_dalys,
        'inhb': inhb, 'inmb': inmb
    }


def calculate_economic_uncertainty(scenario1, scenario2, wtp=80, n_bootstrap=1000):
    df1 = scenario_dfs[scenario1]
    df2 = scenario_dfs[scenario2]
    n = len(df1)

    point_differences = get_scenario_differences(scenario_dfs, scenario=scenario2, wtp=80)

    # Store bootstrap results
    bootstrap_deaths_averted = []
    bootstrap_dalys_averted = []
    bootstrap_cost_diff = []
    bootstrap_mortality_reduction = []
    bootstrap_accessed_oxygen = []
    bootstrap_icer_deaths = []
    bootstrap_icer_dalys = []
    bootstrap_inhbs = []
    bootstrap_inmbs = []

    # Bootstrap from original individual outcomes, with paired resampling
    for _ in range(n_bootstrap):
        # Sample indices to get same individuals from both scenarios
        indices = np.random.choice(n, size=n, replace=True)

        # Get bootstrap samples of original outcomes
        mortality_scenario1_resampled = df1['mortality_outcome'].astype(int).iloc[indices]
        mortality_scenario2_resampled = df2['mortality_outcome'].astype(int).iloc[indices]
        dalys_scenario1_resampled = df1['DALYs_discounted'].iloc[indices]
        dalys_scenario2_resampled = df2['DALYs_discounted'].iloc[indices]
        costs_scenario1_resampled = df1['total_costs'].iloc[indices]
        costs_scenario2_resampled = df2['total_costs'].iloc[indices]

        # bootstrap samples of oxygen need and oxygen provided
        need_oxygen = df2['oxygen_saturation'].iloc[indices].apply(lambda x: 1 if x == '<90%' else 0).sum()
        oxygen_provided = df2[[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx',
                               f'oxygen_provided_scenario_{scenario}_{dx_accuracy}_hw_dx']].iloc[indices].apply(
            lambda x: 1 if (x[0] == '2' or x[0] == '1b') and x[1] == True else 0, axis=1).sum()
        fup_oxygen_provided = df2[[f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx',
                               f'oxygen_provided_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx']].iloc[indices]\
            .apply(lambda x: 1 if (x[0] == '2' or x[0] == '1b') and x[1] == True else 0, axis=1).sum()

        access_to_oxygen = (oxygen_provided + fup_oxygen_provided) / need_oxygen
        bootstrap_accessed_oxygen.append(access_to_oxygen)

        # Calculate differences for this sample

        # First get individual-level paired differences
        mort_diff = mortality_scenario1_resampled - mortality_scenario2_resampled
        dalys_diff = dalys_scenario1_resampled - dalys_scenario2_resampled
        cost_diff = costs_scenario2_resampled - costs_scenario1_resampled

        # Then aggregate to population level
        total_mort_diff = mort_diff.sum()
        bootstrap_deaths_averted.append(total_mort_diff)

        total_dalys_diff = dalys_diff.sum()
        bootstrap_dalys_averted.append(total_dalys_diff)

        total_cost_diff = cost_diff.sum()
        bootstrap_cost_diff.append(total_cost_diff)

        # mortality_reduction
        mortality_reduction = mort_diff.mean() / mortality_scenario1_resampled.mean()
        bootstrap_mortality_reduction.append(mortality_reduction)

        # Calculate ICER and INHB using population totals
        if total_mort_diff != 0:
            icer_deaths = total_cost_diff / total_mort_diff
            bootstrap_icer_deaths.append(icer_deaths)
        if total_dalys_diff != 0:
            icer_dalys = total_cost_diff / total_dalys_diff
            bootstrap_icer_dalys.append(icer_dalys)

        # inhb = (((df1['mortality_outcome'].astype(int)*(1-np.exp(-0.03*(54.7 - df1['age_exact_years']))))/0.03) - (
        #     (df2['mortality_outcome'].astype(int)*(1-np.exp(-0.03*(54.7 - df2['age_exact_years']))))/0.03)) - (total_cost_diff/wtp)
        # bootstrap_inhbs.append(inhb)

        inhb = total_dalys_diff - (total_cost_diff/wtp)
        bootstrap_inhbs.append(inhb)
        inmb = (total_dalys_diff * wtp) - total_cost_diff
        bootstrap_inmbs.append(inmb)

        # check for normality
        skewness = stats.skew(bootstrap_icer_deaths)
        kurtosis = stats.kurtosis(bootstrap_icer_deaths)

    return {
        'deaths_averted': point_differences['deaths_averted'],
        'dalys_averted': point_differences['dalys_averted'],
        'incremental_cost': point_differences['incremental_cost'],
        'icer_deaths': point_differences['icer_deaths'],
        'icer_dalys': point_differences['icer_dalys'],
        'inhb': point_differences['inhb'],
        'inmb': point_differences['inmb'],
        'death_averted_mean': np.mean(bootstrap_deaths_averted),
        'death_averted_ci': np.percentile(bootstrap_deaths_averted, [2.5, 97.5]),
        'dalys_averted_mean': np.mean(bootstrap_dalys_averted),
        'dalys_averted_ci': np.percentile(bootstrap_dalys_averted, [2.5, 97.5]),
        'cost_diff_mean': np.mean(bootstrap_cost_diff),
        'cost_diff_ci': np.percentile(bootstrap_cost_diff, [2.5, 97.5]),
        'mortality_reduction_mean': np.mean(bootstrap_mortality_reduction),
        'mortality_reduction_ci': np.percentile(bootstrap_mortality_reduction, [2.5, 97.5]),
        'access_to_oxygen_mean': np.mean(bootstrap_accessed_oxygen),
        'access_to_oxygen_ci': np.percentile(bootstrap_accessed_oxygen, [2.5, 97.5]),
        'icer_deaths_mean': np.mean(bootstrap_icer_deaths),
        'bootstrap_normality': (skewness, kurtosis),
        'icer_deaths_ci': np.percentile(bootstrap_icer_deaths, [2.5, 97.5]),
        'icer_dalys_mean': np.mean(bootstrap_icer_dalys),
        'icer_dalys_ci': np.percentile(bootstrap_icer_dalys, [2.5, 97.5]),
        'inhb_mean': np.mean(bootstrap_inhbs),
        'inhb_ci': np.percentile(bootstrap_inhbs, [2.5, 97.5]),
        'inmb_mean': np.mean(bootstrap_inmbs),
        'inmb_ci': np.percentile(bootstrap_inmbs, [2.5, 97.5]),
    }

summary_statistics = {}
for scenario in scenarios[1:]:
    summary_statistics[scenario] = \
        calculate_economic_uncertainty(scenario1='baseline_ant', scenario2=scenario, wtp=80, n_bootstrap=1000)

scenario_statistics = summary_statistics
debug_point = 0


def get_frontier_points(scenario_statistics):
    """Get points for cost-effectiveness frontier."""
    # Extract mean costs and effects for each scenario
    # Add origin point (baseline)
    points = [{'scenario': 'baseline_ant', 'cost': 0, 'effect': 0}]

    # Extract mean costs and effects for each scenario
    for scenario in scenario_statistics:
        points.append({
            'scenario': scenario,
            'cost': scenario_statistics[scenario]['cost_diff_mean'],
            'effect': scenario_statistics[scenario]['dalys_averted_mean']
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
    # for i in range(2, len(frontier)):
    #     if (frontier[i]['cost'] / frontier[i]['effect']) < (frontier[i-1]['cost'] / frontier[i-1]['effect']):
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
    scenarios = list(scenario_statistics.keys())

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Find max y value (including CI upper bounds)
    max_cost = max(scenario_statistics[s]['cost_diff_ci'][1] for s in scenarios)
    # Find max and min x values for DALYs
    max_dalys = max(scenario_statistics[s]['dalys_averted_ci'][1] for s in scenarios)
    min_dalys = min(scenario_statistics[s]['dalys_averted_ci'][0] for s in scenarios)

    # Plot reference lines
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Colors for different scenarios

    # Define base colors for each scenario
    BASE_COLOURS = {
        'baseline_ant': {
            'base': '#004080', # Navy Blue
            'level2': '#1a5da0', # Cobalt Blue
            'level1b': '#3379c0',  # Royal Blue
            'level1a': '#4d96df',  # Steel Blue
            'level0': '#66b2ff'  # Cornflower Blue
        },
        'existing_psa': {
            'base': '#e65100', # Dark Orange
            'level2': '#ec670a', # Medium-Dark Orange
            'level1b': '#f37c13',  # Medium Orange
            'level1a': '#f9921d',  # Medium-Light Orange
            'level0': '#ffa726'  # Light Orange
        },
        'planned_psa': {
            'base': '#1b5e20', # Dark Green
            'level2': '#3e7c42', # Medium-Dark Green
            'level1b': '#609a64',  # Medium Green
            'level1a': '#83b885',  # Medium-Light Green
            'level0': '#a5d6a7'  # Light Green
        }
    }
    colours = BASE_COLOURS

    # Define markers by PO implementation level
    po_level_markers = {
        'base': 'o',  # circle
        'level2': 's',   # square
        'level1b': 'd',  # diamond
        'level1a': '*',  # star
        'level0': '^'   # triangle
    }

    # Add point for baseline_ant
    plt.scatter(0, 0, color=colours['baseline_ant']['base'], marker=po_level_markers['base'],
                label='baseline_ant')
    # Plot each scenario
    for scenario in scenarios:
        # Get pre-calculated statistics
        cost_stats = {'mean': scenario_statistics[scenario]['cost_diff_mean'],
                      'ci_lower': scenario_statistics[scenario]['cost_diff_ci'][0],
                      'ci_upper': scenario_statistics[scenario]['cost_diff_ci'][1]}
        daly_stats = {'mean': scenario_statistics[scenario]['dalys_averted_mean'],
                      'ci_lower': scenario_statistics[scenario]['dalys_averted_ci'][0],
                      'ci_upper': scenario_statistics[scenario]['dalys_averted_ci'][1]}

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
    plt.plot(frontier_x, frontier_y, 'k--', color='grey', alpha=0.2, label='CE frontier')

    # Get ICER value and format it
    n_points = len(frontier_points)
    for i in range(1, n_points):
        frontier_scenario = frontier_points[i]['scenario']
        icer_value = scenario_statistics[frontier_scenario]['icer_dalys_mean']
        icer_value_rounded = (np.round(icer_value)).astype(int)
        icer_text = f"ICER={icer_value_rounded}$/DALY"
        # Add ICER annotation
        plt.annotate(
            icer_text,
            xy=(scenario_statistics[frontier_scenario]['dalys_averted_mean'],
                scenario_statistics[frontier_scenario]['cost_diff_mean']-40000))

    # Set y-axis limits to match max cost
    plt.ylim(-max_cost * 0.1, max_cost * 1.1)  # Add 10% padding

    # Format y-axis to show values in 100,000s
    def y_fmt(y, pos):
        """Format y-axis to show values in full units, add commas"""
        return '{:,.0f}'.format(y)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(y_fmt))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(y_fmt))

    # Customize plot
    plt.xlabel('DALYs Averted')
    plt.ylabel('Incremental Cost ($)')
    plt.title('Cost-Effectiveness Plane')

    # Get the current handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Create custom labels dictionary (map original labels to new ones)
    custom_labels = {
        'baseline_ant': 'No_Ox_&_No_PO',
        'baseline_ant_with_po_level2': 'No_Ox_&_PO_1level',
        'baseline_ant_with_po_level1b': 'No_Ox_&_PO_2levels',
        'baseline_ant_with_po_level1a': 'No_Ox_&_PO_3levels',
        'baseline_ant_with_po_level0': 'No_Ox_&_PO_4levels',
        'existing_psa': 'Low_Ox_&_No_PO',
        'existing_psa_with_po_level2': 'Low_Ox_&_PO_1level',
        'existing_psa_with_po_level1b': 'Low_Ox_&_PO_2levels',
        'existing_psa_with_po_level1a': 'Low_Ox_&_PO_3levels',
        'existing_psa_with_po_level0': 'Low_Ox_&_PO_4levels',
        'planned_psa': 'High_Ox_&_No_PO',
        'planned_psa_with_po_level2': 'High_Ox_&_PO_1level',
        'planned_psa_with_po_level1b': 'High_Ox_&_PO_2levels',
        'planned_psa_with_po_level1a': 'High_Ox_&_PO_3levels',
        'planned_psa_with_po_level0': 'High_Ox_&_PO_4levels',
    }

    # Replace labels with custom ones where available
    new_labels = [custom_labels.get(label, label) for label in labels]
    frontier_handle = [handles[0]]
    frontier_label = [labels[0]]
    scenarios_handle = handles[1:]
    scenarios_label = new_labels[1:]

    # Position legend at the bottom of the plot
    legend_frontier = plt.gca().legend(
        frontier_handle, frontier_label, loc='upper center', bbox_to_anchor=(0.1, -0.1), ncol=1,
        frameon=False)
    plt.gca().add_artist(legend_frontier)
    plt.legend(scenarios_handle, scenarios_label, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5,
               frameon=False)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    return plt.gcf()

# Apply
fig = plot_ce_plane(scenario_statistics)
plt.show()

# Optionally save the figure
# fig.savefig('ce_plane.png', dpi=300, bbox_inches='tight')

# # # Save a dataframe with the key information for the paper
key_table = {}
cleaned_values = {}
for scenario in scenario_statistics.keys():
    # Initialize the dictionary for this scenario
    cleaned_values[scenario] = {}
    for key, values in scenario_statistics[scenario].items():
        if key.startswith(('dalys_averted', 'inhb')):
            rounded_values = (np.round(values / 100) * 100).astype(int)
            cleaned_values[scenario][key] = rounded_values
        elif key.startswith(('icer', 'deaths')):
            rounded_values = (np.round(values)).astype(int)
            cleaned_values[scenario][key] = rounded_values
        elif key.startswith(('cost', 'inmb')) or 'cost' in key:
            rounded_values = (np.round(values / 1000) * 1000).astype(int)
            cleaned_values[scenario][key] = rounded_values
        elif key.startswith('mortality'):
            if isinstance(values, np.ndarray):
                rounded_values = ['{:.2f}'.format(v*100) for v in values]
            else:
                rounded_values = '{:.2f}'.format(values*100)
            cleaned_values[scenario][key] = rounded_values
        elif key.startswith('access'):
            rounded_values = (np.round(values * 100)).astype(int)
            cleaned_values[scenario][key] = rounded_values
        else:
            cleaned_values[scenario][key] = values

# Create the table with key outputs with their 95% CI
key_table = pd.DataFrame(columns=['scenario', 'access_to_oxygen',
                                  'total_deaths', 'total_costs',
                                  'mortality_reduction', 'dalys_averted',
                                  'icer_dalys', 'inhb'])


def format_values_for_table(data_dict, key):
    mean = data_dict.get(f'{key}_mean', None)
    ci = data_dict.get(f'{key}_ci', None)

    if mean is not None and ci is not None and len(ci) == 2:
        return f"{mean} ({ci[0]}, {ci[1]})"
    return None

# Convert the dictionary to rows in the DataFrame
rows = []
for scenario, dict in cleaned_values.items():
    row = {
        'scenario': scenario,
        'access_to_oxygen': format_values_for_table(dict, key='access_to_oxygen'),
        'total_deaths': dict.get('total_deaths', None),
        'total_costs': dict.get('total_cost', None),
        'mortality_reduction': format_values_for_table(dict, key='mortality_reduction'),
        'dalys_averted': format_values_for_table(dict, key='dalys_averted'),
        'icer_dalys': format_values_for_table(dict, key='icer_dalys'),
        'inhb': format_values_for_table(dict, key='inhb'),
    }
    rows.append(row)

key_table = pd.DataFrame(rows)
