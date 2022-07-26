"""This script will generate a table that describes all the types of IncidentCase that are created, and the associated
risk of death for each when:
 * Not Treated
 * Treatment (antibiotic effectiveness = default, pulse oximeter & oxygen available = No)
 * Treatment (antibiotic effectiveness = default, pulse oximeter & oxygen available = Yes)
 * Treatment (antibiotic effectiveness = perfect, pulse oximeter & oxygen available = No)
 * Treatment (antibiotic effectiveness = default, pulse oximeter & oxygen available = Yes)
"""

from pathlib import Path
from typing import List

import pandas as pd

import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import DateOffset

from tlo import Date, Simulation, logging, Module
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    alri,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
    Metadata,
)
from tlo.methods.alri import (
    AlriCureEvent,
    AlriDeathEvent,
    AlriIncidentCase,
    AlriIncidentCase_Lethal_DangerSigns_Pneumonia,
    AlriIncidentCase_NonLethal_Fast_Breathing_Pneumonia,
    AlriLoggingEvent,
    AlriNaturalRecoveryEvent,
    AlriPollingEvent,
    AlriPropertiesOfOtherModules,
    HSI_Alri_Treatment,
    Models,
    _make_treatment_and_diagnosis_perfect,
    _make_high_risk_of_death, _make_hw_diagnosis_perfect,
)
from tlo.util import sample_outcome

MODEL_POPSIZE = 10_000
MIN_SAMPLE_OF_NEW_CASES = 100
NUM_REPS_FOR_EACH_CASE = 10


def get_sim(popsize):
    """Return a simulation (composed of only <5 years old) that has run for 0 days."""
    resourcefilepath = Path('./resources')
    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=0)

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
        AlriPropertiesOfOtherModules(),
    )
    sim.modules['Demography'].parameters['max_age_initial'] = 5
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=sim.start_date)
    return sim


# Create simulation (as characteristics in the population affect the chance of incidence with each pathogen)
sim1 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_perfect_diagnosis = sim1.modules['Alri']
_make_hw_diagnosis_perfect(alri_module_with_perfect_diagnosis)

sim2 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_perfect_treatment_and_diagnosis = sim2.modules['Alri']
_make_treatment_and_diagnosis_perfect(alri_module_with_perfect_treatment_and_diagnosis)

hsi = HSI_Alri_Treatment(module=alri_module_with_perfect_diagnosis, person_id=None)


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
                [(k, v) for k, v in sample_outcome(probs=probs_of_acquiring_pathogen, rng=alri_module_with_perfect_diagnosis.rng).items()]
            )

        # Return dataframe in which person_id is replaced with age and sex (ignoring variation in vaccine /
        # under-nutrition).
        return pd.DataFrame(columns=['person_id', 'pathogen'], data=new_alri)\
            .merge(sim1.population.props[['age_exact_years', 'sex']],
                   right_index=True, left_on=['person_id'], how='left')\
            .drop(columns=['person_id'])

    def char_of_incident_case(sex,
                              age_exact_years,
                              pathogen,
                              va_hib_all_doses=False,
                              va_pneumo_all_doses=False,
                              un_clinical_acute_malnutrition="well",
                              ) -> dict:
        """Return the characteristics that are determined by IncidentCase (over 1000 iterations), given an infection
        caused by the pathogen"""
        incident_case = AlriIncidentCase(module=alri_module_with_perfect_diagnosis, person_id=None, pathogen=None)

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
    overall_case_mix = []   # This will be the mix of cases, represeting the characterists of the case as well as the pathogen and who they are incident to

    for x in incident_case_mix.itertuples():
        overall_case_mix.extend(
            char_of_incident_case(sex=x.sex, age_exact_years=x.age_exact_years, pathogen=x.pathogen)
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
):
    """Return the percentage by which the treatment reduce the risk of death"""

    # Get Treatment classification
    classification_for_treatment_decision = hsi._get_disease_classification_for_treatment_decision(
        age_exact_years=age_exact_years,
        symptoms=symptoms,
        oxygen_saturation=oxygen_saturation,
        facility_level='2',  # <-- assumes that the diagnosis occurs at level '2'
        use_oximeter=oximeter_available,
    )

    imci_symptom_based_classification = hsi._get_imci_classification_based_on_symptoms(
        child_is_younger_than_2_months=(age_exact_years < 2.0 / 12.0),
        symptoms=symptoms,
    )

    ultimate_treatment = hsi._ultimate_treatment_indicated_for_patient(
        classification_for_treatment_decision=classification_for_treatment_decision,
        age_exact_years=age_exact_years
    )

    # "Treatment Fails" is the probability that a death is averted (if one is schedule)
    if treatment_perfect:
        alri_module = alri_module_with_perfect_treatment_and_diagnosis
    else:
        alri_module = alri_module_with_perfect_diagnosis

    treatment_fails = alri_module._prob_treatment_fails(
        antibiotic_provided=ultimate_treatment['antibiotic_indicated'],
        oxygen_provided=ultimate_treatment['oxygen_indicated'] if oxygen_available else False,
        imci_symptom_based_classification=imci_symptom_based_classification,
        SpO2_level=oxygen_saturation,
        disease_type=disease_type,
        complications=list(complications),
        symptoms=symptoms,
        hiv_infected_and_not_on_art=False,
        un_clinical_acute_malnutrition='well',
    )

    # Return percentage probability of treatment success
    return 100.0 * (1.0 - treatment_fails)


def generate_table():
    """Return table providing a representative case mix of persons with Alri, the intrinsic risk of death and the
    efficacy of treatment under different conditions."""

    # Get Case Mix
    df = generate_case_mix()

    # Consider risk of death for this person, intrinsically and under different conditions of treatments
    risk_of_death = list()
    for x in df.itertuples():
        risk_of_death.append({
            'prob_die_if_no_treatment': alri_module_with_perfect_diagnosis.models.prob_die_of_alri(
                age_exact_years=x.age_exact_years,
                sex=x.sex,
                pathogen=x.pathogen,
                disease_type=x.disease_type,
                SpO2_level=x.oxygen_saturation,
                complications=x.complications,
                danger_signs='danger_signs' in x.symptoms,
                un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
            ),

            'treatment_efficacy_if_perfect_treatment_and_with_oximeter_and_oxygen': treatment_efficacy(
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
            ),

            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen': treatment_efficacy(
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
            ),

            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen': treatment_efficacy(
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
            ),

            'classification_for_treatment_decision_with_oximeter': hsi._get_disease_classification_for_treatment_decision(
                age_exact_years=x.age_exact_years,
                symptoms=x.symptoms,
                oxygen_saturation=x.oxygen_saturation,
                facility_level='2',  # <-- assumes that the diagnosis occurs at level '2'
                use_oximeter=True,
            ),

            'classification_for_treatment_decision_without_oximeter': hsi._get_disease_classification_for_treatment_decision(
                age_exact_years=x.age_exact_years,
                symptoms=x.symptoms,
                oxygen_saturation=x.oxygen_saturation,
                facility_level='2',  # <-- assumes that the diagnosis occurs at level '2'
                use_oximeter=False,
            ),
        })
    return df.join(pd.DataFrame(risk_of_death))


def main():
    table = generate_table()

    print(table)

    def summarize_by(df: pd.DataFrame, by: List[str], columns: [List[str]]) -> pd.DataFrame:
        """Helper function returns dataframe that summarizes the dataframe provided using groupby, with by arguements,
        and provides columns as follows: [size-of-the-group, mean-of-column-1, mean-of-column-2, ...]"""
        return pd.DataFrame({'fraction': df.groupby(by=by).size()}).apply(lambda x: x/x.sum(), axis=0)\
            .join(df.groupby(by=by)[columns].mean())

    # Examine effectiveness of oximeter/oxygen by need for oxygen
    y = summarize_by(
        df=table,
        by=['oxygen_saturation'],
        columns=[
            'prob_die_if_no_treatment',
            'treatment_efficacy_if_perfect_treatment_and_with_oximeter_and_oxygen',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen',
            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen',
        ]
    )
    # ---> ** Answer is that the oxygen therapy does work, but the number of children with that <90% are 10% of entire case mix. **

    # Examine breakdown by disease classification
    y = summarize_by(
        df=table,
        by=['oxygen_saturation', 'classification_for_treatment_decision_with_oximeter'],
        columns=[
            'prob_die_if_no_treatment',
            'treatment_efficacy_if_perfect_treatment_and_with_oximeter_and_oxygen',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen',
            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen',
        ]
    )

    # --> Query the (lack of) treatment effect for those with hypoxia 90-92%
    # --> Query death rate for those that have cough_and_cold
    # --> Query whether a treatment for cough_or_cold without severe hypoxic, with no antibiotic, has no effect at all.

    # Examine danger_signs as a predictor of SpO2 < 90%
    x = pd.crosstab(
        pd.Series(table['oxygen_saturation'] == "<90%", name='SpO2<90%'),
        pd.Series(table['symptoms'].apply(lambda x: 'danger_signs' in x), name='has_danger_signs'),
        margins=True, normalize=True,
    )
    # --> Looking just at danger_signs picks-up half of those that really do have very low oxygen. The pulse_oximter
    #     picks up the other half.

    # todo X- reduction in risk due to oxygen is 25%; is the the right magnitude?
    # todo - trace it back to the underlying parameters




if __name__ == "__main__":
    main()

    # todo 5- [look at the hw classification done and the true classification]
