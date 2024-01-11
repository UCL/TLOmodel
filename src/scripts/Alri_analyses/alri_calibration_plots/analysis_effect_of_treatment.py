"""This script will generate a table that describes a representative mix of all the IncidentCases that are created, and
 the associated diagnosis and risk of death for each under various conditions of treatments/non-treatment."""
import random
from pathlib import Path
from typing import List
import datetime
from tlo.util import random_date, sample_outcome

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from tlo import Date, Simulation
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import (
    alri,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)
from tlo.methods.alri import (
    AlriIncidentCase,
    AlriPollingEvent,
    AlriPropertiesOfOtherModules,
    HSI_Alri_Treatment,
    _make_hw_diagnosis_perfect,
    _make_treatment_and_diagnosis_perfect,
)
from tlo.util import sample_outcome

MODEL_POPSIZE = 15_000
MIN_SAMPLE_OF_NEW_CASES = 200
NUM_REPS_FOR_EACH_CASE = 20

_facility_level = '2'  # <-- assumes that the diagnosis/treatment occurs at this level

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
        AlriPropertiesOfOtherModules(),
    )
    sim.modules['Demography'].parameters['max_age_initial'] = 5
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=sim.start_date)
    return sim


# Create simulation (This is needed to generate a population with representative characteristics and to initialise the
# Alri module.)

# Alri module with default values
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
hsi_with_perfect_diagnosis_and_treatment = HSI_Alri_Treatment(module=alri_module_with_perfect_treatment_and_diagnosis, person_id=None)


def generate_case_mix() -> pd.DataFrame:
    """Generate table of all the cases that may be created"""

    def get_incident_case_mix() -> pd.DataFrame:
        """Return a representative mix of new incidence alri cases (age, sex of person and the pathogen)."""

        alri_polling_event = AlriPollingEvent(module=alri_module_with_perfect_diagnosis)

        # Get probabilities for each person of being infected with each pathogen
        probs_of_acquiring_pathogen = alri_polling_event.get_probs_of_acquiring_pathogen(
            interval_as_fraction_of_a_year=1.0)

        # Sample who is infected and with what pathogen
        new_alri = []
        while len(new_alri) < MIN_SAMPLE_OF_NEW_CASES:
            new_alri.extend(
                [(k, v) for k, v in
                 sample_outcome(probs=probs_of_acquiring_pathogen, rng=alri_module_with_perfect_diagnosis.rng).items()]
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
    un_clinical_acute_malnutrition,
    hiv_infected_and_not_on_art,
    # facility_level
):
    """Return the percentage by which the treatment reduce the risk of death"""
    # Decide which hsi configuration to use:
    if hw_dx_perfect:
        hsi = hsi_with_perfect_diagnosis
        alri_module = alri_module_with_perfect_diagnosis
    else:
        hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment
        alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment

    # Get Treatment classification
    classification_for_treatment_decision = hsi._get_disease_classification_for_treatment_decision(
        age_exact_years=age_exact_years, symptoms=symptoms, oxygen_saturation=oxygen_saturation, facility_level='2',
        use_oximeter=oximeter_available, hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
        un_clinical_acute_malnutrition=un_clinical_acute_malnutrition)

    imci_symptom_based_classification = alri_module_with_perfect_diagnosis.get_imci_classification_based_on_symptoms(
        child_is_younger_than_2_months=(age_exact_years < 2.0 / 12.0),
        symptoms=symptoms, facility_level='2', hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
        un_clinical_acute_malnutrition=un_clinical_acute_malnutrition)

    # Get the treatment selected based on classification given
    ultimate_treatment = alri_module._ultimate_treatment_indicated_for_patient(
        classification_for_treatment_decision=classification_for_treatment_decision,
        age_exact_years=age_exact_years,
        facility_level='2',
        oxygen_saturation=oxygen_saturation,
    )

    # Decide which alri_module configuration to use:
    if treatment_perfect:
        alri_module = alri_module_with_perfect_treatment_and_diagnosis
    else:
        alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment

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
        hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
        un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
        pre_referral_oxygen='not_applicable'
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
            hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
            un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
            pre_referral_oxygen='not_applicable'
        ))

    # Return percentage probability of treatment success
    return (100.0 * (1.0 - treatment_fails)), classification_for_treatment_decision


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
    for x in df.itertuples():
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

            # Treatment Efficacy with * PERFECT HW Diagnosis *
            'treatment_efficacy_if_perfect_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx': treatment_efficacy(
                # Information about the patient:
                age_exact_years=x.age_exact_years,
                symptoms=x.symptoms,
                oxygen_saturation=x.oxygen_saturation,
                disease_type=x.disease_type,
                complications=x.complications,
                hiv_infected_and_not_on_art=x.hiv_not_on_art,
                un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,

                # Information about the care that can be provided:
                oximeter_available=True,
                oxygen_available=True,
                treatment_perfect=True,
                hw_dx_perfect=True,
            )[0],

            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx': treatment_efficacy(
                # Information about the patient:
                age_exact_years=x.age_exact_years,
                symptoms=x.symptoms,
                oxygen_saturation=x.oxygen_saturation,
                disease_type=x.disease_type,
                complications=x.complications,
                hiv_infected_and_not_on_art=x.hiv_not_on_art,
                un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,

                # Information about the care that can be provided:
                oximeter_available=True,
                oxygen_available=True,
                treatment_perfect=False,
                hw_dx_perfect=True,
            )[0],

            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx': treatment_efficacy(
                # Information about the patient:
                age_exact_years=x.age_exact_years,
                symptoms=x.symptoms,
                oxygen_saturation=x.oxygen_saturation,
                disease_type=x.disease_type,
                complications=x.complications,
                hiv_infected_and_not_on_art=x.hiv_not_on_art,
                un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,

                # Information about the care that can be provided:
                oximeter_available=False,
                oxygen_available=False,
                treatment_perfect=False,
                hw_dx_perfect=True,
            )[0],

            'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx':
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
                    oximeter_available=True,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                )[0],

            'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx':
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
                    oximeter_available=False,
                    oxygen_available=True,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                )[0],

            # Treatment Efficacy with * IMPERFECT HW Diangosis *
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_imperfect_hw_dx': treatment_efficacy(
                # Information about the patient:
                age_exact_years=x.age_exact_years,
                symptoms=x.symptoms,
                oxygen_saturation=x.oxygen_saturation,
                disease_type=x.disease_type,
                complications=x.complications,
                hiv_infected_and_not_on_art=x.hiv_not_on_art,
                un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,

                # Information about the care that can be provided:
                oximeter_available=True,
                oxygen_available=True,
                treatment_perfect=False,
                hw_dx_perfect=False,
            )[0],

            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_imperfect_hw_dx':
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
                    oximeter_available=False,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                )[0],

            'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_imperfect_hw_dx':
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
                    oximeter_available=True,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                )[0],

            'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_imperfect_hw_dx':
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
                oximeter_available=False,
                oxygen_available=True,
                treatment_perfect=False,
                hw_dx_perfect=False,
            )[0],

            # Classifications
            'classification_for_treatment_decision_with_oximeter_perfect_accuracy':
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
                    oximeter_available=True,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                )[1],

            'classification_for_treatment_decision_without_oximeter_perfect_accuracy':
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
                    oximeter_available=False,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                )[1],

            'classification_for_treatment_decision_with_oximeter_imperfect_accuracy':
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
                    oximeter_available=True,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                )[1],

            'classification_for_treatment_decision_without_oximeter_imperfect_accuracy':
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
                    oximeter_available=False,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                )[1],

        })

    return df.join(pd.DataFrame(risk_of_death))


if __name__ == "__main__":
    table = generate_table()
    table = table.assign(
        has_danger_signs=lambda df: df['symptoms'].apply(lambda x: 'danger_signs' in x),
        needs_oxygen=lambda df: df['oxygen_saturation'] == "<90%",
    )


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
    truth = table["classification_for_treatment_decision_with_oximeter_perfect_accuracy"]


    def cross_tab(truth: pd.Series, dx: pd.Series):
        """Return cross-tab between truth and dx and count number of incongruent rows."""
        print(f"% cases mis-classified is {100 * (truth != dx).mean()}%")
        return pd.crosstab(truth, dx, normalize='index')


    # THEORETICAL "total error" that occurs without oximeter: TRUTH versus 'Classification without an oximeter'
    # (under perfect HW accuracy)
    xtab_vs_without_oximeter_perfect_hw_dx = cross_tab(
        truth=truth, dx=table['classification_for_treatment_decision_without_oximeter_perfect_accuracy'],
    )
    # 5%

    # REAL "total error" that occurs without oximeter: TRUTH versus 'Classification without an oximeter'
    # (under actual HW accuracy)
    xtab_vs_without_oximeter_imperfect_hw_dx = cross_tab(
        truth=truth, dx=table['classification_for_treatment_decision_without_oximeter_imperfect_accuracy'],
    )
    # 32%

    # REAL "total error" that occurs with oximeter: TRUTH versus 'Classification with an oximeter'
    # (under actual HW accuracy)
    xtab_vs_with_oximeter_imperfect_hw_dx = cross_tab(
        truth=truth, dx=table["classification_for_treatment_decision_with_oximeter_imperfect_accuracy"],
    )  # 24%

    print(f"\n{xtab_vs_without_oximeter_perfect_hw_dx=}"
          f"\n{xtab_vs_without_oximeter_imperfect_hw_dx=}"
          f"\n{xtab_vs_with_oximeter_imperfect_hw_dx=}"
          )

    # truth_danger_signs_pneumonia = \
    #     table["classification_for_treatment_decision_with_oximeter_perfect_accuracy"] == "danger_signs_pneumonia"
    #
    # correctly_dx_with_po_when_misdiagnosed_without_imperfect_accuracy = len(
    #     table.loc[truth_danger_signs_pneumonia
    #               & (table["classification_for_treatment_decision_with_oximeter_imperfect_accuracy"]
    #                  == "danger_signs_pneumonia")
    #               & (table["classification_for_treatment_decision_without_oximeter_imperfect_accuracy"]
    #                  != "danger_signs_pneumonia")
    #               ]
    # ) / len(table.loc[truth_danger_signs_pneumonia])  # 32%
    #
    # correctly_dx_with_po_when_misdiagnosed_without_perfect_accuracy = len(
    #     table.loc[truth_danger_signs_pneumonia
    #               & (table["classification_for_treatment_decision_with_oximeter_perfect_accuracy"]
    #                  == "danger_signs_pneumonia")
    #               & (table["classification_for_treatment_decision_without_oximeter_perfect_accuracy"]
    #                  != "danger_signs_pneumonia")
    #               ]
    # ) / len(table.loc[truth_danger_signs_pneumonia])  # 12%
    #
    # print(f"\n{correctly_dx_with_po_when_misdiagnosed_without_imperfect_accuracy}"
    #       f"\n{correctly_dx_with_po_when_misdiagnosed_without_perfect_accuracy}")

    # Examine risk of death and treatment effectiveness by disease classification and oxygen saturation
    # And include estimate of % of deaths by scaling using "prob_die_if_no_treatment".
    # UNDER PERFECT HW DX
    risk_of_death = summarize_by(
        df=table,
        by=['oxygen_saturation', 'classification_for_treatment_decision_without_oximeter_perfect_accuracy'],
        columns=[
            'prob_die_if_no_treatment',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx',
            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx',
            'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx',
        ]
    ).assign(
        fraction_of_deaths=lambda df: (
            (df.fraction * df.prob_die_if_no_treatment) / (df.fraction * df.prob_die_if_no_treatment).sum()
        )
    )
    print(f"{risk_of_death=}")

    # risk of deaths if no treatment
    (risk_of_death.fraction * risk_of_death.prob_die_if_no_treatment).sum()  # 0.0342 -- 3.42%

    # risk of death with treatment with oximeter_and_oxygen_perfect_hw_dx
    (risk_of_death.fraction * risk_of_death.prob_die_if_no_treatment * (
        (100 - risk_of_death.treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx) / 100)
     ).sum()  # 0.00604 -- 0.06% deaths


    # UNDER PERFECT HW DX
    risk_of_death_imperfect_dx = summarize_by(
        df=table,
        by=['oxygen_saturation', 'classification_for_treatment_decision_without_oximeter_perfect_accuracy'],
        columns=[
            'prob_die_if_no_treatment',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_imperfect_hw_dx',
            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_imperfect_hw_dx',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_imperfect_hw_dx',
            'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_imperfect_hw_dx',
        ]
    ).assign(
        fraction_of_deaths=lambda df: (
            (df.fraction * df.prob_die_if_no_treatment) / (df.fraction * df.prob_die_if_no_treatment).sum()
        )
    )
    print(f"{risk_of_death_imperfect_dx=}")

    # ------
    # Look at where a person would receive a different diagnoses with/without oximeter
    diff_classification = table.loc[
        (table['classification_for_treatment_decision_with_oximeter_perfect_accuracy']
         != table['classification_for_treatment_decision_without_oximeter_perfect_accuracy']),
        [
            'age_exact_years',
            'symptoms',
            'classification_for_treatment_decision_with_oximeter_perfect_accuracy',
            'classification_for_treatment_decision_without_oximeter_perfect_accuracy',
            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx',
        ]
    ]

    # # In all cases, confirm they have exactly the same treatment effectiveness when no oxygen is available.
    # assert (
    #     diff_classification['treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx']
    #     ==
    #     diff_classification['treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx']
    # ).all()

    # # ... but that the availability of oxygen improves treatment effectiveness when there is a diff in diagnosis.
    # assert (
    #     diff_classification['treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx']
    #     >
    #     diff_classification['treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx']
    # ).all()

    # Overall summary figure: Number of deaths in the cohort Deaths broken down by ( disease / oxygen_saturation) when
    # * No Treatment
    # * Treatment antibiotics only (no oxygen, no oximeter)
    # * Treatment with antibiotics and oxygen (no oximeter)
    # * Treatment with antibiotics and oxygen and use of oximeter
    # ... under assumptions of (i) normal treatment and normal dx accuracy; (ii) normal treatment and perfect dx
    # accuracy

    disease_classification = table["classification_for_treatment_decision_with_oximeter_perfect_accuracy"]
    # disease_classification = table['classification_for_treatment_decision_without_oximeter_perfect_accuracy']
    low_oxygen = (table["oxygen_saturation"] == "<90%").replace({True: '<90%', False: ">=90%"})
    # low_oxygen = (table["oxygen_saturation"])
    fraction = risk_of_death['fraction']
    number_cases = table.groupby(by=[disease_classification, low_oxygen]).size()

    res = {
        "Perfect HW Dx Accuracy": {
            # "No Treatment": table['prob_die_if_no_treatment'].groupby(by=[disease_classification, low_oxygen]).sum(),
            "Antiobiotics only": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table['treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx'] / 100.0
                 )).groupby(by=[disease_classification, low_oxygen]).sum(),
            "+ oxygen": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table['treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx']
                 / 100.0
                 )).groupby(by=[disease_classification, low_oxygen]).sum(),
            "+ oximeter": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx'] / 100.0
                 )).groupby(by=[disease_classification, low_oxygen]).sum(),
            "+ oximeter & oxygen": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table['treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx'] / 100.0
                 )).groupby(by=[disease_classification, low_oxygen]).sum(),
        },

        "Normal HW Dx Accuracy": {
            # "No Treatment": table['prob_die_if_no_treatment'].groupby(by=[disease_classification, low_oxygen]).sum(),
            "Antiobiotics only": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table['treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_imperfect_hw_dx'] / 100.0
                 )).groupby(by=[disease_classification, low_oxygen]).sum(),
            "+ oxygen": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table['treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_imperfect_hw_dx']
                 / 100.0
                 )).groupby(by=[disease_classification, low_oxygen]).sum(),
            "+ oximeter": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_imperfect_hw_dx'] / 100.0
                 )).groupby(by=[disease_classification, low_oxygen]).sum(),
            "+ oximeter & oxygen": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table['treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_imperfect_hw_dx'] / 100.0
                 )).groupby(by=[disease_classification, low_oxygen]).sum(),
        }
    }

    from matplotlib.collections import LineCollection
    import numpy as np

    results = (100_000 / len(table)) * pd.concat({k: pd.DataFrame(v) for k, v in res.items()}, axis=1)

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
        reorderlist = [('cough_or_cold', '>=90%'),
                       ('cough_or_cold', '<90%'),
                       ('fast_breathing_pneumonia', '>=90%'),
                       ('fast_breathing_pneumonia', '<90%'),
                       ('chest_indrawing_pneumonia', '>=90%'),
                       ('chest_indrawing_pneumonia', '<90%'),
                       ('danger_signs_pneumonia', '>=90%'),
                       ('danger_signs_pneumonia', '<90%')
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
                         ('ci-pneumonia', 'SpO2<90%'): 'mediumspringgreen',
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
                         ('ci-pneumonia', 'SpO2<90%'): 'mediumspringgreen',
                         ('ds-pneumonia', 'SpO2>=90%'): 'firebrick',
                         ('ds-pneumonia', 'SpO2<90%'): 'darksalmon'
                         }

    else:
        raise ValueError(f'Index size not recognised {results.index.size}')

    fig, axs = plt.subplots(ncols=2, nrows=1, sharey=True, constrained_layout=True)
    for i, ix in enumerate(results.columns.levels[0]):
        ax = axs[i]
        # results.loc[:, (ix, slice(None))].T.plot.bar(stacked=True, ax=ax, legend=False)
        results.loc[:, (ix, slice(None))].T.plot.bar(stacked=True, ax=ax, width=0.6, legend=False,
                                                     color=results.index.map(assign_colors))
        ax.set_xticklabels(results.loc[:, (ix, slice(None))].columns.levels[1])
        # ax.set_xlabel('Intervention package')
        ax.set_ylabel('Deaths per 100,000 cases of ALRI')
        ax.set_title(f"{ix}", fontsize=10)
        ax.grid(axis='y')
    handles, labels = axs[-1].get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), title='Case Type', loc='upper left', bbox_to_anchor=(1, 1),
              fontsize=7)
    # fig.suptitle('Deaths Under Different Interventions Combinations', fontsize=14, fontweight='semibold')
    fig.suptitle('IMCI care management at facility level 1a', fontsize=12,  fontweight='semibold')
    fig.show()
    # fig.savefig(Path('./outputs') / ('with no treatment hc current policy' + datestamp + ".pdf"), format='pdf')
    fig.savefig(Path('./outputs') / ('imperfect dx - intervention bars hc current policy' + datestamp + ".pdf"), format='pdf')
    plt.close(fig)

    # cfr
    number_cases = table.groupby(by=[disease_classification, low_oxygen]).size()
    will_die = (table['prob_die_if_no_treatment']).groupby(by=[disease_classification, low_oxygen]).sum()

    cfr = will_die / number_cases
    overall_cfr = (cfr * fraction).sum()  # should be ~ 5.2% (95%CI 4.9-5.5) ref: Agweyu et al.2018 (Lancet) - hospital only
    prop_death = will_die / will_die.sum()

    # get the stats of impact between interventions
    # base comparator - antibiotics only under perfect hw dx
    deaths_antibiotics_perfect_hw = (table['prob_die_if_no_treatment'] * (
        1.0 - table['treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx'] / 100.0)
                ).groupby(by=[disease_classification, low_oxygen]).sum()

    # pulse oximeter only under perfect hw dx
    deaths_po_only_perfect_hw = (table['prob_die_if_no_treatment'] * (
        1.0 - table['treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx'] / 100.0)
                                     ).groupby(by=[disease_classification, low_oxygen]).sum()

    # oxygen only under perfect hw dx
    deaths_ox_only_perfect_hw = (table['prob_die_if_no_treatment'] * (
        1.0 - table['treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx'] / 100.0)
                                 ).groupby(by=[disease_classification, low_oxygen]).sum()

    # oxygen only under perfect hw dx
    deaths_po_and_ox_perfect_hw = (table['prob_die_if_no_treatment'] * (
        1.0 - table['treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx'] / 100.0)
                                 ).groupby(by=[disease_classification, low_oxygen]).sum()

    impact_po = deaths_po_only_perfect_hw.mean() / deaths_antibiotics_perfect_hw.mean()  # 0.887704
    impact_ox = deaths_ox_only_perfect_hw.mean() / deaths_antibiotics_perfect_hw.mean()  # 0.76990223%
    impact_po_and_ox = deaths_po_and_ox_perfect_hw.mean() / deaths_antibiotics_perfect_hw.mean()  # 0.671578

    # Repeat for normal HW Dx
    # base comparator - antibiotics only under normal hw dx
    deaths_antibiotics_normal_hw = (table['prob_die_if_no_treatment'] * (
        1.0 - table['treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_imperfect_hw_dx'] / 100.0)
                                     ).groupby(by=[disease_classification, low_oxygen]).sum()

    # pulse oximeter only under perfect hw dx
    deaths_po_only_normal_hw = (table['prob_die_if_no_treatment'] * (
        1.0 - table['treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_imperfect_hw_dx'] / 100.0)
                                 ).groupby(by=[disease_classification, low_oxygen]).sum()

    # oxygen only under perfect hw dx
    deaths_ox_only_normal_hw = (table['prob_die_if_no_treatment'] * (
        1.0 - table['treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_imperfect_hw_dx'] / 100.0)
                                 ).groupby(by=[disease_classification, low_oxygen]).sum()

    # oxygen only under perfect hw dx
    deaths_po_and_ox_normal_hw = (table['prob_die_if_no_treatment'] * (
        1.0 - table['treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_imperfect_hw_dx'] / 100.0)
                                   ).groupby(by=[disease_classification, low_oxygen]).sum()

    # impact reduction on mortality
    impact_po_n = deaths_po_only_normal_hw.mean() / deaths_antibiotics_normal_hw.mean()  # 0.772817
    impact_ox_n = deaths_ox_only_normal_hw.mean() / deaths_antibiotics_normal_hw.mean()  # 0.0.912565
    impact_po_and_ox_n = deaths_po_and_ox_normal_hw.mean() / deaths_antibiotics_normal_hw.mean()  # 0.6535055


# get the CFR for each classification
    # cfr_no_treatment = res['Normal HW Dx Accuracy']['No Treatment'] / number_cases

    under2mo = table[["age_exact_years", 'classification_for_treatment_decision_with_oximeter_perfect_accuracy']].apply(
        lambda x: 1 if (x[0] < 1 / 6) and (x[1] == 'danger_signs_pneumonia') else 0, axis=1).sum()

    # -------------------------------------------------------------------------------------------------------------
    # Scale the treatment failure for 1 st line IV antibiotics for danger_signs_pneumonia - to get the baseline TF

    df = table.drop(table[table.age_exact_years < 1 / 6].index)

    df_ds_pneumonia = df.drop(df.index[df["classification_for_treatment_decision_without_oximeter_perfect_accuracy"] != 'danger_signs_pneumonia'])
    ds_pneumonia_total = len(df_ds_pneumonia)

    # get the no risk
    not_risked = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                                  'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if ((x[0] != '<90%') and ('danger_signs' not in x[1]) and (x[2] != 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                        (x[4] == 'well') and (x[5] == False)) else 0, axis=1).sum()
    # get with risk
    with_risk = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                                 'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if ((x[0] == '<90%') or ('danger_signs' in x[1]) or (x[2] == 'pneumonia') or any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) or
                        (x[4] != 'well') or (x[5] == True)) else 0, axis=1).sum()

    prop_not_risked = not_risked / ds_pneumonia_total  # 0.14831

    # get the SpO2 without any other signs
    below90SpO2_only = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                           'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] == '<90%') and ('danger_signs' not in x[1]) and (x[2] != 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == False) else 0, axis=1).sum()

    prop_below90SpO2_only = below90SpO2_only / ds_pneumonia_total  # 0.041495

    # get the danger signs without any other signs
    ds_only = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                  'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] != '<90%') and ('danger_signs' in x[1]) and (x[2] != 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == False) else 0, axis=1).sum()

    prop_ds_only = ds_only / ds_pneumonia_total  # 0.17009

    # get the CXR+ without any other signs
    abnormal_cxr_only = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                            'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] != '<90%') and ('danger_signs' not in x[1]) and (x[2] == 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == False) else 0, axis=1).sum()

    prop_abnormal_cxr_only = abnormal_cxr_only / ds_pneumonia_total  # 0.10039

    # get the pulmonary complications without any other signs
    pc_only = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                               'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] != '<90%') and ('danger_signs' not in x[1]) and (x[2] != 'pneumonia') and any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == False) else 0, axis=1).sum()

    prop_pc_only = pc_only / ds_pneumonia_total  # 0

    # get the SpO2 with danger sign
    below90SpO2_and_ds = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                                          'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] == '<90%') and ('danger_signs' in x[1]) and (x[2] != 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == False) else 0, axis=1).sum()

    prop_below90SpO2_and_ds = below90SpO2_and_ds / ds_pneumonia_total  # 0.0394

    # get the SpO2 with ds and pc
    below90SpO2_and_ds_and_pc = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                                                 'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] == '<90%') and ('danger_signs' in x[1]) and (x[2] != 'pneumonia') and any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == False) else 0, axis=1).sum()

    prop_below90SpO2_and_ds_and_pc = below90SpO2_and_ds_and_pc / ds_pneumonia_total  # 0.0

    # get SpO2 and CXR+
    below90SpO2_and_abnormal_cxr = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                                                    'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] == '<90%') and ('danger_signs' not in x[1]) and (x[2] == 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == False) else 0, axis=1).sum()

    prop_below90SpO2_and_abnormal_cxr = below90SpO2_and_abnormal_cxr / ds_pneumonia_total  # 0.03575

    below90SpO2_and_pc = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                                          'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] == '<90%') and ('danger_signs' not in x[1]) and (x[2] != 'pneumonia') and any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == False) else 0, axis=1).sum()

    prop_below90SpO2_pc = below90SpO2_and_pc / ds_pneumonia_total  # 0.0

    below90SpO2_and_abnormal_cxr_and_pc = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                                                           'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] == '<90%') and ('danger_signs' not in x[1]) and (x[2] == 'pneumonia') and any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == False) else 0, axis=1).sum()

    prop_below90SpO2_and_abnormal_cxr_and_pc = below90SpO2_and_abnormal_cxr_and_pc / ds_pneumonia_total  # 0.018688

    # get the SpO2 + danger sign + CXR+
    below90SpO2_and_ds_and_abnormal_cxr = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                                                           'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] == '<90%') and ('danger_signs' in x[1]) and (x[2] == 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == False) else 0, axis=1).sum()

    prop_below90SpO2_and_ds_and_abnormal_cxr = below90SpO2_and_ds_and_abnormal_cxr / ds_pneumonia_total  # 0.0348079

    # get the SpO2 + danger sign + CXR+ + pc
    below90SpO2_and_ds_and_abnormal_cxr_and_pc = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                                                                  'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] == '<90%') and ('danger_signs' in x[1]) and (x[2] == 'pneumonia') and any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == False) else 0, axis=1).sum()

    prop_below90SpO2_and_ds_and_abnormal_cxr_and_pc = below90SpO2_and_ds_and_abnormal_cxr_and_pc / ds_pneumonia_total

    # get the danger sign + CXR+
    ds_and_abnormal_cxr = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                                           'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] != '<90%') and ('danger_signs' in x[1]) and (x[2] == 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == False) else 0, axis=1).sum()

    prop_ds_and_abnormal_cxr = ds_and_abnormal_cxr / ds_pneumonia_total  # 0.114797

    # get the danger sign + pc
    ds_and_pc = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                                 'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] != '<90%') and ('danger_signs' in x[1]) and (x[2] != 'pneumonia') and any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == False) else 0, axis=1).sum()

    prop_ds_and_pc = ds_and_pc / ds_pneumonia_total  # 0.0

    # get danger sign + CXR+ + pc
    ds_and_abnormal_cxr_and_pc = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                                                  'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] != '<90%') and ('danger_signs' in x[1]) and (x[2] == 'pneumonia') and any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == False) else 0, axis=1).sum()

    prop_ds_and_abnormal_cxr_and_pc = ds_and_abnormal_cxr_and_pc / ds_pneumonia_total  # 0.05761

    # get CXR+ and pc
    abnormal_cxr_and_pc = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                                           'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] != '<90%') and ('danger_signs' not in x[1]) and (x[2] == 'pneumonia') and any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == False) else 0, axis=1).sum()

    prop_abnormal_cxr_and_pc = abnormal_cxr_and_pc / ds_pneumonia_total  # 0.0504115

    # Also add HIV, MAM, SAM
    # HIV not on ART
    hiv_only = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                                'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] != '<90%') and ('danger_signs' not in x[1]) and (x[2] != 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == True) else 0, axis=1).sum()

    prop_hiv_only = hiv_only / ds_pneumonia_total  # 0.018775

    # HIV not on ART, and SpO2<90%
    hiv_and_below90SpO2 = df_ds_pneumonia[['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
                                'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] == '<90%') and ('danger_signs' not in x[1]) and (x[2] != 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == True) else 0, axis=1).sum()

    prop_hiv_and_below90SpO2 = hiv_and_below90SpO2 / ds_pneumonia_total  # 0.00462969

    # HIV not on ART, and danger signs
    hiv_and_ds = df_ds_pneumonia[
        ['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
         'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] != '<90%') and ('danger_signs' in x[1]) and (x[2] != 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == True) else 0, axis=1).sum()

    prop_hiv_and_ds = hiv_and_ds / ds_pneumonia_total  # 0.020919

    # # HIV not on ART, and SpO2<90%, danger signs
    hiv_and_below90SpO2_and_ds = df_ds_pneumonia[
        ['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
         'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] == '<90%') and ('danger_signs' in x[1]) and (x[2] != 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == True) else 0, axis=1).sum()

    prop_hiv_and_below90SpO2_and_ds = hiv_and_below90SpO2_and_ds / ds_pneumonia_total  # 0.005744

    # # HIV not on ART, CXR+
    hiv_and_abnormal_cxr = df_ds_pneumonia[
        ['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
         'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] != '<90%') and ('danger_signs' not in x[1]) and (x[2] == 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == True) else 0, axis=1).sum()

    prop_hiv_and_abnormal_cxr = hiv_and_abnormal_cxr / ds_pneumonia_total  # 0.0112311

    # # HIV not on ART, CXR+, ds
    hiv_and_ds_and_abnormal_cxr = df_ds_pneumonia[
        ['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
         'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] != '<90%') and ('danger_signs' in x[1]) and (x[2] == 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == True) else 0, axis=1).sum()

    prop_hiv_and_ds_and_abnormal_cxr = hiv_and_ds_and_abnormal_cxr / ds_pneumonia_total  # 0.014231

    # # HIV not on ART, and SpO2<90%, CXR+
    hiv_and_below90SpO2_and_abnormal_cxr = df_ds_pneumonia[
        ['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
         'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] == '<90%') and ('danger_signs' not in x[1]) and (x[2] == 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == True) else 0, axis=1).sum()

    prop_hiv_and_below90SpO2_and_abnormal_cxr = hiv_and_below90SpO2_and_abnormal_cxr / ds_pneumonia_total  # 0.003343

    # # HIV not on ART, and PC
    hiv_and_pc = df_ds_pneumonia[
        ['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
         'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] != '<90%') and ('danger_signs' not in x[1]) and (x[2] != 'pneumonia') and any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == True) else 0, axis=1).sum()

    prop_hiv_and_pc = hiv_and_pc / ds_pneumonia_total  # 0.0

    # # HIV not on ART, ds, and PC
    hiv_and_ds_and_pc = df_ds_pneumonia[
        ['oxygen_saturation', 'symptoms', 'disease_type', 'complications',
         'un_clinical_acute_malnutrition', 'hiv_not_on_art']].apply(
        lambda x: 1 if (x[0] != '<90%') and ('danger_signs' in x[1]) and (x[2] != 'pneumonia') and any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[3]) and
                       (x[4] == 'well') and (x[5] == True) else 0, axis=1).sum()

    prop_hiv_and_ds_and_pc = hiv_and_ds_and_pc / ds_pneumonia_total  # 0.0


    risk_cases = abnormal_cxr_and_pc + \
                 ds_and_abnormal_cxr_and_pc + \
                 ds_and_abnormal_cxr + \
                 below90SpO2_and_ds_and_abnormal_cxr_and_pc + \
                 below90SpO2_and_ds_and_abnormal_cxr + \
                 below90SpO2_and_ds + \
                 below90SpO2_and_pc + \
                 below90SpO2_and_abnormal_cxr + \
                 pc_only + \
                 abnormal_cxr_only + \
                 ds_only + \
                 below90SpO2_only + \
                 ds_and_pc + \
                 below90SpO2_and_abnormal_cxr_and_pc + below90SpO2_and_ds_and_pc

    # unscaled TF
    unscaled_tf_no_risk = 0.171
    unscaled_tf_below90SpO2_only = unscaled_tf_no_risk * 1.28
    unscaled_tf_ds_only = unscaled_tf_no_risk * 1.55
    unscaled_tf_abnormal_cxr_only = unscaled_tf_no_risk * 1.71
    unscaled_tf_pc_only = unscaled_tf_no_risk * 2.31
    unscaled_tf_below90SpO2_and_ds = unscaled_tf_no_risk * 1.28 * 1.55
    unscaled_tf_below90SpO2_and_abnormal_cxr = unscaled_tf_no_risk * 1.28 * 1.71
    unscaled_tf_below90SpO2_and_pc = unscaled_tf_no_risk * 1.28 * 2.31
    unscaled_tf_below90SpO2_and_ds_and_abnormal_cxr = unscaled_tf_no_risk * 1.28 * 1.55 * 1.71
    unscaled_tf_below90SpO2_and_ds_and_abnormal_cxr_and_pc = unscaled_tf_no_risk * 1.28 * 1.55 * 1.71 * 2.31
    unscaled_tf_below90SpO2_and_abnormal_cxr_and_pc = unscaled_tf_no_risk * 1.28 * 1.71 * 2.31
    unscaled_tf_below90SpO2_and_ds_and_pc = unscaled_tf_no_risk * 1.28 * 1.55 * 2.31
    unscaled_tf_ds_and_pc = unscaled_tf_no_risk * 1.55 * 2.31
    unscaled_tf_ds_and_abnormal_cxr = unscaled_tf_no_risk * 1.55 * 1.71
    unscaled_tf_ds_and_abnormal_cxr_and_pc = unscaled_tf_no_risk * 1.55 * 1.71 * 2.31
    unscaled_tf_abnormal_cxr_and_pc = unscaled_tf_no_risk * 1.71 * 2.31

    total_unscaled = ((unscaled_tf_no_risk * (ds_pneumonia_total - risk_cases)) + (
        unscaled_tf_below90SpO2_only * below90SpO2_only) + (
                          unscaled_tf_ds_only * ds_only) + (
                          unscaled_tf_abnormal_cxr_only * abnormal_cxr_only) + (
                          unscaled_tf_pc_only * pc_only) + (
                          unscaled_tf_below90SpO2_and_ds * below90SpO2_and_ds) + (
                          unscaled_tf_below90SpO2_and_abnormal_cxr * below90SpO2_and_abnormal_cxr) + (
                          unscaled_tf_below90SpO2_and_pc * below90SpO2_and_pc) + (unscaled_tf_ds_and_pc * ds_and_pc) + (
                          unscaled_tf_below90SpO2_and_ds_and_abnormal_cxr * below90SpO2_and_ds_and_abnormal_cxr) + (
                          unscaled_tf_below90SpO2_and_abnormal_cxr_and_pc * below90SpO2_and_abnormal_cxr_and_pc) + (
                          unscaled_tf_below90SpO2_and_ds_and_pc + below90SpO2_and_ds_and_pc) + (
                          unscaled_tf_below90SpO2_and_ds_and_abnormal_cxr_and_pc * below90SpO2_and_ds_and_abnormal_cxr_and_pc) + (
                          unscaled_tf_ds_and_abnormal_cxr * ds_and_abnormal_cxr) + (
                          unscaled_tf_ds_and_abnormal_cxr_and_pc * ds_and_abnormal_cxr_and_pc) + (
                          unscaled_tf_abnormal_cxr_and_pc * abnormal_cxr_and_pc)) / ds_pneumonia_total

    scaling_factor = 0.171 / total_unscaled
    baseline_tf = 0.171 * scaling_factor  # base line TF is 0.0684

    def scaled_baseline_tf(df):
        """" scale the treatment failure of IV antibiotics for danger signs pneumonia """

        unscaled_lm = \
            LinearModel(
                LinearModelType.MULTIPLICATIVE,
                0.171,
                Predictor('disease_type', external=True).when('pneumonia', 1.71),
                Predictor('complications', external=True).when(True, 2.31),
                Predictor('hiv_not_on_art', external=True).when(True, 1.8),
                Predictor('un_clinical_acute_malnutrition', external=True).when('MAM', 1.48),
                Predictor('un_clinical_acute_malnutrition', external=True).when('SAM', 2.02),
                Predictor('oxygen_saturation', external=True).when(True, 1.28),
                Predictor('symptoms', external=True).when(True, 1.55),
                # Predictor('referral_hc', external=True).when(True, 1.66),
            )

        # make unscaled linear model
        unscaled_tf = unscaled_lm.predict(df,
                                          disease_type=df['disease_type'],
                                          complications=any(c in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for c in df.complications),
                                          hiv_not_on_art=df['hiv_not_on_art'],
                                          un_clinical_acute_malnutrition=df['un_clinical_acute_malnutrition'],
                                          oxygen_saturation=df.oxygen_saturation == '<90%',
                                          symptoms='danger_signs' in df.symptoms,
                                          # referral_hc=any(l in ['0', '1a'] for l in df.seek_level) and
                                          #             df.classification_for_treatment_decision_without_oximeter_perfect_accuracy == 'danger_signs_pneumonia'
                                          )

        cases_combo = unscaled_tf.unique()
        total_each_combo = unscaled_tf.value_counts()
        total_unscale_tf = (cases_combo * total_each_combo).sum() / total_each_combo.sum()

        scaling_factor = 0.171 / total_unscale_tf

        scaled_baseline_tf = 0.171 * scaling_factor

        return scaled_baseline_tf

    get_base_tf = scaled_baseline_tf(df=df_ds_pneumonia)

    def get_prob_of_outcome_in_baseline_group(or_value, prob_ref, prop_case_group, prevalence) -> dict:
        """
        Helper function to convert odds ratio (OR) to risk ratio (RR) and adjust for the overall prevalence,
        it returns the probability of outcome in the unexposed group (reference group)
        :param or_value: odds ratio of case group for the outcome
        :param prob_ref: prevalence of outcome in reference group
        :param prop_case_group: proportion of case group (with the outcome) over total (case + ref group)
        :param prevalence: overall prevalence (joined groups)
        :return: returns the risk ratio ('rr'), and the baseline probability of outcome in the reference group
        """

        # Convert OR to RR with the following equation
        rr = or_value / ((1 - prob_ref) + (prob_ref * or_value))

        # adjust the probability values using the RR, the two group proportions
        # and the overall prevalence of the risk factor / outcome.
        adjusted_p = prevalence / (prop_case_group * rr + (1 - prop_case_group))

        return dict({'rr': rr, 'adjusted_p': adjusted_p})


    get_rr_death_referral_hc = get_prob_of_outcome_in_baseline_group(
        or_value=1.72, prob_ref=0.050284, prop_case_group=0.242, prevalence=0.06376822)  # 1.65990

    # ----------------------------------------------------------------------------------------------------------
    # Now get the baseline treatment failure for oral antibiotics to treat non-severe pneumonia

    fb_pneumonia = (
        df["classification_for_treatment_decision_with_oximeter_perfect_accuracy"] == 'fast_breathing_pneumonia').sum()
    ci_pneumonia = (
        df["classification_for_treatment_decision_with_oximeter_perfect_accuracy"] == 'chest_indrawing_pneumonia').sum()

    non_sev_pneumonia = fb_pneumonia + ci_pneumonia

    classification = ['fast_breathing_pneumonia', 'chest_indrawing_pneumonia']

    # get the no risk
    non_sev_not_risked = df[['oxygen_saturation', 'disease_type', 'complications',
                             'classification_for_treatment_decision_with_oximeter_perfect_accuracy']].apply(
        lambda x: 1 if ((x[0] == '>=93%') and (x[1] != 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[2])) and
                       (x[3] == 'fast_breathing_pneumonia' or x[3] == 'chest_indrawing_pneumonia') else 0, axis=1).sum()
    # get with risk
    non_sev_with_risk = df[['oxygen_saturation', 'disease_type', 'complications',
                            'classification_for_treatment_decision_with_oximeter_perfect_accuracy']].apply(
        lambda x: 1 if ((x[0] != '>=93%') or (x[1] == 'pneumonia') or any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[2])) and
                       (x[3] == 'fast_breathing_pneumonia' or x[3] == 'chest_indrawing_pneumonia') else 0, axis=1).sum()

    prop_non_sev_not_risked = non_sev_not_risked / non_sev_pneumonia

    # get the SpO2 without any other signs
    non_sev_below93SpO2_only = df[['oxygen_saturation', 'disease_type', 'complications',
                                   'classification_for_treatment_decision_with_oximeter_perfect_accuracy']].apply(
        lambda x: 1 if (x[0] != '>=93%') and (x[1] != 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[2]) and
                       (x[3] == 'fast_breathing_pneumonia' or x[3] == 'chest_indrawing_pneumonia') else 0, axis=1).sum()

    prop_non_sev_below93SpO2_only = non_sev_below93SpO2_only / non_sev_pneumonia

    # get the CXR+ without any other signs
    non_sev_abnormal_cxr_only = df[['oxygen_saturation', 'disease_type', 'complications',
                                    'classification_for_treatment_decision_with_oximeter_perfect_accuracy']].apply(
        lambda x: 1 if (x[0] == '>=93%') and (x[1] == 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[2]) and
                       (x[3] == 'fast_breathing_pneumonia' or x[3] == 'chest_indrawing_pneumonia') else 0, axis=1).sum()

    prop_non_sev_abnormal_cxr_only = non_sev_abnormal_cxr_only / non_sev_pneumonia

    # get the pulmonary complications without any other signs
    non_sev_pc_only = df[['oxygen_saturation', 'disease_type', 'complications',
                          'classification_for_treatment_decision_with_oximeter_perfect_accuracy']].apply(
        lambda x: 1 if (x[0] == '>=93%') and (x[1] != 'pneumonia') and any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[2]) and
                       (x[3] == 'fast_breathing_pneumonia' or x[3] == 'chest_indrawing_pneumonia') else 0, axis=1).sum()

    prop_non_sev_pc_only = non_sev_pc_only / non_sev_pneumonia

    # get the SpO2 with ds and pc
    non_sev_below93SpO2_and_pc = df[['oxygen_saturation', 'disease_type', 'complications',
                                     'classification_for_treatment_decision_with_oximeter_perfect_accuracy']].apply(
        lambda x: 1 if (x[0] != '>=93%') and (x[1] != 'pneumonia') and any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[2]) and
                       (x[3] == 'fast_breathing_pneumonia' or x[3] == 'chest_indrawing_pneumonia') else 0, axis=1).sum()

    prop_non_sev_below93SpO2_and_pc = non_sev_below93SpO2_and_pc / non_sev_pneumonia

    # get SpO2 and CXR+
    non_sev_below93SpO2_and_abnormal_cxr = df[['oxygen_saturation', 'disease_type', 'complications',
                                               'classification_for_treatment_decision_with_oximeter_perfect_accuracy']].apply(
        lambda x: 1 if (x[0] != '>=93%') and (x[1] == 'pneumonia') and not any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[2]) and
                       (x[3] == 'fast_breathing_pneumonia' or x[3] == 'chest_indrawing_pneumonia') else 0, axis=1).sum()

    prop_non_sev_below93SpO2_and_abnormal_cxr = non_sev_below93SpO2_and_abnormal_cxr / non_sev_pneumonia

    # get the SpO2 <93 + CXR+ + PC
    non_sev_below93SpO2_and_abnormal_cxr_and_pc = df[
        ['oxygen_saturation', 'disease_type', 'complications',
         'classification_for_treatment_decision_with_oximeter_perfect_accuracy']].apply(
        lambda x: 1 if (x[0] != '>=93%') and (x[1] == 'pneumonia') and any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[2]) and
                       (x[3] == 'fast_breathing_pneumonia' or x[3] == 'chest_indrawing_pneumonia') else 0, axis=1).sum()

    prop_non_sev_below93SpO2_and_abnormal_cxr_and_pc = non_sev_below93SpO2_and_abnormal_cxr_and_pc / non_sev_pneumonia

    # get CXR+ + pc
    non_sev_abnormal_cxr_and_pc = df[['oxygen_saturation', 'disease_type', 'complications',
                                      'classification_for_treatment_decision_with_oximeter_perfect_accuracy']].apply(
        lambda x: 1 if (x[0] == '>=93%') and (x[1] == 'pneumonia') and any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x[2]) and
                       (x[3] == 'fast_breathing_pneumonia' or x[3] == 'chest_indrawing_pneumonia') else 0, axis=1).sum()

    prop_non_sev_abnormal_cxr_and_pc = non_sev_abnormal_cxr_and_pc / non_sev_pneumonia

    non_sev_risk_cases = non_sev_abnormal_cxr_and_pc + \
                         non_sev_below93SpO2_and_abnormal_cxr_and_pc + \
                         non_sev_below93SpO2_and_abnormal_cxr + \
                         non_sev_below93SpO2_only + \
                         non_sev_below93SpO2_and_pc + \
                         non_sev_pc_only + \
                         non_sev_abnormal_cxr_only

    # unscaled TF
    unscaled_tf_non_sev_no_risk = 0.108
    unscaled_tf_non_sev_below93SpO2_only = unscaled_tf_non_sev_no_risk * 1.695
    unscaled_tf_non_sev_abnormal_cxr_only = unscaled_tf_non_sev_no_risk * 1.71
    unscaled_tf_non_sev_pc_only = unscaled_tf_non_sev_no_risk * 2.31
    unscaled_tf_non_sev_below93SpO2_and_abnormal_cxr = unscaled_tf_non_sev_no_risk * 1.695 * 1.71
    unscaled_tf_non_sev_below93SpO2_and_pc = unscaled_tf_non_sev_no_risk * 1.695 * 2.31
    unscaled_tf_non_sev_below93SpO2_and_abnormal_cxr_and_pc = unscaled_tf_non_sev_no_risk * 1.695 * 1.71 * 2.31
    unscaled_tf_non_sev_abnormal_cxr_and_pc = unscaled_tf_non_sev_no_risk * 1.71 * 2.31

    total_unscaled_non_sev = ((unscaled_tf_non_sev_no_risk * non_sev_not_risked) + (
        unscaled_tf_non_sev_below93SpO2_only * non_sev_below93SpO2_only) + (
                                  unscaled_tf_non_sev_abnormal_cxr_only * non_sev_abnormal_cxr_only) + (
                                  unscaled_tf_non_sev_pc_only * non_sev_pc_only) + (
                                  unscaled_tf_non_sev_below93SpO2_and_abnormal_cxr * non_sev_below93SpO2_and_abnormal_cxr) + (
                                  unscaled_tf_non_sev_below93SpO2_and_pc * non_sev_below93SpO2_and_pc) + (
                                  unscaled_tf_non_sev_below93SpO2_and_abnormal_cxr_and_pc * non_sev_below93SpO2_and_abnormal_cxr_and_pc) + (
                                  unscaled_tf_non_sev_abnormal_cxr_and_pc * non_sev_abnormal_cxr_and_pc)) / non_sev_pneumonia

    scaling_factor_non_sev = 0.1045 / total_unscaled_non_sev
    baseline_tf_oral_antibiotics = 0.1045 * scaling_factor_non_sev  # base line TF is 0.0684
