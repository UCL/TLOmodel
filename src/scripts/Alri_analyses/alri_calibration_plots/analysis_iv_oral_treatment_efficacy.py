"""This script will generate a table that describes a representative mix of all the IncidentCases that are created, and
 the associated diagnosis and risk of death for each under various conditions of treatments/non-treatment."""

from pathlib import Path
from typing import List
import os
import datetime
from openpyxl import load_workbook
import pickle

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tlo import Date, Simulation
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
    _make_low_risk_of_death
)
from tlo.util import sample_outcome

MODEL_POPSIZE = 150_000
MIN_SAMPLE_OF_NEW_CASES = 200
NUM_REPS_FOR_EACH_CASE = 20

# Date for saving the image for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

_facility_level = '2'  # <-- assumes that the diagnosis/treatment occurs at this level
seed_run_dfs = {}


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
        AlriPropertiesOfOtherModules(),
    )
    sim.modules['Demography'].parameters['max_age_initial'] = 5
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=sim.start_date)
    return sim


# Create simulation (This is needed to generate a population with representative characteristics
# and to initialise the Alri module.)

# Alri module with default values
sim0 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment = sim0.modules['Alri']
hsi_with_imperfect_diagnosis_and_imperfect_treatment = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment, person_id=None)

# Alri module with perfect diagnosis (and imperfect treatment)
sim1 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_perfect_diagnosis = sim1.modules['Alri']
_make_hw_diagnosis_perfect(alri_module_with_perfect_diagnosis)
# _make_low_risk_of_death(alri_module_with_perfect_diagnosis)
hsi_with_perfect_diagnosis = HSI_Alri_Treatment(module=alri_module_with_perfect_diagnosis, person_id=None)

# Alri module with perfect diagnosis and perfect treatment
sim2 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_perfect_treatment_and_diagnosis = sim2.modules['Alri']
_make_treatment_and_diagnosis_perfect(alri_module_with_perfect_treatment_and_diagnosis)


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
                                          'va_hib_all_doses', 'va_pneumo_all_doses',
                                          'un_clinical_acute_malnutrition']],
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
    un_clinical_acute_malnutrition,
    hiv_infected_and_not_on_art,
    duration_in_days_of_alri,
    oximeter_available,
    oxygen_available,
    treatment_perfect,
    hw_dx_perfect,
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
        age_exact_years=age_exact_years, symptoms=symptoms, oxygen_saturation=oxygen_saturation,
        facility_level=_facility_level, use_oximeter=oximeter_available,
        hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
        un_clinical_acute_malnutrition=un_clinical_acute_malnutrition)

    imci_symptom_based_classification = hsi._get_imci_classification_based_on_symptoms(
        child_is_younger_than_2_months=(age_exact_years < 2.0 / 12.0),
        symptoms=symptoms, facility_level=_facility_level, hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
        un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
    )

    # Get the treatment selected based on classification given
    ultimate_treatment = alri_module._ultimate_treatment_indicated_for_patient(
        classification_for_treatment_decision=classification_for_treatment_decision,
        age_exact_years=age_exact_years,
        use_oximeter=oximeter_available,
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
        pre_referral_oxygen='not_applicable',
    )

    # Return percentage probability of treatment success
    return 100.0 * (1.0 - treatment_fails)


def treatment_efficacy_for_each_classification(
    age_exact_years,
    symptoms,
    oxygen_saturation,
    disease_type,
    complications,
    un_clinical_acute_malnutrition,
    hiv_infected_and_not_on_art,
    oximeter_available,
    oxygen_available,
    classification_for_treatment
):
    # Decide which hsi configuration to use:
    hsi = hsi_with_perfect_diagnosis
    alri_module = alri_module_with_perfect_diagnosis  # (perfect diagnosis and normal treatment effects)

    def ultimate_treatment_for_classification(classification):
        classification_for_treatment_decision = classification
        ultimate_treatment = alri_module._ultimate_treatment_indicated_for_patient(
            classification_for_treatment_decision=classification_for_treatment_decision,
            age_exact_years=age_exact_years,
            oxygen_saturation=oxygen_saturation,
            facility_level=_facility_level
        )

        return ultimate_treatment

    # get the IMCI symptoms based classification
    imci_symptom_based_classification = hsi._get_imci_classification_based_on_symptoms(
        child_is_younger_than_2_months=(age_exact_years < 2.0 / 12.0),
        symptoms=symptoms, facility_level=_facility_level, hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
        un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
    )

    # apply TFs by oral/IV antibiotic +/- oxygen (proxy by classification)
    ultimate_treatment = ultimate_treatment_for_classification(classification=classification_for_treatment)

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
        pre_referral_oxygen='not_applicable',
        this_is_follow_up=False
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
            pre_referral_oxygen='not_applicable',
            this_is_follow_up=False
        ))

    # Return percentage probability of treatment success
    return 100.0 * (1.0 - treatment_fails)


def generate_table():
    """Return table providing a representative case mix of persons with Alri, the intrinsic risk of death and the
    efficacy of treatment under different conditions."""

    # Get Case Mix
    df = generate_case_mix()

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

            'treatment_efficacy_iv_antibiotics_with_oxygen':
                treatment_efficacy_for_each_classification(
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
                    classification_for_treatment='danger_signs_pneumonia'
                ),

            'treatment_efficacy_iv_antibiotics_without_oxygen':
                treatment_efficacy_for_each_classification(
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
                    classification_for_treatment='danger_signs_pneumonia'
                ),

            'treatment_efficacy_oral_antibiotics_only':
                treatment_efficacy_for_each_classification(
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
                    classification_for_treatment='chest_indrawing_pneumonia' if any(
                        s in ['chest_indrawing', 'danger_signs', 'respiratory_distress'] for s in x.symptoms)
                    else 'fast_breathing_pneumonia'
                ),

            'treatment_efficacy_no_antibiotics':
                treatment_efficacy_for_each_classification(
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
                    classification_for_treatment='cough_or_cold'
                ),

            # Classifications
            'classification_for_treatment_decision_with_oximeter_perfect_accuracy':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=_facility_level, use_oximeter=True, hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition),

            'classification_for_treatment_decision_without_oximeter_perfect_accuracy':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years, symptoms=x.symptoms, oxygen_saturation=x.oxygen_saturation,
                    facility_level=_facility_level, use_oximeter=False,
                    hiv_infected_and_not_on_art=x.hiv_not_on_art,
                    un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition)
        })

    return df.join(pd.DataFrame(risk_of_death))


if __name__ == "__main__":
    table = generate_table()
    table = table.assign(
        has_danger_signs=lambda df: df['symptoms'].apply(lambda x: 'danger_signs' in x),
        needs_oxygen=lambda df: df['oxygen_saturation'] == "<90%",
        pulmonary_complication=lambda df: df['complications'].apply(lambda x: any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax'] for e in x)),
        bacteraemia=lambda df: df['complications'].apply(lambda x: any(
            e in ['bacteraemia'] for e in x)),
        bacterial_infection=lambda df: df[['pathogen', 'bacterial_coinfection']].apply(lambda x: (
            (x[0] in sim0.modules['Alri'].pathogens['bacterial']) or pd.notnull(x[1])), axis=1),
        any_complication=lambda df: df['complications'].apply(lambda x: any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax', 'bacteraemia', 'hypoxaemia'] for e in x)),
        other_complications=lambda df: df['complications'].apply(lambda x: any(
            e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax', 'bacteraemia'] for e in x)),
        severe_pneumonia=lambda df: df['symptoms'].apply(lambda x: any(s in ['danger_signs', 'respiratory_distress'] for s in x)),
        all_severe=lambda df: df[['symptoms', 'oxygen_saturation']].apply(lambda x: any(s in ['danger_signs', 'respiratory_distress'] for s in x[0]) or x[1] == "<90%", axis=1),
    )

    # table.drop(table.loc[table['age_exact_years'] < 2.0 / 12.0].index, inplace=True)

    def summarize_by(df: pd.DataFrame, by: List[str], columns: [List[str]]) -> pd.DataFrame:
        """Helper function returns dataframe that summarizes the dataframe provided using groupby, with by arguements,
        and provides columns as follows: [size-of-the-group, mean-of-column-1, mean-of-column-2, ...]"""
        return pd.DataFrame({'fraction': df.groupby(by=by).size()}).apply(lambda x: x / x.sum(), axis=0) \
            .join(df.groupby(by=by)[columns].mean())


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


    # # # Treatment efficacy table # # #
    risk_of_death = summarize_by(
        df=table,
        # df=table[table[['complications', 'bacterial_infection']].apply(lambda x: len(x[0]) > 0 and x[1] == True, axis=1)],
        by=[
        'oxygen_saturation', 'has_danger_signs',
        #     'any_complication',
        #     'other_complications',
        #     'all_severe',
            # 'bacterial_infection',
            # 'bacteraemia',
            # 'pulmonary_complication', 'hv_inf', 'un_clinical_acute_malnutrition',
            'classification_for_treatment_decision_without_oximeter_perfect_accuracy',
            'disease_type'
        ],
        columns=[
            'prob_die_if_no_treatment',
            'treatment_efficacy_iv_antibiotics_with_oxygen',
            'treatment_efficacy_iv_antibiotics_without_oxygen',
            'treatment_efficacy_oral_antibiotics_only',
            'treatment_efficacy_no_antibiotics'
        ]
    ).assign(
        fraction_of_deaths=lambda df: (
            (df.fraction * df.prob_die_if_no_treatment) / (df.fraction * df.prob_die_if_no_treatment).sum()
        )
    )
    print(f"{risk_of_death=}")

    # -----------------------------
    # Data to save in excel
    df = pd.DataFrame(columns=[risk_of_death.columns])

    if not os.path.exists(f'output_treatment_efficacies_reduced_mort{datestamp}.xlsx'):
        df.to_excel(f'./output_treatment_efficacies_reduced_mort{datestamp}.xlsx', sheet_name=f'Treatment_eff', index=True)

    # new_data = pd.DataFrame(risk_of_death.items, index=risk_of_death.index, columns=[risk_of_death.columns])
    new_data = risk_of_death

    # Properly use ExcelWriter with existing workbook
    book = load_workbook(f'./output_treatment_efficacies_reduced_mort{datestamp}.xlsx')

    with pd.ExcelWriter(f'./output_treatment_efficacies_reduced_mort{datestamp}.xlsx', engine='openpyxl') as writer:
        writer.book = book
        writer.sheets = {ws.title: ws for ws in book.worksheets}
        new_data.to_excel(writer,
                          sheet_name=f'Treatment_eff',
                          header=True,
                          index=True)

    # Save the table output
    with open(f'debug_output_treatment_efficacies_reduced_mort{datestamp}.pkl', 'wb') as f:
        pickle.dump(table, f)

debug_point = 0
