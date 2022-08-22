"""This script will generate a table that describes a representative mix of all the IncidentCases that are created, and
 the associated diagnosis and risk of death for each under various conditions of treatments/non-treatment."""

from pathlib import Path
from typing import List

import pandas as pd
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
)
from tlo.util import sample_outcome

MODEL_POPSIZE = 15_000
MIN_SAMPLE_OF_NEW_CASES = 200
NUM_REPS_FOR_EACH_CASE = 20

_facility_level = '2'  # <-- assumes that the diagnosis/treatment occurs at this level


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
                 sample_outcome(probs=probs_of_acquiring_pathogen, rng=alri_module_with_perfect_diagnosis.rng).items()]
            )

        # Return dataframe in which person_id is replaced with age and sex (ignoring variation in vaccine /
        # under-nutrition).
        return pd.DataFrame(columns=['person_id', 'pathogen'], data=new_alri) \
            .merge(sim1.population.props[['age_exact_years', 'sex']],
                   right_index=True, left_on=['person_id'], how='left') \
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
    overall_case_mix = []  # This will be the mix of cases, representing the characteristics of the case as well as the
    #                        pathogen and who they are incident to.

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
    hw_dx_perfect,
):
    """Return the percentage by which the treatment reduce the risk of death"""
    # Decide which hsi configuration to use:
    if hw_dx_perfect:
        hsi = hsi_with_perfect_diagnosis
    else:
        hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment

    # Get Treatment classification
    classification_for_treatment_decision = hsi._get_disease_classification_for_treatment_decision(
        age_exact_years=age_exact_years,
        symptoms=symptoms,
        oxygen_saturation=oxygen_saturation,
        facility_level=_facility_level,
        use_oximeter=oximeter_available,
    )

    imci_symptom_based_classification = hsi._get_imci_classification_based_on_symptoms(
        child_is_younger_than_2_months=(age_exact_years < 2.0 / 12.0),
        symptoms=symptoms,
    )

    ultimate_treatment = alri_module_with_perfect_treatment_and_diagnosis._ultimate_treatment_indicated_for_patient(
        classification_for_treatment_decision=classification_for_treatment_decision,
        age_exact_years=age_exact_years
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
        any_complications=len(complications) > 0,
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

            # Treatment Efficacy with * PERFECT HW Diagnosis *
            'treatment_efficacy_if_perfect_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx': treatment_efficacy(
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
            ),

            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx': treatment_efficacy(
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
            ),

            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx': treatment_efficacy(
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
            ),

            'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx':
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
            ),

            'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx':
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
            ),

            # Treatment Efficacy with * IMPERFECT HW Diangosis *
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_imperfect_hw_dx':
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
            ),

            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_imperfect_hw_dx':
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
            ),

            'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_imperfect_hw_dx':
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
            ),

            'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_imperfect_hw_dx':
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
            ),

            # Classifications
            'classification_for_treatment_decision_with_oximeter_perfect_accuracy':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    facility_level=_facility_level,
                    use_oximeter=True,
            ),

            'classification_for_treatment_decision_without_oximeter_perfect_accuracy':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    facility_level=_facility_level,
                    use_oximeter=False,
            ),

            'classification_for_treatment_decision_with_oximeter_imperfect_accuracy':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    facility_level=_facility_level,
                    use_oximeter=True,
            ),

            'classification_for_treatment_decision_without_oximeter_imperfect_accuracy':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    facility_level=_facility_level,
                    use_oximeter=False,
            ),

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

    truth_danger_signs_pneumonia = \
        table["classification_for_treatment_decision_with_oximeter_perfect_accuracy"] == "danger_signs_pneumonia"

    correctly_dx_with_po_when_misdiagnosed_without_imperfect_accuracy = len(
        table.loc[truth_danger_signs_pneumonia
                  & (table["classification_for_treatment_decision_with_oximeter_imperfect_accuracy"]
                     == "danger_signs_pneumonia")
                  & (table["classification_for_treatment_decision_without_oximeter_imperfect_accuracy"]
                     != "danger_signs_pneumonia")
                  ]
    ) / len(table.loc[truth_danger_signs_pneumonia])  # 32%

    correctly_dx_with_po_when_misdiagnosed_without_perfect_accuracy = len(
        table.loc[truth_danger_signs_pneumonia
                  & (table["classification_for_treatment_decision_with_oximeter_perfect_accuracy"]
                     == "danger_signs_pneumonia")
                  & (table["classification_for_treatment_decision_without_oximeter_perfect_accuracy"]
                     != "danger_signs_pneumonia")
                  ]
    ) / len(table.loc[truth_danger_signs_pneumonia])  # 12%

    print(f"\n{correctly_dx_with_po_when_misdiagnosed_without_imperfect_accuracy}"
          f"\n{correctly_dx_with_po_when_misdiagnosed_without_perfect_accuracy}")

    # Examine risk of death and treatment effectiveness by disease classification and oxygen saturation
    # And include estimate of % of deaths by scaling using "prob_die_if_no_treatment".
    # UNDER PERFECT HW DX
    risk_of_death = summarize_by(
        df=table,
        by=['oxygen_saturation', 'classification_for_treatment_decision_with_oximeter_perfect_accuracy'],
        columns=[
            'prob_die_if_no_treatment',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx',
            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx',
        ]
    ).assign(
        fraction_of_deaths=lambda df: (
            (df.fraction * df.prob_die_if_no_treatment) / (df.fraction * df.prob_die_if_no_treatment).sum()
        )
    )
    print(f"{risk_of_death=}")

    # Examine risk of death and treatment effectiveness for "danger_signs_pneumonia" & SpO2<90%
    treatment_effectiveness = summarize_by(
        df=table,
        by=['oxygen_saturation', 'classification_for_treatment_decision_with_oximeter_perfect_accuracy'],
        columns=[
            'prob_die_if_no_treatment',
            'treatment_efficacy_if_perfect_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx',
            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx',
            'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx',
        ]
    ).loc[("<90%", "danger_signs_pneumonia")]
    print(f"{treatment_effectiveness}")

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

    # In all cases, confirm they have exactly the same treatment effectiveness when no oxygen is available.
    assert (
        diff_classification['treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx']
        ==
        diff_classification['treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx']
    ).all()

    # ... but that the availability of oxygen improves treatment effectiveness when there is a diff in diagnosis.
    assert (
        diff_classification['treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx']
        >
        diff_classification['treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx']
    ).all()

    # Overall summary figure: Number of deaths in the cohort Deaths broken down by ( disease / oxygen_saturation) when
    # * No Treatment
    # * Treatment antibiotics only (no oxygen, no oximeter)
    # * Treatment with antibiotics and oxygen (no oximeter)
    # * Treatment with antibiotics and oxygen and use of oximeter
    # ... under assumptions of (i) normal treatment and normal dx accuracy; (ii) normal treatment and perfect dx
    # accuracy

    disease_classification = table["classification_for_treatment_decision_with_oximeter_perfect_accuracy"]
    low_oxygen = (table["oxygen_saturation"] == "<90%").replace({True: 'SpO2<90%', False: "SpO2>=90%"})
    res = {
        "Perfect HW Dx Accuracy": {
            "No Treatment": table['prob_die_if_no_treatment'].groupby(by=[disease_classification, low_oxygen]).sum(),
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
                 table['treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx'] / 100.0
                 )).groupby(by=[disease_classification, low_oxygen]).sum(),
        },

        "Normal HW Dx Accuracy": {
            "No Treatment": table['prob_die_if_no_treatment'].groupby(by=[disease_classification, low_oxygen]).sum(),
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
                 table['treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_imperfect_hw_dx'] / 100.0
                 )).groupby(by=[disease_classification, low_oxygen]).sum(),
        }
    }

    results = (100_000 / len(table)) * pd.concat({k: pd.DataFrame(v) for k, v in res.items()}, axis=1)
    fig, axs = plt.subplots(ncols=2, nrows=1, sharey=True, )
    for i, ix in enumerate(results.columns.levels[0]):
        ax = axs[i]
        results.loc[:, (ix, slice(None))].T.plot.bar(stacked=True, ax=ax, legend=False)
        ax.set_xticklabels(results.loc[:, (ix, slice(None))].columns.levels[1])
        ax.set_xlabel('Treatment Assumption')
        ax.set_ylabel('Deaths per 100,000 cases of ALRI')
        ax.set_title(f"{ix}", fontsize=10)
        ax.grid(axis='y')
    handles, labels = axs[-1].get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), title='Case Type', loc='upper left', fontsize=8)
    fig.suptitle('Deaths to Alri Under Different Treatments', fontsize=14)
    fig.tight_layout()
    fig.show()
    plt.close(fig)
