"""This script is used to verify the output values from the natural history."""

from pathlib import Path
from typing import List

import numpy as np
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

MODEL_POPSIZE = 30_000
MIN_SAMPLE_OF_NEW_CASES = 200
NUM_REPS_FOR_EACH_CASE = 20

_facility_level = '2'  # <-- assumes that the diagnosis/treatment occurs at this level


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


# ------------------------------------------------------------------
# generate the dataframe with the cases
df = generate_case_mix()

# Verify the numbers of the natural history

# group by disease type: pneumonia vs other_alri
df_by_group = df.groupby('disease_type')
df_pneumonia = df_by_group.get_group('pneumonia')
df_other_alri = df_by_group.get_group('other_alri')

# total cases per group
total_pneumonia_cases = df_pneumonia.index.size
total_other_alri_cases = df_other_alri.index.size
total_alri_cases = df.index.size

# # # # # # # # # # PARAMETER VALUES FOR RESOURCE FILE # # # # # # # # # #

# # # # # # # # Get the proportions from the natural history output to estimate parameter values # # # # # # # #

# PARAMETER - proportion_pneumonia_in_alri
prop_abnormal_cxr = total_pneumonia_cases / total_alri_cases

# proportion under 2 months old
under_2_months = df.loc[df['age_exact_years'] < 2.0 / 12.0].index.size
prop_under_2mo = under_2_months / df.index.size

# PARAMETER - proportion_bacterial_infection_in_pneumonia
bacterial_infection_pneumonia = df_pneumonia[['pathogen', 'bacterial_coinfection']].apply(lambda x: 1 if (
    (x[0] in sim0.modules['Alri'].pathogens['bacterial']) or pd.notnull(x[1])) else 0, axis=1).sum()

bacterial_inf_in_pneumonia = bacterial_infection_pneumonia / total_pneumonia_cases

# PARAMETER - proportion_bacterial_infection_in_other_alri
bacterial_infection_other_alri = df_other_alri[['pathogen', 'bacterial_coinfection']].apply(lambda x: 1 if (
    (x[0] in sim0.modules['Alri'].pathogens['bacterial']) or pd.notnull(x[1])) else 0, axis=1).sum()

prop_bacterial_inf_in_other_alri = bacterial_infection_other_alri / total_other_alri_cases

# --------------------------------------------------------
# Parameter values for OTHER ALRI
# number of cases NOT included in the model of mortality
no_mortality_cases = df[['symptoms', 'disease_type', 'complications']].apply(
    lambda x: 1 if not any(s in ['chest_indrawing', 'danger_signs', 'respiratory_distress'] for s in x[0]) and
                   (x[1] == 'other_alri') and (len(x[2]) == 0) else 0, axis=1).sum()

# the proportion of cases not in the mortality model / (assumed not sought care, for adjustmnet of params)
prop_no_mortality = no_mortality_cases / total_other_alri_cases

# denominator for uncomplicated other alri
total_uncomplicated_other_alri = df_other_alri['complications'].apply(lambda x: 1 if (len(x) == 0) else 0).sum()

# get the proportion of danger signs in non-hypoxaemic, non-bacteraemic other alri
tot_ds_in_uncomplicated_other_alri = df_other_alri[['complications', 'symptoms']].apply(
    lambda x: 1 if (len(x[0]) == 0) and 'danger_signs' in x[1] else 0, axis=1).sum()

# PARAMETER - prob_danger_signs_in_other_alri (before applying complications)
prop_ds_in_uncomplicated_other_alri = tot_ds_in_uncomplicated_other_alri / total_uncomplicated_other_alri

# # # CHECK IF CALIBRATED --- TODO: check this

# proportion of general danger signs from other alri ( may or may not have respiratory distress)
# PCV13 data tab general_ds_all_level, m - 12.98% - 16.68% (without missing) on average in all ALRI - 1.24x more in CXR+
# should be between minimum 11.78% to max ~ 15.15% in Other ALRI (at least)
total_ds_other_alri = df_other_alri['symptoms'].apply(lambda x: 1 if 'danger_signs' in x else 0).sum()
prop_ds_other_alri = total_ds_other_alri / total_other_alri_cases  # 0.16551

# proportion of all danger signs (ds and rd) from other alri
# PCV13 data tab sev_signs_rd - 22.93% - 28.45% (without missing) on average in all ALRI - 1.24x more in CXR+
# should be between 20.823% to max 25.836% in other ALRI (at least)
total_ds_and_rd_other_alri = df_other_alri['symptoms'].apply(
    lambda x: 1 if 'danger_signs' in x or 'respiratory_distress' in x else 0).sum()
prop_ds_and_rd_other_alri = total_ds_and_rd_other_alri / total_other_alri_cases  # 0.3142

# Check for chest-indrawing in other alri ---------
# get the proportion of chest indrawing in non-complicated other alri
tot_ci_in_uncomplicated_other_alri = df_other_alri[['complications', 'symptoms']].apply(
    lambda x: 1 if (len(x[0]) == 0) and 'chest_indrawing' in x[1] else 0, axis=1).sum()
# should be 49.07% as input value
prop_ci_in_uncomplicated_other_alri = tot_ci_in_uncomplicated_other_alri / total_uncomplicated_other_alri  # 0.56065

# # # CHECK IF CALIBRATED ---
# proportion of chest indrawing from other alri regardless of SpO2 level
# PCV13 data tab chest_all_level - 50.59% - 62.93% (without missing) on average in all ALRI - 1.161x more in CXR+
# should be between minimum 47.375% to max 58.93% in other ALRI
# Or 0.85 (Rees et al 2020) * 0.75 = 0.6375 --- keep as max value
total_ci_other_alri = df_other_alri['symptoms'].apply(lambda x: 1 if 'chest_indrawing' in x else 0).sum()
prop_ci_other_alri = total_ci_other_alri / total_other_alri_cases  # 0.61214 todo: too low

# -----------------------------------------------
# # # # # NOW CHECK FOR PNEUMONIA GROUP # # # # #

# number of cases assumed not seeking care (no complications with only fast-breathing)
no_mortality_cases_pneum = df_pneumonia[['symptoms', 'complications']].apply(
    lambda x: 1 if ('chest_indrawing' not in x[0] or 'danger_signs' not in x[0])
    and (len(x[1]) == 0) else 0, axis=1).sum()

# the proportion of cases not in the mortality model / (assumed not sought care, for adjustmnet of parameters)
prop_no_mortality_pneum = no_mortality_cases / total_pneumonia_cases

# denominator for uncomplicated pneumonia
total_uncomplicated_pneumonia = df_pneumonia['complications'].apply(lambda x: 1 if (len(x) == 0) else 0).sum()
# denominator for complicated pneumonia
total_complicated_pneumonia = df_pneumonia['complications'].apply(lambda x: 1 if (len(x) != 0) else 0).sum()
prop_complicated_pneumonia = total_complicated_pneumonia / total_pneumonia_cases  # 0.4517

total_pc_pneumonia = df_pneumonia['complications'].apply(lambda x: 1 if any(
    e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pneumothorax', 'bacteraemia'] for e in x) else 0).sum()
prop_pc_pneumonia = total_pc_pneumonia / total_pneumonia_cases  # 0.207

# get the proportion of danger signs in uncomplicated pneumonia
tot_ds_in_uncomplicated_pneumonia = df_pneumonia[['complications', 'symptoms']].apply(
    lambda x: 1 if (len(x[0]) == 0) and 'danger_signs' in x[1] else 0, axis=1).sum()

# PARAMETER - prob_danger_signs_in_pneumonia (before applying complications) - input value = 0.1403
prop_ds_in_uncomplicated_pneumonia = tot_ds_in_uncomplicated_pneumonia / total_uncomplicated_pneumonia  # 0.15919

# get the proportion of chest-indrawing in uncomplicated pneumonia group
tot_ci_in_uncomplicated_pneumonia = df_pneumonia[['complications', 'symptoms']].apply(
    lambda x: 1 if (len(x[0]) == 0) and 'chest_indrawing' in x[1] else 0, axis=1).sum()
total_uncomplicated_pneumonia = df_pneumonia['complications'].apply(lambda x: 1 if (len(x) == 0) else 0).sum()

# confirm this equals prob_chest_indrawing_in_pneumonia (before applying complications) - input value 56.97%
prop_ci_in_uncompliced_pneumonia = tot_ci_in_uncomplicated_pneumonia / total_uncomplicated_pneumonia  # 0.641

# Target proportion of chest indrawing in complicated pneumonia ~ 68.42% or higher (or 74% target)
tot_ci_in_complicated_pneumonia = df_pneumonia[['complications', 'symptoms']].apply(
    lambda x: 1 if (len(x[0]) != 0) and 'chest_indrawing' in x[1] else 0, axis=1).sum()
prop_ci_in_complicated_pneumonia = tot_ci_in_complicated_pneumonia / total_complicated_pneumonia  # 0.7867

# # # CHECK IF CALIBRATED
# proportion of danger signs in CXR+ -- ~ 0.28 * 0.26148 (overlap) = 20.678 use as MAX
# PCV13 data tab general_ds_all_level, m - 12.98% - 16.68% (without missing) on average in all ALRI - 1.24x more in CXR+
# should be between 0.1462% to 18.78% in Pneumonia (based on PCV13)
# Then adjust to the non-ALRI in SpO2>=93% in PCV13 - 0.1878*(1-0.273)*1.1747+(0.1878*0.273) = 0.212833 (USE AS MAX)
# (based on general danger signs / may or may not have respiratory distress)
danger_signs_in_pneumonia = df_pneumonia['symptoms'].apply(lambda x: 1 if 'danger_signs' in x else 0).sum()
prop_danger_signs_in_pneumonia = danger_signs_in_pneumonia / total_pneumonia_cases  # 0.21711

# proportion of all danger signs (general ds and rd) in CXR+
# PCV13 data tab sev_signs_rd - 22.93% - 28.45% (without missing) on average in all ALRI - 1.24x more in CXR+
# should be between minimum 25.821% to 32.037% in other ALRI (at least)
ds_and_rd_in_pneumonia = df_pneumonia['symptoms'].apply(lambda x: 1 if ('danger_signs' in x or 'respiratory_distress' in x) else 0).sum()
prop_ds_and_rd_in_pneumonia = ds_and_rd_in_pneumonia / total_pneumonia_cases  # 0.39972

# proportion of chest_indrawing in CXR+ regardless of SpO2 level
# PCV13 data tab chest_all_level - 50.59% - 62.93% (without missing) on average in all ALRI - 1.161x more in CXR+
# should be between minimum 55% to 68.42% in Pneumonia (based on PCV13) - AT LEAST
chest_indrawing_in_pneumonia = df_pneumonia['symptoms'].apply(lambda x: 1 if 'chest_indrawing' in x else 0).sum()
prop_chest_indrawing_in_pneumonia = chest_indrawing_in_pneumonia / total_pneumonia_cases  # 0.7069 todo: too low?

# ------------------------------------------------------------------
# CHECK IF CALIBRATED - Symptom-based classification of pneumonia # # # # # # # # # # # #
# must keep fast-breathing pneumonia at at least ~ 33 %

# general danger signs only (with or without resp distress)
gn_danger_signs_pneumonia = df['symptoms'].apply(lambda x: 1 if 'danger_signs' in x else 0).sum()
# danger_signs_pneumonia = df['symptoms'].apply(
#     lambda x: 1 if ('danger_signs' in x) or ('respiratory_distress' in x) else 0).sum()

danger_signs_pneumonia = df['symptoms'].apply(
    lambda x: 1 if ('danger_signs' in x) or all(s in x for s in ['respiratory_distress', 'chest_indrawing']) else 0).sum()

chest_indrawing_pneumonia = df['symptoms'].apply(
    lambda x: 1 if 'chest_indrawing' in x and 'danger_signs' not in x and 'respiratory_distress' not in x else 0).sum()
fast_breathing_pneumonia = df['symptoms'].apply(
    lambda x: 1 if 'tachypnoea' in x and 'danger_signs' not in x and 'respiratory_distress' not in x and
                   'chest_indrawing' not in x else 0).sum()

prop_gn_danger_signs_pneumonia_class = gn_danger_signs_pneumonia / total_alri_cases

prop_danger_signs_pneumonia_class = danger_signs_pneumonia / total_alri_cases  # 0.329013
prop_chest_indrawing_pneumonia_class = chest_indrawing_pneumonia / total_alri_cases  # 0.366031
prop_fast_breathing_pneumonia_class = fast_breathing_pneumonia / total_alri_cases   # 0.231768
prop_cough_cold = 1 - (prop_danger_signs_pneumonia_class + prop_chest_indrawing_pneumonia_class + prop_fast_breathing_pneumonia_class)  # 0.07318

prop_non_severe = 1 - (prop_danger_signs_pneumonia_class + prop_chest_indrawing_pneumonia_class)  # 0.304955    # should be at least 33%

# -------------------------------------------------------------
# # # # CALIBRATION CHECK # # # #
# get the proportion of SpO2<90% in each classification group

# FAST-BREATHING PNEUMONIA CLASSIFICATION ------------------------------------------------
# SpO2<90% in fast-breathing pneumonia -- ~ 2.89% - 3.07% (PCV13 data - tab classification3 oxygensat_3levels, (m vs -) row)
tot_fast_breathing_SpO2_below90 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '<90%') and 'tachypnoea' in x[1] and
    'danger_signs' not in x[1] and 'respiratory_distress' not in x[1] and 'chest_indrawing' not in x[1] else 0, axis=1).sum()
prop_fb_spo2_below_90 = tot_fast_breathing_SpO2_below90 / fast_breathing_pneumonia  # 0.01889

prop_SpO2_below90_in_non_sev_pneumonia = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '<90%') and 'danger_signs' not in x[1] and 'respiratory_distress' not in x[1]
                   and 'chest_indrawing' not in x[1] else 0,
    axis=1).sum() / (total_alri_cases - (chest_indrawing_pneumonia + danger_signs_pneumonia))  # 0.016

# SpO2 <93% should be ~8% - in fast-breathing pneumonia (ref: hypoxaemia prevalence study)
# ~9.01% - 9.56% (PCV13 data - tab classification3 oxygensat_3levels, (m vs -) row)
tot_fast_breathing_SpO2_below93 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] != '>=93%') and 'tachypnoea' in x[1] and
    'danger_signs' not in x[1] and 'respiratory_distress' not in x[1] and 'chest_indrawing' not in x[1] else 0, axis=1).sum()
prop_SpO2_below93_in_fb_pneumonia = tot_fast_breathing_SpO2_below93 / fast_breathing_pneumonia  # 0.0836

# SpO2 <93% should be ~8% - in fast-breathing pneumonia (ref: hypoxaemia prevalence study)
# include those classified as cough or cold
prop_SpO2_below90_in_non_sev_pneumonia_all_ds = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '<90%') and 'danger_signs' not in x[1] and 'respiratory_distress' not in x[1]
                   and 'chest_indrawing' not in x[1] else 0, axis=1).sum() / (
    total_alri_cases - (chest_indrawing_pneumonia + danger_signs_pneumonia))  # 0.01678

prop_SpO2_below93_in_non_sev_pneumonia_all_ds = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] != '>=93%') and 'danger_signs' not in x[1] and 'respiratory_distress' not in x[1] and
                   'chest_indrawing' not in x[1] else 0, axis=1).sum() / (
    total_alri_cases - (chest_indrawing_pneumonia + danger_signs_pneumonia))  # 0.074

# CHEST-INDRAWING PNEUMONIA CLASSIFICATION ------------------------------------------------
# SpO2<90% in chest indrawing pneumonia
# keep between 7.44% - 8.58%
# (PCV13 data - tab chest_all_level oxygensat_3levels if ( general_ds_all_level!=1 & resp_distress2!=1) ,m ro, (m vs -) row) - use ~ 7.5% as minimum
# CAP MAX AT 10.1% ----- 0.0858 * 1.17474 (adjust for those non-ALRI cases in PCV13) = 0.10079
tot_chest_indrawing_SpO2_below90 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '<90%') and 'chest_indrawing' in x[1] and ('danger_signs' not in x[1] and 'respiratory_distress' not in x[1]) else 0, axis=1).sum()
prop_ci_spo2_below_90 = tot_chest_indrawing_SpO2_below90 / chest_indrawing_pneumonia  # ~ 0.0943

# SpO2 <93% in chest indrawing pneumonia
# keep between 15.79% - 18.22%
# (PCV13 data - tab chest_all_level oxygensat_93 if ( general_ds_all_level!=1 & resp_distress2!=1), (m vs -) row) - use ~16% as minimum
# CAP MAX AT 21.4%% ----- 0.1822 * 1.17474 (adjust for those non-ALRI cases in PCV13) = 0.21403
tot_chest_indrawing_SpO2_below93 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] != '>=93%') and 'chest_indrawing' in x[1] and ('danger_signs' not in x[1] and 'respiratory_distress' not in x[1]) else 0, axis=1).sum()
prop_SpO2_below93_in_ci_pneumonia = tot_chest_indrawing_SpO2_below93 / chest_indrawing_pneumonia  # ~ 0.22479 todo: little too high?

# DANGER-SIGNS PNEUMONIA CLASSIFICATION ------------------------------------------------
# SpO2<90% in general danger signs pneumonia
# PCV13 data tab general_ds_all_level oxygensat_3levels, (m vs -) row)
# Keep between 17.36% - 19.72% (without missing) on average in all ALRI - 1.24x more in CXR+
# CAP MAX AT 23.2% ----- 0.1972 * 1.17474 (adjust for those non-ALRI cases in PCV13) = 0.23165
tot_danger_signs_SpO2_below90 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '<90%') and 'danger_signs' in x[1] else 0, axis=1).sum()
prop_ds_spo2_below_90 = tot_danger_signs_SpO2_below90 / gn_danger_signs_pneumonia  # 0.21886  # todo: too high? - but OK?
# prevalence of hypoxaemia in hospitalised children with severe pneumonia was 40% in Bangladesh study Rahman et al 2021


# should be between 15.765% to max 17.91% in Other ALRI (no hard cap)
tot_danger_signs_SpO2_below90_other = df_other_alri[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '<90%') and 'danger_signs' in x[1] else 0, axis=1).sum()
prop_ds_spo2_below_90_other = tot_danger_signs_SpO2_below90_other / df_other_alri['symptoms'].apply(
    lambda x: 1 if 'danger_signs' in x else 0).sum()  # 0.16981

# should be between 19.55% to max 22.21% in Pneumonia (no hard cap)
tot_danger_signs_SpO2_below90_pneum = df_pneumonia[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '<90%') and 'danger_signs' in x[1] else 0, axis=1).sum()
prop_ds_spo2_below_90_pneum = tot_danger_signs_SpO2_below90_pneum / df_pneumonia['symptoms'].apply(
    lambda x: 1 if 'danger_signs' in x else 0).sum()  # 0.27

# SpO2 <93% in general danger signs alri -------------
# PCV13 data tab general_ds_all_level oxygensat_93, (m vs -) row)
# keep between 27.46% - 31.19% (PCV13 data - tab general_ds_all_level oxygensat_93, (m vs -) row) - use 27.5% as minimum
# CAP MAX AT 36.7% ----- 0.3119 * 1.17474 (adjust for thos non-ALRI cases in PCV13) = 0.366
tot_gn_danger_signs_SpO2_below93 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] != '>=93%') and 'danger_signs' in x[1] else 0, axis=1).sum()
prop_SpO2_below93_in_gn_ds_pneumonia = tot_gn_danger_signs_SpO2_below93 / gn_danger_signs_pneumonia  # 0.3605

# should be between 24.94% to max 28.32% in Other ALRI (no hard cap)
tot_gn_danger_signs_SpO2_below93_other = df_other_alri[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] != '>=93%') and 'danger_signs' in x[1] else 0, axis=1).sum()
prop_gn_ds_spo2_below93_other = tot_gn_danger_signs_SpO2_below93_other / df_other_alri['symptoms'].apply(
    lambda x: 1 if 'danger_signs' in x else 0).sum()  # 0.28

# should be between 30.92% to max 35.12% in Pneumonia (no hard cap)
tot_gn_danger_signs_SpO2_below93_pneum = df_pneumonia[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] != '>=93%') and 'danger_signs' in x[1] else 0, axis=1).sum()
prop_gn_ds_spo2_below93_pneum = tot_gn_danger_signs_SpO2_below93_pneum / df_pneumonia['symptoms'].apply(
    lambda x: 1 if 'danger_signs' in x else 0).sum()  # 0.44381

# SpO2 <93% in all danger signs alri - general ds + rd -------------
# add resp distress in the cases
# PCV13 data tab sev_signs_rd oxygensat_93, (m vs -) row) -- 29.09% - 32.61%
# CAP MAX at 38.5% ---- 0.3261 * 1.1747 (count for thos non-ALRI cases in PCV13) = 0.383
tot_all_danger_signs_SpO2_below93 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] != '>=93%') and ('danger_signs' in x[1] or 'respiratory_distress' in x[1]) else 0, axis=1).sum()
prop_SpO2_below93_in_all_ds_pneumonia = tot_all_danger_signs_SpO2_below93 / danger_signs_pneumonia  # 0.39659 todo: too high - but OK?

# SpO2 <93% should be ~ < 41% in chest indrawing and danger signs pneumonia  - USE AS MAXIMUM VALE (cap at 35%)
tot_ci_or_ds_SpO2_below93 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] != '>=93%') and ('chest_indrawing' in x[1] or 'danger_signs' in x[1]or 'respiratory_distress' in x[1]) else 0, axis=1).sum()
prop_SpO2_below93_in_ci_or_ds_pneumonia = tot_ci_or_ds_SpO2_below93 / (
    danger_signs_pneumonia + chest_indrawing_pneumonia)  # 0.306115

# overall SpO2<93% (PREVALENCE ~28% in Lancet for Africa) -- input value 0.237
tot_SpO2_below93 = df['oxygen_saturation'].apply(lambda x: 1 if (x != '>=93%') else 0).sum()
prop_SpO2_below93 = tot_SpO2_below93 / total_alri_cases

# Prevalence of hypoxaemia SpO2<93% in the PCV13 data is ~ 15.21% - 17.04% - tab classification3 oxygensat_93, (m vs -) row
# PCV13 data - SpO2 <93% in chest indrawing and danger signs pneumonia is -->  2300+954 / 10204+2594 (without missing values) = 25.426%
# PCV13 data - SpO2 <93% in chest indrawing and danger signs pneumonia is -->  2300+954 / 11303+3029 (with missing values) = 22.7%
# --- 22.7% - 25.426% vs 41% in the Lancet systematic review
# 0.227 / 0.41 = 0.5537, or 0.25426 / 0.41 = 0.62 fraction difference between PCV13 data and the Lancet systematic review for proportion of hypoxaemia in chest-indrawing and danger signs pneumonia
# if multiplying 0.5537 by the hypoxaemia prevalence input (28%) = 15.5% ~~~~ similar to the prevalence in the PCV13 data
# if multiplying 0.64 by the hypoxaemia prevalence input (28%) = 17.36% ~~~~ similar to the prevalence in the PCV13 data

# or 0.1521 / 0.28 = 0.543, or 0.1704 / 0.28 = 0.6085 fraction difference in the prevalence of hypoxaemia between PCV13 and Lancet
# if multiplying 0.543 by the hypoxaemia prevalence input (22.7% ) = 12.32% ~~~~ similar to the prevalence in the PCV13 data (15.21%)
# if multiplying 0.608 by the hypoxaemia prevalence input (25.426% ) = 15.47% ~~~~ similar to the prevalence in the PCV13 data (15.21%)

# NOTE USE fraction = 0.6085 TO CHECK THE OUTPUTS

# -----------------------------------------------------------------------------------------------------------------
# CALIBRATION CHECK - Check the signs and symptoms in SpO2 levels -------------------------------------------------
# in ALL ALRI cases
total_SpO2_below90 = df['oxygen_saturation'].apply(lambda x: 1 if x == '<90%' else 0).sum()
total_SpO2_90_92 = df['oxygen_saturation'].apply(lambda x: 1 if x == '90-92%' else 0).sum()
# in Pneumonia group
total_SpO2_below90_pneumonia = df_pneumonia['oxygen_saturation'].apply(lambda x: 1 if x == '<90%' else 0).sum()
total_SpO2_90_92_pneumonia = df_pneumonia['oxygen_saturation'].apply(lambda x: 1 if x == '90-92%' else 0).sum()

# # # # # # SpO2 < 90% # # # # # #
# chest indrawing in SpO2<90% input value 0.837 PCV13 //
# Cap at max value ~ 89.28% (PCV13 data, removed missing values, with missing values 0.8370)
prop_ci_in_SpO2_below90 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '<90%') and 'chest_indrawing' in x[1] else 0, axis=1).sum() / total_SpO2_below90  # 0.8875

# breakdown by disease type
# Pneumonia group
prop_ci_in_SpO2_below90_pneum = df_pneumonia[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '<90%') and 'chest_indrawing' in x[1] else 0, axis=1).sum() / total_SpO2_below90_pneumonia  # 0.9001
# Other ALRI group
prop_ci_in_SpO2_below90_other = df_other_alri[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '<90%') and 'chest_indrawing' in x[1] else 0, axis=1).sum() / \
                                df_other_alri['oxygen_saturation'].apply(lambda x: 1 if x == '<90%' else 0).sum()  # 0.871038

# danger_signs in SpO2<90% input value 0.3041 // Cap at max value ~ 33.95% (PCV13 data, removed missing values)
prop_ds_in_SpO2_below90 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '<90%') and 'danger_signs' in x[1] else 0, axis=1).sum() / total_SpO2_below90  # 0.35485

# resp. distress in SpO2<90% input value 0.4373 // Cap at max value ~ 48.57% (PCV13 data, removed missing values)
prop_rd_in_SpO2_below90 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '<90%') and 'respiratory_distress' in x[1] else 0, axis=1).sum() / total_SpO2_below90  # 0.4924

# # # # # # SpO2 between 90-92% # # # # # #
# chest indrawing in SpO2 90-92% input value 0.445 //
# Cap at max value ~ 77.91% (PCV13 data, removed missing values, with missing values, 63.33)
# should keep between 0.445 and 0.6333
prop_ci_in_SpO2_90_92 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '90-92%') and 'chest_indrawing' in x[1] else 0, axis=1).sum() / total_SpO2_90_92  # 0.7589

# danger_signs in SpO2 90-92% input value 0.168 // Cap at max value ~ 21.44% (PCV13 data, removed missing values)
prop_ds_in_SpO2_90_92 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '90-92%') and 'danger_signs' in x[1] else 0, axis=1).sum() / total_SpO2_90_92  # 0.22127

# resp. distress in SpO2 90-92% input value 0.2501 // Cap at max value ~ 33.20% (PCV13 data, removed missing values)
prop_rd_in_SpO2_90_92 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '90-92%') and 'respiratory_distress' in x[1] else 0, axis=1).sum() / total_SpO2_90_92  # 0.32455

# # # # # # Overall hypoxaemia / SpO2 <93% # # # # # #
# keep between: 53.7% and 73.26% (CAP MAX 83%)
# chest indrawing in SpO2 <93% should be ~ 53.7% (from McCollum WHO study) depends on the prop of SpO2<90 / 90-92%
# or ~ 73.26% (PCV13 data, including missing values in denominator, ~ 83.85% removed missing values)
# tab chest_all_level oxygensat_93, col
prop_ci_in_SpO2_below93 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] != '>=93%') and 'chest_indrawing' in x[1] else 0, axis=1).sum() / (
                              total_SpO2_below90 + total_SpO2_90_92)  # 0.822

# keep between 20.3% and 25.4% (CAP MAX 28%)
# danger_signs in SpO2 <93% should be ~ 22.5% (from McCollum WHO study) depends on the prop of SpO2<90 / 90-92%
# or ~ 23.43% (PCV13 data, including missing values in analyses, ~ 27.95% removed missing values)
# tab general_ds_all_level oxygensat_93, col
prop_ds_in_SpO2_below93 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] != '>=93%') and 'danger_signs' in x[1] else 0, axis=1).sum() / (
                              total_SpO2_below90 + total_SpO2_90_92)  # 0.2868

# keep between 34.13% (PCV13 data, including missing values in analyses, ~ 41.37% removed missing values)
# respiratory distress in SpO2 <93% should be <41%
prop_rd_in_SpO2_below93 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] != '>=93%') and 'respiratory_distress' in x[1] else 0, axis=1).sum() / (
                              total_SpO2_below90 + total_SpO2_90_92)  # 0.40692

# ------------------------------------------------------------------------------------------------------------
# CALIBRATION CHECK:

# NO HYPOXAEMIA

# chest indrawing -------------
# NO HYPOXAEMIA
# chest indrawing in the SpO2>=93% - should be at least 56.71% (PCV 13 data - tab chest_all_level oxygensat_93, col)
# or at least 66.72%
prop_ci_in_normal_SpO2 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '>=93%') and 'chest_indrawing' in x[1] else 0, axis=1).sum() / \
                         (total_alri_cases - (total_SpO2_below90 + total_SpO2_90_92))  # 0.5998

# ALL CASES
# chest indrawing in all cases should be at least 62.93% (PCV13 data, overall prob chest indrawing)
# 0.6293 * 1.1747 = 73.92% CAP MAX AT 74%
# AVERAGE should be around= 0.74 * 0.4215(CXR+) + 0.6375 * (1-0.4215) = 67.85%
prob_ci_in_all_alri = df['symptoms'].apply(
    lambda x: 1 if 'chest_indrawing' in x else 0).sum() / total_alri_cases  # 0.6521 TODO: too low?

# GENERAL DANGER SIGNS -------------
# NO HYPOXAEMIA
# danger signs in the SpO2>=93% --- should be at least 7.38% (PCV 13 data, cap max at 11.73% removed missing values)
# keep max of 16% = 11.73% * 1.1747
prop_ds_in_normal_SpO2 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '>=93%') and 'danger_signs' in x[1] else 0, axis=1).sum() / \
                         (total_alri_cases - (total_SpO2_below90 + total_SpO2_90_92))  # 0.15665
# ALL CASES
# general danger signs in all cases should be at least 16.68% (PCV13 data, overall prob general_ds_all_level)
prob_ds_in_all_alri = df['symptoms'].apply(
    lambda x: 1 if 'danger_signs' in x else 0).sum() / total_alri_cases  # 0.1873

# resp distress in the SpO2>=93% --- input value: 0.086 in other alri and 0.1225 in pneumonia
# keep max 11.06 % - 17.22% (PCV 13 data, with missing) or adjusted for non-ALRI cases in PCV13 - 12.99% - 20.23%
prop_rd_in_normal_SpO2 = df[['oxygen_saturation', 'symptoms']].apply(
    lambda x: 1 if (x[0] == '>=93%') and 'respiratory_distress' in x[1] else 0, axis=1).sum() / \
                         (total_alri_cases - (total_SpO2_below90 + total_SpO2_90_92))  # 0.195288

# respiratory distress in pneumonia group should be a maximum of 35% (or 18% as target input value?)
prob_rd_in_pneumonia = df_pneumonia['symptoms'].apply(
    lambda x: 1 if 'respiratory_distress' in x else 0).sum() / total_pneumonia_cases  # 0.286418

# respiratory distress in other alri group should be a maximum of 25% (or 15% as target input value?)
prob_rd_in_other_alri = df_other_alri['symptoms'].apply(
    lambda x: 1 if 'respiratory_distress' in x else 0).sum() / total_other_alri_cases  # 0.21491

# respiratory distress in all cases should be ~ max 22.15% (PCV13 data, overall prob resp distress2)
prob_rd_in_all_alri = df['symptoms'].apply(
    lambda x: 1 if 'respiratory_distress' in x else 0).sum() / total_alri_cases  # 0.2451
# -------------------------------------------------------------------------------------------------------------

# seek care levels

seek_levels = df['symptoms'].apply(
    lambda x: alri_module_with_perfect_diagnosis.seek_care_level(x))

df_seek = seek_levels.to_frame(name='level')
total_seek_level = df_seek.groupby('level').size()
prop_level = total_seek_level / df_seek.groupby('level').size().sum()


df2 = df.join(df_seek)

hypox_seek_level = df2.loc[df2['oxygen_saturation'] == '<90%'].groupby('level').size()
prop_hypox_seek_level = hypox_seek_level / hypox_seek_level.sum()

