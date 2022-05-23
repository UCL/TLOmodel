"""
Childhood Acute Lower Respiratory Infection Module

Overview
--------
Individuals are exposed to the risk of infection by a pathogen (and potentially also with a bacterial
co-infection / secondary infection) that can cause one of two types of acute lower respiratory infection (Alri)
modelled in TLO.
The disease is manifested as either pneumonia or other alri (including bronchiolitis).

During an episode (prior to recovery - either naturally or cured with treatment), symptoms are manifested
and there may be complications (e.g. local pulmonary complication: pleural effusion, empyema, lung abscess,
pneumothorax; and/or systemic complications: sepsis; and/or complications regarding oxygen exchange: hypoxaemia.
The complications onset at the time of disease onset.

The individual may recover naturally or die. The risk of death depends on the type of disease and the presence of some
of the complications.

Health care seeking is prompted by the onset of the symptom. The individual can be treated; if successful the risk of
death is lowered and they are cured (symptom resolved) some days later.

Outstanding issues
------------------
* All HSI events
* Follow-up appointments for initial HSI events.
* Double check parameters and consumables codes for the HSI events.
* Duration of Alri Event is not informed by data

Following PRs:
---------------
PR3: Achieve a basic calibration of the model for incidence and deaths,
adjusting healthcare seeking behaviour and efficacy of treatment accordingly.

PR4: Achieve a basic calibration of the HSI outputs

Issue #438

"""

import types
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.util import random_date, sample_outcome

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITION
# ---------------------------------------------------------------------------------------------------------

class Alri(Module):
    """This is the disease module for Acute Lower Respiratory Infections."""

    INIT_DEPENDENCIES = {
        'Demography',
        'Epi',
        'Hiv',
        'Lifestyle',
        'NewbornOutcomes',
        'SymptomManager',
        'Wasting',
    }

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    pathogens = {
        'viral': [
            'RSV',
            'Rhinovirus',
            'HMPV',
            'Parainfluenza',
            'Influenza',
            'other_viral_pathogens'  # <-- Coronaviruses NL63, 229E OC43 and HKU1,
            # Cytomegalovirus, Parechovirus/Enterovirus, Adenovirus, Bocavirus
        ],
        'bacterial': [
            'Strep_pneumoniae_PCV13',  # <--  risk of acquisition is affected by the pneumococcal vaccine
            'Strep_pneumoniae_non_PCV13',  # <--  risk of acquisition is not affected by the pneumococcal vaccine
            'Hib',
            'H.influenzae_non_type_b',
            'Staph_aureus',
            'Enterobacteriaceae',  # includes E. coli, Enterobacter species, and Klebsiella species
            'other_Strepto_Enterococci',  # includes Streptococcus pyogenes and Enterococcus faecium
            'other_bacterial_pathogens'
            # <-- includes Bordetella pertussis, Chlamydophila pneumoniae,
            # Legionella species, Mycoplasma pneumoniae, Moraxella catarrhalis, Non-fermenting gram-negative
            # rods (Acinetobacter species and Pseudomonas species), Neisseria meningitidis
        ],
        'fungal/other': [
            'P.jirovecii',
            'other_pathogens_NoS'
        ]
    }

    # Make set of all pathogens combined:
    all_pathogens = sorted(set(chain.from_iterable(pathogens.values())))

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        f"ALRI_{path}": Cause(gbd_causes={'Lower respiratory infections'}, label='Lower respiratory infections')
        for path in all_pathogens
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        f"ALRI_{path}": Cause(gbd_causes={'Lower respiratory infections'}, label='Lower respiratory infections')
        for path in all_pathogens
    }

    # Declare the disease types:
    disease_types = sorted({
        'pneumonia',
        'other_alri'
    })

    # Declare the Alri complications:
    complications = sorted({
        'pneumothorax',
        'pleural_effusion',
        'empyema',
        'lung_abscess',
        'sepsis',
        'hypoxaemia'
    })

    PARAMETERS = {
        # Incidence rate by pathogens  -----
        'base_inc_rate_ALRI_by_RSV':
            Parameter(Types.LIST,
                      'baseline incidence rate of Alri caused by RSV in age groups 0-11, 12-23, 24-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_Rhinovirus':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Rhinovirus in age groups 0-11, 12-23, 24-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_HMPV':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by HMPV in age groups 0-11, 12-23, 24-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_Parainfluenza':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Parainfluenza 1-4 in age groups 0-11, 12-23, 24-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_Strep_pneumoniae_PCV13':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Streptoccocus pneumoniae PCV13 serotype '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_Strep_pneumoniae_non_PCV13':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Streptoccocus pneumoniae non PCV13 serotype '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_Hib':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Haemophilus influenzae type b '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_H.influenzae_non_type_b':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Haemophilus influenzae non-type b '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_Enterobacteriaceae':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Enterobacteriaceae in age groups 0-11, 12-23, '
                      '24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_other_Strepto_Enterococci':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by other streptococci and Enterococci including '
                      'Streptococcus pyogenes and Enterococcus faecium in age groups 0-11, 12-23, 24-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_Staph_aureus':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Staphylococcus aureus '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_Influenza':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Influenza type A, B, and C '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_P.jirovecii':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by P. jirovecii in age groups 0-11, '
                      '12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_other_viral_pathogens':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by other viral pathogens in age groups 0-11, 12-23, '
                      '24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_other_bacterial_pathogens':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by other viral pathogens in age groups 0-11, 12-23, '
                      '24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_other_pathogens_NoS':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by other pathogens not otherwise specified'
                      ' in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),

        # Proportions of what disease type (pneumonia/ bronchiolitis/ other alri) -----
        'proportion_pneumonia_in_RSV_ALRI':
            Parameter(Types.LIST,
                      'proportion of RSV-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Rhinovirus_ALRI':
            Parameter(Types.LIST,
                      'proportion of Rhinovirus-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_HMPV_ALRI':
            Parameter(Types.LIST,
                      'proportion of HMPV-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Parainfluenza_ALRI':
            Parameter(Types.LIST,
                      'proportion of Parainfluenza-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Strep_pneumoniae_PCV13_ALRI':
            Parameter(Types.LIST,
                      'proportion of S. pneumoniae PCV13-type-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Strep_pneumoniae_non_PCV13_ALRI':
            Parameter(Types.LIST,
                      'proportion of S. pneumoniae non PCV13-type-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Hib_ALRI':
            Parameter(Types.LIST,
                      'proportion of Hib ALRI-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_H.influenzae_non_type_b_ALRI':
            Parameter(Types.LIST,
                      'proportion of H.influenzae non type-b-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Staph_aureus_ALRI':
            Parameter(Types.LIST,
                      'proportion of S. aureus-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Enterobacteriaceae_ALRI':
            Parameter(Types.LIST,
                      'proportion of Enterobacteriaceae-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_other_Strepto_Enterococci_ALRI':
            Parameter(Types.LIST,
                      'proportion of other Streptococci- and Enterococci-attributed ALRI manifesting as pneumonia'
                      'by age, (based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Influenza_ALRI':
            Parameter(Types.LIST,
                      'proportion of Influenza-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_P.jirovecii_ALRI':
            Parameter(Types.LIST,
                      'proportion of P. jirovecii-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_other_viral_pathogens_ALRI':
            Parameter(Types.LIST,
                      'proportion of other viral pathogens-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_other_bacterial_pathogens_ALRI':
            Parameter(Types.LIST,
                      'proportion of other bacterial pathogens-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_other_pathogens_NoS_ALRI':
            Parameter(Types.LIST,
                      'proportion of other pathogens NoS-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),

        # Risk factors for incidence infection -----
        'rr_ALRI_HIV/AIDS':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for children with HIV+/AIDS '
                      ),
        'rr_ALRI_incomplete_measles_immunisation':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for children with incomplete measles immunisation'
                      ),
        'rr_ALRI_low_birth_weight':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for infants with low birth weight'
                      ),
        'rr_ALRI_non_exclusive_breastfeeding':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for not exclusive breastfeeding upto 6 months'
                      ),
        'rr_ALRI_indoor_air_pollution':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for indoor air pollution'
                      ),
        # Probability of bacterial co- / secondary infection -----
        'prob_viral_pneumonia_bacterial_coinfection':
            Parameter(Types.REAL,
                      'probability of primary viral pneumonia having a bacterial co-infection'
                      ),
        'proportion_bacterial_coinfection_pathogen':
            Parameter(Types.LIST,
                      'list of proportions of each bacterial pathogens in a viral/bacterial co-infection pneumonia,'
                      'the current values used are the pathogen attributable fractions (AFs) from PERCH for all ages. '
                      'The AFs were scaled to bacterial pathogens only causes - value assumed to be the proportions '
                      'of bacterial pathogens causing co-/secondary infection'
                      ),

        # Duration of disease - natural history
        'max_alri_duration_in_days_without_treatment':
            Parameter(Types.REAL,
                      'maximum duration in days of untreated ALRI episode, assuming an average of 7 days'
                      ),

        # Probability of complications -----
        'overall_progression_to_severe_ALRI':
            Parameter(Types.REAL,
                      'probability of progression to severe ALRI'
                      ),
        'prob_pulmonary_complications_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of pulmonary complications in (CXR+) pneumonia'
                      ),
        'prob_pleural_effusion_in_pulmonary_complicated_pneumonia':
            Parameter(Types.REAL,
                      'probability of pleural effusion in pneumonia with pulmonary complications'
                      ),
        'prob_empyema_in_pulmonary_complicated_pneumonia':
            Parameter(Types.REAL,
                      'probability of empyema in pneumonia with pulmonary complications'
                      ),
        'prob_lung_abscess_in_pulmonary_complicated_pneumonia':
            Parameter(Types.REAL,
                      'probability of lung abscess in pneumonia with pulmonary complications'
                      ),
        'prob_pneumothorax_in_pulmonary_complicated_pneumonia':
            Parameter(Types.REAL,
                      'probability of pneumothorax in pneumonia with pulmonary complications'
                      ),
        'prob_hypoxaemia_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of hypoxaemia in pneumonia cases'
                      ),
        'prob_hypoxaemia_in_other_alri':
            Parameter(Types.REAL,
                      'probability of hypoxaemia in bronchiolitis and other alri cases'
                      ),
        'prob_bacteraemia_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of bacteraemia in pneumonia'
                      ),
        'prob_progression_to_sepsis_with_bacteraemia':
            Parameter(Types.REAL,
                      'probability of progression to sepsis from bactereamia'
                      ),
        'proportion_hypoxaemia_with_SpO2<90%':
            Parameter(Types.REAL,
                      'proportion of hypoxaemic children with SpO2 <90%'
                      ),

        # Risk of death parameters -----
        'overall_CFR_ALRI':
            Parameter(Types.REAL,
                      'overall case-fatality rate of ALRI (Not used in the simulation but is saved here as it is the '
                      'target for calibration of for the overall case-fatality rate of ALRI).)'
                      ),
        'baseline_odds_alri_death':
            Parameter(Types.REAL,
                      'baseline odds of alri death, no risk factors'
                      ),
        'or_death_ALRI_age<2mo':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for infants aged less than 2 months'
                      ),
        'or_death_ALRI_P.jirovecii':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for P.jirovecii infection'
                      ),
        'or_death_ALRI_HIV/AIDS':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for HIV/AIDS children'
                      ),
        'or_death_ALRI_SAM':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for SAM'
                      ),
        'or_death_ALRI_MAM':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for MAM'
                      ),
        'or_death_ALRI_male':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for male children'
                      ),
        'or_death_ALRI_SpO2<93%':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for SpO2<=92%'
                      ),
        'or_death_ALRI_severe_underweight':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for severely underweight children'
                      ),
        'or_death_ALRI_danger_signs':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for very severe pneumonia (presenting danger signs)'
                      ),

        # Probability of symptom development -----
        'prob_cough_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of cough in pneumonia'
                      ),
        'prob_difficult_breathing_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of difficulty breathing in pneumonia'
                      ),
        'prob_fever_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of fever in pneumonia'
                      ),
        'prob_chest_indrawing_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of chest indrawing in pneumonia'
                      ),
        'prob_tachypnoea_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of tachypnoea in pneumonia'
                      ),
        'prob_danger_signs_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of any danger sign in pneumonia, including: : unable to drink, convulsions, '
                      'cyanosis, head nodding/bobbing, irritability, abnormally sleepy, lethargy, '
                      'nasal flaring, grunting'
                      ),
        'prob_cough_in_other_alri':
            Parameter(Types.REAL,
                      'probability of cough in bronchiolitis or other alri'
                      ),
        'prob_difficult_breathing_in_other_alri':
            Parameter(Types.REAL,
                      'probability of difficulty breathing in bronchiolitis or other alri'
                      ),
        'prob_fever_in_other_alri':
            Parameter(Types.REAL,
                      'probability of fever in bronchiolitis or other alri'
                      ),
        'prob_tachypnoea_in_other_alri':
            Parameter(Types.REAL,
                      'probability of tachypnoea in bronchiolitis or other alri'
                      ),
        'prob_chest_indrawing_in_other_alri':
            Parameter(Types.REAL,
                      'probability of chest wall indrawing in bronchiolitis or other alri'
                      ),
        'prob_danger_signs_in_other_alri':
            Parameter(Types.REAL,
                      'probability of any danger signs in bronchiolitis or other alri'
                      ),
        'prob_danger_signs_in_sepsis':
            Parameter(Types.REAL,
                      'probability of any danger signs in ALRI complicated by sepsis'
                      ),
        'prob_danger_signs_in_SpO2<90%':
            Parameter(Types.REAL,
                      'probability of any danger signs in children with SpO2 <90%'
                      ),
        'prob_danger_signs_in_SpO2_90-92%':
            Parameter(Types.REAL,
                      'probability of any danger signs in children with SpO2 between 90-92%'
                      ),
        'prob_chest_indrawing_in_SpO2<90%':
            Parameter(Types.REAL,
                      'probability of chest indrawing in children with SpO2 <90%'
                      ),
        'prob_chest_indrawing_in_SpO2_90-92%':
            Parameter(Types.REAL,
                      'probability of chest indrawing in children with SpO2 between 90-92%'
                      ),

        # Parameters governing the effects of vaccine ----------------
        'rr_Strep_pneum_VT_ALRI_with_PCV13_age<2y':
            Parameter(Types.REAL,
                      'relative rate of acquiring S. pneumoniae vaccine-type Alri '
                      'for children under 2 years of age immunised wth PCV13'
                      ),
        'rr_Strep_pneum_VT_ALRI_with_PCV13_age2to5y':
            Parameter(Types.REAL,
                      'relative rate of acquiring S. pneumoniae vaccine-type Alri '
                      'for children aged 2 to 5 immunised wth PCV13'
                      ),
        'rr_all_strains_Strep_pneum_ALRI_with_PCV13':
            Parameter(Types.REAL,
                      'relative rate of acquiring S. pneumoniae all types Alri '
                      'for children immunised wth PCV13'
                      ),
        'effectiveness_Hib_vaccine_on_Hib_strains':
            Parameter(Types.REAL,
                      'effectiveness of Hib vaccine against H. influenzae typ-b ALRI'
                      ),
        'rr_Hib_ALRI_with_Hib_vaccine':
            Parameter(Types.REAL,
                      'relative rate of acquiring H. influenzae type-b Alri '
                      'for children immunised wth Hib vaccine'
                      ),

        # Parameters governing treatment effectiveness and associated behaviours ----------------
        'days_between_treatment_and_cure':
            Parameter(Types.INT, 'number of days between any treatment being given in an HSI and the cure occurring.'
                      ),
        '3day_amoxicillin_for_fast_breathing_treatment_failure_or_relapse':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 6 or relapse by day 14 '
                      'of 3-day course amoxicillin for treating fast-breathing pneumonia'
                      ),
        '5day_amoxicillin_for_fast_breathing_treatment_failure_or_relapse':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 6 or relapse by day 14 '
                      'of 3-day course amoxicillin for treating fast-breathing pneumonia'
                      ),
        '3day_amoxicillin_for_chest_indrawing_treatment_failure_or_relapse':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 6 or relapse by day 14 '
                      'of 3-day course amoxicillin for treating chest-indrawing pneumonia'
                      ),
        '5day_amoxicillin_for_chest_indrawing_treatment_failure_or_relapse':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 6 or relapse by day 14 '
                      'of 5-day course amoxicillin for treating chest-indrawing pneumonia'
                      ),
        '1st_line_antibiotic_for_severe_pneumonia_treatment_failure_by_day2':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 2 '
                      'of first line antibiotic treatment for severe pneumonia'
                      ),
        'rr_1st_line_treatment_failure_low_oxygen_saturation':
            Parameter(Types.REAL,
                      'relative rate of treatment failure with first line of antibiotics '
                      'among children with low oxygen saturation < 90%'
                      ),
        '2nd_line_antibiotic_for_severe_pneumonia_mortality':
            Parameter(Types.REAL,
                      'mortality among children who changed to 2nd line antibiotics '
                      ),

        'sensitivity_of_classification_of_fast_breathing_pneumonia_facility_level0':
            Parameter(Types.REAL,
                      'sensitivity of correct classification and treatment decision by the HSA trained in iCCM,'
                      ' using paper-based tools, for fast-breathing pneumonia'
                      ),
        'sensitivity_of_classification_of_danger_signs_pneumonia_facility_level0':
            Parameter(Types.REAL,
                      'sensitivity of correct classification and referral decision by the HSA trained in iCCM,'
                      ' using paper-based tools, for danger-signs pneumonia'
                      ),
        'sensitivity_of_classification_of_non_severe_pneumonia_facility_level1':
            Parameter(Types.REAL,
                      'sensitivity of correct classification and treatment decision for non-severe pneumonia '
                      'at facility level 1a and 1b'
                      ),
        'sensitivity_of_classification_of_severe_pneumonia_facility_level1':
            Parameter(Types.REAL,
                      'sensitivity of correct classification and referral decision for severe pneumonia '
                      'at facility level 1a and 1b'
                      ),
        'sensitivity_of_classification_of_non_severe_pneumonia_facility_level2':
            Parameter(Types.REAL,
                      'sensitivity of correct classification and treatment decision for non-severe pneumonia '
                      'at facility level 2'
                      ),
        'sensitivity_of_classification_of_severe_pneumonia_facility_level2':
            Parameter(Types.REAL,
                      'sensitivity of correct classification and treatment decision for severe pneumonia '
                      'at facility level 2'
                      ),
        'prob_iCCM_severe_pneumonia_treated_as_fast_breathing_pneumonia':
            Parameter(Types.REAL,
                      'probability of misdiagnosis of iCCM severe pneumonia (with chest-indrawing or danger signs) '
                      'given treatment for non-severe pneumonia (fast-breathing) '
                      'at facility level O'
                      ),
        'prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia':
            Parameter(Types.REAL,
                      'probability of misdiagnosis of IMCI severe pneumonia (with danger signs) '
                      'given treatment for non-severe pneumonia ( fast-breathing or chest-indrawing) '
                      'at facility level 1a/1b/2'
                      ),
    }

    PROPERTIES = {
        # ---- Alri status ----
        'ri_current_infection_status':
            Property(Types.BOOL,
                     'Does the person currently have an infection with a pathogen that can cause Alri.'
                     ),

        # ---- The pathogen which is the attributed cause of Alri ----
        'ri_primary_pathogen':
            Property(Types.CATEGORICAL,
                     'If infected, what is the pathogen with which the person is currently infected. (np.nan if not '
                     'infected)',
                     categories=list(all_pathogens)
                     ),
        # ---- The bacterial pathogen which is the attributed co-/secondary infection ----
        'ri_secondary_bacterial_pathogen':
            Property(Types.CATEGORICAL,
                     'If infected, is there a secondary bacterial pathogen (np.nan if none or not applicable)',
                     categories=list(pathogens['bacterial'])
                     ),
        # ---- The underlying Alri condition ----
        'ri_disease_type':
            Property(Types.CATEGORICAL, 'If infected, what disease type is the person currently suffering from.',
                     categories=disease_types
                     ),
        # ---- The peripheral oxygen saturation level ----
        'ri_SpO2_level':
            Property(Types.CATEGORICAL, 'Peripheral oxygen saturation level (Sp02), measure for hypoxaemia',
                     categories=['<90%', '90-92%', '>=93%']
                     ),

        # ---- Treatment Status ----
        'ri_on_treatment': Property(Types.BOOL, 'Is this person currently receiving treatment.'),

        # < --- (N.B. Other properties of the form 'ri_complication_{complication-name}' are added later.) -->

        # ---- Internal variables to schedule onset and deaths due to Alri ----
        'ri_start_of_current_episode': Property(Types.DATE,
                                                'date of onset of current Alri event (pd.NaT is not infected)'),
        'ri_scheduled_recovery_date': Property(Types.DATE,
                                               '(scheduled) date of recovery from current Alri event (pd.NaT is not '
                                               'infected or episode is scheduled to end in death)'),
        'ri_scheduled_death_date': Property(Types.DATE,
                                            '(scheduled) date of death caused by current Alri event (pd.NaT is not '
                                            'infected or episode will not cause death)'),
        'ri_end_of_current_episode': Property(Types.DATE,
                                              'date on which the last episode of Alri is resolved, (including allowing '
                                              'for the possibility that a cure is scheduled following onset). This is '
                                              'used to determine when a new episode can begin. This stops successive'
                                              ' episodes interfering with one another.'),
    }

    def __init__(self, name=None, resourcefilepath=None, log_indivdual=None, do_checks=False):
        super().__init__(name)

        # Store arguments provided
        self.resourcefilepath = resourcefilepath
        self.do_checks = do_checks

        assert (log_indivdual is None or isinstance(log_indivdual, int)) and (not isinstance(log_indivdual, bool))
        self.log_individual = log_indivdual

        # Initialise the pointer to where the models will be stored:
        self.models = None

        # Maximum duration of an episode (beginning with infection and ending with recovery)
        self.max_duration_of_episode = None

        # dict to hold the DALY weights
        self.daly_wts = dict()

        # store the consumables needed
        self.consumables_used_in_hsi = dict()

        # Pointer to store the logging event used by this module
        self.logging_event = None

    def read_parameters(self, data_folder):
        """
        * Setup parameters values used by the module
        * Define symptoms
        """
        self.load_parameters_from_dataframe(
            pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Alri.xlsx', sheet_name='Parameter_values')
        )

        self.check_params_read_in_ok()

        self.define_symptoms()

    def check_params_read_in_ok(self):
        """Check that every value has been read-in successfully"""
        for param_name, param_type in self.PARAMETERS.items():
            assert param_name in self.parameters, f'Parameter "{param_name}" ' \
                                                  f'is not read in correctly from the resourcefile.'
            assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert isinstance(self.parameters[param_name],
                              param_type.python_type), f'Parameter "{param_name}" ' \
                                                       f'is not read in correctly from the resourcefile.'

    def define_symptoms(self):
        """Define the symptoms that this module will use"""
        all_symptoms = {
            'cough', 'difficult_breathing', 'fever', 'tachypnoea', 'chest_indrawing', 'danger_signs'
        }

        for symptom_name in sorted(all_symptoms):
            if symptom_name not in self.sim.modules['SymptomManager'].generic_symptoms:
                self.sim.modules['SymptomManager'].register_symptom(
                    Symptom(name=symptom_name, emergency_in_children=(symptom_name == 'danger_signs'))
                    # (associates the symptom with the 'average' healthcare seeking, apart from "danger_signs",
                    #  which is an emergency symptom in children.
                )

    def pre_initialise_population(self):
        """Define columns for complications at run-time"""
        for complication in self.complications:
            Alri.PROPERTIES[f"ri_complication_{complication}"] = Property(
                Types.BOOL, f"Whether this person has complication {complication}"
            )

    def initialise_population(self, population):
        """
        Sets that there is no one with Alri at initiation.
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals

        # ---- Key Current Status Classification Properties ----
        df.loc[df.is_alive, 'ri_current_infection_status'] = False
        df.loc[df.is_alive, 'ri_primary_pathogen'] = np.nan
        df.loc[df.is_alive, 'ri_secondary_bacterial_pathogen'] = np.nan
        df.loc[df.is_alive, 'ri_disease_type'] = np.nan
        df.loc[df.is_alive, [f"ri_complication_{complication}" for complication in self.complications]] = False
        df.loc[df.is_alive, 'ri_SpO2_level'] = ">=93%"

        # ---- Internal values ----
        df.loc[df.is_alive, 'ri_start_of_current_episode'] = pd.NaT
        df.loc[df.is_alive, 'ri_scheduled_recovery_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_scheduled_death_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_end_of_current_episode'] = pd.NaT
        df.loc[df.is_alive, 'ri_on_treatment'] = False

    def initialise_simulation(self, sim):
        """
        Prepares for simulation:
        * Schedules the main polling event
        * Schedules the main logging event
        * Establishes the linear models and other data structures using the parameters that have been read-in
        """
        p = self.parameters

        # Schedule the main polling event (to first occur immediately)
        sim.schedule_event(AlriPollingEvent(self), sim.date)

        # Schedule the main logging event (to first occur in one year)
        self.logging_event = AlriLoggingEvent(self)
        sim.schedule_event(self.logging_event, sim.date + DateOffset(days=364))

        if self.log_individual is not None:
            # Schedule the individual check logging event (to first occur immediately, and to occur every day)
            sim.schedule_event(AlriIndividualLoggingEvent(self), sim.date)

        if self.do_checks:
            # Schedule the event that does checking every day:
            sim.schedule_event(AlriCheckPropertiesEvent(self), sim.date)

        # Generate the model that determine the Natural History of the disease:
        self.models = Models(self)

        # Get DALY weights
        if 'HealthBurden' in self.sim.modules.keys():
            self.daly_wts['daly_non_severe_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
            self.daly_wts['daly_severe_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=46)

        # Define the max episode duration
        self.max_duration_of_episode = DateOffset(
            days=p['max_alri_duration_in_days_without_treatment'] + p['days_between_treatment_and_cure']
        )

        # Look-up and store the consumables that are required for each HSI
        self.look_up_consumables()

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        # ---- Key Current Status Classification Properties ----
        df.at[child_id, 'ri_current_infection_status'] = False
        df.loc[child_id, ['ri_primary_pathogen',
                          'ri_secondary_bacterial_pathogen',
                          'ri_disease_type']] = np.nan
        df.at[child_id, [f"ri_complication_{complication}" for complication in self.complications]] = False
        df.at[child_id, 'ri_SpO2_level'] = ">=93%"

        # ---- Internal values ----
        df.loc[child_id, ['ri_start_of_current_episode',
                          'ri_scheduled_recovery_date',
                          'ri_scheduled_death_date',
                          'ri_end_of_current_episode']] = pd.NaT
        df.at[child_id, 'ri_on_treatment'] = False

    def report_daly_values(self):
        """Report DALY incurred in the population in the last month due to ALRI"""
        df = self.sim.population.props

        # get the list of people with severe pneumonia: danger signs AND either cough or difficult breathing
        has_danger_signs = set(self.sim.modules['SymptomManager'].who_has('danger_signs')) & set(
            self.sim.modules['SymptomManager'].who_has('cough')).union(
            self.sim.modules['SymptomManager'].who_has('difficult_breathing')
        )

        # get the list of people with non-severe pneumonia
        has_fast_breathing_or_chest_indrawing_but_not_danger_signs = set(
            self.sim.modules['SymptomManager'].who_has('tachypnoea')
        ).union(
            self.sim.modules['SymptomManager'].who_has('chest_indrawing')
        ) - has_danger_signs

        # report the DALYs occurred
        total_daly_values = pd.Series(data=0.0, index=df.index[df.is_alive])
        total_daly_values.loc[has_danger_signs] = self.daly_wts['daly_severe_ALRI']
        total_daly_values.loc[
            has_fast_breathing_or_chest_indrawing_but_not_danger_signs] = self.daly_wts['daly_non_severe_ALRI']

        # Split out by pathogen that causes the Alri
        dummies_for_pathogen = pd.get_dummies(df.loc[total_daly_values.index, 'ri_primary_pathogen'], dtype='float')
        daly_values_by_pathogen = dummies_for_pathogen.mul(total_daly_values, axis=0)

        # add prefix to label according to the name of the causes of disability declared
        daly_values_by_pathogen = daly_values_by_pathogen.add_prefix('ALRI_')
        return daly_values_by_pathogen

    def look_up_consumables(self):
        """Look up and store the consumables item codes used in each of the HSI."""
        # TODO @ Ines - these doses by age pattern look weird -- 1 month old has highest dose!!?!

        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name

        def get_dosage_for_age_in_months(age_in_whole_months: float, doses_by_age_in_months: Dict[int, float]):
            """Returns the dose corresponding to age, using the lookup provided in `doses`. The format of `doses` is:
             {<higher_age_boundary_of_age_group_in_months>: <dose>}."""

            for upper_age_bound_in_months, _dose in sorted(doses_by_age_in_months.items()):
                if age_in_whole_months < upper_age_bound_in_months:
                    return _dose
            return _dose

        # # # # # # Dosages by age # # # # # #

        # Antibiotic therapy -------------------

        # Antibiotics for non-severe pneumonia - oral amoxicillin for 5 days
        self.consumables_used_in_hsi['Amoxicillin_tablet_or_suspension_5days'] = {
            get_item_code(item='Amoxycillin 250mg_1000_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {12: 0.006, 36: 0.012, np.inf: 0.018}
                                                          ),
            get_item_code(item='Amoxycillin 125mg/5ml suspension, PFR_0.025_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {12: 1, 36: 2, np.inf: 3}
                                                          ),
        }

        self.consumables_used_in_hsi['Amoxicillin_tablet_or_suspension_3days'] = {
            get_item_code(item='Amoxycillin 250mg_1000_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {12: 0.01, 36: 0.02, np.inf: 0.03}
                                                          ),
            get_item_code(item='Amoxycillin 125mg/5ml suspension, PFR_0.025_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {12: 1, 36: 2, np.inf: 3}
                                                          ),
        }

        # Antibiotic therapy for severe pneumonia - ampicillin package
        self.consumables_used_in_hsi['Ampicillin_gentamicin_therapy_for_severe_pneumonia'] = {
            get_item_code(item='Ampicillin injection 500mg, PFR_each_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {4: 8, 12: 16, 36: 24, np.inf: 40}
                                                          ),
            get_item_code(item='Gentamicin Sulphate 40mg/ml, 2ml_each_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {4: 2.81, 12: 4.69, 36: 7.03, np.inf: 9.37}
                                                          ),
            get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST'): 1,
            get_item_code(item='Syringe, needle + swab'): 1
        }

        # Antibiotic therapy for severe pneumonia - benzylpenicillin package when ampicillin is not available
        self.consumables_used_in_hsi['Benzylpenicillin_gentamicin_therapy_for_severe_pneumonia'] = {
            get_item_code(item='Benzylpenicillin 3g (5MU), PFR_each_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {4: 8, 12: 15, 36: 24, np.inf: 34}
                                                          ),
            get_item_code(item='Gentamicin Sulphate 40mg/ml, 2ml_each_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {4: 2.81, 12: 4.69, 36: 7.03, np.inf: 9.37}
                                                          ),
            get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST'): 1,
            get_item_code(item='Syringe, needle + swab'): 1
        }

        # Second line of antibiotics for severe pneumonia
        self.consumables_used_in_hsi['Ceftriaxone_therapy_for_severe_pneumonia'] = {
            get_item_code(item='Ceftriaxone 1g, PFR_each_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {4: 1.5, 12: 3, 36: 5, np.inf: 7}
                                                          ),
            get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST'): 1,
            get_item_code(item='Syringe, needle + swab'): 1
        }

        # Second line of antibiotics for severe pneumonia, if Staph is suspected
        self.consumables_used_in_hsi['2nd_line_Antibiotic_therapy_for_severe_staph_pneumonia'] = {
            get_item_code(item='cloxacillin 500 mg, powder for injection_50_IDA'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {4: 5.6, 12: 11.2, 36: 16.8, np.inf: 22.4}
                                                          ),
            get_item_code(item='Gentamicin Sulphate 40mg/ml, 2ml_each_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {4: 2.81, 12: 4.69, 36: 7.03, np.inf: 9.37}
                                                          ),
            get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST'): 1,
            get_item_code(item='Syringe, needle + swab'): 1
        }

        # First dose of antibiotic before referral -------------------

        # Referral process in iCCM for severe pneumonia, and at health centres for HIV exposed/infected
        self.consumables_used_in_hsi['First_dose_oral_amoxicillin_for_referral'] = {
            get_item_code(item='Amoxycillin 250mg_1000_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {12: 0.001, 36: 0.002, np.inf: 0.003}
                                                          ),
        }
        # Referral process at health centres for severe cases
        self.consumables_used_in_hsi['First_dose_IM_antibiotics_for_referral'] = {
            get_item_code(item='Ampicillin injection 500mg, PFR_each_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {4: 0.4, 12: 0.8, 36: 1.4, np.inf: 2}
                                                          ),
            get_item_code(item='Gentamicin Sulphate 40mg/ml, 2ml_each_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {4: 0.56, 12: 0.94, 36: 1.41, np.inf: 1.87}
                                                          ),
            get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST'): 1,
            get_item_code(item='Syringe, needle + swab'): 1
        }

        # Oxygen, pulse oximetry and x-ray -------------------

        # Oxygen for hypoxaemia
        self.consumables_used_in_hsi['Oxygen_Therapy'] = {
            get_item_code(item='Oxygen, 1000 liters, primarily with oxygen cylinders'): 1,
            get_item_code(item='Nasal prongs'): 1
        }

        # Pulse oximetry
        self.consumables_used_in_hsi['Pulse_oximetry'] = {
            get_item_code(item='Oxygen, 1000 liters, primarily with oxygen cylinders'): 1
        }
        # use oxygen code to fill in consumable availability for pulse oximetry

        # X-ray scan
        self.consumables_used_in_hsi['X_ray_scan'] = {
            get_item_code(item='X-ray'): 1
        }

        # Optional consumables -------------------

        # Paracetamol
        self.consumables_used_in_hsi['Paracetamol_tablet'] = {
            get_item_code(item='Paracetamol, tablet, 100 mg'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {36: 12, np.inf: 18}
                                                          ),
        }

        # Maintenance of fluids via nasograstric tube
        self.consumables_used_in_hsi['Fluid_Maintenance'] = {
            get_item_code(item='Tube, nasogastric CH 8_each_CMST'): 1
        }

        # Bronchodilator
        # inhaled
        self.consumables_used_in_hsi['Inhaled_Brochodilator'] = {
            get_item_code(item='Salbutamol sulphate 1mg/ml, 5ml_each_CMST'): 2
        }

        # oral
        self.consumables_used_in_hsi['Oral_Brochodilator'] = {
            get_item_code(item='Salbutamol, syrup, 2 mg/5 ml'): 1,
            get_item_code(item='Salbutamol, tablet, 4 mg'): 1
        }

    def end_episode(self, person_id):
        """End the episode infection for a person (i.e. reset all properties to show no current infection or
        complications).
        This is called by AlriNaturalRecoveryEvent and AlriCureEvent.
        NB. 'ri_end_of_current_episode is not reset: this is used to prevent new infections from occurring whilst HSI
        from a previous episode may still be scheduled to occur.
        """
        df = self.sim.population.props

        # Reset properties to show no current infection:
        new_properties = {
            'ri_current_infection_status': False,
            'ri_primary_pathogen': np.nan,
            'ri_secondary_bacterial_pathogen': np.nan,
            'ri_disease_type': np.nan,
            'ri_SpO2_level': '>=93%',
            'ri_on_treatment': False,
            'ri_start_of_current_episode': pd.NaT,
            'ri_scheduled_recovery_date': pd.NaT,
            'ri_scheduled_death_date': pd.NaT,
        }
        df.loc[person_id, new_properties.keys()] = new_properties.values()

        # Remove all existing complications
        df.loc[person_id, [f"ri_complication_{c}" for c in self.complications]] = False

        # Resolve all the symptoms immediately
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id, disease_module=self)

    def cancel_death_and_schedule_cure(self, person_id):
        """Cancels a scheduled date of death due to Alri for a person, and schedules the CureEvent.
        This is called within do_alri_treatment function.
        Cancelling scheduled death date prior to the scheduling the CureEvent prevent deaths happening
        in the time between a treatment being given and the cure event occurring.
        :param person_id:
        :return:
        """
        df = self.sim.population.props

        # cancel death date
        self.sim.population.props.at[person_id, 'ri_scheduled_death_date'] = pd.NaT

        # Determine cure date, and update recovery date
        cure_date = self.sim.date + DateOffset(days=self.parameters['days_between_treatment_and_cure'])
        df.at[person_id, 'ri_scheduled_recovery_date'] = cure_date

        # Schedule the CureEvent
        self.sim.schedule_event(AlriCureEvent(self, person_id), cure_date)

    def check_properties(self):
        """This is used in debugging to make sure that the configuration of properties is correct"""

        df = self.sim.population.props

        # identify those who currently have an infection with a pathogen that can cause ALRI:
        curr_inf = df['is_alive'] & df['ri_current_infection_status']
        not_curr_inf = df['is_alive'] & ~df['ri_current_infection_status']

        # For those with no current infection, variables about the current infection should be null
        assert df.loc[not_curr_inf, [
            'ri_primary_pathogen',
            'ri_secondary_bacterial_pathogen',
            'ri_disease_type',
            'ri_start_of_current_episode',
            'ri_scheduled_recovery_date',
            'ri_scheduled_death_date']
        ].isna().all().all()

        # For those with no current infection, 'ri_end_of_current_episode' should be null or in the past or within the
        # period for which the episode can last.
        assert (
            df.loc[not_curr_inf, 'ri_end_of_current_episode'].isna() |
            (df.loc[not_curr_inf, 'ri_end_of_current_episode'] <= self.sim.date) |
            (
                (df.loc[not_curr_inf, 'ri_end_of_current_episode'] - self.sim.date).dt.days
                <= self.max_duration_of_episode.days
            )
        ).all()

        # For those with no current infection, there should be no treatment
        assert not df.loc[not_curr_inf, 'ri_on_treatment'].any()

        # For those with no current infection, there should be no complications
        assert not df.loc[
            not_curr_inf, [f"ri_complication_{c}" for c in self.complications]
        ].any().any()

        # For those with current infection, variables about the current infection should not be null
        assert not df.loc[curr_inf, [
            'ri_primary_pathogen',
            'ri_disease_type']
        ].isna().any().any()

        # For those with current infection, dates relating to this episode should make sense
        # - start is in the past and end is in the future
        assert (df.loc[curr_inf, 'ri_start_of_current_episode'] <= self.sim.date).all()
        assert (df.loc[curr_inf, 'ri_end_of_current_episode'] >= self.sim.date).all()

        # - a person has exactly one of a recovery_date _or_ a death_date
        assert ((~df.loc[curr_inf, 'ri_scheduled_recovery_date'].isna()) | (
            ~df.loc[curr_inf, 'ri_scheduled_death_date'].isna())).all()
        assert (df.loc[curr_inf, 'ri_scheduled_recovery_date'].isna() != df.loc[
            curr_inf, 'ri_scheduled_death_date'].isna()).all()

        #  If that primary pathogen is bacterial then there should be np.nan for secondary_bacterial_pathogen:
        assert df.loc[
            curr_inf & df['ri_primary_pathogen'].isin(self.pathogens['bacterial']), 'ri_secondary_bacterial_pathogen'
        ].isna().all()

        # There should be consistency between the properties for oxygen saturation and the presence of the complication
        # hypoxaemia
        assert (df.loc[df.is_alive & df['ri_complication_hypoxaemia'], 'ri_SpO2_level'] != '>=93%').all()
        assert (df.loc[df.is_alive & ~df['ri_complication_hypoxaemia'], 'ri_SpO2_level'] == '>=93%').all()

    def impose_symptoms_for_complication(self, person_id, complication, oxygen_saturation, duration_in_days):
        """Impose symptoms for a complication."""
        symptoms = sorted(self.models.symptoms_for_complication(
            complication=complication, oxygen_saturation=oxygen_saturation))

        # pick_days_following_onset = self.rng.randint(0, duration_in_days)
        # date_onset_symptoms = self.sim.date + DateOffset(days=pick_days_following_onset)

        self.sim.modules['SymptomManager'].change_symptom(
            person_id=person_id,
            symptom_string=symptoms,
            add_or_remove='+',
            date_of_onset=self.sim.date,
            duration_in_days=duration_in_days,
            disease_module=self,
        )

    def _treatment_fails(self, imci_symptom_based_classification: str, needs_oxygen: bool, antibiotic_provided: str,
                         oxygen_provided: bool) -> bool:
        """Returns True if the treatment specified will prevent death."""

        def _raise_error():
            raise ValueError(f"No treatment effectiveness defined: {imci_symptom_based_classification=}, "
                             f"{needs_oxygen=}, {antibiotic_provided=}")

        p = self.parameters

        if not needs_oxygen:
            # For no hypoxaemia (SpO2 >= 90%) -----

            if imci_symptom_based_classification == 'chest_indrawing_pneumonia':
                if antibiotic_provided == '5day_oral_amoxicillin':
                    return p['5day_amoxicillin_for_chest_indrawing_treatment_failure_or_relapse'] > \
                           self.rng.random_sample()
                elif antibiotic_provided == '3day_oral_amoxicillin':
                    return p['3day_amoxicillin_for_chest_indrawing_treatment_failure_or_relapse'] >\
                           self.rng.random_sample()
                else:
                    _raise_error()

            # fast-breathing pneumonia
            elif imci_symptom_based_classification == 'fast_breathing_pneumonia':
                if antibiotic_provided == '3day_oral_amoxicillin':
                    return p['3day_amoxicillin_for_fast_breathing_treatment_failure_or_relapse'] > \
                           self.rng.random_sample()
                elif antibiotic_provided == '5day_oral_amoxicillin':
                    return p['5day_amoxicillin_for_chest_indrawing_treatment_failure_or_relapse'] > \
                           self.rng.random_sample()
                else:
                    _raise_error()

            # danger-signs pneumonia
            elif imci_symptom_based_classification in ('danger_signs_pneumonia', 'serious_bacterial_infection'):
                if antibiotic_provided == '1st_line_IV_antibiotics':
                    return p['1st_line_antibiotic_for_severe_pneumonia_treatment_failure_by_day2'] \
                           > self.rng.random_sample()
                elif antibiotic_provided == '5day_oral_amoxicillin':
                    return True
                else:
                    _raise_error()

            elif imci_symptom_based_classification == "cough_or_cold":
                return False  # Treatment cannot 'fail' for a cough_or_cold

            else:
                _raise_error()

        else:
            # todo - @Ines - this should depend on whether or not oxygen is actually provided!
            # For hypoxaemia (SpO2 < 90%) -----
            if imci_symptom_based_classification in ('fast_breathing_pneumonia', 'chest_indrawing_pneumonia'):
                # todo @Ines - treatment always fails, no matter what is provided! Ask consultant and make this depend
                #  whether oxygen provided. ("Ask him if this classification gets oral antibiotics and oxygen- does it
                #  have an effect").
                return True

            # danger-signs pneumonia
            elif imci_symptom_based_classification in ('danger_signs_pneumonia', 'serious_bacterial_infection'):
                if antibiotic_provided == '1st_line_IV_antibiotics':
                    return (
                               p['1st_line_antibiotic_for_severe_pneumonia_treatment_failure_by_day2']
                               * p['rr_1st_line_treatment_failure_low_oxygen_saturation']
                           ) > self.rng.random_sample()
                else:
                    return True

            elif imci_symptom_based_classification == "cough_or_cold":
                return False  # Treatment cannot 'fail' for a cough_or_cold

            else:
                _raise_error()

    def do_effects_of_treatment(self, person_id, antibiotic_provided: str, oxygen_provided: bool) -> None:
        """Helper function that enacts the effects of a treatment to Alri caused by a pathogen.
        It will only do something if the Alri is caused by a pathogen (this module).
        * Prevent any death event that may be scheduled from occurring
        * Schedules a follow-up appointment if condition not improving (by day 6 or by day 14)
        """
        df = self.sim.population.props
        person = df.loc[person_id]

        if not person.ri_current_infection_status:
            return

        # Record that the person is now on treatment:
        self.sim.population.props.at[person_id, 'ri_on_treatment'] = True
        self.logging_event.new_treated()

        # Gather underlying properties that will affect success of treatment
        needs_oxygen = person.ri_SpO2_level == '<90%'
        imci_symptom_based_classification = self.get_imci_classification_based_on_symptoms(
            child_is_younger_than_2_months=person.age_exact_years < (2.0 / 12.0),
            symptoms=self.sim.modules['SymptomManager'].has_what(person_id)
        )

        # Will the treatment fail (depends on the treatment given as well as underlying properties)
        treatment_fails = self._treatment_fails(
            antibiotic_provided=antibiotic_provided, oxygen_provided=oxygen_provided,
            imci_symptom_based_classification=imci_symptom_based_classification, needs_oxygen=needs_oxygen)

        # Cancel death if the treatment is successful:
        if not treatment_fails:
            self.cancel_death_and_schedule_cure(person_id)

    def on_presentation(self, person_id, hsi_event):
        """Action taken when a child (under 5 years old) presents at a generic appointment (emergency or non-emergency)
         with symptoms of `cough` or `difficult_breathing`."""

        self.record_sought_care_for_alri()

        # We give all persons an out-patient appointment at the current facility level.
        self.sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_event=HSI_Alri_Treatment(person_id=person_id, module=self,
                                         facility_level=hsi_event.ACCEPTED_FACILITY_LEVEL),
            topen=self.sim.date,
            tclose=self.sim.date + pd.DateOffset(days=1),
            priority=1
        )

    def record_sought_care_for_alri(self):
        """Count that the person is seeking care"""
        self.logging_event.new_seeking_care()

    @staticmethod
    def get_imci_classification_based_on_symptoms(child_is_younger_than_2_months: bool, symptoms: list) -> str:
        """Based on age and symptoms, classify WHO-pneumonia severity. This is regarded as the *TRUE* classification
         based on symptoms. It will return one of: {
             'serious_bacterial_infection',
             'fast_breathing_pneumonia',
             'danger_signs_pneumonia',
             'chest_indrawing_pneumonia,
             'cough_or_cold'
        }."""

        if child_is_younger_than_2_months:
            if ('chest_indrawing' in symptoms) or ('danger_signs' in symptoms):
                return 'serious_bacterial_infection'
            elif 'tachypnoea' in symptoms:
                return 'fast_breathing_pneumonia'
            else:
                return 'cough_or_cold'

        else:
            if 'danger_signs' in symptoms:
                return 'danger_signs_pneumonia'
            elif 'chest_indrawing' in symptoms:
                return 'chest_indrawing_pneumonia'
            elif 'tachypnoea' in symptoms:
                return 'fast_breathing_pneumonia'
            else:
                return 'cough_or_cold'


class Models:
    """Helper-class to store all the models that specify the natural history of the Alri disease"""

    def __init__(self, module):
        self.module = module
        self.p = module.parameters
        self.rng = module.rng

        # dict that will hold the linear models for incidence risk for each pathogen
        self.incidence_equations_by_pathogen = dict()

        # set-up the linear models for the incidence risk for each pathogen
        self.make_model_for_acquisition_risk()

    def make_model_for_acquisition_risk(self):
        """"Model for the acquisition of a primary pathogen that can cause ALri"""
        p = self.p
        df = self.module.sim.population.props

        def make_scaled_linear_model_for_incidence(target_prob_by_age):
            """Return a linear model that is scaled for each the age-groups.
            It does this by first making an unscaled linear model with an intercept of 1.0. Using this, it calculates
             the resulting mean incidence rate for the population (by age-group). It then creates a new linear model
             with an adjusted age effect so that the average incidence in the population (by age-group) matches the
             target.
            NB. It will not work if there are no persons of that age in the population.
            """

            def make_linear_model(age_effects=None):
                """Make the linear model based for a particular pathogen and a particular age-group."""

                if age_effects is None:
                    age_effects = {
                        0: 1.0,
                        1: 1.0,
                        2: 1.0,
                        3: 1.0,
                        4: 1.0,
                    }

                return LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    1.0,
                    Predictor(
                        'age_years',
                        conditions_are_mutually_exclusive=True,
                        conditions_are_exhaustive=True).when(0, age_effects[0])
                                                       .when(1, age_effects[1])
                                                       .when(2, age_effects[2])
                                                       .when(3, age_effects[3])
                                                       .when(4, age_effects[4])
                                                       .when('>= 5', 0.0),
                    Predictor('li_wood_burn_stove').when(False, p['rr_ALRI_indoor_air_pollution']),
                    Predictor().when('(va_measles_all_doses == False) & (age_years >= 1)',
                                     p['rr_ALRI_incomplete_measles_immunisation']),
                    Predictor().when('(hv_inf == True) & (hv_art!= "on_VL_suppressed")', p['rr_ALRI_HIV/AIDS']),
                    Predictor().when('(nb_breastfeeding_status != "exclusive") & (age_exact_years < 1/6)',
                                     p['rr_ALRI_non_exclusive_breastfeeding'])
                )

            # Unpack `target_prob_by_age` to a dict with keys that match `age_years_exact` = 0, 1, 2, 3, 4
            _target = {
                0: target_prob_by_age[0],
                1: target_prob_by_age[1],
                2: target_prob_by_age[2],
                3: target_prob_by_age[2],
                4: target_prob_by_age[2],
            }

            # make unscaled linear model
            unscaled_lm = make_linear_model()

            # Compute scaling factors for each age-group:
            _age_effect = (pd.Series(_target) /
                           unscaled_lm.predict(df.loc[df.is_alive & (df.age_years < 5)]).groupby(by=df.age_years).mean()
                           ).fillna(1.0).to_dict()

            # Return the linear model with the appropriate scaling factor
            return make_linear_model(age_effects=_age_effect)

        for patho in self.module.all_pathogens:
            self.incidence_equations_by_pathogen[patho] = \
                make_scaled_linear_model_for_incidence(target_prob_by_age=self.p[f'base_inc_rate_ALRI_by_{patho}'])

    def compute_risk_of_acquisition(self, pathogen, df):
        """Compute the risk of a pathogen, using the linear model created and the df provided"""
        p = self.p

        # Run linear model to get baseline risk
        baseline = self.incidence_equations_by_pathogen[pathogen].predict(df)

        # apply the reduced risk of acquisition for those vaccinated
        if pathogen == "Strep_pneumoniae_PCV13":
            baseline.loc[df['va_pneumo_all_doses'] & (df['age_years'] < 2)] \
                *= p['rr_Strep_pneum_VT_ALRI_with_PCV13_age<2y']
            baseline.loc[df['va_pneumo_all_doses'] & (df['age_years'].between(2, 5))] \
                *= p['rr_Strep_pneum_VT_ALRI_with_PCV13_age2to5y']
        elif pathogen == "Hib":
            baseline.loc[df['va_hib_all_doses']] *= p['rr_Hib_ALRI_with_Hib_vaccine']

        return baseline

    def determine_disease_type_and_secondary_bacterial_coinfection(self, pathogen, age,
                                                                   va_hib_all_doses, va_pneumo_all_doses):
        """Determines:
         * the disease that is caused by infection with this pathogen (from among self.disease_types)
         * if there is a bacterial coinfection associated that will cause the dominant disease.

         Note that the disease_type is 'bacterial_pneumonia' if primary pathogen is viral and there is a secondary
         bacterial coinfection.
         """
        p = self.p

        # Determine the disease type - pneumonia or other_alri
        if (
            (age < 1) and (p[f'proportion_pneumonia_in_{pathogen}_ALRI'][0] > self.rng.random_sample())
        ) or (
            (1 <= age < 5) and (p[f'proportion_pneumonia_in_{pathogen}_ALRI'][1] > self.rng.random_sample())
        ):
            disease_type = 'pneumonia'
        else:
            disease_type = 'other_alri'

        # Determine bacterial-coinfection
        if pathogen in self.module.pathogens['viral']:
            if disease_type == 'pneumonia':
                if p['prob_viral_pneumonia_bacterial_coinfection'] > self.rng.random_sample():
                    bacterial_coinfection = self.secondary_bacterial_infection(va_hib_all_doses=va_hib_all_doses,
                                                                               va_pneumo_all_doses=va_pneumo_all_doses)
                else:
                    bacterial_coinfection = np.nan
            else:
                # brochiolitis/other_alri (viral)
                bacterial_coinfection = np.nan
        else:
            # No bacterial co-infection in primary bacterial or fungal cause
            bacterial_coinfection = np.nan

        assert disease_type in self.module.disease_types
        assert bacterial_coinfection in (self.module.pathogens['bacterial'] + [np.nan])

        return disease_type, bacterial_coinfection

    def secondary_bacterial_infection(self, va_hib_all_doses, va_pneumo_all_doses):
        """Determine which specific bacterial pathogen causes a secondary coinfection, or if there is no secondary
        bacterial infection (due to the effects of the pneumococcal vaccine).
        """
        p = self.p

        # get probability of bacterial coinfection with each pathogen
        probs = dict(zip(
            self.module.pathogens['bacterial'], p['proportion_bacterial_coinfection_pathogen']))

        # Edit the probability that the coinfection will be of `Strep_pneumoniae_PCV13` if the person has had
        # the pneumococcal vaccine:
        if va_pneumo_all_doses:
            probs['Strep_pneumoniae_PCV13'] *= p['rr_Strep_pneum_VT_ALRI_with_PCV13_age2to5y']

        # Edit the probability that the coinfection will be of `Hib` if the person has had
        # the hib vaccine:
        if va_hib_all_doses:
            probs['Hib'] *= p['rr_Hib_ALRI_with_Hib_vaccine']

        # Add in the probability that there is none (to ensure that all probabilities sum to 1.0)
        probs['_none_'] = 1.0 - sum(probs.values())

        # return the random selection of bacterial coinfection (including possibly np.nan for 'none')
        outcome = self.rng.choice(list(probs.keys()), p=list(probs.values()))

        return outcome if outcome != '_none_' else np.nan

    def get_complications_that_onset(self, disease_type, primary_path_is_bacterial, has_secondary_bacterial_inf):
        """Determine the set of complication for this person"""
        p = self.p

        probs = defaultdict(float)

        # probabilities for local pulmonary complications
        prob_pulmonary_complications = p['prob_pulmonary_complications_in_pneumonia']
        if disease_type == 'pneumonia':
            if prob_pulmonary_complications > self.rng.random_sample():
                for c in ['pneumothorax', 'pleural_effusion', 'lung_abscess', 'empyema']:
                    probs[c] += p[f'prob_{c}_in_pulmonary_complicated_pneumonia']

            # probabilities for systemic complications
            if primary_path_is_bacterial or has_secondary_bacterial_inf:
                probs['sepsis'] += p['prob_bacteraemia_in_pneumonia'] * p['prob_progression_to_sepsis_with_bacteraemia']

            probs['hypoxaemia'] += p['prob_hypoxaemia_in_pneumonia']

        elif disease_type == 'other_alri':
            probs['hypoxaemia'] += p['prob_hypoxaemia_in_other_alri']

        # determine which complications are onset:
        complications = {c for c, p in probs.items() if p > self.rng.random_sample()}

        return complications

    def get_oxygen_saturation(self, complication_set):
        """Set peripheral oxygen saturation"""

        if 'hypoxaemia' in complication_set:
            if self.p['proportion_hypoxaemia_with_SpO2<90%'] > self.rng.random_sample():
                return '<90%'
            else:
                return '90-92%'
        else:
            return '>=93%'

    def symptoms_for_disease(self, disease_type):
        """Determine set of symptom (before complications) for a given instance of disease"""
        p = self.p

        assert disease_type in self.module.disease_types

        probs = {
            symptom: p[f'prob_{symptom}_in_{disease_type}']
            for symptom in [
                'cough', 'difficult_breathing', 'fever', 'tachypnoea', 'chest_indrawing', 'danger_signs']
        }

        # determine which symptoms are onset:
        symptoms = {s for s, p in probs.items() if p > self.rng.random_sample()}

        return symptoms

    def symptoms_for_complication(self, complication, oxygen_saturation):
        """Probability of each symptom for a person given a complication"""
        p = self.p

        probs = defaultdict(float)

        if complication == 'hypoxaemia':
            if oxygen_saturation == '<90%':
                probs = {
                    'danger_signs': p['prob_danger_signs_in_SpO2<90%'],
                    'chest_indrawing': p['prob_chest_indrawing_in_SpO2<90%']
                }
            elif oxygen_saturation == '90-92%':
                probs = {
                    'danger_signs': p['prob_danger_signs_in_SpO2_90-92%'],
                    'chest_indrawing': p['prob_chest_indrawing_in_SpO2_90-92%']
                }

        elif complication == 'sepsis':
            probs = {
                'danger_signs': p['prob_danger_signs_in_sepsis']
            }

        # determine which symptoms are onset:
        symptoms = {s for s, p in probs.items() if p > self.rng.random_sample()}

        return symptoms

    def will_die_of_alri(self, person_id):
        """Determine if person will die from Alri. Returns True/False"""
        p = self.p
        df = self.module.sim.population.props
        person = df.loc[person_id]
        # check if any complications - death occurs only if a complication is present
        any_complications = person[[f'ri_complication_{c}' for c in self.module.complications]].any()

        # Baseline risk:
        odds_death = p['baseline_odds_alri_death']

        # The effect of age:
        if person['age_exact_years'] < 1.0 / 6.0:
            odds_death *= p['or_death_ALRI_age<2mo']

        # The effect of gender:
        if person['sex'] == 'M':
            odds_death *= p['or_death_ALRI_male']

        # The effect of P.jirovecii infection:
        if person['ri_primary_pathogen'] == 'P.jirovecii':
            odds_death *= p['or_death_ALRI_P.jirovecii']

        # The effect of hypoxaemia:
        if person['ri_complication_hypoxaemia']:
            odds_death *= p['or_death_ALRI_SpO2<93%']

        # The effect of factors defined in other modules:
        # HIV
        if person['hv_inf'] & (person['hv_art'] != "on_VL_suppressed"):
            odds_death *= p['or_death_ALRI_HIV/AIDS']

        # malnutrition:
        if person['un_clinical_acute_malnutrition'] == 'SAM':
            odds_death *= p['or_death_ALRI_SAM']
        elif person['un_clinical_acute_malnutrition'] == 'MAM':
            odds_death *= p['or_death_ALRI_MAM']

        # Convert odds to probability
        risk_death = odds_death / (1 + odds_death)

        if any_complications:
            return risk_death > self.rng.random_sample()
        else:
            return False


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class AlriPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """This is the main event that runs the acquisition of pathogens that cause Alri.
    It determines who is infected and when and schedules individual IncidentCase events to represent onset.

    A known issue is that Alri events are scheduled based on the risk of current age but occur a short time
    later when the children will be slightly older. This means that when comparing the model output with data, the
    model slightly under-represents incidence among younger age-groups and over-represents incidence among older
    age-groups. This is a small effect when the frequency of the polling event is high."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=2))
        self.fraction_of_a_year_until_next_polling_event = self.compute_fraction_of_year_between_polling_event()

    def compute_fraction_of_year_between_polling_event(self):
        """Compute fraction of a year that elapses between polling event. This is used to adjust the risk of
        infection"""
        return (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'Y')

    def apply(self, population):
        """Determine who will become infected and schedule for them an AlriComplicationOnsetEvent"""

        df = population.props
        m = self.module
        models = m.models

        # Compute the incidence rate for each person getting Alri and then convert into a probability
        # getting all children that do not currently have an Alri episode (never had or last episode resolved)
        mask_could_get_new_alri_event = (
            df.is_alive & (df.age_years < 5) & ~df.ri_current_infection_status &
            ((df.ri_end_of_current_episode < self.sim.date) | pd.isnull(df.ri_end_of_current_episode))
        )

        # Compute the incidence rate for each person acquiring Alri
        inc_of_acquiring_alri = pd.DataFrame(index=df.loc[mask_could_get_new_alri_event].index)
        for pathogen in m.all_pathogens:
            inc_of_acquiring_alri[pathogen] = models.compute_risk_of_acquisition(
                pathogen=pathogen,
                df=df.loc[mask_could_get_new_alri_event]
            )

        probs_of_acquiring_pathogen = 1 - np.exp(
            -inc_of_acquiring_alri * self.fraction_of_a_year_until_next_polling_event
        )

        # Sample to find outcomes:
        outcome = sample_outcome(probs=probs_of_acquiring_pathogen, rng=self.module.rng)

        # For persons that will become infected with a particular pathogen:
        for person_id, pathogen in outcome.items():
            #  Create the event for the onset of infection:
            self.sim.schedule_event(
                event=AlriIncidentCase(
                    module=self.module,
                    person_id=person_id,
                    pathogen=pathogen,
                ),
                date=random_date(self.sim.date, self.sim.date + self.frequency - pd.DateOffset(days=1), m.rng)
            )


class AlriIncidentCase(Event, IndividualScopeEventMixin):
    """This Event is for the onset of the infection that causes Alri. It is scheduled by the AlriPollingEvent."""

    def __init__(self, module, person_id, pathogen):
        super().__init__(module, person_id=person_id)
        self.pathogen = pathogen

    def apply(self, person_id):
        """
        * Determines the disease and complications associated with this case
        * Updates all the properties so that they pertain to this current episode of Alri
        * Imposes the symptoms
        * Schedules relevant natural history event {(either AlriNaturalRecoveryEvent or AlriDeathEvent)}
        * Updates the counters in the log accordingly.
        """
        df = self.sim.population.props  # shortcut to the dataframe
        person = df.loc[person_id]
        m = self.module
        p = m.parameters
        rng = self.module.rng
        models = m.models

        # The event should not run if the person is not currently alive:
        if not person.is_alive:
            return

        # Log the incident case:
        self.module.logging_event.new_case(age=person.age_years, pathogen=self.pathogen)

        # ----------------- Determine the Alri disease type and bacterial coinfection for this case -----------------
        disease_type, bacterial_coinfection = models.determine_disease_type_and_secondary_bacterial_coinfection(
            age=person.age_years, pathogen=self.pathogen,
            va_hib_all_doses=person.va_hib_all_doses, va_pneumo_all_doses=person.va_pneumo_all_doses)

        # ----------------------- Duration of the Alri event -----------------------
        duration_in_days_of_alri = rng.randint(1, p['max_alri_duration_in_days_without_treatment'])
        # assumes uniform interval around mean duration of 7 days, with range 14 days

        # Date for outcome (either recovery or death) with uncomplicated Alri
        date_of_outcome = m.sim.date + DateOffset(days=duration_in_days_of_alri)

        # Define 'episode end' date. This the date when this episode ends. It is the last possible data that any HSI
        # could affect this episode.
        episode_end = date_of_outcome + DateOffset(days=p['days_between_treatment_and_cure'])

        # Update the properties in the dataframe:
        _chars = {
            'ri_current_infection_status': True,
            'ri_primary_pathogen': self.pathogen,
            'ri_secondary_bacterial_pathogen': bacterial_coinfection,
            'ri_disease_type': disease_type,
            'ri_on_treatment': False,
            'ri_start_of_current_episode': self.sim.date,
            'ri_scheduled_recovery_date': pd.NaT,
            'ri_scheduled_death_date': pd.NaT,
            'ri_end_of_current_episode': episode_end,
        }
        df.loc[person_id, _chars.keys()] = _chars.values()

        # ----------------------------------- Clinical Symptoms -----------------------------------
        # impose clinical symptoms for new uncomplicated Alri
        self.impose_symptoms_for_uncomplicated_disease(person_id=person_id, disease_type=disease_type,
                                                       duration_in_days=duration_in_days_of_alri)

        # ----------------------------------- Complications  -----------------------------------
        self.impose_complications(person_id=person_id, duration_in_days=duration_in_days_of_alri)

        # ----------------------------------- Outcome  -----------------------------------
        if models.will_die_of_alri(person_id=person_id):
            self.sim.schedule_event(AlriDeathEvent(self.module, person_id), date_of_outcome)
            df.loc[person_id, ['ri_scheduled_death_date', 'ri_scheduled_recovery_date']] = [date_of_outcome, pd.NaT]
        else:
            self.sim.schedule_event(AlriNaturalRecoveryEvent(self.module, person_id), date_of_outcome)
            df.loc[person_id, ['ri_scheduled_recovery_date', 'ri_scheduled_death_date']] = [date_of_outcome, pd.NaT]

    def impose_symptoms_for_uncomplicated_disease(self, person_id, disease_type, duration_in_days):
        """
        Imposes the clinical symptoms to uncomplicated Alri. These symptoms are not set to auto-resolve
        """
        m = self.module
        models = m.models
        symptoms = sorted(models.symptoms_for_disease(disease_type=disease_type))
        m.sim.modules['SymptomManager'].change_symptom(
            person_id=person_id,
            symptom_string=symptoms,
            date_of_onset=m.sim.date,
            duration_in_days=duration_in_days,
            add_or_remove='+',
            disease_module=m,
        )

    def impose_complications(self, person_id, duration_in_days):
        """Choose a set of complications for this person and onset these all instantaneously."""

        df = self.sim.population.props
        m = self.module
        models = m.models
        person = df.loc[person_id]

        # Determine complications
        complications_that_onset = models.get_complications_that_onset(
            disease_type=person['ri_disease_type'],
            primary_path_is_bacterial=person['ri_primary_pathogen'] in self.module.pathogens['bacterial'],
            has_secondary_bacterial_inf=pd.notnull(person.ri_secondary_bacterial_pathogen)
        )
        oxygen_saturation = models.get_oxygen_saturation(complication_set=complications_that_onset)

        # Update dataframe
        df.loc[person_id, [f"ri_complication_{complication}" for complication in complications_that_onset]] = True
        df.loc[person_id, 'ri_SpO2_level'] = oxygen_saturation

        # Onset symptoms
        # pick_days_following_onset = m.rng.randint(0, (date_of_outcome - m.sim.date).days)
        # date_onset_symptoms = m.sim.date + DateOffset(days=pick_days_following_onset)
        for complication in sorted(complications_that_onset):
            m.impose_symptoms_for_complication(person_id=person_id,
                                               complication=complication,
                                               oxygen_saturation=oxygen_saturation,
                                               duration_in_days=duration_in_days
                                               )


class AlriNaturalRecoveryEvent(Event, IndividualScopeEventMixin):
    """This is the Natural Recovery event. It is scheduled by the AlriIncidentCase Event for someone who will recover
    from the infection even if no care received. It calls the 'end_infection' function."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]

        # The event should not run if the person is not currently alive
        if not person.is_alive:
            return

        # Check if person should really recover:
        if (
            person.ri_current_infection_status and
            (person.ri_scheduled_recovery_date == self.sim.date) and
            pd.isnull(person.ri_scheduled_death_date)
        ):
            # Log the recovery
            self.module.logging_event.new_recovered_case(
                age=person.age_years,
                pathogen=person.ri_primary_pathogen
            )

            # Do the episode:
            self.module.end_episode(person_id=person_id)


class AlriCureEvent(Event, IndividualScopeEventMixin):
    """This is the cure event. It is scheduled by an HSI treatment event. It enacts the actual "cure" of the person
    that is caused (after some delay) by the treatment administered."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]

        # The event should not run if the person is not currently alive
        if not person.is_alive:
            return

        # Check if person should really be cured:
        if person.ri_current_infection_status:
            # Log the cure:
            pathogen = person.ri_primary_pathogen
            self.module.logging_event.new_cured_case(
                age=person.age_years,
                pathogen=pathogen
            )

            # End the episode:
            self.module.end_episode(person_id=person_id)


class AlriDeathEvent(Event, IndividualScopeEventMixin):
    """This Event is for the death of someone that is caused by the infection with a pathogen that causes Alri."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        person = df.loc[person_id]

        # The event should not run if the person is not currently alive
        if not person.is_alive:
            return

        # Check if person should really die of Alri:
        if (
            person.ri_current_infection_status and
            (person.ri_scheduled_death_date == self.sim.date) and
            pd.isnull(person.ri_scheduled_recovery_date)
        ):
            # Do the death:
            pathogen = person.ri_primary_pathogen
            self.module.sim.modules['Demography'].do_death(
                individual_id=person_id,
                cause='ALRI_' + pathogen,
                originating_module=self.module
            )

            # Log the death in the Alri logging system
            self.module.logging_event.new_death(
                age=person.age_years,
                pathogen=pathogen
            )


# ---------------------------------------------------------------------------------------------------------
# ==================================== HEALTH SYSTEM INTERACTION EVENTS ===================================
# ---------------------------------------------------------------------------------------------------------

class HSI_Alri_Treatment(HSI_Event, IndividualScopeEventMixin):
    """HSI event for treating uncomplicated pneumonia. This event runs for every presentation and represents all the
    interactions with the healthcare system at all the levels."""

    def __init__(self, module: Module, person_id: int, facility_level: str = "0", inpatient: bool = False,
                 is_followup: bool = False):
        super().__init__(module, person_id=person_id)
        self._treatment_id_stub = 'Alri_Pneumonia_Treatment'
        self._facility_levels = ("0", "1a", "1b", "2")  # Health facility levels at which care may be provided
        assert facility_level in self._facility_levels

        self.is_followup = is_followup  # (if True, then HSI has no effect and is not rescheduled if never ran).

        if not inpatient:
            self._as_out_patient(facility_level)
        else:
            self._as_in_patient(facility_level)

        self._age_exact_years = self.sim.population.props.at[person_id, 'age_exact_years']

    def _as_out_patient(self, facility_level):
        """Cast this HSI as an out-patient appointment."""
        self.TREATMENT_ID = f'{self._treatment_id_stub}_Outpatient{"_Followup" if self.is_followup else ""}'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({
            ('ConWithDCSA' if facility_level == '0' else 'Under5OPD'): 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level

    def _as_in_patient(self, facility_level):
        """Cast this HSI as an in-patient appointment."""
        self.TREATMENT_ID = f'{self._treatment_id_stub}_Inpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.ACCEPTED_FACILITY_LEVEL = facility_level
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 3})
        assert not self.is_followup, 'A Follow-up appointment cannot be an in-patient appointment.'

    def _refer_to_next_level_up(self):
        """Schedule a copy of this event to occur again today at the next level-up (if there is a next level-up)."""

        def _next_in_sequence(seq: tuple, x: str):
            """Return next value in a sequence of strings, or None if at end of sequence."""
            if x == seq[-1]:
                return None

            for i, _x in enumerate(seq):
                if _x == x:
                    return seq[i + 1]

        _next_level_up = _next_in_sequence(self._facility_levels, self.ACCEPTED_FACILITY_LEVEL)

        if _next_level_up is not None:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Alri_Treatment(
                    module=self.module,
                    person_id=self.target,
                    inpatient=self._is_as_in_patient,
                    facility_level=_next_level_up
                ),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=1),
                priority=0)

    def _refer_to_become_inpatient(self):
        """Schedule a copy of this event to occur today at this level as an in-patient appointment, at the higher level
        of the current level and level '1b'."""

        the_higher_level_between_this_level_and_1b = '1b' if self.ACCEPTED_FACILITY_LEVEL in ('0', '1a', '1b') else '2'

        self.sim.modules['HealthSystem'].schedule_hsi_event(
            HSI_Alri_Treatment(
                module=self.module,
                person_id=self.target,
                inpatient=True,
                facility_level=the_higher_level_between_this_level_and_1b
            ),
            topen=self.sim.date,
            tclose=self.sim.date + pd.DateOffset(days=1),
            priority=0)

    def _schedule_follow_up_at_same_facility_as_outpatient(self):
        """Schedule a copy of this event to occur in 5 days time as a 'follow-up' appointment at this level as an
        out-patient."""

        self.sim.modules['HealthSystem'].schedule_hsi_event(
            HSI_Alri_Treatment(
                module=self.module,
                person_id=self.target,
                inpatient=False,
                facility_level=self.ACCEPTED_FACILITY_LEVEL,
                is_followup=True
            ),
            topen=self.sim.date + pd.DateOffset(days=5),
            tclose=None,
            priority=0)

    @property
    def _is_as_in_patient(self):
        """True if this HSI_Event is cast as an in-patient appointment"""
        return self.TREATMENT_ID.endswith('Inpatient')

    def _get_cons(self, _item_str: str) -> bool:
        """True if all of a group of consumables (identified by a string) is available, or if no group is
        identified."""
        if _item_str is not None:
            return self.get_consumables(
                item_codes={
                    k: v(self._age_exact_years) if isinstance(v, types.LambdaType) else v
                    for k, v in self.module.consumables_used_in_hsi[_item_str].items()
                })
        else:
            return True

    def _get_any_cons(self, _item_str: str) -> bool:
        """True if any of a group of consumables (identified by a string) is available, or if no group is
        identified."""
        if _item_str is not None:
            return any(self.get_consumables(
                item_codes={
                    k: v(self._age_exact_years) if isinstance(v, types.LambdaType) else v
                    for k, v in self.module.consumables_used_in_hsi[_item_str].items()
                },
                return_individual_results=True
            ).values())
        else:
            return True

    def _assess_and_treat(self, age_exact_years, oxygen_saturation, symptoms):
        """This routine is called when in every HSI. It classifies the disease of the child and commissions treatment
        accordingly."""

        classification_for_treatment_decision = self._get_disease_classification(
            age_exact_years=age_exact_years,
            symptoms=symptoms,
            oxygen_saturation=oxygen_saturation,
            facility_level=self.ACCEPTED_FACILITY_LEVEL,
            use_oximeter=self.get_consumables(item_codes=self.module.consumables_used_in_hsi['Pulse_oximetry'])
        )

        self._provide_bronchodilator_if_wheeze(
            facility_level=self.ACCEPTED_FACILITY_LEVEL,
            symptoms=symptoms
        )

        self._do_action_given_classification(
            classification_for_treatment_decision=classification_for_treatment_decision,
            has_staph_aureus=self._has_staph_aureus(),
            facility_level=self.ACCEPTED_FACILITY_LEVEL
        )

    def _has_staph_aureus(self):
        """Returns True if the person has Staph. aureus as either primary or secondary infection"""
        person_id = self.target
        infections = self.sim.population.props.loc[
            person_id, ['ri_primary_pathogen', 'ri_secondary_bacterial_pathogen']
        ].to_list()
        return 'Staph_aureus' in infections

    def _get_imci_classification_based_on_symptoms(self, child_is_younger_than_2_months: bool, symptoms: list) -> str:
        """Based on age and symptoms, classify WHO-pneumonia severity. This is regarded as the *TRUE* classification
         based on symptoms. It will return one of: {
             'serious_bacterial_infection',
             'fast_breathing_pneumonia',
             'danger_signs_pneumonia',
             'chest_indrawing_pneumonia,
             'cough_or_cold'
        }."""
        return self.module.get_imci_classification_based_on_symptoms(
            child_is_younger_than_2_months=child_is_younger_than_2_months, symptoms=symptoms)

    @staticmethod
    def _get_imci_classification_by_SpO2_measure(child_is_younger_than_2_months: bool,
                                                 oxygen_saturation: bool) -> str:
        """Return classification based on age and oxygen_saturation. It will return one of: {
             'serious_bacterial_infection',     <-- implies needs oxygen
             'danger_signs_pneumonia',          <-- implies needs oxygen
             ''                                 <-- implies does not need oxygen
        }."""

        if oxygen_saturation == '<90%':
            if child_is_younger_than_2_months:
                return 'serious_bacterial_infection'
            else:
                return 'danger_signs_pneumonia'
        else:
            return ''

    def _get_classification_given_by_health_worker(self,
                                                   imci_classification_based_on_symptoms: str,
                                                   facility_level: str) -> str:
        """Determine the classification of the disease that would be given by the health worker at a particular
        facility_level, allowing for the probability of incorrect classification. It will return one of: {
             <the argument provided for `imci_symptom_based_classification`>
             'cough_or_cold',
             'fast_breathing_pneumonia',
             'chest_indrawing_pneumonia'

        }."""

        rand = self.module.rng.random_sample
        rand_choice = self.module.rng.choice
        p = self.module.parameters
        # todo- @ines --- please check the below. I found a few things that seemed mislabelled, so have gone ahead to
        #  correct them following the dominant pattern.

        if facility_level == '0':
            if imci_classification_based_on_symptoms == 'fast_breathing_pneumonia':
                if rand() < p['sensitivity_of_classification_of_fast_breathing_pneumonia_facility_level0']:
                    return imci_classification_based_on_symptoms
                else:
                    return 'cough_or_cold'

            elif imci_classification_based_on_symptoms in ('chest_indrawing_pneumonia', 'danger_signs_pneumonia'):
                # todo - @Ines - should the line above include 'serious_bacterial_infection'...?
                if rand() < p['sensitivity_of_classification_of_danger_signs_pneumonia_facility_level0']:
                    return imci_classification_based_on_symptoms
                else:
                    return rand_choice(
                        ['fast_breathing_pneumonia', 'cough_or_cold'],
                        p=[
                            p['prob_iCCM_severe_pneumonia_treated_as_fast_breathing_pneumonia'],
                            1.0 - p['prob_iCCM_severe_pneumonia_treated_as_fast_breathing_pneumonia']
                        ]
                    )
            else:
                return 'cough_or_cold'

        elif facility_level in ('1a', '1b'):
            if imci_classification_based_on_symptoms in ('fast_breathing_pneumonia', 'chest_indrawing_pneumonia'):
                if rand() < p['sensitivity_of_classification_of_non_severe_pneumonia_facility_level1']:
                    return imci_classification_based_on_symptoms
                else:
                    return 'cough_or_cold'

            elif imci_classification_based_on_symptoms == 'danger_signs_pneumonia':
                if rand() < p['sensitivity_of_classification_of_severe_pneumonia_facility_level1']:
                    return imci_classification_based_on_symptoms
                else:
                    return rand_choice(
                        ['chest_indrawing_pneumonia', 'cough_or_cold'],
                        p=[
                            p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia'],
                            1.0 - p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia']
                        ]
                    )

            else:
                # (Perfect diagnosis accuracy for 'cough_or_cold' & 'serious_bacterial_infection')
                return imci_classification_based_on_symptoms

        else:  # facility_level 2 or above
            if imci_classification_based_on_symptoms in ('fast_breathing_pneumonia', 'chest_indrawing_pneumonia'):
                if rand() < p['sensitivity_of_classification_of_non_severe_pneumonia_facility_level2']:
                    return imci_classification_based_on_symptoms
                else:
                    return 'cough_or_cold'

            elif imci_classification_based_on_symptoms == 'danger_signs_pneumonia':

                if rand() < p['sensitivity_of_classification_of_severe_pneumonia_facility_level2']:
                    return imci_classification_based_on_symptoms
                else:
                    return rand_choice(
                        ['chest_indrawing_pneumonia', 'cough_or_cold'],
                        p=[
                            p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia'],
                            1.0 - p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia']
                        ]
                    )

            else:
                # (Perfect diagnosis accuracy for 'cough_or_cold' & 'serious_bacterial_infection')
                return imci_classification_based_on_symptoms

    def _get_disease_classification(self, age_exact_years, symptoms, oxygen_saturation, facility_level, use_oximeter
                                    ) -> str:
        """Returns the classification of disease, which may be based on the results of the pulse oximetry (if available)
         or the health worker's own classification. It will be one of: {
                 'serious_bacterial_infection',     (symptoms-based assessment / spO2 assessment implies needs oxygen)
                 'danger_signs_pneumonia',          (symptoms-based assessment / spO2 assessment implies need oxygen)
                 'fast_breathing_pneumonia',        (symptoms-based assessment)
                 'chest_indrawing_pneumonia,        (symptoms-based assessment)
                 'cough_or_cold'                    (symptoms-based assessment)
         }."""

        child_is_younger_than_2_months = age_exact_years < (2.0 / 12.0)

        imci_classification_based_on_symptoms = self._get_imci_classification_based_on_symptoms(
            child_is_younger_than_2_months=child_is_younger_than_2_months,
            symptoms=symptoms)

        imci_classification_by_SpO2_measure = self._get_imci_classification_by_SpO2_measure(
            child_is_younger_than_2_months=child_is_younger_than_2_months,
            oxygen_saturation=oxygen_saturation)

        hw_assigned_classification = self._get_classification_given_by_health_worker(
            imci_classification_based_on_symptoms=imci_classification_based_on_symptoms,
            facility_level=facility_level)

        _classification = imci_classification_by_SpO2_measure \
            if use_oximeter and (imci_classification_by_SpO2_measure != '') else hw_assigned_classification

        logger.info(
            key='classification',
            data={'facility_level': facility_level,
                  'symptom_classification': imci_classification_based_on_symptoms,
                  'pulse_ox_classification': imci_classification_by_SpO2_measure,
                  'hw_classification': hw_assigned_classification,
                  'final_classification': _classification}
        )

        return _classification

    def _do_action_given_classification(self, classification_for_treatment_decision, has_staph_aureus, facility_level):
        """Do the actions that are required given a particular classification"""

        def _try_treatment(antibiotic_indicated: str, oxygen_indicated: bool) -> None:
            """Try to provide a `treatment_indicated` and refer to next level if the consumables are not available."""

            antibiotic_consumables_available = self._get_any_cons('Amoxicillin_tablet_or_suspension_5days') \
                if antibiotic_indicated == '5day_oral_amoxicillin' \
                else (self._get_any_cons('Amoxicillin_tablet_or_suspension_3days') if
                      antibiotic_indicated == '3day_oral_amoxicillin'
                      else (any([self._get_cons('Ampicillin_gentamicin_therapy_for_severe_pneumonia'),
                                 self._get_cons('Benzylpenicillin_gentamicin_therapy_for_severe_pneumonia')]))
                      )

            oxygen_available = self._get_cons('Oxygen_Therapy')
            oxygen_indicated_and_available_or_oxygen_not_indicated = (oxygen_available and oxygen_indicated) or \
                                                                     (not oxygen_indicated)

            if antibiotic_consumables_available and oxygen_indicated_and_available_or_oxygen_not_indicated:
                self.module.do_effects_of_treatment(
                    person_id=self.target,
                    antibiotic_provided=antibiotic_indicated,
                    oxygen_provided=(oxygen_available and oxygen_indicated)
                )
                self._schedule_follow_up_at_same_facility_as_outpatient()
            else:
                self._refer_to_next_level_up()

        def _provide_consumable_and_refer(cons: str) -> None:
            """Provide a consumable (ignoring availability) and refer patient to next level up."""
            if cons is not None:
                _ = self._get_cons(cons)
            self._refer_to_next_level_up()

        def do_if_fast_breathing_pneumonia(facility_level):
            """What to do if classification is `fast_breathing`."""
            _try_treatment(antibiotic_indicated='3day_oral_amoxicillin', oxygen_indicated=False)

        def do_if_chest_indrawing_pneumonia(facility_level):
            """What to do if classification is `chest_indrawing_pneumonia`."""
            if facility_level == '0':
                _provide_consumable_and_refer('First_dose_oral_amoxicillin_for_referral')
            else:
                _try_treatment(antibiotic_indicated='5day_oral_amoxicillin', oxygen_indicated=False)

        def do_if_danger_signs_pneumonia(facility_level):
            """What to do if classification is `danger_signs_pneumonia"""
            # _ = self._get_cons('Ceftriaxone_therapy_for_severe_pneumonia')  # this is second line antibiotic

            if has_staph_aureus:
                _ = self._get_cons('2nd_line_Antibiotic_therapy_for_severe_staph_pneumonia')

            if facility_level == '0':
                _provide_consumable_and_refer('First_dose_oral_amoxicillin_for_referral')

            else:
                _ = self._get_cons('Inhaled_Brochodilator')

                if not self._is_as_in_patient:
                    _ = self._get_cons('First_dose_IM_antibiotics_for_referral')
                    self._refer_to_become_inpatient()

                else:
                    _try_treatment(antibiotic_indicated='1st_line_IV_antibiotics', oxygen_indicated=True)

        def do_if_serious_bacterial_infection(facility_level):
            """What to do if `serious_bacterial_infection`."""

            _ = self._get_cons('Ceftriaxone_therapy_for_severe_pneumonia')

            if self._is_as_in_patient and facility_level in ('1b', '2'):
                _try_treatment(antibiotic_indicated='1st_line_IV_antibiotics', oxygen_indicated=True)
            else:
                _ = self._get_cons('First_dose_IM_antibiotics_for_referral')
                self._refer_to_become_inpatient()

        def do_if_cough_or_cold(facility_level):
            """What to do if `cough_or_cold`."""
            pass  # Do nothing

        # Do the appropriate action
        {
            'fast_breathing_pneumonia': do_if_fast_breathing_pneumonia,
            'chest_indrawing_pneumonia': do_if_chest_indrawing_pneumonia,
            'danger_signs_pneumonia': do_if_danger_signs_pneumonia,
            'serious_bacterial_infection': do_if_serious_bacterial_infection,
            'cough_or_cold': do_if_cough_or_cold
        }[classification_for_treatment_decision](facility_level=facility_level)

    def _provide_bronchodilator_if_wheeze(self, facility_level, symptoms):
        """Provide bronchodilator if wheeze is among the symptoms"""
        if 'wheeze' in symptoms:
            if facility_level == '1a':
                _ = self._get_cons('Inhaled_Brochodilator')
            else:
                _ = self._get_cons('Brochodilator_and_Steroids')

    def apply(self, person_id, squeeze_factor):
        """Assess and attempt to treat the person."""

        # Do nothing if this is a follow-up appointment:
        if self.is_followup:
            return

        # Do nothing if the person is not currently infected and currently experiencing an episode
        person = self.sim.population.props.loc[person_id]
        if not person.ri_current_infection_status and (
            person.ri_start_of_current_episode <= self.sim.date <= person.ri_end_of_current_episode
        ):
            return

        # Do nothing if the persons does not have indicating symptoms
        symptoms = self.sim.modules['SymptomManager'].has_what(person_id)
        if not {'cough', 'difficult_breathing'}.intersection(symptoms):
            return

        # Do nothing if the person is already on treatment
        if person.ri_on_treatment:
            return

        # If the HSI is at level 0 and is for a child aged less than 2 months, refer to the next level.
        if (self.ACCEPTED_FACILITY_LEVEL == '0') and (person.age_exact_years < 2.0 / 12.0):
            self._refer_to_next_level_up()

        # Attempt treatment:
        self._assess_and_treat(age_exact_years=person.age_exact_years,
                               symptoms=symptoms,
                               oxygen_saturation=person.ri_SpO2_level)

    def never_ran(self):
        """If this event never ran (and is not a follow-up appointment), refer to next level up."""
        if not self.is_followup:
            self._refer_to_next_level_up()


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class AlriLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """This Event logs the number of incident cases that have occurred since the previous logging event.
    Analysis scripts expect that the frequency of this logging event is once per year."""

    def __init__(self, module):
        # This event to occur every year
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

        # initialise trakcers of incident cases, new recoveries, new treatments and deaths due to ALRI
        age_grps = {**{0: "0", 1: "1", 2: "2-4", 3: "2-4", 4: "2-4"}, **{x: "5+" for x in range(5, 100)}}

        self.trackers = dict()
        self.trackers['incident_cases'] = Tracker(age_grps=age_grps, pathogens=self.module.all_pathogens)
        self.trackers['recovered_cases'] = Tracker(age_grps=age_grps, pathogens=self.module.all_pathogens)
        self.trackers['cured_cases'] = Tracker(age_grps=age_grps, pathogens=self.module.all_pathogens)
        self.trackers['deaths'] = Tracker(age_grps=age_grps, pathogens=self.module.all_pathogens)
        self.trackers['seeking_care'] = Tracker()
        self.trackers['treated'] = Tracker()

    def new_case(self, **kwargs):
        self.trackers['incident_cases'].add_one(**kwargs)

    def new_recovered_case(self, **kwargs):
        self.trackers['recovered_cases'].add_one(**kwargs)

    def new_cured_case(self, **kwargs):
        self.trackers['cured_cases'].add_one(**kwargs)

    def new_death(self, **kwargs):
        self.trackers['deaths'].add_one(**kwargs)

    def new_seeking_care(self, **kwargs):
        self.trackers['seeking_care'].add_one(**kwargs)

    def new_treated(self, **kwargs):
        self.trackers['treated'].add_one(**kwargs)

    def apply(self, population):
        """
        Log:
        1) Number of new cases, by age-group and by pathogen since the last logging event
        2) Total number of cases, recovery, treatments and deaths since the last logging event
        """

        # 1) Number of new cases, by age-group and by pathogen, since the last logging event
        logger.info(
            key='incidence_count_by_age_and_pathogen',
            data=self.trackers['incident_cases'].report_current_counts(),
            description='Pathogens incident case counts since last logging event'
        )

        # 2) Total number of in all the trackers since the last logging event
        logger.info(
            key='event_counts',
            data={k: v.report_current_total() for k, v in self.trackers.items()},
            description='Counts of trackers since last logging event'
        )

        # 3) Reset the trackers
        for tracker in self.trackers.values():
            tracker.reset()


class Tracker:
    """Helper class to be a counter for number of events occurring by age-group and by pathogen."""

    def __init__(self, age_grps: dict = {}, pathogens: list = []):
        """Create and initalise tracker"""

        # Check and store parameters
        self.pathogens = pathogens
        self.age_grps_lookup = age_grps
        self.unique_age_grps = sorted(set(self.age_grps_lookup.values()))

        # Initialise Tracker
        self.tracker = None
        self.reset()

    def reset(self):
        """Produce a dict of the form: { <Age-Grp>: {<Pathogen>: <Count>} } if age-groups and pathogens are specified;
        otherwise the tracker is an integer."""
        if (len(self.unique_age_grps) == 0) and (len(self.pathogens) == 0):
            self.tracker = 0
        else:
            self.tracker = {
                age: dict(zip(self.pathogens, [0] * len(self.pathogens))) for age in self.unique_age_grps
            }

    def add_one(self, age=None, pathogen=None):
        """Increment counter by one for a specific age and pathogen"""
        if (age is None) and (pathogen is None):
            # increment by one, not specifically by age and pathogen:
            self.tracker += 1

        else:
            # increment by one, specifically by age and pathogen:
            assert age in self.age_grps_lookup, 'Age not recognised'
            assert pathogen in self.pathogens, 'Pathogen not recognised'

            age_grp = self.age_grps_lookup[age]
            self.tracker[age_grp][pathogen] += 1

    def report_current_counts(self):
        return self.tracker

    def report_current_total(self):
        if not isinstance(self.tracker, int):
            total = 0
            for _a in self.tracker.keys():
                total += sum(self.tracker[_a].values())
            return total
        else:
            return self.tracker


# ---------------------------------------------------------------------------------------------------------
#   DEBUGGING / TESTING EVENTS
# ---------------------------------------------------------------------------------------------------------

class AlriCheckPropertiesEvent(RegularEvent, PopulationScopeEventMixin):
    """This event runs daily and checks properties are in the right configuration. Only use whilst debugging!"""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=1))

    def apply(self, population):
        self.module.check_properties()


class AlriIndividualLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """This Event logs the daily occurrence to a single individual child."""

    def __init__(self, module):
        # This logging event to occur every day
        super().__init__(module, frequency=DateOffset(days=1))

        self.person_id = self.module.log_individual
        assert self.person_id in module.sim.population.props.index, 'The person identified to be logged does not exist.'

    def apply(self, population):
        """Log all properties for this module"""
        if self.person_id is not None:
            df = self.sim.population.props
            logger.info(
                key='log_individual',
                data=df.loc[self.person_id, self.module.PROPERTIES.keys()].to_dict(),
                description='Properties for one person (the first under-five-year-old in the dataframe), each day.'
            )


class AlriPropertiesOfOtherModules(Module):
    """For the purpose of the testing, this module generates the properties upon which the Alri module relies"""

    INIT_DEPENDENCIES = {'Demography'}

    # Though this module provides some properties from NewbornOutcomes we do not list
    # NewbornOutcomes in the ALTERNATIVE_TO set to allow using in conjunction with
    # SimplifiedBirths which can also be used as an alternative to NewbornOutcomes
    ALTERNATIVE_TO = {'Hiv', 'Epi', 'Wasting'}

    PROPERTIES = {
        'hv_inf': Property(Types.BOOL, 'temporary property'),
        'hv_art': Property(Types.CATEGORICAL, 'temporary property',
                           categories=["not", "on_VL_suppressed", "on_not_VL_suppressed"]),
        'nb_low_birth_weight_status': Property(Types.CATEGORICAL, 'temporary property',
                                               categories=['extremely_low_birth_weight', 'very_low_birth_weight',
                                                           'low_birth_weight', 'normal_birth_weight']),

        'nb_breastfeeding_status': Property(Types.CATEGORICAL, 'temporary property',
                                            categories=['none', 'non_exclusive', 'exclusive']),
        'va_pneumo_all_doses': Property(Types.BOOL, 'temporary property'),
        'va_hib_all_doses': Property(Types.BOOL, 'temporary property'),
        'va_measles_all_doses': Property(Types.BOOL, 'temporary property'),
        'un_clinical_acute_malnutrition': Property(Types.CATEGORICAL, 'temporary property',
                                                   categories=['MAM', 'SAM', 'well']),
    }

    def __init__(self, name=None):
        super().__init__(name)

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        df = population.props
        df.loc[df.is_alive, 'hv_inf'] = False
        df.loc[df.is_alive, 'hv_art'] = 'not'
        df.loc[df.is_alive, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.loc[df.is_alive, 'nb_breastfeeding_status'] = 'non_exclusive'
        df.loc[df.is_alive, 'va_pneumo_all_doses'] = False
        df.loc[df.is_alive, 'va_hib_all_doses'] = False
        df.loc[df.is_alive, 'va_measles_all_doses'] = False
        df.loc[df.is_alive, 'un_clinical_acute_malnutrition'] = 'well'

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        df = self.sim.population.props
        df.at[child, 'hv_inf'] = False
        df.at[child, 'hv_art'] = 'not'
        df.at[child, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.at[child, 'nb_breastfeeding_status'] = 'non_exclusive'
        df.at[child, 'va_pneumo_all_doses'] = False
        df.at[child, 'va_hib_all_doses'] = False
        df.at[child, 'va_measles_all_doses'] = False
        df.at[child, 'un_clinical_acute_malnutrition'] = 'well'


class AlriIncidentCase_Lethal_Severe_Pneumonia(AlriIncidentCase):
    """This Event can be used for testing and is a drop-in replacement of `AlriIncidentCase`. It always produces an
    infection that will be lethal and should be classified as 'danger_signs_pneumonia' if the person is
    older than 2 months, and `serious_bacterial_infection` otherwise."""

    def __init__(self, module, person_id, pathogen):
        super().__init__(module, person_id=person_id, pathogen=pathogen)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters

        self.module.logging_event.new_case(age=df.at[person_id, 'age_years'], pathogen=self.pathogen)

        disease_type = 'pneumonia'
        duration_in_days_of_alri = 10
        date_of_outcome = self.module.sim.date + DateOffset(days=duration_in_days_of_alri)
        episode_end = date_of_outcome + DateOffset(days=p['days_between_treatment_and_cure'])

        _chars = {
            'ri_current_infection_status': True,
            'ri_primary_pathogen': self.pathogen,
            'ri_secondary_bacterial_pathogen': np.nan,
            'ri_disease_type': disease_type,
            'ri_on_treatment': False,
            'ri_start_of_current_episode': self.sim.date,
            'ri_scheduled_recovery_date': pd.NaT,
            'ri_scheduled_death_date': date_of_outcome,
            'ri_end_of_current_episode': episode_end,
            'ri_complication_hypoxaemia': True,
            'ri_SpO2_level': "<90%"
        }
        df.loc[person_id, _chars.keys()] = _chars.values()

        # make probability of severe symptoms high
        params = self.module.parameters
        severe_symptoms = {
            'chest_indrawing', 'danger_signs'
        }
        for p in params:
            if any([p.startswith(f"prob_{symptom}") for symptom in severe_symptoms]):
                if isinstance(params[p], float):
                    params[p] = 1.0
                else:
                    params[p] = [1.0] * len(params[p])

        self.impose_symptoms_for_uncomplicated_disease(person_id=person_id, disease_type=disease_type,
                                                       duration_in_days=duration_in_days_of_alri)

        self.module.impose_symptoms_for_complication(
            person_id=person_id, complication='hypoxaemia', oxygen_saturation="<90%", duration_in_days=5)

        self.sim.schedule_event(AlriDeathEvent(self.module, person_id), date_of_outcome)

        age_less_than_2_months = df.at[person_id, 'age_exact_years'] < (2.0 / 12.0)
        correct_classification = 'serious_bacterial_infection' if age_less_than_2_months else 'danger_signs_pneumonia'
        assert correct_classification == \
               self.module.get_imci_classification_based_on_symptoms(
                   child_is_younger_than_2_months=age_less_than_2_months,
                   symptoms=self.sim.modules['SymptomManager'].has_what(person_id))


class AlriIncidentCase_NonLethal_Fast_Breathing_Pneumonia(AlriIncidentCase):
    """This Event can be used for testing and is a drop-in replacement of `AlriIncidentCase`. It always produces an
    infection that will be non-lethal and should be classified as a fast_breathing_pneumonia."""

    def __init__(self, module, person_id, pathogen):
        super().__init__(module, person_id=person_id, pathogen=pathogen)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters

        self.module.logging_event.new_case(age=df.at[person_id, 'age_years'], pathogen=self.pathogen)

        disease_type = 'pneumonia'
        duration_in_days_of_alri = 10
        date_of_outcome = self.module.sim.date + DateOffset(days=duration_in_days_of_alri)
        episode_end = date_of_outcome + DateOffset(days=p['days_between_treatment_and_cure'])

        _chars = {
            'ri_current_infection_status': True,
            'ri_primary_pathogen': self.pathogen,
            'ri_secondary_bacterial_pathogen': np.nan,
            'ri_disease_type': disease_type,
            'ri_on_treatment': False,
            'ri_start_of_current_episode': self.sim.date,
            'ri_scheduled_recovery_date': date_of_outcome,
            'ri_scheduled_death_date': pd.NaT,
            'ri_end_of_current_episode': episode_end,
            'ri_complication_hypoxaemia': False,
            'ri_SpO2_level': ">=93%"
        }
        df.loc[person_id, _chars.keys()] = _chars.values()

        # make probability of mild symptoms very high, and severe symptoms low
        params = self.sim.modules['Alri'].parameters
        fast_breathing_pneumonia_symptoms = {'cough', 'tachypnoea'}
        other_symptoms = {'difficult_breathing', 'chest_indrawing', 'danger_signs'}

        for p in params:
            if any([p.startswith(f"prob_{symptom}") for symptom in fast_breathing_pneumonia_symptoms]):
                if isinstance(params[p], float):
                    params[p] = 1.0
                else:
                    params[p] = [1.0] * len(params[p])
            if any([p.startswith(f"prob_{symptom}") for symptom in other_symptoms]):
                params[p] = 0.0

        self.impose_symptoms_for_uncomplicated_disease(person_id=person_id, disease_type=disease_type,
                                                       duration_in_days=duration_in_days_of_alri)

        self.sim.schedule_event(AlriNaturalRecoveryEvent(self.module, person_id), date_of_outcome)

        assert 'fast_breathing_pneumonia' == \
               self.module.get_imci_classification_based_on_symptoms(
                   child_is_younger_than_2_months=False, symptoms=self.sim.modules['SymptomManager'].has_what(person_id)
               )
