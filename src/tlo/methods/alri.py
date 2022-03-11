"""
Childhood Acute Lower Respiratory Infection Module

Overview
--------
Individuals are exposed to the risk of infection by a pathogen (and potentially also with a potential bacterial
co-infection infection) that can cause one of two types of acute lower respiratory infection (Alri) modelled in TLO.
The disease is manifested as either pneumonia or other alri including bronchiolitis.

During an episode (prior to recovery - either naturally or cured with treatment), symptom are manifest and there may be
complications (e.g. local pulmonary complication: pleural effusion, empyema, lung abscess, pneumothorax; and/or
systemic complications: sepsis; and/or complications regarding oxygen exchange: hypoxaemia.
The complications onset at the time of disease onset

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

PR4: Elaborate the HSI system to the extent needed.

Issue #438

"""

from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.dxmanager import DxTest
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
        'rr_ALRI_crowding':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for children living in crowed households (>7 pph)'
                      ),  # TODO: change to wealth?
        'rr_ALRI_underweight':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for underweight children'
                      ),  # TODO: change to SAM/MAM?

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
        'sensitivity_of_assessment_and_classification_of_IMCI_pneumonia':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 2 '
                      'of first line antibiotic treatment for severe pneumonia'
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

        # < --- other properties of the form 'ri_complication_{complication-name}' are added later -->

        # ---- Internal variables to schedule onset and deaths due to Alri ----
        'ri_start_of_current_episode': Property(Types.DATE,
                                                'date of onset of current Alri event (pd.NaT is not infected)'),
        'ri_scheduled_recovery_date': Property(Types.DATE,
                                               '(scheduled) date of recovery from current Alri event (pd.NaT is not '
                                               'infected or episode is scheduled to end in death)'),
        'ri_scheduled_death_date': Property(Types.DATE,
                                            '(scheduled) date of death caused by current Alri event (pd.NaT is not '
                                            'infected or episode will not cause death)'),
        'ri_end_of_current_episode':
            Property(Types.DATE, 'date on which the last episode of Alri is resolved, (including '
                                 'allowing for the possibility that a cure is scheduled following onset). '
                                 'This is used to determine when a new episode can begin. '
                                 'This stops successive episodes interfering with one another.'),
        'ri_ALRI_tx_start_date': Property(Types.DATE,
                                          'start date of Alri treatment for current episode (pd.NaT is not infected or'
                                          ' treatment has not begun)'),
        'ri_symptom_based_pneumonia_classification':
            Property(Types.CATEGORICAL, 'ICCM/IMCI symptoms based classification',
                     categories=['cough_or_cold', 'fast_breathing_pneumonia',
                                 'chest_indrawing_pneumonia', 'danger_signs_pneumonia',
                                 'serious_bacterial_infection', 'not_handled_at_facility_0']
                     ),
        'ri_tx_oxygen_therapy':
            Property(Types.BOOL, 'is the person on treatment receiving oxygen therapy for the current illness'
                     ),
        'ri_tx_antibiotic_therapy':
            Property(Types.CATEGORICAL,
                     'antibiotic therapy for the current illness',
                     categories=['3day_oral_amoxicillin', '5day_oral_amoxicillin',
                                 '1st_line_IV_antibiotics', '2nd_line_IV_antibiotics']
                     ),

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

        # Pointer to store the logging event used by this module
        self.logging_df_per_episode = None

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
        df.loc[df.is_alive, [
            f"ri_complication_{complication}" for complication in self.complications]
        ] = False
        df.loc[df.is_alive, 'ri_SpO2_level'] = ">=93%"

        # ---- Internal values ----
        df.loc[df.is_alive, 'ri_start_of_current_episode'] = pd.NaT
        df.loc[df.is_alive, 'ri_scheduled_recovery_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_scheduled_death_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_end_of_current_episode'] = pd.NaT
        df.loc[df.is_alive, 'ri_on_treatment'] = False
        df.loc[df.is_alive, 'ri_ALRI_tx_start_date'] = pd.NaT

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
        sim.schedule_event(self.logging_event, sim.date + DateOffset(years=1))

        # self.logging_df_per_episode = AlriAllCasesLoggingEvent(self)
        # sim.schedule_event(self.logging_df_per_episode, sim.date + DateOffset(years=1))

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
            days=(p['max_alri_duration_in_days_without_treatment'] +
                  p['days_between_treatment_and_cure']))
        # 14 days max duration of an episode (natural history) + 14 days to allow treatment

        # Look-up and store the consumables that are required for each HSI
        self.look_up_consumables()

        # register the dx_tests
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(

            # correct classification and treatment plan for iCCM fast-breathing pneumonia
            assess_and_classify_iCCM_fast_breathing_pneumonia=DxTest(
                property='ri_symptom_based_pneumonia_classification',
                sensitivity=p['sensitivity_of_classification_of_fast_breathing_pneumonia_facility_level0'],
                target_categories=['fast_breathing_pneumonia']),

            # correct classification and referral for iCCM danger signs pneumonia (includes chest-indrawing)
            assess_and_classify_iCCM_danger_signs_pneumonia=DxTest(
                property='ri_symptom_based_pneumonia_classification',
                sensitivity=p['sensitivity_of_classification_of_danger_signs_pneumonia_facility_level0'],
                target_categories=['chest_indrawing_pneumonia', 'danger_signs_pneumonia']),

            # correct classification and treatment plan for IMCI non-severe pneumonia at facility level 1a/1b
            # (includes fast-breathing and chest-indrawing)
            assess_and_classify_IMCI_non_severe_pneumonia_level1=DxTest(
                property='ri_symptom_based_pneumonia_classification',
                sensitivity=p['sensitivity_of_classification_of_non_severe_pneumonia_facility_level1'],
                target_categories=['fast_breathing_pneumonia', 'chest_indrawing_pneumonia']),

            # correct classification and referral for IMCI severe pneumonia at facility level 1a/1b
            assess_and_classify_IMCI_severe_pneumonia_level1=DxTest(
                property='ri_symptom_based_pneumonia_classification',
                sensitivity=p['sensitivity_of_classification_of_severe_pneumonia_facility_level1'],
                target_categories=['danger_signs_pneumonia']),

            # correct classification and treatment plan for IMCI non-severe pneumonia at facility level 2
            assess_and_classify_IMCI_non_severe_pneumonia_level2=DxTest(
                property='ri_symptom_based_pneumonia_classification',
                sensitivity=p['sensitivity_of_classification_of_non_severe_pneumonia_facility_level2'],
                target_categories=['fast_breathing_pneumonia', 'chest_indrawing_pneumonia']),

            # correct classification and referral for IMCI severe pneumonia at facility level 2
            assess_and_classify_IMCI_severe_pneumonia_level2=DxTest(
                property='ri_symptom_based_pneumonia_classification',
                sensitivity=p['sensitivity_of_classification_of_severe_pneumonia_facility_level2'],
                target_categories=['danger_signs_pneumonia']),
        )

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

        get_item_codes_from_package = self.sim.modules['HealthSystem'].get_item_codes_from_package_name
        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name

        # Treatment of non-severe pneumonia in the health facility (TLO package)
        self.consumables_used_in_hsi['IMCI_Treatment_non_severe_pneumonia'] = \
            get_item_codes_from_package(package='Pneumonia treatment (children)')
        # Treatment severe pneumonia in hospital (TLO package)
        self.consumables_used_in_hsi['IMCI_Treatment_severe_pneumonia'] = \
            get_item_codes_from_package(package='Treatment of severe pneumonia')

        # ------------- Community (iCCM) -------------
        # Treatment of non-severe pneumonia in the community
        self.consumables_used_in_hsi['iCCM_Antibiotic_Therapy_for_pneumonia'] = \
            [get_item_code(item='Paracetamol, tablet, 100 mg')] + \
            [get_item_code(item='Amoxycillin 250mg_1000_CMST')]

        # Referral process in the community for severe pneumonia
        self.consumables_used_in_hsi['First_dose_antibiotic_for_referral_iCCM'] = \
            [get_item_code(item='Amoxycillin 250mg_1000_CMST')]

        # ------------- Health centres (IMCI) -------------

        # Treatment at health centres for severe cases if no referral
        self.consumables_used_in_hsi['IMCI_Antibiotic_therapy_for_severe_pneumonia'] = \
            [get_item_code(item='Ampicillin injection 500mg, PFR_each_CMST')] + \
            [get_item_code(item='Gentamicin Sulphate 40mg/ml, 2ml_each_CMST')] + \
            [get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST')] + \
            [get_item_code(item='Syringe, needle + swab')]

        # Referral process at health centres for severe cases
        self.consumables_used_in_hsi['First_dose_antibiotic_for_referral_IMCI'] = \
            [get_item_code(item='Ampicillin injection 500mg, PFR_each_CMST')] + \
            [get_item_code(item='Gentamicin Sulphate 40mg/ml, 2ml_each_CMST')] + \
            [get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST')] + \
            [get_item_code(item='Syringe, needle + swab')]

        # Referral process at health centres for HIV exposed/infected (oral amoxicillin)
        self.consumables_used_in_hsi['First_dose_amoxicillin_for_HIV_referral_IMCI'] = \
            [get_item_code(item='Amoxycillin 250mg_1000_CMST')]

        # Antibiotics for non-severe pneumonia - oral amoxicillin for 5 days
        self.consumables_used_in_hsi['Amoxicillin_suspension_or_tablet'] = \
            [get_item_code(item='Amoxycillin 125mg/5ml suspension, PFR_0.025_CMST')] + \
            [get_item_code(item='Amoxycillin 250mg_1000_CMST')]

        # Bronchodilator
        self.consumables_used_in_hsi['Inhaled_Brochodilator'] = \
            [get_item_code(item='Salbutamol sulphate 1mg/ml, 5ml_each_CMST')]
        # if not available inhaled bronchodilator, give oral
        self.consumables_used_in_hsi['Brochodilator_syrup'] = \
            [get_item_code(item='Salbutamol, syrup, 2 mg/5 ml')]
        self.consumables_used_in_hsi['Brochodilator_tablet'] = \
            [get_item_code(item='Salbutamol, tablet, 4 mg')]

        # ------------- Hospital (IMCI) -------------

        # First line of antibiotics for severe pneumonia
        self.consumables_used_in_hsi['1st_line_Antibiotic_Therapy_for_Severe_Pneumonia'] = \
            [get_item_code(item='Benzylpenicillin 3g (5MU), PFR_each_CMST')] + \
            [get_item_code(item='Gentamicin Sulphate 40mg/ml, 2ml_each_CMST')] + \
            [get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST')] + \
            [get_item_code(item='Syringe, needle + swab')]

        # Second line of antibiotics for severe pneumonia
        self.consumables_used_in_hsi['2nd_line_Antibiotic_Therapy_for_Severe_Pneumonia'] = \
            [get_item_code(item='Ceftriaxone 1g, PFR_each_CMST')] + \
            [get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST')] + \
            [get_item_code(item='Syringe, needle + swab')]

        # Second line of antibiotics for severe pneumonia, if Staph is suspected
        self.consumables_used_in_hsi['2nd_line_Antibiotic_Therapy_for_Severe_Staph_Pneumonia'] = \
            [get_item_code(item='cloxacillin 500 mg, powder for injection_50_IDA')] + \
            [get_item_code(item='Gentamicin Sulphate 40mg/ml, 2ml_each_CMST')] + \
            [get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST')] + \
            [get_item_code(item='Syringe, needle + swab')] + \
            [get_item_code(item='Cloxacillin discs 5mcg_50_CMST')]

        # Oxygen for hypoxaemia
        self.consumables_used_in_hsi['Oxygen_Therapy'] = \
            [get_item_code(item='Oxygen, 1000 liters, primarily with oxygen concentrators')] + \
            [get_item_code(item='Nasal prongs')]

        # Pulse oximetry
        self.consumables_used_in_hsi['Pulse_oximetry'] = \
            [get_item_code(item='Oxygen, 1000 liters, primarily with oxygen concentrators')]
        # use oxygen code to fill in consumable availability for pulse oximetry

        # X-ray scan
        self.consumables_used_in_hsi['X_ray_scan'] = \
            [get_item_code(item='X-ray')]

        # Treat wheeze
        self.consumables_used_in_hsi['Brochodilator_and_Steroids'] = \
            [get_item_code(item='Salbutamol sulphate 1mg/ml, 5ml_each_CMST')] + \
            [get_item_code(item='Prednisolone 5mg_100_CMST')]

        # Maintenance of fluids via nasograstric tube
        self.consumables_used_in_hsi['Fluid_Maintenance'] = \
            [get_item_code(item='Tube, nasogastric CH 8_each_CMST')]

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
            'ri_ALRI_tx_start_date': pd.NaT,
            'ri_symptom_based_pneumonia_classification': np.nan,
            'ri_tx_oxygen_therapy': False,
            'ri_tx_antibiotic_therapy': np.nan,
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
        assert df.loc[not_curr_inf, 'ri_ALRI_tx_start_date'].isna().all()

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

        # If person is on treatment, they should have a treatment start date
        assert (df.loc[curr_inf, 'ri_on_treatment'] != df.loc[curr_inf, 'ri_ALRI_tx_start_date'].isna()).all()

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

    @staticmethod
    def imci_classification_based_on_symptoms(age_exact_years, symptoms, facility_level):
        """Based on symptoms presented, classify WHO-pneumonia severity at each facility level."""
        # TODO: get other danger signs in iCCM when issue 429 is resolved (!?!?!?!?!?!)
        # todo iccm_danger_signs = symptoms.append() other symptoms child may have that is considered severe in iCCM

        child_is_younger_than_2_months = age_exact_years < (1.0 / 6.0)

        # for children aged < 2 months
        if child_is_younger_than_2_months:
            if facility_level == '0':
                return 'not_handled_at_facility_0'
            else:
                if ('chest_indrawing' in symptoms) or ('danger_signs' in symptoms):
                    return 'serious_bacterial_infection'
                elif 'tachypnoea' in symptoms:
                    return 'fast_breathing_pneumonia'
                else:
                    return 'cough_or_cold'

        # iCCM and IMCI classifications
        # for children aged >= 2 months
        else:
            if 'danger_signs' in symptoms:
                return 'danger_signs_pneumonia'
            elif 'chest_indrawing' in symptoms:
                return 'chest_indrawing_pneumonia'
            elif 'tachypnoea' in symptoms:
                return 'fast_breathing_pneumonia'
            else:
                return 'cough_or_cold'

    def imci_classification_with_SpO2_measure(self, symptom_classification, oxygen_saturation):
        """
        Based on symptoms and potential use of pulse oximetry for measurement of hypoxaemia,
        and return classification based on symptoms and SpO2 level
        """

        if oxygen_saturation == '<90%':
            return 'danger_signs_pneumonia'
        else:
            return symptom_classification

    def classification_given_by_health_worker(self, symptoms, imci_symptom_based_classification,
                                              hsi_event, facility_level):
        """
        Apply the sensitivities of correct classification and treatment decision - quality of care -

        :return:
        """

        rng = self.rng
        p = self.parameters

        run_quality_care = \
            lambda _dx_test: self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=_dx_test,
                                                                                     hsi_event=hsi_event)
        # Now apply the quality of care
        if facility_level == '0':
            if imci_symptom_based_classification == 'fast_breathing_pneumonia':
                if run_quality_care('assess_and_classify_iCCM_fast_breathing_pneumonia'):
                    return imci_symptom_based_classification
                else:
                    return 'cough_or_cold'
            elif imci_symptom_based_classification in ('chest_indrawing_pneumonia', 'danger_signs_pneumonia'):
                if run_quality_care('assess_and_classify_iCCM_danger_signs_pneumonia'):
                    return imci_symptom_based_classification
                else:
                    return rng.choice(['fast_breathing_pneumonia', 'cough_or_cold'],
                                      p=[p['prob_iCCM_severe_pneumonia_treated_as_fast_breathing_pneumonia'],
                                         1 - p['prob_iCCM_severe_pneumonia_treated_as_fast_breathing_pneumonia']])

            elif imci_symptom_based_classification in ('cough_or_cold', 'not_handled_at_facility_0'):
                return imci_symptom_based_classification
            else:
                raise ValueError(f'symptom based classification {imci_symptom_based_classification}'
                                 f'with symptoms {symptoms}'
                                 f' not recognised at facility level 0')

        if facility_level in ('1a', '1b'):
            if imci_symptom_based_classification in ('fast_breathing_pneumonia', 'chest_indrawing_pneumonia'):
                if run_quality_care('assess_and_classify_IMCI_non_severe_pneumonia_level1'):
                    return imci_symptom_based_classification
                else:
                    return 'cough_or_cold'
            elif imci_symptom_based_classification == 'danger_signs_pneumonia':
                if run_quality_care('assess_and_classify_IMCI_severe_pneumonia_level1'):
                    return imci_symptom_based_classification
                else:
                    return rng.choice(['chest_indrawing_pneumonia', 'cough_or_cold'],  # chest_indrawing_pneumonia reflects non-severe pneumonia, applies for fast_breathing pneumonia
                                      p=[p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia'],
                                         1 - p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia']])
            elif imci_symptom_based_classification in ('cough_or_cold', 'serious_bacterial_infection'):
                return imci_symptom_based_classification  # assume 100% correct diagnosis
            else:
                raise ValueError(f'symptom based classification {imci_symptom_based_classification} '
                                 f'with symptoms {symptoms}'
                                 f' not recognised at facility level 1')

        if facility_level == '2':
            if imci_symptom_based_classification in ('fast_breathing_pneumonia', 'chest_indrawing_pneumonia'):
                if run_quality_care('assess_and_classify_IMCI_non_severe_pneumonia_level2'):
                    return imci_symptom_based_classification
                else:
                    return 'cough_or_cold'
            elif imci_symptom_based_classification == 'danger_signs_pneumonia':
                if run_quality_care('assess_and_classify_IMCI_severe_pneumonia_level2'):
                    return imci_symptom_based_classification
                else:
                    return rng.choice(['chest_indrawing_pneumonia', 'cough_or_cold'],
                                      p=[p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia'],
                                         1 - p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia']])
            elif imci_symptom_based_classification in ('cough_or_cold', 'serious_bacterial_infection'):
                return imci_symptom_based_classification  # assume 100% correct diagnosis
            else:
                raise ValueError(f'symptom based classification {imci_symptom_based_classification}'
                                 f'with symptoms {symptoms}'
                                 f' not recognised at facility level 2')

    def final_classification(self, hw_assigned_classification, oxygen_saturation, oximeter_available):
        """
        Based on symptoms and health workers sensitivity of assessment and classification,
        check for the use of pulse oximetry for measurement of hypoxaemia,
        and return the final classification
        """

        if oximeter_available and (oxygen_saturation == '<90%'):
            return 'danger_signs_pneumonia'
        else:
            return hw_assigned_classification

    def do_action_given_classification(self, person_id, classification, symptoms, oxygen_saturation, hsi_event):
        """Do the actions that are required given a particular classification"""
        # todo @ines/tim - this is the bit to pay attention to: we need to make sure that we know what to do for ANY

        # Create shortcuts:
        df = self.sim.population.props
        schedule_hsi = \
            lambda _event: self.sim.modules['HealthSystem'].schedule_hsi_event(_event, priority=0,
                                                                               topen=self.sim.date, tclose=None)
        get_cons = \
            lambda _item_str: hsi_event.get_consumables(item_codes=self.consumables_used_in_hsi[_item_str])
        get_any_cons = \
            lambda _item_strs: any(hsi_event.get_consumables(item_codes=self.consumables_used_in_hsi[_item_strs],
                                                             return_individual_results=True).values())
        do_treatment = \
            lambda _treatment_str: self.do_effects_of_alri_treatment(person_id=person_id,
                                                                     hsi_event=hsi_event, treatment=[_treatment_str])

        if classification is None:
            raise ValueError(f'for person {person_id} classification {classification} is None, for HSI {hsi_event} '
                             f'at facility {hsi_event.ACCEPTED_FACILITY_LEVEL} with symptoms {symptoms}')

        # Define actions for each classification:
        def do_if_fast_breathing_pneumonia(facility_level):
            if facility_level == '0':
                if get_cons('iCCM_Antibiotic_Therapy_for_pneumonia'):
                    do_treatment('5day_oral_amoxicillin')
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: treatment given for person {person_id} with '
                                      f'fast_breathing_pneumonia {classification} at level (0) {facility_level}  hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                else:
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: no antibiotics available for person {person_id} with '
                                      f'fast_breathing_pneumonia {classification} at level (0) {facility_level}  hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                    schedule_hsi(HSI_IMCI_Pneumonia_Treatment_Outpatient_level_1a(person_id=person_id, module=self))

            elif facility_level == '1a':
                if get_any_cons('Amoxicillin_suspension_or_tablet'):
                    do_treatment('5day_oral_amoxicillin')
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: treatment GIVEN for person {person_id} with '
                                      f'fast_breathing_pneumonia {classification} at level (1a) {facility_level} hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                else:
                    schedule_hsi(HSI_IMCI_Pneumonia_Treatment_Outpatient_level_1b(person_id=person_id, module=self))
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: no antibiotics available for person {person_id} with '
                                      f'fast_breathing_pneumonia {classification} at level (1a) {facility_level}  hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')

            elif facility_level == '1b':
                if get_any_cons('Amoxicillin_suspension_or_tablet'):
                    do_treatment('5day_oral_amoxicillin')
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: treatment GIVEN for person {person_id} with '
                                      f'fast_breathing_pneumonia {classification} at level (1b) {facility_level} hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                else:
                    schedule_hsi(HSI_IMCI_Pneumonia_Treatment_Outpatient_level_2(person_id=person_id, module=self))
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: no antibiotics available for person {person_id} with '
                                      f'fast_breathing_pneumonia {classification} at level (1b) {facility_level}  hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')

            elif facility_level == '2':
                if get_any_cons('Amoxicillin_suspension_or_tablet'):
                    do_treatment('5day_oral_amoxicillin')
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: treatment GIVEN for person {person_id} with '
                                      f'fast_breathing_pneumonia {classification} at level (2) {facility_level} hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                else:
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: no antibiotics available for person {person_id} with '
                                      f'fast_breathing_pneumonia {classification} at level (2) {facility_level}  hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')

            else:
                raise ValueError(f'facility_level{facility_level} not recognised for fast_breathing_pneumonia')

        def do_if_chest_indrawing_pneumonia(facility_level):
            if facility_level == '0':
                get_cons('First_dose_antibiotic_for_referral_iCCM')
                schedule_hsi(HSI_IMCI_Pneumonia_Treatment_Outpatient_level_1a(person_id=person_id, module=self))
                logger.debug(key='message',
                             data=f'ALRI_HSI_Event: NO treatment for person {person_id} with '
                                  f'chest_indrawing_pneumonia {classification} at level (0) {facility_level}  hsi {hsi_event} '
                                  f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')

            elif facility_level == '1a':
                if get_cons('Amoxicillin_suspension_or_tablet'):
                    do_treatment('5day_oral_amoxicillin')
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: treatment GIVEN for person {person_id} with '
                                      f'chest_indrawing_pneumonia {classification} at level (1a) {facility_level} hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                else:
                    schedule_hsi(HSI_IMCI_Pneumonia_Treatment_Outpatient_level_1b(person_id=person_id, module=self))
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: NO treatment for person {person_id} with '
                                      f'chest_indrawing_pneumonia {classification} at level (1a) {facility_level}  hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')

            elif facility_level == '1b':
                if get_cons('Amoxicillin_suspension_or_tablet'):
                    do_treatment('5day_oral_amoxicillin')
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: treatment GIVEN for person {person_id} with '
                                      f'chest_indrawing_pneumonia {classification} at level (1b) {facility_level} hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                else:
                    schedule_hsi(HSI_IMCI_Pneumonia_Treatment_Outpatient_level_2(person_id=person_id, module=self))
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: NO treatment for person {person_id} with '
                                      f'chest_indrawing_pneumonia {classification} at level (1b) {facility_level}  hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')

            elif facility_level == '2':
                if get_any_cons('Amoxicillin_suspension_or_tablet'):
                    do_treatment('5day_oral_amoxicillin')
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: treatment GIVEN for person {person_id} with '
                                      f'chest_indrawing_pneumonia {classification} at level (2) {facility_level} hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                else:
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: NO treatment for person {person_id} with '
                                      f'chest_indrawing_pneumonia {classification} at level (2) {facility_level}  hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')

            else:
                raise ValueError(f'facility_level{facility_level} not recognised for chest_indrawing_pneumonia')

        def do_if_danger_signs_pneumonia(facility_level):
            # Get bronchodilator if wheeze and at facility_level 1a, 1b, 2
            if 'wheeze' in symptoms:
                if facility_level == '1a':
                    get_cons('Inhaled_Brochodilator')
                elif facility_level in ('1b', '2'):
                    get_cons('Brochodilator_and_Steroids')

            if facility_level == '0':
                get_cons('First_dose_antibiotic_for_referral_iCCM')
                logger.debug(key='message',
                             data=f'ALRI_HSI_Event: NO treatment for person {person_id} with '
                                  f'danger_signs_pneumonia {classification} at level (0) {facility_level} hsi {hsi_event} '
                                  f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                schedule_hsi(HSI_IMCI_Pneumonia_Treatment_Inpatient_level_1b(person_id=person_id, module=self))

            elif facility_level == '1a':
                get_cons('First_dose_antibiotic_for_referral_IMCI')
                logger.debug(key='message',
                             data=f'ALRI_HSI_Event: NO treatment for person {person_id} with '
                                  f'danger_signs_pneumonia {classification} at level (1a) {facility_level} hsi {hsi_event} '
                                  f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                schedule_hsi(HSI_IMCI_Pneumonia_Treatment_Inpatient_level_2(person_id=person_id, module=self))

            elif facility_level == '1b':
                if oxygen_saturation == '<90%':  # need oxygen to survive
                    if get_cons('Oxygen_Therapy') and get_cons('1st_line_Antibiotic_Therapy_for_Severe_Pneumonia'):
                        do_treatment(['1st_line_IV_antibiotics', 'oxygen'])
                        logger.debug(key='message',
                                     data=f'ALRI_HSI_Event: treatment GIVEN for person {person_id} with '
                                          f'danger_signs_pneumonia {classification} at level (1b) {facility_level} hsi {hsi_event} '
                                          f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                    else:
                        logger.debug(key='message',
                                     data=f'ALRI_HSI_Event: NO treatment for person {person_id} with '
                                          f'danger_signs_pneumonia {classification} at level (1b) {facility_level} hsi {hsi_event} '
                                          f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                        schedule_hsi(HSI_IMCI_Pneumonia_Treatment_Inpatient_level_2(person_id=person_id, module=self))

                else:
                    # TODO: add on a script analysis giving oxygen at SpO2 <93%
                    if get_cons('1st_line_Antibiotic_Therapy_for_Severe_Pneumonia'):
                        do_treatment('1st_line_IV_antibiotics')
                        logger.debug(key='message',
                                     data=f'ALRI_HSI_Event: treatment GIVEN for person {person_id} with '
                                          f'danger_signs_pneumonia {classification} at level (1b) {facility_level} hsi {hsi_event} '
                                          f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                    else:
                        logger.debug(key='message',
                                     data=f'ALRI_HSI_Event: NO treatment for person {person_id} with '
                                          f'danger_signs_pneumonia {classification} at level (1b) {facility_level} hsi {hsi_event} '
                                          f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                        schedule_hsi(HSI_IMCI_Pneumonia_Treatment_Inpatient_level_2(person_id=person_id, module=self))

            elif facility_level == '2':
                if oxygen_saturation == '<90%':  # need oxygen to survive
                    if get_cons('Oxygen_Therapy') and get_cons('1st_line_Antibiotic_Therapy_for_Severe_Pneumonia'):
                        do_treatment(['1st_line_IV_antibiotics', 'oxygen'])
                        logger.debug(key='message',
                                     data=f'ALRI_HSI_Event: treatment GIVEN for person {person_id} with '
                                          f'danger_signs_pneumonia {classification} at level (2) {facility_level} hsi {hsi_event} '
                                          f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                else:
                    get_cons('Oxygen_Therapy')
                    # TODO: add on a script analysis giving oxygen at SpO2 <93%
                    if get_cons('1st_line_Antibiotic_Therapy_for_Severe_Pneumonia'):
                        do_treatment('1st_line_IV_antibiotics')
                        logger.debug(key='message',
                                     data=f'ALRI_HSI_Event: treatment GIVEN for person {person_id} with '
                                          f'danger_signs_pneumonia {classification} at level (2) {facility_level} hsi {hsi_event} '
                                          f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                    else:
                        logger.debug(key='message',
                                     data=f'ALRI_HSI_Event: NO treatment for person {person_id} with '
                                          f'danger_signs_pneumonia {classification} at level (2) {facility_level} hsi {hsi_event} '
                                          f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')

            else:
                raise ValueError(f'facility_level{facility_level} not recognised for danger_signs_pneumonia')

        def do_if_serious_bacterial_infection(facility_level):
            if facility_level == '1a':
                schedule_hsi(HSI_IMCI_Pneumonia_Treatment_Inpatient_level_2(person_id=person_id, module=self))
                logger.debug(key='message',
                             data=f'ALRI_HSI_Event: NO treatment for person {person_id} with '
                                  f'serious_bacterial_infection {classification} at level (1a) {facility_level} hsi {hsi_event} '
                                  f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')

            elif facility_level == '1b':
                if get_cons('Oxygen_Therapy') and get_cons('1st_line_Antibiotic_Therapy_for_Severe_Pneumonia'):
                    do_treatment(['1st_line_IV_antibiotics', 'oxygen'])
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: treatment GIVEN for person {person_id} with '
                                      f'serious_bacterial_infection {classification} at level (1b) {facility_level} hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                else:
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: NO treatment  for person {person_id} with '
                                      f'serious_bacterial_infection {classification} at level (1b) {facility_level} hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                    schedule_hsi(HSI_IMCI_Pneumonia_Treatment_Inpatient_level_2(person_id=person_id, module=self))

            elif facility_level == '2':
                if get_cons('Oxygen_Therapy') and get_cons('1st_line_Antibiotic_Therapy_for_Severe_Pneumonia'):
                    do_treatment(['1st_line_IV_antibiotics', 'oxygen'])
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: treatment GIVEN for person {person_id} with '
                                      f'serious_bacterial_infection {classification} at level (2) {facility_level} hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                else:
                    logger.debug(key='message',
                                 data=f'ALRI_HSI_Event: NO treatment  for person {person_id} with '
                                      f'serious_bacterial_infection {classification} at level (2) {facility_level} hsi {hsi_event} '
                                      f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
            else:
                raise ValueError(f'facility_level{facility_level} not recognised for serious_bacterial_infection')

        def do_if_not_handled_at_facility_0(facility_level):
            if facility_level == '0':
                logger.debug(key='message',
                             data=f'ALRI_HSI_Event: not handled at level 0 person {person_id} with '
                                  f'symptoms {symptoms} {self.sim.modules["SymptomManager"].has_what(person_id)} '
                                  f'at level (0) {facility_level} hsi {hsi_event} '
                                  f'with ri_on_treatment property = {df.loc[person_id, "ri_on_treatment"]}')
                schedule_hsi(HSI_IMCI_Pneumonia_Treatment_Outpatient_level_1a(person_id=person_id, module=self))
            else:
                raise ValueError(f'facility_level {facility_level} not recognised for not_handled_at_facility_0')

        def do_if_cough_or_cold(facility_level):
            logger.debug(key='message',
                         data=f'ALRI_HSI_Event: cough or common cold classification for {person_id}, '
                              f'presenting at facility level {facility_level}, '
                              f'check their symptoms {symptoms} ,  {self.sim.modules["SymptomManager"].has_what(person_id)} '
                              f'and properties '
                              f'{self.sim.population.props.loc[person_id, self.PROPERTIES.keys()].to_dict()}')
            assert facility_level != '2'

        do_mapping = {
            'fast_breathing_pneumonia': do_if_fast_breathing_pneumonia,
            'chest_indrawing_pneumonia': do_if_chest_indrawing_pneumonia,
            'danger_signs_pneumonia': do_if_danger_signs_pneumonia,
            'serious_bacterial_infection': do_if_serious_bacterial_infection,
            'not_handled_at_facility_0': do_if_not_handled_at_facility_0,
            'cough_or_cold': do_if_cough_or_cold
        }

        do_mapping[classification](facility_level=hsi_event.ACCEPTED_FACILITY_LEVEL)

    def assess_and_classify_cough_or_difficult_breathing_level(self, person_id, hsi_event):
        """This routine is called when cough or difficulty breathing is a symptom for a child attending
        a Generic HSI Appointment at level 0, or a Generic Emergency First Appt at level 1b.
        It is also called by HSI events owned by the ALRI module."""

        df = self.sim.population.props
        person = df.loc[person_id]
        symptoms = self.sim.modules['SymptomManager'].has_what(person_id)

        if not (('cough' in symptoms) or ('difficult_breathing' in symptoms)):
            return

        # get the iCCM/IMCI classification based on symptoms alone
        symptom_based_classification = self.imci_classification_based_on_symptoms(
            age_exact_years=person.age_exact_years,
            symptoms=symptoms,
            facility_level=hsi_event.ACCEPTED_FACILITY_LEVEL)

        # update the property
        df.at[person_id, 'ri_symptom_based_pneumonia_classification'] = symptom_based_classification

        # classification based on SpO2
        SpO2_based_classification = self.imci_classification_with_SpO2_measure(
            symptom_classification=symptom_based_classification,
            oxygen_saturation=person.ri_SpO2_level
        )

        # classification given by the health workers
        hw_assigned_classification = self.classification_given_by_health_worker(
            hsi_event=hsi_event,
            symptoms=symptoms,
            imci_symptom_based_classification=symptom_based_classification,
            facility_level=hsi_event.ACCEPTED_FACILITY_LEVEL)

        # if pulse oximeter available, get the changes in classification
        classification = self.final_classification(
            hw_assigned_classification=hw_assigned_classification,
            oxygen_saturation=person.ri_SpO2_level,
            oximeter_available=hsi_event.get_consumables(item_codes=self.consumables_used_in_hsi['Pulse_oximetry'])
        )

        # Do the actions required, given this classification
        self.do_action_given_classification(
            person_id=person_id,
            classification=classification,
            symptoms=symptoms,
            oxygen_saturation=person.ri_SpO2_level,
            hsi_event=hsi_event)

        # log the event for this person
        logger.info(
            key='classification_and_treatment',
            data={'person': person_id,
                  'symptom_classification': symptom_based_classification,
                  'pulse_ox_classification': SpO2_based_classification,
                  'hw_classification': hw_assigned_classification,
                  'final_classification': classification,
                  'facility_level': hsi_event.ACCEPTED_FACILITY_LEVEL,
                  'treatment': None}
        )

    def do_effects_of_alri_treatment(self, person_id, hsi_event, treatment):
        """Helper function that enacts the effects of a treatment to Alri caused by a pathogen.
        It will only do something if the Alri is caused by a pathogen (this module).
        * Log the treatment
        * Prevent any death event that may be scheduled from occurring (prior to the cure event)
        * Schedules the cure event, at which the episode is ended
        * Schedules a follow-up appointment if condition not improving (by day 6 or by day 14)  # todo: not yet added
        """

        df = self.sim.population.props
        imci_classification = df.at[person_id, 'ri_symptom_based_pneumonia_classification']

        df = self.sim.population.props
        person = df.loc[person_id]
        p = self.parameters

        # Do nothing if the person is not infected with a pathogen that can cause ALRI
        if not person['ri_current_infection_status']:
            return

        # log the event for this person
        logger.info(
            key='classification_and_treatment',
            data={'person': person_id,
                  'treatment': treatment}
        )

        # Record that the person is now on treatment:
        df.loc[person_id, ('ri_on_treatment', 'ri_ALRI_tx_start_date')] = (True, self.sim.date)

        def treatment_will_fail(treatment):
            """Determine if the treatment will fail"""
            # First check for the treatment given to the correct classification for the condition -----
            if treatment == '1st_line_IV_antibiotics' and (imci_classification == 'danger_sings_pneumonia'):
                df.at[person_id, 'ri_tx_antibiotic_therapy'] = '5day_oral_amoxicillin'
                return p['1st_line_antibiotic_for_severe_pneumonia_treatment_failure_by_day2'] > self.rng.rand()

            if treatment == '3day_oral_amoxicillin' and (
                    imci_classification == 'fast_breathing_pneumonia'):
                return self.parameters['3day_amoxicillin_for_fast_breathing_treatment_failure_or_relapse'] > self.rng.rand()

            if treatment == '5day_oral_amoxicillin' and (
                    imci_classification == 'chest_indrawing_pneumonia'):
                return self.parameters['5day_amoxicillin_for_chest_indrawing_treatment_failure_or_relapse'] > self.rng.rand()

            # Check for the treatment given to the incorrect classification for the condition -----
            # for classification == 'chest_indrawing_pneumonia' given oral antibiotics as per fast_breathing_pneumonia
            if treatment == '3day_oral_amoxicillin' and (
                    imci_classification == 'chest_indrawing_pneumonia'):
                return self.parameters['3day_amoxicillin_for_chest_indrawing_treatment_failure_or_relapse'] > self.rng.rand()

            # for classification == 'danger_signs_pneumonia' given oral antibiotics as per non-severe pneumonia
            if treatment == '5day_oral_amoxicillin' and (
                    imci_classification == 'danger_signs_pneumonia'):
                return 0.1 > self.rng.rand()

        if not treatment_will_fail(treatment):
            self.cancel_death_and_schedule_cure(person_id)
        else:
            if treatment == 'IMCI_Treatment_severe_pneumonia' and (imci_classification == 'danger_sings_pneumonia'):
                hsi_event.get_consumables(
                    item_codes=self.consumables_used_in_hsi['2nd_line_Antibiotic_Therapy_for_Severe_Pneumonia'])
            #TODO: schedule follow-ups here


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
                        conditions_are_exhaustive=True
                    ).when(0, age_effects[0])
                     .when(1, age_effects[1])
                     .when(2, age_effects[2])
                     .when(3, age_effects[3])
                     .when(4, age_effects[4])
                     .when('>= 5', 0.0),
                    Predictor('li_wood_burn_stove').when(False, p['rr_ALRI_indoor_air_pollution']),
                    Predictor().when('(va_measles_all_doses == False) & (age_years >= 1)',
                                     p['rr_ALRI_incomplete_measles_immunisation']),
                    Predictor().when('(hv_inf == True) & (hv_art!= "on_VL_suppressed")', p['rr_ALRI_HIV/AIDS']),
                    Predictor('un_clinical_acute_malnutrition').when('SAM', p['rr_ALRI_underweight']),
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
                           unscaled_lm.predict(df.loc[df.is_alive]).groupby(by=df.age_years).mean()
                           ).to_dict()

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
            (age < 1) and (p[f'proportion_pneumonia_in_{pathogen}_ALRI'][0] > self.rng.rand())
        ) or (
            (1 <= age < 5) and (p[f'proportion_pneumonia_in_{pathogen}_ALRI'][1] > self.rng.rand())
        ):
            disease_type = 'pneumonia'
        else:
            disease_type = 'other_alri'

        # Determine bacterial-coinfection
        if pathogen in self.module.pathogens['viral']:
            if disease_type == 'pneumonia':
                if p['prob_viral_pneumonia_bacterial_coinfection'] > self.rng.rand():
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
            if prob_pulmonary_complications > self.rng.rand():
                for c in ['pneumothorax', 'pleural_effusion', 'lung_abscess', 'empyema']:
                    probs[c] += p[f'prob_{c}_in_pulmonary_complicated_pneumonia']
                    # TODO: lung abscess, empyema should only apply to (primary or secondary) bacteria ALRIs

            # probabilities for systemic complications
            if primary_path_is_bacterial or has_secondary_bacterial_inf:
                probs['sepsis'] += p['prob_bacteraemia_in_pneumonia'] * p['prob_progression_to_sepsis_with_bacteraemia']

            probs['hypoxaemia'] += p['prob_hypoxaemia_in_pneumonia']

        elif disease_type == 'other_alri':
            probs['hypoxaemia'] += p['prob_hypoxaemia_in_other_alri']

        # determine which complications are onset:
        complications = {c for c, p in probs.items() if p > self.rng.rand()}

        return complications

    def get_oxygen_saturation(self, complication_set):
        """Set peripheral oxygen saturation"""

        if 'hypoxaemia' in complication_set:
            if self.p['proportion_hypoxaemia_with_SpO2<90%'] > self.rng.rand():
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
        symptoms = {s for s, p in probs.items() if p > self.rng.rand()}

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
        symptoms = {s for s, p in probs.items() if p > self.rng.rand()}

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
            return risk_death > self.rng.rand()
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
        df.loc[person_id,
               (
                   'ri_current_infection_status',
                   'ri_primary_pathogen',
                   'ri_secondary_bacterial_pathogen',
                   'ri_disease_type',
                   'ri_on_treatment',
                   'ri_start_of_current_episode',
                   'ri_scheduled_recovery_date',
                   'ri_scheduled_death_date',
                   'ri_end_of_current_episode',
                   'ri_ALRI_tx_start_date'
               )] = (
            True,
            self.pathogen,
            bacterial_coinfection,
            disease_type,
            False,
            self.sim.date,
            pd.NaT,
            pd.NaT,
            episode_end,
            pd.NaT
        )

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
# ==================================== HEALTH SYSTEM INTERACTION EVENTS ====================================
# ---------------------------------------------------------------------------------------------------------

class HSI_IMCI_Pneumonia_Treatment_Outpatient_level_1a(HSI_Event, IndividualScopeEventMixin):
    """
    HSI event for treating uncomplicated pneumonia at the primary level (health centres)
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Pneumonia_Treatment_Outpatient_level_1a'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        person = df.loc[person_id]

        # Exit if the person is not alive or is not currently infected:
        if not (person.is_alive and person.ri_current_infection_status):
            return

        self.module.assess_and_classify_cough_or_difficult_breathing_level(person_id, hsi_event=self)


class HSI_IMCI_Pneumonia_Treatment_Outpatient_level_1b(HSI_Event, IndividualScopeEventMixin):
    """
    HSI event for treating uncomplicated pneumonia at the primary level hospitals
    (community/rural hospitals, CHAM hospitals)
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Pneumonia_Treatment_Outpatient_level_1b'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        person = df.loc[person_id]

        # Exit if the person is not alive or is not currently infected:
        if not (person.is_alive and person.ri_current_infection_status):
            return

        self.module.assess_and_classify_cough_or_difficult_breathing_level(person_id, hsi_event=self)


class HSI_IMCI_Pneumonia_Treatment_Outpatient_level_2(HSI_Event, IndividualScopeEventMixin):
    """
    HSI event for treating uncomplicated pneumonia at the district level / referral services
    (district hospitals)
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Pneumonia_Treatment_Outpatient_level_2'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '2'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        person = df.loc[person_id]

        # Exit if the person is not alive or is not currently infected:
        if not (person.is_alive and person.ri_current_infection_status):
            return

        self.module.assess_and_classify_cough_or_difficult_breathing_level(person_id, hsi_event=self)


class HSI_IMCI_Pneumonia_Treatment_Inpatient_level_1b(HSI_Event, IndividualScopeEventMixin):
    """
    HSI event for inpatient treatment of severe pneumonia at community/rusl hospitals and CHAM hospitals.
    It requires inpatient bed days.

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Pneumonia_Treatment_Inpatient_level_1b'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'InpatientDays': 2, 'IPAdmission': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 3})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        person = df.loc[person_id]

        # Exit if the person is not alive or is not currently infected:
        if not (person.is_alive and person.ri_current_infection_status):
            return

        self.module.assess_and_classify_cough_or_difficult_breathing_level(person_id, hsi_event=self)


class HSI_IMCI_Pneumonia_Treatment_Inpatient_level_2(HSI_Event, IndividualScopeEventMixin):
    """
    HSI event for inpatient treatment of severe pneumonia at referral hospital for that district.
    It requires inpatient bed days.

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Pneumonia_Treatment_Inpatient_level_2'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'InpatientDays': 2, 'IPAdmission': 1})
        self.ACCEPTED_FACILITY_LEVEL = '2'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 3})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        person = df.loc[person_id]

        # Exit if the person is not alive or is not currently infected:
        if not (person.is_alive and person.ri_current_infection_status):
            return

        self.module.assess_and_classify_cough_or_difficult_breathing_level(person_id, hsi_event=self)


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

    def new_case(self, age, pathogen):
        self.trackers['incident_cases'].add_one(age=age, pathogen=pathogen)

    def new_recovered_case(self, age, pathogen):
        self.trackers['recovered_cases'].add_one(age=age, pathogen=pathogen)

    def new_cured_case(self, age, pathogen):
        self.trackers['cured_cases'].add_one(age=age, pathogen=pathogen)

    def new_death(self, age, pathogen):
        self.trackers['deaths'].add_one(age=age, pathogen=pathogen)

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
            description='pathogens incident case counts in the last year'
        )

        # 2) Total number of cases, recovery, treatments and deaths since the last logging event
        logger.info(
            key='event_counts',
            data={k: v.report_current_total() for k, v in self.trackers.items()},
            description='Counts of cases, recovery, treatment and death in the last year'
        )

        # 3) Reset the trackers
        for tracker in self.trackers.values():
            tracker.reset()


class Tracker:
    """Helper class to be a counter for number of events occuring by age-group and by pathogen."""

    def __init__(self, age_grps: dict, pathogens: list):
        """Create and initalise tracker"""

        # Check and store parameters
        self.pathogens = pathogens
        self.age_grps_lookup = age_grps
        self.unique_age_grps = sorted(set(self.age_grps_lookup.values()))

        # Initialise Tracker
        self.tracker = None
        self.reset()

    def reset(self):
        """Produce a dict of the form: { <Age-Grp>: {<Pathogen>: <Count>} }"""
        self.tracker = {
            age: dict(zip(self.pathogens, [0] * len(self.pathogens))) for age in self.unique_age_grps
        }

    def add_one(self, age, pathogen):
        """Increment counter by one for a specific age and pathogen"""
        assert age in self.age_grps_lookup, 'Age not recognised'
        assert pathogen in self.pathogens, 'Pathogen not recognised'

        # increment by one:
        age_grp = self.age_grps_lookup[age]
        self.tracker[age_grp][pathogen] += 1

    def report_current_counts(self):
        return self.tracker

    def report_current_total(self):
        total = 0
        for _a in self.tracker.keys():
            total += sum(self.tracker[_a].values())
        return total

#
# class AlriAllCasesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
#     """This is the AlriAllCasesLoggingEvent.
#     This collects all new cases' information.
#     """
#
#     def __init__(self, module):
#         self.repeat = 1
#         super().__init__(module, frequency=DateOffset(years=self.repeat))
#
#         self.track_new_case_info = dict()
#         self.store_dataframe = pd.DataFrame([
#             'age_years',
#             [f'ri_complication_{complication}' for complication in self.module.complications],
#             'ri_SpO2_level',
#             'ri_start_of_current_episode',
#             'ri_scheduled_recovery_date',
#             'ri_scheduled_death_date',
#             'ri_ALRI_tx_start_date',
#             'ri_symptom_based_pneumonia_classification',
#             'ri_tx_oxygen_therapy',
#             'ri_tx_antibiotic_therapy'])
#
#     def log_new_case_df(self, person_df):
#         self.store_dataframe.append(person_df)
#
#     def apply(self, population):
#         df = self.sim.population.props
#
#         # Previous Year...
#         previous_year = self.sim.date - np.timedelta64(1, 'Y')
#
#         # all cases started within the past year
#         # past_year_cases = df.loc[df.ri_start_of_current_episode] < self.sim.date
#
#         dataframe = self.store_dataframe
#
#         print(self.store_dataframe)
#         print(dataframe)
#
#         logger.info(
#             key='new_case_event_info',
#             data={self.store_dataframe.items()},
#             description='Dataframe for new cases'
#         )
#
#         # 3) Reset the trackers
#         for tracker in self.track_new_case_info:
#             self.track_new_case_info[tracker] = 0
#

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
    """This Event for testing that is a drop-in replacement of `AlriIncidentCase`. It always produces an infection
    that will be lethal and should be classified as a severe pneumonia"""

    def __init__(self, module, person_id, pathogen):
        super().__init__(module, person_id=person_id, pathogen=pathogen)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters

        self.module.logging_event.new_case(age=df.at[person_id, 'age_years'], pathogen=self.pathogen)

        disease_type = 'pneumonia'
        duration_in_days_of_alri = 5
        date_of_outcome = self.module.sim.date + DateOffset(days=duration_in_days_of_alri)
        episode_end = date_of_outcome + DateOffset(days=p['days_between_treatment_and_cure'])

        chars = {
            'ri_current_infection_status': True,
            'ri_primary_pathogen': self.pathogen,
            'ri_secondary_bacterial_pathogen': np.nan,
            'ri_disease_type': disease_type,
            'ri_on_treatment': False,
            'ri_start_of_current_episode': self.sim.date,
            'ri_scheduled_recovery_date': pd.NaT,
            'ri_scheduled_death_date': date_of_outcome,
            'ri_end_of_current_episode': episode_end,
            'ri_ALRI_tx_start_date': pd.NaT,
            'ri_complication_hypoxaemia': True,
            'ri_SpO2_level': "<90%"
        }
        df.loc[person_id, chars.keys()] = chars.values()

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
