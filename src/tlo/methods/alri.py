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
The complications are onset at the time of disease onset.

The individual may recover naturally or die. The risk of death depends on the type of disease and the presence of some
of the complications.

Health care seeking is prompted by the onset of the symptom. The individual can be treated; if successful the risk of
death is lowered and the person is "cured" (symptom resolved) some days later.
"""

import types
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict, List

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

    # todo add imci classifications
    # todo - what is associated between disease type and different classifications?

    all_symptoms = {
        'cough', 'difficult_breathing', 'cyanosis', 'fever', 'tachypnoea', 'chest_indrawing', 'danger_signs'
    }

    # Declare the Alri complications:
    complications = sorted({
        'pneumothorax',
        'pleural_effusion',
        'empyema',
        'lung_abscess',
        'sepsis',
        'hypoxaemia'
    })

    # Declare the antibiotics that can be used
    antibiotics = sorted({
        'Amoxicillin_tablet_or_suspension_3days',
        'Amoxicillin_tablet_or_suspension_5days',
        'Amoxicillin_tablet_or_suspension_7days',
        '1st_line_IV_antibiotics',
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
        'rr_ALRI_wasting':  # todo - @ines - this parameter is not used: I will delete it.
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for infants with WASTING (whz<2sd)'
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
        'base_odds_death_ALRI_age<2mo':
            Parameter(Types.REAL,
                      'baseline odds of death from ALRI for young infants aged 0 month and severe pneumonia '
                      '(base group)'
                      ),
        'or_death_ALRI_age<2mo_very_severe_pneumonia':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for young infants with very severe pneumonia'
                      ),
        'or_death_ALRI_age<2mo_P.jirovecii':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for young infants with P. jirovecii infection'
                      ),
        'or_death_ALRI_age<2mo_by_month_increase_in_age':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for young infants by 1 month increase in age (1 month olds)'
                      ),

        'base_odds_death_ALRI_age2_59mo':
            Parameter(Types.REAL,
                      'baseline odds of death from ALRI for children aged 2 months, male, no SAM, '
                      'and non-severe pneumonia classification (base group)'
                      ),
        'or_death_ALRI_age2_59mo_female':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children who are female'
                      ),
        'or_death_ALRI_age2_59mo_very_severe_pneumonia':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children with very severe pneumonia'
                      ),
        'or_death_ALRI_age2_59mo_P.jirovecii':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children with P. jirovecii infection'
                      ),
        'or_death_ALRI_age2_59mo_by_month_increase_in_age':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI by 1 month increase in age for 2 to 59 months olds'
                      ),
        'or_death_ALRI_age2_59mo_SAM':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children with severe acute malnutrition'
                      ),

        'or_death_ALRI_SpO2<90%':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children with oxygen saturation < 90%, '
                      'base group: SpO2 <=93%'

                      ),
        'or_death_ALRI_SpO2_90_92%':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children with oxygen saturation between 09 to 92%, '
                      'base group: SpO2 <=93%'
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

        # todo - @ines --- what is meaning of "tmp param"???
        'tf_1st_line_antibiotic_for_severe_pneumonia':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'rr_tf_1st_line_antibiotics_if_cyanosis':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'rr_tf_1st_line_antibiotics_if_SpO2<90%':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'rr_tf_1st_line_antibiotics_if_abnormal_CXR':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'rr_tf_1st_line_antibiotics_if_MAM':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'rr_tf_1st_line_antibiotics_if_HIV/AIDS':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'rr_tf_1st_line_antibiotics_if_SAM':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'or_mortality_improved_oxygen_systems':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'tf_3day_amoxicillin_for_fast_breathing_with_SpO2>=90%':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 6 or relapse by day 14 '
                      'of 3-day course amoxicillin for treating fast-breathing pneumonia'
                      ),
        'tf_3day_amoxicillin_for_chest_indrawing_with_SpO2>=90%':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 6 or relapse by day 14 '
                      'of 3-day course amoxicillin for treating chest-indrawing pneumonia '
                      'without hypoxaemia (SpO2>=90%)'
                      ),
        'tf_5day_amoxicillin_for_chest_indrawing_with_SpO2>=90%':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 6 or relapse by day 14 '
                      'of 5-day course amoxicillin for treating chest-indrawing pneumonia '
                      'without hypoxaemia (SpO2>=90%)'
                      ),
        'tf_7day_amoxicillin_for_fast_breathing_pneumonia_in_young_infants':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 6 or relapse by day 14 '
                      'of 5-day course amoxicillin for treating chest-indrawing pneumonia '
                      'without hypoxaemia (SpO2>=90%)'
                      ),
        'tf_oral_amoxicillin_only_for_severe_pneumonia_with_SpO2>=90%':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 2 '
                      'for oral amoxicillin given to severe pneumonia (danger-signs)  without hypoxaemia (SpO2>=93%)'
                      ),
        'tf_oral_amoxicillin_only_for_non_severe_pneumonia_with_SpO2<90%':
            Parameter(Types.REAL,
                      'probability of treatment failure or relapse'
                      'for oral amoxicillin given to non-severe pneumonia (fast-breathing or chest-indrawing) '
                      'with hypoxaemia (SpO2<90%)'
                      ),
        'tf_oral_amoxicillin_only_for_severe_pneumonia_with_SpO2<90%':
            Parameter(Types.REAL,
                      'probability of treatment failure or relapse'
                      'for oral amoxicillin given to severe pneumonia (danger-signs) '
                      'with hypoxaemia (SpO2<90%)'
                      ),
        'tf_2nd_line_antibiotic_for_severe_pneumonia':
            Parameter(Types.REAL,
                      'probability of treatment failure by end of IV therapy'
                      'for 2nd line antibiotic either cloxacillin or ceftriaxone '
                      'to treat severe pneumonia (danger-signs) '
                      ),

        # sensitivities for correct classification by health workers
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

        # todo - @ines - what are these? Can we delete them??
        # extra
        'prob_cyanosis_in_other_alri':  # todo - NOT USED
            Parameter(Types.REAL,
                      'probability of cyanosis in bronchiolitis or other alri'
                      ),
        'prob_cyanosis_in_pneumonia':  # todo - NOT USED
            Parameter(Types.REAL,
                      'probability of cyanosis in pneumonia'
                      ),

        'prob_cyanosis_in_SpO2<90%':  # todo - it is used
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'or_care_seeking_perceived_severe_illness':  # todo - not used.
            Parameter(Types.BOOL,
                      'tmp param'
                      ),

        'pulse_oximeter_and_oxygen_is_available':
            Parameter(Types.CATEGORICAL,
                      'Control the availability of the pulse oximeter and oxygen. "Default" does not over-ride '
                      'availability; "Yes" forces them to be available; "No" forces them to not be available',
                      categories=['Yes', 'No', 'Default']
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
        for symptom_name in sorted(self.all_symptoms):
            if symptom_name not in self.sim.modules['SymptomManager'].generic_symptoms:
                if symptom_name == 'danger_signs':
                    self.sim.modules['SymptomManager'].register_symptom(
                        Symptom(name=symptom_name,
                                emergency_in_children=True))
                elif symptom_name == 'chest_indrawing':
                    self.sim.modules['SymptomManager'].register_symptom(
                        Symptom(name=symptom_name,
                                odds_ratio_health_seeking_in_children=2.4))  # <-- todo do not harcode!
                else:
                    self.sim.modules['SymptomManager'].register_symptom(
                        Symptom(name=symptom_name))
                    # (Associates the symptoms with the 'average' healthcare seeking, apart from "danger_signs", which
                    # is an emergency symptom in children, and "chest_indrawing" which does have increased healthcare
                    # seeking.)

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

        # override consumables availability
        self.over_ride_availability_of_certain_consumables()

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

    def over_ride_availability_of_certain_consumables(self):
        """Over-ride the availability of certain consumables, according the parameter values provided."""
        p = self.parameters
        item_code_pulse_oximeter = list(self.consumables_used_in_hsi['Pulse_oximetry'].keys())
        item_code_oxygen = list(self.consumables_used_in_hsi['Oxygen_Therapy'].keys())
        all_item_codes = list(set(item_code_pulse_oximeter + item_code_oxygen))

        if p['pulse_oximeter_and_oxygen_is_available'] in ('Yes', 'No'):
            over_ride = {
                _item: (1.0 if p['pulse_oximeter_and_oxygen_is_available'] == "Yes" else 0.0)
                for _item in all_item_codes
            }
            self.sim.modules['HealthSystem'].override_availability_of_consumables(over_ride)

    def look_up_consumables(self):
        """Look up and store the consumables item codes used in each of the HSI."""

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
                                                          {2: 0, 12: 0.006, 36: 0.012, np.inf: 0.018}
                                                          ),
            get_item_code(item='Amoxycillin 125mg/5ml suspension, PFR_0.025_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {2: 0, 12: 1, 36: 2, np.inf: 3}
                                                          ),
        }

        # Antibiotics for non-severe pneumonia - oral amoxicillin for 3 days
        self.consumables_used_in_hsi['Amoxicillin_tablet_or_suspension_3days'] = {
            get_item_code(item='Amoxycillin 250mg_1000_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {2: 0, 12: 0.01, 36: 0.02, np.inf: 0.03}
                                                          ),
            get_item_code(item='Amoxycillin 125mg/5ml suspension, PFR_0.025_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {2: 0, 12: 1, 36: 2, np.inf: 3}
                                                          ),
        }

        # Antibiotics for non-severe pneumonia - oral amoxicillin for 7 days for young infants only
        self.consumables_used_in_hsi['Amoxicillin_tablet_or_suspension_7days'] = {
            get_item_code(item='Amoxycillin 250mg_1000_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {1: 0.004, 2: 0.006, np.inf: 0.01}
                                                          ),
            get_item_code(item='Amoxycillin 125mg/5ml suspension, PFR_0.025_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {1: 0.4, 2: 0.5, np.inf: 1}
                                                          ),
        }

        # Antibiotic therapy for severe pneumonia - ampicillin package
        self.consumables_used_in_hsi['1st_line_IV_antibiotics'] = {
            get_item_code(item='Ampicillin injection 500mg, PFR_each_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {1: 3.73, 2: 5.6, 4: 8, 12: 16, 36: 24, np.inf: 40}
                                                          ),
            get_item_code(item='Gentamicin Sulphate 40mg/ml, 2ml_each_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {1: 0.7, 2: 1.4, 4: 2.81, 12: 4.69, 36: 7.03, np.inf: 9.37}
                                                          ),
            get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST'): 1,
            get_item_code(item='Syringe, Autodisable SoloShot IX '): 1
        }

        # # Antibiotic therapy for severe pneumonia - benzylpenicillin package when ampicillin is not available
        # self.consumables_used_in_hsi['Benzylpenicillin_gentamicin_therapy_for_severe_pneumonia'] = {
        #     get_item_code(item='Benzylpenicillin 3g (5MU), PFR_each_CMST'):
        #         lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
        #                                                   {1: 2, 2: 5, 4: 8, 12: 15, 36: 24, np.inf: 34}
        #                                                   ),
        #     get_item_code(item='Gentamicin Sulphate 40mg/ml, 2ml_each_CMST'):
        #         lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
        #                                                   {1: 0.7, 2: 1.4, 4: 2.81, 12: 4.69, 36: 7.03, np.inf: 9.37}
        #                                                   ),
        #     get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST'): 1,
        #     get_item_code(item='Syringe, Autodisable SoloShot IX '): 1
        # }

        # # Second line of antibiotics for severe pneumonia
        # self.consumables_used_in_hsi['Ceftriaxone_therapy_for_severe_pneumonia'] = {
        #     get_item_code(item='Ceftriaxone 1g, PFR_each_CMST'):
        #         lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
        #                                                   {4: 1.5, 12: 3, 36: 5, np.inf: 7}
        #                                                   ),
        #     get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST'): 1,
        #     get_item_code(item='Syringe, Autodisable SoloShot IX '): 1
        # }
        #
        # # Second line of antibiotics for severe pneumonia, if Staph is suspected
        # self.consumables_used_in_hsi['2nd_line_Antibiotic_therapy_for_severe_staph_pneumonia'] = {
        #     get_item_code(item='Flucloxacillin 250mg, vial, PFR_each_CMST'):
        #         lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
        #                                                   {2: 21, 4: 22.4, 12: 37.3, 36: 67.2, 60: 93.3, np.inf: 140}
        #                                                   ),
        #     get_item_code(item='Gentamicin Sulphate 40mg/ml, 2ml_each_CMST'):
        #         lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
        #                                                   {4: 2.81, 12: 4.69, 36: 7.03, 60: 9.37, np.inf: 13.6}
        #                                                   ),
        #     get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST'): 1,
        #     get_item_code(item='Syringe, Autodisable SoloShot IX '): 1,
        #     get_item_code(item='Flucloxacillin 250mg_100_CMST'):
        #         lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
        #                                                   {4: 0.42, 36: 0.84, 60: 1.68, np.inf: 1.68}
        #                                                   ),
        # }

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
            get_item_code(item='Syringe, Autodisable SoloShot IX '): 1
        }

        # Oxygen, pulse oximetry and x-ray -------------------

        # Oxygen for hypoxaemia
        self.consumables_used_in_hsi['Oxygen_Therapy'] = {
            get_item_code(item='Oxygen, 1000 liters, primarily with oxygen cylinders'): 1,
            # get_item_code(item='Nasal prongs'): 1  # todo @Ines - can this be deleted (why just commented out?)
        }

        # Pulse oximetry
        # todo - @Ines: just noting that this is looking-up oxygen rather than pulse oximeter.
        self.consumables_used_in_hsi['Pulse_oximetry'] = {
            get_item_code(item='Oxygen, 1000 liters, primarily with oxygen cylinders'): 1
        }

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

    def _treatment_fails(self, **kwargs) -> bool:
        """ Determine whether a treatment fails or not: Returns `True` if the treatment fails.
        Treatment failures are dependent on the underlying IMCI classification by symptom, the need for oxygen (if
        SpO2 < 90%), and the type of antibiotic therapy (oral vs. IV/IM)."""
        p_fail = self._prob_treatment_fails(**kwargs)
        assert p_fail is not None, f"no probability of failure is recorded, {kwargs=}"
        return p_fail > self.rng.random_sample()

    def _prob_treatment_fails(self,
                              imci_symptom_based_classification: str,
                              SpO2_level: str,
                              disease_type: str,
                              complications: List[str],
                              symptoms: List[str],
                              hiv_infected_and_not_on_art: bool,
                              un_clinical_acute_malnutrition: str,
                              antibiotic_provided: str,
                              oxygen_provided: bool
                              ) -> float:
        """Returns the probability of treatment failure. Treatment failures are dependent on the underlying IMCI
        classification by symptom, the need for oxygen (if SpO2 < 90%), and the type of antibiotic therapy (oral vs.
         IV/IM)."""

        assert (antibiotic_provided in self.antibiotics) or (antibiotic_provided == ''), \
            f"Not recognised {antibiotic_provided=}"

        p = self.parameters

        needs_oxygen = SpO2_level == "<90%"
        any_complications = len(complications) > 0

        if (antibiotic_provided == '') and (imci_symptom_based_classification == "cough_or_cold"):
            # If no antibiotic is provided, the treatment definitely fails, as it cannot have any effect!
            return 1.0

        if imci_symptom_based_classification == 'danger_signs_pneumonia':

            if antibiotic_provided == '1st_line_IV_antibiotics':
                # danger_signs_pneumonia given 1st line IV antibiotic:

                # Baseline risk of treatment failure (... if oxygen is also provided)
                risk_tf_1st_line_antibiotics = p['tf_1st_line_antibiotic_for_severe_pneumonia']

                # The effect of central cyanosis
                if 'cyanosis' in symptoms:
                    risk_tf_1st_line_antibiotics *= p['rr_tf_1st_line_antibiotics_if_cyanosis']

                if needs_oxygen:
                    risk_tf_1st_line_antibiotics *= p['rr_tf_1st_line_antibiotics_if_SpO2<90%']

                if disease_type == 'pneumonia':
                    risk_tf_1st_line_antibiotics *= p['rr_tf_1st_line_antibiotics_if_abnormal_CXR']

                # The effect of HIV
                if hiv_infected_and_not_on_art:
                    risk_tf_1st_line_antibiotics *= p['rr_tf_1st_line_antibiotics_if_HIV/AIDS']

                # The effect of acute malnutrition
                if un_clinical_acute_malnutrition == 'MAM':
                    risk_tf_1st_line_antibiotics *= p['rr_tf_1st_line_antibiotics_if_MAM']
                elif un_clinical_acute_malnutrition == 'SAM':
                    risk_tf_1st_line_antibiotics *= p['rr_tf_1st_line_antibiotics_if_SAM']

                if needs_oxygen and not oxygen_provided:
                    # Elevate risk, according to odds ratio "or_if_no_oxygen"
                    or_if_no_oxygen = 1.0 / p['or_mortality_improved_oxygen_systems']
                    to_odds = lambda pr: pr / (1.0 - pr)
                    to_prob = lambda odds: odds / (1.0 + odds)
                    risk_tf_1st_line_antibiotics = to_prob(to_odds(risk_tf_1st_line_antibiotics) * or_if_no_oxygen)

                return min(1.0, risk_tf_1st_line_antibiotics)

            else:
                # danger_signs_pneumonia given oral antibiotics (probably due to misdiagnosis)
                # todo - Should this depend on oxygen receive or not (should it used the same parameter value as above) - have put something in so that it makes sense; perhaps this should be the parameter from above?
                if needs_oxygen:
                    if oxygen_provided:
                        return p['tf_oral_amoxicillin_only_for_severe_pneumonia_with_SpO2<90%']
                    else:
                        return min(1.0, 2.0 * p['tf_oral_amoxicillin_only_for_severe_pneumonia_with_SpO2<90%'])
                else:
                    return p['tf_oral_amoxicillin_only_for_severe_pneumonia_with_SpO2>=90%']

        else:
            # Non-severe classifications ('fast_breathing_pneumonia', 'chest_indrawing_pneumonia, 'cough_or_cold')...
            if not needs_oxygen:
                # Non-severe classifications antibiotics that do not need oxygen

                if imci_symptom_based_classification == 'chest_indrawing_pneumonia':
                    if antibiotic_provided == 'Amoxicillin_tablet_or_suspension_5days':
                        return p['tf_5day_amoxicillin_for_chest_indrawing_with_SpO2>=90%']
                    elif antibiotic_provided == 'Amoxicillin_tablet_or_suspension_3days':
                        return p['tf_3day_amoxicillin_for_chest_indrawing_with_SpO2>=90%']
                    else:
                        return min(p['tf_5day_amoxicillin_for_chest_indrawing_with_SpO2>=90%'],
                                   p['tf_3day_amoxicillin_for_chest_indrawing_with_SpO2>=90%'])
                        # todo Need an else (in case they are something else) For now, I've put in the lower of the two others

                elif imci_symptom_based_classification == 'fast_breathing_pneumonia':
                    if antibiotic_provided == 'Amoxicillin_tablet_or_suspension_3days':
                        return p['tf_3day_amoxicillin_for_fast_breathing_with_SpO2>=90%']
                    elif antibiotic_provided == 'Amoxicillin_tablet_or_suspension_7days':
                        return p['tf_7day_amoxicillin_for_fast_breathing_pneumonia_in_young_infants']
                    else:
                        return min(p['tf_3day_amoxicillin_for_fast_breathing_with_SpO2>=90%'],
                                   p['tf_7day_amoxicillin_for_fast_breathing_pneumonia_in_young_infants'])
                        # todo Need an else (in case they are something else) For now, I've put in the lower of the two others

                # No pneumonia (by IMCI classification)
                elif imci_symptom_based_classification == "cough_or_cold" and not any_complications:
                    return 0.0  # Treatment cannot 'fail' for a cough_or_cold without complications and no need of oxygen

                elif imci_symptom_based_classification == "cough_or_cold" and any_complications:
                    return p['tf_5day_amoxicillin_for_chest_indrawing_with_SpO2>=90%']

            else:
                # Non-severe classifications given oral antibiotics that do need oxygen -----
                # todo - it should depend on receiving oxygen or not!!! Have put something it so that it makes sense
                if oxygen_provided:
                    return p['tf_oral_amoxicillin_only_for_non_severe_pneumonia_with_SpO2<90%']
                else:
                    return min(1.0, 2.0 * p['tf_oral_amoxicillin_only_for_non_severe_pneumonia_with_SpO2<90%'])

    def do_effects_of_treatment(self, person_id, antibiotic_provided: str, oxygen_provided: bool):
        """Helper function that enacts the effects of a treatment to Alri caused by a pathogen.
        It will only do something if the Alri is caused by a pathogen (this module).
        * Prevent any death event that may be scheduled from occurring
        * Returns True for failed treatment to further schedule    # todo - the return thing seems weird -- sometimes None, T/F wrong way around?
        a follow-up appointment if condition not improving (by day 6 or by day 14)
        """
        # todo rewrite the docstring above
        # todo refactor so that it's based on the characteristics of the person directly
        # todo nb that antibiotic_provided = '' means no antibiotic provided.

        df = self.sim.population.props
        person = df.loc[person_id]

        if not person.ri_current_infection_status:
            return

        # Record that the person is now on treatment:
        self.sim.population.props.at[person_id, 'ri_on_treatment'] = True
        self.logging_event.new_treated()

        # Gather underlying properties that will affect success of treatment
        SpO2_level = person.ri_SpO2_level
        symptoms = self.sim.modules['SymptomManager'].has_what(person_id)
        imci_symptom_based_classification = self.get_imci_classification_based_on_symptoms(
            child_is_younger_than_2_months=person.age_exact_years < (2.0 / 12.0),
            symptoms=symptoms,
        )
        disease_type = person.ri_disease_type
        hiv_infected_and_not_on_art = person.hv_inf and (person.hv_art != "on_VL_suppressed")
        un_clinical_acute_malnutrition = person.un_clinical_acute_malnutrition
        complications = [_c for _c in self.complications if person[f"ri_complication_{_c}"]]

        # Will the treatment fail:
        treatment_fails = self._treatment_fails(
            imci_symptom_based_classification=imci_symptom_based_classification,
            SpO2_level=SpO2_level,
            disease_type=disease_type,
            complications=complications,
            symptoms=symptoms,
            hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
            un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
            antibiotic_provided=antibiotic_provided,
            oxygen_provided=oxygen_provided,
        )

        # Cancel death (if there is one) if the treatment does not fail:
        if not treatment_fails:
            self.cancel_death_and_schedule_cure(person_id)
        # else:
        #     return treatment_fails   # todo ??!!??!??!!

    def on_presentation(self, person_id, hsi_event):
        """Action taken when a child (under 5 years old) presents at a generic appointment (emergency or non-emergency)
         with symptoms of `cough` or `difficult_breathing`."""

        self.record_sought_care_for_alri()

        # All persons have an initial out-patient appointment at the current facility level.
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
             'fast_breathing_pneumonia',
             'danger_signs_pneumonia',
             'chest_indrawing_pneumonia,
             'cough_or_cold'
        }."""
        if child_is_younger_than_2_months:
            if ('chest_indrawing' in symptoms) or ('danger_signs' in symptoms):
                return 'danger_signs_pneumonia'
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

    def determine_disease_type_and_secondary_bacterial_coinfection(self,
                                                                   pathogen,
                                                                   age_exact_years,
                                                                   va_hib_all_doses,
                                                                   va_pneumo_all_doses):
        """Determines:
         * the disease that is caused by infection with this pathogen (from among self.disease_types)
         * if there is a bacterial coinfection associated that will cause the dominant disease.

         Note that the disease_type is 'bacterial_pneumonia' if primary pathogen is viral and there is a secondary
         bacterial coinfection.
         """
        p = self.p

        # Determine the disease type - pneumonia or other_alri
        if (
            (age_exact_years < 1.0) and (p[f'proportion_pneumonia_in_{pathogen}_ALRI'][0] > self.rng.random_sample())
        ) or (
            (1.0 <= age_exact_years < 5.0) and (
            p[f'proportion_pneumonia_in_{pathogen}_ALRI'][1] > self.rng.random_sample())
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
                    # TODO: @Ines - turn this into an issue - lung abscess, empyema should only apply to
                    #  (primary or secondary) bacteria ALRIs. Requires two sets of parameters for the prob
                    #  that a complication is one of possible types for those with and without bacterial infection
                    #  (i.e. the vector of probabilities must sum to 1.0 in either case).

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

    def symptoms_for_disease(self, disease_type) -> set:
        """Determine set of symptom (before complications) for a given instance of disease"""
        p = self.p

        assert disease_type in self.module.disease_types

        probs = {
            symptom: p[f'prob_{symptom}_in_{disease_type}']
            for symptom in [
                'cough', 'difficult_breathing', 'cyanosis', 'fever', 'tachypnoea',
                'chest_indrawing', 'danger_signs']
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
                    'chest_indrawing': p['prob_chest_indrawing_in_SpO2<90%'],
                    'cyanosis': p['prob_cyanosis_in_SpO2<90%']
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

    def will_die_of_alri(self, **kwargs) -> bool:
        """Determine if person will die from Alri"""
        return self.prob_die_of_alri(**kwargs) > self.rng.random_sample()

    def prob_die_of_alri(self,
                         age_exact_years: float,
                         sex: str,
                         pathogen: str,
                         disease_type: str,
                         SpO2_level: str,
                         complications: List[str],
                         danger_signs: bool,
                         un_clinical_acute_malnutrition: str,
                         ) -> float:
        """Returns the probability that someone will die of ALRI"""
        # todo - this needs radical tidying up!

        p = self.p

        # Death does not occur if the disease_type is "other_alri"
        if disease_type == "other_alri":
            return 0.0

        # Death does not occur if there are no complications
        if 0 == len(complications):
            return 0.0

        # Baseline risk for young infants aged less than 2 months:
        odds_death_age_lt2mo = p['base_odds_death_ALRI_age<2mo']

        # The effect of age increase by 1 month:
        if 1.0 / 12.0 < age_exact_years < 1.0 / 6.0:
            odds_death_age_lt2mo *= p['or_death_ALRI_age<2mo_by_month_increase_in_age']

        # Baseline risk for children aged 2 to 59 months:
        odds_death_age2_59mo = p['base_odds_death_ALRI_age2_59mo']

        # The effect of age by 1 month increase in age:
        # age range for the 3 to 59 months (2 month old as baseline)
        age_range_in_months = range(3, 60)
        one_month_in_a_year = 1.0 / 12.0

        # todo sort this out !!!!! ;-)
        for i in age_range_in_months:
            if (i * one_month_in_a_year) <= age_exact_years < ((i + 1) * one_month_in_a_year):
                odds_death_age2_59mo *= (p['or_death_ALRI_age2_59mo_by_month_increase_in_age'] ** i)

        # The effect of disease severity
        if danger_signs:
            odds_death_age_lt2mo *= p['or_death_ALRI_age<2mo_very_severe_pneumonia']

        # The effect of P.jirovecii infection:
        if pathogen == 'P.jirovecii':
            odds_death_age_lt2mo *= p['or_death_ALRI_age<2mo_P.jirovecii']

        # The effect of gender:
        if sex == 'F':
            odds_death_age2_59mo *= p['or_death_ALRI_age2_59mo_female']

        # The effect of disease severity
        if danger_signs:
            odds_death_age2_59mo *= p['or_death_ALRI_age2_59mo_very_severe_pneumonia']

        # The effect of P.jirovecii infection:
        if pathogen == 'P.jirovecii':
            odds_death_age2_59mo *= p['or_death_ALRI_age2_59mo_P.jirovecii']

        # The effect of factors defined in other modules:
        # malnutrition:
        if un_clinical_acute_malnutrition == 'SAM':
            odds_death_age2_59mo *= p['or_death_ALRI_age2_59mo_SAM']

        # The effect of hypoxaemia for all ages:
        if SpO2_level == '<90%':
            odds_death_age_lt2mo *= p['or_death_ALRI_SpO2<90%']
            odds_death_age2_59mo *= p['or_death_ALRI_SpO2<90%']
        if SpO2_level == '90-92%':
            odds_death_age_lt2mo *= p['or_death_ALRI_SpO2_90_92%']
            odds_death_age2_59mo *= p['or_death_ALRI_SpO2_90_92%']
        # todo @ ines --- if this level of saturation does not indicate for oxygen then it doesn't make sense that it increase risk of death. So, have made it so such persons can reach treatment....

        # todo - check this works!
        if 'sepsis' in complications:
            odds_death_age_lt2mo *= 151.9  # todo --- this seems VERY HIGH and why is it hardcoded?
            odds_death_age2_59mo *= 151.9  # todo --- this seems VERY HIGH and why is it hardcoded?

        if 'pneumothorax' in complications:
            odds_death_age_lt2mo *= 77.4  # todo --- this seems VERY HIGH and why is it hardcoded?
            odds_death_age2_59mo *= 77.4  # todo --- this seems VERY HIGH and why is it hardcoded?

        # Convert odds to probability
        risk_death_age_lt2mo = odds_death_age_lt2mo / (1 + odds_death_age_lt2mo)
        risk_death_age2_59mo = odds_death_age2_59mo / (1 + odds_death_age2_59mo)

        if age_exact_years < 2.0 / 12.0:
            return risk_death_age_lt2mo
        else:
            return risk_death_age2_59mo


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

    @property
    def fraction_of_year_between_polling_event(self):
        """Return the fraction of a year that elapses between polling event. This is used to adjust the risk of
        infection"""
        return (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'Y')

    def get_probs_of_acquiring_pathogen(self, interval_as_fraction_of_a_year: float):
        """Return the probability of each person in the dataframe acquiring each pathogen, during the time interval
        specified (as fraction of a year)."""
        df = self.sim.population.props
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

        return 1 - np.exp(
            -inc_of_acquiring_alri * interval_as_fraction_of_a_year
        )

    def get_persons_for_new_alri(self) -> Dict:
        """Returns Dict of the form {<person_id>: <pathogen>} representing the persons who will become infected and with
        what pathogen."""
        # Get probabilities for each person of being infected with each pathogen
        probs_of_acquiring_pathogen = self.get_probs_of_acquiring_pathogen(
            interval_as_fraction_of_a_year=self.fraction_of_year_between_polling_event)

        # Sample to find outcomes (who is infected and with what pathogen):
        return sample_outcome(probs=probs_of_acquiring_pathogen, rng=self.module.rng)

    def apply(self, population):
        """Determine who will become infected and schedule for them an AlriComplicationOnsetEvent"""

        new_alri = self.get_persons_for_new_alri()

        # For each person that will have ALRI, schedule an AlriIncidentCase event
        for person_id, pathogen in new_alri.items():
            #  Create the event for the onset of infection:
            self.sim.schedule_event(
                event=AlriIncidentCase(
                    module=self.module,
                    person_id=person_id,
                    pathogen=pathogen,
                ),
                date=random_date(self.sim.date, self.sim.date + self.frequency - pd.DateOffset(days=1), self.module.rng)
            )


class AlriIncidentCase(Event, IndividualScopeEventMixin):
    """This Event is for the onset of the infection that causes Alri. It is scheduled by the AlriPollingEvent."""

    def __init__(self, module, person_id, pathogen):
        super().__init__(module, person_id=person_id)
        self.pathogen = pathogen

    def determine_nature_of_the_case(self,
                                     age_exact_years,
                                     sex,
                                     pathogen,
                                     va_hib_all_doses,
                                     va_pneumo_all_doses,
                                     un_clinical_acute_malnutrition,
                                     ) -> Dict:
        """Determine all the characteristics of this case:
            * disease_type,
            * bacterial_coinfection,
            * duration_in_days_of_alri,
            * date_of_outcome,
            * episode_end,
            * symptoms,
            * complications,
            * oxygen_saturation,
            * will_die,
        """
        models = self.module.models
        rng = self.module.rng
        params = self.module.parameters
        date = self.module.sim.date

        # ----------------- Determine the Alri disease type and bacterial coinfection for this case -----------------
        disease_type, bacterial_coinfection = \
            models.determine_disease_type_and_secondary_bacterial_coinfection(
                age_exact_years=age_exact_years,
                pathogen=pathogen,
                va_hib_all_doses=va_hib_all_doses,
                va_pneumo_all_doses=va_pneumo_all_doses,
            )

        # ----------------------- Duration of the Alri event and episode -----------------------
        duration_in_days_of_alri = rng.randint(3, params[
            'max_alri_duration_in_days_without_treatment'])  # todo changed this 1-->3 to help with problems about timeliness of healthcare seeking; could chnage it back now that we have instant healthcare seeking

        # Date for outcome (either recovery or death) with uncomplicated Alri
        date_of_outcome = date + DateOffset(days=duration_in_days_of_alri)

        # Define 'episode end' date. This the date when this episode ends. It is the last possible data that any HSI
        # could affect this episode.
        episode_end = date_of_outcome + DateOffset(days=params['days_between_treatment_and_cure'])

        # ----------------------------------- Clinical Symptoms -----------------------------------
        symptoms_for_uncomplicated_disease = models.symptoms_for_disease(disease_type=disease_type)

        # ----------------------------------- Complications  -----------------------------------
        complications = models.get_complications_that_onset(
            disease_type=disease_type,
            primary_path_is_bacterial=(pathogen in self.module.pathogens['bacterial']),
            has_secondary_bacterial_inf=pd.notnull(bacterial_coinfection)
        )

        oxygen_saturation = models.get_oxygen_saturation(complication_set=complications)

        def get_symptoms_for_complications(complications: set, oxygen_saturation: str) -> set:
            """Return the set of symptoms consistent with the set of complications that are onset."""
            symptoms_for_complications = set()
            for complication in complications:
                symptoms_for_complications = symptoms_for_complications.union(
                    models.symptoms_for_complication(complication=complication, oxygen_saturation=oxygen_saturation))
            return symptoms_for_complications

        symptoms_for_complications = get_symptoms_for_complications(complications=complications,
                                                                    oxygen_saturation=oxygen_saturation)

        all_symptoms = sorted(symptoms_for_uncomplicated_disease.union(symptoms_for_complications))

        # ----------------------------------- Whether Will Die  -----------------------------------
        will_die = models.will_die_of_alri(
            age_exact_years=age_exact_years,
            sex=sex,
            pathogen=pathogen,
            disease_type=disease_type,
            SpO2_level=oxygen_saturation,
            complications=complications,
            danger_signs=('danger_signs' in all_symptoms),
            # todo <--- is this right? # turn danger_signs into a property??
            un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
        )

        return {
            'disease_type': disease_type,
            'bacterial_coinfection': bacterial_coinfection,
            'duration_in_days_of_alri': duration_in_days_of_alri,
            'date_of_outcome': date_of_outcome,
            'episode_end': episode_end,
            'symptoms': all_symptoms,
            'complications': complications,
            'oxygen_saturation': oxygen_saturation,
            'will_die': will_die,
        }

    def apply_characteristics_of_the_case(self, person_id, chars):
        """Update properties of the individual, impose symptoms and schedule events to reflect the characteristics of
        the case."""
        df = self.sim.population.props

        # Update the properties in the dataframe:
        _chars = {
            'ri_current_infection_status': True,
            'ri_primary_pathogen': self.pathogen,
            'ri_secondary_bacterial_pathogen': chars['bacterial_coinfection'],
            'ri_disease_type': chars['disease_type'],
            'ri_SpO2_level': chars['oxygen_saturation'],
            'ri_on_treatment': False,
            'ri_start_of_current_episode': self.sim.date,
            'ri_scheduled_recovery_date': pd.NaT,
            'ri_scheduled_death_date': pd.NaT,
            'ri_end_of_current_episode': chars['episode_end'],
        }
        df.loc[person_id, _chars.keys()] = _chars.values()
        df.loc[person_id, [f"ri_complication_{_complication}" for _complication in chars['complications']]] = True

        # Impose symptoms
        self.module.sim.modules['SymptomManager'].change_symptom(
            person_id=person_id,
            symptom_string=chars['symptoms'],
            add_or_remove='+',
            disease_module=self.module,
        )

        # Schedule death or recovery event
        if chars['will_die']:
            self.sim.schedule_event(AlriDeathEvent(self.module, person_id), chars['date_of_outcome'])
            df.loc[person_id, ['ri_scheduled_death_date', 'ri_scheduled_recovery_date']] = \
                [chars['date_of_outcome'], pd.NaT]
        else:
            self.sim.schedule_event(AlriNaturalRecoveryEvent(self.module, person_id), chars['date_of_outcome'])
            df.loc[person_id, ['ri_scheduled_recovery_date', 'ri_scheduled_death_date']] = \
                [chars['date_of_outcome'], pd.NaT]

        # Log the incident case:
        self.module.logging_event.new_case(age=df.at[person_id, "age_years"], pathogen=self.pathogen)

        # Log the complications to the tracker
        if set(chars['complications']).intersection(['pneumothorax', 'pleural_effusion', 'empyema', 'lung_abscess']):
            self.module.logging_event.new_pulmonary_complication_case()
        if 'sepsis' in chars['complications']:
            self.module.logging_event.new_systemic_complication_case()
        if 'hypoxaemia' in chars['complications']:
            self.module.logging_event.new_hypoxaemic_case()
        if (
            ("danger_signs_pneumonia" == self.module.get_imci_classification_based_on_symptoms(
                symptoms=chars['symptoms'],
                child_is_younger_than_2_months=df.at[person_id, "age_years"] < 2)
            ) and (
            chars['oxygen_saturation'] == "<90%")
        ):
            self.module.logging_event.new_danger_signs_pneumonia_case()

    def apply(self, person_id):
        """Determines and enacts all the characteristics of the case (symotoms, complications, outcome)"""
        df = self.sim.population.props
        person = df.loc[person_id]

        # The event should not run if the person is not currently alive:
        if not person.is_alive:
            return

        # Get the characteristics of the case
        chars = self.determine_nature_of_the_case(
            pathogen=self.pathogen,
            sex=person.sex,
            age_exact_years=person.age_exact_years,
            va_hib_all_doses=person.va_hib_all_doses,
            va_pneumo_all_doses=person.va_pneumo_all_doses,
            un_clinical_acute_malnutrition=person.un_clinical_acute_malnutrition,
        )

        # Apply the characteristics to this case:
        self.apply_characteristics_of_the_case(person_id=person_id, chars=chars)


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
                pathogen=pathogen,
                sp02_level=person.ri_SpO2_level
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

        if person_id is not None:
            self._age_exact_years = self.sim.population.props.at[person_id, 'age_exact_years']

    def _as_out_patient(self, facility_level):
        """Cast this HSI as an out-patient appointment."""
        self.TREATMENT_ID = f'{self._treatment_id_stub}_Outpatient{"_Followup" if self.is_followup else ""}'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({
            ('ConWithDCSA' if facility_level == '0' else 'Under5OPD'): 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level

    def _as_in_patient(self, facility_level):
        """Cast this HSI as an in-patient appointment."""
        self.TREATMENT_ID = f'{self._treatment_id_stub}_Inpatient{"_Followup" if self.is_followup else ""}'  # todo - Follow-up as inpatient!??!??!
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.ACCEPTED_FACILITY_LEVEL = facility_level
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 7})
        # assert not self.is_followup, 'A Follow-up appointment cannot be an in-patient appointment.' todo - this removeed? bring back!?

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

        # TODO: BUG - if refer to be inpatient will it still be re-assessed in the
        #  HSI apply function with _assess_and_treat ??
        #  This would not be correct as then they re registered as inpatient but having outpatient treatments given.
        #  I think I have fixed it now

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
        # todo - make inpatient?

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
        return 'Inpatient' in self.TREATMENT_ID

    def _get_cons(self, _item_str: str) -> bool:
        """True if all of a group of consumables (identified by a string) is available, (if no group is
        provided, then raise ValueError)"""
        if _item_str is not None:
            return self.get_consumables(
                item_codes={
                    k: v(self._age_exact_years) if isinstance(v, types.LambdaType) else v
                    for k, v in self.module.consumables_used_in_hsi[_item_str].items()
                })
        else:
            raise ValueError(f'Consumable pack not provided')

    def _get_any_cons(self, _item_str: str) -> bool:
        """True if any of a group of consumables (identified by a string) is available, (if no group is
        provided, then raise ValueError)"""
        # todo - could accept list of stings, to accomodate the availability of other antibiotics
        if _item_str is not None:
            return any(self.get_consumables(
                item_codes={
                    k: v(self._age_exact_years) if isinstance(v, types.LambdaType) else v
                    for k, v in self.module.consumables_used_in_hsi[_item_str].items()
                },
                return_individual_results=True
            ).values())
        else:
            raise ValueError(f'Consumable pack not provided')

    def _assess_and_treat(self, age_exact_years, oxygen_saturation, symptoms):
        """This routine is called in every HSI. It classifies the disease of the child and commissions treatment
        accordingly."""

        use_oximeter = self._get_cons('Pulse_oximetry')

        # Assessment process for first appointments (all start as outpatient) or outpatient referrals
        if not self._is_as_in_patient and not self.is_followup:

            classification_for_treatment_decision = self._get_disease_classification_for_treatment_decision(
                age_exact_years=age_exact_years,
                symptoms=symptoms,
                oxygen_saturation=oxygen_saturation,
                facility_level=self.ACCEPTED_FACILITY_LEVEL,
                use_oximeter=use_oximeter,
            )

            self._provide_bronchodilator_if_wheeze(
                facility_level=self.ACCEPTED_FACILITY_LEVEL,
                symptoms=symptoms,
            )

            self._do_action_given_classification(
                classification_for_treatment_decision=classification_for_treatment_decision,
                age_exact_years=age_exact_years,
                facility_level=self.ACCEPTED_FACILITY_LEVEL,
            )

        # assessment process for follow-ups with treatment failure or urgent referrals (both inpatient)
        # todo - seem that Ines' new logic that says assume its danger_signs_pneumonia if there person is already an in-patient?????
        elif self._is_as_in_patient:
            self._do_action_given_classification(
                classification_for_treatment_decision='danger_signs_pneumonia',  # assumed for sbi for < 2 months
                age_exact_years=age_exact_years,
                facility_level=self.ACCEPTED_FACILITY_LEVEL,
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
             'fast_breathing_pneumonia',
             'danger_signs_pneumonia',
             'chest_indrawing_pneumonia,
             'cough_or_cold'
        }."""
        # todo --- what is this indirection for!?!?!?
        return self.module.get_imci_classification_based_on_symptoms(
            child_is_younger_than_2_months=child_is_younger_than_2_months, symptoms=symptoms)

    @staticmethod
    def _get_imci_classification_by_SpO2_measure(oxygen_saturation: bool) -> str:
        """Return classification based on age and oxygen_saturation. It will return one of: {
             'danger_signs_pneumonia',          <-- implies needs oxygen
             ''                                 <-- implies does not need oxygen
        }."""

        if oxygen_saturation == '<90%':
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
        # todo logic could be simplified a bit here (collecting up all the else)
        rand = self.module.rng.random_sample
        rand_choice = self.module.rng.choice
        p = self.module.parameters

        if facility_level == '0':
            if imci_classification_based_on_symptoms == 'fast_breathing_pneumonia':
                if rand() < p['sensitivity_of_classification_of_fast_breathing_pneumonia_facility_level0']:
                    return imci_classification_based_on_symptoms
                else:
                    return 'cough_or_cold'

            elif imci_classification_based_on_symptoms in ('chest_indrawing_pneumonia', 'danger_signs_pneumonia'):
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
                # (Perfect diagnosis accuracy for 'cough_or_cold')
                # serious bacterial infections (under danger_signs_pneumonia) not determined at level 0 -
                # need to refer to facility < 2 months old
                return imci_classification_based_on_symptoms

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
                assert imci_classification_based_on_symptoms == 'cough_or_cold'
                # (Assume perfect diagnosis accuracy for 'cough_or_cold')
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
                # (Perfect diagnosis accuracy for 'cough_or_cold')
                return imci_classification_based_on_symptoms

    def _get_disease_classification_for_treatment_decision(self,
                                                           age_exact_years,
                                                           symptoms,
                                                           oxygen_saturation,
                                                           facility_level,
                                                           use_oximeter,
                                                           ) -> str:
        """Returns the classification of disease for the purpose of treatment, which may be based on the results of the
         pulse oximetry (if available) or the health worker's own classification. It will be one of: {
                 'danger_signs_pneumonia',          (symptoms-based assessment OR spO2 assessment): implies need oxygen
                 'fast_breathing_pneumonia',        (symptoms-based assessment)
                 'chest_indrawing_pneumonia',       (symptoms-based assessment)
                 'cough_or_cold'                    (symptoms-based assessment)
         }."""

        child_is_younger_than_2_months = age_exact_years < (2.0 / 12.0)

        imci_classification_based_on_symptoms = self._get_imci_classification_based_on_symptoms(
            child_is_younger_than_2_months=child_is_younger_than_2_months,
            symptoms=symptoms)

        imci_classification_by_SpO2_measure = self._get_imci_classification_by_SpO2_measure(
            oxygen_saturation=oxygen_saturation)

        hw_assigned_classification = self._get_classification_given_by_health_worker(
            imci_classification_based_on_symptoms=imci_classification_based_on_symptoms,
            facility_level=facility_level)

        # Final classification gives precedence to the spO2 measure if it can be used:
        _classification = imci_classification_by_SpO2_measure \
            if (use_oximeter and (imci_classification_by_SpO2_measure != '')) else hw_assigned_classification

        assert _classification in ('danger_signs_pneumonia', 'fast_breathing_pneumonia', 'chest_indrawing_pneumonia',
                                   'cough_or_cold')

        return _classification

    @staticmethod
    def _ultimate_treatment_indicated_for_patient(classification_for_treatment_decision, age_exact_years, ):
        """Return a Dict of the form {'antibiotic_indicated': <>, 'oxygen_indicated': <>} which expresses what the
         treatment is that the patient should be provided with ideally and ultimately, if consumables are available (and
         following an admission to in-patient, if required)."""
        # todo assert that classification_for_treatment_decision is in the set of allowable valyes

        if classification_for_treatment_decision == 'fast_breathing_pneumonia':
            return {
                'antibiotic_indicated': (
                    'Amoxicillin_tablet_or_suspension_7days' if age_exact_years < 2.0 / 12.0
                    else 'Amoxicillin_tablet_or_suspension_3days'),
                'oxygen_indicated': False
            }

        elif classification_for_treatment_decision == 'chest_indrawing_pneumonia':
            return {
                'antibiotic_indicated': 'Amoxicillin_tablet_or_suspension_5days',
                'oxygen_indicated': False
            }

        elif classification_for_treatment_decision == 'danger_signs_pneumonia':
            return {
                'antibiotic_indicated': '1st_line_IV_antibiotics',
                'oxygen_indicated': True
            }

        elif classification_for_treatment_decision == "cough_or_cold":
            return {
                'antibiotic_indicated': '',
                'oxygen_indicated': False
            }

    def _do_action_given_classification(self,
                                        classification_for_treatment_decision,
                                        age_exact_years,
                                        facility_level,
                                        ):
        """Do the actions that are required given a particular classification and the current facility level. This
        entails referrals upwards and/or admission at in-patient, and when at the appropriate level, trying to provide
        the ideal treatment."""

        def _provide_consumable_and_refer(cons: str) -> None:
            """Provide a consumable (ignoring availability) and refer patient to next level up."""
            if cons is not None:
                _ = self._get_cons(cons)
            self._refer_to_next_level_up()

        def _try_treatment(antibiotic_indicated: str, oxygen_indicated: bool) -> None:
            """Try to provide a `treatment_indicated` and refer to next level if the consumables are not available."""

            antibiotic_available = self._get_any_cons(antibiotic_indicated)
            oxygen_available = self._get_cons('Oxygen_Therapy')
            oxygen_indicated_and_available_or_oxygen_not_indicated = (oxygen_available and oxygen_indicated) or \
                                                                     (not oxygen_indicated)

            if antibiotic_available and oxygen_indicated_and_available_or_oxygen_not_indicated:
                # Everything indicated is available --> do effects of this treatment
                self.module.do_effects_of_treatment(
                    person_id=self.target,
                    antibiotic_provided=antibiotic_indicated,
                    oxygen_provided=(oxygen_available and oxygen_indicated)
                )
                # # todo -- what is the logic she wants!?!??!?!?
                # if treatment_failed and not self.is_followup:  # only 1 possible follow-up appt for an episode
                #     self._schedule_follow_up_at_same_facility_as_outpatient()
            elif facility_level != '2':
                # Something that is needed is not available -> Refer up if possible.
                self._refer_to_next_level_up()
            else:
                # Something is not available, but no referral up is possible. Do Treatment with what is available.
                self.module.do_effects_of_treatment(
                    person_id=self.target,
                    antibiotic_provided=antibiotic_indicated if antibiotic_available else '',
                    oxygen_provided=(oxygen_available and oxygen_indicated)
                )
                # if treatment_failed and not self.is_followup:
                #     self._schedule_follow_up_at_same_facility_as_outpatient()

        def _do_if_fast_breathing_pneumonia():
            """What to do if classification is `fast_breathing`."""
            _try_treatment(
                **self._ultimate_treatment_indicated_for_patient(
                    classification_for_treatment_decision='fast_breathing_pneumonia',
                    age_exact_years=age_exact_years,
                )
            )

        def _do_if_chest_indrawing_pneumonia():
            """What to do if classification is `chest_indrawing_pneumonia`."""
            if facility_level == '0':
                _provide_consumable_and_refer('First_dose_oral_amoxicillin_for_referral')
            else:
                _try_treatment(
                    **self._ultimate_treatment_indicated_for_patient(
                        classification_for_treatment_decision='chest_indrawing_pneumonia',
                        age_exact_years=age_exact_years,
                    )
                )

        def _do_if_danger_signs_pneumonia():
            """What to do if classification is `danger_signs_pneumonia."""

            if facility_level == '0':
                _provide_consumable_and_refer('First_dose_oral_amoxicillin_for_referral')
            else:
                if not self._is_as_in_patient:
                    _ = self._get_cons('First_dose_IM_antibiotics_for_referral')
                    self._refer_to_become_inpatient()

                else:
                    _try_treatment(
                        **self._ultimate_treatment_indicated_for_patient(
                            classification_for_treatment_decision='danger_signs_pneumonia',
                            age_exact_years=age_exact_years,
                        )
                    )

        def _do_if_cough_or_cold():
            """What to do if `cough_or_cold`. (Nothing!) """
            pass

        # Do the appropriate action
        {
            'fast_breathing_pneumonia': _do_if_fast_breathing_pneumonia,
            'chest_indrawing_pneumonia': _do_if_chest_indrawing_pneumonia,
            'danger_signs_pneumonia': _do_if_danger_signs_pneumonia,
            'cough_or_cold': _do_if_cough_or_cold
        }[classification_for_treatment_decision]()

    def _provide_bronchodilator_if_wheeze(self, facility_level, symptoms):
        """Provide bronchodilator if wheeze is among the symptoms"""
        if 'wheeze' in symptoms:
            if facility_level == '1a':
                _ = self._get_cons('Inhaled_Brochodilator')
            else:
                _ = self._get_cons('Brochodilator_and_Steroids')

    def provide_2nd_line_antibiotics(self, antibiotic_provided, first_line_failed):
        """Provide 2nd line of antibiotics when the 1st line are not working"""
        # todo !?!?!?!?!?!?!!?!!!?!!? Get rid of this :-)
        if first_line_failed and (antibiotic_provided == '1st_line_IV_antibiotics'):
            if self._has_staph_aureus():
                return self._get_cons('2nd_line_Antibiotic_therapy_for_severe_staph_pneumonia')
            else:
                return self._get_cons('Ceftriaxone_therapy_for_severe_pneumonia')

    def apply(self, person_id, squeeze_factor):
        """Assess and attempt to treat the person."""

        # refer follow-up appointments to inpatient care (if follow-up only applies to those with TF)
        # todo- tim - is the logic now that follow-up is only in the case of treatment failure!!!??
        if self.is_followup and not self._is_as_in_patient:
            self._refer_to_become_inpatient()

        # Do nothing if the person is not currently infected and currently experiencing an episode
        person = self.sim.population.props.loc[person_id]
        if not person.ri_current_infection_status and (
            person.ri_start_of_current_episode <= self.sim.date <= person.ri_end_of_current_episode
        ):
            return

        # Do nothing if the persons does not have indicating symptoms
        symptoms = self.sim.modules['SymptomManager'].has_what(person_id)
        if not {'cough', 'difficult_breathing'}.intersection(symptoms):
            return self.make_appt_footprint({})

        # Do nothing if the person is already on treatment
        if person.ri_on_treatment:
            return self.make_appt_footprint({})

        # If the HSI is at level 0 and is for a child aged less than 2 months, refer to the next level.
        if (self.ACCEPTED_FACILITY_LEVEL == '0') and (person.age_exact_years < 2.0 / 12.0):
            self._refer_to_next_level_up()

        # Attempt treatment:
        self._assess_and_treat(age_exact_years=person.age_exact_years,
                               symptoms=symptoms,
                               oxygen_saturation=person.ri_SpO2_level,
                               )

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
        self.trackers['deaths_due_to_untreated_hypoxaemia'] = Tracker()
        self.trackers['seeking_care'] = Tracker()
        self.trackers['treated'] = Tracker()
        self.trackers['pulmonary_complication_cases'] = Tracker()
        self.trackers['systemic_complication_cases'] = Tracker()
        self.trackers['hypoxaemic_cases'] = Tracker()

    def new_case(self, **kwargs):
        self.trackers['incident_cases'].add_one(**kwargs)

    def new_recovered_case(self, **kwargs):
        self.trackers['recovered_cases'].add_one(**kwargs)

    def new_cured_case(self, **kwargs):
        self.trackers['cured_cases'].add_one(**kwargs)

    def new_death(self, **kwargs):
        self.trackers['deaths'].add_one(age=kwargs['age'], pathogen=kwargs['pathogen'])
        if kwargs['sp02_level'] == '90-92%':
            self.trackers['deaths_due_to_untreated_hypoxaemia'].add_one()

    def new_seeking_care(self, **kwargs):
        self.trackers['seeking_care'].add_one(**kwargs)

    def new_treated(self, **kwargs):
        self.trackers['treated'].add_one(**kwargs)

    def new_pulmonary_complication_case(self, **kwargs):
        self.trackers['pulmonary_complication_cases'].add_one(**kwargs)

    def new_systemic_complication_case(self, **kwargs):
        self.trackers['systemic_complication_cases'].add_one(**kwargs)

    def new_hypoxaemic_case(self, **kwargs):
        self.trackers['hypoxaemic_cases'].add_one(**kwargs)

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
        """Create and initialise tracker"""

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


class AlriIncidentCase_Lethal_DangerSigns_Pneumonia(AlriIncidentCase):
    """This Event can be used for testing and is a drop-in replacement of `AlriIncidentCase`. It always produces an
    infection that will be lethal and should be classified as 'danger_signs_pneumonia'"""

    def __init__(self, module, person_id, pathogen):
        super().__init__(module, person_id=person_id, pathogen=pathogen)

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters

        self.apply_characteristics_of_the_case(
            person_id=person_id,
            chars={
                'disease_type': 'pneumonia',
                'bacterial_coinfection': np.nan,
                'duration_in_days_of_alri': 10,
                'date_of_outcome': self.module.sim.date + DateOffset(days=10),
                'episode_end': self.module.sim.date + DateOffset(days=10 + params['days_between_treatment_and_cure']),
                'symptoms': ['cough', 'chest_indrawing', 'danger_signs'],
                'complications': ['hypoxaemia'],
                'oxygen_saturation': "<90%",
                'will_die': True,
            }
        )

        assert 'danger_signs_pneumonia' == self.module.get_imci_classification_based_on_symptoms(
            child_is_younger_than_2_months=df.at[person_id, 'age_exact_years'] < (2.0 / 12.0),
            symptoms=self.sim.modules['SymptomManager'].has_what(person_id)
        )


class AlriIncidentCase_NonLethal_Fast_Breathing_Pneumonia(AlriIncidentCase):
    """This Event can be used for testing and is a drop-in replacement of `AlriIncidentCase`. It always produces an
    infection that will be non-lethal and should be classified as a fast_breathing_pneumonia."""

    def __init__(self, module, person_id, pathogen):
        super().__init__(module, person_id=person_id, pathogen=pathogen)

    def apply(self, person_id):
        params = self.module.parameters

        self.apply_characteristics_of_the_case(
            person_id=person_id,
            chars={
                'disease_type': 'pneumonia',
                'bacterial_coinfection': np.nan,
                'duration_in_days_of_alri': 10,
                'date_of_outcome': self.module.sim.date + DateOffset(days=10),
                'episode_end': self.module.sim.date + DateOffset(days=10 + params['days_between_treatment_and_cure']),
                'symptoms': ['cough', 'tachypnoea'],
                'complications': ['hypoxaemia'],
                'oxygen_saturation': ">=93%",
                'will_die': False,
            }
        )

        assert 'fast_breathing_pneumonia' == \
               self.module.get_imci_classification_based_on_symptoms(
                   child_is_younger_than_2_months=False, symptoms=self.sim.modules['SymptomManager'].has_what(person_id)
               )


def _make_hw_diagnosis_perfect(alri_module):
    """Modify the parameters of an instance of the Alri module so that sensitivity of all diagnosis steps is perfect."""
    p = alri_module.parameters

    p['sensitivity_of_classification_of_fast_breathing_pneumonia_facility_level0'] = 1.0
    p['sensitivity_of_classification_of_danger_signs_pneumonia_facility_level0'] = 1.0
    p['sensitivity_of_classification_of_non_severe_pneumonia_facility_level1'] = 1.0
    p['sensitivity_of_classification_of_severe_pneumonia_facility_level1'] = 1.0
    p['sensitivity_of_classification_of_non_severe_pneumonia_facility_level2'] = 1.0
    p['sensitivity_of_classification_of_severe_pneumonia_facility_level2'] = 1.0


def _make_treatment_and_diagnosis_perfect(alri_module):
    """Modify the parameters of an instance of the Alri module so that treatment and diagnosis is perfect."""

    _make_hw_diagnosis_perfect(alri_module)

    p = alri_module.parameters

    # The probability of treatment failure to be 0.0
    p['tf_1st_line_antibiotic_for_severe_pneumonia'] = 0.0
    p['tf_2nd_line_antibiotic_for_severe_pneumonia'] = 0.0

    p['tf_3day_amoxicillin_for_fast_breathing_with_SpO2>=90%'] = 0.0

    p['tf_3day_amoxicillin_for_chest_indrawing_with_SpO2>=90%'] = 0.0
    p['tf_5day_amoxicillin_for_chest_indrawing_with_SpO2>=90%'] = 0.0

    p['tf_7day_amoxicillin_for_fast_breathing_pneumonia_in_young_infants'] = 0.0

    p['tf_oral_amoxicillin_only_for_severe_pneumonia_with_SpO2>=90%'] = 0.0
    p['tf_oral_amoxicillin_only_for_severe_pneumonia_with_SpO2<90%'] = 0.0

    p['tf_oral_amoxicillin_only_for_non_severe_pneumonia_with_SpO2<90%'] = 0.0


def _make_high_risk_of_death(alri_module):
    """Modify the parameters of an instance of the Alri module so that the risk of death is high."""
    params = alri_module.parameters
    params['base_odds_death_ALRI_age<2mo'] *= 5.0
    params['base_odds_death_ALRI_age2_59mo'] *= 5.0
