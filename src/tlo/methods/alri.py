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
pneumothorax; and/or systemic complications: bacteraemia; and/or complications regarding oxygen exchange: hypoxaemia.
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
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from tlo import DAYS_IN_YEAR, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.util import random_date, sample_outcome

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Helper function for conversion between odds and probabilities
to_odds = lambda pr: pr / (1.0 - pr)  # noqa: E731
to_prob = lambda odds: odds / (1.0 + odds)  # noqa: E731

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

    classifications = sorted({
        'danger_signs_pneumonia',
        'fast_breathing_pneumonia',
        'chest_indrawing_pneumonia',
        'cough_or_cold'
    })

    all_symptoms = {
        'cough', 'difficult_breathing', 'fever', 'tachypnoea', 'chest_indrawing', 'danger_signs', 'respiratory_distress'
    }

    # Declare the Alri complications:
    complications = sorted({
        'pneumothorax',
        'pleural_effusion',
        'empyema',
        'lung_abscess',
        'bacteraemia',
        'hypoxaemia'
    })

    # Declare the antibiotics that can be used
    antibiotics = sorted({
        'Amoxicillin_tablet_or_suspension_3days',
        'Amoxicillin_tablet_or_suspension_5days',
        'Amoxicillin_tablet_or_suspension_7days',
        '1st_line_IV_ampicillin_gentamicin',
        '1st_line_IV_benzylpenicillin_gentamicin',
        '2nd_line_IV_ceftriaxone',
        '2nd_line_IV_flucloxacillin_gentamicin',
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
        'rr_ALRI_wasting':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri children with WHZ<-2'
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
        'prev_hypoxaemia_in_alri':
            Parameter(Types.REAL,
                      'prevalence of hypoxaemia in all ALRI'
                      ),

        'or_hypoxaemia_in_abnormal_CXR':
            Parameter(Types.REAL,
                      'Odds ratio of hypoxaemia in CXR+ compared to ref CXR-'
                      ),
        'or_hypoxaemia_in_pc_pneumonia':
            Parameter(Types.REAL,
                      'Odds ratio of hypoxaemia in in pulmonary complicated pneumonia compared to non-pc pneumonia'
                      ),
        'assumed_prev_hypoxaemia_in_normal_CXR':
            Parameter(Types.REAL,
                      'assumed prevalence of hypoxaemia in CXR- / captured in other ALRI group'
                      ),
        'prev_bacteraemia_in_alri':
            Parameter(Types.REAL,
                      'prevalence of bacteraemia in all ALRI'
                      ),
        'assumed_prev_bacteraemia_in_normal_CXR':
            Parameter(Types.REAL,
                      'assumed prevalence of bacteraemia in CXR- / captured in other ALRI group'
                      ),
        'or_bacteraemia_in_abnormal_CXR':
            Parameter(Types.REAL,
                      'odds ratio of bacteraemia in CXR+ / captured in pneumonia group vs CXR- (ref)'
                      ),
        'assumed_prev_bacteraemia_in_non_pc_pneumonia':
            Parameter(Types.REAL,
                      'assumed prevalence of bacteraemia in non-pulmonary complicated pneumonia'
                      ),
        'assumed_prev_hypoxaemia_in_non_pc_pneumonia':
            Parameter(Types.REAL,
                      'assumed prevalence of hypoxaemia in non-pulmonary complicated pneumonia'
                      ),
        'or_bacteraemia_in_pc_pneumonia':
            Parameter(Types.REAL,
                      'odds ratio bactereamia if already complicated with pulmonary complications'
                      ),
        'proportion_hypoxaemia_with_SpO2<90%':
            Parameter(Types.REAL,
                      'proportion of hypoxaemic children with SpO2 <90%'
                      ),

        # Risk of death parameters -----
        'scaler_on_risk_of_death':
            Parameter(Types.REAL,
                      "Multiplicative scaler on the overall risk of death, for the purpose of calibration."
                      ),
        'base_odds_death_ALRI_age<2mo':
            Parameter(Types.REAL,
                      'baseline odds of death from ALRI for young infants aged 0 month and severe pneumonia '
                      '(base group)'
                      ),
        'or_death_ALRI_age<2mo_danger_signs':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for young infants with any danger signs'
                      ),
        'or_death_ALRI_age<2mo_SpO2<90%':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for young infants with SpO2<90%'
                      ),
        'or_death_ALRI_age<2mo_SpO2_90_92%':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for young infants with SpO2 between 90-92%'
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
        'or_death_ALRI_age2_59mo_chest_indrawing':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children with chest wall indrawing'
                      ),
        'or_death_ALRI_age2_59mo_danger_signs':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children with any danger signs'
                      ),
        'or_death_ALRI_age2_59mo_in_2_5mo':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for 2-5mo (12-59 ref) for 2 to 59 months olds'
                      ),
        'or_death_ALRI_age2_59mo_in_6_11mo':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for 6-11mo (12-59 ref) for 2 to 59 months olds'
                      ),
        'or_death_ALRI_age2_59mo_SAM':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children with severe acute malnutrition'
                      ),
        'or_death_ALRI_age2_59mo_-3<=WHZ<-2':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children with moderate wasting, -3<=WHZ<-2'
                      ),
        'or_death_ALRI_age2_59mo_WHZ<-3':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children with severe wasting, WHZ<-3'
                      ),
        'or_death_ALRI_age2_59mo_SpO2<90%':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children aged 2 to 59 months with oxygen saturation <90%, '
                      'base group: SpO2 <=93%'
                      ),
        'or_death_ALRI_age2_59mo_SpO2_90_92%':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children aged 2 to 59 months with '
                      'oxygen saturation between 90 to 92%, base group: SpO2 <=93%'
                      ),
        'or_death_ALRI_respiratory_distress':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children with respiratory distress'
                      ),
        'or_death_ALRI_abnormal_CXR':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children with abnormal CXR (pneumonia disease type)'
                      ),
        'or_death_ALRI_pulmonary_complications':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children with pulmonary complications'
                      ),
        'or_death_ALRI_bacteraemia':
            Parameter(Types.REAL,
                      'odds ratio of death from ALRI for children with bacteraemia'
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
        'prob_danger_signs_in_pulmonary_complications':
            Parameter(Types.REAL,
                      'probability of danger signs in pneumonia with pulmonary complications'
                      ),
        'prob_chest_indrawing_in_pulmonary_complications':
            Parameter(Types.REAL,
                      'probability of chest indrawing in pneumonia with pulmonary complications'
                      ),
        'or_severe_symptoms_in_severe_pulmonary_complications':
            Parameter(Types.REAL,
                      'increase odds ratio of severe symptoms in severe pulmonary complicated pneumonia'
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
        'tf_1st_line_antibiotic_for_severe_pneumonia':
            Parameter(Types.REAL,
                      'Risk of treatment failure for a person with danger_signs_pneumonia being treated with first line'
                      'intravenous antibiotics'
                      ),
        'rr_tf_1st_line_antibiotics_if_general_danger_signs':
            Parameter(Types.REAL,
                      'Relative Risk for treatment failure for persons with danger signs pneumonia being treated with '
                      'first line intravenous antibiotics if the person has any general danger signs'
                      ),
        'rr_tf_1st_line_antibiotics_if_respiratory_distress':
            Parameter(Types.REAL,
                      'Relative Risk for treatment failure for persons with danger signs pneumonia being treated with '
                      'first line intravenous antibiotics if the person has respiratory distress'
                      ),
        'rr_tf_1st_line_antibiotics_if_SpO2<90%':
            Parameter(Types.REAL,
                      'Relative Risk for treatment failure for persons with danger_signs_pneumonia being treated with '
                      'first line intravenous antibiotics if the person has oxygen saturation < 90%.'
                      ),
        'rr_tf_1st_line_antibiotics_if_abnormal_CXR':
            Parameter(Types.REAL,
                      'Relative Risk for treatment failure for persons with danger_signs_pneumonia being treated with '
                      'first line intravenous antibiotics if the person has a disease_type of "pneumonia".'
                      ),
        'rr_tf_1st_line_antibiotics_if_MAM':
            Parameter(Types.REAL,
                      'Relative Risk for treatment failure for persons with danger_signs_pneumonia being treated with '
                      'first line intravenous antibiotics if the person has un_clinical_acute_malnutrition == "MAM".'
                      ),
        'rr_tf_1st_line_antibiotics_if_SAM':
            Parameter(Types.REAL,
                      'Relative Risk for treatment failure for persons with danger_signs_pneumonia being treated with '
                      'first line intravenous antibiotics if the person has un_clinical_acute_malnutrition == "SAM".'
                      ),
        'rr_tf_1st_line_antibiotics_if_HIV/AIDS':
            Parameter(Types.REAL,
                      'Relative Risk for treatment failure for persons with danger_signs_pneumonia being treated with '
                      'first line intravenous antibiotics if the person has HIV and is not currently being treated.'
                      ),
        'or_mortality_improved_oxygen_systems':
            Parameter(Types.REAL,
                      'Odds Ratio for the effect of oxygen provision to a person that needs oxygen who receives it, '
                      'compares to a patient who does not. N.B. The inverse of this is used to reflect the increase in '
                      'odds of death for a patient that needs oxygen but does not receive it.'
                      ),
        'or_tf_oral_antibiotics_if_SpO2_90_92%':
            Parameter(Types.REAL,
                      'Odds Ratio for the effect of oxygen provision to a person with SpO2 90-92% '
                      'compares to a patient who does not. N.B. The inverse of this is used to reflect the increase in '
                      'odds of death for a patient that needs oxygen but does not receive it.'
                      ),
        'tf_3day_amoxicillin_for_fast_breathing_with_SpO2>=90%':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 6 or relapse by day 14 of 3-day course amoxicillin for'
                      ' treating fast-breathing pneumonia'
                      ),
        'tf_3day_amoxicillin_for_chest_indrawing_with_SpO2>=90%':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 6 or relapse by day 14 of 3-day course amoxicillin for '
                      'treating chest-indrawing pneumonia without hypoxaemia (SpO2>=90%)'
                      ),
        'tf_5day_amoxicillin_for_chest_indrawing_with_SpO2>=90%':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 6 or relapse by day 14 of 5-day course amoxicillin for '
                      'treating chest-indrawing pneumonia without hypoxaemia (SpO2>=90%)'
                      ),
        'tf_7day_amoxicillin_for_fast_breathing_pneumonia_in_young_infants':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 6 or relapse by day 14 of 5-day course amoxicillin for '
                      'treating chest-indrawing pneumonia without hypoxaemia (SpO2>=90%)'
                      ),
        'tf_2nd_line_antibiotic_for_severe_pneumonia':
            Parameter(Types.REAL,
                      'probability of treatment failure by end of IV therapy for 2nd line antibiotic either cloxacillin'
                      ' or ceftriaxone to treat severe pneumonia (danger-signs)'
                      ),
        'rr_tf_if_given_1st_line_IV_antibiotics_for_pneumonia_with_SpO2<90%':
            Parameter(Types.REAL,
                      'relative risk of treatment failure if pneumonia with SpO2<90% were given 1st line IV antibiotics'
                      'compared to oral antibiotics'
                      ),
        'rr_tf_if_given_2nd_line_IV_antibiotics_for_pneumonia_with_SpO2<90%':
            Parameter(Types.REAL,
                      'relative risk of treatment failure if pneumonia with SpO2<90% were given 2nd line IV antibiotics'
                      'compared to oral antibiotics'
                      ),
        'rr_tf_if_given_1st_line_IV_antibiotics_for_pneumonia_with_SpO2>=90%':
            Parameter(Types.REAL,
                      'relative risk of treatment failure if pneumonia with SpO2>=90% were given '
                      '1st line IV antibiotics compared to oral antibiotics'
                      ),
        'rr_tf_if_given_2nd_line_IV_antibiotics_for_pneumonia_with_SpO2>=90%':
            Parameter(Types.REAL,
                      'relative risk of treatment failure if pneumonia with SpO2>=90% were given '
                      '2nd line IV antibiotics compared to oral antibiotics'
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

        'or_care_seeking_if_perceived_severe':
            Parameter(Types.REAL,
                      'The Odds Ratio for healthcare seeking if perceived severe'
                      ),

        'pulse_oximeter_and_oxygen_is_available':
            Parameter(Types.CATEGORICAL,
                      'Control the availability of the pulse oximeter and oxygen. "Default" does not over-ride '
                      'availability; "Yes" forces them to be available; "No" forces them to not be available',
                      categories=['Yes', 'No', 'Default']
                      ),
        'apply_oxygen_indication_to_SpO2_measurement':
            Parameter(Types.CATEGORICAL,
                      'SpO2 measurement level for oxygen indication '
                      '<90% (current policy) or 90-92% (potential policy)',
                      categories=['<90%', '<93%']
                      ),
        'use_static_simulation_run':
            Parameter(Types.BOOL,
                      'Turn on/off the parameter to determine the use of static simulation run for analysis'
                      'The default is True (1)'
                      ),
        # 'allow_use_oximetry_for_non_severe_classifications':
        #     Parameter(Types.BOOL,
        #               'Turn on/off the use of pulse oximetry on non-severe classifications in assigning treatment '
        #               'to determine the SpO2 level for oxygen indication. The default is False (0)'
        #               ),
        'prob_hw_decision_for_oxygen_provision_when_po_unavailable':
            Parameter(Types.REAL,
                      'sensitivity of health worker decision in oxygen provision for danger signs penumonia '
                      'without the use of pulse oximeter'
                      ),
        'or_tf_oral_antibiotics_if_SpO2<90%':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'or_tf_oral_antibiotics_if_MAM':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'or_tf_oral_antibiotics_if_SAM':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'or_tf_oral_antibiotics_if_concurrent_malaria':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'rr_tf_if_given_parenteral_antibiotics_for_pneumonia_with_SpO2<90%':
            Parameter(Types.REAL,
                      'tmp param'
                      ),

        'rr_tf_oral_antibiotics_if_danger_signs':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'rr_tf_oral_antibiotics_if_repiratory_distress':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'rr_tf_if_given_parenteral_antibiotics_for_pneumonia_with_SpO2>=90%':
            Parameter(Types.REAL,
                      'relative risk of treatment failure if non-severe pneumonia with SpO2>90% '
                      'were given parenteral antibiotics'
                      ),
        'prob_respiratory_distress_in_pneumonia':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'prob_respiratory_distress_in_other_alri':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'prob_respiratory_distress_in_SpO2<90%':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'prob_respiratory_distress_in_SpO2_90-92%':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'prob_respiratory_distress_in_pulmonary_complications':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'or_danger_signs_in_alri_with_respiratory_distress':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'prob_danger_signs_in_no_respiratory_distress_SpO2>=93%':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'or_respiratory_distress_in_alri_with_chest_indrawing':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'prob_respiratory_distress_in_no_chest_indrawing_SpO2>=93%':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'prob_danger_signs_in_no_respiratory_distress_SpO2<90%':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'prob_respiratory_distress_in_no_chest_indrawing_SpO2<90%':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'prob_danger_signs_in_no_respiratory_distress_SpO2_90-92%':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'prob_respiratory_distress_in_no_chest_indrawing_SpO2_90-92%':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'or_fever_in_complicated_alri':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'or_tachypnoea_in_complicated_alri':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'prob_respiratory_distress_in_no_chest_indrawing_pc':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'prob_danger_signs_in_no_respiratory_distress_pc':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'base_prob_care_seeking_by_level':
            Parameter(Types.LIST,
                      'tmp param'
                      ),
        'or_care_seeking_hospital_age<2mo':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'or_care_seeking_hospital_age2_11mo':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'or_care_seeking_hospital_age12_23mo':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'or_care_seeking_hospital_danger_signs':
            Parameter(Types.REAL,
                      'tmp param'
                      ),
        'or_care_seeking_hospital_respiratory_distress':
            Parameter(Types.REAL,
                      'tmp param'
                      ),

        # parameters values derived from the model output
        'proportion_pneumonia_in_alri':
            Parameter(Types.REAL,
                      'proportion of pneumonia in the ALRI cases (pneumonia/other_alri)'
                      ),
        'proportion_bacterial_infection_in_pneumonia':
            Parameter(Types.REAL,
                      'proportion of cases with a primary or secondary bacterial infection in pneumonia group'
                      ),
        'proportion_bacterial_infection_in_other_alri':
            Parameter(Types.REAL,
                      'proportion of cases with a primary or secondary bacterial infection in other ALRI group'
                      ),

        # parameters for the CEA Lancet commission scenarios
        'scenario_existing_psa_ox_coverage_by_facility':
            Parameter(Types.LIST,
                      'Coverages of oxygen across the facility levels [2, 1b, 1a] for existing PSA scenario'
                      ),
        'scenario_planned_psa_ox_coverage_by_facility':
            Parameter(Types.LIST,
                      'Coverages of oxygen across the facility levels [2, 1b, 1a] for planned PSA scenario'
                      ),
        'scenario_all_district_psa_ox_coverage_by_facility':
            Parameter(Types.LIST,
                      'Coverages of oxygen across the facility levels [2, 1b, 1a] for all district PSA scenario'
                      ),
        'scenario_baseline_ant_ox_coverage_by_facility':
            Parameter(Types.LIST,
                      'Coverages of oxygen across the facility levels [2, 1b, 1a] for baseline scenario'
                      ),
        'scenario_existing_psa_po_coverage_by_facility':
            Parameter(Types.LIST,
                      'Coverages of PO across the facility levels [2, 1b, 1a] for existing PSA scenario'
                      ),
        'scenario_planned_psa_po_coverage_by_facility':
            Parameter(Types.LIST,
                      'Coverages of PO across the facility levels [2, 1b, 1a] for planned PSA scenario'
                      ),
        'scenario_all_district_psa_po_coverage_by_facility':
            Parameter(Types.LIST,
                      'Coverages of PO across the facility levels [2, 1b, 1a] for all district PSA scenario'
                      ),

        'oxygen_unit_cost_by_po_implementation_existing_psa_perfect_hw_dx':
            Parameter(Types.LIST,
                      'Oxygen unit cost by PO implementation in existing PSA scenario under perfect conditions: '
                      'None, level 2, level 1b, level 1a, level 0'
                      ),
        'oxygen_unit_cost_by_po_implementation_planned_psa_perfect_hw_dx':
            Parameter(Types.LIST,
                      'Oxygen unit cost by PO implementation in planned PSA scenario under perfect conditions: '
                      'None, level 2, level 1b, level 1a, level 0'
                      ),
        'oxygen_unit_cost_by_po_implementation_all_district_psa_perfect_hw_dx':
            Parameter(Types.LIST,
                      'Oxygen unit cost by PO implementation in All district PSA scenario under perfect conditions: '
                      'None, level 2, level 1b, level 1a, level 0'
                      ),
        'oxygen_unit_cost_by_po_implementation_baseline_ant_perfect_hw_dx':
            Parameter(Types.LIST,
                      'Oxygen unit cost in baseline scenario of antibiotics only under perfect conditions'
                      ),
        'oxygen_unit_cost_by_po_implementation_existing_psa_imperfect_hw_dx':
            Parameter(Types.LIST,
                      'Oxygen unit cost by PO implementation in existing PSA scenario under imperfect conditions: '
                      'None, level 2, level 1b, level 1a, level 0'
                      ),
        'oxygen_unit_cost_by_po_implementation_planned_psa_imperfect_hw_dx':
            Parameter(Types.LIST,
                      'Oxygen unit cost by PO implementation in planned PSA scenario under imperfect conditions: '
                      'None, level 2, level 1b, level 1a, level 0'
                      ),
        'oxygen_unit_cost_by_po_implementation_all_district_psa_imperfect_hw_dx':
            Parameter(Types.LIST,
                      'Oxygen unit cost by PO implementation in All district PSA scenario under imperfect conditions: '
                      'None, level 2, level 1b, level 1a, level 0'
                      ),
        'oxygen_unit_cost_by_po_implementation_baseline_ant_imperfect_hw_dx':
            Parameter(Types.LIST,
                      'Oxygen unit cost in baseline scenario of antibiotics only under imperfect conditions'
                      ),

        'referral_rate_severe_cases_from_hc':
            Parameter(Types.REAL,
                      'Successfully referred % of severe pneumonia diagnosed in health centres'
                      ),
        'prop_facility_referred_to':
            Parameter(Types.LIST,
                      'Proportions of referred cases to respective facility levels 2, 1b'
                      ),
        'sought_follow_up_care':
            Parameter(Types.REAL,
                      'Sought follow-up care after oral treatment failure'
                      ),
        'pulse_oximeter_usage_rate':
            Parameter(Types.REAL,
                      'Health worker usage rate of pulse oximeter'
                      ),
        'rr_tf_follow_up_care_with_initial_incorrect_care':
            Parameter(Types.REAL,
                      'Relative risk of treatment failure at follow-up with initial care incorrect'
                      'prior to referral'
                      ),
        'or_tf_non_stabilised_with_oxygen_prior_to_referral':
            Parameter(Types.REAL,
                      'Odds ratio of treatment failure for non-stabilised hypoxaemia cases at the health centres '
                      'prior to referral'
                      ),
        'or_tf_at_follow_up_care_with_previous_oral_antibiotic':
            Parameter(Types.REAL,
                      'Odds ratio of treatment failure for severe cases with initial treatment with oral antibiotics'
                      ),

        'availability_amoxicillin_tablet_by_facility_level':
            Parameter(Types.LIST,
                      'Probability of availability of oral amoxicillin tablet by facility level [0, 1a, 1b, 2, 3]'
                      ),
        'availability_amoxicillin_suspension_by_facility_level':
            Parameter(Types.LIST,
                      'Probability of availability of oral amoxicillin suspension by facility level [0, 1a, 1b, 2, 3]'
                      ),
        'availability_gentamicin_by_facility_level':
            Parameter(Types.LIST,
                      'Probability of availability of gentamicin for injection by facility level [0, 1a, 1b, 2, 3]'
                      ),
        'availability_ampicillin_by_facility_level':
            Parameter(Types.LIST,
                      'Probability of availability of ampicillin for injection by facility level [0, 1a, 1b, 2, 3]'
                      ),
        'availability_benzylpenicillin_3g_by_facility_level':
            Parameter(Types.LIST,
                      'Probability of availability of benzylpenicillin 3g for injection by facility level '
                      '[0, 1a, 1b, 2, 3]'
                      ),
        'availability_benzylpenicillin_1g_by_facility_level':
            Parameter(Types.LIST,
                      'Probability of availability of benzylpenicillin 1g for injection by facility level '
                      '[0, 1a, 1b, 2, 3]'
                      ),
        'availability_ceftriaxone_by_facility_level':
            Parameter(Types.LIST,
                      'Probability of availability of ceftriaxone for injection by facility level [0, 1a, 1b, 2, 3]'
                      ),
        'availability_fluoxacillin_by_facility_level':
            Parameter(Types.LIST,
                      'Probability of availability of fluoxacillin for injection by facility level [0, 1a, 1b, 2, 3]'
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

        # store the treatment given at initial appointment
        self.store_treatment_info = dict()

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
                        Symptom.emergency(name=symptom_name, which='children')
                    )
                elif symptom_name in ('fast_breathing', 'chest_indrawing', 'respiratory_distress'):
                    self.sim.modules['SymptomManager'].register_symptom(
                        Symptom(name=symptom_name,
                                odds_ratio_health_seeking_in_children=self.parameters[
                                    f'or_care_seeking_if_perceived_severe']))
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
        df.loc[child_id, [f"ri_complication_{complication}" for complication in self.complications]] = False
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
        total_daly_values.loc[
            sorted(has_danger_signs)
        ] = self.daly_wts['daly_severe_ALRI']
        total_daly_values.loc[
            sorted(has_fast_breathing_or_chest_indrawing_but_not_danger_signs)
        ] = self.daly_wts['daly_non_severe_ALRI']

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
        self.consumables_used_in_hsi['1st_line_IV_ampicillin_gentamicin'] = {
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
        self.consumables_used_in_hsi['1st_line_IV_benzylpenicillin_gentamicin'] = {
            get_item_code(item='Benzylpenicillin 3g (5MU), PFR_each_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {1: 2, 2: 5, 4: 8, 12: 15, 36: 24, np.inf: 34}
                                                          ),
            get_item_code(item='Gentamicin Sulphate 40mg/ml, 2ml_each_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {1: 0.7, 2: 1.4, 4: 2.81, 12: 4.69, 36: 7.03, np.inf: 9.37}
                                                          ),
            get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST'): 1,
            get_item_code(item='Syringe, Autodisable SoloShot IX '): 1
        }

        # Second line of antibiotics for severe pneumonia, if Staph not suspected
        self.consumables_used_in_hsi['2nd_line_IV_ceftriaxone'] = {
            get_item_code(item='Ceftriaxone 1g, PFR_each_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {4: 1.5, 12: 3, 36: 5, np.inf: 7}
                                                          ),
            get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST'): 1,
            get_item_code(item='Syringe, Autodisable SoloShot IX '): 1
        }

        # Second line of antibiotics for severe pneumonia, if Staph is suspected
        self.consumables_used_in_hsi['2nd_line_IV_flucloxacillin_gentamicin'] = {
            get_item_code(item='Flucloxacillin 250mg, vial, PFR_each_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {2: 21, 4: 22.4, 12: 37.3, 36: 67.2, 60: 93.3, np.inf: 140}
                                                          ),
            get_item_code(item='Gentamicin Sulphate 40mg/ml, 2ml_each_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {4: 2.81, 12: 4.69, 36: 7.03, 60: 9.37, np.inf: 13.6}
                                                          ),
            get_item_code(item='Cannula iv  (winged with injection pot) 16_each_CMST'): 1,
            get_item_code(item='Syringe, Autodisable SoloShot IX '): 1,
            get_item_code(item='Flucloxacillin 250mg_100_CMST'):
                lambda _age: get_dosage_for_age_in_months(int(_age * 12.0),
                                                          {4: 0.42, 36: 0.84, 60: 1.68, np.inf: 1.68}
                                                          ),
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
            get_item_code(item='Syringe, Autodisable SoloShot IX '): 1
        }

        # Oxygen, pulse oximetry and x-ray -------------------

        # Oxygen for hypoxaemia
        self.consumables_used_in_hsi['Oxygen_Therapy'] = {
            get_item_code(item='Oxygen, 1000 liters, primarily with oxygen cylinders'): 1,
        }

        # Pulse oximetry
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

        # Bronchodilator - inhaled
        self.consumables_used_in_hsi['Inhaled_Brochodilator'] = {
            get_item_code(item='Salbutamol sulphate 1mg/ml, 5ml_each_CMST'): 2
        }

        # Bronchodilator - oral
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

    def do_effects_of_treatment_and_return_outcome(self, person_id, antibiotic_provided: str, oxygen_provided: bool):
        """Helper function that enacts the effects of a treatment to Alri caused by a pathogen.
        It will only do something if the Alri is caused by a pathogen (this module).
        * Prevent any death event that may be scheduled from occurring
        * Returns 'success' if the treatment is successful and 'failure' is it is not successful."""

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
        disease_type = person.ri_disease_type
        hiv_infected_and_not_on_art = person.hv_inf and (person.hv_art != "on_VL_suppressed")
        un_clinical_acute_malnutrition = person.un_clinical_acute_malnutrition
        complications = [_c for _c in self.complications if person[f"ri_complication_{_c}"]]

        imci_symptom_based_classification = self.get_imci_classification_based_on_symptoms(
            facility_level='2',
            child_is_younger_than_2_months=person.age_exact_years < (2.0 / 12.0),
            symptoms=symptoms,
            hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
            un_clinical_acute_malnutrition=un_clinical_acute_malnutrition
        )

        # Will the treatment fail:
        treatment_fails = self.models.treatment_fails(
            imci_symptom_based_classification=imci_symptom_based_classification,
            SpO2_level=SpO2_level,
            disease_type=disease_type,
            age_exact_years=person.age_exact_years,
            complications=complications,
            symptoms=symptoms,
            hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
            un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
            antibiotic_provided=antibiotic_provided,
            oxygen_provided=oxygen_provided,
            pre_referral_oxygen='not_applicable'
        )

        # store the information of the first appointment
        self.store_treatment_info = {'person_id': person_id, 'antibiotic_provided': antibiotic_provided}

        # Cancel death (if there is one) if the treatment does not fail (i.e. it works) and return indication of
        # sucesss or failure.
        if not treatment_fails:
            self.cancel_death_and_schedule_cure(person_id)
            return 'success'
        else:
            return 'failure'

    def seek_care_level(self, symptoms, age) -> str:
        """Care seeking at facility levels, based on symptom severity"""

        prop_seek_level = {'2': self.parameters['base_prob_care_seeking_by_level'][0],
                           '1b': self.parameters['base_prob_care_seeking_by_level'][1],
                           '1a': self.parameters['base_prob_care_seeking_by_level'][2],
                           '0': self.parameters['base_prob_care_seeking_by_level'][3]}

        # increase seeking at higher levels by age
        age_gp = '<2mo' if age < 1/6 else '2_11mo' if 1/6 <= age < 1 else '12_23mo' if 1 <= age < 2 else '24_59mo'
        if age < 2:
            prop_seek_level.update({
                '2': to_prob(to_odds(prop_seek_level['2']) *
                             self.parameters[f'or_care_seeking_hospital_age{age_gp}']),
                '1b': to_prob(to_odds(prop_seek_level['1b']) *
                              self.parameters[f'or_care_seeking_hospital_age{age_gp}']),
                '1a': to_prob(to_odds(prop_seek_level['1a']) *
                              (1 / self.parameters[f'or_care_seeking_hospital_age{age_gp}'])),
                '0': to_prob(to_odds(prop_seek_level['0']) *
                             (1 / self.parameters[f'or_care_seeking_hospital_age{age_gp}']))})

        # Increase seeking at higher levels if severe symptoms
        if any(sev_symptom in ['danger_signs', 'respiratory_distress'] for sev_symptom in symptoms):
            if 'chest_indrawing' in symptoms:
                prop_seek_level.update({
                    '2': to_prob(to_odds(prop_seek_level['2']) *
                                 self.parameters[f'or_care_seeking_if_perceived_severe']),
                    '1b': to_prob(to_odds(prop_seek_level['1b']) *
                                  self.parameters[f'or_care_seeking_if_perceived_severe']),
                    '1a': to_prob(to_odds(prop_seek_level['1a']) *
                                  (1 / self.parameters[f'or_care_seeking_if_perceived_severe'])),
                    '0': to_prob(to_odds(prop_seek_level['0']) *
                                 (1 / self.parameters[f'or_care_seeking_if_perceived_severe']))})

            for symptom_name in ('danger_signs', 'respiratory_distress'):
                if symptom_name in symptoms:
                    prop_seek_level.update({
                        '2': to_prob(to_odds(prop_seek_level['2']) *
                                     self.parameters[f'or_care_seeking_hospital_{symptom_name}']),
                        '1b': to_prob(to_odds(prop_seek_level['1b']) *
                                      self.parameters[f'or_care_seeking_hospital_{symptom_name}']),
                        '1a': to_prob(to_odds(prop_seek_level['1a']) *
                                      (1 / self.parameters[f'or_care_seeking_hospital_{symptom_name}'])),
                        '0': to_prob(to_odds(prop_seek_level['0']) *
                                     (1 / self.parameters[f'or_care_seeking_hospital_{symptom_name}']))})

        sum_of_probs = prop_seek_level['2'] + prop_seek_level['1b'] + prop_seek_level['1a'] + prop_seek_level['0']
        prop_seek_level.update({'2': prop_seek_level['2']/sum_of_probs,
                                '1b': prop_seek_level['1b']/sum_of_probs,
                                '1a': prop_seek_level['1a']/sum_of_probs,
                                '0': prop_seek_level['0']/sum_of_probs})

        seek_level = self.rng.choice(list(prop_seek_level.keys()), p=list(prop_seek_level.values()))

        return seek_level

    def on_presentation(self, person_id, symptoms, hsi_event):
        """Action taken when a child (under 5 years old) presents at a generic appointment (emergency or non-emergency)
         with symptoms of `cough` or `difficult_breathing`."""
        df = self.sim.population.props

        self.record_sought_care_for_alri()

        if self.parameters['use_static_simulation_run']:
            seek_level = self.seek_care_level(symptoms, age=df.loc[person_id, 'age_exact_years'])
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Alri_Treatment(person_id=person_id, module=self,
                                             facility_level=seek_level),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=1),
                priority=1
            )
        else:
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
    def get_imci_classification_based_on_symptoms(child_is_younger_than_2_months: bool, symptoms: list,
                                                  facility_level: str,
                                                  hiv_infected_and_not_on_art: bool,
                                                  un_clinical_acute_malnutrition: str) -> str:
        """Based on age and symptoms, classify WHO-pneumonia severity. This is regarded as the *TRUE* classification
         based on symptoms. It will return one of: {
             'fast_breathing_pneumonia',
             'danger_signs_pneumonia',
             'chest_indrawing_pneumonia,
             'cough_or_cold'
        }."""

        # # # For hospital classification # # #
        if facility_level in ('2', '3'):
            # young infants classifications
            if child_is_younger_than_2_months:
                if ('danger_signs' in symptoms) or ('respiratory_distress' in symptoms) or (
                        'chest_indrawing' in symptoms):
                    return 'danger_signs_pneumonia'
                elif 'tachypnoea' in symptoms:
                    if hiv_infected_and_not_on_art or (un_clinical_acute_malnutrition != 'well'):
                        return 'danger_signs_pneumonia'
                    else:
                        return 'fast_breathing_pneumonia'
                else:
                    return 'cough_or_cold'
            # 2-59 months old classifications
            else:
                if ('danger_signs' in symptoms) or ('respiratory_distress' in symptoms):
                    return 'danger_signs_pneumonia'
                elif 'chest_indrawing' in symptoms:
                    if hiv_infected_and_not_on_art or (un_clinical_acute_malnutrition == 'SAM'):
                        return 'danger_signs_pneumonia'
                    else:
                        return 'chest_indrawing_pneumonia'
                elif 'tachypnoea' in symptoms:
                    if hiv_infected_and_not_on_art or (un_clinical_acute_malnutrition == 'SAM'):
                        return 'danger_signs_pneumonia'
                    else:
                        return 'fast_breathing_pneumonia'
                else:
                    return 'cough_or_cold'

        # # # For village clinics, rural hospitals and health centres classification # # #
        elif facility_level in ('0', '1a', '1b'):
            # young infants classifications
            if child_is_younger_than_2_months:
                if ('danger_signs' in symptoms) or ('chest_indrawing' in symptoms):
                    return 'danger_signs_pneumonia'
                elif 'tachypnoea' in symptoms:
                    if hiv_infected_and_not_on_art or (un_clinical_acute_malnutrition != 'well'):
                        return 'danger_signs_pneumonia'
                    else:
                        return 'fast_breathing_pneumonia'
                else:
                    return 'cough_or_cold'
            # 2-59 months old classifications
            else:
                if 'danger_signs' in symptoms:
                    return 'danger_signs_pneumonia'
                elif 'chest_indrawing' in symptoms:
                    if hiv_infected_and_not_on_art or (un_clinical_acute_malnutrition == 'SAM'):
                        return 'danger_signs_pneumonia'
                    else:
                        return 'chest_indrawing_pneumonia'
                elif 'tachypnoea' in symptoms:
                    if hiv_infected_and_not_on_art or (un_clinical_acute_malnutrition == 'SAM'):
                        return 'danger_signs_pneumonia'
                    else:
                        return 'fast_breathing_pneumonia'
                else:
                    return 'cough_or_cold'

        else:
            raise ValueError(f'Unrecognised facility level {facility_level}')

    def referral_from_hc(self, classification_for_treatment_decision, facility_level):
        """ function used in CEA static run
        return referred up, new facility level """
        p = self.parameters

        # referral of cases at the lower level to district and rural hospitals
        referral = p['referral_rate_severe_cases_from_hc'] > self.rng.random_sample()
        referred_to_level = self.rng.choice(['2', '1b'], p=[p['prop_facility_referred_to'][0],
                                                            p['prop_facility_referred_to'][1]])

        # get those that need referral based on classification given
        needs_referral = True if classification_for_treatment_decision == 'danger_signs_pneumonia' and \
                                 facility_level in ('1a', '0') else False

        if needs_referral:
            # needs referral and was referred
            if referral:
                next_facility_level = referred_to_level  # now new level decides on treatment
                return needs_referral, referral, next_facility_level
            # needs referral not referred
            else:
                return needs_referral, referral, facility_level
        # don't need referral
        else:
            return needs_referral, False, facility_level

    def follow_up_treatment_failure(self, original_classification_given, symptoms):

        """ function in static simulation - follow-up of cases with treatment failure
        Apply classification of chest-indrawing to previous fast-breathing pneumonia HW classification
        Apply classification danger signs pneumonia to previous chest-indrawing or danger signs HW given classifications """

        # apply a porportion that will seek care:
        p = self.parameters
        prob_sought_fup_care = p['sought_follow_up_care']

        if prob_sought_fup_care < 1.0:
            if any(sev_symptom in ['danger_signs', 'respiratory_distress'] for sev_symptom in symptoms):
                if 'danger_signs' in symptoms:
                    prob_sought_fup_care = to_prob(to_odds(prob_sought_fup_care) * p[f'or_care_seeking_hospital_danger_signs'])
                if 'respiratory_distress' in symptoms:
                    prob_sought_fup_care = to_prob(to_odds(prob_sought_fup_care) * p[f'or_care_seeking_hospital_respiratory_distress'])
                # if 'chest_indrawing' in symptoms:
                #     prob_sought_fup_care = to_prob(to_odds(prob_sought_fup_care) * p[f'or_care_seeking_if_perceived_severe'])
            else:
                if 'chest_indrawing' not in symptoms:
                    prob_sought_fup_care = to_prob(to_odds(prob_sought_fup_care) * (1 / p[f'or_care_seeking_if_perceived_severe']))
                else:
                    prob_sought_fup_care = p['sought_follow_up_care']

        prob_fup_care = min(1.0, prob_sought_fup_care)
        sought_follow_up_care = prob_fup_care > self.rng.random_sample()

        # get a follow-up classification
        # cough or cold, and fast-breathing pneumonia classifications will be given oral antibiotics, treat outpatient
        if original_classification_given in ('cough_or_cold', 'fast_breathing_pneumonia'):
            followup_classification = 'chest_indrawing_pneumonia'
        # chest-indrawing and danger signs pneumonia classifications will be given inpatient care
        else:
            followup_classification = 'danger_signs_pneumonia'

        return sought_follow_up_care, followup_classification

    # @staticmethod
    def _ultimate_treatment_indicated_for_patient(self, classification_for_treatment_decision, age_exact_years,
                                                  facility_level, oxygen_saturation) -> Dict:
        """Return a Dict of the form {'antibiotic_indicated': Tuple[str], 'oxygen_indicated': <>} which expresses what
         the treatment is that the patient _should_ be provided with ultimately (i.e., if consumables are available and
         following an admission to in-patient, if required).
         For the antibiotic indicated, the first in the list is the one for which effectiveness parameters are defined;
         the second is assumed to be a drop-in replacement with the same effectiveness etc., and is used only when the
         first is found to be not available."""
        p = self.parameters

        # Change this variable when looking into the benefit of using SpO2 <93% as indication for oxygen
        oxygen_indicated = oxygen_saturation == '<90%' if p['apply_oxygen_indication_to_SpO2_measurement'] == '<90%' \
            else (
            (not oxygen_saturation == '>=93%') if p['apply_oxygen_indication_to_SpO2_measurement'] == '<93%' else False)

        # if no oximeter available, oxygen indication is determined by the heath worker's decision (ds-pneumonia)
        # oxygen_indicated_without_po = \
        #     p['prob_hw_decision_for_oxygen_provision_when_po_unavailable'] > self.rng.random_sample()

        # treatment for cough or cold classification
        if classification_for_treatment_decision == "cough_or_cold":
            return {
                'antibiotic_indicated': (
                    '',  # <-- First choice antibiotic -- none for cough_or_cold
                ),
                'oxygen_indicated': False
            }

        # treatment for fast-breathing pneumonia classification
        elif classification_for_treatment_decision == 'fast_breathing_pneumonia':
            return {
                'antibiotic_indicated': (
                    'Amoxicillin_tablet_or_suspension_7days' if age_exact_years < 2.0 / 12.0
                    else 'Amoxicillin_tablet_or_suspension_3days',  # <-- # <-- First choice antibiotic
                ),
                'oxygen_indicated': False
            }

        # treatment for chest-indrawing pneumonia classification
        elif classification_for_treatment_decision == 'chest_indrawing_pneumonia':
            return {
                'antibiotic_indicated': (
                    'Amoxicillin_tablet_or_suspension_5days',  # <-- # <-- First choice antibiotic
                ),
                'oxygen_indicated': False
            }

        # treatment for danger signs pneumonia classification
        elif classification_for_treatment_decision == 'danger_signs_pneumonia':
            # assume that PO at inpatient care is available if oxygen systems are implemented
            if facility_level in ('2', '1b'):
                return {
                    'antibiotic_indicated': (
                        '1st_line_IV_ampicillin_gentamicin',  # <-- # <-- First choice antibiotic
                        '1st_line_IV_benzylpenicillin_gentamicin',  # <-- If the first choice not available
                        '2nd_line_IV_ceftriaxone',  # <-- If the first line choices not available
                    ),
                    'oxygen_indicated': oxygen_indicated
                }
            elif facility_level in ('1a', '0'):
                return {
                    'antibiotic_indicated': (
                        'Amoxicillin_tablet_or_suspension_5days',  # <-- # <-- First choice antibiotic
                    ),
                    'oxygen_indicated': oxygen_indicated
                }

        else:
            raise ValueError(f'Classification not recognised: {classification_for_treatment_decision}')

    # # @staticmethod
    # def _ultimate_treatment_indicated_for_patient(self, classification_for_treatment_decision, age_exact_years,
    #                                               facility_level, oxygen_saturation, use_oximeter) -> Dict:
    #     """Return a Dict of the form {'antibiotic_indicated': Tuple[str], 'oxygen_indicated': <>} which expresses what
    #      the treatment is that the patient _should_ be provided with ultimately (i.e., if consumables are available and
    #      following an admission to in-patient, if required).
    #      For the antibiotic indicated, the first in the list is the one for which effectiveness parameters are defined;
    #      the second is assumed to be a drop-in replacement with the same effectiveness etc., and is used only when the
    #      first is found to be not available."""
    #     p = self.parameters
    #     # referral = False  # assume 100% referrals
    #     referral = 0.5 > self.rng.random_sample()
    #
    #     # Change this variable when looking into the benefit of using SpO2 <93% as indication for oxygen
    #     oxygen_indicated = oxygen_saturation == '<90%' if p['apply_oxygen_indication_to_SpO2_measurement'] == '<90%' \
    #         else (
    #         (not oxygen_saturation == '>=93%') if p['apply_oxygen_indication_to_SpO2_measurement'] == '<93%' else False)
    #
    #     # if no oximeter available, oxygen indication is determined by the heath worker's decision (ds-pneumonia)
    #     # oxygen_indicated_without_po = \
    #     #     p['prob_hw_decision_for_oxygen_provision_when_po_unavailable'] > self.rng.random_sample()
    #
    #     # treatment for cough or cold classification
    #     if classification_for_treatment_decision == "cough_or_cold":
    #         return {
    #             'antibiotic_indicated': (
    #                 '',  # <-- First choice antibiotic -- none for cough_or_cold
    #             ),
    #             'oxygen_indicated': False
    #         }
    #
    #     # treatment for fast-breathing pneumonia classification
    #     elif classification_for_treatment_decision == 'fast_breathing_pneumonia':
    #         return {
    #             'antibiotic_indicated': (
    #                 'Amoxicillin_tablet_or_suspension_7days' if age_exact_years < 2.0 / 12.0
    #                 else 'Amoxicillin_tablet_or_suspension_3days',  # <-- # <-- First choice antibiotic
    #             ),
    #             'oxygen_indicated': False
    #         }
    #
    #     # treatment for chest-indrawing pneumonia classification
    #     elif classification_for_treatment_decision == 'chest_indrawing_pneumonia':
    #         if (referral and facility_level == '0') or (facility_level in ('2', '1b', '1a')):
    #             return {
    #                 'antibiotic_indicated': (
    #                     'Amoxicillin_tablet_or_suspension_5days',  # <-- # <-- First choice antibiotic
    #                 ),
    #                 'oxygen_indicated': False
    #             }
    #         else:
    #             return {
    #                 'antibiotic_indicated': (
    #                     'Amoxicillin_tablet_or_suspension_7days' if age_exact_years < 2.0 / 12.0
    #                     else 'Amoxicillin_tablet_or_suspension_3days',  # <-- # <-- First choice antibiotic
    #                 ),
    #                 'oxygen_indicated': False
    #             }
    #
    #     # treatment for danger signs pneumonia classification
    #     elif classification_for_treatment_decision == 'danger_signs_pneumonia':
    #         # assume that PO at inpatient care is available if oxygen systems are implemented
    #         if (referral and facility_level in ('0', '1a')) or (facility_level in ('2', '1b')):
    #             return {
    #                 'antibiotic_indicated': (
    #                     '1st_line_IV_ampicillin_gentamicin',  # <-- # <-- First choice antibiotic
    #                     '1st_line_IV_benzylpenicillin_gentamicin',  # <-- If the first choice not available
    #                     '2nd_line_IV_ceftriaxone',  # <-- If the first line choices not available
    #                 ),
    #                 'oxygen_indicated': oxygen_indicated
    #             }
    #         else:
    #             return {
    #                 'antibiotic_indicated': (
    #                     'Amoxicillin_tablet_or_suspension_5days',  # <-- # <-- First choice antibiotic
    #                 ),
    #                 'oxygen_indicated': False
    #             }
    #
    #     else:
    #         raise ValueError(f'Classification not recognised: {classification_for_treatment_decision}')


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
                    Predictor().when('(hv_inf == True) & (hv_art != "on_VL_suppressed")', p['rr_ALRI_HIV/AIDS']),
                    Predictor('un_WHZ_category').when('WHZ<-3' or '-3<=WHZ<-2', p['rr_ALRI_wasting']),
                    Predictor('nb_low_birth_weight_status').when('extremely_low_birth_weight' or
                                                                 'very_low_birth_weight' or 'low_birth_weight',
                                                                 p['rr_ALRI_low_birth_weight']),
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
        col = 0 if age_exact_years < 1.0 else 1
        prob_is_pneumonia = p[f'proportion_pneumonia_in_{pathogen}_ALRI'][col]

        if prob_is_pneumonia > self.rng.random_sample():
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

    @staticmethod
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

    def get_complications_that_onset(self, disease_type,
                                     primary_path_is_bacterial, has_secondary_bacterial_inf):
        """Determine the set of complication for this person"""
        p = self.p

        probs = defaultdict(float)

        # get the probabilities of hypoxaemia in other alri knowing the prevalence, OR for CXR+, and CXR+/- proportion
        rr_and_prob_hypoxaemia_in_other_alri = self.get_prob_of_outcome_in_baseline_group(
            or_value=p['or_hypoxaemia_in_abnormal_CXR'],
            prob_ref=p['assumed_prev_hypoxaemia_in_normal_CXR'],
            prop_case_group=p['proportion_pneumonia_in_alri'],
            prevalence=p['prev_hypoxaemia_in_alri'])

        # get the probabilities of hypoxaemia by presence of PC in Pneumonia # OR = 2.236 ref Kamal Masarweh et al 2021
        rr_and_base_prob_hypoxaemia_in_non_pc = self.get_prob_of_outcome_in_baseline_group(
            or_value=p['or_hypoxaemia_in_pc_pneumonia'],
            prob_ref=p['assumed_prev_hypoxaemia_in_non_pc_pneumonia'],
            prop_case_group=p['prob_pulmonary_complications_in_pneumonia'],
            prevalence=rr_and_prob_hypoxaemia_in_other_alri['adjusted_p'] * rr_and_prob_hypoxaemia_in_other_alri['rr'])

        # get the probabilities of bacteraemia in other alri knowing the prevalence, OR for CXR+, and CXR+/- proportion
        rr_and_prob_bacteraemia_in_other_alri = self.get_prob_of_outcome_in_baseline_group(
            or_value=p['or_bacteraemia_in_abnormal_CXR'],
            prob_ref=p['assumed_prev_bacteraemia_in_normal_CXR'],
            prop_case_group=p['proportion_pneumonia_in_alri'],
            prevalence=p['prev_bacteraemia_in_alri'])

        # get the probabilities of bacteraemia by presence of pulmonary complicated pneumonia
        rr_and_base_prob_bacteraemia = self.get_prob_of_outcome_in_baseline_group(
            or_value=p['or_bacteraemia_in_pc_pneumonia'],
            prob_ref=p['assumed_prev_bacteraemia_in_non_pc_pneumonia'],
            prop_case_group=p['prob_pulmonary_complications_in_pneumonia'],
            prevalence=rr_and_prob_bacteraemia_in_other_alri['adjusted_p'] *
                       rr_and_prob_bacteraemia_in_other_alri['rr'])  # to get prevalence of bacteraemia in CXR+

        # Sort the probabilities to the disease types
        # For 'pneumonia' disease
        if disease_type == 'pneumonia':

            # get probabilities for local pulmonary complications
            if p['prob_pulmonary_complications_in_pneumonia'] > self.rng.random_sample():

                # get the probabilities for each pulmonary complication
                for c in ['pleural_effusion', 'pneumothorax']:  # regardless of bacterial/viral/fungal pathogen
                    probs[c] = p[f'prob_{c}_in_pulmonary_complicated_pneumonia']

                for c in ['empyema', 'lung_abscess']:
                    if primary_path_is_bacterial or has_secondary_bacterial_inf:
                        if probs['pleural_effusion'] > self.rng.random_sample():
                            probs['empyema'] = p[f'prob_empyema_in_pulmonary_complicated_pneumonia'] * \
                                               (1 / p['proportion_bacterial_infection_in_pneumonia'])
                        probs['lung_abscess'] = p[f'prob_lung_abscess_in_pulmonary_complicated_pneumonia'] * \
                                                (1 / p['proportion_bacterial_infection_in_pneumonia'])
                    else:
                        probs[c] = 0

                # get the probability of bacteraemia in pulmonary complicated pneumonia
                if primary_path_is_bacterial or has_secondary_bacterial_inf:
                    probs['bacteraemia'] = \
                        rr_and_base_prob_bacteraemia['adjusted_p'] * rr_and_base_prob_bacteraemia['rr'] * \
                        (1 / p['proportion_bacterial_infection_in_pneumonia'])
                    # proportion of bacterial infection/co-infection in pneumonia from model output

                # get the probability of hypoxaemia pulmonary complicated pneumonia
                probs['hypoxaemia'] = rr_and_base_prob_hypoxaemia_in_non_pc['adjusted_p'] * \
                                      rr_and_base_prob_hypoxaemia_in_non_pc['rr']

            # if no pulmonary complication is present
            else:
                # get the probability of bacteraemia in non-pulmonary complicated pneumonia
                if primary_path_is_bacterial or has_secondary_bacterial_inf:
                    probs['bacteraemia'] = rr_and_base_prob_bacteraemia['adjusted_p'] * \
                                           (1 / p['proportion_bacterial_infection_in_pneumonia'])

                # get the probability of hypoxaemia in non-pulmonary complicated pneumonia
                probs['hypoxaemia'] = rr_and_base_prob_hypoxaemia_in_non_pc['adjusted_p']

        # # # # # # # # #
        # For 'other_alri' disease
        elif disease_type == 'other_alri':

            # probability of hypoxaemia in other ALRI
            probs['hypoxaemia'] = rr_and_prob_hypoxaemia_in_other_alri['adjusted_p']

            # probability of bacteraemia in other ALRI
            if primary_path_is_bacterial or has_secondary_bacterial_inf:
                probs['bacteraemia'] = rr_and_prob_bacteraemia_in_other_alri['adjusted_p'] * \
                                       (1 / p['proportion_bacterial_infection_in_other_alri'])
                # proportion of bacterial infection/co-infection in other alri from model output

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

    def symptoms_for_uncomplicated_disease(self, disease_type) -> set:
        """Determine set of symptom (before complications) for a given instance of disease"""
        p = self.p

        assert disease_type in self.module.disease_types

        # apply main ALRI symptoms
        probs = {
            symptom: p[f'prob_{symptom}_in_{disease_type}']
            for symptom in [
                'cough', 'difficult_breathing', 'fever', 'tachypnoea', 'chest_indrawing']
        }
        # determine which main symptoms are onset:
        symptoms = {s for s, p in probs.items() if p > self.rng.random_sample()}

        # apply probability of respiratory distress based on the presence of chest-indrawing
        probs_rd = defaultdict(float)

        # get the probability of resp. distress without chest-indrawing (base group) and the relative risk from the OR
        rr_and_prob_rd_in_alri_without_ci = self.get_prob_of_outcome_in_baseline_group(
            or_value=p['or_respiratory_distress_in_alri_with_chest_indrawing'],
            prob_ref=p['prob_respiratory_distress_in_no_chest_indrawing_SpO2>=93%'],
            prop_case_group=p[f'prob_chest_indrawing_in_{disease_type}'],
            prevalence=p[f'prob_respiratory_distress_in_{disease_type}'])

        # get the probability of danger signs without respiratory distress (base group) and the RR from the OR
        rr_and_prob_ds_in_alri_without_rd = self.get_prob_of_outcome_in_baseline_group(
            or_value=p['or_danger_signs_in_alri_with_respiratory_distress'],
            prob_ref=p['prob_danger_signs_in_no_respiratory_distress_SpO2>=93%'],
            prop_case_group=p[f'prob_respiratory_distress_in_{disease_type}'],
            prevalence=p[f'prob_danger_signs_in_{disease_type}'])

        if 'chest_indrawing' in symptoms:
            probs_rd = {'respiratory_distress':
                            rr_and_prob_rd_in_alri_without_ci['adjusted_p'] * rr_and_prob_rd_in_alri_without_ci['rr']}
        else:
            probs_rd = {'respiratory_distress': rr_and_prob_rd_in_alri_without_ci['adjusted_p']}

        # determine which symptoms are onset:
        symptoms_rd = {s for s, p in probs_rd.items() if p > self.rng.random_sample()}

        # apply probability of danger signs based on the presence of respiratory distress
        probs_ds = defaultdict(float)

        if 'respiratory_distress' in symptoms_rd:
            probs_ds = {
                'danger_signs': rr_and_prob_ds_in_alri_without_rd['adjusted_p'] *
                                rr_and_prob_ds_in_alri_without_rd['rr'],
            }
        else:
            probs_ds = {'danger_signs': rr_and_prob_ds_in_alri_without_rd['adjusted_p']}

        # determine which symptoms are onset:
        symptoms_ds = {s for s, p in probs_ds.items() if p > self.rng.random_sample()}

        # return all the symptoms:
        all_symptoms = symptoms.union(symptoms_rd.union(symptoms_ds))

        return all_symptoms

    def symptoms_for_complicated_disease(self, disease_type, complications, oxygen_saturation) -> set:
        """Probability of each symptom for a person given a complication"""
        p = self.p

        # apply main ALRI symptoms
        probs = {
            symptom: p[f'prob_{symptom}_in_{disease_type}']
            for symptom in [
                'cough', 'difficult_breathing', 'fever', 'tachypnoea']
        }
        # increase the odds of having symptoms in complicated disease
        probs.update({
            symptom: to_prob(to_odds(p[f'prob_{symptom}_in_{disease_type}']) * p[f'or_{symptom}_in_complicated_alri'])
            for symptom in [
                'fever', 'tachypnoea']
        })

        # determine which main symptoms are onset in severe PC
        general_symptoms = {s for s, p in probs.items() if p > self.rng.random_sample()}

        # set of dictionaries storing probability of symptoms for hypoxaemia
        probs_hypo = defaultdict(float)
        probs_hypo_ci = defaultdict(float)
        probs_hypo_rd = defaultdict(float)
        probs_hypo_ds = defaultdict(float)

        # set of dictionaries storing probability of symptoms for pulmonary complications (PE and Emp)
        probs_pc = defaultdict(float)
        probs_pc_ci = defaultdict(float)
        probs_pc_rd = defaultdict(float)
        probs_pc_ds = defaultdict(float)

        # set of dictionaries storing probability of symptoms for pulmonary complications (PE and Emp)
        probs_spc = defaultdict(float)
        probs_spc_ci = defaultdict(float)
        probs_spc_rd = defaultdict(float)
        probs_spc_ds = defaultdict(float)

        if 'hypoxaemia' in complications:  # with or without pulmonary complications
            oxygensat = oxygen_saturation if oxygen_saturation == '<90%' else '_90-92%'

            # get the probability of resp. distress without chest-indrawing (base group) and the RR from the OR
            rr_and_prob_rd_in_hypoxaemic_alri_without_ci = self.get_prob_of_outcome_in_baseline_group(
                or_value=p['or_respiratory_distress_in_alri_with_chest_indrawing'],
                prob_ref=p[f'prob_respiratory_distress_in_no_chest_indrawing_SpO2{oxygensat}'],
                prop_case_group=p[f'prob_chest_indrawing_in_SpO2{oxygensat}'],
                prevalence=p[f'prob_respiratory_distress_in_SpO2{oxygensat}'])

            # get the probability of danger signs without resp distress (base group) and the RR from the OR
            rr_and_prob_ds_in_hypoxaemic_alri_without_rd = self.get_prob_of_outcome_in_baseline_group(
                or_value=p['or_danger_signs_in_alri_with_respiratory_distress'],
                prob_ref=p[f'prob_danger_signs_in_no_respiratory_distress_SpO2{oxygensat}'],
                prop_case_group=p[f'prob_respiratory_distress_in_SpO2{oxygensat}'],
                prevalence=p[f'prob_danger_signs_in_SpO2{oxygensat}'])

            # apply probability of chest indrawing
            probs_hypo_ci = {
                'chest_indrawing': p[f'prob_chest_indrawing_in_SpO2{oxygensat}']
            }

            if probs_hypo_ci['chest_indrawing'] > self.rng.random_sample():
                probs_hypo_rd = {'respiratory_distress': rr_and_prob_rd_in_hypoxaemic_alri_without_ci['adjusted_p'] *
                                                         rr_and_prob_rd_in_hypoxaemic_alri_without_ci['rr']}
            else:
                probs_hypo_rd = {'respiratory_distress': rr_and_prob_rd_in_hypoxaemic_alri_without_ci['adjusted_p']}

            if probs_hypo_rd['respiratory_distress'] > self.rng.random_sample():
                probs_hypo_ds = {'danger_signs': rr_and_prob_ds_in_hypoxaemic_alri_without_rd['adjusted_p'] *
                                                 rr_and_prob_ds_in_hypoxaemic_alri_without_rd['rr']}
            else:
                probs_hypo_ds = {'danger_signs': rr_and_prob_ds_in_hypoxaemic_alri_without_rd['adjusted_p']}

        # determine which symptoms are onset from hypoxaemia:
        probs_hypo.update(probs_hypo_ci)
        probs_hypo.update(probs_hypo_rd)
        probs_hypo.update(probs_hypo_ds)

        symptoms_hypoxaemia = {s for s, p in probs_hypo.items() if p > self.rng.random_sample()}

        # pulmonary complications - PE and Emp can mimic non-severe pneumonia:
        for pc in ['pleural_effusion', 'empyema']:
            if pc in complications:
                # get the probability of resp. distress without chest-indrawing (base group) and the RR from the OR
                rr_and_prob_rd_in_pc_alri_without_ci = self.get_prob_of_outcome_in_baseline_group(
                    or_value=p['or_respiratory_distress_in_alri_with_chest_indrawing'],
                    prob_ref=p[f'prob_respiratory_distress_in_no_chest_indrawing_pc'],
                    prop_case_group=p[f'prob_chest_indrawing_in_pulmonary_complications'],
                    prevalence=p[f'prob_respiratory_distress_in_pulmonary_complications'])

                # get the probability of danger signs without respiratory distress (base group) and the RR from the OR
                rr_and_prob_ds_in_pc_alri_without_rd = self.get_prob_of_outcome_in_baseline_group(
                    or_value=p['or_danger_signs_in_alri_with_respiratory_distress'],
                    prob_ref=p[f'prob_danger_signs_in_no_respiratory_distress_pc'],
                    prop_case_group=p[f'prob_respiratory_distress_in_pulmonary_complications'],
                    prevalence=p[f'prob_danger_signs_in_pulmonary_complications'])

                # apply probability of chest indrawing
                probs_pc_ci = {
                    'chest_indrawing': p[f'prob_chest_indrawing_in_pulmonary_complications']
                }

                if probs_pc_ci['chest_indrawing'] > self.rng.random_sample():
                    probs_pc_rd = {'respiratory_distress': rr_and_prob_rd_in_pc_alri_without_ci['adjusted_p'] *
                                                           rr_and_prob_rd_in_pc_alri_without_ci['rr']}
                else:
                    probs_pc_rd = {'respiratory_distress': rr_and_prob_rd_in_pc_alri_without_ci['adjusted_p']}

                if probs_pc_rd['respiratory_distress'] > self.rng.random_sample():
                    probs_pc_ds = {'danger_signs': rr_and_prob_ds_in_pc_alri_without_rd['adjusted_p'] *
                                                   rr_and_prob_ds_in_pc_alri_without_rd['rr']}
                else:
                    probs_pc_ds = {'danger_signs': rr_and_prob_ds_in_pc_alri_without_rd['adjusted_p']}

        # determine which symptoms are onset from pulmonary complications:
        probs_pc.update(probs_pc_ci)  # add chest indrawing
        probs_pc.update(probs_pc_rd)  # add respiratory distress
        probs_pc.update(probs_pc_ds)  # add danger signs

        symptoms_pc = {s for s, p in probs_pc.items() if p > self.rng.random_sample()}

        # severe pulmonary complications - LA and Pth:
        for spc in ['lung_abscess', 'pneumothorax']:
            if spc in complications:
                # increase the odds of having fever and fast breathing symptoms in complicated disease
                probs_spc.update({
                    symptom: to_prob(to_odds(probs[symptom]) * p[f'or_{symptom}_in_complicated_alri'])
                    for symptom in [
                        'fever', 'tachypnoea']
                })

                # get the probability of resp. distress without chest-indrawing (base group) and the RR from the OR
                rr_and_prob_rd_in_spc_alri_without_ci = self.get_prob_of_outcome_in_baseline_group(
                    or_value=p['or_respiratory_distress_in_alri_with_chest_indrawing'],
                    prob_ref=to_prob(to_odds(p['prob_respiratory_distress_in_no_chest_indrawing_pc']) *
                                     p['or_severe_symptoms_in_severe_pulmonary_complications']),
                    prop_case_group=to_prob(to_odds(p['prob_chest_indrawing_in_pulmonary_complications']) *
                                            p['or_severe_symptoms_in_severe_pulmonary_complications']),
                    prevalence=to_prob(to_odds(p['prob_respiratory_distress_in_pulmonary_complications']) *
                                       p['or_severe_symptoms_in_severe_pulmonary_complications']))

                # get the probability of danger signs without respiratory distress (base group) and the RR from the OR
                rr_and_prob_ds_in_spc_alri_without_rd = self.get_prob_of_outcome_in_baseline_group(
                    or_value=p['or_danger_signs_in_alri_with_respiratory_distress'],
                    prob_ref=to_prob(to_odds(p[f'prob_danger_signs_in_no_respiratory_distress_pc']) *
                                     p['or_severe_symptoms_in_severe_pulmonary_complications']),
                    prop_case_group=to_prob(to_odds(p['prob_respiratory_distress_in_pulmonary_complications']) *
                                            p['or_severe_symptoms_in_severe_pulmonary_complications']),
                    prevalence=to_prob(to_odds(p['prob_danger_signs_in_pulmonary_complications']) *
                                       p['or_severe_symptoms_in_severe_pulmonary_complications']))

                # apply probability of chest indrawing
                probs_spc_ci = {
                    'chest_indrawing': p[f'prob_chest_indrawing_in_pulmonary_complications']
                }

                if probs_spc_ci['chest_indrawing'] > self.rng.random_sample():
                    probs_spc_rd = {'respiratory_distress': rr_and_prob_rd_in_spc_alri_without_ci['adjusted_p'] *
                                                            rr_and_prob_rd_in_spc_alri_without_ci['rr']}
                else:
                    probs_spc_rd = {'respiratory_distress': rr_and_prob_rd_in_spc_alri_without_ci['adjusted_p']}

                if probs_spc_rd['respiratory_distress'] > self.rng.random_sample():
                    probs_spc_ds = {'danger_signs': rr_and_prob_ds_in_spc_alri_without_rd['adjusted_p'] *
                                                    rr_and_prob_ds_in_spc_alri_without_rd['rr']}
                else:
                    probs_spc_ds = {'danger_signs': rr_and_prob_ds_in_spc_alri_without_rd['adjusted_p']}

        # determine which symptoms are onset from pulmonary complications:
        probs_spc.update(probs_spc_ci)  # add chest indrawing
        probs_spc.update(probs_spc_rd)  # add respiratory distress
        probs_spc.update(probs_spc_ds)  # add danger signs

        symptoms_spc = {s for s, p in probs_spc.items() if p > self.rng.random_sample()}

        # join pulmonary complications
        symptoms_all_pc = symptoms_pc.union(symptoms_spc)

        # return all the symptoms:
        all_symptoms = general_symptoms.union(symptoms_hypoxaemia.union(symptoms_all_pc))

        return all_symptoms

    def will_die_of_alri(self, **kwargs) -> bool:
        """Determine if person will die from Alri"""
        return self.prob_die_of_alri(**kwargs) > self.rng.random_sample()

    def prob_die_of_alri(self,
                         age_exact_years: float,
                         sex: str,
                         bacterial_infection: bool,
                         disease_type: str,
                         all_symptoms: List[str],
                         SpO2_level: str,
                         complications: List[str],
                         un_clinical_acute_malnutrition: str,
                         ) -> float:
        """Returns the probability that such a case of ALRI will be lethal (if untreated)."""
        p = self.p
        df = self.module.sim.population.props

        # base group is CXR- without complications, fast-breathing or cough/cold only

        def get_odds_of_death(age_in_whole_months):
            """Returns odds of death given age in whole months."""

            def get_odds_of_death_for_under_two_month_old(age_in_whole_months):
                return p[f'base_odds_death_ALRI_{_age_}']

            def get_odds_of_death_for_over_two_month_old(age_in_whole_months):
                if 2 <= age_in_whole_months <= 5:
                    return p[f'base_odds_death_ALRI_{_age_}'] * p[f'or_death_ALRI_{_age_}_in_2_5mo']
                elif 6 <= age_in_whole_months <= 11:
                    return p[f'base_odds_death_ALRI_{_age_}'] * p[f'or_death_ALRI_{_age_}_in_6_11mo']
                else:
                    return p[f'base_odds_death_ALRI_{_age_}']

            return get_odds_of_death_for_under_two_month_old(age_in_whole_months=age_in_whole_months) if \
                age_in_whole_months < 2 else \
                get_odds_of_death_for_over_two_month_old(age_in_whole_months=age_in_whole_months)

        age_in_whole_months = int(np.floor(age_exact_years * 12.0))
        is_under_two_months_old = age_in_whole_months < 2
        _age_ = "age<2mo" if is_under_two_months_old else "age2_59mo"

        # Get baseline odds of death given age
        odds_death = get_odds_of_death(age_in_whole_months=age_in_whole_months)

        # Modify odds of death based on other factors:
        if 'danger_signs' in all_symptoms:
            odds_death *= p[f'or_death_ALRI_{_age_}_danger_signs']

        if SpO2_level == '<90%':
            odds_death *= p[f'or_death_ALRI_{_age_}_SpO2<90%']
        elif SpO2_level == '90-92%':
            odds_death *= p[f'or_death_ALRI_{_age_}_SpO2_90_92%']

        if not is_under_two_months_old:
            if sex == 'F':
                odds_death *= p[f'or_death_ALRI_{_age_}_female']

            if un_clinical_acute_malnutrition == 'SAM':
                odds_death *= p[f'or_death_ALRI_{_age_}_SAM']

            if 'chest_indrawing' in all_symptoms:
                odds_death *= p[f'or_death_ALRI_{_age_}_chest_indrawing']

            if 'respiratory_distress' in all_symptoms:
                odds_death *= p['or_death_ALRI_respiratory_distress']

        if 'bacteraemia' in complications:
            odds_death *= p['or_death_ALRI_bacteraemia']

        if disease_type == 'pneumonia':
            odds_death *= p['or_death_ALRI_abnormal_CXR']

        if any(pc in ['pleural_effusion', 'empyema', 'pneumothorax', 'lung_abscess'] for pc in complications):
            odds_death *= p['or_death_ALRI_pulmonary_complications']

        if bacterial_infection in self.module.pathogens['bacterial']:
            odds_death *= 4.01

        return min(1.0, to_prob(odds_death))

        # # Adjustments ----------------------------------
        # if 0 == len(complications):
        #     # Adjust the natural risk of death for those uncomplicated CXR- viral causes, without severe symptoms
        #     if disease_type == 'other_alri' and not any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection not in self.module.pathogens['bacterial']:
        #         return min(1.0, 0.6 * to_prob(odds_death))  # these wouldn't be be in the Lazzerini dataset
        #         # or within less than 1%
        #
        #     # Adjust the natural risk of death for those uncomplicated CXR- bacterial causes, without severe symptoms
        #     elif disease_type == 'other_alri' and not any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection in self.module.pathogens['bacterial']:
        #         return min(1.0, 0.8 * to_prob(odds_death))  # these wouldn't be be in the Lazzerini dataset
        #         # or within less than 1%
        #
        #     # Adjust the natural risk of death for those uncomplicated CXR- viral causes, with severe symptoms
        #     elif disease_type == 'other_alri' and any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection not in self.module.pathogens['bacterial']:
        #         return min(1.0, 0.8 * to_prob(odds_death))  # antibiotic treatment has no effect on outcome
        #
        #     # Adjust the natural risk of death for those uncomplicated CXR- bacterial causes, with severe symptoms
        #     elif disease_type == 'other_alri' and any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection in self.module.pathogens['bacterial']:
        #         return min(1.0, 1 * to_prob(odds_death))  # death for these cases = with/without treatment
        #
        #     # Adjust the natural risk of death for those uncomplicated CXR+ viral causes, without severe symptoms
        #     elif disease_type == 'pneumonia' and not any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection not in self.module.pathogens['bacterial']:
        #         return min(1.0, 1.1 * to_prob(odds_death))
        #
        #     # Adjust the natural risk of death for those uncomplicated CXR+ bacterial causes, without severe symptoms
        #     elif disease_type == 'pneumonia' and not any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection in self.module.pathogens['bacterial']:
        #         return min(1.0, 1.2 * to_prob(odds_death))
        #
        #     # Adjust the natural risk of death for those uncomplicated CXR+ viral causes, with severe symptoms
        #     elif disease_type == 'pneumonia' and any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection not in self.module.pathogens['bacterial']:
        #         return min(1.0, 1.2 * to_prob(odds_death))
        #
        #     # Adjust the natural risk of death for those uncomplicated CXR+ bacteriral causes, with severe symptoms
        #     elif disease_type == 'pneumonia' and any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection in self.module.pathogens['bacterial']:
        #         return min(1.0, 1.3 * to_prob(odds_death))
        #
        # # Now with complications
        # # Adjust the natural risk of death for those complicated CXR- viral causes, without severe symptoms
        # elif 0 < len(complications):
        #     if disease_type == 'other_alri' and not any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection not in self.module.pathogens['bacterial']:
        #         return min(1.0, 1.1 * to_prob(
        #             odds_death))
        #
        #     # Adjust the natural risk of death for those complicated CXR- bacterial causes, without severe symptoms
        #     elif disease_type == 'other_alri' and not any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection in self.module.pathogens['bacterial']:
        #         return min(1.0, 1.2 * to_prob(odds_death))
        #
        #     # Adjust the natural risk of death for those complicated CXR- viral causes, with severe symptoms
        #     elif disease_type == 'other_alri' and any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection not in self.module.pathogens['bacterial']:
        #         return min(1.0, 1.2 * to_prob(odds_death))
        #
        #     # Adjust the natural risk of death for those complicated CXR- bacterial causes, with severe symptoms
        #     elif disease_type == 'other_alri' and any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection in self.module.pathogens['bacterial']:
        #         return min(1.0, 1.3 * to_prob(odds_death))
        #
        #     # Adjust the natural risk of death for those complicated CXR+ viral causes, without severe symptoms
        #     elif disease_type == 'pneumonia' and not any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection not in self.module.pathogens['bacterial']:
        #         return min(1.0, 1.2 * to_prob(odds_death))
        #
        #     # Adjust the natural risk of death for those uncomplicated CXR+ bacterial causes, without severe symptoms
        #     elif disease_type == 'pneumonia' and not any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection in self.module.pathogens['bacterial']:
        #         return min(1.0, 1.3 * to_prob(odds_death))
        #
        #     # Adjust the natural risk of death for those uncomplicated CXR+ viral causes, with severe symptoms
        #     elif disease_type == 'pneumonia' and any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection not in self.module.pathogens['bacterial']:
        #         return min(1.0, 1.3 * to_prob(odds_death)) # (len(complications) *
        #
        #     # Adjust the natural risk of death for those uncomplicated CXR+ bacterial causes, with severe symptoms
        #     elif disease_type == 'pneumonia' and any(symptom in [
        #         'respiratory_distress', 'danger_signs'] for symptom in all_symptoms) and \
        #         bacterial_infection in self.module.pathogens['bacterial']:
        #         return min(1.0, 1.4 * to_prob(odds_death))
        #
        # else:
        #     raise ValueError('Case type missing in the adjustment')

    def treatment_fails(self, **kwargs) -> bool:
        """Determine whether a treatment fails or not: Returns `True` if the treatment fails."""
        p_fail = self._prob_treatment_fails(**kwargs)
        assert p_fail is not None, f"no probability of failure is recorded, {kwargs=}"
        return p_fail > self.rng.random_sample()

    def coverage_of_oxygen(self, scenario):
        """ This function is used in the static simulation in the CEA analysis for the Lancet Commission scenarios.
        Availability of oxygen is hierarchical - apply cumulative probability of coverage.
        Note: extra availabilities at lower facilities represent the greater number of facilities """

        p = self.p
        oxygen_available = list()
        oxygen_available = p[f'scenario_{scenario}_ox_coverage_by_facility'] > self.rng.random_sample(
            len(p[f'scenario_{scenario}_ox_coverage_by_facility']))

        return oxygen_available

    def _prob_treatment_fails(self,
                              imci_symptom_based_classification: str,
                              SpO2_level: str,
                              disease_type: str,
                              age_exact_years: float,
                              symptoms: list,
                              complications: list,
                              hiv_infected_and_not_on_art: bool,
                              un_clinical_acute_malnutrition: str,
                              antibiotic_provided: str,
                              oxygen_provided: bool,
                              pre_referral_oxygen: str,
                              this_is_follow_up: bool,
                              ) -> float:
        """Returns the probability of treatment failure. Treatment failures are dependent on the underlying IMCI
        classification by symptom, the need for oxygen (if SpO2 < 90%), and the type of antibiotic therapy (oral vs.
         IV/IM). NB. antibiotic_provided = '' means no antibiotic provided."""

        assert antibiotic_provided in self.module.antibiotics + [''], f"Not recognised {antibiotic_provided=}"

        p = self.p

        def _group_antibiotic(antibiotic_provided):
            if antibiotic_provided.startswith('1st_line_IV'):
                return '1st_line_IV_antibiotics'
            elif antibiotic_provided.startswith('2nd_line_IV'):
                return '2nd_line_IV_antibiotics'
            elif antibiotic_provided.startswith('Amoxicillin'):
                return 'oral_antibiotics'
            elif antibiotic_provided == '':
                return ''
            else:
                raise ValueError(f'{antibiotic_provided} not in any group')

        antibiotic_provided_grp = _group_antibiotic(antibiotic_provided)

        needs_oxygen = SpO2_level == '<90%'
        might_need_oxygen = SpO2_level == '90-92%'

        def modify_failure_risk_when_does_not_get_oxygen_but_needs_oxygen(_risk):
            """Define the effect size for the increase in the risk of treatment failure if a person need oxygen but does
             not receive it. The parameter is an odds ratio, so to use it with the risk, there has to be conversion
             between odds and probabilities."""
            age_in_whole_months = int(np.floor(age_exact_years * 12.0))
            is_under_two_months_old = age_in_whole_months < 2
            _age_ = "age<2mo" if is_under_two_months_old else "age2_59mo"

            # if the risk of treatmnet failure is already 1.0 with oxygen provided, return that value
            if _risk == 1.0:
                return _risk

            elif needs_oxygen:
                or_if_does_not_get_oxygen_but_needs_oxygen = 1.0 / p['or_mortality_improved_oxygen_systems']
                return min(1.0, to_prob(to_odds(_risk) * or_if_does_not_get_oxygen_but_needs_oxygen))

            elif might_need_oxygen:
                # first calculate the relative ORs between or_death for SpO2< 90% and 90-92%
                relative_or_mortality_in_low_ox_sat = \
                    p[f'or_death_ALRI_{_age_}_SpO2<90%'] / p[f'or_death_ALRI_{_age_}_SpO2_90_92%']
                # convert OR mortality of improved oxygen systems to likelihood (%)
                convert_or_into_likelihood = 1 - p['or_mortality_improved_oxygen_systems']
                p_tsuccess_for_oxygen_provided_to_SpO2_90_92 = \
                    convert_or_into_likelihood / relative_or_mortality_in_low_ox_sat
                # this returns in likelihood % -- convert back to OR
                assumed_or_tf_if_oxygen_provided_to_SpO2_90_92 = 1 - p_tsuccess_for_oxygen_provided_to_SpO2_90_92
                or_if_does_not_get_oxygen_but_might_need_oxygen = 1.0 / assumed_or_tf_if_oxygen_provided_to_SpO2_90_92
                return min(1.0, to_prob(to_odds(_risk) * or_if_does_not_get_oxygen_but_might_need_oxygen))

        def modify_failure_risk_when_follow_up(_risk):

            # if the risk of treatment failure is already 1.0, return that value
            if _risk == 1.0:
                return _risk

            if this_is_follow_up:
                _risk = to_prob(to_odds(_risk) * p['or_tf_at_follow_up_care_with_previous_oral_antibiotic'])
                if any([needs_oxygen, might_need_oxygen]) or (any(['danger_signs', 'respiratory_distress']) in symptoms):
                    _risk *= p['rr_tf_follow_up_care_with_initial_incorrect_care']
                return min(1.0, _risk)
                # assume a risk of increase failure for oral antibiotics if no care was provided at the first appt
            else:
                return _risk

        def modify_failure_risk_when_stabilised_with_ox_at_hc(_risk):
            # if the risk of treatment failure is already 1.0, return that value
            if _risk == 1.0:
                return _risk

            if needs_oxygen and pre_referral_oxygen == 'not_provided':
                _risk = to_prob(to_odds(_risk) * p['or_tf_non_stabilised_with_oxygen_prior_to_referral'])
                return min(1.0, _risk)
            else:
                return _risk

        def _treatment_failure_oral_antibiotics(risk_tf_oral_antibiotics):
            """Return the risk of treatment failure by oral antibiotics for non-severe pneumonia classification"""

            risk_tf_oral_antibiotics = risk_tf_oral_antibiotics

            if SpO2_level == '<90%':
                risk_tf_oral_antibiotics = to_prob(
                    to_odds(risk_tf_oral_antibiotics) * p['or_tf_oral_antibiotics_if_SpO2<90%'])

            if SpO2_level == '90-92%':
                risk_tf_oral_antibiotics = to_prob(
                    to_odds(risk_tf_oral_antibiotics) * p['or_tf_oral_antibiotics_if_SpO2_90_92%'])

            if un_clinical_acute_malnutrition == 'MAM':
                risk_tf_oral_antibiotics = to_prob(
                    to_odds(risk_tf_oral_antibiotics) * p['or_tf_oral_antibiotics_if_MAM'])

            if un_clinical_acute_malnutrition == 'SAM':
                risk_tf_oral_antibiotics = to_prob(
                    to_odds(risk_tf_oral_antibiotics) * p['or_tf_oral_antibiotics_if_SAM'])

            if 'danger_signs' in symptoms:
                risk_tf_oral_antibiotics *= p['rr_tf_oral_antibiotics_if_danger_signs']

            if 'respiratory_distress' in symptoms:
                risk_tf_oral_antibiotics *= p['rr_tf_oral_antibiotics_if_repiratory_distress']

            # if 'ma_is_infected' and not 'ma_tx':
            #     risk_tf_oral_antibiotics = to_prob(
            #         to_odds(risk_tf_oral_antibiotics *  p['or_tf_oral_antibiotics_if_concurrent_malaria']))

            # use the same RR of treatment failure of IV antibiotics for those with CXR+ (mostly pneumonia group)
            # for consistency of risk factors between the two treatment groups (oral / IV)
            if disease_type == 'pneumonia':
                risk_tf_oral_antibiotics *= p['rr_tf_1st_line_antibiotics_if_abnormal_CXR']

            # The effect of HIV
            if hiv_infected_and_not_on_art:
                risk_tf_oral_antibiotics *= p['rr_tf_1st_line_antibiotics_if_HIV/AIDS']

            # return risk_tf_oral_antibiotics
            return min(1.0, risk_tf_oral_antibiotics)

        def _treatment_failure_IV_antibiotics():
            """Return the risk of treatment failure by parenteral antibiotics for severe pneumonia classification"""

            # Baseline risk of treatment failure ( ref group: non-need oxygen or if oxygen is also provided)
            risk_tf_iv_antibiotics = p['tf_2nd_line_antibiotic_for_severe_pneumonia'] if \
                antibiotic_provided_grp == '2nd_line_IV_antibiotics' \
                else p['tf_1st_line_antibiotic_for_severe_pneumonia']

            if 'danger_signs' in symptoms:
                risk_tf_iv_antibiotics *= p['rr_tf_1st_line_antibiotics_if_general_danger_signs']

            if 'respiratory_distress' in symptoms:
                risk_tf_iv_antibiotics *= p['rr_tf_1st_line_antibiotics_if_respiratory_distress']

            if SpO2_level == "<90%":
                risk_tf_iv_antibiotics *= p['rr_tf_1st_line_antibiotics_if_SpO2<90%']

            if disease_type == 'pneumonia':
                risk_tf_iv_antibiotics *= p['rr_tf_1st_line_antibiotics_if_abnormal_CXR']

            # The effect of acute malnutrition
            if un_clinical_acute_malnutrition == 'MAM':
                risk_tf_iv_antibiotics *= p['rr_tf_1st_line_antibiotics_if_MAM']
            elif un_clinical_acute_malnutrition == 'SAM':
                risk_tf_iv_antibiotics *= p['rr_tf_1st_line_antibiotics_if_SAM']

            # The effect of HIV
            if hiv_infected_and_not_on_art:
                risk_tf_iv_antibiotics *= p['rr_tf_1st_line_antibiotics_if_HIV/AIDS']

            return min(1.0, risk_tf_iv_antibiotics)

        def _prob_treatment_fails_when_danger_signs_pneumonia():
            """Return probability treatment fails when the true classification (by symptoms)
             is danger_signs_pneumonia."""

            if antibiotic_provided_grp == '':
                return 1.0  # If no antibiotic is provided the treatment fails

            elif antibiotic_provided_grp in ('1st_line_IV_antibiotics', '2nd_line_IV_antibiotics'):

                # danger_signs_pneumonia given 1st line IV antibiotic:
                iv_tf_ = _treatment_failure_IV_antibiotics()
                iv_tf_ = modify_failure_risk_when_follow_up(iv_tf_)
                iv_tf_ = modify_failure_risk_when_stabilised_with_ox_at_hc(iv_tf_)

                if any([needs_oxygen, might_need_oxygen]) and not oxygen_provided:
                    return modify_failure_risk_when_does_not_get_oxygen_but_needs_oxygen(iv_tf_)
                else:
                    return iv_tf_

            # danger_signs_pneumonia given oral antibiotics (probably due to misdiagnosis) with or without SpO2<90%
            elif antibiotic_provided_grp == 'oral_antibiotics':

                oral_tf_ = min(1.0, _treatment_failure_oral_antibiotics(p['tf_3day_amoxicillin_for_chest_indrawing_with_SpO2>=90%']))

                # # get the TF of IV antibiotics without oxygen provision
                # iv_tf_ = modify_failure_risk_when_does_not_get_oxygen_but_needs_oxygen(
                #     _treatment_failure_IV_antibiotics()) if any([needs_oxygen, might_need_oxygen]) and \
                #                                             not oxygen_provided else _treatment_failure_IV_antibiotics()
                #
                # # adjusted_oral_tf_ = iv_tf_ * (1 / (0.65 * fraction))
                # oral_tf_ = iv_tf_ * 1.3537

                oral_tf_ = modify_failure_risk_when_follow_up(oral_tf_)
                oral_tf_ = modify_failure_risk_when_stabilised_with_ox_at_hc(oral_tf_)

                return oral_tf_
            else:
                raise ValueError(f'Unrecognised antibiotic {antibiotic_provided_grp}')

        def _prob_treatment_fails_when_fast_breathing_pneumonia():
            """Return probability treatment fails when the true classification (by symptoms)
             is fast_breathing_pneumonia."""
            if antibiotic_provided_grp == '':
                return 1.0  # If no antibiotic is provided the treatment fails

            # oral antibiotics
            elif antibiotic_provided_grp == 'oral_antibiotics':
                oral_tf_ = _treatment_failure_oral_antibiotics(
                    p['tf_7day_amoxicillin_for_fast_breathing_pneumonia_in_young_infants']) if \
                    age_exact_years < 1 / 6 else \
                    _treatment_failure_oral_antibiotics(p['tf_3day_amoxicillin_for_fast_breathing_with_SpO2>=90%'])
                oral_tf_ = modify_failure_risk_when_follow_up(oral_tf_)
                oral_tf_ = modify_failure_risk_when_stabilised_with_ox_at_hc(oral_tf_)
                return oral_tf_

            # If symptom-based classification == fast-breathing pneum with SpO2<90% - give IV antibiotics + oxygen
            elif antibiotic_provided_grp in ('1st_line_IV_antibiotics', '2nd_line_IV_antibiotics'):
                # danger_signs_pneumonia given 1st line IV antibiotic:
                iv_tf_ = min(1.0, _treatment_failure_IV_antibiotics())
                iv_tf_ = modify_failure_risk_when_follow_up(iv_tf_)
                iv_tf_ = modify_failure_risk_when_stabilised_with_ox_at_hc(iv_tf_)

                if any([needs_oxygen, might_need_oxygen]) and not oxygen_provided:
                    return modify_failure_risk_when_does_not_get_oxygen_but_needs_oxygen(iv_tf_)
                else:
                    return iv_tf_

                # # base oral TF for SpO2<90% or SpO2>=90% * RR of IV antibiotics
                # rr_tf_iv = p['rr_tf_if_given_1st_line_IV_antibiotics_for_pneumonia_with_SpO2<90%'] if \
                #     needs_oxygen and antibiotic_provided_grp=='1st_line_IV_antibiotics' else \
                #     p['rr_tf_if_given_2nd_line_IV_antibiotics_for_pneumonia_with_SpO2<90%'] if \
                #         needs_oxygen and antibiotic_provided_grp=='2nd_line_IV_antibiotics' else \
                #         p['rr_tf_if_given_1st_line_IV_antibiotics_for_pneumonia_with_SpO2>=90%'] if \
                #             might_need_oxygen and antibiotic_provided_grp=='1st_line_IV_antibiotics' else \
                #     p['rr_tf_if_given_2nd_line_IV_antibiotics_for_pneumonia_with_SpO2>=90%'] if \
                #         might_need_oxygen and antibiotic_provided_grp=='2nd_line_IV_antibiotics' else 0.8025 if \
                #         antibiotic_provided_grp=='1st_line_IV_antibiotics' else 0.91485
                #
                # iv_tf_ = _treatment_failure_oral_antibiotics(
                #     p['tf_7day_amoxicillin_for_fast_breathing_pneumonia_in_young_infants']) * rr_tf_iv if \
                #     age_exact_years < 1 / 6 else \
                #     _treatment_failure_oral_antibiotics(
                #         p['tf_3day_amoxicillin_for_fast_breathing_with_SpO2>=90%']) * rr_tf_iv
                #
                # iv_tf_ = min(1.0, iv_tf_)
                # iv_tf_ = modify_failure_risk_when_follow_up(iv_tf_)
                # iv_tf_ = modify_failure_risk_when_stabilised_with_ox_at_hc(iv_tf_)
                #
                # # adjust the TF base on oxygen provision
                # if any([needs_oxygen, might_need_oxygen]) and not oxygen_provided:
                #     return modify_failure_risk_when_does_not_get_oxygen_but_needs_oxygen(iv_tf_)
                # else:
                #     return iv_tf_

            else:
                raise ValueError(f'Unrecognised antibiotic{antibiotic_provided_grp}.')

        def _prob_treatment_fails_when_chest_indrawing_pneumonia():
            """Return probability treatment fails when the true classification (by symptoms)
            is chest_indrawing_pneumonia."""

            if antibiotic_provided_grp == '':
                return 1.0  # If no antibiotic is provided the treatment fails

            # oral antibiotics
            elif antibiotic_provided_grp == 'oral_antibiotics':
                if antibiotic_provided == 'Amoxicillin_tablet_or_suspension_3days':
                    oral_tf_3days = _treatment_failure_oral_antibiotics(
                        p['tf_3day_amoxicillin_for_chest_indrawing_with_SpO2>=90%'])
                    oral_tf_3days = modify_failure_risk_when_follow_up(oral_tf_3days)
                    oral_tf_3days = modify_failure_risk_when_stabilised_with_ox_at_hc(oral_tf_3days)
                    return oral_tf_3days

                elif antibiotic_provided == 'Amoxicillin_tablet_or_suspension_5days':
                    oral_tf_5days = _treatment_failure_oral_antibiotics(
                        p['tf_5day_amoxicillin_for_chest_indrawing_with_SpO2>=90%'])
                    oral_tf_5days = modify_failure_risk_when_follow_up(oral_tf_5days)
                    oral_tf_5days = modify_failure_risk_when_stabilised_with_ox_at_hc(oral_tf_5days)
                    return oral_tf_5days
                else:
                    raise ValueError(f'Unrecognised antibiotic{antibiotic_provided}.')

            # 1st line IV antibiotic:
            elif antibiotic_provided_grp in ('1st_line_IV_antibiotics', '2nd_line_IV_antibiotics'):
                # danger_signs_pneumonia given 1st line IV antibiotic:
                iv_tf_ = min(1.0, _treatment_failure_IV_antibiotics())
                iv_tf_ = modify_failure_risk_when_follow_up(iv_tf_)
                iv_tf_ = modify_failure_risk_when_stabilised_with_ox_at_hc(iv_tf_)

                if any([needs_oxygen, might_need_oxygen]) and not oxygen_provided:
                    return modify_failure_risk_when_does_not_get_oxygen_but_needs_oxygen(iv_tf_)
                else:
                    return iv_tf_

                # # base oral TF for SpO2<90% or SpO2>=90% * RR of IV antibiotics
                # rr_tf_iv = p['rr_tf_if_given_1st_line_IV_antibiotics_for_pneumonia_with_SpO2<90%'] if \
                #     needs_oxygen and antibiotic_provided_grp=='1st_line_IV_antibiotics' else \
                #     p['rr_tf_if_given_2nd_line_IV_antibiotics_for_pneumonia_with_SpO2<90%'] if \
                #         needs_oxygen and antibiotic_provided_grp=='2nd_line_IV_antibiotics' else \
                #         p['rr_tf_if_given_1st_line_IV_antibiotics_for_pneumonia_with_SpO2>=90%'] if \
                #             might_need_oxygen and antibiotic_provided_grp=='1st_line_IV_antibiotics' else \
                #             p['rr_tf_if_given_2nd_line_IV_antibiotics_for_pneumonia_with_SpO2>=90%'] if \
                #                 might_need_oxygen and antibiotic_provided_grp=='2nd_line_IV_antibiotics' else 0.8025 if \
                #                 antibiotic_provided_grp=='1st_line_IV_antibiotics' else 0.91485
                #
                # iv_tf_ = _treatment_failure_oral_antibiotics(
                #     p['tf_5day_amoxicillin_for_chest_indrawing_with_SpO2>=90%']) * rr_tf_iv
                #
                # iv_tf_ = min(1.0, iv_tf_)
                # iv_tf_ = modify_failure_risk_when_follow_up(iv_tf_)
                # iv_tf_ = modify_failure_risk_when_stabilised_with_ox_at_hc(iv_tf_)
                #
                # # adjust the TF base on oxygen provision
                # if any([needs_oxygen, might_need_oxygen]) and not oxygen_provided:
                #     return modify_failure_risk_when_does_not_get_oxygen_but_needs_oxygen(iv_tf_)
                # else:
                #     return iv_tf_

            else:
                raise ValueError(f'Unrecognised antibiotic{antibiotic_provided_grp}.')

        def _prob_treatment_fails_when_cough_or_cold():
            """Return probability treatment fails when the true classification (by symptoms) is "cough_or_cold."""
            if antibiotic_provided_grp == '':
                return 1.0

            # oral antibiotics (use TF for most non-severe pneumonia treatment)
            elif antibiotic_provided_grp == 'oral_antibiotics':
                oral_tf_ = _treatment_failure_oral_antibiotics(
                    p['tf_3day_amoxicillin_for_fast_breathing_with_SpO2>=90%'])
                oral_tf_ = modify_failure_risk_when_follow_up(oral_tf_)
                oral_tf_ = modify_failure_risk_when_stabilised_with_ox_at_hc(oral_tf_)
                return oral_tf_

            # If symptom-based classification == cough/cold classification with SpO2<90% - give IV antibiotics + oxygen
            elif antibiotic_provided_grp in ('1st_line_IV_antibiotics', '2nd_line_IV_antibiotics'):
                # danger_signs_pneumonia given 1st line IV antibiotic:
                iv_tf_ = min(1.0, _treatment_failure_IV_antibiotics())
                iv_tf_ = modify_failure_risk_when_follow_up(iv_tf_)
                iv_tf_ = modify_failure_risk_when_stabilised_with_ox_at_hc(iv_tf_)

                if any([needs_oxygen, might_need_oxygen]) and not oxygen_provided:
                    return modify_failure_risk_when_does_not_get_oxygen_but_needs_oxygen(iv_tf_)
                else:
                    return iv_tf_

                # # base oral TF for SpO2<90% or SpO2>=90% * RR of IV antibiotics
                # rr_tf_iv = p['rr_tf_if_given_1st_line_IV_antibiotics_for_pneumonia_with_SpO2<90%'] if \
                #     needs_oxygen and antibiotic_provided_grp=='1st_line_IV_antibiotics' else \
                #     p['rr_tf_if_given_2nd_line_IV_antibiotics_for_pneumonia_with_SpO2<90%'] if \
                #         needs_oxygen and antibiotic_provided_grp=='2nd_line_IV_antibiotics' else \
                #         p['rr_tf_if_given_1st_line_IV_antibiotics_for_pneumonia_with_SpO2>=90%'] if \
                #             might_need_oxygen and antibiotic_provided_grp=='1st_line_IV_antibiotics' else \
                #             p['rr_tf_if_given_2nd_line_IV_antibiotics_for_pneumonia_with_SpO2>=90%'] if \
                #                 might_need_oxygen and antibiotic_provided_grp=='2nd_line_IV_antibiotics' else 0.8025 if \
                #                 antibiotic_provided_grp=='1st_line_IV_antibiotics' else 0.91485
                #
                # iv_tf_ = _treatment_failure_oral_antibiotics(
                #     p['tf_3day_amoxicillin_for_fast_breathing_with_SpO2>=90%']) * rr_tf_iv
                #
                # iv_tf_ = min(1.0, iv_tf_)
                # iv_tf_ = modify_failure_risk_when_follow_up(iv_tf_)
                # iv_tf_ = modify_failure_risk_when_stabilised_with_ox_at_hc(iv_tf_)
                #
                # # adjust the TF base on oxygen provision
                # if any([needs_oxygen, might_need_oxygen]) and not oxygen_provided:
                #     return modify_failure_risk_when_does_not_get_oxygen_but_needs_oxygen(iv_tf_)
                # else:
                #     return iv_tf_

            else:
                raise ValueError(f'Unrecognised antibiotic{antibiotic_provided_grp}.')

        if imci_symptom_based_classification == 'danger_signs_pneumonia':
            return min(1.0, _prob_treatment_fails_when_danger_signs_pneumonia())

        elif imci_symptom_based_classification == 'fast_breathing_pneumonia':
            return min(1.0, _prob_treatment_fails_when_fast_breathing_pneumonia())

        elif imci_symptom_based_classification == 'chest_indrawing_pneumonia':
            return min(1.0, _prob_treatment_fails_when_chest_indrawing_pneumonia())

        elif imci_symptom_based_classification == 'cough_or_cold':
            return min(1.0, _prob_treatment_fails_when_cough_or_cold())

        else:
            raise ValueError('Unrecognised imci_symptom_based_classification.')


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
        return (self.sim.date + self.frequency - self.sim.date) / pd.Timedelta(days=DAYS_IN_YEAR)

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
        duration_in_days_of_alri = rng.randint(1, params['max_alri_duration_in_days_without_treatment'])

        # Date for outcome (either recovery or death) with uncomplicated Alri
        date_of_outcome = date + DateOffset(days=duration_in_days_of_alri)

        # Define 'episode end' date. This the date when this episode ends. It is the last possible data that any HSI
        # could affect this episode.
        episode_end = date_of_outcome + DateOffset(days=params['days_between_treatment_and_cure'])

        # ----------------------------------- Complications  -----------------------------------
        complications = models.get_complications_that_onset(
            disease_type=disease_type,
            primary_path_is_bacterial=(pathogen in self.module.pathogens['bacterial']),
            has_secondary_bacterial_inf=pd.notnull(bacterial_coinfection)
        )

        oxygen_saturation = models.get_oxygen_saturation(complication_set=complications)

        # ----------------------------------- Clinical Symptoms -----------------------------------
        # apply signs and symptoms for uncomplicated cases or for complicated cases
        all_symptoms = \
            models.symptoms_for_uncomplicated_disease(disease_type=disease_type) if len(complications) == 0 else \
                models.symptoms_for_complicated_disease(disease_type=disease_type, complications=complications,
                                                        oxygen_saturation=oxygen_saturation)

        # ----------------------------------- Whether Will Die  -----------------------------------
        will_die = models.will_die_of_alri(
            age_exact_years=age_exact_years,
            sex=sex,
            bacterial_infection=any(
                b_patho in [pathogen, bacterial_coinfection] for b_patho in self.module.pathogens['bacterial']),
            disease_type=disease_type,
            SpO2_level=oxygen_saturation,
            complications=complications,
            all_symptoms=all_symptoms,
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
        if 'bacteraemia' in chars['complications']:
            self.module.logging_event.new_systemic_complication_case()
        if 'hypoxaemia' in chars['complications']:
            self.module.logging_event.new_hypoxaemic_case()

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
                 is_followup_following_treatment_failure: bool = False):
        super().__init__(module, person_id=person_id)
        self._treatment_id_stub = 'Alri_Pneumonia_Treatment'
        self._facility_levels = ("0", "1a", "1b", "2")  # Health facility levels at which care may be provided
        assert facility_level in self._facility_levels
        self.is_followup_following_treatment_failure = is_followup_following_treatment_failure

        if not inpatient:
            self._as_out_patient(facility_level)
        else:
            self._as_in_patient(facility_level)

        if person_id is not None:
            self._age_exact_years = self.sim.population.props.at[person_id, 'age_exact_years']

    def _as_out_patient(self, facility_level):
        """Cast this HSI as an out-patient appointment."""
        self.TREATMENT_ID = \
            f'{self._treatment_id_stub}_Outpatient{"_Followup" if self.is_followup_following_treatment_failure else ""}'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({
            ('ConWithDCSA' if facility_level == '0' else 'Under5OPD'): 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level

    def _as_in_patient(self, facility_level):
        """Cast this HSI as an in-patient appointment."""
        self.TREATMENT_ID = \
            f'{self._treatment_id_stub}_Inpatient{"_Followup" if self.is_followup_following_treatment_failure else ""}'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.ACCEPTED_FACILITY_LEVEL = facility_level
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 7})

    def _refer_to_next_level_up(self):
        """Schedule this event to occur again today at the next level-up (if there is a next level-up)."""

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

    def _schedule_follow_up_following_treatment_failure(self):
        """Schedule a copy of this event to occur in 5 days time as a 'follow-up' appointment at this level
        (if above "0") and as an in-patient (for most cases).
        (rare outpatient at follow-up, only if no treatment was given at initial appointment)."""
        self.sim.modules['HealthSystem'].schedule_hsi_event(
            HSI_Alri_Treatment(
                module=self.module,
                person_id=self.target,
                inpatient=self._is_as_in_patient,
                facility_level=self.ACCEPTED_FACILITY_LEVEL if self.ACCEPTED_FACILITY_LEVEL != "0" else "1a",
                is_followup_following_treatment_failure=True,
            ),
            topen=self.sim.date + pd.DateOffset(days=5),
            tclose=None,
            priority=0)

    @property
    def _is_as_in_patient(self):
        """True if this HSI_Event is cast as an in-patient appointment"""
        return 'Inpatient' in self.TREATMENT_ID

    def _get_cons(self, _arg: Union[str, Tuple[str]]) -> bool:
        """Checks availability of group (or groups) of consumables identified by a string (or tuple of strings).
         * If the argument is a string, True is returned if _all_ of the item_codes in that group are available.
         * If the argument is a tuple of strings, True is returned if, for _any_ of the groups identified by the
           strings, _all_ of the item_codes in that group are available. This is suitable if there is a preferred group
           of consumables to use, but one or more 'back-ups' that could be used if the first is not available."""

        if isinstance(_arg, str):
            return self._get_cons_group(item_group_str=_arg)
        elif isinstance(_arg, tuple):
            return self._get_cons_with_backups(tuple_of_item_group_str=_arg)
        else:
            raise ValueError("Argument is neither a tuple not a string.")

    def _get_cons_group(self, item_group_str: str) -> bool:
        """True if _all_ of a group of consumables (identified by a string) is available."""
        if item_group_str is not None:
            return self.get_consumables(
                item_codes={
                    k: v(self._age_exact_years) if isinstance(v, types.LambdaType) else v
                    for k, v in self.module.consumables_used_in_hsi[item_group_str].items()
                })
        else:
            raise ValueError('String for the group of consumables not provided')

    def _get_cons_with_backups(self, tuple_of_item_group_str: Tuple[str]):
        """Returns True if `_get_cons_group` is True for any in a tuple of strings (each identifying a group of
        consumables). It works by attempting `_get_cons_group` with the first entry in a tuple of strings; and only
        tries the next entry if it is not available; repeats until _get_cons_group is found to be True for an entry."""
        for _item_group_str in tuple_of_item_group_str:
            if self._get_cons_group(_item_group_str):
                return True  # _get_cons is True for at least one entry
        return False  # _get_cons_group is not True for any entry.

    def _assess_and_treat(self, age_exact_years, oxygen_saturation, symptoms,
                          hiv_infected_and_not_on_art, un_clinical_acute_malnutrition):
        """This routine is called in every HSI. It classifies the disease of the child and commissions treatment
        accordingly."""

        # pulse_oximeter_available = self._get_cons('Pulse_oximetry')

        if not self._is_as_in_patient:

            pulse_oximeter_available = self._get_cons('Pulse_oximetry')
            # Assessment process if not an in-patient:
            classification_for_treatment_decision = self._get_disease_classification_for_treatment_decision(
                age_exact_years=age_exact_years, symptoms=symptoms, oxygen_saturation=oxygen_saturation,
                facility_level=self.ACCEPTED_FACILITY_LEVEL, use_oximeter=pulse_oximeter_available,
                hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
                un_clinical_acute_malnutrition=un_clinical_acute_malnutrition)

            self._provide_bronchodilator_if_wheeze(
                facility_level=self.ACCEPTED_FACILITY_LEVEL,
                symptoms=symptoms,
            )

            self._do_action_given_classification(
                classification_for_treatment_decision=classification_for_treatment_decision,
                age_exact_years=age_exact_years,
                facility_level=self.ACCEPTED_FACILITY_LEVEL,
                use_oximeter=pulse_oximeter_available,
                oxygen_saturation=oxygen_saturation,
            )

        else:
            # For in-patients, provide treatments as though classification_for_treatment_decision =
            # 'danger_signs_pneumonia', as this is the reasonable clinical presumption for in-patients.

            self._do_action_given_classification(
                classification_for_treatment_decision='danger_signs_pneumonia',  # assumed for sbi for < 2 months
                age_exact_years=age_exact_years,
                facility_level=self.ACCEPTED_FACILITY_LEVEL,
                use_oximeter=self._get_cons('Pulse_oximetry'),
                oxygen_saturation=oxygen_saturation,
            )

    def _has_staph_aureus(self):
        """Returns True if the person has Staph. aureus as either primary or secondary infection"""
        person_id = self.target
        infections = self.sim.population.props.loc[
            person_id, ['ri_primary_pathogen', 'ri_secondary_bacterial_pathogen']
        ].to_list()
        return 'Staph_aureus' in infections

    def _get_imci_classification_based_on_symptoms(self, child_is_younger_than_2_months: bool, symptoms: list,
                                                   facility_level: str,
                                                   hiv_infected_and_not_on_art: bool,
                                                   un_clinical_acute_malnutrition: str) -> str:
        """Based on age and symptoms, classify WHO-pneumonia severity. This is regarded as the *TRUE* classification
         based on symptoms. It will return one of: {
             'fast_breathing_pneumonia',
             'danger_signs_pneumonia',
             'chest_indrawing_pneumonia,
             'cough_or_cold'
        }."""
        # person_id = self.target
        # if self.sim.population.props.loc[person_id, 'un_clinical_acute_malnutrition'] == 'SAM':
        #     return 'danger_signs_pneumonia'
        # else:
        return self.module.get_imci_classification_based_on_symptoms(
            child_is_younger_than_2_months=child_is_younger_than_2_months, symptoms=symptoms,
            facility_level=facility_level,
            hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
            un_clinical_acute_malnutrition=un_clinical_acute_malnutrition)

    # @staticmethod
    def _get_imci_classification_by_SpO2_measure(self, oxygen_saturation: bool) -> str:
        """Return classification based on age and oxygen_saturation. It will return one of: {
             'danger_signs_pneumonia',          <-- implies needs oxygen
             ''                                 <-- implies does not need oxygen
        }."""

        p = self.module.parameters
        # Change this variable when looking into the benefit of using SpO2 <93% as indication for oxygen
        oxygen_needed = oxygen_saturation == '<90%' if p['apply_oxygen_indication_to_SpO2_measurement'] == '<90%' \
            else (
            (not oxygen_saturation == '>=93%') if p['apply_oxygen_indication_to_SpO2_measurement'] == '<93%' else False)

        if oxygen_needed:
            return 'danger_signs_pneumonia'
        else:
            return ''

        # # SpO2<90% apply severe classification to get inpatient care
        # if oxygen_needed:
        #     return 'danger_signs_pneumonia'
        # else:
        #     return ''

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

        # applying rng here to avoid seed inconsistency

        # classifications at the facility level 0 ---------
        # hw performance in correctly classifying severe pneumonia
        correct_hw_classification_severe_pneumonia_level0 = \
            rand() < p['sensitivity_of_classification_of_danger_signs_pneumonia_facility_level0']
        # alternative classifications given to danger signs pneumonia
        incorrect_classification_given_to_severe_pneumonia_level0 = rand_choice(
            ['fast_breathing_pneumonia', 'cough_or_cold'],
            p=[p['prob_iCCM_severe_pneumonia_treated_as_fast_breathing_pneumonia'],
                1.0 - p['prob_iCCM_severe_pneumonia_treated_as_fast_breathing_pneumonia']])

        # hw performance in correctly classifying fast-breathing pneumonia
        correct_hw_classification_nonsev_pneumonia_level0 = \
            rand() < p['sensitivity_of_classification_of_fast_breathing_pneumonia_facility_level0']

        # classifications at the facility level 1 ---------
        # hw performance in correctly classifying severe pneumonia
        correct_hw_classification_severe_pneumonia_level1 = \
            rand() < p['sensitivity_of_classification_of_severe_pneumonia_facility_level1']
        # alternative classifications given to danger signs pneumonia
        incorrect_classification_given_to_severe_pneumonia_level1 = rand_choice(
            ['chest_indrawing_pneumonia', 'cough_or_cold'],
            p=[p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia'],
                1.0 - p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia']])
        # hw performance in correctly classifying fast-breathing pneumonia
        correct_hw_classification_nonsev_pneumonia_level1 = \
            rand() < p['sensitivity_of_classification_of_non_severe_pneumonia_facility_level1']

        # classifications at the facility level 2 ---------
        # hw performance in correctly classifying severe pneumonia
        correct_hw_classification_severe_pneumonia_level2 = \
            rand() < p['sensitivity_of_classification_of_severe_pneumonia_facility_level2']
        # alternative classifications given to danger signs pneumonia
        incorrect_classification_given_to_severe_pneumonia_level2 = rand_choice(
            ['chest_indrawing_pneumonia', 'cough_or_cold'],
            p=[p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia'],
               1.0 - p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia']
               ])
        # hw performance in correctly classifying fast-breathing pneumonia
        correct_hw_classification_nonsev_pneumonia_level2 = \
            rand() < p['sensitivity_of_classification_of_non_severe_pneumonia_facility_level2']

        def _classification_at_facility_level_0(imci_classification_based_on_symptoms):
            """Return classification if it does at facility level 0"""
            if imci_classification_based_on_symptoms in ('chest_indrawing_pneumonia', 'danger_signs_pneumonia'):
                if correct_hw_classification_severe_pneumonia_level0:
                    return imci_classification_based_on_symptoms
                else:
                    return incorrect_classification_given_to_severe_pneumonia_level0
                    # return rand_choice(
                    #     ['fast_breathing_pneumonia', 'cough_or_cold'],
                    #     p=[
                    #         p['prob_iCCM_severe_pneumonia_treated_as_fast_breathing_pneumonia'],
                    #         1.0 - p['prob_iCCM_severe_pneumonia_treated_as_fast_breathing_pneumonia']
                    #     ]
                    # )

            elif imci_classification_based_on_symptoms == 'fast_breathing_pneumonia':
                if correct_hw_classification_nonsev_pneumonia_level0:
                    return imci_classification_based_on_symptoms
                else:
                    return 'cough_or_cold'

            elif imci_classification_based_on_symptoms == "cough_or_cold":
                return "cough_or_cold"

            else:
                raise ValueError(f'Classification not recognised: {imci_classification_based_on_symptoms}')

        def _classification_at_facility_level_1(imci_classification_based_on_symptoms):
            """Return classification if it does at facility level 1"""
            if imci_classification_based_on_symptoms == 'danger_signs_pneumonia':
                if correct_hw_classification_severe_pneumonia_level1:
                    return imci_classification_based_on_symptoms
                else:
                    return incorrect_classification_given_to_severe_pneumonia_level1
                    # return rand_choice(
                    #     ['chest_indrawing_pneumonia', 'cough_or_cold'],
                    #     p=[
                    #         p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia'],
                    #         1.0 - p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia']
                    #     ]
                    # )

            elif imci_classification_based_on_symptoms in ('fast_breathing_pneumonia', 'chest_indrawing_pneumonia'):
                if correct_hw_classification_nonsev_pneumonia_level1:
                    return imci_classification_based_on_symptoms
                else:
                    return 'cough_or_cold'

            elif imci_classification_based_on_symptoms == "cough_or_cold":
                return "cough_or_cold"

            else:
                raise ValueError(f'Classification not recognised: {imci_classification_based_on_symptoms}')

        def _classification_at_facility_level_2(imci_classification_based_on_symptoms):
            """Return classification if it does at facility level 2"""
            if imci_classification_based_on_symptoms == 'danger_signs_pneumonia':
                if correct_hw_classification_severe_pneumonia_level2:
                    return imci_classification_based_on_symptoms
                else:
                    return incorrect_classification_given_to_severe_pneumonia_level2
                    # return rand_choice(
                    #     ['chest_indrawing_pneumonia', 'cough_or_cold'],
                    #     p=[
                    #         p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia'],
                    #         1.0 - p['prob_IMCI_severe_pneumonia_treated_as_non_severe_pneumonia']
                    #     ]
                    # )

            elif imci_classification_based_on_symptoms in ('fast_breathing_pneumonia', 'chest_indrawing_pneumonia'):
                if correct_hw_classification_nonsev_pneumonia_level2:
                    return imci_classification_based_on_symptoms
                else:
                    return 'cough_or_cold'

            elif imci_classification_based_on_symptoms == "cough_or_cold":
                return "cough_or_cold"

            else:
                raise ValueError(f'Classification not recognised: {imci_classification_based_on_symptoms}')

        if facility_level == "0":
            classification = _classification_at_facility_level_0(imci_classification_based_on_symptoms)
        elif facility_level in ("1a", "1b"):
            classification = _classification_at_facility_level_1(imci_classification_based_on_symptoms)
        elif facility_level == "2":
            classification = _classification_at_facility_level_2(imci_classification_based_on_symptoms)
        else:
            raise ValueError(f"Facility Level not recognised: {facility_level}")

        assert classification in self.module.classifications
        return classification

    def _get_disease_classification_for_treatment_decision(self,
                                                           age_exact_years,
                                                           symptoms,
                                                           oxygen_saturation,
                                                           facility_level,
                                                           use_oximeter,
                                                           hiv_infected_and_not_on_art,
                                                           un_clinical_acute_malnutrition
                                                           ) -> str:
        """Returns the classification of disease for the purpose of treatment, which may be based on the results of the
         pulse oximetry (if available) or the health worker's own classification. It will be one of: {
                 'danger_signs_pneumonia',          (symptoms-based assessment OR spO2 assessment): implies need oxygen
                 'fast_breathing_pneumonia',        (symptoms-based assessment)
                 'chest_indrawing_pneumonia',       (symptoms-based assessment)
                 'cough_or_cold'                    (symptoms-based assessment)
         }.
         :param hiv_infected_and_not_on_art:
         :param un_clinical_acute_malnutrition: """
        p = self.module.parameters
        rand = self.module.rng.random_sample

        child_is_younger_than_2_months = age_exact_years < (2.0 / 12.0)

        imci_classification_based_on_symptoms = self._get_imci_classification_based_on_symptoms(
            child_is_younger_than_2_months=child_is_younger_than_2_months,
            symptoms=symptoms, facility_level=facility_level,
            hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
            un_clinical_acute_malnutrition=un_clinical_acute_malnutrition)

        hw_assigned_classification = self._get_classification_given_by_health_worker(
            imci_classification_based_on_symptoms=imci_classification_based_on_symptoms,
            facility_level=facility_level)

        imci_classification_by_SpO2_measure = self._get_imci_classification_by_SpO2_measure(
            oxygen_saturation=oxygen_saturation)

        # Final classification gives precedence to the spO2 measure if it can be used:
        consistent_po_use = rand() < p['pulse_oximeter_usage_rate']

        _classification = imci_classification_by_SpO2_measure \
            if (use_oximeter and consistent_po_use and
                (imci_classification_by_SpO2_measure != '')) else hw_assigned_classification

        assert _classification in self.module.classifications

        return _classification

    def _do_action_given_classification(self,
                                        classification_for_treatment_decision,
                                        age_exact_years,
                                        facility_level,
                                        use_oximeter,
                                        oxygen_saturation,
                                        ):
        """Do the actions that are required given a particular classification and the current facility level. This
        entails referrals upwards and/or admission at in-patient, and when at the appropriate level, trying to provide
        the ideal treatment."""
        p = self.module.parameters
        rng = self.module.rng

        def _provide_consumable_and_refer(cons: str) -> None:
            """Provide a consumable (ignoring availability) and refer patient to next level up."""
            if cons is not None:
                _ = self._get_cons(cons)
            self._refer_to_next_level_up()

        def _try_treatment(antibiotic_indicated: Tuple[str], oxygen_indicated: bool) -> None:
            """Try to provide a `treatment_indicated` and refer to next level if the consumables are not available."""

            assert [_antibiotic in self.module.antibiotics + [''] for _antibiotic in antibiotic_indicated]

            # antibiotic_available = True
            # antibiotic_provided = antibiotic_indicated[0] if antibiotic_available else ''

            if antibiotic_indicated[0] != '':
                antibiotic_available = self._get_cons(antibiotic_indicated)
                antibiotic_provided = antibiotic_indicated[0] if antibiotic_available else ''

            else:
                antibiotic_available = True
                antibiotic_provided = ''

            oxygen_available = self._get_cons('Oxygen_Therapy')
            oxygen_provided = (oxygen_available and oxygen_indicated)

            all_things_needed_available = antibiotic_available and (
                (oxygen_available and oxygen_indicated) or (not oxygen_indicated)
            )

            if (not all_things_needed_available) and (facility_level != '2'):
                # Something that is needed is not available -> Refer up if possible.
                self._refer_to_next_level_up()

            else:
                # Provide treatment with what is available
                treatment_outcome = self.module.do_effects_of_treatment_and_return_outcome(
                    person_id=self.target,
                    antibiotic_provided=antibiotic_provided,
                    oxygen_provided=oxygen_provided
                )
                if treatment_outcome == 'failure':
                    # for inpatients provide 2nd line IV antibiotic if 1st line failed
                    if antibiotic_provided.startswith('1st_line_IV'):
                        # if Staph is suspected
                        if self._has_staph_aureus():
                            second_line_available = self._get_cons('2nd_line_IV_flucloxacillin_gentamicin')
                            if second_line_available:
                                self.module.do_effects_of_treatment_and_return_outcome(
                                    person_id=self.target,
                                    antibiotic_provided='2nd_line_IV_flucloxacillin_gentamicin',
                                    oxygen_provided=oxygen_provided
                                )
                        else:
                            # if 1st line IV fails
                            second_line_available = self._get_cons('2nd_line_IV_ceftriaxone')
                            if second_line_available:
                                self.module.do_effects_of_treatment_and_return_outcome(
                                    person_id=self.target,
                                    antibiotic_provided='2nd_line_IV_ceftriaxone',
                                    oxygen_provided=oxygen_provided
                                )

                    # oral antibiotics failure - seek follow-up appointment
                    elif antibiotic_provided.startswith('Amoxicillin_tablet'):
                        if not self.is_followup_following_treatment_failure:
                            # apply a 30% will follow-up if oral treatment fails for those without a schedule recovery date
                            if self.sim.population.props.loc[self.target, 'ri_scheduled_recovery_date'] == pd.NaT:
                                if 0.5 > rng.random_sample():
                                    self._schedule_follow_up_following_treatment_failure()

        def _do_if_fast_breathing_pneumonia():
            """What to do if classification is `fast_breathing`."""

            _try_treatment(
                **self.module._ultimate_treatment_indicated_for_patient(
                    classification_for_treatment_decision='fast_breathing_pneumonia',
                    age_exact_years=age_exact_years,
                    facility_level=facility_level,
                    oxygen_saturation=oxygen_saturation
                )
            )

        def _do_if_chest_indrawing_pneumonia():
            """What to do if classification is `chest_indrawing_pneumonia`."""
            if facility_level == '0':
                _provide_consumable_and_refer('First_dose_oral_amoxicillin_for_referral')
            else:
                _try_treatment(
                    **self.module._ultimate_treatment_indicated_for_patient(
                        classification_for_treatment_decision='chest_indrawing_pneumonia',
                        age_exact_years=age_exact_years,
                        facility_level=facility_level,
                        oxygen_saturation=oxygen_saturation
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
                        **self.module._ultimate_treatment_indicated_for_patient(
                            classification_for_treatment_decision='danger_signs_pneumonia',
                            age_exact_years=age_exact_years,
                            facility_level=facility_level,
                            oxygen_saturation=oxygen_saturation
                        )
                    )

        def _do_if_cough_or_cold():
            """What to do if `cough_or_cold`."""
            _try_treatment(
                **self.module._ultimate_treatment_indicated_for_patient(
                    classification_for_treatment_decision='cough_or_cold',
                    age_exact_years=age_exact_years,
                    facility_level=facility_level,
                    oxygen_saturation=oxygen_saturation
                )
            )

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

    def _do_on_follow_up_following_treatment_failure(self, person_id):
        """Things to do for a patient who is having this HSI following a failure of an earlier treatment.
        A further drug will be used but this will have no effect on the chance of the person dying."""

        p = self.module.parameters
        # Do nothing if the person is not currently infected and currently experiencing an episode
        person = self.sim.population.props.loc[person_id]
        if not person.ri_current_infection_status and (
            person.ri_start_of_current_episode <= self.sim.date <= person.ri_end_of_current_episode
        ):
            return

        # HIV status, not on ART
        hiv_infected_and_not_on_art = person.hv_inf and (person.hv_art != "on_VL_suppressed")
        # acute malnutrition status
        un_clinical_acute_malnutrition = person.un_clinical_acute_malnutrition

        pulse_oximeter_available = self._get_cons('Pulse_oximetry')
        # Assessment process if not an in-patient:
        classification_for_treatment_decision = self._get_disease_classification_for_treatment_decision(
            age_exact_years=person.age_exact_years, symptoms=person.symptoms,
            oxygen_saturation=person.oxygen_saturation, facility_level=self.ACCEPTED_FACILITY_LEVEL,
            use_oximeter=pulse_oximeter_available, hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
            un_clinical_acute_malnutrition=un_clinical_acute_malnutrition)
        self._provide_bronchodilator_if_wheeze(
            facility_level=self.ACCEPTED_FACILITY_LEVEL,
            symptoms=person.symptoms,
        )

        # get a follow-up classification
        followup_classification = ''
        # cough or cold, and fast-breathing pneumonia classifications will be given oral antibiotics, treat outpatient
        if classification_for_treatment_decision in ('cough_or_cold', 'fast_breathing_pneumonia'):
            followup_classification = 'chest_indrawing_pneumonia'
        # chest-indrawing and danger signs pneumonia classifications will be given inpatient care
        else:
            followup_classification = 'danger_signs_pneumonia'

        # do the effect of treatment - cure determine in the function
        self._do_action_given_classification(
            classification_for_treatment_decision=followup_classification,
            age_exact_years=person.age_exact_years,
            facility_level=self.ACCEPTED_FACILITY_LEVEL,
            use_oximeter=pulse_oximeter_available,
            oxygen_saturation=person.oxygen_saturation,
        )

    def apply(self, person_id, squeeze_factor):
        """Assess and attempt to treat the person."""

        if not self.is_followup_following_treatment_failure:

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

            # HIV status, not on ART
            hiv_infected_and_not_on_art = person.hv_inf and (person.hv_art != "on_VL_suppressed")
            # acute malnutrition status
            un_clinical_acute_malnutrition = person.un_clinical_acute_malnutrition

            # Attempt treatment:
            self._assess_and_treat(age_exact_years=person.age_exact_years,
                                   symptoms=symptoms,
                                   oxygen_saturation=person.ri_SpO2_level,
                                   hiv_infected_and_not_on_art=hiv_infected_and_not_on_art,
                                   un_clinical_acute_malnutrition=un_clinical_acute_malnutrition

                                   )

        else:
            self._do_on_follow_up_following_treatment_failure(person_id)

    def never_ran(self):
        """If this event never ran, refer to next level up."""
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

        self.trackers = {
            'incident_cases': Tracker(age_grps=age_grps, pathogens=self.module.all_pathogens),
            'recovered_cases': Tracker(age_grps=age_grps, pathogens=self.module.all_pathogens),
            'cured_cases': Tracker(age_grps=age_grps, pathogens=self.module.all_pathogens),
            'deaths': Tracker(age_grps=age_grps, pathogens=self.module.all_pathogens),
            'deaths_among_persons_with_SpO2<90%': Tracker(),
            'seeking_care': Tracker(),
            'treated': Tracker(),
            'pulmonary_complication_cases': Tracker(),
            'systemic_complication_cases': Tracker(),
            'hypoxaemic_cases': Tracker()
        }

    def new_case(self, **kwargs):
        self.trackers['incident_cases'].add_one(**kwargs)

    def new_recovered_case(self, **kwargs):
        self.trackers['recovered_cases'].add_one(**kwargs)

    def new_cured_case(self, **kwargs):
        self.trackers['cured_cases'].add_one(**kwargs)

    def new_death(self, **kwargs):
        self.trackers['deaths'].add_one(age=kwargs['age'], pathogen=kwargs['pathogen'])
        if kwargs['sp02_level'] == '<90%':
            self.trackers['deaths_among_persons_with_SpO2<90%'].add_one()

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

    def __init__(self, age_grps: Dict = None, pathogens: List = None):
        """Create and initialise tracker"""

        # Check and store parameters
        self.pathogens = [] if pathogens is None else pathogens
        self.age_grps_lookup = {} if age_grps is None else age_grps
        self.unique_age_grps = sorted(set(self.age_grps_lookup.values()))

        # Initialise Tracker
        self.tracker = None
        self.reset()

    def reset(self):
        """Produce a dict of the form: { <Age-Grp>: {<Pathogen>: <Count>} } if age-groups and pathogens are specified;
        otherwise the tracker is an integer."""

        # if collections are not empty
        if self.unique_age_grps and self.pathogens:
            self.tracker = {
                age: dict(zip(self.pathogens, [0] * len(self.pathogens))) for age in self.unique_age_grps
            }
        else:
            self.tracker = 0

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
        'hv_inf': Property(Types.BOOL, 'DUMMY version of the property for hv_inf'),
        'hv_art': Property(Types.CATEGORICAL, 'DUMMY version of the property for hv_art.',
                           categories=['not', 'on_VL_suppressed', 'on_not_VL_suppressed']),

        'va_pneumo_all_doses': Property(Types.BOOL, 'DUMMY version of the property for va_pneumo_all_doses '
                                                    'whether all doses have been received of the pneumococcal vaccine'),
        'va_hib_all_doses': Property(Types.BOOL, 'DUMMY version of the property for va_pneumo_all_doses '
                                                 'whether all doses have been received of the Hib vaccine'),
        'va_measles_all_doses': Property(Types.BOOL, 'DUMMY version of the property for va_pneumo_all_doses '
                                                     'whether all doses have been received of the measles vaccine'),

        'un_clinical_acute_malnutrition': Property(
            Types.CATEGORICAL, 'DUMMY version of the property for un_clinical_acute_malnutrition'
                               'clinical acute malnutrition state based on WHZ',
            categories=['MAM', 'SAM', 'well']),
        'un_WHZ_category': Property(
            Types.CATEGORICAL, 'DUMMY version of the property for un_WHZ_category'
                               'height-for-age z-score group',
            categories=['WHZ<-3', '-3<=WHZ<-2', 'WHZ>=-2']),

        # NOTE: 'nb_breastfeeding_status' and 'nb_low_birth_weight_status' already in simplified births!!
    }

    def __init__(self, name=None, hiv_prev=0.01, art_cov=0.9, moderate_wasting=0.02444, severe_wasting=0.00156,
                 va_pcv13=0.91, va_hib=0.91, va_measles=0.65):
        super().__init__(name)
        self.hiv_prev = hiv_prev
        self.art_cov = art_cov
        self.moderate_wasting = moderate_wasting  # values from Data UNICEF 2015-16 Survey ID 41450
        self.severe_wasting = severe_wasting  # -- 4.7% all wasted, 1.3% severe
        self.va_pcv13 = va_pcv13
        self.va_hib = va_hib
        self.va_measles = va_measles

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        df = population.props

        df.loc[df.is_alive, "hv_inf"] = self.rng.rand(sum(df.is_alive)) < self.hiv_prev
        df.loc[(df.is_alive & df.hv_inf), "hv_art"] = pd.Series(
            self.rng.rand(sum(df.is_alive & df.hv_inf)) < self.art_cov).replace(
            {True: "on_VL_suppressed", False: "not"}).values

        # df.loc[df.is_alive, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        # df.loc[df.is_alive, 'nb_breastfeeding_status'] = 'non_exclusive'
        df.loc[df.is_alive, 'va_pneumo_all_doses'] = self.rng.rand(sum(df.is_alive)) < self.va_pcv13
        df.loc[df.is_alive, 'va_hib_all_doses'] = self.rng.rand(sum(df.is_alive)) < self.va_hib
        df.loc[df.is_alive, 'va_measles_all_doses'] = self.rng.rand(sum(df.is_alive)) < self.va_measles

        df.loc[df.is_alive, 'un_WHZ_category'] = self.rng.choice(
            ['WHZ<-3', '-3<=WHZ<-2', 'WHZ>=-2'],
            p=[self.severe_wasting, self.moderate_wasting, 0.974], size=sum(df.is_alive))

        df.loc[df.is_alive & (df.un_WHZ_category == 'WHZ>=-2'), 'un_clinical_acute_malnutrition'] = 'well'
        df.loc[df.is_alive & (df.un_WHZ_category == '-3<=WHZ<-2'), 'un_clinical_acute_malnutrition'] = 'MAM'
        df.loc[df.is_alive & (df.un_WHZ_category == 'WHZ<-3'), 'un_clinical_acute_malnutrition'] = 'SAM'

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        df = self.sim.population.props
        df.at[child, "hv_inf"] = self.rng.rand() < self.hiv_prev

        if df.at[child, "hv_inf"]:
            df.at[child, "hv_art"] = "on_VL_suppressed" if self.rng.rand() < self.art_cov else "not"

        # df.at[child, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        # df.at[child, 'nb_breastfeeding_status'] = 'non_exclusive'
        df.at[child, 'va_pneumo_all_doses'] = False
        df.at[child, 'va_hib_all_doses'] = False
        df.at[child, 'va_measles_all_doses'] = False
        df.at[child, 'un_clinical_acute_malnutrition'] = 'well'
        df.at[child, 'un_WHZ_category'] = 'WHZ>=-2'


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
            symptoms=self.sim.modules['SymptomManager'].has_what(person_id), facility_level='2',
            hiv_infected_and_not_on_art=False,
            un_clinical_acute_malnutrition='well'
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
                   child_is_younger_than_2_months=False,
                   symptoms=self.sim.modules['SymptomManager'].has_what(person_id), facility_level='2',
                   hiv_infected_and_not_on_art=False,
                   un_clinical_acute_malnutrition='well'
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
    p['prob_hw_decision_for_oxygen_provision_when_po_unavailable'] = 1.0


def _set_current_policy(alri_module):
    """Modify the parameters of an instance of the Alri module so that sensitivity of all diagnosis steps is perfect.
    And under current policy parameters"""
    p = alri_module.parameters

    p['apply_oxygen_indication_to_SpO2_measurement'] = '<90%'
    # p['allow_use_oximetry_for_non_severe_classifications'] = False


def _set_new_policy(alri_module):
    """Modify the parameters of an instance of the Alri module so that sensitivity of all diagnosis steps is perfect.
    And under new policy parameters"""
    p = alri_module.parameters

    p['apply_oxygen_indication_to_SpO2_measurement'] = '<93%'
    # p['allow_use_oximetry_for_non_severe_classifications'] = True


def _reduce_hw_dx_sensitivity(alri_module):
    """Modify the parameters of an instance of the Alri module so that sensitivity of
    health workers diagnostic accuracy is at 30%"""
    p = alri_module.parameters

    p['sensitivity_of_classification_of_fast_breathing_pneumonia_facility_level0'] = 0.3
    p['sensitivity_of_classification_of_danger_signs_pneumonia_facility_level0'] = 0.3
    p['sensitivity_of_classification_of_non_severe_pneumonia_facility_level1'] = 0.3
    p['sensitivity_of_classification_of_severe_pneumonia_facility_level1'] = 0.3
    p['sensitivity_of_classification_of_non_severe_pneumonia_facility_level2'] = 0.3
    p['sensitivity_of_classification_of_severe_pneumonia_facility_level2'] = 0.3

    # Change respective cost for this condition
    p['oxygen_unit_cost_by_po_implementation_existing_psa_imperfect_hw_dx'] = \
        [0.0167841508524565, 0.00926022115997598, 0.00635414926419273, 0.00446207421001029, 0.00429132200096448]
    p['oxygen_unit_cost_by_po_implementation_planned_psa_imperfect_hw_dx'] = \
        [0.0142747188864335, 0.00826537109557841, 0.00551370534960769, 0.00383253630298794, 0.0036877605349842]
    p['oxygen_unit_cost_by_po_implementation_all_district_psa_imperfect_hw_dx'] = \
        [0.0286146949852482, 0.0165326378779786, 0.0110544589933642, 0.00769319228358011, 0.00723704307846086]


def _prioritise_oxygen_to_hospitals(alri_module):
    """Modify the parameters of an instance of the Alri module so that coverage of oxygen is
    higher at the hospital levels"""
    p = alri_module.parameters

    p['scenario_existing_psa_ox_coverage_by_facility'] = [1, 0.03, 0]
    p['scenario_planned_psa_ox_coverage_by_facility'] = [1, 1, 0.94]

    # Change respective cost for this condition
    p['oxygen_unit_cost_by_po_implementation_existing_psa_imperfect_hw_dx'] = \
        [0.00645054596605153, 0.00603118423066993, 0.00519326397877533, 0.00394158505920955, 0.00380774765607967]
    p['oxygen_unit_cost_by_po_implementation_planned_psa_imperfect_hw_dx'] = \
        [0.00547761189189517, 0.0051350610605493, 0.00436034217101729, 0.00329684408052527, 0.00318278915942758]
    p['oxygen_unit_cost_by_po_implementation_all_district_psa_imperfect_hw_dx'] = \
        [0.0109790253401836, 0.0102926367248479, 0.00874632510421351, 0.00661490893230276, 0.00638693209973874]


def _make_treatment_perfect(alri_module):
    """Modify the parameters of an instance of the Alri module so that treatment is perfectly effective."""
    p = alri_module.parameters

    # The probability of treatment failure to be 0.0
    p['tf_1st_line_antibiotic_for_severe_pneumonia'] = 0.0
    p['tf_2nd_line_antibiotic_for_severe_pneumonia'] = 0.0
    p['tf_3day_amoxicillin_for_fast_breathing_with_SpO2>=90%'] = 0.0
    p['tf_3day_amoxicillin_for_chest_indrawing_with_SpO2>=90%'] = 0.0
    p['tf_5day_amoxicillin_for_chest_indrawing_with_SpO2>=90%'] = 0.0
    p['tf_7day_amoxicillin_for_fast_breathing_pneumonia_in_young_infants'] = 0.0


def _make_treatment_ineffective(alri_module):
    """Modify the parameters of an instance of the Alri module so that treatment is completely ineffective."""
    p = alri_module.parameters

    # The probability of treatment failure to be 1.0
    p['tf_1st_line_antibiotic_for_severe_pneumonia'] = 1.0
    p['tf_2nd_line_antibiotic_for_severe_pneumonia'] = 1.0
    p['tf_3day_amoxicillin_for_fast_breathing_with_SpO2>=90%'] = 1.0
    p['tf_3day_amoxicillin_for_chest_indrawing_with_SpO2>=90%'] = 1.0
    p['tf_5day_amoxicillin_for_chest_indrawing_with_SpO2>=90%'] = 1.0
    p['tf_7day_amoxicillin_for_fast_breathing_pneumonia_in_young_infants'] = 1.0


def _make_treatment_and_diagnosis_perfect(alri_module):
    """Modify the parameters of an instance of the Alri module so that treatment and diagnosis is perfect."""

    _make_hw_diagnosis_perfect(alri_module)
    _make_treatment_perfect(alri_module)
    _make_perfect_conditions(alri_module)


def _make_perfect_conditions(alri_module):
    """Modify the parameters of an instance of the Alri module so that consitions are perfect:
    - referral rate, pulse oximeter usage rate, and follow-up care."""
    p = alri_module.parameters

    p['referral_rate_severe_cases_from_hc'] = 1.0
    p['pulse_oximeter_usage_rate'] = 1.0
    # p['sought_follow_up_care'] = 1.0


def _make_high_risk_of_death(alri_module):
    """Modify the parameters of an instance of the Alri module so that the risk of death is high."""
    params = alri_module.parameters
    params['base_odds_death_ALRI_age<2mo'] *= 5.0
    params['base_odds_death_ALRI_age2_59mo'] *= 5.0


# def _set_scenario_existing_psa_ox(alri_module):
#     """Modify the parameters of scenario_existing_psa_ox_coverage_by_facility for CEA """
#     p = alri_module.parameters
#
#     p['scenario_existing_psa_ox_coverage_by_facility'] = '<93%'
