"""
Childhood Acute Lower Respiratory Infection Module

Overview
--------
Individuals are exposed to the risk of onset of an acute lower respiratory infection (Alri).
The disease is manifested as viral pneumonia, bacterial pneumonia or bronchiolitis
caused by one primary agent at a time, which can also have a co-infection or secondary bacterial infection.
During an episode (prior to recovery - either naturally or cured with treatment),
the symptom of cough or difficult breathing is present in addition to other possible symptoms.
Alri may cause associated complications, such as,
local pulmonary complication: pleural effusuion, empyema, lung abscess, pneumothorax,
and systemic complications: sepsis, meningitis, and respiratory failure, leading to multi-organ dysfunction and death.
The individual may recover naturally or die.

Health care seeking is prompted by the onset of the symptom cough or respiratory symptoms.
The individual can be treated; if successful the risk of death is lowered
and they are cured (symptom resolved) some days later.

Outstanding issues
------------------
* Follow-up appointments for initial HSI events.
* Double check parameters and consumables codes for the HSI events.

"""

# #todo - properties: rename "
# 'number_cases': df.loc[df.is_alive, 'ri_ALRI_cases_counter'].sum(),
# 'number_recovered': df.loc[df.is_alive, 'ri_ALRI_recovery_counter'].sum(),
# 'number_treated': df.loc[df.is_alive, 'ri_ALRI_treatment_counter'].sum(),
# 'number_died': df.loc[df.is_alive, 'ri_ALRI_death_counter'].sum()


import copy
from pathlib import Path

import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography
from tlo.methods.causes import Cause
from tlo.methods.symptommanager import Symptom
from tlo.methods.healthsystem import HSI_Event
from tlo.util import BitsetHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

class Alri(Module):
    """This is the disease module for Acute Lower Respiritory Infections."""

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare the pathogen types + pathogens:
    viral_patho = {'RSV',
                   'Rhinovirus',
                   'HMPV',
                   'Parainfluenza',
                   'Influenza',
                   'Adenovirus',
                   'Bocavirus',
                   'other_viral_pathogens'  # Coronaviruses NL63, 229E OC43 and HKU1, Cytomegalovirus,
                                            # Parechovirus/Enterovirus
                   }

    bacterial_patho = {'Strep_pneumoniae_PCV13',
                       'Strep_pneumoniae_non_PCV13',
                       'Hib',
                       'H.influenzae_non_type_b',
                       'Staph_aureus',
                       'Enterobacteriaceae',        # includes E. coli, Enterobacter species, and Klebsiella species
                       'other_Strepto_Enterococci', # includes Streptococcus pyogenes and Enterococcus faecium
                       'other_bacterial_pathogens'
                       # includes Bordetella pertussis, Chlamydophila pneumoniae,
                       # Legionella species, Mycoplasma pneumoniae, Moraxella catarrhalis, Non-fermenting gram-negative
                       # rods (Acinetobacter species and Pseudomonas species), Neisseria meningitidis
                       }

    fungal_patho = {'P.jirovecii'}

    # Make set of all pathogens combined:
    pathogens = viral_patho | bacterial_patho | fungal_patho

    # todo: @ines -- is the definition of causes of death and disability suitable? Please look into this.
    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        f"ALRI_{path}":
            Cause(gbd_causes={'Lower respiratory infections'}, label='Lower Lower respiratory infections')
        for path in pathogens
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        f"ALRI_{path}":
            Cause(gbd_causes={'Lower respiratory infections'}, label='Lower Lower respiratory infections')
        for path in pathogens
    }

    # Declare the disease types:
    disease_type = {
        'bacterial_pneumonia', 'viral_pneumonia', 'fungal_pneumonia', 'bronchiolitis'
    }

    # Declare the Alri complications:
    complications = {'pneumothorax',
                     'pleural_effusion',
                     'empyema',
                     'lung_abscess',
                     'sepsis',
                     'meningitis',
                     'respiratory_failure'}


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
                      'baseline incidence of Alri caused by Enterobacteriaceae in age groups 0-11, 12-23, 24-59 months,'
                      ' per child per year'
                      ),
        'base_inc_rate_ALRI_by_other_Strepto_Enterococci':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by other streptococci and Enterococci including '
                      'Streptococcus pyogenes and Enterococcus faecium in age groups 0-11, 12-23, 24-59 months,'
                      ' per child per year'
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
                      'baseline incidence of Alri caused by P. jirovecii in age groups 0-11, 12-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_Adenovirus':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by adenovirus in age groups 0-11, 12-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_Bocavirus':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by bocavirus in age groups 0-11, 12-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_other_viral_pathogens':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by other viral pathogens in age groups 0-11, 12-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_other_bacterial_pathogens':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by other viral pathogens in age groups 0-11, 12-59 months, '
                      'per child per year'
                      ),

        # Risk factors parameters -----
        'rr_ALRI_HHhandwashing':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for children with household handwashing with soap '
                      ),
        'rr_ALRI_HIV_untreated':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for children with untreated HIV positive status'
                      ),
        'rr_ALRI_underweight':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for underweight children'
                      ),
        'rr_ALRI_low_birth_weight':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for infants with low birth weight'
                      ),
        'rr_ALRI_not_excl_breastfeeding':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for not exclusive breastfeeding upto 6 months'
                      ),
        'rr_ALRI_indoor_air_pollution':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for indoor air pollution'
                      ),
        # 'rr_ALRI_pneumococcal_vaccine': Parameter
        # (Types.REAL, 'relative rate of acquiring Alri for pneumonococcal vaccine'
        #  ),
        # 'rr_ALRI_haemophilus_vaccine': Parameter
        # (Types.REAL, 'relative rate of acquiring Alri for haemophilus vaccine'
        #  ),
        # 'rr_ALRI_influenza_vaccine': Parameter
        # (Types.REAL, 'relative rate of acquiring Alri for influenza vaccine'
        #  ),
        # 'r_progress_to_severe_ALRI': Parameter
        # (Types.LIST,
        #  'probability of progressing from non-severe to severe Alri by age category '
        #  'HIV negative, no SAM'
        #  ),

        # Probability of bacterial co- / secondary infection
        'prob_viral_pneumonia_bacterial_coinfection':
            Parameter(Types.REAL,
                      'probability of primary viral pneumonia having a bacterial co-infection'
                      ),
        'proportion_bacterial_coinfection_pathogen':
            Parameter(Types.LIST,
                      'list of proportions of each bacterial pathogens in a co-infection pneumonia'
                      ),
        'prob_secondary_bacterial_infection_in_bronchiolitis':
            Parameter(Types.REAL,
                      'probability of viral bronchiolitis having a bacterial infection'
                      ),

        # Probability of complications -----
        'prob_hypoxia_by_viral_pneumonia':
            Parameter(Types.REAL,
                      'probability of hypoxia caused by primary viral pneumonia'
                      ),
        'prob_hypoxia_by_bacterial_pneumonia':
            Parameter(Types.REAL,
                      'probability of hypoxia caused by primary or secondary bacterial pneumonia'
                      ),
        'prob_hypoxia_by_bronchiolitis':
            Parameter(Types.REAL,
                      'probability of hypoxia caused by viral bronchiolitis'
                      ),
        'prob_respiratory_failure_when_SpO2<93%':
            Parameter(Types.REAL,
                      'probability of respiratory failure when peripheral oxygen level is lower than 93%'
                      ),


        'prob_respiratory_failure_to_multiorgan_dysfunction':
            Parameter(Types.REAL,
                      'probability of respiratory failure causing multi-organ dysfunction'
                      ),
        'prob_sepsis_by_viral_pneumonia':
            Parameter(Types.REAL,
                      'probability of sepsis caused by primary viral pneumonia'
                      ),
        'prob_sepsis_by_bacterial_pneumonia':
            Parameter(Types.REAL,
                      'probability of sepsis caused by primary or secondary bacterial pneumonia'
                      ),
        'prob_sepsis_by_bronchiolitis':
            Parameter(Types.REAL,
                      'probability of sepsis caused by viral bronchiolitis'
                      ),
        'prob_sepsis_to_multiorgan_dysfunction':
            Parameter(Types.REAL,
                      'probability of sepsis causing multi-organ dysfunction'
                      ),
        'prob_meningitis_by_bacterial_pneumonia':
            Parameter(Types.REAL,
                      'probability of meningitis caused by primary or secondary bacterial pneumonia'
                      ),
        'prob_pleural_effusion_by_bacterial_pneumonia':
            Parameter(Types.REAL,
                      'probability of pleural effusion caused by primary or secondary bacterial pneumonia'
                      ),
        'prob_pleural_effusion_by_viral_pneumonia':
            Parameter(Types.REAL,
                      'probability of pleural effusion caused by primary viral pneumonia'
                      ),
        'prob_pleural_effusion_by_bronchiolitis':
            Parameter(Types.REAL,
                      'probability of pleural effusion caused by bronchiolitis'
                      ),
        'prob_pleural_effusion_to_empyema':
            Parameter(Types.REAL,
                      'probability of pleural effusion developing into empyema'
                      ),
        'prob_empyema_to_sepsis':
            Parameter(Types.REAL,
                      'probability of (bacterial) empyema causing sepsis'
                      ),
        'prob_lung_abscess_by_bacterial_pneumonia':
            Parameter(Types.REAL,
                      'probability of a lung abscess caused by primary or secondary bacterial pneumonia'
                      ),
        'prob_pneumothorax_by_bacterial_pneumonia':
            Parameter(Types.REAL,
                      'probability of pneumothorax caused by primary or secondary bacterial pneumonia'
                      ),
        'prob_pneumothorax_by_viral_pneumonia':
            Parameter(Types.REAL,
                      'probability of pneumothorax caused by primary viral pneumonia'
                      ),
        'prob_pneumothorax_by_bronchiolitis':
            Parameter(Types.REAL,
                      'probability of atelectasis/ lung collapse caused by bronchiolitis'
                      ),
        'prob_pneumothorax_to_respiratory_failure':
            Parameter(Types.REAL,
                      'probability of pneumothorax causing respiratory failure'
                      ),
        'prob_lung_abscess_to_sepsis':
            Parameter(Types.REAL,
                      'probability of lung abscess causing sepsis'
                      ),

        # death parameters -----
        'base_death_rate_ALRI_by_bacterial_pneumonia':
            Parameter(Types.REAL,
                      'baseline death rate from bacterial pneumonia, base age 0-11 months'
                      ),
        'base_death_rate_ALRI_by_bronchiolitis':
            Parameter(Types.REAL,
                      'baseline death rate from bronchiolitis, base age 0-11 months'
                      ),
        'base_death_rate_ALRI_by_viral_pneumonia':
            Parameter(Types.REAL,
                      'baseline death rate from viral pneumonia, base age 0-11 months'
                      ),
        'base_death_rate_ALRI_by_fungal_pneumonia':
            Parameter(Types.REAL,
                      'baseline death rate from fungal pneumonia, base age 0-11 months'
                      ),
        'rr_death_ALRI_sepsis':
            Parameter(Types.REAL,
                      'relative death rate from Alri for sepsis, base age 0-11 months'
                      ),
        'rr_death_ALRI_respiratory_failure':
            Parameter(Types.REAL,
                      'relative death rate from Alri for respiratory failure, base age 0-11 months'
                      ),
        'rr_death_ALRI_meningitis':
            Parameter(Types.REAL,
                      'relative death rate from Alri for meningitis, base age 0-11 months'
                      ),
        'rr_death_ALRI_age12to23mo':
            Parameter(Types.REAL,
                      'death rate of Alri for children aged 12 to 23 months'
                      ),
        'rr_death_ALRI_age24to59mo':
            Parameter(Types.REAL,
                      'death rate of Alri for children aged 24 to 59 months'
                      ),
        'rr_death_ALRI_HIV':
            Parameter(Types.REAL,
                      'death rate of Alri for children with HIV not on ART'
                      ),
        'rr_death_ALRI_SAM':
            Parameter(Types.REAL,
                      'death rate of Alri for children with severe acute malnutrition'
                      ),
        'rr_death_ALRI_low_birth_weight':
            Parameter(Types.REAL,
                      'death rate of Alri for children with low birth weight (applicable to infants)'
                      ),

        # Proportions of what disease type (viral Alri) -----
        'proportion_viral_pneumonia_by_RSV':
            Parameter(Types.REAL,
                      'proportion of RSV infection causing viral pneumonia'
                      ),
        'proportion_viral_pneumonia_by_Rhinovirus':
            Parameter(Types.REAL,
                      'proportion of Rhinovirus infection causing viral pneumonia'
                      ),
        'proportion_viral_pneumonia_by_HMPV':
            Parameter(Types.REAL,
                      'proportion of HMPV infection causing viral pneumonia'
                      ),
        'proportion_viral_pneumonia_by_Parainfluenza':
            Parameter(Types.REAL,
                      'proportion of Parainfluenza infection causing viral pneumonia'
                      ),
        'proportion_viral_pneumonia_by_Influenza':
            Parameter(Types.REAL,
                      'proportion of Influenza infection causing viral pneumonia'
                      ),
        'proportion_viral_pneumonia_by_Adenovirus':
            Parameter(Types.REAL,
                      'proportion of Adenovirus infection causing viral pneumonia'
                      ),
        'proportion_viral_pneumonia_by_Bocavirus':
            Parameter(Types.REAL,
                      'proportion of Bocavirus infection causing viral pneumonia'
                      ),
        'proportion_viral_pneumonia_by_other_viral_pathogens':
            Parameter(Types.REAL,
                      'proportion of other pathogens infection causing viral pneumonia'
                      ),

        # Probability of symptom development -----
        'prob_fever_uncomplicated_ALRI_by_disease_type':
            Parameter(Types.LIST,
                      'list of probabilities of having fever by bacterial pneumonia, viral pneumonia and bronchiolitis'
                      ),
        'prob_cough_uncomplicated_ALRI_by_disease_type':
            Parameter(Types.LIST,
                      'list of probabilities of having cough by bacterial pneumonia, viral pneumonia and bronchiolitis'
                      ),
        'prob_difficult_breathing_uncomplicated_ALRI_by_disease_type':
            Parameter(Types.LIST,
                      'list of probabilities of difficult breathing by '
                      'bacterial pneumonia, viral pneumonia and bronchiolitis'
                      ),
        'prob_fast_breathing_uncomplicated_ALRI_by_disease_type':
            Parameter(Types.LIST,
                      'list of probabilities of fast breathing by '
                      'bacterial pneumonia, viral pneumonia and bronchiolitis'
                      ),
        'prob_chest_indrawing_uncomplicated_ALRI_by_disease_type':
            Parameter(Types.LIST,
                      'list of probabilities of chest indrawing by '
                      'bacterial pneumonia, viral pneumonia and bronchiolitis'
                      ),
        'prob_danger_signs_uncomplicated_ALRI_by_disease_type':
            Parameter(Types.LIST,
                      'list of probabilities of danger signs by '
                      'bacterial pneumonia, viral pneumonia and bronchiolitis'
                      ),

        # Additional signs and symptoms from complications -----
        'prob_loss_of_appetite_adding_from_pleural_effusion':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of loss of appetite from pleural effusion'
                      ),
        'prob_loss_of_appetite_adding_from_empyema':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of loss of appetite from empyema'
                      ),
        'prob_severe_respiratory_distress_adding_from_respiratory_failure':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of severe respiratory distress from respiratory failure'
                      ),
        'prob_severe_respiratory_distress_adding_from_sepsis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of severe respiratory distress from sepsis'
                      ),
        # second round of signs/symptoms added from each complication
        # Pneumothorax ------------
        'prob_chest_pain_adding_from_pneumothorax':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of chest pain / pleurisy from pneumothorax'
                      ),
        'prob_cyanosis_adding_from_pneumothorax':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of cyanosis from pneumothorax'
                      ),
        'prob_difficult_breathing_adding_from_pneumothorax':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of difficult breathing from pneumothorax'
                      ),
        # pleural effusion -----------
        'prob_chest_pain_adding_from_pleural_effusion':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of chest pain / pleurisy from pleural effusion'
                      ),
        'prob_fever_adding_from_pleural_effusion':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of fever from pleural effusion'
                      ),
        'prob_difficult_breathing_adding_from_pleural_effusion':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of dyspnoea/ difficult breathing from pleural effusion'
                      ),
        # Empyema ------------
        'prob_chest_pain_adding_from_empyema':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of chest pain / pleurisy from empyema'
                      ),
        'prob_fever_adding_from_empyema':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of fever from empyema'
                      ),
        'prob_respiratory_distress_adding_from_empyema':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of cough with sputum from respiratory distress'
                      ),
        # Lung abscess --------------
        'prob_chest_pain_adding_from_lung_abscess':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of chest pain / pleurisy from lung abscess'
                      ),
        'prob_fast_breathing_adding_from_lung_abscess':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of fast_breathing/ fast breathing from lung abscess'
                      ),
        'prob_fever_adding_from_lung_abscess':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of fever from lung abscess'
                      ),
        # Hypoxaemic Respiratory failure
        'prob_difficult_breathing_adding_from_respiratory_failure':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of difficult breathing from respiratory failure'
                      ),
        'prob_cyanosis_adding_from_respiratory_failure':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of cyanosis from respiratory failure'
                      ),
        'prob_fast_breathing_adding_from_respiratory_failure':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of fast_breathing from respiratory failure'
                      ),
        'prob_danger_signs_adding_from_respiratory_failure':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of danger signs from respiratory failure'
                      ),
        # meningitis ------------
        'prob_headache_adding_from_meningitis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of headache from meningitis'
                      ),
        'prob_fever_adding_from_meningitis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of fever from meningitis'
                      ),
        'prob_danger_signs_adding_from_meningitis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of danger_signs from meningitis'
                      ),
        # Sepsis ----------------
        'prob_fast_breathing_adding_from_sepsis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of fast_breathing from sepsis'
                      ),
        'prob_danger_signs_adding_from_sepsis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of danger signs from sepsis'
                      ),
        'prob_fever_adding_from_sepsis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of fever from sepsis'
                      ),

        # other parameters
        'days_between_treatment_and_cure':
            Parameter(Types.INT, 'number of days between any treatment being given in an HSI and the cure occurring.'
                      ),
        'rr_ALRI_PCV13':
            Parameter(Types.REAL,
                      'relative rate of Alri with the PCV13'
                      ),
        'rr_ALRI_hib_vaccine':
            Parameter(Types.REAL,
                      'relative rate of Alri with the hib vaccination'
                      ),
        'rr_ALRI_RSV_vaccine':
            Parameter(Types.REAL,
                      'relative rate of Alri with the RSV vaccination'
                      ),

        'prob_of_cure_for_uncomplicated_pneumonia_given_IMCI_pneumonia_treatment':
            Parameter(Types.REAL,
                      'probability of cure for uncomplicated pneumonia given IMCI pneumonia treatment'
                      ),
        'prob_of_cure_for_pneumonia_with_severe_complication_given_IMCI_severe_pneumonia_treatment':
            Parameter(Types.REAL,
                      'probability of cure for pneumonia with severe complications given IMCI pneumonia treatment'
                      ),
        'prob_seek_follow_up_care_after_treatment_failure':
            Parameter(Types.REAL,
                      'probability of seeking follow-up care after treatment failure'
                      ),
        'oxygen_therapy_effectiveness_ALRI':
            Parameter(Types.REAL,
                      'effectiveness of oxygen therapy on death from Alri with respiratory failure'
                      ),
        'antibiotic_therapy_effectiveness_ALRI':
            Parameter(Types.REAL,
                      'effectiveness of antibiotic therapy on death from Alri with bacterial cause'
                      ),

    }

    # todo; @Ines -- some of the labelling of the properties is confusing -- is for currrent or last episode?
    PROPERTIES = {
        # ---- Alri status ----
        'ri_current_ALRI_status':
            Property(Types.BOOL,
                     'Alri status (current or last episode)'
                     ),
        # ---- Treatment Status ----
        # todo- @ines: i saw you using a property like this but it wasn't defined, so have defined it here.
        'ri_ALRI_treatment':
            Property(Types.BOOL,
                    'Whether this person is currentlt receiving treatment for a current episode of ALRI'
                    ),
        # ---- The pathogen which is the attributed cause of Alri ----
        'ri_primary_ALRI_pathogen':
            Property(Types.CATEGORICAL,
                     'Attributable pathogen for the current Alri event',
                     categories=list(pathogens) + ['not_applicable']
                     ),
        # ---- The bacterial pathogen which is the attributed co-/secondary infection ----
        'ri_secondary_bacterial_pathogen':
            Property(Types.CATEGORICAL,
                     'Secondary bacterial pathogen for the current Alri event',
                     categories=list(bacterial_patho) + ['none'] + ['not_applicable']
                     ),
        # ---- The underlying Alri condition ----
        'ri_ALRI_disease_type':
            Property(Types.CATEGORICAL, 'underlying Alri condition',
                     categories=['viral_pneumonia', 'bacterial_pneumonia', 'fungal_pneumonia',
                                 'bronchiolitis'] + ['not_applicable']
                     ),
        # ---- Complications associated with Alri ----
        'ri_ALRI_complications':
            Property(Types.INT,
                     'bitset column, stores complications that arose from the Alri event'
                     ),
        'ri_peripheral_oxygen_saturation':
            Property(Types.CATEGORICAL,
                     'Level of peripheral oxygen saturation to be read by a pulse oximetry',
                     categories=['SpO2<93%', 'SpO2_93-100%']),
        # ---- Symptoms associated with Alri ----
        'ri_current_ALRI_symptoms':
            Property(Types.INT,
                     'bitset column, stores symptoms of current Alri event'
                     ),

        # ---- Internal variables to schedule onset and deaths due to Alri ----
        'ri_ALRI_event_date_of_onset': Property(Types.DATE, 'date of onset of current Alri event'),
        'ri_ALRI_event_recovered_date': Property(Types.DATE, 'date of recovery from current Alri event'),
        'ri_ALRI_event_death_date': Property(Types.DATE, 'date of death caused by current Alri event'),
        'ri_end_of_last_alri_episode':
            Property(Types.DATE, 'date on which the last episode of Alri is resolved, (including '
                                 'allowing for the possibility that a cure is scheduled following onset). '
                                 'This is used to determine when a new episode can begin. '
                                 'This stops successive episodes interfering with one another.'),

        # ---- Temporary Variables: To be replaced with the properties of other modules ----
        'tmp_malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'tmp_low_birth_weight': Property(Types.BOOL, 'temporary property - low birth weight'),
        'tmp_hv_inf': Property(Types.BOOL, 'temporary property - hiv infection'),
        'tmp_exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'tmp_continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'tmp_pneumococcal_vaccination': Property(Types.BOOL, 'temporary property - streptococcus pneumoniae vaccine'),
        'tmp_haemophilus_vaccination': Property(Types.BOOL, 'temporary property - H. influenzae type b vaccine'),
        'tmp_influenza_vaccination': Property(Types.BOOL, 'temporary property - flu vaccine'),

        # ---- Health System interventions / Treatment properties ----
        'ri_ALRI_tx_start_date': Property(Types.DATE, 'start date of Alri treatment for current event'),

        # ---- Counter properties (for Logging use) ----
        'ri_ALRI_cases_counter':
            Property(Types.INT,
                     'annual counter for Alri cases'
                     ),
        'ri_ALRI_recovery_counter':
            Property(Types.INT,
                     'annual counter for Alri recovery episodes'
                     ),
        'ri_ALRI_treatment_counter':
            Property(Types.INT,
                     'annual counter for Alri treatment episodes'
                     ),
        'ri_ALRI_death_counter':
            Property(Types.INT,
                     'annual counter for Alri death episodes'
                     ),

    }

    def __init__(self, name=None, resourcefilepath=None, log_indivdual=False):
        super().__init__(name)

        # Store arguments provided
        self.resourcefilepath = resourcefilepath
        self.log_individual = log_indivdual

        # equations for the incidence of Alri by pathogen:
        self.incidence_equations_by_pathogen = dict()

        # equations for the proportions of Alri diseases:
        self.proportions_of_ALRI_disease_types_by_pathogen = dict()

        # equations for the probabilities of secondary bacterial superinfection:
        self.prob_secondary_bacterial_infection = None

        # equations for the development of Alri-associated complications:
        self.risk_of_decreased_peripheral_oxygen_level = None
        self.risk_of_developing_ALRI_complications = dict()
        self.risk_of_progressing_to_severe_complications = dict()

        # Linear Model for predicting the risk of death:
        self.mortality_equations_by_disease = dict()

        # dict to hold the probability of onset of different types of symptom given underlying complications:
        self.prob_symptoms_uncomplicated_ALRI = dict()
        self.prob_extra_symptoms_complications = dict()

        # dict to to store the information regarding HSI management of disease:
        self.child_disease_management_information = dict()

        # dict to hold the DALY weights
        self.daly_wts = dict()

        # will store the logging event used by this module
        self.logging_event = None

        # Define the symptoms that this module will use:
        # todo - @ines: is it right that 'danger_signs' is an indepednet symptom?
        self.symptoms = {
            'fever', 'cough', 'difficult_breathing', 'fast_breathing', 'chest_indrawing', 'chest_pain',
            'cyanosis', 'respiratory_distress', 'danger_signs'
        }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
        self.load_parameters_from_dataframe(
            pd.read_excel(
                Path(self.resourcefilepath) / 'ResourceFile_Alri.xlsx', sheet_name='Parameter_values'))

        self.check_params_read_in_ok()

        # Declare symptoms that this modules will cause and which are not included in the generic symptoms:
        for symptom_name in self.symptoms:
            if symptom_name not in self.sim.modules['SymptomManager'].generic_symptoms:
                self.sim.modules['SymptomManager'].register_symptom(
                    Symptom(name=symptom_name)  # (give non-generic symptom 'average' healthcare seeking)
                )

    def check_params_read_in_ok(self):
        """Check that every value has been read-in successfully"""
        for param_name, param_type in self.PARAMETERS.items():
            assert param_name in self.parameters, f'Parameter "{param_name}" ' \
                                                  f'is not read in correctly from the resourcefile.'
            assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert isinstance(self.parameters[param_name],
                              param_type.python_type), f'Parameter "{param_name}" ' \
                                                       f'is not read in correctly from the resourcefile.'

    def initialise_population(self, population):
        """
        Sets that there is no one with Alri at initiation.
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals

        # ---- Key Current Status Classification Properties ----
        df.loc[df.is_alive, 'ri_current_ALRI_status'] = False
        df.loc[df.is_alive, 'ri_primary_ALRI_pathogen'].values[:] = 'not_applicable'
        df.loc[df.is_alive, 'ri_current_ALRI_symptoms'] = 0
        df.loc[df.is_alive, 'ri_secondary_bacterial_pathogen'] = 'not_applicable'
        df.loc[df.is_alive, 'ri_ALRI_disease_type'] = 'not_applicable'
        df.loc[df.is_alive, 'ri_ALRI_complications'] = 0

        # ---- Internal values ----
        df.loc[df.is_alive, 'ri_ALRI_event_date_of_onset'] = pd.NaT
        df.loc[df.is_alive, 'ri_ALRI_event_recovered_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_ALRI_event_death_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_end_of_last_alri_episode'] = pd.NaT

        df.loc[df.is_alive, 'ri_ALRI_treatment'] = False
        df.loc[df.is_alive, 'ri_ALRI_tx_start_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_peripheral_oxygen_saturation'] = 'SpO2_93-100%'

        # ---- Temporary values ----
        df.loc[df.is_alive, 'tmp_malnutrition'] = False
        df.loc[df.is_alive, 'tmp_hv_inf'] = False
        df.loc[df.is_alive, 'tmp_low_birth_weight'] = False
        df.loc[df.is_alive, 'tmp_exclusive_breastfeeding'] = False
        df.loc[df.is_alive, 'tmp_continued_breastfeeding'] = False

        # ---- Counter properties ----
        df.loc[df.is_alive, 'ri_ALRI_cases_counter'] = 0
        df.loc[df.is_alive, 'ri_ALRI_recovery_counter'] = 0
        df.loc[df.is_alive, 'ri_ALRI_treatment_counter'] = 0
        df.loc[df.is_alive, 'ri_ALRI_death_counter'] = 0

        # This biset property stores set of symptoms that can occur
        # todo - @ines - this seems to be replicating what the SymptomManager does?
        self.ALRI_symptoms = BitsetHandler(self.sim.population, 'ri_current_ALRI_symptoms',
                                           ['fever', 'cough', 'difficult_breathing', 'fast_breathing', 'headache',
                                            'chest_indrawing', 'chest_pain', 'cyanosis', 'respiratory_distress',
                                            'danger_signs'])

        # This biset property stores set of complications that can occur
        self.ALRI_complications = BitsetHandler(self.sim.population, 'ri_ALRI_complications',
                                                ['pneumothorax', 'pleural_effusion', 'empyema', 'lung_abscess',
                                                 'sepsis', 'meningitis', 'respiratory_failure'])

    def initialise_simulation(self, sim):
        """
        Prepares for simulation:
        * Schedules the main polling event
        * Schedules the main logging event
        * Establishes the linear models and other data structures using the parameters that have been read-in
        """

        # Schedule the main polling event (to first occur immediately)
        sim.schedule_event(AlriPollingEvent(self), sim.date)

        # Schedule the main logging event (to first occur in one year)
        self.logging_event = AlriLoggingEvent(self)
        sim.schedule_event(self.logging_event, sim.date + DateOffset(years=1))

        if self.log_individual:
            # Schedule the individual check logging event (to first occur immediately, and to occur every day)
            sim.schedule_event(AlriIindividualCheckLoggingEvent(self), sim.date)

        # Get DALY weights
        # get_daly_weight = self.sim.modules['HealthBurden'].get_daly_weight
        if 'HealthBurden' in self.sim.modules.keys():
            self.daly_wts['daly_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
            self.daly_wts['daly_severe_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
            self.daly_wts['daly_very_severe_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=46)
            # todo @ines - 'daly_very_severe_ALRI' is never used: is that right?

        # Make the linear models:
        self.make_linear_models()

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        # ---- Key Current Status Classification Properties ----
        df.at[child_id, 'ri_current_ALRI_status'] = False
        df.at[child_id, 'ri_primary_ALRI_pathogen'] = 'not_applicable'
        df.at[child_id, 'ri_current_ALRI_symptoms'] = 0
        df.at[child_id, 'ri_secondary_bacterial_pathogen'] = 'not_applicable'
        df.at[child_id, 'ri_ALRI_disease_type'] = 'not_applicable'
        df.at[child_id, 'ri_ALRI_complications'] = 0

        # ---- Internal values ----
        df.at[child_id, 'ri_ALRI_event_date_of_onset'] = pd.NaT
        df.at[child_id, 'ri_ALRI_event_recovered_date'] = pd.NaT
        df.at[child_id, 'ri_ALRI_event_death_date'] = pd.NaT
        df.at[child_id, 'ri_end_of_last_alri_episode'] = pd.NaT

        # ---- Temporary values ----
        df.at[child_id, 'tmp_malnutrition'] = False
        df.at[child_id, 'tmp_hv_inf'] = False
        df.at[child_id, 'tmp_low_birth_weight'] = False
        df.at[child_id, 'tmp_exclusive_breastfeeding'] = False
        df.at[child_id, 'tmp_continued_breastfeeding'] = False

    def report_daly_values(self):
        """Report DALY incurred in the population in the last month due to ALRI"""

        df = self.sim.population.props

        total_daly_values = pd.Series(data=0.0, index=df.index[df.is_alive])

        total_daly_values.loc[
            self.sim.modules['SymptomManager'].who_has('fast_breathing')] = self.daly_wts['daly_ALRI']

        total_daly_values.loc[
            self.sim.modules['SymptomManager'].who_has('danger_signs')] = self.daly_wts['daly_severe_ALRI']

        # Split out by pathogen that causes the Alri
        dummies_for_pathogen = pd.get_dummies(df.loc[total_daly_values.index,
                                                     'ri_primary_ALRI_pathogen'],
                                              dtype='float').drop(columns='not_applicable')
        daly_values_by_pathogen = dummies_for_pathogen.mul(total_daly_values, axis=0)

        # add prefix to label according to the name of the causes of disability declared
        daly_values_by_pathogen = daly_values_by_pathogen.add_prefix('ALRI_')
        return daly_values_by_pathogen

    def make_linear_models(self):
        """Make all the linear models used in the simulation"""
        p = self.parameters
        df = self.sim.population.props

        # =====================================================================================================
        # APPLY A LINEAR MODEL FOR THE ACQUISITION OF A PRIMARY PATHOGEN FOR Alri
        # --------------------------------------------------------------------------------------------
        # Make a dict to hold the equations that govern the probability that a person acquires Alri
        # that is caused (primarily) by a pathogen

        def make_scaled_linear_model_for_incidence(patho):
            """Makes the unscaled linear model with default intercept of 1. Calculates the mean incidents rate for
            0-year-olds and then creates a new linear model with adjusted intercept so incidents in 0-year-olds
            matches the specified value in the model when averaged across the population
            """

            def make_naive_linear_model(patho, intercept=1.0):
                """Make the linear model based exactly on the parameters specified"""
                # todo - @ines - 'va_pneumo' is not defined?

                base_inc_rate = f'base_inc_rate_ALRI_by_{patho}'
                return LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept,
                    Predictor('age_years').when('.between(0,0)', p[base_inc_rate][0])
                        .when('.between(1,1)', p[base_inc_rate][1])
                        .when('.between(2,4)', p[base_inc_rate][2])
                        .otherwise(0.0),
                    Predictor('li_no_access_handwashing').when(False, p['rr_ALRI_HHhandwashing']),
                    Predictor('li_wood_burn_stove').when(False, p['rr_ALRI_indoor_air_pollution']),
                    Predictor('tmp_hv_inf').when(True, p['rr_ALRI_HIV_untreated']),
                    # Predictor().when(
                    #     "va_pneumo == '>1' & "
                    #     "((ri_primary_ALRI_pathogen == 'streptococcus') | "
                    #     "(ri_secondary_bacterial_pathogen == 'streptococcus'))",
                    #     p['rr_ALRI_PCV13']),
                    Predictor('tmp_malnutrition').when(True, p['rr_ALRI_underweight']),
                    Predictor('tmp_exclusive_breastfeeding').when(False, p['rr_ALRI_not_excl_breastfeeding'])
                )

            unscaled_lm = make_naive_linear_model(patho)
            target_mean = p[f'base_inc_rate_ALRI_by_{patho}'][0]
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 0)]).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean)
            scaled_lm = make_naive_linear_model(patho, intercept=scaled_intercept)

            # check by applying the model to mean incidence of 0-year-olds
            assert (target_mean - scaled_lm.predict(df.loc[df.is_alive & (df.age_years == 0)]).mean()) < 1e-10
            return scaled_lm

        for patho in self.pathogens:
            self.incidence_equations_by_pathogen[patho] = make_scaled_linear_model_for_incidence(patho)

        # =====================================================================================================
        # APPLY PROBABILITY OF CO- / SECONDARY BACTERIAL INFECTION
        # -----------------------------------------------------------------------------------------------------
        # Create a linear model equation for the probability of a secondary bacterial superinfection
        self.prob_secondary_bacterial_infection = \
            LinearModel(LinearModelType.ADDITIVE,
                        0.0,
                        Predictor()
                        .when(
                            "ri_primary_ALRI_pathogen.isin(['RSV', 'rhinovirus', 'hMPV', "
                            "'parainfluenza', 'influenza']) & "
                            "(ri_ALRI_disease_type == 'viral_pneumonia') & "
                            "(ri_secondary_bacterial_pathogen =='not_applicable')",
                            p['prob_viral_pneumonia_bacterial_coinfection']),
                        Predictor()
                        .when(
                            "ri_primary_ALRI_pathogen.isin(['RSV', 'rhinovirus', 'hMPV', "
                            "'parainfluenza', 'influenza']) & "
                            "(ri_ALRI_disease_type == 'bronchiolitis') & "
                            "(ri_secondary_bacterial_pathogen =='not_applicable')",
                            p['prob_secondary_bacterial_infection_in_bronchiolitis'])
                        )

        # =====================================================================================================
        # APPLY LINEAR MODEL TO DETERMINE PROBABILITY OF COMPLICATIONS
        # -----------------------------------------------------------------------------------------------------
        # Create a linear model for the risk of hypoxia - decrease in peripheral oxygen level
        self.risk_of_decreased_peripheral_oxygen_level = \
            LinearModel(LinearModelType.ADDITIVE,
                        0.0,
                        Predictor('ri_primary_ALRI_pathogen' or 'ri_secondary_bacterial_pathogen')
                        .when(
                            ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                            "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                            "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                            p['prob_hypoxia_by_bacterial_pneumonia']),
                        Predictor()
                        .when(
                            "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                            "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                            "(ri_ALRI_disease_type == 'viral_pneumonia')",
                            p['prob_hypoxia_by_viral_pneumonia']),
                        Predictor()
                        .when(
                            "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                            "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                            "(ri_ALRI_disease_type == 'bronchiolitis')",
                            p['prob_hypoxia_by_bronchiolitis']),
                        )

        # Create linear models for the risk of acquiring complications from uncomplicated Alri
        self.risk_of_developing_ALRI_complications.update({
            'pneumothorax':
                LinearModel(LinearModelType.ADDITIVE,
                            0.0,
                            Predictor('ri_primary_ALRI_pathogen' or 'ri_secondary_bacterial_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_pneumothorax_by_bacterial_pneumonia']).otherwise(0.0),
                            Predictor()
                            .when(
                                "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "(ri_ALRI_disease_type == 'viral_pneumonia') ",
                                p['prob_pneumothorax_by_viral_pneumonia']),
                            Predictor()
                            .when(
                                "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "(ri_ALRI_disease_type == 'bronchiolitis') ",
                                p['prob_pneumothorax_by_bronchiolitis'])
                            ),

            'pleural_effusion':
                LinearModel(LinearModelType.ADDITIVE,
                            0.0,
                            Predictor('ri_primary_ALRI_pathogen' or 'ri_secondary_bacterial_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_pleural_effusion_by_bacterial_pneumonia']).otherwise(0.0),
                            Predictor()
                            .when(
                                "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "(ri_ALRI_disease_type == 'viral_pneumonia') ",
                                p['prob_pleural_effusion_by_viral_pneumonia']),
                            Predictor()
                            .when(
                                "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "(ri_ALRI_disease_type == 'bronchiolitis') ",
                                p['prob_pleural_effusion_by_bronchiolitis'])
                            ),

            'empyema':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_ALRI_complications').apply(
                                lambda x: p['prob_pleural_effusion_to_empyema']
                                if x & self.ALRI_complications.element_repr('pleural_effusion') else 0)
                            ),

            'lung_abscess':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen' or 'ri_secondary_bacterial_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_lung_abscess_by_bacterial_pneumonia'])
                            .otherwise(0.0)
                            ),

            'sepsis':
                LinearModel(LinearModelType.ADDITIVE,
                            0.0,
                            Predictor('ri_primary_ALRI_pathogen' or 'ri_secondary_bacterial_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_sepsis_by_bacterial_pneumonia']).otherwise(0.0),
                            Predictor()
                            .when(
                                "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "(ri_ALRI_disease_type == 'viral_pneumonia') ",
                                p['prob_sepsis_by_viral_pneumonia']),
                            Predictor()
                            .when(
                                "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "(ri_ALRI_disease_type == 'bronchiolitis') ",
                                p['prob_sepsis_by_bronchiolitis']),
                            ),

            'meningitis':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen' or 'ri_secondary_bacterial_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_meningitis_by_bacterial_pneumonia'])
                            .otherwise(0.0)
                            ),

            'respiratory_failure':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_peripheral_oxygen_saturation')
                            .when('SpO2<93%', p['prob_respiratory_failure_when_SpO2<93%'])
                            .otherwise(0.0)
                            )

        }),

        # check that equations have been declared for each complication
        assert self.complications == set(list(self.risk_of_developing_ALRI_complications.keys()))

        # Create linear models for the risk of developing severe complications
        self.risk_of_progressing_to_severe_complications.update({
            'respiratory_failure':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_ALRI_complications').apply(
                                lambda x: p['prob_pneumothorax_to_respiratory_failure']
                                if x & self.ALRI_complications.element_repr('pneumothorax') else 0)
                            ),

            'sepsis':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_ALRI_complications').apply(
                                lambda x: p['prob_lung_abscess_to_sepsis']
                                if x & self.ALRI_complications.element_repr('lung_abscess') else 0),
                            Predictor('ri_ALRI_complications').apply(
                                lambda x: p['prob_empyema_to_sepsis']
                                if x & self.ALRI_complications.element_repr('empyema') else 0),
                            )
        })

        # =====================================================================================================
        # APPLY PROBABILITY OF SYMPTOMS TO EACH Alri DISEASE TYPE (UNCOMPLICATED AND WITH COMPLICATIONS)
        # -----------------------------------------------------------------------------------------------------
        # Make a dict containing the probability of symptoms given acquisition of (uncomplicated) Alri,
        # by disease type
        def make_symptom_probs(disease_type):
            if disease_type == 'bacterial_pneumonia':
                return {
                    'fever': p['prob_fever_uncomplicated_ALRI_by_disease_type'][0],
                    'cough': p['prob_cough_uncomplicated_ALRI_by_disease_type'][0],
                    'difficult_breathing': p['prob_difficult_breathing_uncomplicated_ALRI_by_disease_type'][0],
                    'fast_breathing': p['prob_fast_breathing_uncomplicated_ALRI_by_disease_type'][0],
                    'chest_indrawing': p['prob_chest_indrawing_uncomplicated_ALRI_by_disease_type'][0],
                    'danger_signs': p['prob_danger_signs_uncomplicated_ALRI_by_disease_type'][0],
                }
            if disease_type == 'viral_pneumonia':
                return {
                    'fever': p['prob_fever_uncomplicated_ALRI_by_disease_type'][1],
                    'cough': p['prob_cough_uncomplicated_ALRI_by_disease_type'][1],
                    'difficult_breathing': p['prob_difficult_breathing_uncomplicated_ALRI_by_disease_type'][1],
                    'fast_breathing': p['prob_fast_breathing_uncomplicated_ALRI_by_disease_type'][1],
                    'chest_indrawing': p['prob_chest_indrawing_uncomplicated_ALRI_by_disease_type'][1],
                    'danger_signs': p['prob_danger_signs_uncomplicated_ALRI_by_disease_type'][1],
                }
            if disease_type == 'bronchiolitis':
                return {
                    'fever': p['prob_fever_uncomplicated_ALRI_by_disease_type'][2],
                    'cough': p['prob_cough_uncomplicated_ALRI_by_disease_type'][2],
                    'difficult_breathing': p['prob_difficult_breathing_uncomplicated_ALRI_by_disease_type'][2],
                    'fast_breathing': p['prob_fast_breathing_uncomplicated_ALRI_by_disease_type'][2],
                    'chest_indrawing': p['prob_chest_indrawing_uncomplicated_ALRI_by_disease_type'][2],
                    'danger_signs': p['prob_danger_signs_uncomplicated_ALRI_by_disease_type'][2],
                }
            if disease_type == 'fungal_pneumonia':  # same as probabilities for viral pneumonia
                return {
                    'fever': p['prob_fever_uncomplicated_ALRI_by_disease_type'][1],
                    'cough': p['prob_cough_uncomplicated_ALRI_by_disease_type'][1],
                    'difficult_breathing': p['prob_difficult_breathing_uncomplicated_ALRI_by_disease_type'][1],
                    'fast_breathing': p['prob_fast_breathing_uncomplicated_ALRI_by_disease_type'][1],
                    'chest_indrawing': p['prob_chest_indrawing_uncomplicated_ALRI_by_disease_type'][1],
                    'danger_signs': p['prob_danger_signs_uncomplicated_ALRI_by_disease_type'][1],
                }

        for disease in Alri.disease_type:
            self.prob_symptoms_uncomplicated_ALRI[disease] = make_symptom_probs(disease)

        # Check that each Alri type has a risk of developing each symptom
        assert self.disease_type == set(list(self.prob_symptoms_uncomplicated_ALRI.keys()))

        # -----------------------------------------------------------------------------------------------------
        # Make a dict containing the probability of additional symptoms given acquisition of complications
        # probability by complication
        def add_complication_symptom_probs(complicat):
            if complicat == 'pneumothorax':
                return {
                    'chest_pain': p[f'prob_chest_pain_adding_from_{complicat}'],
                    'cyanosis': p[f'prob_cyanosis_adding_from_{complicat}'],
                    'difficult_breathing': p[f'prob_difficult_breathing_adding_from_{complicat}'],
                }
            if complicat == 'pleural_effusion':
                return {
                    'chest_pain': p[f'prob_chest_pain_adding_from_{complicat}'],
                    'fever': p[f'prob_fever_adding_from_{complicat}'],
                    'difficult_breathing': p[f'prob_difficult_breathing_adding_from_{complicat}'],
                }
            if complicat == 'empyema':
                return {
                    'chest_pain': p[f'prob_chest_pain_adding_from_{complicat}'],
                    'fever': p[f'prob_fever_adding_from_{complicat}'],
                    'respiratory_distress': p[f'prob_respiratory_distress_adding_from_{complicat}'],
                }
            if complicat == 'lung_abscess':
                return {
                    'chest_pain': p[f'prob_chest_pain_adding_from_{complicat}'],
                    'fast_breathing': p[f'prob_fast_breathing_adding_from_{complicat}'],
                    'fever': p[f'prob_fever_adding_from_{complicat}'],
                }
            if complicat == 'respiratory_failure':
                return {
                    'cyanosis': p[f'prob_cyanosis_adding_from_{complicat}'],
                    'fast_breathing': p[f'prob_fast_breathing_adding_from_{complicat}'],
                    'difficult_breathing': p[f'prob_difficult_breathing_adding_from_{complicat}'],
                    'danger_signs': p[f'prob_danger_signs_adding_from_{complicat}'],
                }
            if complicat == 'sepsis':
                return {
                    'fever': p[f'prob_fever_adding_from_{complicat}'],
                    'fast_breathing': p[f'prob_fast_breathing_adding_from_{complicat}'],
                    'danger_signs': p[f'prob_danger_signs_adding_from_{complicat}'],

                }
            if complicat == 'meningitis':
                return {
                    'fever': p[f'prob_fever_adding_from_{complicat}'],
                    'headache': p[f'prob_headache_adding_from_{complicat}'],
                    'danger_signs': p[f'prob_danger_signs_adding_from_{complicat}'],
                }

        for complication in Alri.complications:
            self.prob_extra_symptoms_complications[complication] = add_complication_symptom_probs(complication)

        # Check that each complication has a risk of developing each symptom
        assert self.complications == set(list(self.prob_extra_symptoms_complications.keys()))

        # =====================================================================================================
        # APPLY A LINEAR MODEL FOR THE RISK OF DEATH DUE TO Alri
        # -----------------------------------------------------------------------------------------------------
        def linear_model_for_death(disease_type):
            return LinearModel(
                LinearModelType.MULTIPLICATIVE,
                1.0,
                Predictor('ri_ALRI_disease_type').when(f'{disease_type}', p[f'base_death_rate_ALRI_by_{disease_type}']),
                Predictor('tmp_hv_inf').when(True, p['rr_death_ALRI_HIV']),
                Predictor('tmp_malnutrition').when(True, p['rr_death_ALRI_SAM']),
                Predictor('tmp_low_birth_weight').when(True, p['rr_death_ALRI_low_birth_weight']),
                Predictor('age_years')  .when('.between(1,1)', p['rr_death_ALRI_age12to23mo'])
                                        .when('.between(2,4)', p['rr_death_ALRI_age24to59mo']),
                Predictor('ri_ALRI_complications').when('sepsis', p['rr_death_ALRI_sepsis']),
                Predictor('ri_ALRI_complications').when('respiratory_failure', p['rr_death_ALRI_respiratory_failure']),
                Predictor('ri_ALRI_complications').when('meningitis', p['rr_death_ALRI_meningitis']),
                # Predictor('ri_oxygen_therapy_given').when(True, p['oxygen_therapy_effectiveness_ALRI']),
                # Predictor('ri_antibiotic_administered').when(True, p['antibiotic_therapy_effectiveness_ALRI']),
            )
        # TODO: can I do a scaled linear model for death at the individual level? - calibrations

        for disease in Alri.disease_type:
            self.mortality_equations_by_disease[disease] = linear_model_for_death(disease)
        # -----------------------------------------------------------------------------------------------------

    def determine_disease_type(self, pathogen, age):
        """Determine the disease that is caused by infection with this pathogen for a particular person at a particular
        time: from among self.disease_type"""
        # todo - @Ines -- I don't think the original version was doing what you thought it was. Can we discuss what is
        #  needed here? I have made a guess at what is useful but please check.

        p = self.parameters

        if patho in self.bacterial_patho:
            disease_type = 'bacterial_pneumonia'

        elif patho in self.viral_patho:
            if age < 2:
                disease_type =  'viral_pneumonia' if (self.rng.rand() < p[f'proportion_viral_pneumonia_by_{patho}']) \
                    else 'bronchiolitis'
            else:
                disease_type =  'viral_pneumonia'

        elif patho in self.fungal_patho:
            disease_type =  'fungal_pneumonia'

        else:
            raise ValueError

        assert disease_type in self.disease_type
        return disease_type

    def uncomplicated_alri_symptoms(self, disease, person_id, duration_in_days):
        """
        Assigns clinical symptoms to uncomplicated Alri.
        It schedules the initial symptoms of the disease and schedules resolution
        :param disease:
        :param person_id:
        :param duration_in_days:
        :return:
        """
        rng = self.rng

        # ----------------------- Allocate symptoms to onset of Alri ----------------------
        prob_symptoms_uncomplicated_alri = self.prob_symptoms_uncomplicated_ALRI[disease]

        symptoms_for_this_person = list()
        for symptom, prob in prob_symptoms_uncomplicated_alri.items():
            if rng.rand() < prob:
                symptoms_for_this_person.append(symptom)
                self.ALRI_symptoms.set([person_id], f'{symptom}')

        # Onset symptoms:
        for symptom in symptoms_for_this_person:
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove='+',
                disease_module=self,
                duration_in_days=duration_in_days
            )

        return symptoms_for_this_person

    def alri_with_complications_symptoms(self, complication, person_id, duration_in_days):
        """
        Assigns clinical symptoms to complications. Symptoms can be resolved by treatment
        :param complication:
        :param person_id:
        :param duration_in_days:
        :return:
        """

        symptoms_from_complications = list()
        # get the symptoms for this complication:
        for symptom, prob in self.prob_extra_symptoms_complications[complication].items():
            if self.rng.rand() < prob:
                symptoms_from_complications.append(symptom)
                # add symptoms from complication to the symptoms list on the bitsetHandler
                self.ALRI_symptoms.set([person_id], f'{symptom}')

        for symptom in symptoms_from_complications:
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove='+',
                disease_module=self,
                duration_in_days=duration_in_days
            )

    def complications_append(self, person_id, complication):
        """
        This function is called at the onset of complications in AlriOnsetEvent
        and AlriSystemicComplicationsOnsetEvent.
        It adds the complication to the list of complications for the person
        :param person_id:
        :param complication:
        :return:
        """
        # if already having this complication, no need to add to the list
        if self.ALRI_complications.has_all([person_id], f'{complication}', first=True):
            return
        else:  # append complications for this person using the BitsetHandler
            self.ALRI_complications.set([person_id], f'{complication}')

    def do_alri_treatment(self, person_id, prob_of_cure):
        """Helper function that enacts the effects of a treatment to Alri caused by a pathogen.
        It will only do something if the Alri is caused by a pathogen (this module). It will not allow any effect
         if the respiratory infection is caused by another module.
        * Log the treatment date
        * Prevents this episode of Alri
         from causing a death
        * Schedules the cure event, at which symptoms are alleviated.
        """
        df = self.sim.population.props
        person = df.loc[person_id]

        if not person.is_alive:
            return

        # Do nothing if the Alri has not been caused by a pathogen
        if not (
            (person.ri_primary_ALRI_pathogen != 'not_applicable') &
            (person.ri_ALRI_event_date_of_onset <= self.sim.date <= person.ri_end_of_last_alri_episode)
        ):
            return

        # Log that the treatment is provided:
        df.at[person_id, 'ri_ALRI_tx_start_date'] = self.sim.date
        df.at[person_id, 'ri_ALRI_treatment_counter'] += 1

        # Determine if the treatment is effective
        if prob_of_cure > self.rng.rand():
            # If treatment is successful: cancel death and schedule cure event
            self.cancel_death_date(person_id)
            self.sim.schedule_event(AlriCureEvent(self, person_id),
                                    self.sim.date + DateOffset(
                                        days=self.parameters['days_between_treatment_and_cure']
                                    ))
        # else:  # not improving seek care or death
        #     self.do_when_not_improving(person_id)

    def cancel_death_date(self, person_id):
        """
        Cancels a scheduled date of death due to Alri for a person. This is called within do_treatment_alri function,
        and prior to the scheduling the CureEvent to prevent deaths happening in the time between
        a treatment being given and the cure event occurring.
        :param person_id:
        :return:
        """
        df = self.sim.population.props
        df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT

    def cancel_complication_onset(self, person_id, complication):
        """
        Cancels the onset of complications for a person,
        when treatment was administered before the scheduled date of onset in
        AlriOnsetEvent and AlriSystemicComplicationsOnsetEvent.
        :param person_id:
        :param complication:
        :return:
        """
        # depending on treatment type: antibiotic administrations vs oxygen supplementation - apply a probability
        # currently simplified to cancel the occurrence of complication if treatment was given before scheduled onset
        self.ALRI_complications.unset([person_id], f'{complication}')

    # def do_when_not_improving(self, person_id):
    #     """
    #     Prolongs the signs and symptoms of disease
    #     """
    #     df = self.sim.population.props
    #     person = df.loc[person_id]
    #     # probability of those who follow-up for further care
    #     # TODO: Care-seeking for those who sought care but following up for further care
    #
    #     # function only for those who sought care and started treatment but not improving
    #     if not (
    #         (person.ri_ALRI_tx_start_date <= self.sim.date)
    #     ):
    #         return
    #
    #     if self.parameters['prob_seek_follow_up_care_after_treatment_failure'] > self.rng.rand():
    #         # schedule follow-up event
    #         self.sim.modules['DxAlgorithmChild'].do_when_facility_level_1(person_id=person_id, hsi_event=self)
    # not working line


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------
class AlriPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the main event that runs the acquisition of pathogens that cause Alri.
        It determines who is infected and when and schedules individual IncidentCase events to represent onset.

        A known issue is that Alri events are scheduled based on the risk of current age but occur a short time
        later when the children will be slightly older. This means that when comparing the model output with data, the
        model slightly under-represents incidence among younger age-groups and over-represents incidence among older
        age-groups. This is a small effect when the frequency of the polling event is high.
    """
    # TODO: how to fix this

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=2))
        # NB. The frequency of the occurrences of this event can be edited safely.

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        p = self.module.parameters
        rng = self.module.rng

        # Compute the incidence rate for each person getting Alri and then convert into a probability
        # getting all children that do not currently have an Alri episode (never had or last episode resolved)
        mask_could_get_new_alri_event = \
            df['is_alive'] & (df['age_years'] < 5) & \
            ((df.ri_end_of_last_alri_episode < self.sim.date) | pd.isnull(df.ri_end_of_last_alri_episode))

        # Compute the incidence rate for each person acquiring Alri
        inc_of_acquiring_alri = pd.DataFrame(index=df.loc[mask_could_get_new_alri_event].index)
        for pathogen in m.pathogens:
            inc_of_acquiring_alri[pathogen] = m.incidence_equations_by_pathogen[pathogen] \
                .predict(df.loc[mask_could_get_new_alri_event])

        # Convert the incidence rates that are predicted by the model into risk of an event occurring before the next
        # polling event
        fraction_of_a_year_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(
            1, 'Y')
        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')
        probs_of_acquiring_pathogen = 1 - np.exp(-inc_of_acquiring_alri * fraction_of_a_year_until_next_polling_event)

        # Create the probability of getting 'any' pathogen:
        # (Assumes that pathogens are mutually exclusive); Prevents probability being greater than 1.0.
        prob_of_acquiring_any_pathogen = probs_of_acquiring_pathogen.sum(axis=1).clip(upper=1.0)
        assert all(prob_of_acquiring_any_pathogen <= 1.0)

        # Determine which persons will acquire any pathogen:
        person_id_that_acquire_pathogen = prob_of_acquiring_any_pathogen.index[
            rng.rand(len(prob_of_acquiring_any_pathogen)) < prob_of_acquiring_any_pathogen]

        # Determine which pathogen each person will acquire (among those who will get a pathogen)
        # and create the event for the onset of new infection
        for person_id in person_id_that_acquire_pathogen:
            # ----------------------- Allocate a pathogen to the person ----------------------
            p_by_pathogen = probs_of_acquiring_pathogen.loc[person_id].values
            normalised_p_by_pathogen = p_by_pathogen / sum(p_by_pathogen)
            pathogen = rng.choice(probs_of_acquiring_pathogen.columns, p=normalised_p_by_pathogen)

            # ----------------- Determine the Alri disease type for this case -----------------
            alri_disease_type_for_this_person = m.proportions_of_ALRI_disease_types_by_pathogen[pathogen]

            # ------- Allocate a secondary bacterial pathogen in co-infection with primary viral pneumonia -------
            bacterial_patho_in_ALRI_coinfection = 'not_applicable'
            if pathogen in self.module.viral_patho:
                prob_bacterial_infection = m.prob_secondary_bacterial_infection.predict(df.loc[[person_id]]).values[0]
                if rng.rand() < prob_bacterial_infection:
                    bacterial_coinfection_pathogen = rng.choice(list(self.module.bacterial_patho),
                                                                p=p['proportion_bacterial_coinfection_pathogen'])
                    bacterial_patho_in_ALRI_coinfection = bacterial_coinfection_pathogen
                    # update to co-infection property
                    alri_disease_type_for_this_person = 'bacterial_pneumonia'
                else:
                    bacterial_patho_in_ALRI_coinfection = 'none'
            if pathogen in self.module.bacterial_patho:
                bacterial_patho_in_ALRI_coinfection = 'not_applicable'
                assert bacterial_patho_in_ALRI_coinfection == 'not_applicable'

            if pathogen in self.module.fungal_patho:
                bacterial_patho_in_ALRI_coinfection = 'none'

            # ----------------------- Allocate a date of onset of Alri ----------------------
            date_onset = self.sim.date + DateOffset(days=np.random.randint(0, days_until_next_polling_event))

            # ----------------------- Duration of the Alri event -----------------------
            duration_in_days_of_alri = max(7, int(
                14 + (-2 + 4 * rng.rand())))  # assumes uniform interval around mean duration with range 4 days
            # TODO: find natural course of disease duration estimates

            # ----------------------- Create the event for the onset of infection -------------------
            self.sim.schedule_event(
                event=AlriIncidentCase(
                    module=self.module,
                    person_id=person_id,
                    pathogen=pathogen,
                    disease_type=alri_disease_type_for_this_person,
                    co_bacterial_patho=bacterial_patho_in_ALRI_coinfection,
                    duration_in_days=duration_in_days_of_alri,
                ),
                date=date_onset
            )


class AlriIncidentCase(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of the infection that causes Alri.
     * Refreshes all the properties so that they pertain to this current episode of Alri
     * Imposes the symptoms
     * Schedules relevant natural history event {(AlriWithPulmonaryComplicationsEvent) and
     (AlriWithSevereComplicationsEvent)
       (either AlriNaturalRecoveryEvent or AlriDeathEvent)}
     * Updates a counter for incident cases
    """
    AGE_GROUPS = {0: '0y', 1: '1y', 2: '2-4y', 3: '2-4y', 4: '2-4y'}

    def __init__(self, module, person_id, pathogen, disease_type, co_bacterial_patho, duration_in_days):
        super().__init__(module, person_id=person_id)
        self.pathogen = pathogen
        self.bacterial_pathogen = co_bacterial_patho
        self.disease = disease_type
        self.duration_in_days = duration_in_days

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        rng = self.module.rng

        # The event should not run if the person is not currently alive
        if not df.at[person_id, 'is_alive']:
            return

        # Update the properties in the dataframe:
        df.at[person_id, 'ri_current_ALRI_status'] = True
        df.at[person_id, 'ri_primary_ALRI_pathogen'] = self.pathogen
        if self.bacterial_pathogen != 'not_applicable':
            df.at[person_id, 'ri_secondary_bacterial_pathogen'] = self.bacterial_pathogen
        df.at[person_id, 'ri_ALRI_disease_type'] = self.disease
        df.at[person_id, 'ri_ALRI_event_date_of_onset'] = self.sim.date
        df.at[person_id, 'ri_ALRI_complications'] = 0  # all disease start as non-severe
        df.at[person_id, 'ri_ALRI_cases_counter'] += 1

        if df.at[person_id, 'ri_primary_ALRI_pathogen'] == any(self.module.bacterial_patho):
            assert df.at[person_id, 'ri_secondary_bacterial_pathogen'] == 'not_applicable'

        # ----------------------------------- clinical symptoms -----------------------------------
        # update clinical symptoms for new uncomplicated Alri
        uncomplicated_symptoms_for_this_person = self.module.uncomplicated_alri_symptoms(
            disease=self.disease, person_id=person_id,
            duration_in_days=self.duration_in_days)

        # date for recovery with uncomplicated Alri
        date_of_outcome = self.module.sim.date + DateOffset(days=self.duration_in_days)

        # COMPLICATIONS -----------------------------------------------------------------------------------------
        # here, assign the probability of hypoxia
        prob_developing_hypoxia = m.risk_of_decreased_peripheral_oxygen_level.predict(
            df.loc[[person_id]]).values[0]
        if rng.rand() < prob_developing_hypoxia:
            df.at[person_id, 'ri_peripheral_oxygen_saturation'] = 'SpO2<93%'

        # here, assign the probability of each complication and schedule the event onset
        complications_for_this_person = list()
        for complication in self.module.complications:
            prob_developing_each_complication = m.risk_of_developing_ALRI_complications[complication].predict(
                df.loc[[person_id]]).values[0]
            if rng.rand() < prob_developing_each_complication:
                complications_for_this_person.append(complication)
        if 'pleural_effusion' in complications_for_this_person:
            if self.disease == 'bacterial_pneumonia':
                prob_developing_empyema = m.risk_of_developing_ALRI_complications['empyema'].predict(
                    df.loc[[person_id]]).values[0]
                if rng.rand() < prob_developing_empyema:
                    complications_for_this_person.append('empyema')

        # if at least one complication developed in the Alri event, schedule the event
        if len(complications_for_this_person) != 0:
            for complication in complications_for_this_person:
                # if complication in self.module.lung_complications:  # schedule for pulmonary complications
                date_onset_complications = self.module.sim.date + DateOffset(
                    days=np.random.randint(0, high=self.duration_in_days))
                # schedule the complication event
                self.sim.schedule_event(AlriOnsetEvent(
                    self.module, person_id,
                    duration_in_days=self.duration_in_days,
                    symptoms=uncomplicated_symptoms_for_this_person,
                    complication=complication
                ),
                    date_onset_complications)

            df.at[person_id, 'ri_ALRI_event_recovered_date'] = pd.NaT
            df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT
        else:
            # if NO complications for this Alri event, schedule a natural recovery
            self.sim.schedule_event(AlriNaturalRecoveryEvent(self.module, person_id),
                                    date_of_outcome)
            df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT

        # Record 'episode end' data. This the date when this episode ends. It is the last possible data that any HSI
        # could affect this episode.
        df.at[person_id, 'ri_end_of_last_alri_episode'] = date_of_outcome + DateOffset(
            days=self.module.parameters['days_between_treatment_and_cure']
        )

        # Determine death outcome -------------------------------------------------------------------------
        for disease in m.disease_type:
            prob_death_from_ALRI = m.mortality_equations_by_disease[disease].predict(df.loc[[person_id]]).values[0]
            if rng.rand() < prob_death_from_ALRI:
                self.sim.schedule_event(AlriDeathEvent(self.module, person_id),
                                        date_of_outcome)
            else:
                self.sim.schedule_event(AlriNaturalRecoveryEvent(self.module, person_id),
                                        date_of_outcome)

        # Add this incident case to the tracker
        (self.module.logging_event).new_case(age=df.at[person_id, 'age_years'], pathogen=self.pathogen)

class AlriOnsetEvent(Event, IndividualScopeEventMixin):
    """
           This Event is for the onset of complications from Alri. For some untreated children,
           this occurs a set number of days after onset of disease.
           It sets the property 'ri_ALRI_complications' to each initial complication
           and schedules further severe complications.
       """

    def __init__(self, module, person_id, duration_in_days, symptoms, complication):
        super().__init__(module, person_id=person_id)
        self.duration_in_days = duration_in_days
        self.complication = complication
        self.symptoms = symptoms

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        rng = self.module.rng

        # terminate the event if the person has already died.
        if not df.at[person_id, 'is_alive']:
            return

        # cancel the schedules onset of complication if treatment was administered for this episode
        if df.at[person_id, 'ri_ALRI_event_date_of_onset'] < df.at[person_id, 'ri_ALRI_tx_start_date'] < self.sim.date:
            self.module.cancel_complication_onset(person_id=person_id, complication=self.complication)
            return

        # else: do the rest below
        # Determine severe complication outcome -------------------------------------------------------------------
        date_of_recovery = df.at[person_id, 'ri_ALRI_event_date_of_onset'] + DateOffset(self.duration_in_days)
        # use the outcome date to get the number of days from onset of lung complication to outcome
        delta_date = date_of_recovery - self.sim.date
        delta_in_days = delta_date.days

        # Schedule the progression of pulmonary complication to severe systemic complications
        date_of_onset_severe_complication = self.sim.date + DateOffset(
            days=np.random.randint(0, high=delta_in_days))

        # add the complications for this person to the ri_ALRI_complication bitsetHandler
        self.module.complications_append(person_id=person_id, complication=self.complication)
        # add the symptoms from the complications to the ri_current_ALRI_symptoms
        self.module.alri_with_complications_symptoms(
            person_id=person_id,
            complication=self.complication,
            duration_in_days=self.duration_in_days)
        # apply the probability for progressing further in the disease
        for severe_complication in ['respiratory_failure', 'sepsis']:
            prob_progressing_severe_complication = \
                m.risk_of_progressing_to_severe_complications[severe_complication].predict(
                    df.loc[[person_id]]).values[0]
            if rng.rand() < prob_progressing_severe_complication:
                self.sim.schedule_event(AlriSystemicComplicationsOnsetEvent(
                    self.module, person_id,
                    duration_in_days=self.duration_in_days,
                    complication=severe_complication),
                    date_of_onset_severe_complication)
            else:
                # schedule recovery event
                # df.at[person_id, 'ri_ALRI_event_recovered_date'] = date_of_recovery
                self.sim.schedule_event(AlriNaturalRecoveryEvent(self.module, person_id), date_of_recovery)


class AlriSystemicComplicationsOnsetEvent(Event, IndividualScopeEventMixin):
    """
        This Event is for the onset of severe complications from Alri. For some untreated children,
        this occurs a set number of days after onset of disease.
        It sets the property 'ri_ALRI_complications' to each complication and schedules the death.
    """

    def __init__(self, module, person_id, duration_in_days, complication):
        super().__init__(module, person_id=person_id)
        self.duration_in_days = duration_in_days
        self.complication = complication

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # terminate the event if the person has already died.
        if not df.at[person_id, 'is_alive']:
            return

        # cancel the schedules onset of complication if treatment was administered for this episode
        if df.at[person_id,
                 'ri_ALRI_event_date_of_onset'] <= df.at[person_id, 'ri_ALRI_tx_start_date'] <= self.sim.date:
            self.module.cancel_complication_onset(person_id=person_id, complication=self.complication)
            return

        # add severe complications to ri_ALRI_complications
        self.module.complications_append(person_id=person_id, complication=self.complication)

        # symptoms from severe complication, add to the initial list of Alri symptoms
        self.module.alri_with_complications_symptoms(
            person_id=person_id,
            complication=self.complication,
            duration_in_days=self.duration_in_days)


class AlriNaturalRecoveryEvent(Event, IndividualScopeEventMixin):
    """
    This is the Natural Recovery event.
    It is part of the natural history and represents the end of an episode of Alri
    It does the following:
        * resolves all symptoms caused by Alri
        * resolves all ri_ properties back to 'none','not_applicable', or False
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]

        # The event should not run if the person is not currently alive
        if not person.is_alive:
            return

        # Confirm that this is event is occurring during a current episode of Alri
        assert person.ri_ALRI_event_date_of_onset <= self.sim.date <= person.ri_end_of_last_alri_episode

        # Check that the person is not scheduled to die in this episode
        assert pd.isnull(person.ri_ALRI_event_death_date)

        # if not pd.isnull(person.ri_ALRI_event_death_date):
        #     return

        # set recovery date for this current episode
        df.at[person_id, 'ri_ALRI_event_recovered_date'] = self.sim.date
        df.at[person_id, 'ri_ALRI_recovery_counter'] += 1

        # clear properties to inital state
        df.at[person_id, 'ri_current_ALRI_status'] = False
        # df.at[person_id, 'ri_ALRI_event_date_of_onset'] = pd.NaT
        df.at[person_id, 'ri_primary_ALRI_pathogen'] = 'not_applicable'
        df.at[person_id, 'ri_secondary_bacterial_pathogen'] = 'not_applicable'
        df.at[person_id, 'ri_ALRI_disease_type'] = 'not_applicable'
        df.at[person_id, 'ri_current_ALRI_symptoms'] = 0
        df.at[person_id, 'ri_ALRI_complications'] = 0
        df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT

        # clear the treatment prperties
        df.at[person_id, 'ri_ALRI_treatment'] = False
        df.at[person_id, 'ri_ALRI_tx_start_date'] = pd.NaT
        # df.at[person_id, 'ri_end_of_last_alri_episode'] = pd.NaT

        # Resolve all the symptoms immediately
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id,
                                                          disease_module=self.sim.modules['Alri'])


class AlriCureEvent(Event, IndividualScopeEventMixin):
    """
       This is the cure event. It is scheduled by an HSI treatment event.
       It enacts the actual "cure" of the person that is caused (after some delay) by the treatment administered.
       It does the following:
           * Sets the date of recovery to today's date
           * Resolves all symptoms caused by Alri
       """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("AlriCureEvent: Stopping Alri treatment and curing person %d", person_id)
        df = self.sim.population.props

        # terminate the event if the person has already died.
        if not df.at[person_id, 'is_alive']:
            return

        # Cure should not happen if the person has already recovered for the current episode
        if df.at[person_id,
                 'ri_ALRI_event_date_of_onset'] <= df.at[person_id, 'ri_ALRI_event_recovered_date'] <= self.sim.date:
            return

        # Confirm that this is event is occurring during a current episode of Alri
        if not (
                   df.at[person_id, 'ri_ALRI_event_date_of_onset']) <= self.sim.date <= \
               (df.at[person_id, 'ri_end_of_last_alri_episode']):
            return

        # This event should only run after the person has received a treatment during this episode
        assert (
            (df.at[person_id, 'ri_ALRI_event_date_of_onset']) <=
            (df.at[person_id, 'ri_ALRI_tx_start_date']) <= self.sim.date)

        # If cure should go ahead, check that it is after when the person has received a treatment during this episode
        assert (
            df.at[person_id, 'ri_ALRI_event_date_of_onset'] <=
            df.at[person_id, 'ri_ALRI_tx_start_date'] <=
            self.sim.date <=
            df.at[person_id, 'ri_end_of_last_alri_episode']
        )

        # Stop the person from dying of Alri (if they were going to die)
        df.at[person_id, 'ri_ALRI_event_recovered_date'] = self.sim.date
        df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT

        df.at[person_id, 'ri_ALRI_recovery_counter'] += 1

        # clear properties to inital state
        # df.at[person_id, 'ri_ALRI_event_date_of_onset'] = pd.NaT
        df.at[person_id, 'ri_current_ALRI_status'] = False
        df.at[person_id, 'ri_primary_ALRI_pathogen'] = 'not_applicable'
        df.at[person_id, 'ri_secondary_bacterial_pathogen'] = 'not_applicable'
        df.at[person_id, 'ri_ALRI_disease_type'] = 'not_applicable'
        df.at[person_id, 'ri_current_ALRI_symptoms'] = 0
        df.at[person_id, 'ri_ALRI_complications'] = 0

        # clear the treatment prperties
        df.at[person_id, 'ri_ALRI_treatment'] = False
        df.at[person_id, 'ri_ALRI_tx_start_date'] = pd.NaT
        # df.at[person_id, 'ri_end_of_last_alri_episode'] = pd.NaT

        # Resolve all the symptoms immediately
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id,
                                                          disease_module=self.sim.modules['Alri'])


class AlriDeathEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the death of someone that is caused by the infection with a pathogen that causes Alri.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # The event should not run if the person is not currently alive
        if not df.at[person_id, 'is_alive']:
            return

        # Confirm that this is event is occurring during a current episode of Alri
        assert (
            (df.at[person_id, 'ri_ALRI_event_date_of_onset']) <=
            self.sim.date <=
            (df.at[person_id, 'ri_end_of_last_alri_episode'])
        )

        # Check if person should still die of Alri
        if (
            df.at[person_id, 'ri_ALRI_event_death_date'] == self.sim.date) and \
            pd.isnull(df.at[person_id, 'ri_ALRI_event_recovered_date']
                      ):
            df.at[person_id, 'ri_ALRI_event_death_date'] = self.sim.date
            self.sim.schedule_event(demography.InstantaneousDeath(self.module,
                                                                  person_id,
                                                                  cause='ALRI_' + df.at[
                                                                      person_id, 'ri_primary_ALRI_pathogen']
                                                                  ), self.sim.date)

            # todo - ensure that pathogen is reported here
            self.module.logging_event.new_death()


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class AlriLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This Event logs the number of incident cases that have occurred since the previous logging event.
    Analysis scripts expect that the frequency of this logging event is once per year.
    """

    def __init__(self, module):
        # This event to occur every year
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

        # initialise trakcers of incident cases, new recoveries, new treatments and deaths due to ALRI
        age_grps= {**{0: "0", 1: "1", 2: "2-4", 3: "2-4", 4: "2-4"}, **{x: "5+" for x in range(5, 100)}}

        self.trackers = dict()
        self.trackers['incident_cases'] = Tracker(age_grps=age_grps, pathogens=self.module.pathogens)
        self.trackers['recovered_cases'] = Tracker(age_grps=age_grps, pathogens=self.module.pathogens)
        self.trackers['treated_cases'] = Tracker(age_grps=age_grps, pathogens=self.module.pathogens)
        self.trackers['deaths'] = Tracker(age_grps=age_grps, pathogens=self.module.pathogens)

    def new_case(self, age, pathogen):
        self.trackers['incident_cases'].add(age=age, pathogen=pathogen)

    def new_recovered_case(self, age, pathogen):
        self.trackers['recovered_cases'].add(age=age, pathogen=pathogen)

    def new_treated_case(self, age, pathogen):
        self.trackers['treated_cases'].add(age=age, pathogen=pathogen)

    def new_death(self):
        self.trackers['deaths'].add(age=age, pathogen=pathogen)

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
            key='counters',
            data={k: v.report_current_total() for k, v in self.trackers.items()},
            description='Counts of cases, recovery, treatment and death in the last year'
        )

        # 3) Reset the trackers
        for trakcer in self.trackers:
            tracker.reset()


class Tracker():
    """Helper class to be a counter for number of events occuring by age-group and by pathogen."""

    def __init__(self, age_grps: dict, pathogens: list):
        """Create and initalise tracker"""

        # Check and store parameters
        self.pathogens = pathogens
        self.age_grps_lookup = age_grps
        self.unique_age_grps = list(set(self.age_grps_lookup.values()))
        self.unique_age_grps.sort()

        # Initialise Tracker
        self.tracker = None
        self.reset()

    def reset(self):
        """Produce a dict of the form: { <Age-Grp>: {<Pathogen>: <Count>} }"""
        self.tracker = {
            age: dict(zip(self.pathogens, [0] * len(self.pathogens))) for age in self.unique_age_grps
        }

    def add(self, age, pathogen):
        """Increment counter by one for a specific age and pathogen"""
        assert age in self.age_grps_lookup, 'Age not recognised'
        assert pathogen in self.pathogens, 'Pathogen no recognised'

        # increment by one:
        age_grp = self.age_grps_lookup[age]
        self.tracker[age_grp][pathogen] += 1

    def report_current_counts(self):
        return self.tracker

    def report_current_total(self):
        return self.tracker.sum()


class AlriIindividualCheckLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This Event logs the daily occurrence to a single individual child.
    """
    def __init__(self, module):
        # This logging event to occur every day
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(days=self.repeat))

        # Find the person to log: the first under-five-year-old in the dataframe
        df = self.sim.population.props
        under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
        self.person_id = under5s.index[0] if len(under5s) else None

    def apply(self, population):
        """Log all properties for this module"""
        if self.person_id:
            df = self.sim.population.props
            logger.info(
                key='log_individual',
                data= df.loc[self.person_id, self.module.PROPERTIES.keys()].to_dict(),
                description='print of properties each day for one person (the first under-five-year-old in the dataframe)'
            )

"""
### Not looking at anything to do with HSI ###

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ==================================== HEALTH SYSTEM INTERACTION EVENTS ====================================
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ---------------------------------------------------------------------------------------------------------
# COMMUNITY LEVEL - iCCM delivered through HSAs -----------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

class HSI_iCCM_Pneumonia_Treatment_level_0(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Define the necessary information for an HSI
        # (These are blank when created; but these should be filled-in by the module that calls it)
        self.TREATMENT_ID = 'HSI_iCCM_Pneumonia_Treatment_level_0'

        # APPP_FOOTPRINT: village clinic event takes small amount of time for DCSA
        appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        appt_footprint['ConWithDCSA'] = 0.5
        # Demonstrate the equivalence with:
        assert appt_footprint == self.make_appt_footprint({'ConWithDCSA': 0.5})

        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ConWithDCSA': 0.5})
        self.ACCEPTED_FACILITY_LEVEL = 0  # Can occur at facility-level 0 / community with HSAs
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~

        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # whole package of interventions
        pkg_code_pneumonia = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Pneumonia treatment (children)',
                            'Intervention_Pkg_Code'])[0]

        # individual items
        item_code1 = pd.unique(
            consumables.loc[consumables['Items'] == 'Amoxycillin 125mg/5ml suspension, PFR_0.025_CMST', 'Item_Code']
        )[0]
        item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'Paracetamol, tablet, 100 mg', 'Item_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code_pneumonia: 1},
                              'Item_Code': {item_code1: 1, item_code2: 1}}

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_pneumonia]:
            logger.debug(key='debug', data='PkgCode1 is available, so use it.')
            self.module.do_alri_treatment(
                person_id=person_id,
                prob_of_cure=self.module.parameters[
                    'prob_of_cure_for_uncomplicated_pneumonia_given_IMCI_pneumonia_treatment']
            )
        else:
            logger.debug(key='debug', data="PkgCode1 is not available, so can't use it.")
            # todo: prbability of referral if no drug available
            # referral to facility level 1 - go through the IMCI algorithm with do_facility_level_1
            self.sim.modules['DxAlgorithmChild'].do_when_facility_level_1(person_id=person_id, hsi_event=self)

        # check to see if all consumables returned (for demonstration purposes):
        # check to see if all consumables returned (for demonstration purposes):
        all_available = (outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_pneumonia]) and\
                        (outcome_of_request_for_consumables['Item_Code'][item_code1]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code2])
        # use helper function instead (for demonstration purposes)
        all_available_using_helper_function = self.get_all_consumables(
            item_codes=[item_code1, item_code2],
            pkg_codes=[pkg_code_pneumonia]
        )
        # Demonstrate equivalence
        assert all_available == all_available_using_helper_function

        # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='debug', data='HSI_iCCM_Pneumonia_Treatment_level_0: did not run')
        pass


class HSI_iCCM_Severe_Pneumonia_Treatment_level_0(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'iCCM_Severe_Pneumonia_Treatment_level_0'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ALERT_OTHER_DISEASES = []
        self.ACCEPTED_FACILITY_LEVEL = 0

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        # store management info:
        care_management_info = dict()

        # first dose of antibiotic is given - give first dose of oral antibiotic
        # (amoxicillin tablet - 250mg)
        # Age 2 months up to 12 months - 1 tablet
        # Age 12 months up to 5 years - 2 tablets

        # give first dose of an appropriate antibiotic
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        item_code_amoxycilin = pd.unique(
            consumables.loc[consumables['Items'] == 'Amoxycillin 250mg_1000_CMST', 'Item_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {}, 'Item_Code': {item_code_amoxycilin: 1}}

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)
        if outcome_of_request_for_consumables['Intervention_Package_Code'][item_code_amoxycilin]:
            care_management_info.update({
                'treatment_plan': 'referral_with_first_dose_antibiotic'})

        # then refer to facility level 1 or 2 - go though the IMCI algorithm
        self.sim.modules['DxAlgorithmChild'].do_when_facility_level_1(person_id=person_id, hsi_event=self)
        self.sim.modules['DxAlgorithmChild'].do_when_facility_level_2(person_id=person_id, hsi_event=self)
        # todo: which facility level is closest to be refered to?
        # todo: what about those wo are lost to follow up? - incorporate in the code


# ---------------------------------------------------------------------------------------------------------
# PRIMARY LEVEL - IMCI delivered at facility level 1 / health centres -------------------------------------
# ---------------------------------------------------------------------------------------------------------

class HSI_IMCI_No_Pneumonia_Treatment_level_1(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Define the necessary information for an HSI - interventions for common cold at facility level 1
        # (These are blank when created; but these should be filled-in by the module that calls it)
        self.TREATMENT_ID = 'HSI_IMCI_No_Pneumonia_Treatment_level_1'

        # APP_FOOTPRINT: health centre event takes small amount of time for health workers
        appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        appt_footprint['Under5OPD'] = 1  # This requires one out patient
        # Demonstrate the equivalence with:
        assert appt_footprint == self.make_appt_footprint({'Under5OPD': 1})

        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1  # Can occur at facility-level 1 / health centres
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        # no consumables, home advice
        # store management info:
        care_management_info = dict()
        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~

        # Make request for some consumables
        care_management_info.update({
            'treatment_plan': 'home_counselling'})
        self.module.child_disease_management_information.update({person_id: care_management_info})


class HSI_IMCI_Pneumonia_Treatment_level_1(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Define the necessary information for an HSI - interventions for non-severe pneumonia at facility level 1
        # (These are blank when created; but these should be filled-in by the module that calls it)
        self.TREATMENT_ID = 'HSI_IMCI_Pneumonia_Treatment_level_1'

        # APP_FOOTPRINT: health centre event takes small amount of time for health workers
        appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        appt_footprint['Under5OPD'] = 1  # This requires one out patient
        # Demonstrate the equivalence with:
        assert appt_footprint == self.make_appt_footprint({'Under5OPD': 1})

        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1  # Can occur at facility-level 1 / health centres
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        # store management info:
        care_management_info = dict()
        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~

        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # whole package of interventions
        pkg_code_pneumonia = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Pneumonia treatment (children)',
                            'Intervention_Pkg_Code'])[0]

        # individual items
        item_code1 = pd.unique(consumables.loc[consumables['Items'] == 'Paracetamol, tablet, 100 mg', 'Item_Code'])[
            0]
        item_code2 = pd.unique(
            consumables.loc[consumables['Items'] == 'Salbutamol, tablet, 4 mg', 'Item_Code'])[0]
        item_code3 = pd.unique(
            consumables.loc[consumables['Items'] == 'Salbutamol, syrup, 2 mg/5 ml', 'Item_Code'])[0]
        item_code4 = pd.unique(
            consumables.loc[consumables['Items'] == 'Salbutamol sulphate 1mg/ml, 5ml_each_CMST', 'Item_Code'])[0]
        item_code5 = pd.unique(
            consumables.loc[consumables['Items'] == 'Amoxycillin 125mg/5ml suspension, PFR_0.025_CMST', 'Item_Code'])[0]
        item_code6 = pd.unique(
            consumables.loc[consumables['Items'] == 'Amoxycillin 250mg_1000_CMST', 'Item_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code_pneumonia: 1},
                              'Item_Code': {item_code1: 1, item_code2: 1, item_code3: 1,
                                            item_code4: 1, item_code5: 1, item_code6: 1}}

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_pneumonia]:
            logger.debug(key='debug', data='PkgCode1 is available, so use it.')
            self.module.do_alri_treatment(
                person_id=person_id,
                prob_of_cure=self.module.parameters[
                    'prob_of_cure_for_uncomplicated_pneumonia_given_IMCI_pneumonia_treatment']
            )
            care_management_info.update({
                'treatment_plan': 'treatment_for_pneumonia'})
            self.module.child_disease_management_information.update({person_id: care_management_info})

        else:
            logger.debug(key='debug', data="PkgCode1 is not available, so can't use it.")
            # todo: probability of referral if no drug available
            self.sim.modules['DxAlgorithmChild'].do_when_facility_level_2(person_id=person_id, hsi_event=self)
            care_management_info.update({
                'treatment_plan': 'no_available_treatment', 'referral_to_level_2': True})

        self.module.child_disease_management_information.update({person_id: care_management_info})

        # todo: If coughing for more than 2 weeks or if having recurrent wheezing, assess for TB or asthma

        # todo: follow-up in 2 days

        # check to see if all consumables returned (for demonstration purposes):
        # check to see if all consumables returned (for demonstration purposes):
        all_available = (outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_pneumonia]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code1]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code2]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code3]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code4]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code5]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code6])
        # use helper function instead (for demonstration purposes)
        all_available_using_helper_function = self.get_all_consumables(
            item_codes=[item_code1, item_code2, item_code3, item_code4, item_code5, item_code6],
            pkg_codes=[pkg_code_pneumonia]
        )
        # Demonstrate equivalence
        assert all_available == all_available_using_helper_function

        # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        actual_appt_footprint['Under5OPD'] = actual_appt_footprint['Under5OPD'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='debug', data='HSI_IMCI_Pneumonia_Treatment_level_1: did not run')
        pass


class HSI_IMCI_Severe_Pneumonia_Treatment_level_1(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Define the necessary information for an HSI - interventions for severe pneumonia at facility level 1
        # (These are blank when created; but these should be filled-in by the module that calls it)
        self.TREATMENT_ID = 'HSI_IMCI_Severe_Pneumonia_Treatment_level_1'

        # APP_FOOTPRINT: health centre event takes small amount of time for health workers
        appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        appt_footprint['Under5OPD'] = 1  # This requires one out patient
        # Demonstrate the equivalence with:
        assert appt_footprint == self.make_appt_footprint({'Under5OPD': 1})

        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1  # Can occur at facility-level 1 / health centres
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        # store management info:
        care_management_info = dict()
        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~

        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # whole package of interventions
        pkg_code_severe_pneumonia = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Treatment of severe pneumonia',
                            'Intervention_Pkg_Code'])[0]

        # individual items
        item_code1 = pd.unique(consumables.loc[consumables['Items'] == 'Benzylpenicillin 3g (5MU), PFR_each_CMST',
                                               'Item_Code'])[0]
        item_code2 = pd.unique(
            consumables.loc[consumables['Items'] == 'Gentamicin Sulphate 40mg/ml, 2ml_each_CMST', 'Item_Code'])[0]
        item_code3 = pd.unique(
            consumables.loc[consumables['Items'] == 'Ceftriaxone 1g, PFR_each_CMST', 'Item_Code'])[0]
        item_code4 = pd.unique(
            consumables.loc[consumables['Items'] == 'Tube, nasogastric CH 8_each_CMST', 'Item_Code'])[0]
        item_code5 = pd.unique(
            consumables.loc[consumables['Items'] == 'Oxygen, 1000 liters, primarily with oxygen concentrators',
                            'Item_Code'])[0]
        item_code6 = pd.unique(
            consumables.loc[consumables['Items'] == 'Prednisolone 5mg_100_CMST', 'Item_Code'])[0]
        item_code7 = pd.unique(
            consumables.loc[consumables['Items'] == 'Salbutamol, syrup, 2 mg/5 mlT', 'Item_Code'])[0]
        item_code8 = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item_code9 = pd.unique(
            consumables.loc[consumables['Items'] == 'Amoxycillin 250mg_1000_CMST', 'Item_Code'])[0]
        item_code10 = pd.unique(
            consumables.loc[consumables['Items'] == 'Cannula iv  (winged with injection pot) 16_each_CMST',
                            'Item_Code'])[0]
        item_code11 = pd.unique(
            consumables.loc[consumables['Items'] == 'X-ray', 'Item_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code_severe_pneumonia: 1},
                              'Item_Code': {item_code1: 1, item_code2: 1, item_code3: 1,
                                            item_code4: 1, item_code5: 1, item_code6: 1,
                                            item_code7: 1, item_code8: 1, item_code9: 1,
                                            item_code10: 1, item_code11: 1}}

        # TODO: note that x-rays, oxygen not found at facility level 1

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_severe_pneumonia]:
            logger.debug(key='debug', data='PkgCode1 is available, so use it.')
            self.module.do_alri_treatment(
                person_id=person_id,
                prob_of_cure=self.module.parameters[
                    'prob_of_cure_for_uncomplicated_pneumonia_given_IMCI_pneumonia_treatment']
            )

            # todo: need to check the dates conflict with cure/ treatment/ death etc...
            care_management_info.update({
                'treatment_plan': 'treatment_for_severe_pneumonia'})

        else:
            logger.debug(key='debug', data="PkgCode1 is not available, so can't use it.")
            # todo: probability of referral if no drug available
            self.sim.modules['DxAlgorithmChild'].do_when_facility_level_2(person_id=person_id, hsi_event=self)
            # TODO: give first dose of antibiotic before referral
            care_management_info.update({
                'treatment_plan': 'no_available_treatment', 'referral_to_level_2': True})

        self.module.child_disease_management_information.update({person_id: care_management_info})
        # todo: If coughing for more than 2 weeks or if having recurrent wheezing, assess for TB or asthma

        # todo: follow-up in 2 days

        # check to see if all consumables returned (for demonstration purposes):
        all_available = \
            (outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_severe_pneumonia]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code1]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code2]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code3]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code4]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code5]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code6]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code7]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code8]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code9]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code10]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code11])

        # use helper function instead (for demonstration purposes)
        all_available_using_helper_function = self.get_all_consumables(
            item_codes=[item_code1, item_code2, item_code3, item_code4, item_code5, item_code6,
                        item_code7, item_code8, item_code9, item_code10, item_code11],
            pkg_codes=[pkg_code_severe_pneumonia]
        )
        # Demonstrate equivalence
        assert all_available == all_available_using_helper_function

        # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        actual_appt_footprint['Under5OPD'] = actual_appt_footprint['Under5OPD'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='debug', data='HSI_IMCI_Severe_Pneumonia_Treatment_level_1: did not run')
        pass


# ---------------------------------------------------------------------------------------------------------
# SECONDARY LEVEL - IMCI delivered at facility level 2 / hospitals ----------------------------------------
# ---------------------------------------------------------------------------------------------------------

class HSI_IMCI_Pneumonia_Treatment_level_2(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Define the necessary information for an HSI - interventions for non-severe pneumonia at facility level 2
        # (These are blank when created; but these should be filled-in by the module that calls it)
        self.TREATMENT_ID = 'HSI_IMCI_Pneumonia_Treatment_level_2'

        # APP_FOOTPRINT: hospital-level event takes a certain amount of time for health workers
        appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        appt_footprint['Under5OPD'] = 1  # This requires one out patient
        # Demonstrate the equivalence with:
        assert appt_footprint == self.make_appt_footprint({'Under5OPD': 1})

        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = 2  # Can occur at facility-level 2 / hospitals
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        # store management info:
        care_management_info = dict()
        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~

        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # whole package of interventions
        pkg_code_pneumonia = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Pneumonia treatment (children)',
                            'Intervention_Pkg_Code'])[0]

        # individual items
        item_code1 = pd.unique(consumables.loc[consumables['Items'] == 'Paracetamol, tablet, 100 mg', 'Item_Code'])[
            0]
        item_code2 = pd.unique(
            consumables.loc[consumables['Items'] == 'Salbutamol, tablet, 4 mg', 'Item_Code'])[0]
        item_code3 = pd.unique(
            consumables.loc[consumables['Items'] == 'Salbutamol, syrup, 2 mg/5 ml', 'Item_Code'])[0]
        item_code4 = pd.unique(
            consumables.loc[consumables['Items'] == 'Salbutamol sulphate 1mg/ml, 5ml_each_CMST', 'Item_Code'])[0]
        item_code5 = pd.unique(
            consumables.loc[consumables['Items'] == 'Amoxycillin 125mg/5ml suspension, PFR_0.025_CMST', 'Item_Code'])[0]
        item_code6 = pd.unique(
            consumables.loc[consumables['Items'] == 'Amoxycillin 250mg_1000_CMST', 'Item_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code_pneumonia: 1},
                              'Item_Code': {item_code1: 1, item_code2: 1, item_code3: 1,
                                            item_code4: 1, item_code5: 1, item_code6: 1}}

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_pneumonia]:
            logger.debug(key='debug', data='PkgCode1 is available, so use it.')
            self.sim.modules['Alri'].do_alri_treatment(
                person_id=person_id,
                prob_of_cure=self.sim.modules['Alri'].parameters[
                    'prob_of_cure_for_uncomplicated_pneumonia_given_IMCI_pneumonia_treatment']
            )
            care_management_info.update({
                'treatment_plan': 'treatment_for_pneumonia'})
        else:
            logger.debug(key='debug', data="PkgCode1 is not available, so can't use it.")
            # todo: probability of referral if no drug available
            care_management_info.update({
                'treatment_plan': 'no_available_treatment', 'referral_to_level_2': True})

        self.module.child_disease_management_information.update({person_id: care_management_info})

        # todo: If coughing for more than 2 weeks or if having recurrent wheezing, assess for TB or asthma

        # todo: follow-up in 2 days

        # check to see if all consumables returned (for demonstration purposes):
        # check to see if all consumables returned (for demonstration purposes):
        all_available = (outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_pneumonia]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code1]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code2]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code3]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code4]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code5]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code6])
        # use helper function instead (for demonstration purposes)
        all_available_using_helper_function = self.get_all_consumables(
            item_codes=[item_code1, item_code2, item_code3, item_code4, item_code5, item_code6],
            pkg_codes=[pkg_code_pneumonia]
        )
        # Demonstrate equivalence
        assert all_available == all_available_using_helper_function

        # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        actual_appt_footprint['Under5OPD'] = actual_appt_footprint['Under5OPD'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='debug', data='HSI_IMCI_Pneumonia_Treatment_level_2: did not run')
        pass

        # todo: follow up after 3 days


class HSI_IMCI_Severe_Pneumonia_Treatment_level_2(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Alri)

        # Define the necessary information for an HSI - interventions for severe pneumonia at facility level 2
        # (These are blank when created; but these should be filled-in by the module that calls it)
        self.TREATMENT_ID = 'HSI_IMCI_Severe_Pneumonia_Treatment_level_2'

        # APP_FOOTPRINT: hospital level event takes small amount of time for health workers
        appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        appt_footprint['IPAdmission'] = 1  # This requires one out patient
        # Demonstrate the equivalence with:
        assert appt_footprint == self.make_appt_footprint({'IPAdmission': 1})

        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'IPAdmission': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1  # Can occur at facility-level 1 / health centres
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return
        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~

        care_management_info = dict()
        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # whole package of interventions
        pkg_code_severe_pneumonia = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Treatment of severe pneumonia',
                            'Intervention_Pkg_Code'])[0]

        # individual items
        item_code1 = pd.unique(consumables.loc[consumables['Items'] == 'Benzylpenicillin 3g (5MU), PFR_each_CMST',
                                               'Item_Code'])[0]
        item_code2 = pd.unique(
            consumables.loc[consumables['Items'] == 'Gentamicin Sulphate 40mg/ml, 2ml_each_CMST', 'Item_Code'])[0]
        item_code3 = pd.unique(
            consumables.loc[consumables['Items'] == 'Ceftriaxone 1g, PFR_each_CMST', 'Item_Code'])[0]
        item_code4 = pd.unique(
            consumables.loc[consumables['Items'] == 'Tube, nasogastric CH 8_each_CMST', 'Item_Code'])[0]
        item_code5 = pd.unique(
            consumables.loc[consumables['Items'] == 'Oxygen, 1000 liters, primarily with oxygen concentrators',
                            'Item_Code'])[0]
        item_code6 = pd.unique(
            consumables.loc[consumables['Items'] == 'Prednisolone 5mg_100_CMST', 'Item_Code'])[0]
        item_code7 = pd.unique(
            consumables.loc[consumables['Items'] == 'Salbutamol, syrup, 2 mg/5 mlT', 'Item_Code'])[0]
        item_code8 = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item_code9 = pd.unique(
            consumables.loc[consumables['Items'] == 'Amoxycillin 250mg_1000_CMST', 'Item_Code'])[0]
        item_code10 = pd.unique(
            consumables.loc[consumables['Items'] == 'Cannula iv  (winged with injection pot) 16_each_CMST',
                            'Item_Code'])[0]
        item_code11 = pd.unique(
            consumables.loc[consumables['Items'] == 'X-ray', 'Item_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code_severe_pneumonia: 1},
                              'Item_Code': {item_code1: 1, item_code2: 1, item_code3: 1,
                                            item_code4: 1, item_code5: 1, item_code6: 1,
                                            item_code7: 1, item_code8: 1, item_code9: 1,
                                            item_code10: 1, item_code11: 1}}

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_severe_pneumonia]:
            logger.debug(key='debug', data='PkgCode1 is available, so use it.')
            self.sim.modules['Alri'].do_alri_treatment(
                person_id=person_id,
                prob_of_cure=self.sim.modules['Alri'].parameters[
                    'prob_of_cure_for_pneumonia_with_severe_complication_given_IMCI_severe_pneumonia_treatment']
            )
            care_management_info.update({
                'treatment_plan': 'treatment_for_severe_pneumonia'})
        else:
            logger.debug(key='debug', data="PkgCode1 is not available, so can't use it.")
            # todo: probability of referral if no drug available
            # self.sim.modules['DxAlgorithmChild'].do_when_facility_level_3(person_id=person_id, hsi_event=self)
            # todo: inpatient bed days
            care_management_info.update({
                'treatment_plan': 'no_available_treatment', 'referral_to_level_2': True})

        # check to see if all consumables returned (for demonstration purposes):
        all_available = \
            (outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_severe_pneumonia]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code1]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code2]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code3]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code4]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code5]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code6]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code7]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code8]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code9]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code10]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code11])

        # use helper function instead (for demonstration purposes)
        all_available_using_helper_function = self.get_all_consumables(
            item_codes=[item_code1, item_code2, item_code3, item_code4, item_code5, item_code6,
                        item_code7, item_code8, item_code9, item_code10, item_code11],
            pkg_codes=[pkg_code_severe_pneumonia]
        )
        # Demonstrate equivalence
        assert all_available == all_available_using_helper_function

        # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        actual_appt_footprint['IPAdmission'] = actual_appt_footprint['IPAdmission'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='debug', data='HSI_IMCI_Severe_Pneumonia_Treatment_level_2: did not run')
        pass
"""
