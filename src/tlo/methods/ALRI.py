"""
Childhood Acute Lower Respiratory Infection Module
Documentation:
04 - Methods Repository/ALRI module - Description.docx
04 - Methods Repository/ResourceFile_ALRI2.xlsx

Overview
--------
Individuals are exposed to the risk of onset of ALRI. They can have viral pneumonia, bacterial pneumonia or
bronchiolitis caused by one primary agent at a time,
which can also have a co-infection or secondary bacterial infection.
During an episode (prior to recovery - either natural or cured), the symptom of cough or difficult breathing is present
in addition to other possible symptoms. ALRI may cause associated complications, such as,
pleural effusuion, empyema, lung abscess, pneumothorax, including severe complications,
such as, sepsis, meningitis and respiratory failure, leading to multi-organ dysfunction and death.
The individual may recover naturally or die.

Health care seeking is prompted by the onset of the symptom cough or respiratory symptoms.
The individual can be treated; if successful the risk of death is lowered
and they are cured (symptom resolved) some days later.

Outstanding issues
------------------
* Follow-up appointments for initial HSI events.
* Double check parameters and consumables codes for the HSI events.

"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from tlo import Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography
from tlo.methods.symptommanager import Symptom
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

class ALRI(Module):
    # Declare the pathogens that this module will simulate:
    pathogens = {
        'RSV',
        'Rhinovirus',
        'HMPV',
        'Parainfluenza',
        'Strep_pneumoniae_PCV13',
        'Strep_pneumoniae_non_PCV13',
        'Hib',
        'H.influenzae_non_type_b',
        'Staph_aureus',
        'Enterobacteriaceae',  # includes E. coli, Enterobacter species, and Klebsiella species,
        'other_Strepto_Enterococci',  # includes Streptococcus pyogenes and Enterococcus faecium
        'Influenza',
        'P.jirovecii',
        'Bocavirus',
        'Adenovirus',
        'other_viral_pathogens',  # Coronaviruses NL63, 229E OC43 and HKU1, Cytomegalovirus, Parechovirus/Enterovirus
        'other_bacterial_pathogens'  # includes Bordetella pertussis, Chlamydophila pneumoniae, Legionella species,
        # Mycoplasma pneumoniae, Moraxella catarrhalis,
        # Non-fermenting gram-negative rods (Acinetobacter species and Pseudomonas species), Neisseria meningitidis
    }

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,  # Disease modules: Any disease module should carry this label.
        Metadata.USES_SYMPTOMMANAGER,  # The 'Symptom Manager' recognises modules with this label.
        Metadata.USES_HEALTHSYSTEM,  # The 'HealthSystem' recognises modules with this label.
        Metadata.USES_HEALTHBURDEN  # The 'HealthBurden' module recognises modules with this label.
    }

    # Declare the ALRI complications:
    complications = {'pneumothorax', 'pleural_effusion', 'empyema', 'lung_abscess',
                     'sepsis', 'meningitis', 'respiratory_failure'}

    # Declare the pathogen types + pathogens:
    viral_patho = {'RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', 'Influenza', 'Adenovirus', 'Bocavirus',
                   'other_viral_pathogens'}

    bacterial_patho = {'Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', 'Hib', 'H.influenzae_non_type_b',
                       'Staph_aureus', 'Enterobacteriaceae', 'other_Strepto_Enterococci', 'other_bacterial_pathogens'}

    fungal_patho = {'P.jirovecii'}

    # Declare the disease types:
    disease_type = {
        'bacterial_pneumonia', 'viral_pneumonia', 'fungal_pneumonia', 'bronchiolitis'
    }

    # Declare the complication types:
    lung_complications = ['pleural_effusion', 'empyema', 'pneumothorax', 'lung_abscess']
    severe_complications = ['sepsis', 'meningitis', 'respiratory_failure']

    PARAMETERS = {
        # Incidence rate by pathogens  -----
        'base_inc_rate_ALRI_by_RSV':
            Parameter(Types.LIST,
                      'baseline incidence rate of ALRI caused by RSV in age groups 0-11, 12-23, 24-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_Rhinovirus':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Rhinovirus in age groups 0-11, 12-23, 24-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_HMPV':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by HMPV in age groups 0-11, 12-23, 24-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_Parainfluenza':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Parainfluenza 1-4 in age groups 0-11, 12-23, 24-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_Strep_pneumoniae_PCV13':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Streptoccocus pneumoniae PCV13 serotype '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_Strep_pneumoniae_non_PCV13':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Streptoccocus pneumoniae non PCV13 serotype '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_Hib':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Haemophilus influenzae type b '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_H.influenzae_non_type_b':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Haemophilus influenzae non-type b '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_Enterobacteriaceae':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Enterobacteriaceae in age groups 0-11, 12-23, 24-59 months,'
                      ' per child per year'
                      ),
        'base_inc_rate_ALRI_by_other_Strepto_Enterococci':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by other streptococci and Enterococci including '
                      'Streptococcus pyogenes and Enterococcus faecium in age groups 0-11, 12-23, 24-59 months,'
                      ' per child per year'
                      ),
        'base_inc_rate_ALRI_by_Staph_aureus':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Staphylococcus aureus '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_Influenza':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Influenza type A, B, and C '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_P.jirovecii':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by P. jirovecii in age groups 0-11, 12-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_Adenovirus':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by adenovirus in age groups 0-11, 12-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_Bocavirus':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by bocavirus in age groups 0-11, 12-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_other_viral_pathogens':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by other viral pathogens in age groups 0-11, 12-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_other_bacterial_pathogens':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by other viral pathogens in age groups 0-11, 12-59 months, '
                      'per child per year'
                      ),

        # Risk factors parameters -----
        'rr_ALRI_HHhandwashing':
            Parameter(Types.REAL,
                      'relative rate of acquiring ALRI for children with household handwashing with soap '
                      ),
        'rr_ALRI_HIV_untreated':
            Parameter(Types.REAL,
                      'relative rate of acquiring ALRI for children with untreated HIV positive status'
                      ),
        'rr_ALRI_underweight':
            Parameter(Types.REAL,
                      'relative rate of acquiring ALRI for underweight children'
                      ),
        'rr_ALRI_low_birth_weight':
            Parameter(Types.REAL,
                      'relative rate of acquiring ALRI for infants with low birth weight'
                      ),
        'rr_ALRI_not_excl_breastfeeding':
            Parameter(Types.REAL,
                      'relative rate of acquiring ALRI for not exclusive breastfeeding upto 6 months'
                      ),
        'rr_ALRI_indoor_air_pollution':
            Parameter(Types.REAL,
                      'relative rate of acquiring ALRI for indoor air pollution'
                      ),
        # 'rr_ALRI_pneumococcal_vaccine': Parameter
        # (Types.REAL, 'relative rate of acquiring ALRI for pneumonococcal vaccine'
        #  ),
        # 'rr_ALRI_haemophilus_vaccine': Parameter
        # (Types.REAL, 'relative rate of acquiring ALRI for haemophilus vaccine'
        #  ),
        # 'rr_ALRI_influenza_vaccine': Parameter
        # (Types.REAL, 'relative rate of acquiring ALRI for influenza vaccine'
        #  ),
        # 'r_progress_to_severe_ALRI': Parameter
        # (Types.LIST,
        #  'probability of progressing from non-severe to severe ALRI by age category '
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
        'prob_respiratory_failure_by_viral_pneumonia':
            Parameter(Types.REAL,
                      'probability of respiratory failure caused by primary viral pneumonia'
                      ),
        'prob_respiratory_failure_by_bacterial_pneumonia':
            Parameter(Types.REAL,
                      'probability of respiratory failure caused by primary or secondary bacterial pneumonia'
                      ),
        'prob_respiratory_failure_by_bronchiolitis':
            Parameter(Types.REAL,
                      'probability of respiratory failure caused by viral bronchiolitis'
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
        # 'prob_sepsis_to_septic_shock': Parameter
        # (Types.REAL, 'probability of sepsis causing septic shock'
        #  ),
        # 'prob_septic_shock_to_multiorgan_dysfunction': Parameter
        # (Types.REAL, 'probability of septic shock causing multi-organ dysfunction'
        #  ),
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
        'r_death_from_ALRI_due_to_sepsis':
            Parameter(Types.REAL,
                      'death rate from ALRI due to sepsis, base age 0-11 months'
                      ),
        'r_death_from_ALRI_due_to_respiratory_failure':
            Parameter(Types.REAL,
                      'death rate from ALRI due to respiratory failure, base age 0-11 months'
                      ),
        'r_death_from_ALRI_due_to_meningitis':
            Parameter(Types.REAL,
                      'death rate from ALRI due to meningitis, base age 0-11 months'
                      ),
        'rr_death_ALRI_age12to23mo':
            Parameter(Types.REAL,
                      'death rate of ALRI for children aged 12 to 23 months'
                      ),
        'rr_death_ALRI_age24to59mo':
            Parameter(Types.REAL,
                      'death rate of ALRI for children aged 24 to 59 months'
                      ),
        'rr_death_ALRI_HIV':
            Parameter(Types.REAL,
                      'death rate of ALRI for children with HIV not on ART'
                      ),
        'rr_death_ALRI_SAM':
            Parameter(Types.REAL,
                      'death rate of ALRI for children with severe acute malnutrition'
                      ),
        'rr_death_ALRI_low_birth_weight':
            Parameter(Types.REAL,
                      'death rate of ALRI for children with low birth weight (applicable to infants)'
                      ),

        # Proportions of what disease type (viral ALRI) -----
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
                      'relative rate of ALRI with the PCV13'
                      ),
        'rr_ALRI_hib_vaccine':
            Parameter(Types.REAL,
                      'relative rate of ALRI with the hib vaccination'
                      ),
        'rr_ALRI_RSV_vaccine':
            Parameter(Types.REAL,
                      'relative rate of ALRI with the RSV vaccination'
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

    }

    PROPERTIES = {
        # ---- ALRI status ----
        'ri_current_ALRI_status':
            Property(Types.BOOL,
                     'ALRI status (current or last episode)'
                     ),
        # ---- The pathogen which is the attributed cause of ALRI ----
        'ri_primary_ALRI_pathogen':
            Property(Types.CATEGORICAL,
                     'Attributable pathogen for the current ALRI event',
                     categories=list(pathogens) + ['not_applicable']
                     ),
        # ---- The bacterial pathogen which is the attributed co-/secondary infection ----
        'ri_secondary_bacterial_pathogen':
            Property(Types.CATEGORICAL,
                     'Secondary bacterial pathogen for the current ALRI event',
                     categories=list(bacterial_patho) + ['none'] + ['not_applicable']
                     ),
        # ---- The underlying ALRI condition ----
        'ri_ALRI_disease_type':
            Property(Types.CATEGORICAL, 'underlying ALRI condition',
                     categories=['viral_pneumonia', 'bacterial_pneumonia', 'fungal_pneumonia',
                                 'bronchiolitis'] + ['not_applicable']
                     ),
        # ---- Complications associated with ALRI ----
        'ri_ALRI_complications':
            Property(Types.LIST,
                     'complications that arose from the current ALRI event',
                     categories=['pneumothorax', 'pleural_effusion', 'empyema',
                                 'lung_abscess', 'sepsis', 'meningitis',
                                 'respiratory_failure'] + ['none'] + ['not_applicable']
                     ),
        # ---- Symptoms associated with ALRI ----
        'ri_current_ALRI_symptoms':
            Property(Types.LIST,
                     'symptoms of current ALRI event',
                     categories=[
                         'fever', 'cough', 'difficult_breathing', 'convulsions', 'lethargy',
                         'fast_breathing', 'chest_indrawing', 'grunting', 'chest_pain',
                         'cyanosis', 'respiratory_distress', 'hypoxia', 'fast_breathing',
                         'danger_signs'
                     ]
                     ),

        # ---- Internal variables to schedule onset and deaths due to ALRI ----
        'ri_ALRI_event_date_of_onset': Property(Types.DATE, 'date of onset of current ALRI event'),
        'ri_ALRI_event_recovered_date': Property(Types.DATE, 'date of recovery from current ALRI event'),
        'ri_ALRI_event_death_date': Property(Types.DATE, 'date of death caused by current ALRI event'),
        'ri_ALRI_pulmonary_complication_date': Property(Types.DATE, 'date of progression to a pulmonary complication'),
        'ri_ALRI_severe_complication_date': Property(Types.DATE, 'date of progression to a severe complication'),
        'ri_end_of_last_alri_episode':
            Property(Types.DATE, 'date on which the last episode of ALRI is resolved, (including '
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
        'ri_ALRI_tx_start_date': Property(Types.DATE, 'start date of ALRI treatment for current event'),

        'ri_chest_auscultations_signs':
            Property(Types.CATEGORICAL, 'findings during chest auscultation examination',
                     categories=['decreased_breath_sounds', 'bronchial_breaths_sounds', 'crackles', 'wheeze',
                                 'abnormal_vocal_resonance', 'pleural_rub'] + ['none'] + ['not_applicable']
                     ),

        'ri_ALRI_antibiotic_treatment_administered':
            Property(Types.CATEGORICAL,
                     'Antibiotic treatment given for the ALRI',
                     categories=['benzyl penicillin injection', 'amoxicillin', 'cotrimoxazole', 'other_antibiotic',
                                 'chlorampheniciol', 'prednisolone'] + ['none'] + ['not_applicable']),
        'ri_peripheral_oxygen_saturation':
            Property(Types.CATEGORICAL,
                     'Level of peripheral oxygen saturation to be read by a pulse oximetry',
                     categories=['SpO2<90%', 'SpO2_90-92%', 'SpO2>92%']),
        'ri_oxygen_therapy_given':
            Property(Types.BOOL,
                     'Oxygen therapy received at the hospital (for severe cases)'),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # equations for the incidence of ALRI by pathogen:
        self.incidence_equations_by_pathogen\
            = dict()

        # equations for the proportions of ALRI diseases:
        self.proportions_of_ALRI_disease_types_by_pathogen = dict()

        # equations for the probabilities of secondary bacterial superinfection:
        self.prob_secondary_bacterial_infection = None

        # equations for the development of ALRI-associated complications:
        self.risk_of_developing_ALRI_complications = dict()
        self.risk_of_progressing_to_severe_complications = dict()

        # Linear Model for predicting the risk of death:
        self.risk_of_death_severe_ALRI = dict()

        # dict to hold the probability of onset of different types of symptom given underlying complications:
        self.prob_symptoms_uncomplicated_ALRI = dict()
        self.prob_extra_symptoms_complications = dict()

        # dict to to store the information regarding HSI management of disease:
        self.child_disease_management_information = dict()

        # dict to hold the DALY weights
        self.daly_wts = dict()

        # dict to hold counters for the number of ALRI events by pathogen and age-group
        # (0yrs, 1yrs, 2-4yrs)
        blank_counter = dict(zip(self.pathogens, [list() for _ in self.pathogens]))

        self.incident_case_tracker_blank = {
            '0y': copy.deepcopy(blank_counter),
            '1y': copy.deepcopy(blank_counter),
            '2-4y': copy.deepcopy(blank_counter),
            '5+y': copy.deepcopy(blank_counter)
        }
        self.incident_case_tracker = copy.deepcopy(self.incident_case_tracker_blank)

        zeros_counter = dict(zip(self.pathogens, [0] * len(self.pathogens)))
        self.incident_case_tracker_zeros = {
            '0y': copy.deepcopy(zeros_counter),
            '1y': copy.deepcopy(zeros_counter),
            '2-4y': copy.deepcopy(zeros_counter),
            '5+y': copy.deepcopy(zeros_counter)
        }

        # # Store the symptoms that this module will use:
        # self.symptoms = {'grunting', 'cyanosis', 'hypoxia', 'danger_signs', 'stridor'}

        # Store the symptoms that this module will use:
        self.symptoms = {
            'fever', 'cough', 'difficult_breathing', 'fast_breathing', 'chest_indrawing', 'chest_pain',
            'cyanosis', 'respiratory_distress', 'danger_signs'
        }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
        p = self.parameters
        self.load_parameters_from_dataframe(
            pd.read_excel(
                Path(self.resourcefilepath) / 'ResourceFile_ALRI2.xlsx', sheet_name='Parameter_values'))

        # Check that every value has been read-in successfully
        for param_name, param_type in self.PARAMETERS.items():
            assert param_name in self.parameters, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert isinstance(self.parameters[param_name],
                              param_type.python_type), f'Parameter "{param_name}" is not read in correctly from the resourcefile.'

        # Register this disease module with the health system
        # self.sim.modules['HealthSystem'].register_disease_module(self)

        # Declare symptoms that this modules will cause and which are not included in the generic symptoms:
        generic_symptoms = self.sim.modules['SymptomManager'].parameters['generic_symptoms']
        for symptom_name in self.symptoms:
            if symptom_name not in generic_symptoms:
                self.sim.modules['SymptomManager'].register_symptom(
                    Symptom(name=symptom_name)  # (give non-generic symptom 'average' healthcare seeking)
                )

    def initialise_population(self, population):
        """Set our property values for the initial population.
        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        :param population: the population of individuals

        Sets that there is no one with ALRI at initiation.
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals

        # ---- Key Current Status Classification Properties ----
        df.loc[df.is_alive, 'ri_ALRI_status'] = False
        df.loc[df.is_alive, 'ri_primary_ALRI_pathogen'].values[:] = 'not_applicable'
        df.loc[df.is_alive, 'ri_current_ALRI_symptoms'] = 'not_applicable'
        df.loc[df.is_alive, 'ri_secondary_bacterial_pathogen'] = 'not_applicable'
        df.loc[df.is_alive, 'ri_ALRI_disease_type'] = 'not_applicable'
        df.loc[df.is_alive, 'ri_ALRI_complications'] = 'not_applicable'

        # ---- Internal values ----
        df.loc[df.is_alive, 'ri_ALRI_event_date_of_onset'] = pd.NaT
        df.loc[df.is_alive, 'ri_ALRI_event_recovered_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_ALRI_event_death_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_ALRI_pulmonary_complication_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_ALRI_severe_complication_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_end_of_last_alri_episode'] = pd.NaT

        df.loc[df.is_alive, 'ri_ALRI_treatment'] = False
        df.loc[df.is_alive, 'ri_ALRI_tx_start_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_chest_auscultations_signs'] = 'not_applicable'
        df.loc[df.is_alive, 'ri_ALRI_antibiotic_treatment_administered'] = 'not_applicable'
        df.loc[df.is_alive, 'ri_peripheral_oxygen_saturation'] = 'SpO2>92%'
        df.loc[df.is_alive, 'ri_oxygen_therapy_given'] = False

        # ---- Temporary values ----
        df.loc[df.is_alive, 'tmp_malnutrition'] = False
        df.loc[df.is_alive, 'tmp_hv_inf'] = False
        df.loc[df.is_alive, 'tmp_low_birth_weight'] = False
        df.loc[df.is_alive, 'tmp_exclusive_breastfeeding'] = False
        df.loc[df.is_alive, 'tmp_continued_breastfeeding'] = False

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.

        Prepares for simulation:
        * Schedules the main polling event
        * Schedules the main logging event
        * Establishes the linear models and other data structures using the parameters that have been read-in
        """

        # Schedule the main polling event (to first occur immidiately)
        sim.schedule_event(AcuteLowerRespiratoryInfectionPollingEvent(self), sim.date + DateOffset(days=0))

        # Schedule the main logging event (to first occur in one year)
        sim.schedule_event(AcuteLowerRespiratoryInfectionLoggingEvent(self), sim.date + DateOffset(days=0))

        # Get DALY weights
        # get_daly_weight = self.sim.modules['HealthBurden'].get_daly_weight
        if 'HealthBurden' in self.sim.modules.keys():
            self.daly_wts['daly_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
            self.daly_wts['daly_severe_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
            self.daly_wts['daly_very_severe_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=46)

        # =====================================================================================================
        # APPLY A LINEAR MODEL FOR THE ACQUISITION OF A PRIMARY PATHOGEN FOR ALRI
        # --------------------------------------------------------------------------------------------
        # Make a dict to hold the equations that govern the probability that a person acquires ALRI
        # that is caused (primarily) by a pathogen
        p = self.parameters

        def make_scaled_linear_model(patho):
            """Makes the unscaled linear model with default intercept of 1. Calculates the mean incidents rate for
            0-year-olds and then creates a new linear model with adjusted intercept so incidents in 0-year-olds
            matches the specified value in the model when averaged across the population
            """

            def make_linear_model(patho, intercept=1.0):
                base_inc_rate = f'base_inc_rate_ALRI_by_{patho}'
                return LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept,
                    Predictor('age_years')
                        .when('.between(0,0)', p[base_inc_rate][0])
                        .when('.between(1,1)', p[base_inc_rate][1])
                        .when('.between(2,4)', p[base_inc_rate][2])
                        .otherwise(0.0),
                    Predictor('li_no_access_handwashing').when(False, p['rr_ALRI_HHhandwashing']),
                    Predictor('li_wood_burn_stove').when(False, p['rr_ALRI_indoor_air_pollution']),
                    Predictor('tmp_hv_inf').when(True, p['rr_ALRI_HIV_untreated']),
                    # Predictor().when(
                    #     "va_pneumo == '>1' & "
                    #     "(ri_primary_ALRI_pathogen | ri_secondary_bacterial_pathogen == 'streptococcus'",
                    #     p['rr_ALRI_PCV13']),
                    Predictor('tmp_malnutrition').when(True, p['rr_ALRI_underweight']),
                    Predictor('tmp_exclusive_breastfeeding').when(False, p['rr_ALRI_not_excl_breastfeeding'])
                )

            df = self.sim.population.props
            unscaled_lm = make_linear_model(patho)
            target_mean = p[f'base_inc_rate_ALRI_by_{patho}'][0]
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 0)]).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean)
            scaled_lm = make_linear_model(patho, intercept=scaled_intercept)
            # check by applying the model to mean incidence of 0-year-olds
            assert (target_mean - scaled_lm.predict(df.loc[df.is_alive & (df.age_years == 0)]).mean()) < 1e-10
            return scaled_lm

        for pathogen in ALRI.pathogens:
            self.incidence_equations_by_pathogen[pathogen] = make_scaled_linear_model(pathogen)

        # check that equations have been declared for each pathogens
        assert self.pathogens == set(list(self.incidence_equations_by_pathogen.keys()))

        # --------------------------------------------------------------------------------------------
        # Linear models for determining the underlying condition as viral or bacterial pneumonia, and bronchiolitis
        # caused by each primary pathogen
        def determine_ALRI_type(patho):
            df = self.sim.population.props
            if patho in ALRI.bacterial_patho:
                return 'bacterial_pneumonia'
            if patho in ALRI.viral_patho:
                under_2_yo = df['is_alive'] & (df['age_years'] < 2)
                for child in under_2_yo:  # bronchiolitis only for those under 2
                    return 'viral_pneumonia' if self.rng.rand() < p[f'proportion_viral_pneumonia_by_{patho}'] else 'bronchiolitis'
                else:
                    return 'viral_pneumonia'
            if patho in ALRI.fungal_patho:
                return 'fungal_pneumonia'

        for pathogen in ALRI.pathogens:
            self.proportions_of_ALRI_disease_types_by_pathogen[pathogen] = determine_ALRI_type(pathogen)

        # check that equations have been declared for each pathogens
        assert self.pathogens == set(list(self.proportions_of_ALRI_disease_types_by_pathogen.keys()))

        # =====================================================================================================
        # APPLY PROBABILITY OF CO- / SECONDARY BACTERIAL INFECTION
        # -----------------------------------------------------------------------------------------------------
        # Create a linear model equation for the probability of a secondary bacterial superinfection
        self.prob_secondary_bacterial_infection = \
            LinearModel(LinearModelType.MULTIPLICATIVE,
                        1.0,
                        Predictor()
                        .when(
                            "ri_primary_ALRI_pathogen.isin(['RSV', 'rhinovirus', 'hMPV', "
                            "'parainfluenza', 'influenza']) & "
                            "ri_ALRI_disease_type == 'viral_pneumonia'"
                            "& ri_secondary_bacterial_pathogen =='not_applicable'",
                            p['prob_viral_pneumonia_bacterial_coinfection']),
                        Predictor()
                        .when(
                            "ri_primary_ALRI_pathogen.isin(['RSV', 'rhinovirus', 'hMPV', "
                            "'parainfluenza', 'influenza']) & "
                            "ri_ALRI_disease_type == 'bronchiolitis' "
                            " & ri_secondary_bacterial_pathogen =='not_applicable'",
                            p['prob_secondary_bacterial_infection_in_bronchiolitis'])
                        )

        # =====================================================================================================
        # APPLY LINEAR MODEL TO DETERMINE PROBABILITY OF COMPLICATIONS
        # -----------------------------------------------------------------------------------------------------
        # Create linear models for the risk of acquiring complications from uncomplicated ALRI
        self.risk_of_developing_ALRI_complications.update({
            'pneumothorax':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen' or 'ri_secondary_bacterial_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_pneumothorax_by_bacterial_pneumonia']),
                            Predictor()
                            .when(
                                "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "ri_ALRI_disease_type == 'viral_pneumonia' ",
                                p['prob_pneumothorax_by_viral_pneumonia']),
                            Predictor()
                            .when(
                                "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "ri_ALRI_disease_type == 'bronchiolitis' ",
                                p['prob_pneumothorax_by_bronchiolitis'])
                            ),

            'pleural_effusion':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen' or 'ri_secondary_bacterial_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_pleural_effusion_by_bacterial_pneumonia']),
                            Predictor()
                            .when(
                                "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "ri_ALRI_disease_type == 'viral_pneumonia' ",
                                p['prob_pleural_effusion_by_viral_pneumonia']),
                            Predictor()
                            .when(
                                "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "ri_ALRI_disease_type == 'bronchiolitis' ",
                                p['prob_pleural_effusion_by_bronchiolitis'])
                            ),

            'empyema':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_ALRI_complications').apply(
                                lambda x: p['prob_pleural_effusion_to_empyema'] if 'pleural_effusion' in x else 0)
                            ),  # TODO: This doesn't work here

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
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen' or 'ri_secondary_bacterial_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_sepsis_by_bacterial_pneumonia']),
                            Predictor()
                            .when(
                                "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "ri_ALRI_disease_type == 'viral_pneumonia' ",
                                p['prob_sepsis_by_viral_pneumonia']),
                            Predictor()
                            .when(
                                "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "ri_ALRI_disease_type == 'bronchiolitis' ",
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
                            Predictor('ri_primary_ALRI_pathogen' or 'ri_secondary_bacterial_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_respiratory_failure_by_bacterial_pneumonia']),
                            Predictor()
                            .when(
                                "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "ri_ALRI_disease_type == 'viral_pneumonia' ",
                                p['prob_respiratory_failure_by_viral_pneumonia']),
                            Predictor()
                            .when(
                                "ri_primary_ALRI_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "ri_ALRI_disease_type == 'bronchiolitis' ",
                                p['prob_respiratory_failure_by_bronchiolitis']),
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
                                lambda x: p['prob_pneumothorax_to_respiratory_failure'] if 'pneumothorax' in x else 0)
                            ),
            'sepsis':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_ALRI_complications').apply(
                                lambda x: p['prob_lung_abscess_to_sepsis'] if 'lung_abscess' in x else 0),
                            Predictor('ri_ALRI_complications').apply(
                                lambda x: p['prob_empyema_to_sepsis'] if 'empyema' in x else 0),
                            )
        })

        # =====================================================================================================
        # APPLY PROBABILITY OF SYMPTOMS TO EACH ALRI DISEASE TYPE (UNCOMPLICATED AND WITH COMPLICATIONS)
        # -----------------------------------------------------------------------------------------------------
        # Make a dict containing the probability of symptoms given acquisition of (uncomplicated) ALRI,
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

        for disease in ALRI.disease_type:
            self.prob_symptoms_uncomplicated_ALRI[disease] = make_symptom_probs(disease)

        # Check that each ALRI type has a risk of developing each symptom
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

        for complication in ALRI.complications:
            self.prob_extra_symptoms_complications[complication] = add_complication_symptom_probs(complication)

        # Check that each complication has a risk of developing each symptom
        assert self.complications == set(list(self.prob_extra_symptoms_complications.keys()))

        # =====================================================================================================
        # APPLY A LINEAR MODEL FOR THE RISK OF DEATH DUE TO ALRI (SEVERE COMPLICATIONS)
        # -----------------------------------------------------------------------------------------------------
        # Create a linear model for the risk of dying due to complications: sepsis, meningitis, respiratory failure
        # TODO: change this - this is wrong maths
        def death_risk(complications_list):
            total = 0
            if 'sepsis' in complications_list:
                total += p['r_death_from_ALRI_due_to_sepsis']
            if 'respiratory_failure' in complications_list:
                total += p['r_death_from_ALRI_due_to_respiratory_failure']
            if 'meningitis' in complications_list:
                total += p['r_death_from_ALRI_due_to_meningitis']
            return total

        self.risk_of_death_severe_ALRI = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1.0,
            Predictor('ri_ALRI_complications').apply(death_risk),
            Predictor('tmp_hv_inf').when(True, p['rr_death_ALRI_HIV']),
            Predictor('tmp_malnutrition').when(True, p['rr_death_ALRI_SAM']),
            Predictor('tmp_low_birth_weight').when(True, p['rr_death_ALRI_low_birth_weight']),
            Predictor('age_years')
                .when('.between(1,1)', p['rr_death_ALRI_age12to23mo'])
                .when('.between(2,4)', p['rr_death_ALRI_age24to59mo']),
            # Predictor()
            #     .when("ri_ALRI_complications == 'respiratory_failure' & "
            #           "ri_ALRI_disease_type.isin(['bronchiolitis', 'viral_pneumonia']) & "
            #           "ri_oxygen_therapy_given == 'True'", p['prob_respiratory_failure_by_bronchiolitis']),
        )
        # -----------------------------------------------------------------------------------------------------

        # Look-up and store the consumables that are required for each HSI
        # self.look_up_consumables()

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        # ---- Key Current Status Classification Properties ----
        df.at[child_id, 'ri_ALRI_status'] = False
        df.at[child_id, 'ri_primary_ALRI_pathogen'] = 'not_applicable'
        df.at[child_id, 'ri_current_ALRI_symptoms'] = 'not_applicable'
        df.at[child_id, 'ri_secondary_bacterial_pathogen'] = 'not_applicable'
        df.at[child_id, 'ri_ALRI_disease_type'] = 'not_applicable'
        df.at[child_id, 'ri_ALRI_complications'] = 'not_applicable'

        # ---- Internal values ----
        df.at[child_id, 'ri_ALRI_event_date_of_onset'] = pd.NaT
        df.at[child_id, 'ri_ALRI_event_recovered_date'] = pd.NaT
        df.at[child_id, 'ri_ALRI_event_recovered_date'] = pd.NaT
        df.at[child_id, 'ri_ALRI_event_death_date'] = pd.NaT
        df.at[child_id, 'ri_ALRI_pulmonary_complication_date'] = pd.NaT
        df.at[child_id, 'ri_ALRI_severe_complication_date'] = pd.NaT
        df.at[child_id, 'ri_ALRI_pulmonary_complication_date'] = pd.NaT
        df.at[child_id, 'ri_end_of_last_alri_episode'] = pd.NaT

        # ---- Temporary values ----
        df.at[child_id, 'tmp_malnutrition'] = False
        df.at[child_id, 'tmp_hv_inf'] = False
        df.at[child_id, 'tmp_low_birth_weight'] = False
        df.at[child_id, 'tmp_exclusive_breastfeeding'] = False
        df.at[child_id, 'tmp_continued_breastfeeding'] = False

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        pass

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        pass

        df = self.sim.population.props
        who_has_symptoms = self.sim.modules['SymptomManager'].who_has

        total_daly_values = pd.Series(data=0.0, index=df.index[df.is_alive])

        total_daly_values.loc[
            self.sim.modules['SymptomManager'].who_has('fast_breathing')] = self.daly_wts['daly_ALRI']

        total_daly_values.loc[
            self.sim.modules['SymptomManager'].who_has('danger_signs')] = self.daly_wts['daly_severe_ALRI']

        # Split out by pathogen that causes the ALRI
        dummies_for_pathogen = pd.get_dummies(df.loc[total_daly_values.index,
                                                     'ri_primary_ALRI_pathogen'],
                                              dtype='float').drop(columns='not_applicable')
        daly_values_by_pathogen = dummies_for_pathogen.mul(total_daly_values, axis=0)

        return daly_values_by_pathogen

    def do_treatment(self, person_id, prob_of_cure):
        """Helper function that enacts the effects of a treatment to ALRI caused by a pathogen.
        It will only do something if the ALRI is caused by a pathogen (this module). It will not allow any effect
         if the respiratory infection is caused by another module.
        * Log the treatment date
        * Prevents this episode of ALRI
         from causing a death
        * Schedules the cure event, at which symptoms are alleviated.
        """
        df = self.sim.population.props
        person = df.loc[person_id]

        if not person.is_alive:
            return

        # Do nothing if the ALRI has not been caused by a pathogen
        if not (
            (person.ri_primary_ALRI_pathogen != 'not_applicable') &
            (person.ri_ALRI_event_date_of_onset <= self.sim.date <= person.ri_end_of_last_alri_episode)
        ):
            return

        # Log that the treatment is provided:
        df.at[person_id, 'ri_ALRI_tx_start_date'] = self.sim.date

        # Determine if the treatment is effective
        if prob_of_cure > self.rng.rand():
            # If treatment is successful: cancel death and schedule cure event
            self.cancel_death_date(person_id)
            self.sim.schedule_event(ALRICureEvent(self, person_id),
                                    self.sim.date + DateOffset(
                                        days=self.parameters['days_between_treatment_and_cure']
                                    ))
        # else:  # not improving seek care or death
        #     self.do_when_not_improving(person_id)

    def cancel_death_date(self, person_id):
        """
        Cancels a scheduled date of death due to ALRI for a person. This is called prior to the scheduling the
        CureEvent to prevent deaths happening in the time between a treatment being given and the cure event occurring.
        :param person_id:
        :return:
        """
        df = self.sim.population.props
        df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT

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
    #         self.sim.modules['DxAlgorithmChild'].do_when_facility_level_1(person_id=person_id, hsi_event=self) # not working line


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------
class AcuteLowerRespiratoryInfectionPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the main event that runs the acquisition of pathogens that cause ALRI.
        It determines who is infected and when and schedules individual IncidentCase events to represent onset.

        A known issue is that ALRI events are scheduled based on the risk of current age but occur a short time
        later when the children will be slightly older. This means that when comparing the model output with data, the
        model slightly under-represents incidence among younger age-groups and over-represents incidence among older
        age-groups. This is a small effect when the frequency of the polling event is high.
    """
    # TODO: how to fix this

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        # NB. The frequency of the occurrences of this event can be edited safely.

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        p = self.module.parameters
        rng = self.module.rng

        # Compute the incidence rate for each person getting ALRI and then convert into a probability
        # getting all children that do not currently have an ALRI episode (never had or last episode resolved)
        mask_could_get_new_alri_event = \
            df['is_alive'] & (df['age_years'] < 5) & \
            ((df.ri_end_of_last_alri_episode < self.sim.date) | pd.isnull(df.ri_end_of_last_alri_episode))

        # Compute the incidence rate for each person acquiring ALRI
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

            # ----------------- Determine the ALRI disease type for this case -----------------
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
            if pathogen in self.module.fungal_patho:
                bacterial_patho_in_ALRI_coinfection = 'none'

            # ----------------------- Allocate a date of onset of ALRI ----------------------
            date_onset = self.sim.date + DateOffset(days=np.random.randint(0, days_until_next_polling_event))

            # ----------------------- Duration of the ALRI event -----------------------
            duration_in_days_of_alri = max(7, int(
                14 + (-2 + 4 * rng.rand())))  # assumes uniform interval around mean duration with range 4 days

            # ----------------------- Create the event for the onset of infection -------------------
            self.sim.schedule_event(
                event=AcuteLowerRespiratoryInfectionIncidentCase(
                    module=self.module,
                    person_id=person_id,
                    pathogen=pathogen,
                    disease_type=alri_disease_type_for_this_person,
                    co_bacterial_patho=bacterial_patho_in_ALRI_coinfection,
                    duration_in_days=duration_in_days_of_alri,
                ),
                date=date_onset
            )


class AcuteLowerRespiratoryInfectionIncidentCase(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of the infection that causes ALRI.
     * Refreshes all the properties so that they pertain to this current episode of ALRI
     * Imposes the symptoms
     * Schedules relevant natural history event {(AlriWithPulmonaryComplicationsEvent) and
     (AlriWithSevereComplicationsEvent)
       (either ALRINaturalRecoveryEvent or ALRIDeathEvent)}
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
        p = self.module.parameters

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
        df.at[person_id, 'ri_ALRI_complications'] = 'none'  # all disease start as non-severe

        # assert self.disease is not None

        # ----------------------- Allocate symptoms to onset of ALRI ----------------------
        prob_symptoms_uncomplicated_alri = m.prob_symptoms_uncomplicated_ALRI[self.disease]

        symptoms_for_this_person = list()
        for symptom, prob in prob_symptoms_uncomplicated_alri.items():
            if rng.rand() < prob:
                symptoms_for_this_person.append(symptom)

        # Onset symptoms:
        for symptom in symptoms_for_this_person:
            self.module.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove='+',
                disease_module=self.module,
                duration_in_days=self.duration_in_days
            )
        df.at[person_id, 'ri_current_ALRI_symptoms'] = symptoms_for_this_person

        # date for recovery with uncomplicated ALRI
        date_of_outcome = self.module.sim.date + DateOffset(days=self.duration_in_days)

        # COMPLICATIONS ------------------------------------------------------------------------------------------
        complications_for_this_person = list()
        for complication in self.module.complications:
            prob_developing_each_complication = m.risk_of_developing_ALRI_complications[complication].predict(
                df.loc[[person_id]]).values[0]
            if rng.rand() < prob_developing_each_complication:
                complications_for_this_person.append(complication)
            if 'pleural_effusion' in complications_for_this_person:
                if self.disease == 'bacterial_pneumonia':
                    if rng.rand() < p['prob_pleural_effusion_to_empyema']:
                        complications_for_this_person.append('empyema')

        # if at least one complication developed in the ALRI event
        if len(complications_for_this_person) != 0:
            for complication in complications_for_this_person:
                if complication in self.module.lung_complications:  # schedule for pulmonary complications
                    date_onset_complications = self.module.sim.date + DateOffset(
                        days=np.random.randint(0, high=self.duration_in_days))
                    # schedule the complication event
                    self.sim.schedule_event(AlriWithPulmonaryComplicationsEvent(
                        self.module, person_id,
                        duration_in_days=self.duration_in_days,
                        symptoms=symptoms_for_this_person,
                        complication=complications_for_this_person),
                        date_onset_complications)
                if complication in self.module.severe_complications:  # schedule for severe complications
                    date_onset_complications = self.module.sim.date + DateOffset(
                        days=np.random.randint(0, high=self.duration_in_days))
                    # schedule the complication event
                    self.sim.schedule_event(AlriWithSevereComplicationsEvent(
                        self.module, person_id,
                        duration_in_days=self.duration_in_days,
                        symptoms=symptoms_for_this_person,
                        complication=complications_for_this_person),
                        date_onset_complications)

            df.at[person_id, 'ri_ALRI_event_recovered_date'] = pd.NaT
            df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT
        else:
            # if NO complications for this ALRI event, schedule a natural recovery
            df.at[person_id, 'ri_ALRI_event_recovered_date'] = date_of_outcome
            self.sim.schedule_event(ALRINaturalRecoveryEvent(self.module, person_id),
                                    date_of_outcome)
            df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT

        # Record 'episode end' data. This the date when this episode ends. It is the last possible data that any HSI
        # could affect this episode.
        df.at[person_id, 'ri_end_of_last_alri_episode'] = date_of_outcome + DateOffset(
            days=self.module.parameters['days_between_treatment_and_cure']
        )

        # Add this incident case to the tracker
        age = df.loc[person_id, ['age_years']]
        if age.values[0] < 5:
            age_grp = age.map({0: '0y', 1: '1y', 2: '2-4y', 3: '2-4y', 4: '2-4y'}).values[0]
        else:
            age_grp = '5+y'
        self.module.incident_case_tracker[age_grp][self.pathogen].append(self.sim.date)


class AlriWithPulmonaryComplicationsEvent(Event, IndividualScopeEventMixin):
    """
           This Event is for the onset of a pulmonary complication from ALRI. For some untreated children,
           this occurs a set number of days after onset of disease.
           It sets the property 'ri_ALRI_complications' to each pulmonary complication
           and schedules severe complications.
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

        # complications for this person
        df.at[person_id, 'ri_ALRI_complications'] = list(self.complication)

        # date of onset of pulmonary complication
        df.at[person_id, 'ri_ALRI_pulmonary_complication_date'] = self.sim.date

        # add to the initial list of uncomplicated ALRI symptoms
        all_symptoms_for_this_person = list(self.symptoms)  # original uncomplicated symptoms list to add to
        # keep only the probabilities for the complications of the person:
        possible_symptoms_by_complication = {key: val for key, val in
                                             self.module.prob_extra_symptoms_complications.items()
                                             if key in list(self.complication)}

        symptoms_from_complications = list()
        for complication in possible_symptoms_by_complication:
            for symptom, prob in possible_symptoms_by_complication[complication].items():
                if self.module.rng.rand() < prob:
                    symptoms_from_complications.append(symptom)
                for i in symptoms_from_complications:
                    # add symptoms from complications to the list
                    all_symptoms_for_this_person.append(
                        i) if i not in all_symptoms_for_this_person \
                        else all_symptoms_for_this_person

        df.at[person_id, 'ri_current_ALRI_symptoms'] = all_symptoms_for_this_person

        for symptom in all_symptoms_for_this_person:
            self.module.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove='+',
                disease_module=self.module,
                duration_in_days=self.duration_in_days
            )

        # Determine severe complication outcome -------------------------------------------------------------------
        date_of_recovery = df.at[person_id, 'ri_ALRI_event_date_of_onset'] + DateOffset(self.duration_in_days)
        # use the outcome date to get the number of days from onset of lung complication to outcome
        delta_date = date_of_recovery - self.sim.date
        delta_in_days = delta_date.days

        date_of_onset_severe_complication = self.sim.date + DateOffset(
            days=np.random.randint(0, high=delta_in_days))

        complications_for_this_person = list()
        for complication in ['respiratory_failure', 'sepsis']:
            prob_progressing_severe_complication = m.risk_of_progressing_to_severe_complications[complication].predict(
                df.loc[[person_id]]).values[0]
            complications_for_this_person.append(complication)
            if rng.rand() < prob_progressing_severe_complication:
                df.at[person_id, 'ri_ALRI_severe_complication_date'] = date_of_onset_severe_complication
                self.sim.schedule_event(AlriWithSevereComplicationsEvent(
                    self.module, person_id,
                    duration_in_days=self.duration_in_days,
                    complication=complications_for_this_person,
                    symptoms=all_symptoms_for_this_person),
                    date_of_onset_severe_complication)

            else:
                df.at[person_id, 'ri_ALRI_event_recovered_date'] = date_of_recovery
                self.sim.schedule_event(ALRINaturalRecoveryEvent(self.module, person_id), date_of_recovery)


class AlriWithSevereComplicationsEvent(Event, IndividualScopeEventMixin):
    """
        This Event is for the onset of any complication from Pneumonia. For some untreated children,
        this occurs a set number of days after onset of disease.
        It sets the property 'ri_ALRI_complications' to each complication and schedules the death.
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

        # complications for this person
        df.at[person_id, 'ri_ALRI_complications'] = list(self.complication)

        # date of onset of severe complication
        df.at[person_id, 'ri_ALRI_severe_complication_date'] = self.sim.date

        # add to the initial list of uncomplicated ALRI symptoms
        all_symptoms_for_this_person = list(self.symptoms)  # original uncomplicated symptoms list to add to
        # keep only the probabilities for the complications of the person:
        possible_symptoms_by_complication = {key: val for key, val in
                                             self.module.prob_extra_symptoms_complications.items()
                                             if key in list(self.complication)}

        symptoms_from_complications = list()
        for complication in possible_symptoms_by_complication:
            for symptom, prob in possible_symptoms_by_complication[complication].items():
                if self.module.rng.rand() < prob:
                    symptoms_from_complications.append(symptom)
                for i in symptoms_from_complications:
                    # add symptoms from complications to the list
                    all_symptoms_for_this_person.append(
                        i) if i not in all_symptoms_for_this_person \
                        else all_symptoms_for_this_person

        df.at[person_id, 'ri_current_ALRI_symptoms'] = all_symptoms_for_this_person

        for symptom in all_symptoms_for_this_person:
            self.module.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove='+',
                disease_module=self.module,
                duration_in_days=self.duration_in_days
            )

        # Determine death outcome -------------------------------------------------------------------------
        date_of_outcome = \
            df.at[person_id, 'ri_ALRI_event_date_of_onset'] + DateOffset(days=self.duration_in_days)

        prob_death_from_ALRI = m.risk_of_death_severe_ALRI.predict(df.loc[[person_id]]).values[0]

        if rng.rand() < prob_death_from_ALRI:
            df.at[person_id, 'ri_ALRI_event_death_date'] = date_of_outcome
            self.sim.schedule_event(ALRIDeathEvent(self.module, person_id),
                                    date_of_outcome)
        else:
            df.at[person_id, 'ri_ALRI_event_recovered_date'] = date_of_outcome
            self.sim.schedule_event(ALRINaturalRecoveryEvent(self.module, person_id),
                                    date_of_outcome)


class ALRINaturalRecoveryEvent(Event, IndividualScopeEventMixin):
    """
    This is the Natural Recovery event.
    It is part of the natural history and represents the end of an episode of ALRI
    It does the following:
        * resolves all symptoms caused by ALRI
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

        # Do nothing if the recovery is not expected for today
        # (Because the person has already recovered, through being cured).
        if not (self.sim.date == person.ri_ALRI_event_recovered_date):
            return

        # Confirm that this is event is occurring during a current episode of ALRI
        assert person.ri_ALRI_event_date_of_onset <= self.sim.date <= person.ri_end_of_last_alri_episode

        # Check that the person is not scheduled to die in this episode
        # assert pd.isnull(person.ri_ALRI_event_death_date)

        if not pd.isnull(person.ri_ALRI_event_death_date):
            return

        # clear other properties
        df.at[person_id, 'ri_current_ALRI_status'] = False
        # ---- Key Current Status Classification Properties ----
        df.at[person_id, 'ri_primary_ALRI_pathogen'] = 'not_applicable'
        df.at[person_id, 'ri_current_ALRI_symptoms'] = 'not_applicable'
        df.at[person_id, 'ri_secondary_bacterial_pathogen'] = 'not_applicable'
        df.at[person_id, 'ri_ALRI_disease_type'] = 'not_applicable'
        df.at[person_id, 'ri_ALRI_complications'] = 'not_applicable'
        df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT
        df.at[person_id, 'ri_ALRI_pulmonary_complication_date'] = pd.NaT
        df.at[person_id, 'ri_ALRI_severe_complication_date'] = pd.NaT

        # clear the treatment prperties
        df.at[person_id, 'ri_ALRI_treatment'] = False
        df.at[person_id, 'ri_ALRI_tx_start_date'] = pd.NaT

        # Resolve all the symptoms immediately
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id,
                                                          disease_module=self.sim.modules['ALRI'])


class ALRICureEvent(Event, IndividualScopeEventMixin):
    """
       This is the cure event. It is scheduled by an HSI treatment event.
       It enacts the actual "cure" of the person that is caused (after some delay) by the treatment administered.
       It does the following:
           * Sets the date of recovery to today's date
           * Resolves all symptoms caused by ALRI
       """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("ALRICureEvent: Stopping ALRI treatment and curing person %d", person_id)
        df = self.sim.population.props

        # terminate the event if the person has already died.
        if not df.at[person_id, 'is_alive']:
            return

        # Cure should not happen if the person has already recovered
        if df.at[person_id, 'ri_ALRI_event_recovered_date'] <= self.sim.date:
            return

        # Confirm that this is event is occurring during a current episode of ALRI
        if not (df.at[person_id, 'ri_ALRI_event_date_of_onset']) \
               <= self.sim.date <= (df.at[person_id, 'ri_end_of_last_alri_episode']):
            # If not, then the event has been caused by another cause of diarrhoea (which may has resolved by now)
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

        # Stop the person from dying of ALRI (if they were going to die)
        df.at[person_id, 'ri_ALRI_event_recovered_date'] = self.sim.date
        df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT

        df.at[person_id, 'ri_current_ALRI_status'] = False
        # ---- Key Current Status Classification Properties ----
        df.at[person_id, 'ri_primary_ALRI_pathogen'] = 'not_applicable'
        df.at[person_id, 'ri_current_ALRI_symptoms'] = 'not_applicable'
        df.at[person_id, 'ri_secondary_bacterial_pathogen'] = 'not_applicable'
        df.at[person_id, 'ri_ALRI_disease_type'] = 'not_applicable'
        df.at[person_id, 'ri_ALRI_complications'] = 'not_applicable'
        df.at[person_id, 'ri_ALRI_event_recovered_date'] = self.sim.date
        df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT
        df.at[person_id, 'ri_ALRI_pulmonary_complication_date'] = pd.NaT
        df.at[person_id, 'ri_ALRI_severe_complication_date'] = pd.NaT

        # clear the treatment prperties
        df.at[person_id, 'ri_ALRI_treatment'] = False
        df.at[person_id, 'ri_ALRI_tx_start_date'] = pd.NaT

        # Resolve all the symptoms immediately
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id,
                                                          disease_module=self.sim.modules['ALRI'])


class ALRIDeathEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the death of someone that is caused by the infection with a pathogen that causes ALRI.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # The event should not run if the person is not currently alive
        if not df.at[person_id, 'is_alive']:
            return

        # Confirm that this is event is occurring during a current episode of ALRI

        assert (
            (df.at[person_id, 'ri_ALRI_event_date_of_onset']) <=
            self.sim.date <=
            (df.at[person_id, 'ri_end_of_last_alri_episode'])
        )

        # Check if person should still die of ALRI
        if (
            df.at[person_id, 'ri_ALRI_event_death_date'] == self.sim.date) and \
            pd.isnull(df.at[person_id, 'ri_ALRI_event_recovered_date']
                      ):
            self.sim.schedule_event(demography.InstantaneousDeath(self.module,
                                                                  person_id,
                                                                  cause='ALRI_' + df.at[
                                                                      person_id, 'ri_primary_ALRI_pathogen']
                                                                  ), self.sim.date)


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
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
            self.module.do_treatment(
                person_id=person_id,
                prob_of_cure=self.module.parameters[
                    'prob_of_cure_for_uncomplicated_pneumonia_given_IMCI_pneumonia_treatment']
            )
        else:
            logger.debug(key='debug', data="PkgCode1 is not available, so can't use it.")
            # todo: prbability of referral if no drug available
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

        # then refer to facility level 1 or 2
        self.sim.modules['DxAlgorithmChild'].do_when_facility_level_1(person_id=person_id, hsi_event=self)
        self.sim.modules['DxAlgorithmChild'].do_when_facility_level_2(person_id=person_id, hsi_event=self)
        # todo: which facility level is closest to be refered to?
        # todo: what about those wo are lost to follow up? - incorporate in the code


# PRIMARY LEVEL - IMCI delivered in facility level 1 / health centres -------------------------------------
# ---------------------------------------------------------------------------------------------------------
class HSI_IMCI_No_Pneumonia_Treatment_level_1(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Define the necessary information for an HSI - interventions for non-severe pneumoni at facility level 1
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

        # Define the necessary information for an HSI - interventions for non-severe pneumoni at facility level 1
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
            self.module.do_treatment(
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

        # Define the necessary information for an HSI - interventions for non-severe pneumoni at facility level 1
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

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)

        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_severe_pneumonia]:
            logger.debug(key='debug', data='PkgCode1 is available, so use it.')
            self.module.do_treatment(
                person_id=person_id,
                prob_of_cure=self.module.parameters[
                    'prob_of_cure_for_uncomplicated_pneumonia_given_IMCI_pneumonia_treatment']
            )
            # self.sim.schedule_event(
            #     ALRICureEvent(self.sim.modules['ALRI'], person_id),
            #     self.sim.date + DateOffset(days=3), # todo: need to check the dates conflict with cure/ treatment/ death etc...
            # )
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
        all_available = (outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_severe_pneumonia]) and \
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


# SECONDARY LEVEL - IMCI delivered in facility level 2 / hospitals -------------------------------------
# ---------------------------------------------------------------------------------------------------------
class HSI_IMCI_Pneumonia_Treatment_level_2(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Define the necessary information for an HSI - interventions for non-severe pneumoni at facility level 1
        # (These are blank when created; but these should be filled-in by the module that calls it)
        self.TREATMENT_ID = 'HSI_IMCI_Pneumonia_Treatment_level_2'

        # APP_FOOTPRINT: hospital-level event takes a certain amount of time for health workers
        appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        appt_footprint['Under5OPD'] = 1  # This requires one out patient
        # Demonstrate the equivalence with:
        assert appt_footprint == self.make_appt_footprint({'Under5OPD': 1})

        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1  # Can occur at facility-level 2 / hospitals
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
            self.module.do_treatment(
                person_id=person_id,
                prob_of_cure=self.module.parameters[
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
            self.module.do_treatment(
                person_id=person_id,
                prob_of_cure=self.module.parameters[
                    'prob_of_cure_for_pneumonia_with_severe_complication_given_IMCI_severe_pneumonia_treatment']
            )
            care_management_info.update({
                'treatment_plan': 'treatment_for_severe_pneumonia'})
        else:
            logger.debug(key='debug', data="PkgCode1 is not available, so can't use it.")
            # todo: probability of referral if no drug available
            # self.sim.modules['DxAlgorithmChild'].do_when_facility_level_3(person_id=person_id, hsi_event=self)
            #todo: inpatient bed days
            care_management_info.update({
                'treatment_plan': 'no_available_treatment', 'referral_to_level_2': True})

        # check to see if all consumables returned (for demonstration purposes):
        all_available = (outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_severe_pneumonia]) and \
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


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class AcuteLowerRespiratoryInfectionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This Event logs the number of incident cases that have occurred since the previous logging event.
    Analysis scripts expect that the frequency of this logging event is once per year.
    """

    def __init__(self, module):
        # This event to occur every year
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(years=self.repeat))
        self.date_last_run = self.sim.date

    def apply(self, population):
        df = self.sim.population.props
        # Convert the list of timestamps into a number of timestamps
        # and check that all the dates have occurred since self.date_last_run
        counts = copy.deepcopy(self.module.incident_case_tracker_zeros)

        for age_grp in self.module.incident_case_tracker.keys():
            for pathogen in self.module.pathogens:
                list_of_times = self.module.incident_case_tracker[age_grp][pathogen]
                counts[age_grp][pathogen] = len(list_of_times)
                for t in list_of_times:
                    assert self.date_last_run <= t <= self.sim.date

        # logger.info(key='incidence_count_by_pathogen', data=counts)

        # get single row of dataframe (but not a series) ----------------
        index_children_with_alri = df.index[df.is_alive & (df.age_exact_years < 5) & df.ri_current_ALRI_status]
        index_children = df.index[df.age_exact_years < 5]
        # individual_child = df.loc[[index_children_with_alri[0]]]
        # alri_check = df.loc[index_children_with_alri]
        # logger.info(key='individual_check',
        #             data=individual_child,
        #             description='following an individual through simulation')
        #
        #
        # properties_for_logging = df.loc[index_children_with_alri[0], [
        #                     'ri_current_ALRI_status',
        #                     'ri_primary_ALRI_pathogen',
        #                     'ri_secondary_bacterial_pathogen',
        #                     'ri_ALRI_disease_type',
        #                     'ri_ALRI_complications',
        #                     'ri_current_ALRI_symptoms',
        #                     'ri_ALRI_event_date_of_onset',
        #                     'ri_ALRI_event_recovered_date',
        #                     'ri_ALRI_severe_complication_date',
        #                     'ri_ALRI_event_death_date',
        #                     'ri_ALRI_tx_start_date',
        #                     'ri_end_of_last_alri_episode'
        #                 ]].to_dict()
        # #
        # logger.info(key='one_child',
        #             data=properties_for_logging,
        #             description='one person')

        logger.info('%s|person_one|%s',
                    self.sim.date,
                    df.loc[index_children[0], [
                        'age_exact_years',
                        'ri_current_ALRI_status',
                        'ri_primary_ALRI_pathogen',
                        'ri_secondary_bacterial_pathogen',
                        'ri_ALRI_disease_type',
                        'ri_ALRI_complications',
                        'ri_current_ALRI_symptoms',
                        'ri_ALRI_event_date_of_onset',
                        'ri_ALRI_event_recovered_date',
                        'ri_ALRI_tx_start_date',
                        'ri_ALRI_pulmonary_complication_date',
                        'ri_ALRI_severe_complication_date',
                        'ri_ALRI_event_death_date',
                        'ri_end_of_last_alri_episode',
                        # 'ri_IMCI_classification_as_gold',
                        # 'ri_health_worker_IMCI_classification'
                    ]].to_dict())

        start_date = Date(2010, 1, 1)
        end_date = Date(2015, 1, 1)

        after_start_date_recovery = df['ri_ALRI_event_recovered_date'] >= start_date
        before_end_date_recovery = df['ri_ALRI_event_recovered_date'] <= end_date
        between_two_dates_recovery = after_start_date_recovery & before_end_date_recovery
        recovered = df.loc[between_two_dates_recovery]

        after_start_date_treatment = df['ri_ALRI_tx_start_date'] >= start_date
        before_end_date_treatment = df['ri_ALRI_tx_start_date'] <= end_date
        between_two_dates_treatment = after_start_date_treatment & before_end_date_treatment
        treated = df.loc[between_two_dates_treatment]

        after_start_date_death = df['ri_ALRI_event_death_date'] >= start_date
        before_end_date_death = df['ri_ALRI_event_death_date'] <= end_date
        between_two_dates_death = after_start_date_death & before_end_date_death
        died = df.loc[between_two_dates_death]

        print(recovered)

        data_for_df = {'recovered': len(after_start_date_recovery), 'treated': len(treated), 'died': len(died)}
        # percentages_df = pd.DataFrame.from_dict(data_for_df)

        logger.info(key='percentages',
                    data=data_for_df,
                    description='proportion of recovery, treated, died')

        # logger.info(key='individual_check',
        #             data=individual_child,
        #             description='following an individual through simulation')
        #
#         # log the information on complications ----------------
#         index_alri_with_complications = df.index[df.is_alive & (df.age_exact_years < 5) & df.ri_current_ALRI_status &
#                                                  (df.ri_ALRI_complications != 'none')]
#         # make a df with children with alri complications as the columns
#         df_alri_complications = pd.DataFrame(index=index_alri_with_complications,
#                                              data=bool(),
#                                              columns=list(self.module.complications))
#         for i in index_alri_with_complications:
#             if 'respiratory_failure' in df.ri_ALRI_complications[i]:
#                 update_df = pd.DataFrame({'respiratory_failure': True}, index=[i])
#                 df_alri_complications.update(update_df)
#             if 'pleural_effusion' in df.ri_ALRI_complications[i]:
#                 update_df = pd.DataFrame({'pleural_effusion': True}, index=[i])
#                 df_alri_complications.update(update_df)
#             if 'empyema' in df.ri_ALRI_complications[i]:
#                 update_df = pd.DataFrame({'empyema': True}, index=[i])
#                 df_alri_complications.update(update_df)
#             if 'lung_abscess' in df.ri_ALRI_complications[i]:
#                 update_df = pd.DataFrame({'lung_abscess': True}, index=[i])
#                 df_alri_complications.update(update_df)
#             if 'sepsis' in df.ri_ALRI_complications[i]:
#                 update_df = pd.DataFrame({'sepsis': True}, index=[i])
#                 df_alri_complications.update(update_df)
#             if 'meningitis' in df.ri_ALRI_complications[i]:
#                 update_df = pd.DataFrame({'meningitis': True}, index=[i])
#                 df_alri_complications.update(update_df)
#
#         count_alri_complications = {}
#         for complic in df_alri_complications.columns:
#             count_complication = df_alri_complications[complic].sum()
#             update_dict = {f'{complic}': count_complication}
#             count_alri_complications.update(update_dict)
#
#         complications_summary = {
#             'count': count_alri_complications,
#             'number_of_children_with_complications': len(index_alri_with_complications),
#             'number_of_children_with_and_without_complications': len(index_children_with_alri)
#         }
#
#         logger.info(key='alri_complications',
#                     data=complications_summary,
#                     description='Summary of complications in a year')
#
#         alri_count = df.groupby(df.ri_ALRI_disease_type).size()
# #         print(alri_count)
#
#         # Reset the counters and the date_last_run --------------------
#         self.module.incident_case_tracker = copy.deepcopy(self.module.incident_case_tracker_blank)
#         self.date_last_run = self.sim.date
#
# # todo : empyema not being added complications
