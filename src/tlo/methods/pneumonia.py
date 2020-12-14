"""
Childhood Acute Lower Respiratory Infection Module
Documentation:
04 - Methods Repository/ALRI module - Description.docx
04 - Methods Repository/ResourceFile_ALRI2.xlsx

Overview
--------
Individuals are exposed to the risk of onset of ALRI. They can have pneumonia or bronchiolitis caused by one primary
agent at a time, which can also have a co-infection or secondary bacterial superinfection.
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
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography
from tlo.methods.symptommanager import Symptom

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
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
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

    PARAMETERS = {
        # Incidence rate by pathogens  -----
        'base_inc_rate_ALRI_by_RSV':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by RSV in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_ALRI_by_Rhinovirus':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Rhinovirus in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_ALRI_by_HMPV':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by HMPV in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_ALRI_by_Parainfluenza':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Parainfluenza 1-4 in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_ALRI_by_Strep_pneumoniae_PCV13':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Streptoccocus pneumoniae PCV13 serotype '
                      'in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_ALRI_by_Strep_pneumoniae_non_PCV13':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Streptoccocus pneumoniae non PCV13 serotype '
                      'in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_ALRI_by_Hib':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Haemophilus influenzae type b '
                      'in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_ALRI_by_H.influenzae_non_type_b':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Haemophilus influenzae non-type b '
                      'in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_ALRI_by_Enterobacteriaceae':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Enterobacteriaceae in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_ALRI_by_other_Strepto_Enterococci':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by other streptococci and Enterococci including '
                      'Streptococcus pyogenes and Enterococcus faecium in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_ALRI_by_Staph_aureus':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Staphylococcus aureus '
                      'in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_ALRI_by_Influenza':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by Influenza type A, B, and C '
                      'in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_ALRI_by_P.jirovecii':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by P. jirovecii in age groups 0-11, 12-59 months'
                      ),
        'base_inc_rate_ALRI_by_Adenovirus':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by adenovirus in age groups 0-11, 12-59 months'
                      ),
        'base_inc_rate_ALRI_by_Bocavirus':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by bocavirus in age groups 0-11, 12-59 months'
                      ),
        'base_inc_rate_ALRI_by_other_viral_pathogens':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by other viral pathogens in age groups 0-11, 12-59 months'
                      ),
        'base_inc_rate_ALRI_by_other_bacterial_pathogens':
            Parameter(Types.LIST,
                      'baseline incidence of ALRI caused by other viral pathogens in age groups 0-11, 12-59 months'
                      ),
        # Risk factors parameters -----
        'rr_ALRI_HHhandwashing':
            Parameter(Types.REAL,
                      'relative rate of ALRI with household handwashing with soap'
                      ),
        'rr_ALRI_HIV_untreated':
            Parameter(Types.REAL,
                      'relative rate of ALRI for untreated HIV positive status'
                      ),
        'rr_ALRI_underweight':
            Parameter(Types.REAL,
                      'relative rate of ALRI for underweight'
                      ),
        'rr_ALRI_low_birth_weight':
            Parameter(Types.REAL,
                      'relative rate of ALRI with low birth weight infants'
                      ),
        'rr_ALRI_not_excl_breastfeeding':
            Parameter(Types.REAL,
                      'relative rate of ALRI for not exclusive breastfeeding upto 6 months'
                      ),
        'rr_ALRI_indoor_air_pollution':
            Parameter(Types.REAL,
                      'relative rate of ALRI for indoor air pollution'
                      ),
        # 'rr_ALRI_pneumococcal_vaccine': Parameter
        # (Types.REAL, 'relative rate of ALRI for pneumonococcal vaccine'
        #  ),
        # 'rr_ALRI_haemophilus_vaccine': Parameter
        # (Types.REAL, 'relative rate of ALRI for haemophilus vaccine'
        #  ),
        # 'rr_ALRI_influenza_vaccine': Parameter
        # (Types.REAL, 'relative rate of ALRI for influenza vaccine'
        #  ),
        # 'r_progress_to_severe_ALRI': Parameter
        # (Types.LIST,
        #  'probability of progressing from non-severe to severe ALRI by age category '
        #  'HIV negative, no SAM'
        #  ),

        # Probability of bacterial co- / superinfection
        'prob_viral_pneumonia_bacterial_coinfection':
            Parameter(Types.REAL,
                      'probability of primary viral pneumonia having a bacterial co-infection'
                      ),
        'porportion_bacterial_coinfection_pathogen':
            Parameter(Types.LIST,
                      'proportions of bacterial pathogens in a co-infection pneumonia'
                      ),
        'prob_secondary_bacterial_infection_in_viral_pneumonia':
            Parameter(Types.REAL,
                      'probability of primary viral pneumonia having a bacterial superinfection'
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
                      'probability of empyema causing sepsis'
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
        'prob_atelectasis_by_bronchiolitis':
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
        'r_death_from_ALRI':
            Parameter(Types.REAL,
                      'death rate from ALRI'
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
        'prob_chest_pain_adding_from_pleural_effusion':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of chest pain / pleurisy from pleural effusion'
                      ),
        'prob_loss_of_appetite_adding_from_pleural_effusion':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of loss of appetite from pleural effusion'
                      ),
        'prob_chest_pain_adding_from_empyema':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of chest pain / pleurisy from empyema'
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
        'prob_tachycardia_adding_from_pneumothorax':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of tachycardia from pneumothorax'
                      ),
        'prob_difficult_breathing_adding_from_pneumothorax':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of difficult breathing from pneumothorax'
                      ),
        'prob_fatigue_adding_from_pneumothorax':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of fatigue from pneumothorax'
                      ),
        'prob_decreased_chest_movement_adding_from_pneumothorax':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of decreased chest wall movement from pneumothorax'
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
        'prob_cough_sputum_adding_from_pleural_effusion':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of cough with sputum from pleural effusion'
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
        'prob_cough_sputum_adding_from_empyema':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of cough with sputum from empyema'
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
        'prob_tachypnoea_adding_from_lung_abscess':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of tachypnoea/ fast breathing from lung abscess'
                      ),
        'prob_cough_sputum_adding_from_lung_abscess':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of cough with sputum from lung abscess'
                      ),
        'prob_fever_adding_from_lung_abscess':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of fever from lung abscess'
                      ),
        'prob_hemoptysis_adding_from_lung_abscess':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of hemoptysis from lung abscess'
                      ),
        'prob_tachycardia_adding_from_lung_abscess':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of tachycardia from lung abscess'
                      ),
        'prob_weight_loss_adding_from_lung_abscess':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of weight loss from lung abscess'
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
        'prob_tachypnoea_adding_from_respiratory_failure':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of tachypnoea/ fast breathing from respiratory failure'
                      ),
        'prob_tachycardia_adding_from_respiratory_failure':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of tachycardia from respiratory failure'
                      ),
        'prob_lethargy_adding_from_respiratory_failure':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of lethargy from respiratory failure'
                      ),
        'prob_restlessness_adding_from_respiratory_failure':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of restlessness from respiratory failure'
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
        'prob_bulging_fontanel_adding_from_meningitis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of bulging fontanel from meningitis'
                      ),
        'prob_convulsions_adding_from_meningitis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of convulsions from meningitis'
                      ),
        'prob_nausea_adding_from_meningitis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of nausea from meningitis'
                      ),
        'prob_vomiting_adding_from_meningitis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of vomiting from meningitis'
                      ),
        'prob_photophobia_adding_from_meningitis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of photophobia from meningitis'
                      ),
        'prob_nuchal_rigidity_adding_from_meningitis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of nuchal rigidity from meningitis'
                      ),
        # Sepsis ----------------
        'prob_lethargy_adding_from_sepsis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of lethargy from sepsis'
                      ),
        'prob_fatigue_adding_from_sepsis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of fatigue from sepsis'
                      ),
        'prob_tachypnoea_adding_from_sepsis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of tachypnoea from sepsis'
                      ),
        'prob_convulsions_adding_from_sepsis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of convulsions from sepsis'
                      ),
        'prob_not_eating_adding_from_sepsis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of not eating from sepsis'
                      ),
        'prob_fever_adding_from_sepsis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of fever from sepsis'
                      ),
        'prob_vomiting_adding_from_sepsis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of vomiting from sepsis'
                      ),
        'prob_no_urination_in_last_12h_adding_from_sepsis':
            Parameter(Types.REAL,
                      'probability of additional signs/symptoms of no urination for 12h from sepsis'
                      ),

        # other properties
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

    }

    PROPERTIES = {
        # ---- The pathogen which is the attributed cause of ALRI ----
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
        # ---- The bacterial pathogen which is the attributed cause of ALRI ----
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
                     categories=['fever', 'cough', 'difficult_breathing', 'convulsions', 'lethargy', 'not_eating', 'restlessness',
                                 'fast_breathing', 'chest_indrawing', 'grunting', 'chest_pain', 'loss_of_appetite',
                                 'cyanosis', 'respiratory_distress', 'hypoxia', 'tachypnoea', 'cough_with_sputum',
                                 'hemoptysis', 'tachycardia', 'decreased_chest_movement', 'fatigue', 'no_urination_in_last_12h',
                                 'danger_signs']
                     ),

        # ---- Internal variables to schedule onset and deaths due to ALRI ----
        'ri_ALRI_event_date_of_onset': Property(Types.DATE, 'date of onset of current ALRI event'),
        'ri_ALRI_event_recovered_date': Property(Types.DATE, 'date of recovery from current ALRI event'),
        'ri_ALRI_event_death_date': Property(Types.DATE, 'date of death caused by current ALRI event'),
        'ri_end_of_last_alri_episode':
            Property(Types.DATE, 'date on which the last episode of ALRI is resolved, (including '
                                 'allowing for the possibility that a cure is scheduled following onset). '
                                 'This is used to determine when a new episode can begin. '
                                 'This stops successive episodes interfering with one another.'),

        'ri_secondary_bacterial_infection_date': Property(Types.DATE, 'date of onset of secondary bacterial infection'),

        # ---- Temporary Variables: To be replaced with the properties of other modules ----
        'tmp_malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'tmp_low_birth_weight': Property(Types.BOOL, 'temporary property - low birth weight'),
        'tmp_hv_inf': Property(Types.BOOL, 'temporary property - hiv infection'),
        'tmp_exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'tmp_continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'tmp_pneumococcal_vaccination': Property(Types.BOOL, 'temporary property - streptococcus ALRIe vaccine'),
        'tmp_haemophilus_vaccination': Property(Types.BOOL, 'temporary property - H. influenzae type b vaccine'),
        'tmp_influenza_vaccination': Property(Types.BOOL, 'temporary property - flu vaccine'),

        # ---- HS interventions / Treatment properties ----
        'ri_ALRI_tx_start_date': Property(Types.DATE, 'start date of ALRI treatment for current event'),
        'ri_ALRI_antibiotic_treatment_administered':
            Property(Types.CATEGORICAL,
                     'Antibiotic treatment given for the ALRI',
                     categories=['benzyl penicillin injection', 'amoxicillin', 'cotrimoxazole', 'other_antibiotic',
                                 'chlorampheniciol', 'prednisolone']),
        'ri_peripheral_oxygen_saturation':
            Property(Types.CATEGORICAL,
                     'Level of peripheral oxygen saturation to be read by a pulse oximetry',
                     categories=['SpO2<90%', 'SpO2_90-92%', 'SpO2>92%']),
        'oxygen_therapy_given':
            Property(Types.BOOL,
                     'Oxygen therapy received at the hospital (for severe cases)'),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # equations for the incidence of ALRI by pathogen:
        self.incidence_equations_by_pathogen = dict()

        # equations for the proportions of ALRI diseases:
        self.proportions_of_ALRI_disease_types_by_pathogen = dict()

        # equation for the probability of viral/bacterial (co-infection) pneumonia:
        self.viral_pneumonia_with_bacterial_coinfection = dict()

        # equations for the probabilities of secondary bacterial superinfection:
        self.prob_secondary_bacterial_infection = None

        # equations for the development of ALRI-associated complications:
        self.risk_of_developing_ALRI_complications = dict()

        # Linear Model for predicting the risk of death:
        self.risk_of_death_severe_ALRI = dict()

        # dict to hold the probability of onset of different types of symptom given underlying complications:
        self.prob_symptoms_uncomplicated_ALRI = dict()
        self.prob_extra_symptoms_complications = dict()

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
            'fever', 'cough', 'difficult_breathing', 'convulsions', 'lethargy', 'not_eating', 'restlessness', 'bulging_fontanel',
            'fast_breathing', 'chest_indrawing', 'grunting', 'chest_pain', 'loss_of_appetite', 'photophobia', 'nuchal_rigidity',
            'cyanosis', 'respiratory_distress', 'hypoxia', 'tachypnoea', 'cough_with_sputum', 'weight_loss', 'nausea',
            'hemoptysis', 'tachycardia', 'decreased_chest_movement', 'fatigue', 'no_urination_in_last_12h', 'hemoptysis',
            'danger_signs'
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
        m = self

        # ---- Key Current Status Classification Properties ----
        df.loc[df.is_alive, 'ri_primary_ALRI_pathogen'].values[:] = 'not_applicable'
        df.loc[df.is_alive, 'ri_current_ALRI_symptoms'] = 'not_applicable'
        df.loc[df.is_alive, 'ri_secondary_bacterial_pathogen'] = 'not_applicable'
        df.loc[df.is_alive, 'ri_ALRI_disease_type'] = 'not_applicable'
        df.loc[df.is_alive, 'ri_ALRI_complications'] = 'not_applicable'

        # ---- Internal values ----
        df.loc[df.is_alive, 'ri_ALRI_event_date_of_onset'] = pd.NaT
        df.loc[df.is_alive, 'ri_ALRI_event_recovered_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_ALRI_event_death_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_end_of_last_alri_episode'] = pd.NaT

        df.loc[df.is_alive, 'ri_ALRI_treatment'] = False
        df.loc[df.is_alive, 'ri_ALRI_tx_start_date'] = pd.NaT

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
        sim.schedule_event(AcuteLowerRespiratoryInfectionPollingEvent(self), sim.date)

        # Schedule the main logging event (to first occur in one year)
        sim.schedule_event(AcuteLowerRespiratoryInfectionLoggingEvent(self), sim.date + DateOffset(years=1))

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
                            p['prob_secondary_bacterial_infection_in_viral_pneumonia']),
                        Predictor()
                        .when(
                            "ri_primary_ALRI_pathogen.isin(['RSV', 'rhinovirus', 'hMPV', "
                            "'parainfluenza', 'influenza']) & "
                            "ri_ALRI_disease_type == 'bronchiolitis' & "
                            "ri_secondary_bacterial_pathogen =='not_applicable' ",
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
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_pneumothorax_by_bacterial_pneumonia'])
                            .when(
                                ".isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', 'Influenza', "
                                "'other_viral_pathogens'])",
                                p['prob_pneumothorax_by_viral_pneumonia'])
                            .otherwise(0.0)
                            ),

            'pleural_effusion':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_pleural_effusion_by_bacterial_pneumonia'])
                            .otherwise(0.0)
                            ),

            'empyema':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_current_ALRI_complications').apply(
                                lambda x: p['prob_pleural_effusion_to_empyema'] if 'pleural_effusion' in x else 0)
                            ),

            'lung_abscess':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen')
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
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_sepsis_by_bacterial_pneumonia'])
                            .when(
                                ".isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', 'Influenza', "
                                "'other_viral_pathogens'])",
                                p['prob_sepsis_by_viral_pneumonia'])
                            .otherwise(0.0)
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
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_respiratory_failure_by_bacterial_pneumonia'])

                            .when(
                                ".isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', 'Influenza', "
                                "'other_viral_pathogens'])",
                                p['prob_respiratory_failure_by_viral_pneumonia'])
                            .otherwise(0.0)
                            ),
        }),

        # check that equations have been declared for each complication
        assert self.complications == set(list(self.risk_of_developing_ALRI_complications.keys()))

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
            if disease_type == 'fungal_pneumonia':
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
                    'tachycardia': p[f'prob_tachycardia_adding_from_{complicat}'],
                    'cyanosis': p[f'prob_cyanosis_adding_from_{complicat}'],
                    'fatigue': p[f'prob_fatigue_adding_from_{complicat}'],
                    'difficult_breathing': p[f'prob_difficult_breathing_adding_from_{complicat}'],
                    'decreased_chest_movement': p[f'prob_decreased_chest_movement_adding_from_{complicat}'],
                }
            if complicat == 'pleural_effusion':
                return {
                    'chest_pain': p[f'prob_chest_pain_adding_from_{complicat}'],
                    'cough_with_sputum': p[f'prob_cough_sputum_adding_from_{complicat}'],
                    'fever': p[f'prob_fever_adding_from_{complicat}'],
                    'difficult_breathing': p[f'prob_difficult_breathing_adding_from_{complicat}'],
                }
            if complicat == 'empyema':
                return {
                    'chest_pain': p[f'prob_chest_pain_adding_from_{complicat}'],
                    'cough_with_sputum': p[f'prob_cough_sputum_adding_from_{complicat}'],
                    'fever': p[f'prob_fever_adding_from_{complicat}'],
                    'respiratory_distress': p[f'prob_respiratory_distress_adding_from_{complicat}'],
                }
            if complicat == 'lung_abscess':
                return {
                    'chest_pain': p[f'prob_chest_pain_adding_from_{complicat}'],
                    'tachypnoea': p[f'prob_tachypnoea_adding_from_{complicat}'],
                    'cough_with_sputum': p[f'prob_cough_sputum_adding_from_{complicat}'],
                    'hemoptysis': p[f'prob_hemoptysis_adding_from_{complicat}'],
                    'fever': p[f'prob_fever_adding_from_{complicat}'],
                    'weight_loss': p[f'prob_weight_loss_adding_from_{complicat}'],
                    'tachycardia': p[f'prob_tachycardia_adding_from_{complicat}'],
                }
            if complicat == 'respiratory_failure':
                return {
                    'cyanosis': p[f'prob_cyanosis_adding_from_{complicat}'],
                    'tachypnoea': p[f'prob_tachypnoea_adding_from_{complicat}'],
                    'tachycardia': p[f'prob_tachycardia_adding_from_{complicat}'],
                    'lethargy': p[f'prob_lethargy_adding_from_{complicat}'],
                    'restlessness': p[f'prob_restlessness_adding_from_{complicat}'],
                    'difficult_breathing': p[f'prob_difficult_breathing_adding_from_{complicat}'],
                }
            if complicat == 'sepsis':
                return {
                    'fever': p[f'prob_fever_adding_from_{complicat}'],
                    'fatigue': p[f'prob_fatigue_adding_from_{complicat}'],
                    'tachypnoea': p[f'prob_tachypnoea_adding_from_{complicat}'],
                    'lethargy': p[f'prob_lethargy_adding_from_{complicat}'],
                    'convulsions': p[f'prob_convulsions_adding_from_{complicat}'],
                    'not_eating': p[f'prob_not_eating_adding_from_{complicat}'],
                    'vomiting': p[f'prob_vomiting_adding_from_{complicat}'],
                    'no_urination_in_last_12h': p[f'prob_no_urination_in_last_12h_adding_from_{complicat}'],
                }
            if complicat == 'meningitis':
                return {
                    'fever': p[f'prob_fever_adding_from_{complicat}'],
                    'headache': p[f'prob_headache_adding_from_{complicat}'],
                    'bulging_fontanel': p[f'prob_bulging_fontanel_adding_from_{complicat}'],
                    'convulsions': p[f'prob_convulsions_adding_from_{complicat}'],
                    'nausea': p[f'prob_nausea_adding_from_{complicat}'],
                    'vomiting': p[f'prob_vomiting_adding_from_{complicat}'],
                    'photophobia': p[f'prob_photophobia_adding_from_{complicat}'],
                    'nuchal_rigidity': p[f'prob_nuchal_rigidity_adding_from_{complicat}'],
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
            Predictor('ri_current_ALRI_complications').apply(death_risk),
            Predictor('tmp_hv_inf').when(True, p['rr_death_ALRI_HIV']),
            Predictor('tmp_malnutrition').when(True, p['rr_death_ALRI_SAM']),
            Predictor('tmp_low_birth_weight').when(True, p['rr_death_ALRI_low_birth_weight']),
            Predictor('age_years')
                .when('.between(1,1)', p['rr_death_ALRI_age12to23mo'])
                .when('.between(2,4)', p['rr_death_ALRI_age24to59mo'])
        )
        # todo: do I need to add risk factors for death from severe complications??
        # -----------------------------------------------------------------------------------------------------

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        # ---- Key Current Status Classification Properties ----
        df.at[child_id, 'ri_primary_ALRI_pathogen'] = 'not_applicable'
        df.at[child_id, 'ri_current_ALRI_symptoms'] = 'not_applicable'
        df.at[child_id, 'ri_secondary_bacterial_pathogen'] = 'not_applicable'
        df.at[child_id, 'ri_ALRI_disease_type'] = 'not_applicable'
        df.at[child_id, 'ri_ALRI_complications'] = 'not_applicable'

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

    def cancel_death_date(self, person_id):
        """
        Cancels a scheduled date of death due to ALRI for a person. This is called prior to the scheduling the
        CureEvent to prevent deaths happening in the time between a treatment being given and the cure event occurring.
        :param person_id:
        :return:
        """
        df = self.sim.population.props
        df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT


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
        super().__init__(module, frequency=DateOffset(months=3))
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
        assert all(prob_of_acquiring_any_pathogen < 1.0)

        # Determine which persons will acquire any pathogen:
        person_id_that_acquire_pathogen = prob_of_acquiring_any_pathogen.index[
            rng.rand(len(prob_of_acquiring_any_pathogen)) < prob_of_acquiring_any_pathogen]

        # Determine which pathogen each person will acquire (among those who will get a pathogen)
        # and create the event for the onset of new infection
        for person_id in person_id_that_acquire_pathogen:
            # ----------------------- Allocate a pathogen to the person ----------------------
            p_by_pathogen = probs_of_acquiring_pathogen.loc[person_id].values
            # print(sum(p_by_pathogen))
            normalised_p_by_pathogen = p_by_pathogen / sum(p_by_pathogen)
            # print(sum(normalised_p_by_pathogen))
            pathogen = rng.choice(probs_of_acquiring_pathogen.columns, p=normalised_p_by_pathogen)

            # ----------------- Determine the ALRI disease type for this case -----------------
            alri_disease_type_for_this_person = m.proportions_of_ALRI_disease_types_by_pathogen[pathogen]
            print(alri_disease_type_for_this_person)

            # ------- Allocate a secondary bacterial pathogen in co-infection with primary viral pneumonia -------
            bacterial_patho_in_viral_pneumonia_coinfection = 'not_applicable'
            if pathogen in self.module.viral_patho:
                if alri_disease_type_for_this_person == 'viral_pneumonia':
                    pneumonia_will_have_bacterial_coinfection = rng.rand() < p['prob_viral_pneumonia_bacterial_coinfection']
                    print(pneumonia_will_have_bacterial_coinfection)
                    if pneumonia_will_have_bacterial_coinfection:
                        bacterial_coinfection_pathogen = rng.choice(list(self.module.bacterial_patho),
                                                                    p=p['porportion_bacterial_coinfection_pathogen'])
                        bacterial_patho_in_viral_pneumonia_coinfection = bacterial_coinfection_pathogen
                        # update to co-infection property
                        alri_disease_type_for_this_person = 'bacterial_pneumonia'
                    else:
                        bacterial_patho_in_viral_pneumonia_coinfection = 'none'
                if alri_disease_type_for_this_person == 'bronchiolitis':
                    bacterial_patho_in_viral_pneumonia_coinfection = 'none'
            if pathogen in self.module.bacterial_patho:
                bacterial_patho_in_viral_pneumonia_coinfection = 'not_applicable'
            if pathogen in self.module.fungal_patho:
                bacterial_patho_in_viral_pneumonia_coinfection = 'none'

            print(alri_disease_type_for_this_person, bacterial_patho_in_viral_pneumonia_coinfection)

            # ----------------------- Allocate a date of onset of ALRI ----------------------
            date_onset = self.sim.date + DateOffset(days=np.random.randint(0, days_until_next_polling_event))

            # ----------------------- Duration of the ALRI event -----------------------
            duration_in_days_of_alri = max(1, int(
                7 + (-2 + 4 * rng.rand())))  # assumes uniform interval around mean duration with range 4 days

            # ----------------------- Create the event for the onset of infection -------------------
            self.sim.schedule_event(
                event=AcuteLowerRespiratoryInfectionIncidentCase(
                    module=self.module,
                    person_id=person_id,
                    pathogen=pathogen,
                    disease_type=alri_disease_type_for_this_person,
                    co_bacterial_patho=bacterial_patho_in_viral_pneumonia_coinfection,
                    duration_in_days=duration_in_days_of_alri,
                ),
                date=date_onset
            )


class AcuteLowerRespiratoryInfectionIncidentCase(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of the infection that causes ALRI.
    """

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
        df.at[person_id, 'ri_current_ALRI_complications'] = 'none'  # all disease start as non-severe

        print(self.disease)

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

        # # ------ Add the event of a SECONDARY BACTERIAL SUPERINFECTION - in viral pneumonia and bronchiolitis ------
        # prob_bacterial_superinfection = m.prob_secondary_bacterial_infection.predict(df.loc[[person_id]]).values[0]
        # if rng.rand() < prob_bacterial_superinfection:
        #     # choose a secondary bacterial pathogen
        #     secondary_bacterial_pathogen = rng.choice(list(self.module.bacterial_patho),
        #                                               p=p['porportion_bacterial_coinfection_pathogen'])
        #     # date of onset of secondary bacterial infection
        #     date_onset_secondary_bacterial = self.module.sim.date + DateOffset(
        #         days=np.random.randint(1, high=self.duration_in_days))  # todo: has to be less than full duration
        #
        #     self.sim.schedule_event(BacterialSuperinfectionEvent(
        #         self.module, person_id,
        #         symptoms=symptoms_for_this_person, duration_in_days=self.duration_in_days,
        #         bacterial_pathogen=secondary_bacterial_pathogen), date=date_onset_secondary_bacterial)

        # COMPLICATIONS ------------------------------------------------------------------------------------------
        complications_for_this_person = list()
        for complication in self.module.complications:
            prob_developing_each_complication = m.risk_of_developing_ALRI_complications[complication].predict(df.loc[[person_id]]).values[0]
            if rng.rand() < prob_developing_each_complication:
                complications_for_this_person.append(complication)
        print(complications_for_this_person)

        if len(complications_for_this_person) != 0:
            for i in complications_for_this_person:
                date_onset_complications = self.module.sim.date + DateOffset(
                    days=np.random.randint(3, high=self.duration_in_days))
                print(i, date_onset_complications)
                self.sim.schedule_event(PneumoniaWithComplicationsEvent(
                    self.module, person_id, duration_in_days=self.duration_in_days, symptoms=symptoms_for_this_person,
                    complication=complications_for_this_person), date_onset_complications)
            df.at[person_id, 'ri_ALRI_event_recovered_date'] = pd.NaT
            df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT
        else:
            df.at[person_id, 'ri_ALRI_event_recovered_date'] = date_of_outcome
            df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT

        # Record 'episode end' data. This the date when this episode ends. It is the last possible data that any HSI
        # could affect this episode.
        df.at[person_id, 'ri_end_of_last_alri_episode'] = date_of_outcome + DateOffset(
            days=self.module.parameters['days_between_treatment_and_cure']
        )

        # self.sim.modules['DxAlgorithmChild'].imnci_as_gold_standard(person_id=person_id)

        # Add this incident case to the tracker
        age = df.loc[person_id, ['age_years']]
        if age.values[0] < 5:
            age_grp = age.map({0: '0y', 1: '1y', 2: '2-4y', 3: '2-4y', 4: '2-4y'}).values[0]
        else:
            age_grp = '5+y'
        self.module.incident_case_tracker[age_grp][self.pathogen].append(self.sim.date)


# class BacterialSuperinfectionEvent(Event, IndividualScopeEventMixin):
#     """
#         This Event is for the onset of a secondary bacterial superinfection
#          in children with primary viral pneumonia or bronchiolitis
#         """
#
#     def __init__(self, module, person_id, symptoms, duration_in_days, bacterial_pathogen):
#         super().__init__(module, person_id=person_id)
#         self.bacterial_pathogen = bacterial_pathogen
#         self.symptoms = symptoms
#         self.duration_in_days = duration_in_days
#
#     def apply(self, person_id):
#         df = self.sim.population.props
#         m = self.module
#         rng = self.module.rng
#
#         # The event should not run if the person is not currently alive
#         if not df.at[person_id, 'is_alive']:
#             return
#
#         # Update the properties in the dataframe:
#         df.at[person_id, 'ri_secondary_bacterial_pathogen'] = self.bacterial_pathogen
#         df.at[person_id, 'ri_ALRI_disease_type'] = 'bacterial_pneumonia'
#         df.at[person_id, 'ri_secondary_bacterial_infection_date'] = self.sim.date
#
#         all_symptoms_for_this_person = list(self.symptoms)
#
#         # Update the symptoms from bacterial cause
#         added_bacterial_cause_symptoms = list()
#         bacterial_pneumonia_add_symptoms = m.prob_symptoms_uncomplicated_ALRI['bacterial_pneumonia']
#         for symptom, prob in bacterial_pneumonia_add_symptoms.items():
#             if rng.rand() < prob:
#                 added_bacterial_cause_symptoms.append(symptom)
#                 for i in added_bacterial_cause_symptoms:
#                     # add symptoms from complications to the list
#                     all_symptoms_for_this_person.append(
#                         i) if i not in all_symptoms_for_this_person \
#                         else all_symptoms_for_this_person
#
#         # Onset symptoms:
#         for symptom in added_bacterial_cause_symptoms:
#             self.module.sim.modules['SymptomManager'].change_symptom(
#                 person_id=person_id,
#                 symptom_string=symptom,
#                 add_or_remove='+',
#                 disease_module=self.module,
#                 duration_in_days=self.duration_in_days
#             )
#         df.at[person_id, 'ri_current_ALRI_symptoms'] = added_bacterial_cause_symptoms


class PneumoniaWithComplicationsEvent(Event, IndividualScopeEventMixin):
    """
        This Event is for the onset of any complication from Pneumonia. For some untreated children,
        this occurs a set number of days after onset of disease.
        It sets the property 'ri_current_ALRI_complications' to each complication and schedules the death.
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

        # add to the initial list of uncomplicated ALRI symptoms
        all_symptoms_for_this_person = list(self.symptoms)  # original uncomplicated symptoms list to add to

        # keep only the probabilities for the complications of the person:
        possible_symptoms_by_complication = {key: val for key, val in
                                             self.module.prob_extra_symptoms_complications.items()
                                             if key in list(self.complication)}
        print(possible_symptoms_by_complication)
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

        print(all_symptoms_for_this_person)
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

        prob_death_from_ALRI = \
            m.risk_of_death_severe_ALRI.predict(df.loc[[person_id]]).values[0]
        death_outcome = rng.rand() < prob_death_from_ALRI

        if death_outcome:
            df.at[person_id, 'ri_ALRI_event_recovered_date'] = pd.NaT
            df.at[person_id, 'ri_ALRI_event_death_date'] = date_of_outcome
            self.sim.schedule_event(PneumoniaDeathEvent(self.module, person_id),
                                    date_of_outcome)
        else:
            df.at[person_id, 'ri_ALRI_event_recovered_date'] = date_of_outcome
            df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT


class PneumoniaCureEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("PneumoniaCureEvent: Stopping ALRI treatment and curing person %d", person_id)
        df = self.sim.population.props

        # terminate the event if the person has already died.
        if not df.at[person_id, 'is_alive']:
            return

        # Cure should not happen if the person has already recovered
        if df.at[person_id, 'ri_ALRI_event_recovered_date'] <= self.sim.date:
            return

        # This event should only run after the person has received a treatment during this episode
        assert (
            (df.at[person_id, 'ri_ALRI_event_date_of_onset']) <=
            (df.at[person_id, 'ri_ALRI_tx_start_date']) <=
            self.sim.date
        )

        # Stop the person from dying of ALRI (if they were going to die)
        df.at[person_id, 'ri_ALRI_event_recovered_date'] = self.sim.date
        df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT

        # clear other properties
        df.at[person_id, 'ri_current_ALRI_status'] = False
        # ---- Key Current Status Classification Properties ----
        df.at[person_id, 'ri_primary_ALRI_pathogen'] = 'not_applicable'
        df.at[person_id, 'ri_current_ALRI_symptoms'] = 'not_applicable'
        df.at[person_id, 'ri_secondary_bacterial_pathogen'] = 'not_applicable'
        df.at[person_id, 'ri_ALRI_disease_type'] = 'not_applicable'
        df.at[person_id, 'ri_ALRI_complications'] = 'not_applicable'

        # clear the treatment prperties
        df.at[person_id, 'ri_ALRI_treatment'] = False
        df.at[person_id, 'ri_ALRI_tx_start_date'] = pd.NaT

        # Resolve all the symptoms immediately
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id,
                                                          disease_module=self.sim.modules['Pneumonia'])


class PneumoniaDeathEvent(Event, IndividualScopeEventMixin):
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

        # Confirm that this is event is occurring during a current episode of diarrhoea
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
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
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

        logger.info(key='incidence_count_by_pathogen', data=counts)

        # get single row of dataframe (but not a series) ----------------
        index_children_with_alri = df.index[df.is_alive & (df.age_exact_years < 5) & df.ri_current_ALRI_status]
        individual_child = df.loc[[index_children_with_alri[0]]]

        logger.debug(key='individual_check',
                     data=individual_child,
                     description='following an individual through simulation')

        # log the information on complications ----------------
        index_alri_with_complications = df.index[df.is_alive & (df.age_exact_years < 5) & df.ri_current_ALRI_status &
                                                 (df.ri_ALRI_complications != 'none')]
        # make a df with children with alri complications as the columns
        df_alri_complications = pd.DataFrame(index=index_alri_with_complications,
                                             data=bool(),
                                             columns=list(self.module.complications))
        for i in index_alri_with_complications:
            if 'respiratory_failure' in df.ri_ALRI_complications[i]:
                update_df = pd.DataFrame({'respiratory_failure': True}, index=[i])
                df_alri_complications.update(update_df)
            if 'pleural_effusion' in df.ri_ALRI_complications[i]:
                update_df = pd.DataFrame({'pleural_effusion': True}, index=[i])
                df_alri_complications.update(update_df)
            if 'empyema' in df.ri_ALRI_complications[i]:
                update_df = pd.DataFrame({'empyema': True}, index=[i])
                df_alri_complications.update(update_df)
            if 'lung_abscess' in df.ri_ALRI_complications[i]:
                update_df = pd.DataFrame({'lung_abscess': True}, index=[i])
                df_alri_complications.update(update_df)
            if 'sepsis' in df.ri_ALRI_complications[i]:
                update_df = pd.DataFrame({'sepsis': True}, index=[i])
                df_alri_complications.update(update_df)
            if 'meningitis' in df.ri_ALRI_complications[i]:
                update_df = pd.DataFrame({'meningitis': True}, index=[i])
                df_alri_complications.update(update_df)
        print(df_alri_complications)

        count_alri_complications = {}
        for complic in df_alri_complications.columns:
            count_complication = df_alri_complications[complic].sum()
            update_dict = {f'{complic}': count_complication}
            count_alri_complications.update(update_dict)
        print(count_alri_complications)

        complications_summary = {
            'count': count_alri_complications,
            'number_of_children_with_complications': len(index_alri_with_complications),
            'number_of_children_with_and_without_complications': len(index_children_with_alri)
        }
        print(complications_summary)

        logger.info(key='alri_complications',
                    data=complications_summary,
                    description='Summary of complications in a year')

        alri_count = df.groupby(df.ri_ALRI_disease_type).size()
        alri_check = df.loc[index_children_with_alri]

        print(alri_count)

        # Reset the counters and the date_last_run --------------------
        self.module.incident_case_tracker = copy.deepcopy(self.module.incident_case_tracker_blank)
        self.date_last_run = self.sim.date

# todo : empyema not being added complications
