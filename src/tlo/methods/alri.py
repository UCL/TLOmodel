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
* All HSI events
* Follow-up appointments for initial HSI events.
* Double check parameters and consumables codes for the HSI events.
"""

# #todo - consider properties: remove/rename "



from pathlib import Path

import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography
from tlo.methods.causes import Cause
from tlo.methods.symptommanager import Symptom
from tlo.util import BitsetHandler
from tlo.methods.healthsystem import HSI_Event

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

    # Declare the pathogen types and specific pathogens:
    viral_patho = {'RSV',
                   'Rhinovirus',
                   'HMPV',
                   'Parainfluenza',
                   'Influenza',
                   'Adenovirus',
                   'Bocavirus',
                   'other_viral_pathogens'
                   # <-- Coronaviruses NL63, 229E OC43 and HKU1, Cytomegalovirus, Parechovirus/Enterovirus
                   }

    bacterial_patho = {'Strep_pneumoniae_PCV13',
                       'Strep_pneumoniae_non_PCV13',
                       'Hib',
                       'H.influenzae_non_type_b',
                       'Staph_aureus',
                       'Enterobacteriaceae',  # includes E. coli, Enterobacter species, and Klebsiella species
                       'other_Strepto_Enterococci',  # includes Streptococcus pyogenes and Enterococcus faecium
                       'other_bacterial_pathogens'
                       # <-- includes Bordetella pertussis, Chlamydophila pneumoniae,
                       # Legionella species, Mycoplasma pneumoniae, Moraxella catarrhalis, Non-fermenting gram-negative
                       # rods (Acinetobacter species and Pseudomonas species), Neisseria meningitidis
                       }

    fungal_patho = {'P.jirovecii'}

    # Make set of all pathogens combined:
    pathogens = viral_patho | bacterial_patho | fungal_patho

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        f"ALRI_{path}":
            Cause(gbd_causes={'Lower respiratory infections'}, label='Lower respiratory infections')
        for path in pathogens
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        f"ALRI_{path}":
            Cause(gbd_causes={'Lower respiratory infections'}, label='Lower Lower respiratory infections')
        for path in pathogens
    }

    # Declare the disease types:
    disease_types = {
        'bacterial_pneumonia', 'viral_pneumonia', 'fungal_pneumonia', 'bronchiolitis'
    }

    # Declare the Alri complications:
    complications = {'pneumothorax',
                     'pleural_effusion',
                     'empyema',
                     'lung_abscess',
                     'sepsis',
                     'meningitis',
                     'respiratory_failure',
                     'peripheral_oxygen_saturation_low'  # <-- Low implies Sp02<93%'
                     }

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
                     categories=list(pathogens)
                     ),
        # ---- The bacterial pathogen which is the attributed co-/secondary infection ----
        'ri_secondary_bacterial_pathogen':
            Property(Types.CATEGORICAL,
                     'If infected, is there a secondary bacterial pathogen (np.nan if none or not applicable)',
                     categories=list(bacterial_patho)
                     ),
        # ---- The underlying Alri condition ----
        'ri_disease_type':
            Property(Types.CATEGORICAL, 'If infected, what disease type is the person currently suffering from.',
                     categories=list(disease_types)
                     ),

        # ---- Treatment Status ----
        'ri_on_treatment': Property(Types.BOOL, 'Is this person currently receiving treatment.'),

        # < --- other properties of the form 'ri_complication_{complication-name}' are added later -->

        # ---- Internal variables to schedule onset and deaths due to Alri ----
        'ri_start_of_current_episode': Property(Types.DATE, 'date of onset of current Alri event (pd.NaT is not infected)'),
        'ri_scheduled_recovery_date': Property(Types.DATE, '(scheduled) date of recovery from current Alri event (pd.NaT is not infected or episode is scheduled to end in death)'),
        'ri_scheduled_death_date': Property(Types.DATE, '(scheduled) date of death caused by current Alri event (pd.NaT is not infected or episode will not cause death)'),
        'ri_end_of_current_episode':
            Property(Types.DATE, 'date on which the last episode of Alri is resolved, (including '
                                 'allowing for the possibility that a cure is scheduled following onset). '
                                 'This is used to determine when a new episode can begin. '
                                 'This stops successive episodes interfering with one another.'),
        'ri_ALRI_tx_start_date': Property(Types.DATE, 'start date of Alri treatment for current episode (pd.NaT is not infected or treatment has not begun)'),
    }

    def __init__(self, name=None, resourcefilepath=None, log_indivdual=False, do_checks=False):
        super().__init__(name)

        # Store arguments provided
        self.resourcefilepath = resourcefilepath
        self.log_individual = log_indivdual
        self.do_checks = do_checks

        # Initialise where the linear models will be stored
        # todo - refactor this so that these are containd with a single dict

        # equations for the incidence of Alri by pathogen:
        self.incidence_equations_by_pathogen = dict()

        # equations for the probabilities of secondary bacterial superinfection:
        self.prob_secondary_bacterial_infection = None

        # equations for the development of Alri-associated complications:
        self.risk_of_developing_ALRI_complications = dict()
        self.risk_of_progressing_to_severe_complications = dict()

        # dict to hold the probability of onset of different types of symptom given underlying complications:
        self.prob_symptoms_uncomplicated_ALRI = dict()
        self.prob_extra_symptoms_complications = dict()

        # Linear Model for predicting the risk of death:
        self.risk_of_death = dict()

        # Maximum duration of an episode (beginning with inection and ending with recovery)
        self.max_duration_of_epsiode = None

        # dict to hold the DALY weights
        self.daly_wts = dict()

        # will store the logging event used by this module
        self.logging_event = None

    def read_parameters(self, data_folder):
        """
        * Setup parameters values used by the module
        * Define symptoms
        """
        self.load_parameters_from_dataframe(
            pd.read_excel(
                Path(self.resourcefilepath) / 'ResourceFile_Alri.xlsx', sheet_name='Parameter_values'))

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
        # todo - @ines: is it right that 'danger_signs' is an indepednet symptom? It seems like this is something that
        #  is determined in the course of a diagnosis (like in diarrhoea module).

        all_symptoms = {
            'fever', 'cough', 'difficult_breathing', 'fast_breathing', 'chest_indrawing', 'chest_pain', 'cyanosis',
            'respiratory_distress', 'danger_signs'
        }

        for symptom_name in all_symptoms:
            if symptom_name not in self.sim.modules['SymptomManager'].generic_symptoms:
                self.sim.modules['SymptomManager'].register_symptom(
                    Symptom(name=symptom_name)
                    # (give non-generic symptom 'average' healthcare seeking)
                )

    def pre_initialise_population(self):
        """Define columns for complications at run-time"""
        Alri.PROPERTIES.update({
            f"ri_complication_{complication}": Property(Types.BOOL,
                                                        f"Whether this person has complication {complication}")
            for complication in self.complications
        })

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

        # Schedule the main polling event (to first occur immediately)
        sim.schedule_event(AlriPollingEvent(self), sim.date)

        # Schedule the main logging event (to first occur in one year)
        self.logging_event = AlriLoggingEvent(self)
        sim.schedule_event(self.logging_event, sim.date + DateOffset(years=1))

        if self.log_individual:
            # Schedule the individual check logging event (to first occur immediately, and to occur every day)
            sim.schedule_event(AlriIndividualLoggingEvent(self), sim.date)

        if self.do_checks:
            # Schedule the event that does checking every day:
            sim.schedule_event(AlriCheckPropertiesEvent(self), sim.date)

        # Make the linear models:
        self.make_linear_models()

        # Get DALY weights
        if 'HealthBurden' in self.sim.modules.keys():
            self.daly_wts['daly_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
            self.daly_wts['daly_severe_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
            self.daly_wts['daly_very_severe_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=46)
            # todo @ines - 'daly_very_severe_ALRI' is never used: is that right?

        # Define the max episode duration
        self.max_duration_of_epsiode = DateOffset(days=self.parameters['days_between_treatment_and_cure'])

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        #todo - vectorize this:

        # ---- Key Current Status Classification Properties ----
        df.at[child_id, 'ri_current_infection_status'] = False
        df.at[child_id, 'ri_primary_pathogen'] = np.nan
        df.at[child_id, 'ri_secondary_bacterial_pathogen'] = np.nan
        df.at[child_id, 'ri_disease_type'] = np.nan
        df.at[child_id, [f"ri_complication_{complication}" for complication in self.complications]] = False

        # ---- Internal values ----
        df.at[child_id, 'ri_start_of_current_episode'] = pd.NaT
        df.at[child_id, 'ri_scheduled_recovery_date'] = pd.NaT
        df.at[child_id, 'ri_scheduled_death_date'] = pd.NaT
        df.at[child_id, 'ri_end_of_current_episode'] = pd.NaT

    def report_daly_values(self):
        """Report DALY incurred in the population in the last month due to ALRI"""
        df = self.sim.population.props

        total_daly_values = pd.Series(data=0.0, index=df.index[df.is_alive])
        total_daly_values.loc[
            self.sim.modules['SymptomManager'].who_has('fast_breathing')] = self.daly_wts['daly_ALRI']
        total_daly_values.loc[
            self.sim.modules['SymptomManager'].who_has('danger_signs')] = self.daly_wts['daly_severe_ALRI']

        # Split out by pathogen that causes the Alri
        dummies_for_pathogen = pd.get_dummies(df.loc[total_daly_values.index, 'ri_primary_pathogen'], dtype='float')
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
                    Predictor('hv_inf').when(True, p['rr_ALRI_HIV_untreated']),
                    Predictor().when(
                        "(tmp_pneumococcal_vaccination == True) & "
                        "((ri_primary_pathogen == 'streptococcus') | "
                        "(ri_secondary_bacterial_pathogen == 'streptococcus'))",
                        p['rr_ALRI_PCV13']),
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
        # todo @ines - is this checking to see if the pathgen is a viral pathogen. We could have a property to say
        #  'primary_ALRI_pathogen_type' to make this simpler?
        self.prob_secondary_bacterial_infection = \
            LinearModel(LinearModelType.ADDITIVE,
                        0.0,
                        Predictor()
                        .when(
                            "ri_primary_pathogen.isin(['RSV', 'rhinovirus', 'hMPV', "
                            "'parainfluenza', 'influenza']) & "
                            "(ri_disease_type == 'viral_pneumonia') & "
                            "(ri_secondary_bacterial_pathogen =='not_applicable')",
                            p['prob_viral_pneumonia_bacterial_coinfection']),
                        Predictor()
                        .when(
                            "ri_primary_pathogen.isin(['RSV', 'rhinovirus', 'hMPV', "
                            "'parainfluenza', 'influenza']) & "
                            "(ri_disease_type == 'bronchiolitis') & "
                            "(ri_secondary_bacterial_pathogen =='not_applicable')",
                            p['prob_secondary_bacterial_infection_in_bronchiolitis'])
                        )

        # =====================================================================================================
        # APPLY LINEAR MODEL TO DETERMINE PROBABILITY OF COMPLICATIONS
        # -----------------------------------------------------------------------------------------------------
        # Create linear models for the risk of acquiring complications from uncomplicated Alri
        # todo - @ines -- I don't think these linear models are doing what you think it is: The 'or' statement won't be
        #  working. Can we discuss what you would like this to do?
        # todo - @ines -- do you want people to be able to get resp failure and sepsis straight away, as well as when following on from pneothoroax or lung abscess/empyem, respectively.

        self.risk_of_developing_ALRI_complications.update({
            'pneumothorax':
                LinearModel(LinearModelType.ADDITIVE,
                            0.0,
                            Predictor('ri_primary_pathogen' or 'ri_secondary_bacterial_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_pneumothorax_by_bacterial_pneumonia']).otherwise(0.0),
                            Predictor()
                            .when(
                                "ri_primary_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "(ri_disease_type == 'viral_pneumonia') ",
                                p['prob_pneumothorax_by_viral_pneumonia']),
                            Predictor()
                            .when(
                                "ri_primary_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "(ri_disease_type == 'bronchiolitis') ",
                                p['prob_pneumothorax_by_bronchiolitis'])
                            ),

            'pleural_effusion':
                LinearModel(LinearModelType.ADDITIVE,
                            0.0,
                            Predictor('ri_primary_pathogen' or 'ri_secondary_bacterial_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_pleural_effusion_by_bacterial_pneumonia']).otherwise(0.0),
                            Predictor()
                            .when(
                                "ri_primary_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "(ri_disease_type == 'viral_pneumonia') ",
                                p['prob_pleural_effusion_by_viral_pneumonia']),
                            Predictor()
                            .when(
                                "ri_primary_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "(ri_disease_type == 'bronchiolitis') ",
                                p['prob_pleural_effusion_by_bronchiolitis'])
                            ),

            'empyema':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            # todo @ines - not sure what the intention is with the below?
                            # Predictor('ri_ALRI_complications').apply(
                            #     lambda x: p['prob_pleural_effusion_to_empyema']
                            #     if x & self.ALRI_complications.element_repr('pleural_effusion') else 0)
                            ),

            'lung_abscess':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_pathogen' or 'ri_secondary_bacterial_pathogen')
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
                            Predictor('ri_primary_pathogen' or 'ri_secondary_bacterial_pathogen')
                            .when(
                                ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                                "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                                "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                                p['prob_sepsis_by_bacterial_pneumonia']).otherwise(0.0),
                            Predictor()
                            .when(
                                "ri_primary_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "(ri_disease_type == 'viral_pneumonia') ",
                                p['prob_sepsis_by_viral_pneumonia']),
                            Predictor()
                            .when(
                                "ri_primary_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                                "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                                "(ri_disease_type == 'bronchiolitis') ",
                                p['prob_sepsis_by_bronchiolitis']),
                            ),

            'meningitis':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_pathogen' or 'ri_secondary_bacterial_pathogen')
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
                            Predictor('ri_complication_peripheral_oxygen_saturation_low')
                            .when(True, p['prob_respiratory_failure_when_SpO2<93%'])
                            .otherwise(0.0)
                            ),

            'peripheral_oxygen_saturation_low':
                LinearModel(LinearModelType.ADDITIVE,
                        0.0,
                        Predictor('ri_primary_pathogen' or 'ri_secondary_bacterial_pathogen')
                        .when(
                            ".isin(['Strep_pneumoniae_PCV13', 'Strep_pneumoniae_non_PCV13', "
                            "'Hib', 'H.influenzae_non_type_b', 'Staph_aureus', 'Enterobacteriaceae', "
                            "'other_Strepto_Enterococci', 'other_bacterial_pathogens'])",
                            p['prob_hypoxia_by_bacterial_pneumonia']),
                        Predictor()
                        .when(
                            "ri_primary_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                            "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                            "(ri_disease_type == 'viral_pneumonia')",
                            p['prob_hypoxia_by_viral_pneumonia']),
                        Predictor()
                        .when(
                            "ri_primary_pathogen.isin(['RSV', 'Rhinovirus', 'HMPV', 'Parainfluenza', "
                            "'Influenza', 'Adenovirus', 'Bocavirus', 'other_viral_pathogens']) & "
                            "(ri_disease_type == 'bronchiolitis')",
                            p['prob_hypoxia_by_bronchiolitis']),
                        )
        })

        # check that equations have been declared for each complication
        assert self.complications == set(list(self.risk_of_developing_ALRI_complications.keys()))

        # Create linear models for the risk of developing severe complications
        self.risk_of_progressing_to_severe_complications.update({
            # - If the person has `pneumothorax` they may progress to `respiratory_failure`
            'respiratory_failure':
                LinearModel.multiplicative(
                    Predictor('ri_complication_pneumothorax')
                        .when(True,p['prob_pneumothorax_to_respiratory_failure'])
                        .otherwise(0.0)
                ),

            # - If the person has `lung_abscess` or `empyema` they may progress to `sepsis`
            # todo - what to when they have both or only one?? this currently means that have to have both and get the product of those probabilities:
            # todo - make the probabilties ADDDDDDDDDDDDD
            'sepsis':
                LinearModel.multiplicative(
                    Predictor('ri_complication_lung_abscess')
                        .when(True, p['prob_lung_abscess_to_sepsis'])
                        .otherwise(0.0),
                    Predictor('ri_complication_empyema')
                        .when(True, p['prob_empyema_to_sepsis'])
                        .otherwise(0.0)
                )
        })

        # =====================================================================================================
        # APPLY PROBABILITY OF SYMPTOMS TO EACH Alri DISEASE TYPE (UNCOMPLICATED AND WITH COMPLICATIONS)
        # -----------------------------------------------------------------------------------------------------
        # todo - combine these into a single function that is called to work out what symptoms someone will have, given
        #  a 'disease' and a 'complication'...?
        # Make a dict containing the probability of symptoms given acquisition of (uncomplicated) Alri,
        # by disease type
        def make_symptom_probs(disease_type):
            """helper function to make the probabilities of each symptom for each type of disease"""

            assert disease_type in self.disease_types
            index = {
                'bacterial_pneumonia': 0,
                'viral_pneumonia': 1,
                'bronchiolitis': 2,
                'fungal_pneumonia': 1  # <-- same as probabilities for viral pneumonia
            }[disease_type]

            return {
                'fever': p['prob_fever_uncomplicated_ALRI_by_disease_type'][index],
                'cough': p['prob_cough_uncomplicated_ALRI_by_disease_type'][index],
                'difficult_breathing': p['prob_difficult_breathing_uncomplicated_ALRI_by_disease_type'][index],
                'fast_breathing': p['prob_fast_breathing_uncomplicated_ALRI_by_disease_type'][index],
                'chest_indrawing': p['prob_chest_indrawing_uncomplicated_ALRI_by_disease_type'][index],
                'danger_signs': p['prob_danger_signs_uncomplicated_ALRI_by_disease_type'][index],
            }

        for disease in self.disease_types:
            self.prob_symptoms_uncomplicated_ALRI[disease] = make_symptom_probs(disease)

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
            elif complicat == 'pleural_effusion':
                return {
                    'chest_pain': p[f'prob_chest_pain_adding_from_{complicat}'],
                    'fever': p[f'prob_fever_adding_from_{complicat}'],
                    'difficult_breathing': p[f'prob_difficult_breathing_adding_from_{complicat}'],
                }
            elif complicat == 'empyema':
                return {
                    'chest_pain': p[f'prob_chest_pain_adding_from_{complicat}'],
                    'fever': p[f'prob_fever_adding_from_{complicat}'],
                    'respiratory_distress': p[f'prob_respiratory_distress_adding_from_{complicat}'],
                }
            elif complicat == 'lung_abscess':
                return {
                    'chest_pain': p[f'prob_chest_pain_adding_from_{complicat}'],
                    'fast_breathing': p[f'prob_fast_breathing_adding_from_{complicat}'],
                    'fever': p[f'prob_fever_adding_from_{complicat}'],
                }
            elif complicat == 'respiratory_failure':
                return {
                    'cyanosis': p[f'prob_cyanosis_adding_from_{complicat}'],
                    'fast_breathing': p[f'prob_fast_breathing_adding_from_{complicat}'],
                    'difficult_breathing': p[f'prob_difficult_breathing_adding_from_{complicat}'],
                    'danger_signs': p[f'prob_danger_signs_adding_from_{complicat}'],
                }
            elif complicat == 'sepsis':
                return {
                    'fever': p[f'prob_fever_adding_from_{complicat}'],
                    'fast_breathing': p[f'prob_fast_breathing_adding_from_{complicat}'],
                    'danger_signs': p[f'prob_danger_signs_adding_from_{complicat}'],

                }
            elif complicat == 'meningitis':
                return {
                    'fever': p[f'prob_fever_adding_from_{complicat}'],
                    'headache': p[f'prob_headache_adding_from_{complicat}'],
                    'danger_signs': p[f'prob_danger_signs_adding_from_{complicat}'],
                }
            elif complicat == 'peripheral_oxygen_saturation_low':
                # (No specific symptoms for 'peripheral_oxygen_saturation_low')
                return {}
            else:
                raise ValueError

        for complication in self.complications:
            self.prob_extra_symptoms_complications[complication] = add_complication_symptom_probs(complication)

        # =====================================================================================================
        # DEFINE A LINEAR MODEL FOR THE RISK OF DEATH DUE TO Alri
        # -----------------------------------------------------------------------------------------------------
        self.risk_of_death = \
            LinearModel.multiplicative(
                Predictor('ri_disease_type')
                    .when(f'fungal_pneumonia', p[f'base_death_rate_ALRI_by_fungal_pneumonia'])
                    .when(f'viral_pneumonia', p[f'base_death_rate_ALRI_by_viral_pneumonia'])
                    .when(f'bacterial_pneumonia', p[f'base_death_rate_ALRI_by_bacterial_pneumonia'])
                    .when(f'bronchiolitis', p[f'base_death_rate_ALRI_by_bronchiolitis']),
                Predictor('hv_inf')
                    .when(True, p['rr_death_ALRI_HIV']),
                Predictor('tmp_malnutrition')
                    .when(True, p['rr_death_ALRI_SAM']),
                Predictor('tmp_low_birth_weight')
                    .when(True, p['rr_death_ALRI_low_birth_weight']),
                Predictor('age_years')
                    .when('.between(1,1)', p['rr_death_ALRI_age12to23mo'])
                    .when('.between(2,4)', p['rr_death_ALRI_age24to59mo']),
                Predictor('ri_complication_sepsis')
                    .when(True, p['rr_death_ALRI_sepsis']),
                Predictor('ri_complication_respiratory_failure')
                    .when(True, p['rr_death_ALRI_respiratory_failure']),
                Predictor('ri_complication_meningitis')
                    .when(True, p['rr_death_ALRI_meningitis']),
            )

        # -----------------------------------------------------------------------------------------------------

    def determine_disease_type(self, pathogen, age):
        """Determine the disease that is caused by infection with this pathogen for a particular person at a particular
        time: from among self.disease_types"""
        # todo - @Ines -- I don't think the original version was doing what you thought it was. Can we discuss what is
        #  needed here? I have made a guess at what is useful but please check.

        p = self.parameters

        if pathogen in self.bacterial_patho:
            disease_type = 'bacterial_pneumonia'

        elif pathogen in self.viral_patho:
            if age < 2:
                disease_type = 'viral_pneumonia' if (
                    self.rng.rand() < p[f'proportion_viral_pneumonia_by_{pathogen}']
                ) else 'bronchiolitis'
            else:
                disease_type = 'viral_pneumonia'

        elif pathogen in self.fungal_patho:
            disease_type = 'fungal_pneumonia'

        else:
            raise ValueError

        assert disease_type in self.disease_types
        return disease_type

    def end_episode(self, person_id):
        """End the episode infection for a person (i.e. reset all properties to show no current infection or
        complications).
        This is called by AlriNaturalRecoveryEvent and AlriCureEvent.
        """
        df = self.sim.population.props

        # Reset properties to show no current infection:
        df.loc[person_id, [
            'ri_current_infection_status',
            'ri_primary_pathogen',
            'ri_secondary_bacterial_pathogen',
            'ri_disease_type',
            'ri_on_treatment',
            'ri_start_of_current_episode',
            'ri_scheduled_recovery_date',
            'ri_scheduled_death_date',
            'ri_ALRI_tx_start_date']
        ] = [
            False,
            np.nan,
            np.nan,
            np.nan,
            False,
            pd.NaT,
            pd.NaT,
            pd.NaT,
            pd.NaT,
        ]
        #  NB> 'ri_end_of_current_episode is not reset: this is used to control behaviour of event (incl. HSI) that
        #  may still be scheduled to occur and to prevent new infections from occuring whilst HSI from a previous
        #  episode occur.

        # Remove all existing complications
        df.loc[person_id, [f"ri_complication_{c}" for c in self.complications]] = False

        # Resolve all the symptoms immediately
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id, disease_module=self)

    def do_treatment(self, person_id, prob_of_cure):
        """Helper function that enacts the effects of a treatment to Alri caused by a pathogen.
        It will only do something if the Alri is caused by a pathogen (this module). It will not allow any effect
         if the respiratory infection is caused by another module.
        * Log the treatment date
        * Prevent any death event that may be scheduled from occuring (prior to the cure event)
        * Schedules the cure event, at which the episode is ended
        """

        df = self.sim.population.props
        person = df.loc[person_id]

        # Do nothing if the person is not alive
        if not person.is_alive:
            return

        # Do nothing if the person is not infected with a pathogen that can cause ALRI
        if not person['ri_current_infection_status']:
            return

        # Record that the person is now on treatment:
        df.loc[person_id, ['ri_on_treatment', 'ri_ALRI_tx_start_date']] = [True, self.sim.date]

        # Determine if the treatment is effective
        if prob_of_cure > self.rng.rand():

            # Cancel the death
            self.cancel_death_date(person_id)

            # Schedule the CureEvent
            cure_date = self.sim.date + DateOffset(days=self.parameters['days_between_treatment_and_cure'])
            self.sim.schedule_event(AlriCureEvent(self, person_id), cure_date)

    def cancel_death_date(self, person_id):
        """
        Cancels a scheduled date of death due to Alri for a person. This is called within do_treatment_alri function,
        and prior to the scheduling the CureEvent to prevent deaths happening in the time between
        a treatment being given and the cure event occurring.
        :param person_id:
        :return:
        """
        self.sim.population.props.at[person_id, 'ri_scheduled_death_date'] = pd.NaT

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
        assert (df.loc[not_curr_inf, 'ri_end_of_current_episode'].isna() |
                (df.loc[not_curr_inf, 'ri_end_of_current_episode'] <= self.sim.date) |
                ((df.loc[not_curr_inf, 'ri_end_of_current_episode'] - self.sim.date).dt.days <= self.max_duration_of_epsiode.days)
                ).all()

        # For those with no current infection, there should be no treatment
        assert not df.loc[not_curr_inf, 'ri_on_treatment'].any()
        assert df.loc[not_curr_inf, 'ri_ALRI_tx_start_date'].isna().all()

        # For those with no current infection, there should be no complications
        assert not df.loc[not_curr_inf,
            [f"ri_complication_{c}" for c in self.complications]
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
        assert ((~df.loc[curr_inf, 'ri_scheduled_recovery_date'].isna()) | (~df.loc[curr_inf, 'ri_scheduled_death_date'].isna())).all()
        assert (df.loc[curr_inf,'ri_scheduled_recovery_date'].isna() != df.loc[curr_inf, 'ri_scheduled_death_date'].isna()).all()

        #  If that primary pathogen is bacterial then there should be np.nan for secondary_bacterial_pathogen:
        assert df.loc[
            curr_inf & df['ri_primary_pathogen'].isin(self.bacterial_patho), 'ri_secondary_bacterial_pathogen'
        ].isna().all()

        # If person is on treatment, they should have a treatment start date
        assert (df.loc[curr_inf, 'ri_on_treatment'] != df.loc[curr_inf, 'ri_ALRI_tx_start_date'].isna()).all()

    def random_date(self, start, end):
        """Generate a random date between `start` and `end` - sampling with precision of the day."""
        return start + DateOffset(days=self.rng.randint(0, (end - start).days))

    def impose_symptoms_for_complication(self, complication, person_id):
        """
        Assign clinical symptoms in respect of a complication.
        """
        for symptom, prob in self.prob_extra_symptoms_complications[complication].items():
            if self.rng.rand() < prob:
                self.sim.modules['SymptomManager'].change_symptom(
                    person_id=person_id,
                    symptom_string=symptom,
                    add_or_remove='+',
                    disease_module=self,
                )

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
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=2))
        self.fraction_of_a_year_until_next_polling_event = self.compute_fraction_of_year_between_polling_event()

    def compute_fraction_of_year_between_polling_event(self):
        """Compute fraction of a year that elapses between polling event. This is used to adjust the risk of infection"""
        return (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'Y')

    def apply(self, population):
        """Determine who will become infected and schedule for them an AlriComplicationOnsetEvent
        """
        df = population.props
        m = self.module
        p = self.module.parameters
        rng = self.module.rng

        # Compute the incidence rate for each person getting Alri and then convert into a probability
        # getting all children that do not currently have an Alri episode (never had or last episode resolved)
        mask_could_get_new_alri_event = (
            df['is_alive'] &
            (df['age_years'] < 5) &
            ~df['ri_current_infection_status'] &
            ((df['ri_end_of_current_episode'] < self.sim.date) | pd.isnull(df['ri_end_of_current_episode']))
        )

        # Compute the incidence rate for each person acquiring Alri
        inc_of_acquiring_alri = pd.DataFrame(index=df.loc[mask_could_get_new_alri_event].index)
        for pathogen in m.pathogens:
            inc_of_acquiring_alri[pathogen] = m.incidence_equations_by_pathogen[pathogen] \
                .predict(df.loc[mask_could_get_new_alri_event])

        probs_of_acquiring_pathogen = 1 - np.exp(
            -inc_of_acquiring_alri * self.fraction_of_a_year_until_next_polling_event
        )

        # Sample to find outcomes:
        outcome = self.sample_outcome(probs_of_acquiring_pathogen)

        # For persons that will become infected with a particular pathogen:
        for person_id, pathogen in outcome.items():
            #  Create the event for the onset of infection:
            self.sim.schedule_event(
                event=AlriIncidentCase(
                    module=self.module,
                    person_id=person_id,
                    pathogen=pathogen,
                ),
                date=m.random_date(self.sim.date, self.sim.date + self.frequency - pd.DateOffset(days=1))
            )

    def sample_outcome(self, df):
        """Helper function to randoly sample outcomes from a set of probabilities.
        Each row in the df is a person and each column is an event that may happen to the person.
        The probabilities of each event are assumed to be independent but mutually exlusive."""

        # todo - this needs checking
        assert (df.sum(axis=1) <= 1.0).all(), "Probabilities across columns cannot sum to more than 1.0"

        # Compare uniform deviate to cumulative sum across columns, after including a "null" column (for no event).
        df['_'] = 1.0 - df.sum(axis=1)  # add implied "none of these events" category
        cumsum = df.cumsum(axis=1)
        draws = pd.Series(self.module.rng.rand(len(cumsum)), index=cumsum.index)
        y = cumsum.gt(draws, axis=0)
        outcome = y.idxmax(axis=1)

        # return as a dict only in those cases where the outcome is one of the events.
        return outcome.loc[~(outcome == '_')].to_dict()


class AlriIncidentCase(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of the infection that causes Alri. It is scheduled by the AlriPollingEvent.
    """

    def __init__(self, module, person_id, pathogen):
        super().__init__(module, person_id=person_id)
        self.pathogen = pathogen

    def apply(self, person_id):
        """
        * Determines the disease and complications associated with this case
        * Updates all the properties so that they pertain to this current episode of Alri
        * Imposes the symptoms
        * Schedules relevant natural history event {(AlriWithPulmonaryComplicationsEvent) and
        (AlriWithSevereComplicationsEvent) (either AlriNaturalRecoveryEvent or AlriDeathEvent)}
        * Updates the counters in the log accordingly.
        """
        df = self.sim.population.props  # shortcut to the dataframe
        person = df.loc[person_id]
        m = self.module
        rng = self.module.rng
        p = m.parameters

        # The event should not run if the person is not currently alive:
        if not person['is_alive']:
            return

        # 0) Add this case to the counter:
        self.module.logging_event.new_case(age=person['age_years'], pathogen=self.pathogen)

        # 1) Determines the disease and complications associated with this case

        # ----------------- Determine the Alri disease type for this case -----------------
        disease = m.determine_disease_type(age=person['age_years'], pathogen=self.pathogen)

        # ----------------- Determine if there is a secondary bacterial infection  -----------------
        if self.pathogen in self.module.viral_patho:
            # pathogen is viral: determine if there is bacterial co-infection
            if m.prob_secondary_bacterial_infection.predict(df.loc[[person_id]], rng):
                bacterial_coinfection = \
                    rng.choice(list(m.bacterial_patho), p=p['proportion_bacterial_coinfection_pathogen'])
                # update to co-infection property
                # (a viral infection with a bacterial secondary infection is classified as bacterial_pneumonia)
                disease = 'bacterial_pneumonia'
            else:
                bacterial_coinfection = np.nan

        else:
            bacterial_coinfection = np.nan

        # ----------------------- Duration of the Alri event -----------------------
        duration_in_days_of_alri = rng.randint(1, 8)  # assumes uniform interval around mean duration with range 4 days
        # todo - make this a parameter and inform with data

        # Date for outcome (either recovery or death) with uncomplicated Alri
        date_of_outcome = self.module.sim.date + DateOffset(days=duration_in_days_of_alri)

        # Define 'episode end' date. This the date when this episode ends. It is the last possible data that any HSI
        # could affect this episode.
        episode_end = date_of_outcome + m.max_duration_of_epsiode

        # 2) Update the properties in the dataframe:
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
               )
        ] = (
            True,
            self.pathogen,
            bacterial_coinfection,
            disease,
            False,
            self.sim.date,
            pd.NaT,
            pd.NaT,
            episode_end,
            pd.NaT
        )

        # ----------------------------------- clinical symptoms -----------------------------------
        # impose clinical symptoms for new uncomplicated Alri
        self.impose_symptoms_for_uncomplicated_disease(person_id=person_id, disease=disease)

        # COMPLICATIONS -----------------------------------------------------------------------------------------
        self.impose_complications(person_id=person_id, date_of_outcome=date_of_outcome)

        # Determine outcome: death or recovery
        if m.risk_of_death.predict(df.loc[[person_id]], rng):
            self.sim.schedule_event(AlriDeathEvent(self.module, person_id), date_of_outcome)
            df.loc[person_id, ['ri_scheduled_death_date', 'ri_scheduled_recovery_date']] = [date_of_outcome, pd.NaT]
        else:
            self.sim.schedule_event(AlriNaturalRecoveryEvent(self.module, person_id), date_of_outcome)
            df.loc[person_id, ['ri_scheduled_recovery_date', 'ri_scheduled_death_date']] = [date_of_outcome, pd.NaT]

    def impose_symptoms_for_uncomplicated_disease(self, person_id, disease):
        """
        Assigns clinical symptoms to uncomplicated Alri. The probabilities of different symptoms are specific to the
        type of Alri disease. These symptoms are not set to auto-resolve
        """
        rng = self.module.rng
        m = self.module

        # ----------------------- Allocate symptoms to onset of Alri ----------------------
        prob_symptoms_uncomplicated_alri = m.prob_symptoms_uncomplicated_ALRI[disease]

        for symptom, prob in prob_symptoms_uncomplicated_alri.items():
            if rng.rand() < prob:
                m.sim.modules['SymptomManager'].change_symptom(
                    person_id=person_id,
                    symptom_string=symptom,
                    add_or_remove='+',
                    disease_module=m,
                )

    def impose_complications(self, person_id, date_of_outcome):
        """Impose the set of complications for this person, and onset these all instantanesouly."""
        # todo - make this go faster by doing one .loc and passing it all updates to the df at the same time.

        df = self.sim.population.props
        rng = self.module.rng
        m = self.module

        # Order complications with "peripheral_oxygen_saturation_low" first (risk of other complications depends on it).
        complications_ordered = ['peripheral_oxygen_saturation_low'] + list(
            m.complications - {'peripheral_oxygen_saturation_low'})
        assert set(complications_ordered) == set(m.complications)
        assert len(complications_ordered) == len(m.complications)

        for complication in complications_ordered:
            if m.risk_of_developing_ALRI_complications[complication].predict(df.loc[[person_id]], rng):
                df.at[person_id, f"ri_complication_{complication}"] = True
                m.impose_symptoms_for_complication(person_id=person_id, complication=complication)

        # Schedule event for a delayed onset of a complication, where appropriate:
        date_of_onset_delayed_complications = m.random_date(self.sim.date, date_of_outcome)
        # todo @ines this choose a date for onset of syestemtic complications randomly between date of infection and date of
        #  outcome - is this what you wanted?
        for complication, lm in m.risk_of_progressing_to_severe_complications.items():
            if lm.predict(df.loc[[person_id]], rng):
                self.sim.schedule_event(
                    AlriDelayedOnsetComplication(person_id=person_id,
                                                 complication=complication,
                                                 module=m),
                    date_of_onset_delayed_complications
                )


class AlriDelayedOnsetComplication(Event, IndividualScopeEventMixin):
    """This Event is for the delayed onset of complications from Alri. It is applicable only to a subset of
    complications."""

    def __init__(self, module, person_id, complication):
        super().__init__(module, person_id=person_id)

        assert complication in ['sepsis', 'respiratory_failure'], \
            'Delayed onset is only possible for certain complications'
        self.complication = complication

    def apply(self, person_id):
        """Apply the complication, if the person is still infected and not on treatment"""
        df = self.sim.population.props  # shortcut to the dataframe
        person = df.loc[person_id]
        m = self.module

        # Do nothing if person is not alive:
        if not df.at[person_id, 'is_alive']:
            return

        # If person is infected, not on treatment and does not already have the complication, add this complication:
        if (person['ri_current_infection_status'] and
            ~person['ri_on_treatment'] and
            ~person[f'ri_complication_{systemic_complication}']
        ):
            df.at[person_id, f'ri_complication_{systemic_complication}'] = True
            m.impose_symptoms_for_complication(complication=systemic_complication, person_id=person_id)


class AlriNaturalRecoveryEvent(Event, IndividualScopeEventMixin):
    """
    This is the Natural Recovery event. It is scheduled by the AlriIncidentCase Event for someone who will recover
    from the infection even if no care received.
    It calls the 'end_infection' function.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]

        # The event should not run if the person is not currently alive
        if not person['is_alive']:
            return

        # Check if person should really recover:
        if (
            person['ri_current_infection_status'] and
            (person['ri_scheduled_recovery_date'] == self.sim.date) and
            pd.isnull(person['ri_scheduled_death_date'])
        ):
            # Log the recovery
            self.module.logging_event.new_recovered_case(
                age=person['age_years'],
                pathogen=person['ri_primary_pathogen']
            )

            # Do the episode:
            self.module.end_episode(person_id=person_id)


class AlriCureEvent(Event, IndividualScopeEventMixin):
    """
       This is the cure event. It is scheduled by an HSI treatment event.
       It enacts the actual "cure" of the person that is caused (after some delay) by the treatment administered.
       """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]

        # The event should not run if the person is not currently alive
        if not person['is_alive']:
            return

        # Check if person should really be cured:
        if (
            person['ri_current_infection_status']
        ):
            # Log the cure:
            pathogen = person['ri_primary_pathogen']
            self.module.logging_event.new_cured_case(
                age=df.at[person_id, 'age_years'],
                pathogen=pathogen
            )

            # End the episode:
            self.module.end_episode(person_id=person_id)


class AlriDeathEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the death of someone that is caused by the infection with a pathogen that causes Alri.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        person = df.loc[person_id]

        # The event should not run if the person is not currently alive
        if not person['is_alive']:
            return

        # Check if person should really die of Alri:
        if (
            person['ri_current_infection_status'] and
            (person['ri_scheduled_death_date'] == self.sim.date) and
            pd.isnull(person['ri_scheduled_recovery_date'])
        ):

            # Do the death:
            pathogen = person['ri_primary_pathogen']
            self.module.sim.modules['Demography'].do_death(
                individual_id=person_id,
                cause='ALRI_' + pathogen,
                originating_module=self.module
            )

            # Log the death in the Alri logging system
            self.module.logging_event.new_death(
                age=df.at[person_id, 'age_years'],
                pathogen=pathogen
            )


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ==================================== HEALTH SYSTEM INTERACTION EVENTS ====================================
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class HSI_Alri_GenericTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    # todo - @Ines -- here is a template for the HSI interaction events. It just shows the checks to use each time.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Alri_GenericTreatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Do the treatment"""

        df = self.sim.population.props
        person = df.loc[person_id]

        # Exit if the person is not alive or is not currently infected:
        if not (person['is_alive'] and person['ri_current_infection_status']):
            return

        # For example, say that probability of cure = 1.0
        prob_of_cure = 1.0
        self.module.do_treatment(person_id=person_id, prob_of_cure=prob_of_cure)


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
        age_grps = {**{0: "0", 1: "1", 2: "2-4", 3: "2-4", 4: "2-4"}, **{x: "5+" for x in range(5, 100)}}

        self.trackers = dict()
        self.trackers['incident_cases'] = Tracker(age_grps=age_grps, pathogens=self.module.pathogens)
        self.trackers['recovered_cases'] = Tracker(age_grps=age_grps, pathogens=self.module.pathogens)
        self.trackers['cured_cases'] = Tracker(age_grps=age_grps, pathogens=self.module.pathogens)
        self.trackers['deaths'] = Tracker(age_grps=age_grps, pathogens=self.module.pathogens)

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


# ---------------------------------------------------------------------------------------------------------
#   DEBUGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class AlriCheckPropertiesEvent(RegularEvent, PopulationScopeEventMixin):
    """This event runs daily and checks properties are in the right configuration. Only use whilst debugging!
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=1))

    def apply(self, population):
        self.module.check_properties()


class AlriIndividualLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This Event logs the daily occurrence to a single individual child.
    """

    def __init__(self, module):
        # This logging event to occur every day
        super().__init__(module, frequency=DateOffset(days=1))

        # Find the person to log: the first under-five-year-old in the dataframe
        df = self.sim.population.props
        under5s = df.loc[df.is_alive & (df['age_years'] < 5)]
        self.person_id = under5s.index[0] if len(under5s) else None

    def apply(self, population):
        """Log all properties for this module"""
        if self.person_id is not None:
            df = self.sim.population.props
            logger.info(
                key='log_individual',
                data=df.loc[self.person_id, self.module.PROPERTIES.keys()].to_dict(),
                description='Properties for one person (the first under-five-year-old in the dataframe), each day.'
            )
