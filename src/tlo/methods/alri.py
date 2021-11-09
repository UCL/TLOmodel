"""
Childhood Acute Lower Respiratory Infection Module

Overview
--------
Individuals are exposed to the risk of infection by a pathogen (and potentially also with a secondary bacterial
infection) that can cause one of several type of acute lower respiratory infection (Alri).
The disease is manifested as either viral pneumonia, bacterial pneumonia or bronchiolitis.

During an episode (prior to recovery - either naturally or cured with treatment), symptom are manifest and there may be
complications (e.g. local pulmonary complication: pleural effusuion, empyema, lung abscess, pneumothorax; and/or
systemic complications: sepsis, meningitis, and respiratory failure). There is a risk that some of these complications
are onset after a delay.

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

Questions
----------

#1:
'daly_very_severe_ALRI' is never used: is that right?

#2:
There are no probabilities describing the risk of onset of the complication "empyema". There is the risk of empyema
being delayed onset (i.e. prob_pleural_effusion_to_empyema) and also the risk that "empyema" leads to sepsis
(prob_empyema_to_sepsis). However, for simplicity  we only have two phases of complication (initial and delayed), so
this is not represented at the moment. I am not sure if it should be or not.

#3:
The risk of death is not affected by delayed-onset complications: should it be?

#4:
Check that every parameter is used and remove those not used (from definition and from excel). I think that are several
not being used (e.g prob_respiratory_failure_to_multiorgan_dysfunction and r_progress_to_severe_ALRI) -- perhaps left
 over from an earlier version? Also, please could you tidy-up ResourceFile_Alri.xlsx? If there are sheets that need to
  remain in there, it would be good if you can explain what each is for on the sheet (and delete anything that is not
   needed).

#5:
Is it right that 'danger_signs' is an indepednet symptom? It seems like this is something that is determined in the
 course of a diagnosis (like in diarrhoea module).
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
        'Hiv',
        'Lifestyle',
        'NewbornOutcomes',
        'SymptomManager',
        'Wasting',
    }

    ADDITIONAL_DEPENDENCIES = {'Epi'}

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
            'Adenovirus',
            'Bocavirus',
            'other_viral_pathogens'
            # <-- Coronaviruses NL63, 229E OC43 and HKU1, Cytomegalovirus, Parechovirus/Enterovirus
        ],
        'bacterial': [
            'Strep_pneumoniae_PCV13',      # <--  risk of acquisition is affected by the pneumococcal vaccine
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
        'fungal': [
            'P.jirovecii'
        ]
    }

    # Make set of all pathogens combined:
    all_pathogens = set(chain.from_iterable(pathogens.values()))

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
    disease_types = {
        'pneumonia', 'bronchiolitis', 'other_alri'
    }

    # Declare the Alri complications:
    complications = {'pneumothorax',
                     'pleural_effusion',
                     'empyema',
                     'lung_abscess',
                     'sepsis',
                     'meningitis',
                     'respiratory_failure',
                     'hypoxia'  # <-- Low implies Sp02<93%'
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

        # Risk factors for incidence infection -----
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

        # Probability of bacterial co- / secondary infection -----
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

        # Risk of death parameters -----
        'base_death_rate_ALRI_by_pneumonia':
            Parameter(Types.REAL,
                      'baseline death rate from pneumonia, for those aged 0-11 months'
                      ),
        'base_death_rate_ALRI_by_bronchiolitis':
            Parameter(Types.REAL,
                      'baseline death rate from bronchiolitis, base age 0-11 months'
                      ),
        'base_death_rate_ALRI_by_other_alri':
            Parameter(Types.REAL,
                      'baseline death rate from other lower respiratory infections, base age 0-11 months'
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
                      'list of probabilities of having fever by pneumonia, bronchiolitis, and other_alri'
                      ),
        'prob_cough_uncomplicated_ALRI_by_disease_type':
            Parameter(Types.LIST,
                      'list of probabilities of having cough by pneumonia, bronchiolitis, and other_alri'
                      ),
        'prob_difficult_breathing_uncomplicated_ALRI_by_disease_type':
            Parameter(Types.LIST,
                      'list of probabilities of difficult breathing by '
                      'pneumonia, bronchiolitis, and other_alri'
                      ),
        'prob_fast_breathing_uncomplicated_ALRI_by_disease_type':
            Parameter(Types.LIST,
                      'list of probabilities of fast breathing by '
                      'pneumonia, bronchiolitis, and other_alri'
                      ),
        'prob_chest_indrawing_uncomplicated_ALRI_by_disease_type':
            Parameter(Types.LIST,
                      'list of probabilities of chest indrawing by '
                      'pneumonia, bronchiolitis, and other_alri'
                      ),
        'prob_danger_signs_uncomplicated_ALRI_by_disease_type':
            Parameter(Types.LIST,
                      'list of probabilities of danger signs by '
                      'pneumonia, bronchiolitis, and other_alri'
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

        # Second round of signs/symptoms added from each complication
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

        # Pleural effusion -----------
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

        # Hypoxaemic Respiratory failure --------------
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

        # Meningitis ------------
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

        # Parameters governing the effects of vaccine ----------------
        'rr_infection_strep_with_pneumococcal_vaccine':
            Parameter(Types.REAL,
                      'relative risk of acquiring the pathogen=="Strep_pneumoniae_PCV13" if has had all doses of '
                      'pneumonococcal vaccine (i.e., `va_pneumo_all_doses`==True), either as a primary pathogen or'
                      'as a secondary bacterial coinfection.'
                      ),
        'rr_infection_hib_haemophilus_vaccine':
            Parameter(Types.REAL,
                      'relaitve risk of acquiring the pathogen=="Hib" if has had all doses of Haemophilus vaccine'
                      ' (i.e., `va_hib_all_doses`==Tue)'
                      ),

        # Parameters governing treatment effectiveness and assoicated behaviours ----------------
        'days_between_treatment_and_cure':
            Parameter(Types.INT, 'number of days between any treatment being given in an HSI and the cure occurring.'
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
                     categories=list(disease_types)
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

        # Maximum duration of an episode (beginning with inection and ending with recovery)
        self.max_duration_of_episode = None

        # dict to hold the DALY weights
        self.daly_wts = dict()

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
            'fever', 'cough', 'difficult_breathing', 'fast_breathing', 'chest_indrawing', 'chest_pain', 'cyanosis',
            'respiratory_distress', 'danger_signs'
        }

        for symptom_name in all_symptoms:
            if symptom_name not in self.sim.modules['SymptomManager'].generic_symptoms:
                self.sim.modules['SymptomManager'].register_symptom(
                    Symptom(name=symptom_name)
                    # (associates the symptom with the 'average' healthcare seeking)
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
        self.max_duration_of_episode = DateOffset(days=self.parameters['days_between_treatment_and_cure'])

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

        # ---- Internal values ----
        df.loc[child_id, ['ri_start_of_current_episode',
                          'ri_scheduled_recovery_date',
                          'ri_scheduled_death_date',
                          'ri_end_of_current_episode']] = pd.NaT

    def report_daly_values(self):
        """Report DALY incurred in the population in the last month due to ALRI"""
        df = self.sim.population.props

        total_daly_values = pd.Series(data=0.0, index=df.index[df.is_alive])
        total_daly_values.loc[
            self.sim.modules['SymptomManager'].who_has('fast_breathing')] = self.daly_wts['daly_non_severe_ALRI']
        total_daly_values.loc[
            self.sim.modules['SymptomManager'].who_has('danger_signs')] = self.daly_wts['daly_severe_ALRI']

        # Split out by pathogen that causes the Alri
        dummies_for_pathogen = pd.get_dummies(df.loc[total_daly_values.index, 'ri_primary_pathogen'], dtype='float')
        daly_values_by_pathogen = dummies_for_pathogen.mul(total_daly_values, axis=0)

        # add prefix to label according to the name of the causes of disability declared
        daly_values_by_pathogen = daly_values_by_pathogen.add_prefix('ALRI_')
        return daly_values_by_pathogen

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
        #  NB. 'ri_end_of_current_episode is not reset: this is used to prevent new infections from occuring whilst
        #  HSI from a previous episode may still be scheduled to occur.

        # Remove all existing complications
        df.loc[person_id, [f"ri_complication_{c}" for c in self.complications]] = False

        # Resolve all the symptoms immediately
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id, disease_module=self)

    def do_treatment(self, person_id, prob_of_cure):
        """Helper function that enacts the effects of a treatment to Alri caused by a pathogen.
        It will only do something if the Alri is caused by a pathogen (this module).
        * Log the treatment
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
        """Cancels a scheduled date of death due to Alri for a person. This is called within do_treatment_alri function,
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

    def impose_symptoms_for_complication(self, complication, person_id):
        """Impose symptoms for a complication."""
        symptoms = self.models.symptoms_for_complication(complication=complication)
        for symptom in symptoms:
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove='+',
                disease_module=self,
            )


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

        def make_scaled_linear_model_for_incidence(patho):
            """Makes the unscaled linear model with default intercept of 1. Calculates the mean incidents rate for
            0-year-olds and then creates a new linear model with adjusted intercept so incidence in 0-year-olds
            matches the specified value in the model when averaged across the population. This will return an unscaled
            linear model if there are no 0-year-olds in the population.
            """

            def make_naive_linear_model(_patho, intercept=1.0):
                """Make the linear model based exactly on the parameters specified"""

                base_inc_rate = f'base_inc_rate_ALRI_by_{_patho}'
                return LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept,
                    Predictor(
                        'age_years',
                        conditions_are_mutually_exclusive=True,
                        conditions_are_exhaustive=True,
                    )
                    .when(0, p[base_inc_rate][0])
                    .when(1, p[base_inc_rate][1])
                    .when('.between(2,4)', p[base_inc_rate][2])
                    .when('> 4', 0.0),
                    Predictor('li_no_access_handwashing').when(False, p['rr_ALRI_HHhandwashing']),
                    Predictor('li_wood_burn_stove').when(False, p['rr_ALRI_indoor_air_pollution']),
                    Predictor('hv_inf').when(True, p['rr_ALRI_HIV_untreated']),
                    Predictor('un_clinical_acute_malnutrition').when('SAM', p['rr_ALRI_underweight']),
                    Predictor('nb_breastfeeding_status').when('exclusive', 1.0)
                                                        .otherwise(p['rr_ALRI_not_excl_breastfeeding'])
                )

            zero_year_olds = df.is_alive & (df.age_years == 0)
            unscaled_lm = make_naive_linear_model(patho)

            # If not 0 year-olds then cannot do scaling, return unscaled linear model
            if sum(zero_year_olds) == 0:
                return unscaled_lm

            # If some 0 year-olds then can do scaling:
            target_mean = p[f'base_inc_rate_ALRI_by_{patho}'][0]
            actual_mean = unscaled_lm.predict(df.loc[zero_year_olds]).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean)
            scaled_lm = make_naive_linear_model(patho, intercept=scaled_intercept)

            # check by applying the model to mean incidence of 0-year-olds
            assert (target_mean - scaled_lm.predict(df.loc[zero_year_olds]).mean()) < 1e-10
            return scaled_lm

        for patho in self.module.all_pathogens:
            self.incidence_equations_by_pathogen[patho] = make_scaled_linear_model_for_incidence(patho)

    def compute_risk_of_aquisition(self, pathogen, df):
        """Compute the risk of a pathogen, using the linear model created and the df provided"""
        p = self.p
        lm = self.incidence_equations_by_pathogen[pathogen]

        # run linear model to get baseline risk
        baseline = lm.predict(df)

        # apply the reduced risk of acquisition for those vaccinated
        if pathogen == "Strep_pneumoniae_PCV13":
            baseline.loc[df['va_pneumo_all_doses']] *= p['rr_infection_strep_with_pneumococcal_vaccine']
        elif pathogen == "Hib":
            baseline.loc[df['va_hib_all_doses']] *= p['rr_infection_hib_haemophilus_vaccine']

        return baseline

    def determine_disease_type_and_secondary_bacterial_coinfection(self, pathogen, age, va_pneumo_all_doses):
        """Determines:
         * the disease that is caused by infection with this pathogen (from among self.disease_types)
         * if there is a bacterial coinfection associated that will cause the dominant disease.

         Note that the disease_type is 'bacterial_pneumonia' if primary pathogen is viral and there is a secondary
         bacterial coinfection.
         """
        p = self.p

        if pathogen in set(self.module.pathogens['bacterial']).union(self.module.pathogens['fungal']):
            disease_type = 'pneumonia'
            bacterial_coinfection = np.nan

        elif pathogen in self.module.pathogens['viral']:
            if (age >= 2) or (p[f'proportion_viral_pneumonia_by_{pathogen}'] > self.rng.rand()):
                disease_type = 'pneumonia'
                if p['prob_viral_pneumonia_bacterial_coinfection'] > self.rng.rand():
                    bacterial_coinfection = self.secondary_bacterial_infection(va_pneumo_all_doses)
                else:
                    bacterial_coinfection = np.nan
            else:
                # likely to be 'bronchiolitis' (if there is no secondary bacterial infection)
                if p['prob_secondary_bacterial_infection_in_bronchiolitis'] > self.rng.rand():
                    bacterial_coinfection = self.secondary_bacterial_infection(va_pneumo_all_doses)
                    disease_type = 'pneumonia' if pd.notnull(bacterial_coinfection) else 'bronchiolitis'
                else:
                    bacterial_coinfection = np.nan
                    disease_type = 'bronchiolitis'
        else:
            raise ValueError('Pathogen is not recognised.')

        assert disease_type in self.module.disease_types
        assert bacterial_coinfection in (self.module.pathogens['bacterial'] + [np.nan])

        return disease_type, bacterial_coinfection

    def secondary_bacterial_infection(self, va_pneumo_all_doses):
        """Determine which specific bacterial pathogen causes a secondary coinfection, or if there is no secondary
        bacterial infection (due to the effects of the pneumococcal vaccine).
        """
        p = self.p

        # get probability of bacterial coinfection with each pathogen
        probs = dict(zip(
            self.module.pathogens['bacterial'],
            p['proportion_bacterial_coinfection_pathogen']))

        # Edit the probability that the coinfection will be of `Strep_pneumoniae_PCV13` if the person has had
        #  the pneumococcal vaccine:
        if va_pneumo_all_doses:
            probs['Strep_pneumoniae_PCV13'] *= self.p['rr_infection_strep_with_pneumococcal_vaccine']

        # Add in the probability that there is none (to ensure that all probabilities sum to 1.0)
        probs['none'] = 1.0 - sum(probs.values())

        # return the random selection of bacterial coinfection (including possibly np.nan for 'none')
        outcome = self.rng.choice(list(probs.keys()), p=list(probs.values()))
        return outcome if not 'none' else np.nan

    def complications(self, person_id):
        """Determine the set of complication for this person"""
        p = self.p
        person = self.module.sim.population.props.loc[person_id]

        primary_path_is_bacterial = person['ri_primary_pathogen'] in self.module.pathogens['bacterial']
        primary_path_is_viral = person['ri_primary_pathogen'] in self.module.pathogens['viral']
        has_secondary_bacterial_inf = pd.notnull(person.ri_secondary_bacterial_pathogen)
        disease_type = person['ri_disease_type']

        probs = defaultdict(float)

        if primary_path_is_bacterial or has_secondary_bacterial_inf:
            for c in ['pneumothorax', 'pleural_effusion', 'lung_abscess', 'sepsis', 'meningitis', 'hypoxia']:
                probs[c] += p[f"prob_{c}_by_bacterial_pneumonia"]

        if primary_path_is_viral and (disease_type == 'pneumonia'):
            for c in ['pneumothorax', 'pleural_effusion', 'sepsis', 'hypoxia']:
                probs[c] += p[f"prob_{c}_by_viral_pneumonia"]

        if primary_path_is_viral and (disease_type == 'bronchiolitis'):
            for c in ['pneumothorax', 'pleural_effusion', 'sepsis', 'hypoxia']:
                probs[c] += p[f"prob_{c}_by_bronchiolitis"]

        # determine which complications are onset:
        complications = {c for c, p in probs.items() if p > self.rng.rand()}

        # Determine if repiratory failure should also be onset (this depends on whether or not hypoxia is onset):
        if 'hypoxia' in complications:
            if p['prob_respiratory_failure_when_SpO2<93%'] > self.rng.rand():
                complications.add('respiratory_failure')

        return complications

    def delayed_complications(self, person_id):
        """Determine the set of delayed complications"""
        p = self.p
        person = self.module.sim.population.props.loc[person_id]

        complications = set()

        # 'respiratory_failure':
        if person['ri_complication_pneumothorax'] and ~person['ri_complication_respiratory_failure']:
            if p['prob_pneumothorax_to_respiratory_failure'] > self.rng.rand():
                complications.add('respiratory_failure')

        # 'sepsis'
        if ~person['ri_complication_sepsis']:
            prob_sepsis = 0.0
            if person['ri_complication_lung_abscess']:
                prob_sepsis += p['prob_lung_abscess_to_sepsis']
            if person['ri_complication_empyema']:
                prob_sepsis += p['prob_empyema_to_sepsis']

            if prob_sepsis > self.rng.rand():
                complications.add('sepsis')

        return complications

    def symptoms_for_disease(self, disease_type):
        """Determine set of symptom (before complications) for a given instance of disease"""
        p = self.p

        assert disease_type in self.module.disease_types
        index = {
            'pneumonia': 0,
            'bronchiolitis': 1,
            'other_alri': 2
        }[disease_type]

        probs = {
            symptom: p[f'prob_{symptom}_uncomplicated_ALRI_by_disease_type'][index]
            for symptom in [
                'fever', 'cough', 'difficult_breathing', 'fast_breathing', 'chest_indrawing', 'danger_signs']
        }

        # determine which symptoms are onset:
        symptoms = {s for s, p in probs.items() if p > self.rng.rand()}

        return symptoms

    def symptoms_for_complication(self, complication):
        """Probability of each symptom for a person given a particular complication"""
        p = self.p

        if complication == 'pneumothorax':
            probs = {
                'chest_pain': p[f'prob_chest_pain_adding_from_{complication}'],
                'cyanosis': p[f'prob_cyanosis_adding_from_{complication}'],
                'difficult_breathing': p[f'prob_difficult_breathing_adding_from_{complication}']
            }
        elif complication == 'pleural_effusion':
            probs = {
                'chest_pain': p[f'prob_chest_pain_adding_from_{complication}'],
                'fever': p[f'prob_fever_adding_from_{complication}'],
                'difficult_breathing': p[f'prob_difficult_breathing_adding_from_{complication}']
            }
        elif complication == 'empyema':
            probs = {
                'chest_pain': p[f'prob_chest_pain_adding_from_{complication}'],
                'fever': p[f'prob_fever_adding_from_{complication}'],
                'respiratory_distress': p[f'prob_respiratory_distress_adding_from_{complication}']
            }
        elif complication == 'lung_abscess':
            probs = {
                'chest_pain': p[f'prob_chest_pain_adding_from_{complication}'],
                'fast_breathing': p[f'prob_fast_breathing_adding_from_{complication}'],
                'fever': p[f'prob_fever_adding_from_{complication}']
            }
        elif complication == 'respiratory_failure':
            probs = {
                'cyanosis': p[f'prob_cyanosis_adding_from_{complication}'],
                'fast_breathing': p[f'prob_fast_breathing_adding_from_{complication}'],
                'difficult_breathing': p[f'prob_difficult_breathing_adding_from_{complication}'],
                'danger_signs': p[f'prob_danger_signs_adding_from_{complication}']
            }
        elif complication == 'sepsis':
            probs = {
                'fever': p[f'prob_fever_adding_from_{complication}'],
                'fast_breathing': p[f'prob_fast_breathing_adding_from_{complication}'],
                'danger_signs': p[f'prob_danger_signs_adding_from_{complication}']
            }
        elif complication == 'meningitis':
            probs = {
                'fever': p[f'prob_fever_adding_from_{complication}'],
                'headache': p[f'prob_headache_adding_from_{complication}'],
                'danger_signs': p[f'prob_danger_signs_adding_from_{complication}']
            }
        elif complication == 'hypoxia':
            # (No specific symptoms for 'hypoxia')
            return set()
        else:
            raise ValueError

        # determine which symptoms are onset:
        symptoms = {s for s, p in probs.items() if p > self.rng.rand()}

        return symptoms

    def death(self, person_id):
        """Determine if person will die from Alri. Returns True/False"""
        p = self.p
        person = self.module.sim.population.props.loc[person_id]

        disease_type = person['ri_disease_type']

        # Baseline risk:
        risk = p[f"base_death_rate_ALRI_by_{disease_type}"]

        # The effect of age:
        if person['age_years'] == 1:
            risk *= p['rr_death_ALRI_age12to23mo']
        elif 2 <= person['age_years'] <= 4:
            risk *= p['rr_death_ALRI_age24to59mo']

        # The effect of complications:
        if person['ri_complication_sepsis']:
            risk *= p['rr_death_ALRI_sepsis']

        if person['ri_complication_respiratory_failure']:
            risk *= p['rr_death_ALRI_respiratory_failure']

        if person['ri_complication_meningitis']:
            risk *= p['rr_death_ALRI_meningitis']

        # The effect of factors defined in other modules:
        if person['hv_inf']:
            risk *= p['rr_death_ALRI_HIV']

        if person['un_clinical_acute_malnutrition'] == 'SAM':
            risk *= p['rr_death_ALRI_SAM']

        if (
            person['nb_low_birth_weight_status'] in
            ['extremely_low_birth_weight', 'very_low_birth_weight', 'low_birth_weight']
        ):
            risk *= p['rr_death_ALRI_low_birth_weight']

        return risk > self.rng.rand()


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class AlriPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the main event that runs the acquisition of pathogens that cause Alri.
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
            inc_of_acquiring_alri[pathogen] = models.compute_risk_of_aquisition(
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
        models = m.models

        # The event should not run if the person is not currently alive:
        if not person.is_alive:
            return

        # Add this case to the counter:
        self.module.logging_event.new_case(age=person.age_years, pathogen=self.pathogen)

        # ----------------- Determine the Alri disease type and bacterial coinfection for this case -----------------
        disease_type, bacterial_coinfection = models.determine_disease_type_and_secondary_bacterial_coinfection(
            age=person.age_years, pathogen=self.pathogen, va_pneumo_all_doses=person.va_pneumo_all_doses)

        # ----------------------- Duration of the Alri event -----------------------
        duration_in_days_of_alri = rng.randint(1, 8)  # assumes uniform interval around mean duration with range 4 days

        # Date for outcome (either recovery or death) with uncomplicated Alri
        date_of_outcome = self.module.sim.date + DateOffset(days=duration_in_days_of_alri)

        # Define 'episode end' date. This the date when this episode ends. It is the last possible data that any HSI
        # could affect this episode.
        episode_end = date_of_outcome + m.max_duration_of_episode

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
        self.impose_symptoms_for_uncomplicated_disease(person_id=person_id, disease_type=disease_type)

        # ----------------------------------- Complications  -----------------------------------
        self.impose_complications(person_id=person_id, date_of_outcome=date_of_outcome)

        # ----------------------------------- Outcome  -----------------------------------
        # Determine outcome: death or recovery
        if models.death(person_id=person_id):
            self.sim.schedule_event(AlriDeathEvent(self.module, person_id), date_of_outcome)
            df.loc[person_id, ['ri_scheduled_death_date', 'ri_scheduled_recovery_date']] = [date_of_outcome, pd.NaT]
        else:
            self.sim.schedule_event(AlriNaturalRecoveryEvent(self.module, person_id), date_of_outcome)
            df.loc[person_id, ['ri_scheduled_recovery_date', 'ri_scheduled_death_date']] = [date_of_outcome, pd.NaT]

    def impose_symptoms_for_uncomplicated_disease(self, person_id, disease_type):
        """
        Imposes the clinical symptoms to uncomplicated Alri. These symptoms are not set to auto-resolve
        """
        m = self.module
        models = m.models

        for symptom in models.symptoms_for_disease(disease_type=disease_type):
            m.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove='+',
                disease_module=m,
            )

    def impose_complications(self, person_id, date_of_outcome):
        """Choose a set of complications for this person and onset these all instantanesouly."""

        df = self.sim.population.props
        m = self.module
        models = m.models

        complications = models.complications(person_id=person_id)
        df.loc[person_id, [f"ri_complication_{complication}" for complication in complications]] = True
        for complication in complications:
            m.impose_symptoms_for_complication(person_id=person_id, complication=complication)

        # Consider delayed-onset of complications and schedule events accordingly
        date_of_onset_delayed_complications = random_date(self.sim.date, date_of_outcome, m.rng)
        delayed_complications = models.delayed_complications(person_id=person_id)
        for delayed_complication in delayed_complications:
            self.sim.schedule_event(
                AlriDelayedOnsetComplication(person_id=person_id,
                                             complication=delayed_complication,
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
        if not person.is_alive:
            return

        # If person is infected, not on treatment and does not already have the complication, add this complication:
        if (
            person.ri_current_infection_status and
            ~person.ri_on_treatment and
            ~person[f'ri_complication_{self.complication}']
        ):
            df.at[person_id, f'ri_complication_{self.complication}'] = True
            m.impose_symptoms_for_complication(complication=self.complication, person_id=person_id)


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
    """
    This Event is for the death of someone that is caused by the infection with a pathogen that causes Alri.
    """

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

class HSI_Alri_GenericTreatment(HSI_Event, IndividualScopeEventMixin):
    """
    This is a template for the HSI interaction events. It just shows the checks to use each time.
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
        if not (person.is_alive and person.ri_current_infection_status):
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
#   DEBUGGING / TESTING EVENTS
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
        'nb_low_birth_weight_status': Property(Types.CATEGORICAL, 'temporary property',
                                               categories=['extremely_low_birth_weight', 'very_low_birth_weight',
                                                           'low_birth_weight', 'normal_birth_weight']),

        'nb_breastfeeding_status': Property(Types.CATEGORICAL, 'temporary property',
                                            categories=['none', 'non_exclusive', 'exclusive']),
        'va_pneumo_all_doses': Property(Types.BOOL, 'temporary property'),
        'va_hib_all_doses': Property(Types.BOOL, 'temporary property'),
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
        df.loc[df.is_alive, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.loc[df.is_alive, 'nb_breastfeeding_status'] = 'non_exclusive'
        df.loc[df.is_alive, 'va_pneumo_all_doses'] = False
        df.loc[df.is_alive, 'va_hib_all_doses'] = False
        df.loc[df.is_alive, 'un_clinical_acute_malnutrition'] = 'well'

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        df = self.sim.population.props
        df.at[child, 'hv_inf'] = False
        df.at[child, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.at[child, 'nb_breastfeeding_status'] = 'non_exclusive'
        df.at[child, 'va_pneumo_all_doses'] = False
        df.at[child, 'va_hib_all_doses'] = False
        df.at[child, 'un_clinical_acute_malnutrition'] = 'well'
