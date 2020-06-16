"""
Childhood pneumonia module
Documentation: 04 - Methods Repository/Method_Child_RespiratoryInfection.xlsx
"""

from tlo import Module, Parameter, Property, Types, logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

class Bronchiolitis(Module):
    # Declare the pathogens that this module will simulate:
    pathogens = {
        'RSV',
        'rhinovirus',
        'hMPV',
        'parainfluenza',
        'influenza',
        'other_pathogens'
    }

    complications = {'pneumothorax', 'pleural_effusion', 'empyema',
                     'lung_abscess', 'sepsis', 'meningitis', 'respiratory_failure'}

    PARAMETERS = {
        'base_incidence_pneumonia_by_agecat': Parameter
        (Types.REAL, 'overall incidence of pneumonia by age category'
         ),
        'pn_attributable_fraction_RSV': Parameter
        (Types.REAL, 'attributable fraction of RSV causing pneumonia'
         ),
        'pn_attributable_fraction_rhinovirus': Parameter
        (Types.REAL, 'attributable fraction of rhinovirus causing pneumonia'
         ),
        'pn_attributable_fraction_hmpv': Parameter
        (Types.REAL, 'attributable fraction of hMPV causing pneumonia'
         ),
        'pn_attributable_fraction_parainfluenza': Parameter
        (Types.REAL, 'attributable fraction of parainfluenza causing pneumonia'
         ),
        'pn_attributable_fraction_streptococcus': Parameter
        (Types.REAL, 'attributable fraction of streptococcus causing pneumonia'
         ),
        'pn_attributable_fraction_hib': Parameter
        (Types.REAL, 'attributable fraction of hib causing pneumonia'
         ),
        'pn_attributable_fraction_TB': Parameter
        (Types.REAL, 'attributable fraction of TB causing pneumonia'
         ),
        'pn_attributable_fraction_staph': Parameter
        (Types.REAL, 'attributable fraction of staphylococcus causing pneumonia'
         ),
        'pn_attributable_fraction_influenza': Parameter
        (Types.REAL, 'attributable fraction of influenza causing pneumonia'
         ),
        'pn_attributable_fraction_jirovecii': Parameter
        (Types.REAL, 'attributable fraction of jirovecii causing pneumonia'
         ),
        'pn_attributable_fraction_other_pathogens': Parameter
        (Types.REAL, 'attributable fraction of jirovecii causing pneumonia'
         ),
        'pn_attributable_fraction_other_cause': Parameter
        (Types.REAL, 'attributable fraction of jirovecii causing pneumonia'
         ),
        'base_inc_rate_pneumonia_by_RSV': Parameter
        (Types.LIST, 'incidence of pneumonia caused by Respiratory Syncytial Virus in age groups 0-11, 12-59 months'
         ),
        'base_inc_rate_pneumonia_by_rhinovirus': Parameter
        (Types.LIST, 'incidence of pneumonia caused by rhinovirus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_pneumonia_by_hMPV': Parameter
        (Types.LIST, 'incidence of pneumonia caused by hMPV in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_pneumonia_by_parainfluenza': Parameter
        (Types.LIST, 'incidence of pneumonia caused by parainfluenza in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_pneumonia_by_streptococcus': Parameter
        (Types.LIST, 'incidence of pneumonia caused by streptoccocus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_pneumonia_by_hib': Parameter
        (Types.LIST, 'incidence of pneumonia caused by hib in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_pneumonia_by_TB': Parameter
        (Types.LIST, 'incidence of pneumonia caused by TB in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_pneumonia_by_staphylococcus': Parameter
        (Types.LIST, 'incidence of pneumonia caused by Staphylococcus aureus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_pneumonia_by_influenza': Parameter
        (Types.LIST, 'incidence of pneumonia caused by influenza in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_pneumonia_by_jirovecii': Parameter
        (Types.LIST, 'incidence of pneumonia caused by P. jirovecii in age groups 0-11, 12-59 months'
         ),
        'base_inc_rate_pneumonia_by_other_pathogens': Parameter
        (Types.LIST, 'incidence of pneumonia caused by other pathogens in age groups 0-11, 12-59 months'
         ),
        'rr_pneumonia_HHhandwashing': Parameter
        (Types.REAL, 'relative rate of pneumonia with household handwashing with soap'
         ),
        'rr_pneumonia_HIV': Parameter
        (Types.REAL, 'relative rate of pneumonia for HIV positive status'
         ),
        'rr_pneumonia_SAM': Parameter
        (Types.REAL, 'relative rate of pneumonia for severe malnutrition'
         ),
        'rr_pneumonia_excl_breastfeeding': Parameter
        (Types.REAL, 'relative rate of pneumonia for exclusive breastfeeding upto 6 months'
         ),
        'rr_pneumonia_cont_breast': Parameter
        (Types.REAL, 'relative rate of pneumonia for continued breastfeeding 6 months to 2 years'
         ),
        'rr_pneumonia_indoor_air_pollution': Parameter
        (Types.REAL, 'relative rate of pneumonia for indoor air pollution'
         ),
        'rr_pneumonia_pneumococcal_vaccine': Parameter
        (Types.REAL, 'relative rate of pneumonia for pneumonococcal vaccine'
         ),
        'rr_pneumonia_hib_vaccine': Parameter
        (Types.REAL, 'relative rate of pneumonia for hib vaccine'
         ),
        'rr_pneumonia_influenza_vaccine': Parameter
        (Types.REAL, 'relative rate of pneumonia for influenza vaccine'
         ),
        'r_progress_to_severe_pneumonia': Parameter
        (Types.LIST,
         'probability of progressing from non-severe to severe pneumonia by age category '
         'HIV negative, no SAM'
         ),
        'prob_respiratory_failure_by_viral_pneumonia': Parameter
        (Types.REAL, 'probability of respiratory failure caused by primary viral pneumonia'
         ),
        'prob_respiratory_failure_by_bacterial_pneumonia': Parameter
        (Types.REAL, 'probability of respiratory failure caused by primary or secondary bacterial pneumonia'
         ),
        'prob_respiratory_failure_to_multiorgan_dysfunction': Parameter
        (Types.REAL, 'probability of respiratory failure causing multi-organ dysfunction'
         ),
        'prob_sepsis_by_viral_pneumonia': Parameter
        (Types.REAL, 'probability of sepsis caused by primary viral pneumonia'
         ),
        'prob_sepsis_by_bacterial_pneumonia': Parameter
        (Types.REAL, 'probability of sepsis caused by primary or secondary bacterial pneumonia'
         ),
        'prob_sepsis_to_septic_shock': Parameter
        (Types.REAL, 'probability of sepsis causing septic shock'
         ),
        'prob_septic_shock_to_multiorgan_dysfunction': Parameter
        (Types.REAL, 'probability of septic shock causing multi-organ dysfunction'
         ),
        'prob_meningitis_by_viral_pneumonia': Parameter
        (Types.REAL, 'probability of meningitis caused by primary viral pneumonia'
         ),
        'prob_meningitis_by_bacterial_pneumonia': Parameter
        (Types.REAL, 'probability of meningitis caused by primary or secondary bacterial pneumonia'
         ),
        'prob_pleural_effusion_by_bacterial_pneumonia': Parameter
        (Types.REAL, 'probability of pleural effusion caused by primary or secondary bacterial pneumonia'
         ),
        'prob_pleural_effusion_to_empyema': Parameter
        (Types.REAL, 'probability of pleural effusion developing into empyema'
         ),
        'prob_empyema_to_sepsis': Parameter
        (Types.REAL, 'probability of empyema causing sepsis'
         ),
        'prob_lung_abscess_by_bacterial_pneumonia': Parameter
        (Types.REAL, 'probability of a lung abscess caused by primary or secondary bacterial pneumonia'
         ),
        'prob_pneumothorax_by_bacterial_pneumonia': Parameter
        (Types.REAL, 'probability of pneumothorax caused by primary or secondary bacterial pneumonia'
         ),
        'prob_pneumothorax_to_respiratory_failure': Parameter
        (Types.REAL, 'probability of pneumothorax causing respiratory failure'
         ),
        'prob_lung_abscess_to_sepsis': Parameter
        (Types.REAL, 'probability of lung abscess causing sepsis'
         ),
        'r_death_from_pneumonia_due_to_meningitis': Parameter
        (Types.REAL, 'death rate from pneumonia due to meningitis'
         ),
        'r_death_from_pneumonia_due_to_sepsis': Parameter
        (Types.REAL, 'death rate from pneumonia due to sepsis'
         ),
        'r_death_from_pneumonia_due_to_respiratory_failure': Parameter
        (Types.REAL, 'death rate from pneumonia due to respiratory failure'
         ),
        # 'rr_death_pneumonia_agelt2mo': Parameter
        # (Types.REAL,
        #  'death rate of pneumonia'
        #  ),
        'rr_death_pneumonia_age12to23mo': Parameter
        (Types.REAL,
         'death rate of pneumonia'
         ),
        'rr_death_pneumonia_age24to59mo': Parameter
        (Types.REAL,
         'death rate of pneumonia'
         ),
        'rr_death_pneumonia_HIV': Parameter
        (Types.REAL,
         'death rate of pneumonia'
         ),
        'rr_death_pneumonia_SAM': Parameter
        (Types.REAL,
         'death rate of pneumonia'
         ),
        'rr_death_pneumonia_low_birth_weight': Parameter
        (Types.REAL,
         'death rate of pneumonia'
         ),
    }

    PROPERTIES = {
        # ---- The pathogen which is the attributed cause of pneumonia ----
        'ri_last_pneumonia_pathogen': Property(Types.CATEGORICAL,
                                               'Attributable pathogen for the last pneumonia event',
                                               categories=list(pathogens) + ['none']),

        # ---- Complications associated with pneumonia ----
        'ri_last_pneumonia_complications': Property(Types.LIST,
                                                    'complications that arose from last pneumonia event',
                                                    categories=['pneumothorax', 'pleural_eff usion', 'empyema',
                                                                'lung_abscess', 'sepsis', 'meningitis',
                                                                'respiratory_failure'] + ['none']
                                                    ),

        # ---- Internal variables to schedule onset and deaths due to pneumonia ----
        'ri_last_pneumonia_date_of_onset': Property(Types.DATE, 'date of onset of last pneumonia event'),
        'ri_last_pneumonia_recovered_date': Property(Types.DATE, 'date of recovery from last pneumonia event'),
        'ri_last_pneumonia_death_date': Property(Types.DATE, 'date of death caused by last pneumonia event'),

        # ---- Temporary Variables: To be replaced with the properties of other modules ----
        'tmp_malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'tmp_low_birth_weight': Property(Types.BOOL, 'temporary property - low birth weight'),
        'tmp_hv_inf': Property(Types.BOOL, 'temporary property - hiv infection'),
        'tmp_exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'tmp_continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'tmp_pneumococcal_vaccination': Property(Types.BOOL, 'temporary property - streptococcus pneumoniae vaccine'),
        'tmp_hib_vaccination': Property(Types.BOOL, 'temporary property - H. influenzae type b vaccine'),
        'tmp_influenza_vaccination': Property(Types.BOOL, 'temporary property - flu vaccine'),

        # ---- Treatment properties ----
        # TODO; Ines -- you;ve introduced these but not initialised them and don;t use them. do you need them?
        'ri_pneumonia_treatment': Property(Types.BOOL, 'currently on pneumonia treatment'),
        'ri_pneumonia_tx_start_date': Property(Types.DATE, 'start date of pneumonia treatment for current event'),

        # 'date_of_progression_severe_pneum': Property
        # (Types.DATE, 'date of progression of disease to severe pneumonia'
        #  ),
        # 'date_of_progression_very_sev_pneum': Property
        # (Types.DATE, 'date of progression of disease to severe pneumonia'
        #  ),
    }

    # declare the symptoms that this module will cause:
    SYMPTOMS = {'fever', 'cough', 'difficult_breathing', 'fast_breathing', 'chest_indrawing', 'grunting',
                'cyanosis', 'severe_respiratory_distress', 'hypoxia', 'danger_signs'}
