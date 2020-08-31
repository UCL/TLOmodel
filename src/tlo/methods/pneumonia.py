"""
Childhood ALRI module
Documentation: 04 - Methods Repository/ResourceFile_Childhood_Pneumonia.xlsx
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

class ALRI(Module):
    # Declare the pathogens that this module will simulate:
    pathogens = {
        'RSV',
        'rhinovirus',
        'hMPV',
        'parainfluenza',
        'streptococcus',
        'hib',
        'TB',
        'staphylococcus',
        'influenza',
        'jirovecii',
        'other_pathogens'
    }

    complications = {'pneumothorax', 'pleural_effusion', 'empyema',
                     'lung_abscess', 'sepsis', 'meningitis', 'respiratory_failure'}

    # Declare the severity levels of the disease:
    pathogen_type = {
        'viral': 'RSV' 'rhinovirus' 'hMPV' 'parainfluenza' 'influenza',
        'bacterial': 'streptococcus' 'hib' 'TB' 'staphylococcus'
    }


    PARAMETERS = {
        'base_incidence_ALRI_by_agecat': Parameter
        (Types.REAL, 'overall incidence of ALRI by age category'
         ),
        'pn_attributable_fraction_RSV': Parameter
        (Types.REAL, 'attributable fraction of RSV causing ALRI'
         ),
        'pn_attributable_fraction_rhinovirus': Parameter
        (Types.REAL, 'attributable fraction of rhinovirus causing ALRI'
         ),
        'pn_attributable_fraction_hmpv': Parameter
        (Types.REAL, 'attributable fraction of hMPV causing ALRI'
         ),
        'pn_attributable_fraction_parainfluenza': Parameter
        (Types.REAL, 'attributable fraction of parainfluenza causing ALRI'
         ),
        'pn_attributable_fraction_streptococcus': Parameter
        (Types.REAL, 'attributable fraction of streptococcus causing ALRI'
         ),
        'pn_attributable_fraction_hib': Parameter
        (Types.REAL, 'attributable fraction of hib causing ALRI'
         ),
        'pn_attributable_fraction_TB': Parameter
        (Types.REAL, 'attributable fraction of TB causing ALRI'
         ),
        'pn_attributable_fraction_staph': Parameter
        (Types.REAL, 'attributable fraction of staphylococcus causing ALRI'
         ),
        'pn_attributable_fraction_influenza': Parameter
        (Types.REAL, 'attributable fraction of influenza causing ALRI'
         ),
        'pn_attributable_fraction_jirovecii': Parameter
        (Types.REAL, 'attributable fraction of jirovecii causing ALRI'
         ),
        'pn_attributable_fraction_other_pathogens': Parameter
        (Types.REAL, 'attributable fraction of jirovecii causing ALRI'
         ),
        'pn_attributable_fraction_other_cause': Parameter
        (Types.REAL, 'attributable fraction of jirovecii causing ALRI'
         ),
        'base_inc_rate_ALRI_by_RSV': Parameter
        (Types.LIST, 'incidence of ALRI caused by Respiratory Syncytial Virus in age groups 0-11, 12-59 months'
         ),
        'base_inc_rate_ALRI_by_rhinovirus': Parameter
        (Types.LIST, 'incidence of ALRI caused by rhinovirus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_hMPV': Parameter
        (Types.LIST, 'incidence of ALRI caused by hMPV in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_parainfluenza': Parameter
        (Types.LIST, 'incidence of ALRI caused by parainfluenza in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_streptococcus': Parameter
        (Types.LIST, 'incidence of ALRI caused by streptoccocus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_hib': Parameter
        (Types.LIST, 'incidence of ALRI caused by hib in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_TB': Parameter
        (Types.LIST, 'incidence of ALRI caused by TB in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_staphylococcus': Parameter
        (Types.LIST, 'incidence of ALRI caused by Staphylococcus aureus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_influenza': Parameter
        (Types.LIST, 'incidence of ALRI caused by influenza in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_jirovecii': Parameter
        (Types.LIST, 'incidence of ALRI caused by P. jirovecii in age groups 0-11, 12-59 months'
         ),
        'base_inc_rate_ALRI_by_other_pathogens': Parameter
        (Types.LIST, 'incidence of ALRI caused by other pathogens in age groups 0-11, 12-59 months'
         ),
        'rr_ALRI_HHhandwashing': Parameter
        (Types.REAL, 'relative rate of ALRI with household handwashing with soap'
         ),
        'rr_ALRI_HIV': Parameter
        (Types.REAL, 'relative rate of ALRI for HIV positive status'
         ),
        'rr_ALRI_SAM': Parameter
        (Types.REAL, 'relative rate of ALRI for severe malnutrition'
         ),
        'rr_ALRI_excl_breastfeeding': Parameter
        (Types.REAL, 'relative rate of ALRI for exclusive breastfeeding upto 6 months'
         ),
        'rr_ALRI_cont_breast': Parameter
        (Types.REAL, 'relative rate of ALRI for continued breastfeeding 6 months to 2 years'
         ),
        'rr_ALRI_indoor_air_pollution': Parameter
        (Types.REAL, 'relative rate of ALRI for indoor air pollution'
         ),
        'rr_ALRI_pneumococcal_vaccine': Parameter
        (Types.REAL, 'relative rate of ALRI for pneumonococcal vaccine'
         ),
        'rr_ALRI_hib_vaccine': Parameter
        (Types.REAL, 'relative rate of ALRI for hib vaccine'
         ),
        'rr_ALRI_influenza_vaccine': Parameter
        (Types.REAL, 'relative rate of ALRI for influenza vaccine'
         ),
        'r_progress_to_severe_ALRI': Parameter
        (Types.LIST,
         'probability of progressing from non-severe to severe ALRI by age category '
         'HIV negative, no SAM'
         ),
        'prob_respiratory_failure_by_viral_ALRI': Parameter
        (Types.REAL, 'probability of respiratory failure caused by primary viral ALRI'
         ),
        'prob_respiratory_failure_by_bacterial_ALRI': Parameter
        (Types.REAL, 'probability of respiratory failure caused by primary or secondary bacterial ALRI'
         ),
        'prob_respiratory_failure_to_multiorgan_dysfunction': Parameter
        (Types.REAL, 'probability of respiratory failure causing multi-organ dysfunction'
         ),
        'prob_sepsis_by_viral_ALRI': Parameter
        (Types.REAL, 'probability of sepsis caused by primary viral ALRI'
         ),
        'prob_sepsis_by_bacterial_ALRI': Parameter
        (Types.REAL, 'probability of sepsis caused by primary or secondary bacterial ALRI'
         ),
        'prob_sepsis_to_septic_shock': Parameter
        (Types.REAL, 'probability of sepsis causing septic shock'
         ),
        'prob_septic_shock_to_multiorgan_dysfunction': Parameter
        (Types.REAL, 'probability of septic shock causing multi-organ dysfunction'
         ),
        'prob_meningitis_by_viral_ALRI': Parameter
        (Types.REAL, 'probability of meningitis caused by primary viral ALRI'
         ),
        'prob_meningitis_by_bacterial_ALRI': Parameter
        (Types.REAL, 'probability of meningitis caused by primary or secondary bacterial ALRI'
         ),
        'prob_pleural_effusion_by_bacterial_ALRI': Parameter
        (Types.REAL, 'probability of pleural effusion caused by primary or secondary bacterial ALRI'
         ),
        'prob_pleural_effusion_to_empyema': Parameter
        (Types.REAL, 'probability of pleural effusion developing into empyema'
         ),
        'prob_empyema_to_sepsis': Parameter
        (Types.REAL, 'probability of empyema causing sepsis'
         ),
        'prob_lung_abscess_by_bacterial_ALRI': Parameter
        (Types.REAL, 'probability of a lung abscess caused by primary or secondary bacterial ALRI'
         ),
        'prob_pneumothorax_by_bacterial_ALRI': Parameter
        (Types.REAL, 'probability of pneumothorax caused by primary or secondary bacterial ALRI'
         ),
        'prob_pneumothorax_to_respiratory_failure': Parameter
        (Types.REAL, 'probability of pneumothorax causing respiratory failure'
         ),
        'prob_lung_abscess_to_sepsis': Parameter
        (Types.REAL, 'probability of lung abscess causing sepsis'
         ),
        'r_death_from_ALRI_due_to_meningitis': Parameter
        (Types.REAL, 'death rate from ALRI due to meningitis'
         ),
        'r_death_from_ALRI_due_to_sepsis': Parameter
        (Types.REAL, 'death rate from ALRI due to sepsis'
         ),
        'r_death_from_ALRI_due_to_respiratory_failure': Parameter
        (Types.REAL, 'death rate from ALRI due to respiratory failure'
         ),
        # 'rr_death_ALRI_agelt2mo': Parameter
        # (Types.REAL,
        #  'death rate of ALRI'
        #  ),
        'rr_death_ALRI_age12to23mo': Parameter
        (Types.REAL,
         'death rate of ALRI'
         ),
        'rr_death_ALRI_age24to59mo': Parameter
        (Types.REAL,
         'death rate of ALRI'
         ),
        'rr_death_ALRI_HIV': Parameter
        (Types.REAL,
         'death rate of ALRI'
         ),
        'rr_death_ALRI_SAM': Parameter
        (Types.REAL,
         'death rate of ALRI'
         ),
        'rr_death_ALRI_low_birth_weight': Parameter
        (Types.REAL,
         'death rate of ALRI'
         ),
    }

    PROPERTIES = {
        # ---- The underlying ALRI condition ----
        'ri_current_ALRI_disease_type': Property(Types.CATEGORICAL, 'underlying ALRI condition',
                                                 categories=['viral pneumonia', 'bacterial pneumonia', 'co-infection',
                                                             'bronchiolitis']),

        # ---- The pathogen which is the attributed cause of ALRI ----
        'ri_primary_ALRI_pathogen': Property(Types.CATEGORICAL,
                                             'Attributable pathogen for the current ALRI event',
                                             categories=list(pathogens) + ['none']),

        # ---- The bacterial pathogen which is the attributed cause of ALRI ----
        'ri_secondary_ALRI_pathogen': Property(Types.CATEGORICAL,
                                               'Secondary bacterial pathogen for the current ALRI event',
                                               categories=list(pathogens) + ['none']),  # todo: only bacterial agents

        # ---- Complications associated with ALRI ----
        'ri_current_ALRI_complications': Property(Types.LIST,
                                                  'complications that arose from the current ALRI event',
                                                  categories=['pneumothorax', 'pleural_effusion', 'empyema',
                                                              'lung_abscess', 'sepsis', 'meningitis',
                                                              'respiratory_failure'] + ['none']
                                                  ),

        # ---- Symptoms associated with ALRI ----
        'ri_current_ALRI_symptoms': Property(Types.LIST,
                                             'symptoms of current ALRI event',
                                             categories=['fever', 'cough', 'difficult_breathing',
                                                         'fast_breathing', 'chest_indrawing', 'grunting',
                                                         'cyanosis', 'severe_respiratory_distress', 'hypoxia',
                                                         'danger_signs']
                                             ),

        # ---- Internal variables to schedule onset and deaths due to ALRI ----
        'ri_ALRI_event_date_of_onset': Property(Types.DATE, 'date of onset of current ALRI event'),
        'ri_ALRI_event_recovered_date': Property(Types.DATE, 'date of recovery from current ALRI event'),
        'ri_ALRI_event_death_date': Property(Types.DATE, 'date of death caused by current ALRI event'),

        # ---- Temporary Variables: To be replaced with the properties of other modules ----
        'tmp_malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'tmp_low_birth_weight': Property(Types.BOOL, 'temporary property - low birth weight'),
        'tmp_hv_inf': Property(Types.BOOL, 'temporary property - hiv infection'),
        'tmp_exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'tmp_continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'tmp_pneumococcal_vaccination': Property(Types.BOOL, 'temporary property - streptococcus ALRIe vaccine'),
        'tmp_hib_vaccination': Property(Types.BOOL, 'temporary property - H. influenzae type b vaccine'),
        'tmp_influenza_vaccination': Property(Types.BOOL, 'temporary property - flu vaccine'),

        # ---- Treatment properties ----
        # TODO; Ines -- you;ve introduced these but not initialised them and don;t use them. do you need them?
        'ri_ALRI_treatment': Property(Types.BOOL, 'currently on ALRI treatment'),
        'ri_ALRI_tx_start_date': Property(Types.DATE, 'start date of ALRI treatment for current event'),
    }

    # declare the symptoms that this module will cause:
    SYMPTOMS = {'fever', 'cough', 'difficult_breathing', 'fast_breathing', 'chest_indrawing', 'grunting',
                'cyanosis', 'severe_respiratory_distress', 'hypoxia', 'danger_signs', 'stridor'}

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # equations for the incidence of ALRI by pathogen:
        self.incidence_equations_by_pathogen = dict()

        # equations for the proportions of ALRI diseases:
        self.proportions_of_ALRI_diseases_caused_by_each_pathogen = dict()

        # equations for the probabilities of secondary bacterial co-infection:
        self.prob_secondary_bacterial_infection = dict()

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

        zeros_counter = dict(zip(self.pathogens, [0]*len(self.pathogens)))
        self.incident_case_tracker_zeros = {
            '0y': copy.deepcopy(zeros_counter),
            '1y': copy.deepcopy(zeros_counter),
            '2-4y': copy.deepcopy(zeros_counter),
            '5+y': copy.deepcopy(zeros_counter)
        }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module
        """
        p = self.parameters
        self.load_parameters_from_dataframe(
            pd.read_excel(
                Path(self.resourcefilepath) / 'ResourceFile_Childhood_Pneumonia.xlsx', sheet_name='Parameter_values'))

        p['prob_secondary_bacterial_infection_from_bronchiolitis'] = 0.03
        p['prob_secondary_bacterial_infection_from_viral_pneumonia'] = 0.3

        # Check that every value has been read-in successfully
        for param_name, type in self.PARAMETERS.items():
            assert param_name in self.parameters, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert isinstance(self.parameters[param_name],
                          type.python_type), f'Parameter "{param_name}" is not read in correctly from the resourcefile.'

        # DALY weights
        # if 'HealthBurden' in self.sim.modules.keys():
        #     p['daly_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
        #     p['daly_severe_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
        #     p['daly_very_severe_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=46)

        # --------------------------------------------------------------------------------------------
        # Make a dict to hold the equations that govern the probability that a person acquires ALRI
        # that is caused (primarily) by a pathogen

        self.incidence_equations_by_pathogen.update({
            'RSV': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('age_years')
                               .when('.between(0,0)', p['base_inc_rate_ALRI_by_RSV'][0])
                               .when('.between(1,1)', p['base_inc_rate_ALRI_by_RSV'][1])
                               .when('.between(2,4)', p['base_inc_rate_ALRI_by_RSV'][2])
                               .otherwise(0.0),
                               Predictor('li_no_access_handwashing')
                               .when(False, p['rr_ALRI_HHhandwashing']),
                               Predictor('li_wood_burn_stove')
                               .when(False, p['rr_ALRI_indoor_air_pollution']),
                               Predictor('tmp_hv_inf')
                               .when(True, p['rr_ALRI_HIV']),
                               Predictor('tmp_malnutrition')
                               .when(True, p['rr_ALRI_SAM']),
                               Predictor('tmp_exclusive_breastfeeding')
                               .when(False, p['rr_ALRI_excl_breastfeeding'])
                               )
        })

        self.incidence_equations_by_pathogen.update({
            'rhinovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                      1.0,
                                      Predictor('age_years')
                                      .when('.between(0,0)', p['base_inc_rate_ALRI_by_rhinovirus'][0])
                                      .when('.between(1,1)', p['base_inc_rate_ALRI_by_rhinovirus'][1])
                                      .when('.between(2,4)', p['base_inc_rate_ALRI_by_rhinovirus'][2])
                                      .otherwise(0.0),
                                      Predictor('li_no_access_handwashing')
                                      .when(False, p['rr_ALRI_HHhandwashing']),
                                      Predictor('li_wood_burn_stove')
                                      .when(False, p['rr_ALRI_indoor_air_pollution']),
                                      Predictor('tmp_hv_inf')
                                      .when(True, p['rr_ALRI_HIV']),
                                      Predictor('tmp_malnutrition')
                                      .when(True, p['rr_ALRI_SAM']),
                                      Predictor('tmp_exclusive_breastfeeding')
                                      .when(False, p['rr_ALRI_excl_breastfeeding'])
                                      )
        })

        self.incidence_equations_by_pathogen.update({
            'hMPV': LinearModel(LinearModelType.MULTIPLICATIVE,
                                1.0,
                                Predictor('age_years')
                                .when('.between(0,0)', p['base_inc_rate_ALRI_by_hMPV'][0])
                                .when('.between(1,1)', p['base_inc_rate_ALRI_by_hMPV'][1])
                                .when('.between(2,4)', p['base_inc_rate_ALRI_by_hMPV'][2])
                                .otherwise(0.0),
                                Predictor('li_no_access_handwashing')
                                .when(False, p['rr_ALRI_HHhandwashing']),
                                Predictor('li_wood_burn_stove')
                                .when(False, p['rr_ALRI_indoor_air_pollution']),
                                Predictor('tmp_hv_inf')
                                .when(True, p['rr_ALRI_HIV']),
                                Predictor('tmp_malnutrition')
                                .when(True, p['rr_ALRI_SAM']),
                                Predictor('tmp_exclusive_breastfeeding')
                                .when(False, p['rr_ALRI_excl_breastfeeding'])
                                )
        })

        self.incidence_equations_by_pathogen.update({
            'parainfluenza': LinearModel(LinearModelType.MULTIPLICATIVE,
                                         1.0,
                                         Predictor('age_years')
                                         .when('.between(0,0)', p['base_inc_rate_ALRI_by_parainfluenza'][0])
                                         .when('.between(1,1)', p['base_inc_rate_ALRI_by_parainfluenza'][1])
                                         .when('.between(2,4)', p['base_inc_rate_ALRI_by_parainfluenza'][2])
                                         .otherwise(0.0),
                                         Predictor('li_no_access_handwashing')
                                         .when(False, p['rr_ALRI_HHhandwashing']),
                                         Predictor('li_wood_burn_stove')
                                         .when(False, p['rr_ALRI_indoor_air_pollution']),
                                         Predictor('tmp_hv_inf')
                                         .when(True, p['rr_ALRI_HIV']),
                                         Predictor('tmp_malnutrition')
                                         .when(True, p['rr_ALRI_SAM']),
                                         Predictor('tmp_exclusive_breastfeeding')
                                         .when(False, p['rr_ALRI_excl_breastfeeding'])
                                         )
        })

        self.incidence_equations_by_pathogen.update({
            'streptococcus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                         1.0,
                                         Predictor('age_years')
                                         .when('.between(0,0)', p['base_inc_rate_ALRI_by_streptococcus'][0])
                                         .when('.between(1,1)', p['base_inc_rate_ALRI_by_streptococcus'][1])
                                         .when('.between(2,4)', p['base_inc_rate_ALRI_by_streptococcus'][2])
                                         .otherwise(0.0),
                                         Predictor('li_no_access_handwashing')
                                         .when(False, p['rr_ALRI_HHhandwashing']),
                                         Predictor('li_wood_burn_stove')
                                         .when(False, p['rr_ALRI_indoor_air_pollution']),
                                         Predictor('tmp_hv_inf')
                                         .when(True, p['rr_ALRI_HIV']),
                                         Predictor('tmp_malnutrition')
                                         .when(True, p['rr_ALRI_SAM']),
                                         Predictor('tmp_exclusive_breastfeeding')
                                         .when(False, p['rr_ALRI_excl_breastfeeding']),
                                         Predictor('tmp_pneumococcal_vaccination')
                                         .when(False, p['rr_ALRI_pneumococcal_vaccine'])
                                         )
        })

        self.incidence_equations_by_pathogen.update({
            'hib': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('age_years')
                               .when('.between(0,0)', p['base_inc_rate_ALRI_by_hib'][0])
                               .when('.between(1,1)', p['base_inc_rate_ALRI_by_hib'][1])
                               .when('.between(2,4)', p['base_inc_rate_ALRI_by_hib'][2])
                               .otherwise(0.0),
                               Predictor('li_no_access_handwashing')
                               .when(False, p['rr_ALRI_HHhandwashing']),
                               Predictor('li_wood_burn_stove')
                               .when(False, p['rr_ALRI_indoor_air_pollution']),
                               Predictor('tmp_hv_inf')
                               .when(True, p['rr_ALRI_HIV']),
                               Predictor('tmp_malnutrition')
                               .when(True, p['rr_ALRI_SAM']),
                               Predictor('tmp_exclusive_breastfeeding')
                               .when(False, p['rr_ALRI_excl_breastfeeding']),
                               Predictor('tmp_hib_vaccination')
                               .when(False, p['rr_ALRI_hib_vaccine'])
                               )
        })

        self.incidence_equations_by_pathogen.update({
            'TB': LinearModel(LinearModelType.MULTIPLICATIVE,
                              1.0,
                              Predictor('age_years')
                              .when('.between(0,0)', p['base_inc_rate_ALRI_by_TB'][0])
                              .when('.between(1,1)', p['base_inc_rate_ALRI_by_TB'][1])
                              .when('.between(2,4)', p['base_inc_rate_ALRI_by_TB'][2])
                              .otherwise(0.0),
                              Predictor('ri_primary_ALRI_pathogen')
                              .when('is not TB', 0.0),
                              Predictor('li_no_access_handwashing')
                              .when(False, p['rr_ALRI_HHhandwashing']),
                              Predictor('li_wood_burn_stove')
                              .when(False, p['rr_ALRI_indoor_air_pollution']),
                              Predictor('tmp_hv_inf')
                              .when(True, p['rr_ALRI_HIV']),
                              Predictor('tmp_malnutrition')
                              .when(True, p['rr_ALRI_SAM']),
                              Predictor('tmp_exclusive_breastfeeding')
                              .when(False, p['rr_ALRI_excl_breastfeeding'])
                              )
        })

        self.incidence_equations_by_pathogen.update({
            'staphylococcus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                          1.0,
                                          Predictor('age_years')
                                          .when('.between(0,0)', p['base_inc_rate_ALRI_by_staphylococcus'][0])
                                          .when('.between(1,1)', p['base_inc_rate_ALRI_by_staphylococcus'][1])
                                          .when('.between(2,4)', p['base_inc_rate_ALRI_by_staphylococcus'][2])
                                          .otherwise(0.0),
                                          Predictor('li_no_access_handwashing')
                                          .when(False, p['rr_ALRI_HHhandwashing']),
                                          Predictor('li_wood_burn_stove')
                                          .when(False, p['rr_ALRI_indoor_air_pollution']),
                                          Predictor('tmp_hv_inf')
                                          .when(True, p['rr_ALRI_HIV']),
                                          Predictor('tmp_malnutrition')
                                          .when(True, p['rr_ALRI_SAM']),
                                          Predictor('tmp_exclusive_breastfeeding')
                                          .when(False, p['rr_ALRI_excl_breastfeeding'])
                                          )
        })

        self.incidence_equations_by_pathogen.update({
            'influenza': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('age_years')
                                     .when('.between(0,0)', p['base_inc_rate_ALRI_by_influenza'][0])
                                     .when('.between(1,1)', p['base_inc_rate_ALRI_by_influenza'][1])
                                     .when('.between(2,4)', p['base_inc_rate_ALRI_by_influenza'][2])
                                     .otherwise(0.0),
                                     Predictor('li_no_access_handwashing')
                                     .when(False, p['rr_ALRI_HHhandwashing']),
                                     Predictor('li_wood_burn_stove')
                                     .when(False, p['rr_ALRI_indoor_air_pollution']),
                                     Predictor('tmp_hv_inf')
                                     .when(True, p['rr_ALRI_HIV']),
                                     Predictor('tmp_malnutrition')
                                     .when(True, p['rr_ALRI_SAM']),
                                     Predictor('tmp_exclusive_breastfeeding')
                                     .when(False, p['rr_ALRI_excl_breastfeeding'])
                                     )
        })

        self.incidence_equations_by_pathogen.update({
            'jirovecii': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('age_years')
                                     .when('.between(0,0)', p['base_inc_rate_ALRI_by_jirovecii'][0])
                                     .when('.between(1,1)', p['base_inc_rate_ALRI_by_jirovecii'][1])
                                     .when('.between(2,4)', p['base_inc_rate_ALRI_by_jirovecii'][2])
                                     .otherwise(0.0),
                                     Predictor('li_no_access_handwashing')
                                     .when(False, p['rr_ALRI_HHhandwashing']),
                                     Predictor('li_wood_burn_stove')
                                     .when(False, p['rr_ALRI_indoor_air_pollution']),
                                     Predictor('tmp_hv_inf')
                                     .when(True, p['rr_ALRI_HIV']),
                                     Predictor('tmp_malnutrition')
                                     .when(True, p['rr_ALRI_SAM']),
                                     Predictor('tmp_exclusive_breastfeeding')
                                     .when(False, p['rr_ALRI_excl_breastfeeding'])
                                     )
        })

        self.incidence_equations_by_pathogen.update({
            'other_pathogens': LinearModel(LinearModelType.MULTIPLICATIVE,
                                           1.0,
                                           Predictor('age_years')
                                           .when('.between(0,0)', p['base_inc_rate_ALRI_by_other_pathogens'][0])
                                           .when('.between(1,1)', p['base_inc_rate_ALRI_by_other_pathogens'][1])
                                           .when('.between(2,4)', p['base_inc_rate_ALRI_by_other_pathogens'][2])
                                           .otherwise(0.0),
                                           Predictor('li_no_access_handwashing')
                                           .when(False, p['rr_ALRI_HHhandwashing']),
                                           Predictor('li_wood_burn_stove')
                                           .when(False, p['rr_ALRI_indoor_air_pollution']),
                                           Predictor('tmp_hv_inf')
                                           .when(True, p['rr_ALRI_HIV']),
                                           Predictor('tmp_malnutrition')
                                           .when(True, p['rr_ALRI_SAM']),
                                           Predictor('tmp_exclusive_breastfeeding')
                                           .when(False, p['rr_ALRI_excl_breastfeeding'])
                                           )
        })

        # check that equations have been declared for each pathogens
        assert self.pathogens == set(list(self.incidence_equations_by_pathogen.keys()))

        # --------------------------------------------------------------------------------------------
        # Linear models for determining the underlying condition as pneumonia caused by each pathogen
        self.proportions_of_ALRI_diseases_caused_by_each_pathogen.update({
            'RSV': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('ri_primary_ALRI_pathogen')
                               .when('is not RSV', 0.0),
                               Predictor('age_years')
                               .when('.between(0,0)', p['p_RSV_inf_causing_pneumonia'][0])
                               .when('.between(1,1)', p['p_RSV_inf_causing_pneumonia'][1])
                               .when('.between(2,4)', p['p_RSV_inf_causing_pneumonia'][2])
                               .otherwise(0.0),
                               ),
            'rhinovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('ri_primary_ALRI_pathogen')
                               .when('is not rhinovirus', 0.0),
                               Predictor('age_years')
                               .when('.between(0,0)', p['p_rhinovirus_inf_causing_pneumonia'][0])
                               .when('.between(1,1)', p['p_rhinovirus_inf_causing_pneumonia'][1])
                               .when('.between(2,4)', p['p_rhinovirus_inf_causing_pneumonia'][2])
                               .otherwise(0.0),
                               ),
            'hMPV': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('ri_primary_ALRI_pathogen')
                               .when('is not hMPV', 0.0),
                               Predictor('age_years')
                               .when('.between(0,0)', p['p_hMPV_inf_causing_pneumonia'][0])
                               .when('.between(1,1)', p['p_hMPV_inf_causing_pneumonia'][1])
                               .when('.between(2,4)', p['p_hMPV_inf_causing_pneumonia'][2])
                               .otherwise(0.0),
                               ),
            'parainfluenza': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('ri_primary_ALRI_pathogen')
                               .when('is not parainfluenza', 0.0),
                               Predictor('age_years')
                               .when('.between(0,0)', p['p_parainfluenza_inf_causing_pneumonia'][0])
                               .when('.between(1,1)', p['p_parainfluenza_inf_causing_pneumonia'][1])
                               .when('.between(2,4)', p['p_parainfluenza_inf_causing_pneumonia'][2])
                               .otherwise(0.0),
                               ),
            'streptococcus': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('ri_primary_ALRI_pathogen')
                               .when(".isin(['RSV', 'rhinovirus', 'hMPV', 'parainfluenza', 'influenza'])",
                                     p['prob_secondary_bacterial_infection'])
                               .otherwise(0.0),
                               Predictor('age_years')
                               .when('.between(0,0)', p['p_streptococcus_inf_causing_pneumonia'][0])
                               .when('.between(1,1)', p['p_streptococcus_inf_causing_pneumonia'][1])
                               .when('.between(2,4)', p['p_streptococcus_inf_causing_pneumonia'][2])
                               .otherwise(0.0)
                               ),
            'hib': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('ri_primary_ALRI_pathogen')
                               .when('is not hib', 0.0),
                               Predictor('age_years')
                               .when('.between(0,0)', p['p_hib_inf_causing_pneumonia'][0])
                               .when('.between(1,1)', p['p_hib_inf_causing_pneumonia'][1])
                               .when('.between(2,4)', p['p_hib_inf_causing_pneumonia'][2])
                               .otherwise(0.0),
                               ),
            'TB': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('ri_primary_ALRI_pathogen')
                               .when('is not TB', 0.0),
                               Predictor('age_years')
                               .when('.between(0,0)', p['p_TB_inf_causing_pneumonia'][0])
                               .when('.between(1,1)', p['p_TB_inf_causing_pneumonia'][1])
                               .when('.between(2,4)', p['p_TB_inf_causing_pneumonia'][2])
                               .otherwise(0.0),
                               ),
            'staphylococcus': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('ri_primary_ALRI_pathogen')
                               .when('is not staphylococcus', 0.0),
                               Predictor('age_years')
                               .when('.between(0,0)', p['p_staphylococcus_inf_causing_pneumonia'][0])
                               .when('.between(1,1)', p['p_staphylococcus_inf_causing_pneumonia'][1])
                               .when('.between(2,4)', p['p_staphylococcus_inf_causing_pneumonia'][2])
                               .otherwise(0.0),
                               ),
            'influenza': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('ri_primary_ALRI_pathogen')
                               .when('is not influenza', 0.0),
                               Predictor('age_years')
                               .when('.between(0,0)', p['p_influenza_inf_causing_pneumonia'][0])
                               .when('.between(1,1)', p['p_influenza_inf_causing_pneumonia'][1])
                               .when('.between(2,4)', p['p_influenza_inf_causing_pneumonia'][2])
                               .otherwise(0.0),
                               ),
            'jirovecii': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('ri_primary_ALRI_pathogen')
                               .when('is not jirovecii', 0.0),
                               Predictor('age_years')
                               .when('.between(0,0)', p['p_jirovecii_inf_causing_pneumonia'][0])
                               .when('.between(1,1)', p['p_jirovecii_inf_causing_pneumonia'][1])
                               .when('.between(2,4)', p['p_jirovecii_inf_causing_pneumonia'][2])
                               .otherwise(0.0),
                               ),
            'other_pathogens': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('ri_primary_ALRI_pathogen')
                               .when('is not other_pathogens', 0.0),
                               Predictor('age_years')
                               .when('.between(0,0)', p['p_other_pathogens_inf_causing_pneumonia'][0])
                               .when('.between(1,1)', p['p_other_pathogens_inf_causing_pneumonia'][1])
                               .when('.between(2,4)', p['p_other_pathogens_inf_causing_pneumonia'][2])
                               .otherwise(0.0),
                               ),
        })

        # Create linear model equation for the probability of a secondary bacterial infection
        self.prob_secondary_bacterial_infection = LinearModel(LinearModelType.MULTIPLICATIVE,
                                         1.0,
                                         Predictor()
                                         .when("ri_primary_ALRI_pathogen.isin(['RSV', 'rhinovirus', 'hMPV', "
                                               "'parainfluenza', 'influenza']) & "
                                               "ri_current_ALRI_disease_type == 'viral pneumonia'",
                                               p['prob_secondary_bacterial_infection_from_viral_pneumonia'])
                                         .otherwise(0.0),
                                         Predictor()
                                         .when("ri_primary_ALRI_pathogen.isin(['RSV', 'rhinovirus', 'hMPV', "
                                               "'parainfluenza', 'influenza']) & "
                                               "ri_current_ALRI_disease_type == 'bronchiolitis'",
                                               p['prob_secondary_bacterial_infection_from_bronchiolitis'])
                                         .otherwise(0.0)
                                         )


        # check that probability of symptoms have been declared for each severity level
        # assert self.severity == set(list(self.prob_symptoms.keys()))
        # --------------------------------------------------------------------------------------------
        # Create linear models for the risk of acquiring complications from uncomplicated ALRI
        self.risk_of_developing_ALRI_complications.update({
            'pneumothorax':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['streptococcus', 'hib', 'TB', 'staphylococcus', 'other_pathogens'])",
                                p['prob_pneumothorax_by_bacterial_ALRI'])
                            .otherwise(0.0)
                            ),

            'pleural_effusion':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['streptococcus', 'hib', 'TB', 'staphylococcus', 'other_pathogens'])",
                                p['prob_pleural_effusion_by_bacterial_ALRI'])
                            .otherwise(0.0)
                            ),

            'empyema':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_current_ALRI_complications')
                            .when('pleural_effusion', p['prob_pleural_effusion_to_empyema'])
                            .otherwise(0.0)
                            ),

            'lung_abscess':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['streptococcus', 'hib', 'TB', 'staphylococcus', 'other_pathogens'])",
                                p['prob_lung_abscess_by_bacterial_ALRI'])
                            .otherwise(0.0)
                            ),

            'sepsis':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['streptococcus', 'hib', 'TB', 'staphylococcus', 'other_pathogens'])",
                                p['prob_sepsis_by_bacterial_ALRI'])
                            .when(
                                ".isin(['RSV', 'rhinovirus', 'hMPV', 'parainfluenza', 'influenza'])",
                                p['prob_sepsis_by_viral_ALRI'])
                            .otherwise(0.0)
                            ),

            'meningitis':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['streptococcus', 'hib', 'TB', 'staphylococcus', 'other_pathogens'])",
                                p['prob_meningitis_by_bacterial_ALRI'])
                            .when(
                                ".isin(['RSV', 'rhinovirus', 'hMPV', 'parainfluenza', 'influenza'])",
                                p['prob_meningitis_by_viral_ALRI'])
                            .otherwise(0.0)
                            ),

            'respiratory_failure':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('ri_primary_ALRI_pathogen')
                            .when(
                                ".isin(['streptococcus', 'hib', 'TB', 'staphylococcus', 'other_pathogens'])",
                                p['prob_respiratory_failure_by_bacterial_ALRI'])

                            .when(
                                ".isin(['RSV', 'rhinovirus', 'hMPV', 'parainfluenza', 'influenza'])",
                                p['prob_respiratory_failure_by_viral_ALRI'])
                            .otherwise(0.0)
                            ),
        }),

        # --------------------------------------------------------------------------------------------
        # Make a dict containing the probability of symptoms onset given acquisition of ALRI
        self.prob_symptoms_uncomplicated_ALRI.update({
            'viral_pneumonia': {
                'fever': 0.7,
                'cough': 0.8,
                'difficult_breathing': 1,
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 0
            },
            'bacterial_pneumonia': {
                'fever': 0.7,
                'cough': 0.8,
                'difficult_breathing': 1,
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 0
            },
            'bronchiolitis': {
                'fever': 0.7,
                'cough': 0.8,
                'difficult_breathing': 1,
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 0
            }
        })

        self.prob_extra_symptoms_complications.update({
            'pneumothorax': {
                'grunting': 0.9,
                'severe_respiratory_distress': 0.5,
                'cyanosis': 1
            },
            'pleural_effusion': {
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 1
            },
            'empyema': {
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 1
            },
            'lung_abscess': {
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 1
            },
            'sepsis': {
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 1
            },
            'meningitis': {
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 1
            },
            'respiratory_failure': {
                'hypoxia': 0.7,
                'danger_signs': 1,
                'stridor': 0.7
            }
        })
        # TODO: add the probabilities of symptoms by severity - in parameters

        # --------------------------------------------------------------------------------------------
        # Create the linear model for the risk of dying due to ALRI
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

        # TODO: duration of ilness - mean 3.0 days (2.0-5.0 days) from PERCH/hopsitalization days

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_population(self, population):
        """Set our property values for the initial population.
        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng
        now = self.sim.date

        # ---- Key Current Status Classification Properties ----
        df['ri_primary_ALRI_pathogen'].values[:] = 'none'

        # ---- Internal values ----
        df['ri_ALRI_event_date_of_onset'] = pd.NaT
        df['ri_ALRI_event_recovered_date'] = pd.NaT
        df['ri_ALRI_event_death_date'] = pd.NaT

        df['ri_ALRI_treatment'] = False
        df['ri_ALRI_tx_start_date'] = pd.NaT

        # ---- Temporary values ----
        df['tmp_malnutrition'] = False
        df['tmp_hv_inf'] = False
        df['tmp_low_birth_weight'] = False
        df['tmp_exclusive_breastfeeding'] = False
        df['tmp_continued_breastfeeding'] = False

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # Schedule the main polling event (to first occur immidiately)
        sim.schedule_event(AcuteLowerRespiratoryInfectionPollingEvent(self), sim.date + DateOffset(months=0))

        # Schedule the main logging event (to first occur in one year)
        sim.schedule_event(PneumoniaLoggingEvent(self), sim.date + DateOffset(years=1))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        # ---- Key Current Status Classification Properties ----
        df.at[child_id, 'gi_primary_ALRI_pathogen'] = 'none'

        # ---- Internal values ----
        df.at[child_id, 'gi_ALRI_event_date_of_onset'] = pd.NaT
        df.at[child_id, 'gi_ALRI_event_recovered_date'] = pd.NaT
        df.at[child_id, 'gi_ALRI_event_death_date'] = pd.NaT

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

        logger.debug('This is Pneumonia, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)
        pass

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        pass

        # logger.debug('This is ALRI reporting my health values')
        # df = self.sim.population.props
        # p = self.parameters
        #
        # total_daly_values = pd.Series(data=0.0, index=df.loc[df['is_alive']].index)
        # total_daly_values.loc[
        #     self.sim.modules['SymptomManager'].who_has('fast_breathing')
        # ] = self.daly_wts['daly_ALRI']
        # total_daly_values.loc[
        #     self.sim.modules['SymptomManager'].who_has('chest_indrawing')
        # ] = self.daly_wts['daly_severe_ALRI']
        # total_daly_values.loc[
        #     self.sim.modules['SymptomManager'].who_has('danger_signs')
        # ] = self.daly_wts['daly_severe_ALRI']
        #
        # # health_values = df.loc[df.is_alive, 'ri_specific_symptoms'].map({
        # #     'none': 0,
        # #     'ALRI': p['daly_ALRI'],
        # #     'severe ALRI': p['daly_severe_ALRI'],
        # #     'very severe ALRI': p['daly_very_severe_ALRI']
        # # })
        # # health_values.name = 'Pneumonia Symptoms'  # label the cause of this disability
        # # return health_values.loc[df.is_alive]  # returns the series
        #
        # # Split out by pathogen that causes the ALRI
        # dummies_for_pathogen = pd.get_dummies(df.loc[total_daly_values.index,
        #                                              'ri_primary_ALRI_pathogen'],
        #                                       dtype='float')
        # daly_values_by_pathogen = dummies_for_pathogen.mul(total_daly_values, axis=0).drop(columns='none')
        #
        # return daly_values_by_pathogen


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------
class AcuteLowerRespiratoryInfectionPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the main event that runs the acquisition of pathogens that cause ALRI.
        It determines who is infected and when and schedules individual IncidentCase events to represent onset.
        A known issue is that ALRI events are scheduled based on the risk of current age but occur a short time
        later when the children have aged.
        """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = self.module.rng

        # Compute the incidence rate for each person getting ALRI and then convert into a probability
        # getting all children that do not have ALRI currently
        mask_could_get_new_alri_event = \
            df['is_alive'] & (df['age_years'] < 5) & ((df['ri_ALRI_event_recovered_date'] <= self.sim.date) |
                                                      pd.isnull(df['ri_ALRI_event_recovered_date']))

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
        # (Assumes that pathogens are mutually exclusive)
        prob_of_acquiring_any_pathogen = probs_of_acquiring_pathogen.sum(axis=1)
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

            # ----------------------- Allocate a date of onset of ALRI ----------------------
            date_onset = self.sim.date + DateOffset(days=np.random.randint(0, days_until_next_polling_event))
            # duration
            duration_in_days_of_alri = max(1, int(
                7 + (-2 + 4 * rng.rand())))  # assumes uniform interval around mean duration with range 4 days

            # ----------------------- Allocate symptoms to onset of ALRI ----------------------
            # possible_symptoms_by_severity = m.prob_symptoms_uncomplicated_ALRI
            # symptoms_for_this_person = list()
            # for symptom, prob in possible_symptoms_by_severity.items():
            #     if rng.rand() < prob:
            #         symptoms_for_this_person.append(symptom)

            # ----------------------- Create the event for the onset of infection -------------------
            # NB. The symptoms are scheduled by the SymptomManager to 'autoresolve' after the duration
            #       of the ALRI.
            self.sim.schedule_event(
                event=AcuteLowerRespiratoryInfectionIncidentCase(
                    module=self.module,
                    person_id=person_id,
                    pathogen=pathogen,
                    duration_in_days=duration_in_days_of_alri,
                    # symptoms=symptoms_for_this_person
                ),
                date=date_onset
            )


class AcuteLowerRespiratoryInfectionIncidentCase(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of the infection that causes ALRI.
    """

    def __init__(self, module, person_id, pathogen, duration_in_days):
        super().__init__(module, person_id=person_id)
        self.pathogen = pathogen
        self.duration_in_days = duration_in_days
        # self.symptoms = symptoms

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        rng = self.module.rng

        # The event should not run if the person is not currently alive
        if not df.at[person_id, 'is_alive']:
            return

        # Update the properties in the dataframe:
        df.at[person_id, 'ri_primary_ALRI_pathogen'] = self.pathogen
        df.at[person_id, 'ri_ALRI_event_date_of_onset'] = self.sim.date
        df.at[person_id, 'ri_current_ALRI_complications'] = 'none' # all disease start as non-severe symptoms
        df.at[person_id, 'ri_current_ALRI_symptoms'] = self.symptoms

        for pathogen, prob in m.proportions_of_ALRI_diseases_caused_by_each_pathogen.items():
            if rng.rand() < prob:
        # determine if it is viral or bacterial ALRI based on pathogen
        if self.pathogen == self.module.pathogen_type['viral']:
            if
            df.at[person_id, 'ri_current_ALRI_disease_type'] = 'primarily viral'
        if self.pathogen == self.module.pathogen_type['bacterial']:
            df.at[person_id, 'ri_current_ALRI_disease_type'] = 'primarily bacterial'

        # ----------------------- Allocate symptoms to onset of ALRI ----------------------
        possible_symptoms_by_severity = m.prob_symptoms_uncomplicated_ALRI[disease_type]
        symptoms_for_this_person = list()
        for symptom, prob in possible_symptoms_by_severity.items():
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

        date_of_outcome = self.module.sim.date + DateOffset(days=self.duration_in_days)

        complications_for_this_person = list()
        for complication in self.module.complications:
            prob_developing_each_complication = m.risk_of_developing_ALRI_complications[complication].predict(
                df.loc[[person_id]]).values[0]
            if rng.rand() < prob_developing_each_complication:
                complications_for_this_person.append(complication)
                df.at[person_id, 'ri_ALRI_event_recovered_date'] = pd.NaT
                df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT
            else:
                df.at[person_id, 'ri_ALRI_event_recovered_date'] = date_of_outcome
                df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT

        if len(complications_for_this_person) != 0:
            for i in complications_for_this_person:
                date_onset_complications = self.module.sim.date + DateOffset(
                    days=np.random.randint(3, high=self.duration_in_days))
                print(i, date_onset_complications)
                self.sim.schedule_event(PneumoniaWithComplicationsEvent(
                    self.module, person_id, duration_in_days=self.duration_in_days, symptoms=self.symptoms,
                    complication=complications_for_this_person), date_onset_complications)

        self.sim.modules['DxAlgorithmChild'].imnci_as_gold_standard(person_id=person_id)

        # Add this incident case to the tracker
        age = df.loc[person_id, ['age_years']]
        if age.values[0] < 5:
            age_grp = age.map({0: '0y', 1: '1y', 2: '2-4y', 3: '2-4y', 4: '2-4y'}).values[0]
        else:
            age_grp = '5+y'
        self.module.incident_case_tracker[age_grp][self.pathogen].append(self.sim.date)


class PneumoniaWithComplicationsEvent(Event, IndividualScopeEventMixin):
        """
            This Event is for the onset of Clinical Severe Pneumonia. For some untreated children,
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
            df.at[person_id, 'ri_current_ALRI_complications'] = list(self.complication)

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

            prob_death_from_ALRI =\
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

        # Stop the person from dying of ALRI (if they were going to die)
        df.at[person_id, 'ri_ALRI_event_recovered_date'] = self.sim.date
        df.at[person_id, 'ri_ALRI_event_death_date'] = pd.NaT

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
        # Check if person should still die of ALRI
        if (df.at[person_id, 'is_alive']) and \
            (df.at[person_id, 'ri_ALRI_event_death_date'] == self.sim.date):
            self.sim.schedule_event(demography.InstantaneousDeath(self.module,
                                                                  person_id,
                                                                  cause='Pneumonia_' + df.at[
                                                                      person_id, 'ri_primary_ALRI_pathogen']
                                                                  ), self.sim.date)


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class PneumoniaLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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

        logger.info('%s|incidence_count_by_pathogen|%s',
                    self.sim.date,
                    counts
                    )

        imci_classification_count = \
            df[df.is_alive & df.age_years.between(0, 5)].groupby('ri_pneumonia_IMCI_classification').size()

        logger.info('%s|imci_classicications_count|%s',
                    self.sim.date,
                    imci_classification_count
                    )

        # Reset the counters and the date_last_run
        self.module.incident_case_tracker = copy.deepcopy(self.module.incident_case_tracker_blank)
        self.date_last_run = self.sim.date
