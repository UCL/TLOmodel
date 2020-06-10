"""
Childhood pneumonia module
Documentation: 04 - Methods Repository/Method_Child_RespiratoryInfection.xlsx
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

class Pneumonia(Module):
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

    # Declare the severity levels of the disease:
    pathogen_type = {
        'viral': 'RSV' 'rhinovirus' 'hMPV' 'parainfluenza' 'influenza',
        'bacterial': 'streptococcus' 'hib' 'TB' 'staphylococcus'
    }


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
                                                    categories=['pneumothorax', 'pleural efusion', 'empyema',
                                                                'lung abscess', 'sepsis', 'meningitis',
                                                                'respiratory failure'] + ['none']
                                                    ),

        # ---- Internal variables to schedule onset and deaths due to pneumonia ----
        'ri_last_pneumonia_date_of_onset': Property(Types.DATE, 'date of onset of last pneumonia event'),
        'ri_last_pneumonia_recovered_date': Property(Types.DATE, 'date of recovery from last pneumonia event'),
        'ri_last_pneumonia_death_date': Property(Types.DATE, 'date of death caused by last pneumonia event'),

        # ---- Temporary Variables: To be replaced with the properties of other modules ----
        'tmp_malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'tmp_low_birth_weight': Property(Types.BOOL, 'temporary property - low birth weight'),
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
    SYMPTOMS = {'fever', 'cough', 'difficult_breathing', 'fast_breathing', 'chest_indrawing', 'danger_signs'}

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # equations for the incidence of pneumonia by pathogen:
        self.incidence_equations_by_pathogen = dict()

        # equations for the development of pneumonia-associated complications:
        self.risk_of_developing_pneumonia_complications = dict()

        # Linear Model for predicting the risk of death:
        self.risk_of_death_severe_pneumonia = dict()

        # dict to hold the probability of onset of different types of symptom given underlying complications:
        self.prob_symptoms = dict()

        # dict to hold the DALY weights
        self.daly_wts = dict()

        # dict to hold counters for the number of pneumonia events by pathogen and age-group
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

        # Check that every value has been read-in successfully
        for param_name, type in self.PARAMETERS.items():
            assert param_name in self.parameters, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert isinstance(self.parameters[param_name],
                          type.python_type), f'Parameter "{param_name}" is not read in correctly from the resourcefile.'

        # DALY weights
        if 'HealthBurden' in self.sim.modules.keys():
            p['daly_pneumonia'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
            p['daly_severe_pneumonia'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
            p['daly_very_severe_pneumonia'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=46)

        # --------------------------------------------------------------------------------------------
        # Make a dict to hold the equations that govern the probability that a person acquires pneumonia
        # that is caused (primarily) by a pathogen

        self.incidence_equations_by_pathogen.update({
            'RSV': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('age_years')
                               .when('.between(0,0)', p['base_inc_rate_pneumonia_by_RSV'][0])
                               .when('.between(1,1)', p['base_inc_rate_pneumonia_by_RSV'][1])
                               .when('.between(2,4)', p['base_inc_rate_pneumonia_by_RSV'][2])
                               .otherwise(0.0),
                               Predictor('li_no_access_handwashing')
                               .when(False, p['rr_pneumonia_HHhandwashing']),
                               Predictor('li_wood_burn_stove')
                               .when(False, p['rr_pneumonia_indoor_air_pollution']),
                               Predictor('hv_inf')
                               .when(True, p['rr_pneumonia_HIV']),
                               Predictor('tmp_malnutrition')
                               .when(True, p['rr_pneumonia_SAM']),
                               Predictor('tmp_exclusive_breastfeeding')
                               .when(False, p['rr_pneumonia_excl_breast'])
                               )
        })

        self.incidence_equations_by_pathogen.update({
            'rhinovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                      1.0,
                                      Predictor('age_years')
                                      .when('.between(0,0)', p['base_inc_rate_pneumonia_by_rhinovirus'][0])
                                      .when('.between(1,1)', p['base_inc_rate_pneumonia_by_rhinovirus'][1])
                                      .when('.between(2,4)', p['base_inc_rate_pneumonia_by_rhinovirus'][2])
                                      .otherwise(0.0),
                                      Predictor('li_no_access_handwashing')
                                      .when(False, p['rr_pneumonia_HHhandwashing']),
                                      Predictor('li_wood_burn_stove')
                                      .when(False, p['rr_pneumonia_indoor_air_pollution']),
                                      Predictor('hv_inf')
                                      .when(True, p['rr_pneumonia_HIV']),
                                      Predictor('tmp_malnutrition')
                                      .when(True, p['rr_pneumonia_SAM']),
                                      Predictor('tmp_exclusive_breastfeeding')
                                      .when(False, p['rr_pneumonia_excl_breast'])
                                      )
        })

        self.incidence_equations_by_pathogen.update({
            'hMPV': LinearModel(LinearModelType.MULTIPLICATIVE,
                                1.0,
                                Predictor('age_years')
                                .when('.between(0,0)', p['base_inc_rate_pneumonia_by_hMPV'][0])
                                .when('.between(1,1)', p['base_inc_rate_pneumonia_by_hMPV'][1])
                                .when('.between(2,4)', p['base_inc_rate_pneumonia_by_hMPV'][2])
                                .otherwise(0.0),
                                Predictor('li_no_access_handwashing')
                                .when(False, p['rr_pneumonia_HHhandwashing']),
                                Predictor('li_wood_burn_stove')
                                .when(False, p['rr_pneumonia_indoor_air_pollution']),
                                Predictor('hv_inf')
                                .when(True, p['rr_pneumonia_HIV']),
                                Predictor('tmp_malnutrition')
                                .when(True, p['rr_pneumonia_SAM']),
                                Predictor('tmp_exclusive_breastfeeding')
                                .when(False, p['rr_pneumonia_excl_breast'])
                                )
        })

        self.incidence_equations_by_pathogen.update({
            'parainfluenza': LinearModel(LinearModelType.MULTIPLICATIVE,
                                         1.0,
                                         Predictor('age_years')
                                         .when('.between(0,0)', p['base_inc_rate_pneumonia_by_parainfluenza'][0])
                                         .when('.between(1,1)', p['base_inc_rate_pneumonia_by_parainfluenza'][1])
                                         .when('.between(2,4)', p['base_inc_rate_pneumonia_by_parainfluenza'][2])
                                         .otherwise(0.0),
                                         Predictor('li_no_access_handwashing')
                                         .when(False, p['rr_pneumonia_HHhandwashing']),
                                         Predictor('li_wood_burn_stove')
                                         .when(False, p['rr_pneumonia_indoor_air_pollution']),
                                         Predictor('hv_inf')
                                         .when(True, p['rr_pneumonia_HIV']),
                                         Predictor('tmp_malnutrition')
                                         .when(True, p['rr_pneumonia_SAM']),
                                         Predictor('tmp_exclusive_breastfeeding')
                                         .when(False, p['rr_pneumonia_excl_breast'])
                                         )
        })

        self.incidence_equations_by_pathogen.update({
            'streptococcus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                         1.0,
                                         Predictor('age_years')
                                         .when('.between(0,0)', p['base_inc_rate_pneumonia_by_streptococcus'][0])
                                         .when('.between(1,1)', p['base_inc_rate_pneumonia_by_streptococcus'][1])
                                         .when('.between(2,4)', p['base_inc_rate_pneumonia_by_streptococcus'][2])
                                         .otherwise(0.0),
                                         Predictor('li_no_access_handwashing')
                                         .when(False, p['rr_pneumonia_HHhandwashing']),
                                         Predictor('li_wood_burn_stove')
                                         .when(False, p['rr_pneumonia_indoor_air_pollution']),
                                         Predictor('hv_inf')
                                         .when(True, p['rr_pneumonia_HIV']),
                                         Predictor('tmp_malnutrition')
                                         .when(True, p['rr_pneumonia_SAM']),
                                         Predictor('tmp_exclusive_breastfeeding')
                                         .when(False, p['rr_pneumonia_excl_breast']),
                                         Predictor('tmp_pneumococcal_vaccination')
                                         .when(False, p['rr_pneumonia_pneumococcal_vaccine'])
                                         )
        })

        self.incidence_equations_by_pathogen.update({
            'hib': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('age_years')
                               .when('.between(0,0)', p['base_inc_rate_pneumonia_by_hib'][0])
                               .when('.between(1,1)', p['base_inc_rate_pneumonia_by_hib'][1])
                               .when('.between(2,4)', p['base_inc_rate_pneumonia_by_hib'][2])
                               .otherwise(0.0),
                               Predictor('li_no_access_handwashing')
                               .when(False, p['rr_pneumonia_HHhandwashing']),
                               Predictor('li_wood_burn_stove')
                               .when(False, p['rr_pneumonia_indoor_air_pollution']),
                               Predictor('hv_inf')
                               .when(True, p['rr_pneumonia_HIV']),
                               Predictor('tmp_malnutrition')
                               .when(True, p['rr_pneumonia_SAM']),
                               Predictor('tmp_exclusive_breastfeeding')
                               .when(False, p['rr_pneumonia_excl_breast']),
                               Predictor('tmp_hib_vaccination')
                               .when(False, p['rr_pneumonia_hib_vaccine'])
                               )
        })

        self.incidence_equations_by_pathogen.update({
            'TB': LinearModel(LinearModelType.MULTIPLICATIVE,
                              1.0,
                              Predictor('age_years')
                              .when('.between(0,0)', p['base_inc_rate_pneumonia_by_TB'][0])
                              .when('.between(1,1)', p['base_inc_rate_pneumonia_by_TB'][1])
                              .when('.between(2,4)', p['base_inc_rate_pneumonia_by_TB'][2])
                              .otherwise(0.0),
                              Predictor('ri_last_pneumonia_pathogen')
                              .when('is not TB', 0.0),
                              Predictor('li_no_access_handwashing')
                              .when(False, p['rr_pneumonia_HHhandwashing']),
                              Predictor('li_wood_burn_stove')
                              .when(False, p['rr_pneumonia_indoor_air_pollution']),
                              Predictor('hv_inf')
                              .when(True, p['rr_pneumonia_HIV']),
                              Predictor('tmp_malnutrition')
                              .when(True, p['rr_pneumonia_SAM']),
                              Predictor('tmp_exclusive_breastfeeding')
                              .when(False, p['rr_pneumonia_excl_breast'])
                              )
        })

        self.incidence_equations_by_pathogen.update({
            'staphylococcus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                          1.0,
                                          Predictor('age_years')
                                          .when('.between(0,0)', p['base_inc_rate_pneumonia_by_staphylococcus'][0])
                                          .when('.between(1,1)', p['base_inc_rate_pneumonia_by_staphylococcus'][1])
                                          .when('.between(2,4)', p['base_inc_rate_pneumonia_by_staphylococcus'][2])
                                          .otherwise(0.0),
                                          Predictor('li_no_access_handwashing')
                                          .when(False, p['rr_pneumonia_HHhandwashing']),
                                          Predictor('li_wood_burn_stove')
                                          .when(False, p['rr_pneumonia_indoor_air_pollution']),
                                          Predictor('hv_inf')
                                          .when(True, p['rr_pneumonia_HIV']),
                                          Predictor('tmp_malnutrition')
                                          .when(True, p['rr_pneumonia_SAM']),
                                          Predictor('tmp_exclusive_breastfeeding')
                                          .when(False, p['rr_pneumonia_excl_breast'])
                                          )
        })

        self.incidence_equations_by_pathogen.update({
            'influenza': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('age_years')
                                     .when('.between(0,0)', p['base_inc_rate_pneumonia_by_influenza'][0])
                                     .when('.between(1,1)', p['base_inc_rate_pneumonia_by_influenza'][1])
                                     .when('.between(2,4)', p['base_inc_rate_pneumonia_by_influenza'][2])
                                     .otherwise(0.0),
                                     Predictor('li_no_access_handwashing')
                                     .when(False, p['rr_pneumonia_HHhandwashing']),
                                     Predictor('li_wood_burn_stove')
                                     .when(False, p['rr_pneumonia_indoor_air_pollution']),
                                     Predictor('hv_inf')
                                     .when(True, p['rr_pneumonia_HIV']),
                                     Predictor('tmp_malnutrition')
                                     .when(True, p['rr_pneumonia_SAM']),
                                     Predictor('tmp_exclusive_breastfeeding')
                                     .when(False, p['rr_pneumonia_excl_breast'])
                                     )
        })

        self.incidence_equations_by_pathogen.update({
            'jirovecii': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('age_years')
                                     .when('.between(0,0)', p['base_inc_rate_pneumonia_by_jirovecii'][0])
                                     .when('.between(1,1)', p['base_inc_rate_pneumonia_by_jirovecii'][1])
                                     .when('.between(2,4)', p['base_inc_rate_pneumonia_by_jirovecii'][2])
                                     .otherwise(0.0),
                                     Predictor('li_no_access_handwashing')
                                     .when(False, p['rr_pneumonia_HHhandwashing']),
                                     Predictor('li_wood_burn_stove')
                                     .when(False, p['rr_pneumonia_indoor_air_pollution']),
                                     Predictor('hv_inf')
                                     .when(True, p['rr_pneumonia_HIV']),
                                     Predictor('tmp_malnutrition')
                                     .when(True, p['rr_pneumonia_SAM']),
                                     Predictor('tmp_exclusive_breastfeeding')
                                     .when(False, p['rr_pneumonia_excl_breast'])
                                     )
        })

        self.incidence_equations_by_pathogen.update({
            'other_pathogens': LinearModel(LinearModelType.MULTIPLICATIVE,
                                           1.0,
                                           Predictor('age_years')
                                           .when('.between(0,0)', p['base_inc_rate_pneumonia_by_other_pathogens'][0])
                                           .when('.between(1,1)', p['base_inc_rate_pneumonia_by_other_pathogens'][1])
                                           .when('.between(2,4)', p['base_inc_rate_pneumonia_by_other_pathogens'][2])
                                           .otherwise(0.0),
                                           Predictor('li_no_access_handwashing')
                                           .when(False, p['rr_pneumonia_HHhandwashing']),
                                           Predictor('li_wood_burn_stove')
                                           .when(False, p['rr_pneumonia_indoor_air_pollution']),
                                           Predictor('hv_inf')
                                           .when(True, p['rr_pneumonia_HIV']),
                                           Predictor('tmp_malnutrition')
                                           .when(True, p['rr_pneumonia_SAM']),
                                           Predictor('tmp_exclusive_breastfeeding')
                                           .when(False, p['rr_pneumonia_excl_breast'])
                                           )
        })

        # check that equations have been declared for each pathogens
        assert self.pathogens == set(list(self.incidence_equations_by_pathogen.keys()))

        # --------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------
        # Make a dict containing the probability of symptoms onset given acquisition of pneumonia
        self.prob_symptoms.update({
            'uncomplicated': {
                'fever': 0.7,
                'cough': 0.8,
                'difficult_breathing': 1,
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 0
            },
            'pneumothorax': {
                'fever': 0.7,
                'cough': 0.8,
                'difficult_breathing': 1,
                'grunting': 0.9,
                'sereve_respiratory_distress': 0.5,
                'cyanosis': 1
            },
            'pleural_effusion': {
                'fever': 0.7,
                'cough': 0.8,
                'difficult_breathing': 1,
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 1
            },
            'empyema': {
                'fever': 0.7,
                'cough': 0.8,
                'difficult_breathing': 1,
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 1
            },
            'lung_abscess': {
                'fever': 0.7,
                'cough': 0.8,
                'difficult_breathing': 1,
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 1
            },
            'sepsis': {
                'fever': 0.7,
                'cough': 0.8,
                'difficult_breathing': 1,
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 1
            },
            'menigitis': {
                'fever': 0.7,
                'cough': 0.8,
                'difficult_breathing': 1,
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 1
            },
            'respiratory_failure': {
                'fever': 0.7,
                'cough': 0.8,
                'difficult_breathing': 1,
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 1
            }
        })
        # TODO: add the probabilities of symptoms by severity - in parameters

        # check that probability of symptoms have been declared for each severity level
        # assert self.severity == set(list(self.prob_symptoms.keys()))
        # --------------------------------------------------------------------------------------------
        # Create linear models for the risk of acquiring complications from uncomplicated pneumonia
        self.risk_of_developing_pneumonia_complications.update({
            'pneumothorax': LinearModel(LinearModelType.MULTIPLICATIVE,
                                        1.0,
                                        Predictor('ri_last_pneumonia_pathogen')
                                        .when('is streptococcus| hib | TB | staphylococcus',
                                              p['prob_pneumothorax_bacterial_pneumonia'])
                                        .otherwise(0.0)
                                        ),

            'pleural_effusion': LinearModel(LinearModelType.MULTIPLICATIVE,
                                            1.0,
                                            Predictor('ri_last_pneumonia_pathogen')
                                            .when('is streptococcus| hib | TB | staphylococcus',
                                                  p['prob_pleural_effusion_by_bacterial_pneumonia'])
                                            .otherwise(0.0)
                                            ),

            'lung_abscess': LinearModel(LinearModelType.MULTIPLICATIVE,
                                        1.0,
                                        Predictor('ri_last_pneumonia_pathogen')
                                        .when('is streptococcus| hib | TB | staphylococcus',
                                              p['prob_pleural_effusion_by_bacterial_pneumonia'])
                                        .otherwise(0.0)
                                        ),

            'sepsis': LinearModel(LinearModelType.MULTIPLICATIVE,
                                  1.0,
                                  Predictor('ri_last_pneumonia_pathogen')
                                  .when('is streptococcus| hib | TB | staphylococcus',
                                        p['prob_sepsis_by_bacterial_pneumonia'])
                                  .when('is RSV | rhinovirus | hMPV | parainfluenza | influenza',
                                        p['prob_sepsis_by_viral_pneumonia'])
                                  .otherwise(0.0)
                                  ),

            'meningitis': LinearModel(LinearModelType.MULTIPLICATIVE,
                                      1.0,
                                      Predictor('ri_last_pneumonia_pathogen')
                                      .when('is streptococcus| hib | TB | staphylococcus',
                                            p['prob_meningitis_by_bacterial_pneumonia'])
                                      .when('is RSV | rhinovirus | hMPV | parainfluenza | influenza',
                                            p['prob_meningitis_by_viral_pneumonia'])
                                      .otherwise(0.0)
                                      ),

            'respiratory_failure': LinearModel(LinearModelType.MULTIPLICATIVE,
                                               1.0,
                                               Predictor('ri_last_pneumonia_pathogen')
                                               .when('is streptococcus| hib | TB | staphylococcus',
                                                     p['prob_respiratory_failure_by_bacterial_pneumonia'])
                                               .when('is RSV | rhinovirus | hMPV | parainfluenza | influenza',
                                                     p['prob_respiratory_failure_by_viral_pneumonia'])
                                               .otherwise(0.0)
                                               ),
        }),

        # --------------------------------------------------------------------------------------------
        # Create the linear model for the risk of dying due to pneumonia
        self.risk_of_death_severe_pneumonia =\
            LinearModel(LinearModelType.MULTIPLICATIVE,
                        1.0,
                        Predictor('ri_last_pneumonia_complications')
                        .when('sepsis', p['r_death_from_pneumonia_due_to_sepsis'])
                        .when('respiratory_failure', p['r_death_from_pneumonia_due_to_respiratory_failure'])
                        .when('meningitis', p['r_death_from_pneumonia_due_to_meningitis'])
                        .otherwise(0.0),
                        Predictor('hv_inf').when(True, p['rr_death_pneumonia_HIV']),
                        Predictor('tmp_malnutrition').when(True, p['rr_death_pneumonia_malnutrition']),
                        Predictor('tmp_low_birth_weight').when(True, p['rr_death_pneumonia_lbw']),
                        Predictor('age_years')
                        .when('.between(1,1)', p['rr_death_pneumonia_age12to23mo'])
                        .when('.between(2,4)', p['rr_death_pneumonia_age24to59mo'])
                        )

        # TODO: duration of ilness - mean 3.0 days (2.0-5.0 days) from PERCH/hopsitalization days

        # Register this disease module with the health system
        # self.sim.modules['HealthSystem'].register_disease_module(self)

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
        df['ri_last_pneumonia_pathogen'].values[:] = 'none'

        # ---- Internal values ----
        df['ri_last_pneumonia_date_of_onset'] = pd.NaT
        df['ri_last_pneumonia_recovered_date'] = pd.NaT
        df['ri_last_pneumonia_death_date'] = pd.NaT

        df['ri_pneumonia_treatment'] = False
        df['ri_pneumonia_tx_start_date'] = pd.NaT

        # ---- Temporary values ----
        df['tmp_malnutrition'] = False
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
        sim.schedule_event(PneumoniaPollingEvent(self), sim.date + DateOffset(months=0))

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
        df.at[child_id, 'gi_last_pneumonia_pathogen'] = 'none'
        df.at[child_id, 'gi_last_clinical_severity'] = 'none'

        # ---- Internal values ----
        df.at[child_id, 'gi_last_pneumonia_date_of_onset'] = pd.NaT
        df.at[child_id, 'gi_last_pneumonia_recovered_date'] = pd.NaT
        df.at[child_id, 'gi_last_pneumonia_death_date'] = pd.NaT

        # ---- Temporary values ----
        df.at[child_id, 'tmp_malnutrition'] = False
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

        # logger.debug('This is pneumonia reporting my health values')
        # df = self.sim.population.props
        # p = self.parameters
        #
        # total_daly_values = pd.Series(data=0.0, index=df.loc[df['is_alive']].index)
        # total_daly_values.loc[
        #     self.sim.modules['SymptomManager'].who_has('fast_breathing')
        # ] = self.daly_wts['daly_pneumonia']
        # total_daly_values.loc[
        #     self.sim.modules['SymptomManager'].who_has('chest_indrawing')
        # ] = self.daly_wts['daly_severe_pneumonia']
        # total_daly_values.loc[
        #     self.sim.modules['SymptomManager'].who_has('danger_signs')
        # ] = self.daly_wts['daly_severe_pneumonia']
        #
        # # health_values = df.loc[df.is_alive, 'ri_specific_symptoms'].map({
        # #     'none': 0,
        # #     'pneumonia': p['daly_pneumonia'],
        # #     'severe pneumonia': p['daly_severe_pneumonia'],
        # #     'very severe pneumonia': p['daly_very_severe_pneumonia']
        # # })
        # # health_values.name = 'Pneumonia Symptoms'  # label the cause of this disability
        # # return health_values.loc[df.is_alive]  # returns the series
        #
        # # Split out by pathogen that causes the pneumonia
        # dummies_for_pathogen = pd.get_dummies(df.loc[total_daly_values.index,
        #                                              'ri_last_pneumonia_pathogen'],
        #                                       dtype='float')
        # daly_values_by_pathogen = dummies_for_pathogen.mul(total_daly_values, axis=0).drop(columns='none')
        #
        # return daly_values_by_pathogen


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------
class PneumoniaPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the main event that runs the acquisition of pathogens that cause Pneumonia.
        It determines who is infected and when and schedules individual IncidentCase events to represent onset.
        A known issue is that pneumonia events are scheduled based on the risk of current age but occur a short time
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

        # Compute the incidence rate for each person getting pneumonia and then convert into a probability
        # getting all children that do not have pneumonia currently
        mask_could_get_new_pneumonia_event = \
            df['is_alive'] & (df['age_years'] < 5) & ((df['ri_last_pneumonia_recovered_date'] <= self.sim.date) |
                                                      pd.isnull(df['ri_last_pneumonia_recovered_date']))

        inc_of_acquiring_pneumonia = pd.DataFrame(index=df.loc[mask_could_get_new_pneumonia_event].index)

        for pathogen in m.pathogens:
            inc_of_acquiring_pneumonia[pathogen] = m.incidence_equations_by_pathogen[pathogen] \
                .predict(df.loc[mask_could_get_new_pneumonia_event])

        # Convert the incidence rates that are predicted by the model into risk of an event occurring before the next
        # polling event
        fraction_of_a_year_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(
            1, 'Y')
        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')
        probs_of_acquiring_pathogen = 1 - np.exp(-inc_of_acquiring_pneumonia * fraction_of_a_year_until_next_polling_event)

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

            # ----------------------- Allocate a date of onset of pneumonia ----------------------
            date_onset = self.sim.date + DateOffset(days=np.random.randint(0, days_until_next_polling_event))
            # duration
            duration_in_days_of_pneumonia = max(1, int(
                7 + (-2 + 4 * rng.rand())))  # assumes uniform interval around mean duration with range 4 days

            # ----------------------- Allocate symptoms to onset of pneumonia ----------------------
            possible_symptoms_by_severity = m.prob_symptoms['non-severe']
            symptoms_for_this_person = list()
            for symptom, prob in possible_symptoms_by_severity.items():
                if rng.rand() < prob:
                    symptoms_for_this_person.append(symptom)

            # ----------------------- Create the event for the onset of infection -------------------
            # NB. The symptoms are scheduled by the SymptomManager to 'autoresolve' after the duration
            #       of the pneumonia.
            self.sim.schedule_event(
                event=PneumoniaIncidentCase(
                    module=self.module,
                    person_id=person_id,
                    pathogen=pathogen,
                    duration_in_days=duration_in_days_of_pneumonia,
                    symptoms=symptoms_for_this_person
                ),
                date=date_onset
            )


class PneumoniaIncidentCase(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of the infection that causes pneumonia.
    """

    def __init__(self, module, person_id, pathogen, duration_in_days, symptoms):
        super().__init__(module, person_id=person_id)
        self.pathogen = pathogen
        self.duration_in_days = duration_in_days
        self.symptoms = symptoms

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        rng = self.module.rng

        # The event should not run if the person is not currently alive
        if not df.at[person_id, 'is_alive']:
            return

        # Update the properties in the dataframe:
        df.at[person_id, 'ri_last_pneumonia_pathogen'] = self.pathogen
        df.at[person_id, 'ri_last_pneumonia_date_of_onset'] = self.sim.date
        df.at[person_id, 'ri_last_clinical_severity'] = 'non-severe' # all disease start as non-severe symptoms

        # Onset symptoms:
        for symptom in self.symptoms:
            self.module.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove='+',
                disease_module=self.module,
                duration_in_days=self.duration_in_days
            )

        # determine if it is viral or bacterial pneumonia based on pathogen
        if self.pathogen == self.module.pathogen_type['viral']:
            df.at[person_id, 'ri_pneumonia_by_pathogen_type'] = 'primarily viral'
        if self.pathogen == self.module.pathogen_type['bacterial']:
            df.at[person_id, 'ri_pneumonia_by_pathogen_type'] = 'primarily bacterial'
            
        # Determine progression to 'severe clinical pneumonia'
        date_of_outcome = self.module.sim.date + DateOffset(days=self.duration_in_days)
        prob_progress_clinical_severe_case = pd.DataFrame(index=[person_id])
        # for disease in m.diseases:
        #     prob_progress_clinical_severe_case = \
        #         m.progression_to_clinical_severe_penumonia[disease].predict(df.loc[[person_id]]).values[0]
        # will_progress_to_severe = rng.rand() < prob_progress_clinical_severe_case

        # if will_progress_to_severe:
        #     df.at[person_id, 'ri_last_pneumonia_recovered_date'] = pd.NaT
        #     df.at[person_id, 'ri_last_pneumonia_death_date'] = pd.NaT
        #     date_onset_clinical_severe = self.module.sim.date + DateOffset(
        #         days=np.random.randint(2, self.duration_in_days))
        #     self.sim.schedule_event(SeverePneumoniaEvent(self.module, person_id,
        #                                                  duration_in_days=self.duration_in_days),
        #                             date_onset_clinical_severe)
        # else:
        #     df.at[person_id, 'ri_last_pneumonia_recovered_date'] = date_of_outcome
        #     df.at[person_id, 'ri_last_pneumonia_death_date'] = pd.NaT

        # Add this incident case to the tracker
        age = df.loc[person_id, ['age_years']]
        if age.values[0] < 5:
            age_grp = age.map({0: '0y', 1: '1y', 2: '2-4y', 3: '2-4y', 4: '2-4y'}).values[0]
        else:
            age_grp = '5+y'
        self.module.incident_case_tracker[age_grp][self.pathogen].append(self.sim.date)


class SeverePneumoniaEvent(Event, IndividualScopeEventMixin):
        """
            This Event is for the onset of Clinical Severe Pneumonia. For some untreated children,
            this occurs a set number of days after onset of disease.
            It sets the property 'ri_last_clinical_severity' to 'severe' and schedules the death.
            """

        def __init__(self, module, person_id, duration_in_days):
            super().__init__(module, person_id=person_id)
            self.duration_in_days = duration_in_days

        def apply(self, person_id):
            df = self.sim.population.props  # shortcut to the dataframe
            m = self.module
            rng = self.module.rng

            # terminate the event if the person has already died.
            if not df.at[person_id, 'is_alive']:
                return

            df.at[person_id, 'ri_last_clinical_severity'] = 'severe'
            possible_symptoms_by_severity = self.module.prob_symptoms['severe']
            symptoms_for_this_person = list()
            for symptom, prob in possible_symptoms_by_severity.items():
                if self.module.rng.rand() < prob:
                    symptoms_for_this_person.append(symptom)

            # Determine death outcome
            date_of_outcome = \
                df.at[person_id, 'ri_last_pneumonia_date_of_onset'] + DateOffset(days=self.duration_in_days)
            prob_death_by_disease = pd.DataFrame(index=[person_id])
            for disease in m.diseases:
                prob_death_by_disease = \
                    m.risk_of_death_clinical_severe_pneumonia[disease].predict(df.loc[[person_id]]).values[0]
            death_outcome = rng.rand() < prob_death_by_disease

            if death_outcome:
                df.at[person_id, 'ri_last_pneumonia_recovered_date'] = pd.NaT
                df.at[person_id, 'ri_last_pneumonia_death_date'] = date_of_outcome
                self.sim.schedule_event(PneumoniaDeathEvent(self.module, person_id),
                                        date_of_outcome)
            else:
                df.at[person_id, 'ri_last_pneumonia_recovered_date'] = date_of_outcome
                df.at[person_id, 'ri_last_pneumonia_death_date'] = pd.NaT


class PneumoniaCureEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("PneumoniaCureEvent: Stopping pneumonia treatment and curing person %d", person_id)
        df = self.sim.population.props

        # terminate the event if the person has already died.
        if not df.at[person_id, 'is_alive']:
            return

        # Stop the person from dying of pneumonia (if they were going to die)
        df.at[person_id, 'ri_last_pneumonia_recovered_date'] = self.sim.date
        df.at[person_id, 'ri_last_pneumonia_death_date'] = pd.NaT

        # clear the treatment prperties
        df.at[person_id, 'ri_pneumonia_treatment'] = False
        df.at[person_id, 'ri_pneumonia_tx_start_date'] = pd.NaT

        # Resolve all the symptoms immediately
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id,
                                                          disease_module=self.sim.modules['Pneumonia'])


class PneumoniaDeathEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the death of someone that is caused by the infection with a pathogen that causes pneumonia.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        # Check if person should still die of pneumonia
        if (df.at[person_id, 'is_alive']) and \
            (df.at[person_id, 'ri_last_pneumonia_death_date'] == self.sim.date):
            self.sim.schedule_event(demography.InstantaneousDeath(self.module,
                                                                  person_id,
                                                                  cause='Pneumonia_' + df.at[
                                                                      person_id, 'ri_last_pneumonia_pathogen']
                                                                  ), self.sim.date)


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

        # Reset the counters and the date_last_run
        self.module.incident_case_tracker = copy.deepcopy(self.module.incident_case_tracker_blank)
        self.date_last_run = self.sim.date
