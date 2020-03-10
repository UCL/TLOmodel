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
        'other pathogens'
    }

    PARAMETERS = {
        'base_incidence_pneumonia_by_RSV': Parameter
        (Types.LIST, 'incidence of pneumonia caused by Respiratory Syncytial Virus in age groups 0-11, 12-59 months'
         ),
        'base_incidence_pneumonia_by_rhinovirus': Parameter
        (Types.LIST, 'incidence of pneumonia caused by rhinovirus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_hMPV': Parameter
        (Types.LIST, 'incidence of pneumonia caused by hMPV in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_parainfluenza': Parameter
        (Types.LIST, 'incidence of pneumonia caused by parainfluenza in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_streptococcus': Parameter
        (Types.LIST, 'incidence of pneumonia caused by streptoccocus 40/41 in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_hib': Parameter
        (Types.LIST, 'incidence of pneumonia caused by hib in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_TB': Parameter
        (Types.LIST, 'incidence of pneumonia caused by TB in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_staphylococcus': Parameter
        (Types.LIST, 'incidence of pneumonia caused by Staphylococcus aureus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_influenza': Parameter
        (Types.LIST, 'incidence of pneumonia caused by influenza in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_incidence_pneumonia_by_jirovecii': Parameter
        (Types.LIST, 'incidence of pneumonia caused by P. jirovecii in age groups 0-11, 12-59 months'
         ),
        'base_incidence_pneumonia_by_other_pathogens': Parameter
        (Types.LIST, 'incidence of pneumonia caused by other pathogens in age groups 0-11, 12-59 months'
         ),
        'rr_ri_pneumonia_HHhandwashing': Parameter
        (Types.REAL, 'relative rate of pneumonia with household handwashing with soap'
         ),
        'rr_ri_pneumonia_HIV': Parameter
        (Types.REAL, 'relative rate of pneumonia for HIV positive status'
         ),
        'rr_ri_pneumonia_SAM': Parameter
        (Types.REAL, 'relative rate of pneumonia for severe malnutrition'
         ),
        'rr_ri_pneumonia_excl_breast': Parameter
        (Types.REAL, 'relative rate of pneumonia for exclusive breastfeeding upto 6 months'
         ),
        'rr_ri_pneumonia_cont_breast': Parameter
        (Types.REAL, 'relative rate of pneumonia for continued breastfeeding 6 months to 2 years'
         ),
        'rr_ri_pneumonia_indoor_air_pollution': Parameter
        (Types.REAL, 'relative rate of pneumonia for indoor air pollution'
         ),
        'rr_ri_pneumonia_pneumococcal_vaccine': Parameter
        (Types.REAL, 'relative rate of pneumonia for pneumonococcal vaccine'
         ),
        'rr_ri_pneumonia_hib_vaccine': Parameter
        (Types.REAL, 'relative rate of pneumonia for hib vaccine'
         ),
        'rr_progress_severe_pneum_viral': Parameter
        (Types.REAL, 'relative rate of pneumonia for viral pathogen'
         ),
        'r_progress_to_severe_pneum': Parameter
        (Types.REAL,
         'probability of progressing from non-severe to severe pneumonia among children aged 2-11 months, '
         'HIV negative, no SAM'
         ),
        'rr_progress_severe_pneum_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for age <2 months'
         ),
        'rr_progress_severe_pneum_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for age 12 to 23 months'
         ),
        'rr_progress_severe_pneum_age24to59mo': Parameter
        (Types.REAL, 'relative rate of progression to severe pneumonia for age 24 to 59 months'
         ),
        'rr_progress_severe_pneum_HIV': Parameter
        (Types.REAL,
         'relative risk of progression to severe pneumonia for HIV positive status'
         ),
        'rr_progress_severe_pneum_SAM': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for severe acute malnutrition'
         ),
        'rr_progress_very_sev_pneum_viral': Parameter
        (Types.REAL, 'relative rate of pneumonia for viral pathogen'
         ),
        'r_progress_to_very_sev_pneum': Parameter
        (Types.REAL,
         'probability of progressing from non-severe to severe pneumonia among children aged 2-11 months, '
         'HIV negative, no SAM'
         ),
        'rr_progress_very_sev_pneum_agelt2mo': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for age <2 months'
         ),
        'rr_progress_very_sev_pneum_age12to23mo': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for age 12 to 23 months'
         ),
        'rr_progress_very_sev_pneum_age24to59mo': Parameter
        (Types.REAL, 'relative rate of progression to severe pneumonia for age 24 to 59 months'
         ),
        'rr_progress_very_sev_pneum_HIV': Parameter
        (Types.REAL,
         'relative risk of progression to severe pneumonia for HIV positive status'
         ),
        'rr_progress_very_sev_pneum_SAM': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for severe acute malnutrition'
         ),
    }

    PROPERTIES = {
        # ---- The pathogen which is the attributed cause of pneumonia ----
        'ri_last_pneumonia_pathogen': Property(Types.CATEGORICAL,
                                               'Attributable pathogen for the last pneumonia event',
                                               categories=list(pathogens) + ['none']),

        # ---- Classification of the severity of pneumonia that is caused ----
        'ri_last_pneumonia_severity': Property(Types.CATEGORICAL,
                                               'severity of pneumonia disease',
                                               categories=['non-severe', 'severe', 'very severe']),

        # ---- Internal variables to schedule onset and deaths due to pneumonia ----
        'ri_last_pneumonia_date_of_onset': Property(Types.DATE, 'date of onset of last pneumonia event'),
        'ri_last_pneumonia_recovered_date': Property(Types.DATE, 'date of recovery from last pneumonia event'),
        'ri_last_pneumonia_death_date': Property(Types.DATE, 'date of death caused by last pneumonia event'),

        # ---- Temporary Variables: To be replaced with the properties of other modules ----
        'tmp_malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
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

        # dict to hold equations in for the incidence of pathogens:
        self.incidence_equations_by_pathogen = dict()

        # Linear Model for predicting the progression of disease from non-severe to severe pneumonia:
        self.progression_to_severe_pneumonia = LinearModel

        # Linear Model for predicting the risk of death:
        self.risk_of_death_pneumonia = LinearModel

        # dict to hold the probability of onset of different types of symptom given a pathgoen:
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
        m = self.module
        self.load_parameters_from_dataframe(
            pd.read_excel(
                Path(self.resourcefilepath) / 'ResourceFile_Childhood_Pneumonia.xlsx', sheet_name='Parameter_values'))

        p['rr_death_pneumonia_agelt2mo'] = 1.4
        p['rr_death_pneumonia_age12to23mo'] = 0.8
        p['rr_death_pneumonia_age24to59mo'] = 0.3
        p['rr_death_pneumonia_HIV'] = 1.4
        p['rr_death_pneumonia_SAM'] = 1.4
        p['IMCI_effectiveness_2010'] = 0.5
        p['dhs_care_seeking_2010'] = 0.6
        p['case_fatality_rate'] = 0.15

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
                               # Predictor('li_no_access_handwashing')
                               # .when(False, m.rr_diarrhoea_HHhandwashing),
                               # Predictor('li_no_clean_drinking_water').
                               # when(False, m.rr_diarrhoea_clean_water),
                               # Predictor('li_unimproved_sanitation').
                               # when(False, m.rr_diarrhoea_improved_sanitation),
                               # # Predictor('hv_inf').
                               # # when(True, m.rr_diarrhoea_HIV),
                               # Predictor('malnutrition').
                               # when(True, m.rr_diarrhoea_SAM),
                               # Predictor('exclusive_breastfeeding').
                               # when(False, m.rr_diarrhoea_excl_breast)
                               )
        })

        self.incidence_equations_by_pathogen.update({
            'rhinovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('age_years')
                               .when('.between(0,0)', p['base_inc_rate_pneumonia_by_rhinovirus'][0])
                               .when('.between(1,1)', p['base_inc_rate_pneumoniaa_by_rhinovirus'][1])
                               .when('.between(2,4)', p['base_inc_rate_pneumoniaa_by_rhinovirus'][2])
                               .otherwise(0.0),
                               # Predictor('li_no_access_handwashing')
                               # .when(False, m.rr_diarrhoea_HHhandwashing),
                               # Predictor('li_no_clean_drinking_water').
                               # when(False, m.rr_diarrhoea_clean_water),
                               # Predictor('li_unimproved_sanitation').
                               # when(False, m.rr_diarrhoea_improved_sanitation),
                               # # Predictor('hv_inf').
                               # # when(True, m.rr_diarrhoea_HIV),
                               # Predictor('malnutrition').
                               # when(True, m.rr_diarrhoea_SAM),
                               # Predictor('exclusive_breastfeeding').
                               # when(False, m.rr_diarrhoea_excl_breast)
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
                                # Predictor('li_no_access_handwashing')
                                # .when(False, m.rr_diarrhoea_HHhandwashing),
                                # Predictor('li_no_clean_drinking_water').
                                # when(False, m.rr_diarrhoea_clean_water),
                                # Predictor('li_unimproved_sanitation').
                                # when(False, m.rr_diarrhoea_improved_sanitation),
                                # # Predictor('hv_inf').
                                # # when(True, m.rr_diarrhoea_HIV),
                                # Predictor('malnutrition').
                                # when(True, m.rr_diarrhoea_SAM),
                                # Predictor('exclusive_breastfeeding').
                                # when(False, m.rr_diarrhoea_excl_breast)
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
                                         # Predictor('li_no_access_handwashing')
                                         # .when(False, m.rr_diarrhoea_HHhandwashing),
                                         # Predictor('li_no_clean_drinking_water').
                                         # when(False, m.rr_diarrhoea_clean_water),
                                         # Predictor('li_unimproved_sanitation').
                                         # when(False, m.rr_diarrhoea_improved_sanitation),
                                         # # Predictor('hv_inf').
                                         # # when(True, m.rr_diarrhoea_HIV),
                                         # Predictor('malnutrition').
                                         # when(True, m.rr_diarrhoea_SAM),
                                         # Predictor('exclusive_breastfeeding').
                                         # when(False, m.rr_diarrhoea_excl_breast)
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
                                         # Predictor('li_no_access_handwashing')
                                         # .when(False, m.rr_diarrhoea_HHhandwashing),
                                         # Predictor('li_no_clean_drinking_water').
                                         # when(False, m.rr_diarrhoea_clean_water),
                                         # Predictor('li_unimproved_sanitation').
                                         # when(False, m.rr_diarrhoea_improved_sanitation),
                                         # # Predictor('hv_inf').
                                         # # when(True, m.rr_diarrhoea_HIV),
                                         # Predictor('malnutrition').
                                         # when(True, m.rr_diarrhoea_SAM),
                                         # Predictor('exclusive_breastfeeding').
                                         # when(False, m.rr_diarrhoea_excl_breast)
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
                               # Predictor('li_no_access_handwashing')
                               # .when(False, m.rr_diarrhoea_HHhandwashing),
                               # Predictor('li_no_clean_drinking_water').
                               # when(False, m.rr_diarrhoea_clean_water),
                               # Predictor('li_unimproved_sanitation').
                               # when(False, m.rr_diarrhoea_improved_sanitation),
                               # # Predictor('hv_inf').
                               # # when(True, m.rr_diarrhoea_HIV),
                               # Predictor('malnutrition').
                               # when(True, m.rr_diarrhoea_SAM),
                               # Predictor('exclusive_breastfeeding').
                               # when(False, m.rr_diarrhoea_excl_breast)
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
                              # todo: add risk factor - TB  - if no TB multiply by 0
                              # Predictor('li_no_access_handwashing')
                              # .when(False, m.rr_diarrhoea_HHhandwashing),
                              # Predictor('li_no_clean_drinking_water').
                              # when(False, m.rr_diarrhoea_clean_water),
                              # Predictor('li_unimproved_sanitation').
                              # when(False, m.rr_diarrhoea_improved_sanitation),
                              # # Predictor('hv_inf').
                              # # when(True, m.rr_diarrhoea_HIV),
                              # Predictor('malnutrition').
                              # when(True, m.rr_diarrhoea_SAM),
                              # Predictor('exclusive_breastfeeding').
                              # when(False, m.rr_diarrhoea_excl_breast)
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
                                          # Predictor('li_no_access_handwashing')
                                          # .when(False, m.rr_diarrhoea_HHhandwashing),
                                          # Predictor('li_no_clean_drinking_water').
                                          # when(False, m.rr_diarrhoea_clean_water),
                                          # Predictor('li_unimproved_sanitation').
                                          # when(False, m.rr_diarrhoea_improved_sanitation),
                                          # # Predictor('hv_inf').
                                          # # when(True, m.rr_diarrhoea_HIV),
                                          # Predictor('malnutrition').
                                          # when(True, m.rr_diarrhoea_SAM),
                                          # Predictor('exclusive_breastfeeding').
                                          # when(False, m.rr_diarrhoea_excl_breast)
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
                                     # Predictor('li_no_access_handwashing')
                                     # .when(False, m.rr_diarrhoea_HHhandwashing),
                                     # Predictor('li_no_clean_drinking_water').
                                     # when(False, m.rr_diarrhoea_clean_water),
                                     # Predictor('li_unimproved_sanitation').
                                     # when(False, m.rr_diarrhoea_improved_sanitation),
                                     # # Predictor('hv_inf').
                                     # # when(True, m.rr_diarrhoea_HIV),
                                     # Predictor('malnutrition').
                                     # when(True, m.rr_diarrhoea_SAM),
                                     # Predictor('exclusive_breastfeeding').
                                     # when(False, m.rr_diarrhoea_excl_breast)
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
                                     # Predictor('li_no_access_handwashing')
                                     # .when(False, m.rr_diarrhoea_HHhandwashing),
                                     # Predictor('li_no_clean_drinking_water').
                                     # when(False, m.rr_diarrhoea_clean_water),
                                     # Predictor('li_unimproved_sanitation').
                                     # when(False, m.rr_diarrhoea_improved_sanitation),
                                     # # Predictor('hv_inf').
                                     # # when(True, m.rr_diarrhoea_HIV),
                                     # Predictor('malnutrition').
                                     # when(True, m.rr_diarrhoea_SAM),
                                     # Predictor('exclusive_breastfeeding').
                                     # when(False, m.rr_diarrhoea_excl_breast)
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
                                           # Predictor('li_no_access_handwashing')
                                           # .when(False, m.rr_diarrhoea_HHhandwashing),
                                           # Predictor('li_no_clean_drinking_water').
                                           # when(False, m.rr_diarrhoea_clean_water),
                                           # Predictor('li_unimproved_sanitation').
                                           # when(False, m.rr_diarrhoea_improved_sanitation),
                                           # # Predictor('hv_inf').
                                           # # when(True, m.rr_diarrhoea_HIV),
                                           # Predictor('malnutrition').
                                           # when(True, m.rr_diarrhoea_SAM),
                                           # Predictor('exclusive_breastfeeding').
                                           # when(False, m.rr_diarrhoea_excl_breast)
                                           )
        })

        # check that equations have been declared for each pathogens
        assert self.pathogens == set(list(self.incidence_equations_by_pathogen.keys()))

        # --------------------------------------------------------------------------------------------
        # Create the linear model for the progression to severe pneumonia
        self.progression_to_severe_pneumonia = \
            LinearModel(LinearModelType.MULTIPLICATIVE,
                        1.0,
                        Predictor('age_years')
                        .when('.between(0,0)', p['r_progress_to_very_sev_penum'][0])
                        .when('.between(1,1)', p['r_progress_to_very_sev_penum'][1])
                        .when('.between(2,4)', p['r_progress_to_very_sev_penum'][2])
                        .otherwise(0.0),
                        # Predictor('has_hiv').when(True, m.rr_progress_very_sev_pneum_HIV),
                        # Predictor('malnutrition').when(True, m.rr_progress_very_sev_pneum_SAM),
                        )

        # eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'RSV') &
        #                                  (df.age_years < 5)] *= 0.7159
        # eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'rhinovirus') &
        #                                  (df.age_years < 5)] *= 0.9506
        # eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'hMPV') &
        #                                  (df.age_years < 5)] *= 0.9512
        # eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'parainfluenza') &
        #                                  (df.age_years < 5)] *= 0.5556
        # eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'streptococcus') &
        #                                  (df.age_years < 5)] *= 2.1087
        # eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'hib') &
        #                                  (df.age_years < 5)] *= 1.6122
        # eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'TB') &
        #                                  (df.age_years < 5)] *= 1.1667
        # eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'staphylococcus') &
        #                                  (df.age_years < 5)] *= 5.2727
        # eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'influenza') &
        #                                  (df.age_years < 5)] *= 1.4
        # eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'P. jirovecii') &
        #                                  (df.age_years < 5)] *= 1.9167

        # --------------------------------------------------------------------------------------------
        # Make a dict containing the probability of symptoms onset given acquisition of pneumonia
        self.prob_symptoms.update({
            'severe': {
                'fever': p['fever_by_rotavirus'],
                'vomiting': p['vomiting_by_rotavirus'],
                'dehydration': p['dehydration_by_rotavirus'],
            },
            'very severe': {
                'cough': p['fever_by_rotavirus'],
                'difficult_breathing': p['fever_by_rotavirus']
            }
        }) # TODO: add the probabilities of symptoms by severity - in parameters

        # --------------------------------------------------------------------------------------------
        # Create the linear model for the risk of dying due to pneumonia
        self.risk_of_death_pneumonia = \
            LinearModel(LinearModelType.MULTIPLICATIVE,
                        1.0,
                        Predictor('ri_last_pneumonia_severity')  ##          <--- fill in
                        .when('non-severe', p['case_fatality_rate_AWD'])
                        .when('severe', p['case_fatality_rate_AWD'])
                        .when('very severe', p['case_fatality_rate_dysentery']),
                        # .when('persistent', p['cfr_persistent_diarrhoea']),
                        # Predictor('age_years')  ##          <--- fill in
                        # .when('.between(1,2)', p['rr_diarr_death_age12to23mo'])
                        # .when('.between(2,4)', p['rr_diarr_death_age24to59mo'])
                        # .otherwise(0.0)
                        ##          < --- TODO: add in current_severe_dehyration
                        # # Predictor('hv_inf').
                        # # when(True, m.rr_diarrhoea_HIV),
                        # Predictor('malnutrition').
                        # when(True, m.rr_diarrhoea_SAM)
                        )

        # TODO: duration of ilness - mean 3.0 days (2.0-5.0 days) from PERCH

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
        df['ri_last_pneumonia_severity'].values[:] = 'none'

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

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        # ---- Key Current Status Classification Properties ----
        df.at[child_id, 'gi_last_pneumonia_pathogen'] = 'none'
        df.at[child_id, 'gi_last_pneumonia_severity'] = 'none'

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

        logger.debug('This is pneumonia reporting my health values')
        df = self.sim.population.props
        p = self.parameters

        total_daly_values = pd.Series(data=0.0, index=df.loc[df['is_alive']].index)
        total_daly_values.loc[
            self.sim.modules['SymptomManager'].who_has('fast_breathing')
        ] = self.daly_wts['daly_pneumonia']
        total_daly_values.loc[
            self.sim.modules['SymptomManager'].who_has('chest_indrawing')
        ] = self.daly_wts['daly_severe_pneumonia']
        total_daly_values.loc[
            self.sim.modules['SymptomManager'].who_has('danger_signs')
        ] = self.daly_wts['daly_severe_pneumonia']

        # health_values = df.loc[df.is_alive, 'ri_specific_symptoms'].map({
        #     'none': 0,
        #     'pneumonia': p['daly_pneumonia'],
        #     'severe pneumonia': p['daly_severe_pneumonia'],
        #     'very severe pneumonia': p['daly_very_severe_pneumonia']
        # })
        # health_values.name = 'Pneumonia Symptoms'  # label the cause of this disability
        # return health_values.loc[df.is_alive]  # returns the series

        # Split out by pathogen that causes the pneumonia
        dummies_for_pathogen = pd.get_dummies(df.loc[total_daly_values.index,
                                                     'ri_last_pneumonia_pathogen'],
                                              dtype='float')
        daly_values_by_pathogen = dummies_for_pathogen.mul(total_daly_values, axis=0).drop(columns='none')

        return daly_values_by_pathogen


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class PneumoniaPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
        This is the main event that runs the acquisition of pathogens that cause Pneumonia.
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
        p = self.module.parameters

        # Compute the incidence rate for each person getting pneumonia and then convert into a probability
        # getting all children that do not have pneumonia currently
        mask_could_get_new_pneumonia_event = \
            df['is_alive'] & (df['age_years'] < 5) & ((df['ri_last_pneumonia_recovered_date'] <= self.sim.date) |
                                                      pd.isnull(df['ri_last_pneumonia_recovered_date']))

        inc_of_acquiring_pathogen = pd.DataFrame(index=df.loc[mask_could_get_new_pneumonia_event].index)

        for pathogen in m.pathogens:
            inc_of_acquiring_pathogen[pathogen] = m.incidence_equations_by_pathogen[pathogen]\
                .predict(df.loc[mask_could_get_new_pneumonia_event])

        # Convert the incidence rates that are predicted by the model into risk of an event occurring before the next
        # polling event
        fraction_of_a_year_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(
            1, 'Y')
        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')
        probs_of_acquiring_pathogen = 1 - np.exp(-inc_of_acquiring_pathogen * fraction_of_a_year_until_next_polling_event)

        # Create the probability of getting 'any' pathogen:
        # (Assumes that pathogens are mutually exclusive)
        prob_of_acquiring_any_pathogen = probs_of_acquiring_pathogen.sum(axis=1)
        assert all(prob_of_acquiring_any_pathogen < 1.0)

        # Determine which persons will acquire any pathogen:
        person_id_that_acquire_pathogen = prob_of_acquiring_any_pathogen.index[
            rng.rand(len(prob_of_acquiring_any_pathogen)) < prob_of_acquiring_any_pathogen
            ]
        # Determine which pathogen each person will acquire (among those who will get a pathogen)
        # and create the event for the onset of new infection
        for person_id in person_id_that_acquire_pathogen:
            # ----------------------- Allocate a pathogen to the person ----------------------
            p_by_pathogen = probs_of_acquiring_pathogen.loc[person_id].values
            normalised_p_by_pathogen = p_by_pathogen / sum(p_by_pathogen)
            pathogen = rng.choice(probs_of_acquiring_pathogen.columns,
                                  p=normalised_p_by_pathogen)

            # ----------------------- Allocate a date of onset of pneumonia ----------------------
            date_onset = self.sim.date + DateOffset(days=np.random.randint(0, days_until_next_polling_event))

            # ----------------------- Determine outcomes for this case ----------------------
            # duration_in_days_of_pneumonia = max(1, int(
            #     m.mean_duration_in_days_of_pneumonia.predict(df.loc[[person_id]]).values[0] + \
            #     (-2 + 4 * rng.rand())  # assumes uniform interval around mean duration with range 4 days
            # ))
            # todo: need to add the severity

            risk_of_death = m.risk_of_death_pneumonia.predict(df.loc[[person_id]]).values[0]
            will_die = rng.rand() < risk_of_death

            # ----------------------- Allocate symptoms to onset of pneumonia ----------------------
            # possible_symptoms_for_this_pathogen = m.prob_symptoms[pathogen]
            # symptoms_for_this_person = list()
            # for symptom, prob in possible_symptoms_for_this_pathogen.items():
            #     if rng.rand() < prob:
            #         symptoms_for_this_person.append(symptom)

            # ----------------------- Create the event for the onset of infection -------------------
            # NB. The symptoms are scheduled by the SymptomManager to 'autoresolve' after the duration
            #       of the diarrhoea.
            self.sim.schedule_event(
                event=PneumoniaIncidentCase(
                    module=self.module,
                    person_id=person_id,
                    pathogen=pathogen,
                    severity=severity_of_pneumonia,
                    duration_in_days=duration_in_days_of_pneumonia,
                    will_die=will_die,
                    symptoms=symptoms_for_this_person
                ),
                date=date_onset
            )


class PneumoniaIncidentCase(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of the infection that causes pneumonia.
    """

    def __init__(self, module, person_id, pathogen, severity, duration_in_days, will_die, symptoms):
        super().__init__(module, person_id=person_id)
        self.pathogen = pathogen
        self.type = severity
        self.duration_in_days = duration_in_days
        self.will_die = will_die
        self.symptoms = symptoms

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # The event should not run if the person is not currently alive
        if not df.at[person_id, 'is_alive']:
            return

        # Update the properties in the dataframe:
        df.at[person_id, 'ri_last_pneumonia_pathogen'] = self.pathogen
        df.at[person_id, 'ri_last_pneumonia_date_of_onset'] = self.sim.date
        df.at[person_id, 'ri_last_pneumonia_type'] = self.type

        # Onset symptoms:
        for symptom in self.symptoms:
            self.module.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove='+',
                disease_module=self.module,
                duration_in_days=self.duration_in_days
            )

        # Determine timing of outcome (either recovery or death)
        date_of_outcome = self.module.sim.date + DateOffset(days=self.duration_in_days)
        if self.will_die:
            df.at[person_id, 'ri_last_pneumonia_recovered_date'] = pd.NaT
            df.at[person_id, 'ri_last_pneumonia_death_date'] = pd.NaT
            date_of_onset_severe_pneumonia = max(self.sim.date, date_of_outcome - DateOffset(
                days=self.module.parameters['days_onset_severe_pneumonia_before_death']))
            self.sim.schedule_event(SeverePneumoniaEvent(self.module, person_id),
                                    date_of_onset_severe_pneumonia)
        else:
            df.at[person_id, 'ri_last_pneumonia_recovered_date'] = date_of_outcome
            df.at[person_id, 'ri_last_pneumonia_death_date'] = pd.NaT

        # Add this incident case to the tracker
        age = df.loc[person_id, ['age_years']]
        if age.values[0] < 5:
            age_grp = age.map({0: '0y', 1: '1y', 2: '2-4y', 3: '2-4y', 4: '2-4y'}).values[0]
        else:
            age_grp = '5+y'
        self.module.incident_case_tracker[age_grp][self.pathogen].append(self.sim.date)

class SeverePneumoniaEvent(Event, IndividualScopeEventMixin):
        """
            This Event is for the onset of Severe Dehydration. This occurs a set number of days prior to death (for untreated
            children). It sets the property 'gi_current_severe_dehydration' to True and schedules the death.
            """

        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)

        def apply(self, person_id):
            df = self.sim.population.props  # shortcut to the dataframe

            # terminate the event if the person has already died.
            if not df.at[person_id, 'is_alive']:
                return

            df.at[person_id, 'ri_last_pneumonia_severity'] = 'severe'

            date_of_death = self.sim.date\
                            + DateOffset(days=self.module.parameters['days_onset_severe_pneumonia_before_death'])
            df.at[person_id, 'ri_last_pneumonia_death_date'] = date_of_death
            self.sim.schedule_event(PneumoniaDeathEvent(self.module, person_id), date_of_death)


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



        # ---------------------------------------------------------------------------------------------------
        # / # / # / # / # / # / # / # / # / PROGRESS TO SEVERE PNEUMONIA # / # / # / # / # / # / # / # / # /
        # ---------------------------------------------------------------------------------------------------
        # Progression in pneumonia severity by age groups
        severe_pneum_prog_2_11mo = \
            pd.Series(m.r_progress_to_severe_penum[0],
                      index=df.index[df.is_alive & (df.age_exact_years >= 1 / 6) & (df.age_exact_years < 1) &
                                     df.ri_pneumonia_status & (df.ri_pneumonia_severity == 'pneumonia')])
        severe_pneum_prog_12_23mo = \
            pd.Series(m.r_progress_to_severe_penum[1],
                      index=df.index[df.is_alive & (df.age_exact_years >= 1) & (df.age_exact_years < 2) &
                                     df.ri_pneumonia_status & (df.ri_pneumonia_severity == 'pneumonia')])
        severe_pneum_prog_24_59mo = \
            pd.Series(m.r_progress_to_severe_penum[2],
                      index=df.index[df.is_alive & (df.age_exact_years >= 2) & (df.age_exact_years < 5) &
                                     df.ri_pneumonia_status & (df.ri_pneumonia_severity == 'pneumonia')])
        # concatenating plus sorting
        eff_prob_prog_severe_pneum = pd.concat([severe_pneum_prog_2_11mo, severe_pneum_prog_12_23mo,
                                                severe_pneum_prog_24_59mo], axis=0).sort_index()

        eff_prob_prog_severe_pneum.loc[
            df.is_alive & (df.ri_pneumonia_severity == 'pneumonia') & df.has_hiv == True &
            (df.age_years < 5)] *= m.rr_progress_severe_pneum_HIV
        eff_prob_prog_severe_pneum.loc[
            df.is_alive & (df.ri_pneumonia_severity == 'pneumonia') & df.malnutrition == True &
            (df.age_years < 5)] *= m.rr_progress_severe_pneum_SAM
        eff_prob_prog_severe_pneum.loc[
            df.is_alive & (df.ri_pneumonia_severity == 'pneumonia') & (df.ri_pneumonia_pathogen_type == 'viral') &
            (df.age_years < 5)] *= m.rr_progress_severe_pneum_viral

        pn_current_pneumonia_idx = \
            df.index[df.is_alive & (df.age_exact_years >= 1 / 6) & (df.age_exact_years < 5) &
                     df.ri_pneumonia_status & (df.ri_pneumonia_severity == 'pneumonia')]

        random_draw = pd.Series(rng.random_sample(size=len(pn_current_pneumonia_idx)),
                                index=pn_current_pneumonia_idx)
        progress_severe_pneum = eff_prob_prog_severe_pneum > random_draw
        progress_severe_pneum_idx = eff_prob_prog_severe_pneum.index[progress_severe_pneum]
        df.loc[progress_severe_pneum_idx, 'ri_pneumonia_severity'] = 'severe pneumonia'
        self_recovery_nonsev_pneum = eff_prob_prog_severe_pneum <= random_draw
        self_recovery_nonsev_pneum_idx = eff_prob_prog_severe_pneum.index[self_recovery_nonsev_pneum]

        # date of progression to severe pneumonia for 2-59 months
        df.loc[progress_severe_pneum_idx, 'date_of_progression_severe_pneum'] = \
            df['date_of_acquiring_pneumonia'] + pd.DateOffset(days=int(rng.random_integers(0, 7)))



        # schedule recovery from non-severe pneumonia
        for person_id in self_recovery_nonsev_pneum_idx:
            self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=person_id),
                                    (df.at[person_id, 'date_of_acquiring_pneumonia'] + DateOffset(
                                        days=int(rng.random_integers(3, 7)))))

            # random_day = self.sim.date + DateOffset(days=int(rng.random_integers(1, 28)))


        # # # # # # # # # # # # # # # # # # SYMPTOMS FROM SEVERE PNEUMONIA # # # # # # # # # # # # # # # # # #

        pn_current_severe_pneum_idx = df.index[df.is_alive & (df.age_years < 5) &
                                               (df.ri_pneumonia_severity == 'severe pneumonia')]
        for individual in pn_current_severe_pneum_idx:
            df.at[individual, 'pn_chest_indrawing'] = True

        eff_prob_cough = pd.Series(0.96, index=pn_current_severe_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_severe_pneum_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_severity == 'severe pneumonia')])
        dfx = pd.concat([eff_prob_cough, random_draw], axis=1)
        dfx.columns = ['eff_prob_cough', 'random number']
        idx_cough = dfx.index[dfx.eff_prob_cough > random_draw]
        df.loc[idx_cough, 'pn_cough'] = True

        eff_prob_difficult_breathing = pd.Series(0.40, index=pn_current_severe_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_severe_pneum_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (
                                            df.ri_pneumonia_severity == 'severe pneumonia')])
        dfx = pd.concat([eff_prob_difficult_breathing, random_draw], axis=1)
        dfx.columns = ['eff_prob_difficult_breathing', 'random number']
        idx_difficult_breathing = dfx.index[dfx.eff_prob_difficult_breathing > random_draw]
        df.loc[idx_difficult_breathing, 'pn_difficult_breathing'] = True

        eff_prob_fast_breathing = pd.Series(0.96, index=pn_current_severe_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_severe_pneum_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (
                                            df.ri_pneumonia_severity == 'severe pneumonia')])
        dfx = pd.concat([eff_prob_fast_breathing, random_draw], axis=1)
        dfx.columns = ['eff_prob_fast_breathing', 'random number']
        idx_fast_breathing = dfx.index[dfx.eff_prob_fast_breathing > random_draw]
        df.loc[idx_fast_breathing, 'pn_fast_breathing'] = True


        # --------------------------------------------------------------------------------------------------------
        # / # / # / # / # / # / # / # / # / PROGRESS TO VERY SEVERE PNEUMONIA # / # / # / # / # / # / # / # / # /
        # --------------------------------------------------------------------------------------------------------
        # Progression in pneumonia severity by age groups

        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.has_hiv == True) & (df.age_years < 5)] *= \
            m.rr_progress_very_sev_pneum_HIV
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & df.malnutrition == True & (df.age_years < 5)] *= \
            m.rr_progress_very_sev_pneum_SAM
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'RSV') &
                                         (df.age_years < 5)] *= 0.7159
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'rhinovirus') &
                                         (df.age_years < 5)] *= 0.9506
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'hMPV') &
                                         (df.age_years < 5)] *= 0.9512
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'parainfluenza') &
                                         (df.age_years < 5)] *= 0.5556
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'streptococcus') &
                                         (df.age_years < 5)] *= 2.1087
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'hib') &
                                         (df.age_years < 5)] *= 1.6122
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'TB') &
                                         (df.age_years < 5)] *= 1.1667
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'staphylococcus') &
                                         (df.age_years < 5)] *= 5.2727
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'influenza') &
                                         (df.age_years < 5)] *= 1.4
        eff_prob_prog_very_sev_pneum.loc[current_sev_pneum & (df.ri_pneumonia_pathogen_type == 'P. jirovecii') &
                                         (df.age_years < 5)] *= 1.9167


        # log the information on attributable pathogens
        pathogen_count = df[df.is_alive & df.age_years.between(0, 5)].groupby('ri_pneumonia_pathogen').size()
        under5 = df[df.is_alive & df.age_years.between(0, 5)]
        logger.info('%s|pneumonia_pathogens|%s', self.sim.date,
                    {'total': sum(pathogen_count),
                     'RSV': pathogen_count['RSV'],
                     'rhinovirus': pathogen_count['rhinovirus'],
                     'hMPV': pathogen_count['hMPV'],
                     'parainfluenza': pathogen_count['parainfluenza'],
                     'strep': pathogen_count['streptococcus'],
                     'hib': pathogen_count['hib'],
                     'TB': pathogen_count['TB'],
                     'staph': pathogen_count['staph'],
                     'influenza': pathogen_count['influenza'],
                     'jirovecii': pathogen_count['P. jirovecii'],
                     })

        # incidence rate by pathogen per 100 child-years
        logger.info('%s|pneumo_incidence_by_patho|%s', self.sim.date,
                    {'total': (sum(pathogen_count) * 4 * 100) / len(under5),
                     'RSV': (pathogen_count['RSV'] * 4 * 100) / len(under5),
                     'rhinovirus': (pathogen_count['rhinovirus'] * 4 * 100) / len(under5),
                     'hMPV': (pathogen_count['hMPV'] * 4 * 100) / len(under5),
                     'parainfluenza': (pathogen_count['parainfluenza'] * 4 * 100) / len(under5),
                     'strep': (pathogen_count['streptococcus'] * 4 * 100) / len(under5),
                     'hib': (pathogen_count['hib'] * 4 * 100) / len(under5),
                     'TB': (pathogen_count['TB'] * 4 * 100) / len(under5),
                     'staph': (pathogen_count['staph'] * 4 * 100) / len(under5),
                     'influenza': (pathogen_count['influenza'] * 4 * 100) / len(under5),
                     'jirovecii': (pathogen_count['P. jirovecii'] * 4 * 100) / len(under5),
                     })

        # TODO: make a graph showing the proportions of pathogens causing severe vs very severe pneum
        # log the proportions of pathogens causing severe and very severe
        severity_pneum_count = df[df.is_alive & df.age_years.between(0, 5)].groupby('ri_pneumonia_severity').size()
        logger.info('%s|severity_pneumonia|%s', self.sim.date,
                    {'total': sum(severity_pneum_count),
                     'pneumonia': severity_pneum_count['pneumonia'],
                     'severe': severity_pneum_count['severe pneumonia'],
                     'very_severe': severity_pneum_count['very severe pneumonia']
                     })

        # # # # # # # # # # # # # # # # # # SYMPTOMS FROM VERY SEVERE PNEUMONIA # # # # # # # # # # # # # # # # # #

        pn_current_very_sev_pneum_idx = df.index[df.is_alive & (df.age_years < 5) &
                                                 (df.ri_pneumonia_severity == 'very severe pneumonia')]

        eff_prob_cough = pd.Series(0.857, index=pn_current_very_sev_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_very_sev_pneum_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_severity == 'very severe pneumonia')])
        dfx = pd.concat([eff_prob_cough, random_draw], axis=1)
        dfx.columns = ['eff_prob_cough', 'random number']
        idx_cough = dfx.index[dfx.eff_prob_cough > random_draw]
        df.loc[idx_cough, 'pn_cough'] = True

        eff_prob_difficult_breathing = pd.Series(0.43, index=pn_current_very_sev_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_very_sev_pneum_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (
                                        df.ri_pneumonia_severity == 'very severe pneumonia')])
        dfx = pd.concat([eff_prob_difficult_breathing, random_draw], axis=1)
        dfx.columns = ['eff_prob_difficult_breathing', 'random number']
        idx_difficult_breathing = dfx.index[dfx.eff_prob_difficult_breathing > random_draw]
        df.loc[idx_difficult_breathing, 'pn_difficult_breathing'] = True

        eff_prob_fast_breathing = pd.Series(0.857, index=pn_current_very_sev_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_very_sev_pneum_idx)),
                                index=df.index[
                                    (df.age_years < 5) & df.is_alive & (
                                        df.ri_pneumonia_severity == 'very severe pneumonia')])
        dfx = pd.concat([eff_prob_fast_breathing, random_draw], axis=1)
        dfx.columns = ['eff_prob_fast_breathing', 'random number']
        idx_fast_breathing = dfx.index[dfx.eff_prob_fast_breathing > random_draw]
        df.loc[idx_fast_breathing, 'pn_fast_breathing'] = True

        eff_prob_chest_indrawing = pd.Series(0.76, index=pn_current_very_sev_pneum_idx)
        random_draw = pd.Series(rng.random_sample(size=len(pn_current_very_sev_pneum_idx)),
                                index=df.index[(df.age_years < 5) & df.is_alive &
                                               (df.ri_pneumonia_severity == 'very severe pneumonia')])
        dfx = pd.concat([eff_prob_chest_indrawing, random_draw], axis=1)
        dfx.columns = ['eff_prob_chest_indrawing', 'random number']
        idx_chest_indrawing = dfx.index[dfx.eff_prob_chest_indrawing > random_draw]
        df.loc[idx_chest_indrawing, 'pn_chest_indrawing'] = True



        # # # # # # # # # # # # # # #  ASSIGN SYMPTOMS # # # # # # # # # # # # # # # #
        if df[df.is_alive & (df.age_exact_years < 5) & df.ri_pneumonia_status]:
            self.module.assign_symptoms(population, new_infections, 'very severe pneumonia')

        # # # # # # ASSIGN DEATH PROBABILITIES BASED ON AGE, SEVERITY AND CO-MORBIDITIES # # # # # #
        # schedule death events for very severe pneumonia
        current_very_sev_pneumonia_idx = \
            df.index[df.is_alive & (df.ri_pneumonia_severity == 'very severe pneumonia') & (df.age_exact_years < 5)]

        # base group 2-11 months of age
        eff_prob_death_pneumonia = \
            pd.Series(m.r_death_pneumonia,
                      index=current_very_sev_pneumonia_idx)
        eff_prob_death_pneumonia.loc[
            df.is_alive & (df.age_exact_years < 1 / 6)] *= m.rr_death_pneumonia_agelt2mo
        eff_prob_death_pneumonia.loc[
            df.is_alive & (df.age_exact_years >= 1) & (df.age_exact_years < 2)] *= m.rr_death_pneumonia_age12to23mo
        eff_prob_death_pneumonia.loc[df.is_alive & (df.age_exact_years >= 2) & (df.age_exact_years < 5)] *= \
            m.rr_death_pneumonia_age24to59mo
        eff_prob_death_pneumonia.loc[df.is_alive & df.has_hiv == True & (df.age_exact_years < 5)] *= \
            m.rr_death_pneumonia_HIV
        eff_prob_death_pneumonia.loc[df.is_alive & df.malnutrition == True & (df.age_exact_years < 5)] *= \
            m.rr_death_pneumonia_SAM

        random_draw_death = \
            pd.Series(rng.random_sample(size=len(current_very_sev_pneumonia_idx)),
                      index=current_very_sev_pneumonia_idx)
        pneum_death = eff_prob_death_pneumonia > random_draw_death
        pneum_death_idx = eff_prob_death_pneumonia.index[pneum_death]
        recover_from_very_sev_pneum = eff_prob_death_pneumonia <= random_draw_death
        recover_from_very_sev_pneum_idx = eff_prob_death_pneumonia.index[recover_from_very_sev_pneum]

        # schedule recovery event from very severe pneumonia
        for person_id in recover_from_very_sev_pneum_idx:
            self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=person_id),
                                    (df.at[person_id, 'date_of_progression_very_sev_pneum'] +
                                     DateOffset(days=int(rng.random_integers(1, 2)))))

        # schedule death event
        for person_id in pneum_death_idx:
            self.sim.schedule_event(DeathFromPneumoniaDisease(self.module, person_id=person_id, cause='pneumonia'),
                                    (df.at[person_id, 'date_of_progression_very_sev_pneum'] +
                                     DateOffset(days=int(rng.random_integers(1, 2)))))
