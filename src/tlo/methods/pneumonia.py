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
    severity = {
        'non-severe',
        'severe'
    }

    # Declare the underlying conditions of the ALRI:
    diseases = {
        'pneumonia',
        'bronchiolitis'
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
        'base_inc_rate_ALRI_by_RSV': Parameter
        (Types.LIST, 'incidence of pneumonia caused by Respiratory Syncytial Virus in age groups 0-11, 12-59 months'
         ),
        'base_inc_rate_ALRI_by_rhinovirus': Parameter
        (Types.LIST, 'incidence of pneumonia caused by rhinovirus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_hMPV': Parameter
        (Types.LIST, 'incidence of pneumonia caused by hMPV in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_parainfluenza': Parameter
        (Types.LIST, 'incidence of pneumonia caused by parainfluenza in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_streptococcus': Parameter
        (Types.LIST, 'incidence of pneumonia caused by streptoccocus 40/41 in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_hib': Parameter
        (Types.LIST, 'incidence of pneumonia caused by hib in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_TB': Parameter
        (Types.LIST, 'incidence of pneumonia caused by TB in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_staphylococcus': Parameter
        (Types.LIST, 'incidence of pneumonia caused by Staphylococcus aureus in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_influenza': Parameter
        (Types.LIST, 'incidence of pneumonia caused by influenza in age groups 0-11, 12-23, 24-59 months'
         ),
        'base_inc_rate_ALRI_by_jirovecii': Parameter
        (Types.LIST, 'incidence of pneumonia caused by P. jirovecii in age groups 0-11, 12-59 months'
         ),
        'base_inc_rate_ALRI_by_other_pathogens': Parameter
        (Types.LIST, 'incidence of pneumonia caused by other pathogens in age groups 0-11, 12-59 months'
         ),
        'rr_ALRI_HHhandwashing': Parameter
        (Types.REAL, 'relative rate of pneumonia with household handwashing with soap'
         ),
        'rr_ALRI_HIV': Parameter
        (Types.REAL, 'relative rate of pneumonia for HIV positive status'
         ),
        'rr_ALRI_SAM': Parameter
        (Types.REAL, 'relative rate of pneumonia for severe malnutrition'
         ),
        'rr_ALRI_excl_breastfeeding': Parameter
        (Types.REAL, 'relative rate of pneumonia for exclusive breastfeeding upto 6 months'
         ),
        'rr_ALRI_cont_breast': Parameter
        (Types.REAL, 'relative rate of pneumonia for continued breastfeeding 6 months to 2 years'
         ),
        'rr_ALRI_indoor_air_pollution': Parameter
        (Types.REAL, 'relative rate of pneumonia for indoor air pollution'
         ),
        'rr_ALRI_pneumococcal_vaccine': Parameter
        (Types.REAL, 'relative rate of pneumonia for pneumonococcal vaccine'
         ),
        'rr_ALRI_hib_vaccine': Parameter
        (Types.REAL, 'relative rate of pneumonia for hib vaccine'
         ),
        'rr_ALRI_influenza_vaccine': Parameter
        (Types.REAL, 'relative rate of pneumonia for influenza vaccine'
         ),
        'p_symptomatic_RSV_inf_causing_pneumonia': Parameter
        (Types.LIST, 'proportion of RSV infections causing pneumonia'
         ),
        'p_symptomatic_rhinovirus_inf_causing_pneumonia': Parameter
        (Types.LIST, 'proportion of rhinovirus infections causing pneumonia'
         ),
        'p_symptomatic_hMPV_inf_causing_pneumonia': Parameter
        (Types.LIST, 'proportion of hMPV infections causing pneumonia'
         ),
        'p_symptomatic_parainfluenza_inf_causing_pneumonia': Parameter
        (Types.LIST, 'proportion of parainfluenza infections causing pneumonia'
         ),
        'p_symptomatic_streptococcus_inf_causing_pneumonia': Parameter
        (Types.LIST, 'proportion of streptococcus infections causing pneumonia'
         ),
        'p_symptomatic_hib_inf_causing_pneumonia': Parameter
        (Types.LIST, 'proportion of hib infections causing pneumonia'
         ),
        'p_symptomatic_TB_inf_causing_pneumonia': Parameter
        (Types.LIST, 'proportion of TB infections causing pneumonia'
         ),
        'p_symptomatic_staphylococcus_inf_causing_pneumonia': Parameter
        (Types.LIST, 'proportion of staphylococcus infections causing pneumonia'
         ),
        'p_symptomatic_influenza_inf_causing_pneumonia': Parameter
        (Types.LIST, 'proportion of influenza infections causing pneumonia'
         ),
        'p_symptomatic_jirovecii_inf_causing_pneumonia': Parameter
        (Types.LIST, 'proportion of jirovecii infections causing pneumonia'
         ),
        'p_symptomatic_other_pathogens_inf_causing_pneumonia': Parameter
        (Types.LIST, 'proportion of other_pathogens infections causing pneumonia'
         ),
        'r_progress_to_severe_pneumonia': Parameter
        (Types.LIST,
         'probability of progressing from non-severe to severe pneumonia by age category '
         'HIV negative, no SAM'
         ),
        'r_progress_to_severe_bronchiolitis': Parameter
        (Types.LIST,
         'probability of progressing from non-severe to severe bronchiolitis  by age category  '
         'HIV negative, no SAM'
         ),
        # 'rr_progress_severe_pneum_age12to23mo': Parameter
        # (Types.REAL,
        #  'relative rate of progression to severe pneumonia for age 12 to 23 months'
        #  ),
        # 'rr_progress_severe_pneum_age24to59mo': Parameter
        # (Types.REAL, 'relative rate of progression to severe pneumonia for age 24 to 59 months'
        #  ),
        'rr_progress_clinical_severe_pneumonia_HIV': Parameter
        (Types.REAL,
         'relative risk of progression to severe pneumonia for HIV positive status'
         ),
        'rr_progress_clinical_severe_pneumonia_SAM': Parameter
        (Types.REAL,
         'relative rate of progression to severe pneumonia for severe acute malnutrition'
         ),
        'r_death_pneumonia': Parameter
        (Types.REAL,
         'death rate of pneumonia'
         ),
        'r_death_bronchiolitis': Parameter
        (Types.REAL,
         'death rate of bronchiolitis'
         ),
        'rr_death_pneumonia_agelt2mo': Parameter
        (Types.REAL,
         'death rate of pneumonia'
         ),
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
    }

    PROPERTIES = {
        # ---- The pathogen which is the attributed cause of pneumonia ----
        'ri_last_ALRI_pathogen': Property(Types.CATEGORICAL,
                                          'Attributable pathogen for the last ALRI event',
                                          categories=list(pathogens) + ['none']),

        # ---- Classification of the severity of pneumonia that is caused ----
        'ri_last_clinical_severity': Property(Types.CATEGORICAL,
                                               'clinical severity of disease',
                                               categories=list(severity) + ['none']),

        # ---- The true underlying condition ----
        'ri_true_underlying_condition': Property(Types.CATEGORICAL,
                                               'true underlying condition',
                                               categories=['pneumonia', 'bronchiolitis']),

        # ---- Internal variables to schedule onset and deaths due to pneumonia ----
        'ri_last_ALRI_date_of_onset': Property(Types.DATE, 'date of onset of last pneumonia event'),
        'ri_last_ALRI_recovered_date': Property(Types.DATE, 'date of recovery from last pneumonia event'),
        'ri_last_ALRI_death_date': Property(Types.DATE, 'date of death caused by last pneumonia event'),

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

        # equations for the incidence of ALRI by pathogen:
        self.incidence_equations_by_pathogen = dict()

        # equations for the proportions of symptomatic ALRI that are pneumonia:
        self.pathogens_causing_pneumonia_as_underlying_condition = dict()

        # equations for predicting the progression of disease from non-severe to severe clinical pneumonia:
        self.progression_to_clinical_severe_penumonia = dict()

        # Linear Model for predicting the risk of death:
        self.risk_of_death_clinical_severe_pneumonia = dict()

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
        self.load_parameters_from_dataframe(
            pd.read_excel(
                Path(self.resourcefilepath) / 'ResourceFile_Pneumonia.xlsx', sheet_name='Parameter_values'))

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
                               .when('.between(0,0)', p['base_inc_rate_ALRI_by_RSV'][0])
                               .when('.between(1,1)', p['base_inc_rate_ALRI_by_RSV'][1])
                               .when('.between(2,4)', p['base_inc_rate_ALRI_by_RSV'][2])
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
                                      .when('.between(0,0)', p['base_inc_rate_ALRI_by_rhinovirus'][0])
                                      .when('.between(1,1)', p['base_inc_rate_ALRI_by_rhinovirus'][1])
                                      .when('.between(2,4)', p['base_inc_rate_ALRI_by_rhinovirus'][2])
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
                                .when('.between(0,0)', p['base_inc_rate_ALRI_by_hMPV'][0])
                                .when('.between(1,1)', p['base_inc_rate_ALRI_by_hMPV'][1])
                                .when('.between(2,4)', p['base_inc_rate_ALRI_by_hMPV'][2])
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
                                         .when('.between(0,0)', p['base_inc_rate_ALRI_by_parainfluenza'][0])
                                         .when('.between(1,1)', p['base_inc_rate_ALRI_by_parainfluenza'][1])
                                         .when('.between(2,4)', p['base_inc_rate_ALRI_by_parainfluenza'][2])
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
                                         .when('.between(0,0)', p['base_inc_rate_ALRI_by_streptococcus'][0])
                                         .when('.between(1,1)', p['base_inc_rate_ALRI_by_streptococcus'][1])
                                         .when('.between(2,4)', p['base_inc_rate_ALRI_by_streptococcus'][2])
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
                               .when('.between(0,0)', p['base_inc_rate_ALRI_by_hib'][0])
                               .when('.between(1,1)', p['base_inc_rate_ALRI_by_hib'][1])
                               .when('.between(2,4)', p['base_inc_rate_ALRI_by_hib'][2])
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
                              .when('.between(0,0)', p['base_inc_rate_ALRI_by_TB'][0])
                              .when('.between(1,1)', p['base_inc_rate_ALRI_by_TB'][1])
                              .when('.between(2,4)', p['base_inc_rate_ALRI_by_TB'][2])
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
                                          .when('.between(0,0)', p['base_inc_rate_ALRI_by_staphylococcus'][0])
                                          .when('.between(1,1)', p['base_inc_rate_ALRI_by_staphylococcus'][1])
                                          .when('.between(2,4)', p['base_inc_rate_ALRI_by_staphylococcus'][2])
                                          .otherwise(0.0),
                                          # Predictor('li_no_access_handwashing')
                                          # .when(False, m.rr_diarrhoea_HHhandwashing),
                                          # Predictor('li_no_clean_drinking_water').
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
                                          .when('.between(0,0)', p['base_inc_rate_ALRI_by_influenza'][0])
                                          .when('.between(1,1)', p['base_inc_rate_ALRI_by_influenza'][1])
                                          .when('.between(2,4)', p['base_inc_rate_ALRI_by_influenza'][2])
                                          .otherwise(0.0),
                                          # Predictor('li_no_access_handwashing')
                                          # .when(False, m.rr_diarrhoea_HHhandwashing),
                                          # Predictor('li_no_clean_drinking_water').
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
                                     .when('.between(0,0)', p['base_inc_rate_ALRI_by_jirovecii'][0])
                                     .when('.between(1,1)', p['base_inc_rate_ALRI_by_jirovecii'][1])
                                     .when('.between(2,4)', p['base_inc_rate_ALRI_by_jirovecii'][2])
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
                                           .when('.between(0,0)', p['base_inc_rate_ALRI_by_other_pathogens'][0])
                                           .when('.between(1,1)', p['base_inc_rate_ALRI_by_other_pathogens'][1])
                                           .when('.between(2,4)', p['base_inc_rate_ALRI_by_other_pathogens'][2])
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
        # Linear models for determining the underlying condition as pneumonia caused by each pathogen
        self.pathogens_causing_pneumonia_as_underlying_condition.update({
            'RSV': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('ri_last_ALRI_pathogen')
                               .when('RSV', 1.0)
                               .otherwise(0.0),
                               Predictor('age_years')
                               .when('.between(0,0)', p['p_symptomatic_RSV_inf_causing_pneumonia'][0])
                               .when('.between(1,1)', p['p_symptomatic_RSV_inf_causing_pneumonia'][1])
                               .when('.between(2,4)', p['p_symptomatic_RSV_inf_causing_pneumonia'][2])
                               .otherwise(0.0),
                               )
        })
        self.pathogens_causing_pneumonia_as_underlying_condition.update({
            'rhinovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                      1.0,
                                      Predictor('ri_last_ALRI_pathogen')
                                      .when('rhinovirus', 1.0)
                                      .otherwise(0.0),
                                      Predictor('age_years')
                                      .when('.between(0,0)', p['p_symptomatic_rhinovirus_inf_causing_pneumonia'][0])
                                      .when('.between(1,1)', p['p_symptomatic_rhinovirus_inf_causing_pneumonia'][1])
                                      .when('.between(2,4)', p['p_symptomatic_rhinovirus_inf_causing_pneumonia'][2])
                                      .otherwise(0.0),
                                      )
        })
        self.pathogens_causing_pneumonia_as_underlying_condition.update({
            'hMPV': LinearModel(LinearModelType.MULTIPLICATIVE,
                                1.0,
                                Predictor('ri_last_ALRI_pathogen')
                                .when('hMPV', 1.0)
                                .otherwise(0.0),
                                Predictor('age_years')
                                .when('.between(0,0)', p['p_symptomatic_hMPV_inf_causing_pneumonia'][0])
                                .when('.between(1,1)', p['p_symptomatic_hMPV_inf_causing_pneumonia'][1])
                                .when('.between(2,4)', p['p_symptomatic_hMPV_inf_causing_pneumonia'][2])
                                .otherwise(0.0),
                                )
        })
        self.pathogens_causing_pneumonia_as_underlying_condition.update({
            'parainfluenza': LinearModel(LinearModelType.MULTIPLICATIVE,
                                         1.0,
                                         Predictor('ri_last_ALRI_pathogen')
                                         .when('parainfluenza', 1.0)
                                         .otherwise(0.0),
                                         Predictor('age_years')
                                         .when('.between(0,0)',
                                               p['p_symptomatic_parainfluenza_inf_causing_pneumonia'][0])
                                         .when('.between(1,1)',
                                               p['p_symptomatic_parainfluenza_inf_causing_pneumonia'][1])
                                         .when('.between(2,4)',
                                               p['p_symptomatic_parainfluenza_inf_causing_pneumonia'][2])
                                         .otherwise(0.0),
                                         )


        })
        self.pathogens_causing_pneumonia_as_underlying_condition.update({
            'streptococcus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                         1.0,
                                         Predictor('ri_last_ALRI_pathogen')
                                         .when('streptococcus', 1.0)
                                         .otherwise(0.0),
                                         Predictor('age_years')
                                         .when('.between(0,0)',
                                               p['p_symptomatic_streptococcus_inf_causing_pneumonia'][0])
                                         .when('.between(1,1)',
                                               p['p_symptomatic_streptococcus_inf_causing_pneumonia'][1])
                                         .when('.between(2,4)',
                                               p['p_symptomatic_streptococcus_inf_causing_pneumonia'][2])
                                         .otherwise(0.0),
                                         )
        })
        self.pathogens_causing_pneumonia_as_underlying_condition.update({
            'hib': LinearModel(LinearModelType.MULTIPLICATIVE,
                               1.0,
                               Predictor('ri_last_ALRI_pathogen')
                               .when('hib', 1.0)
                               .otherwise(0.0),
                               Predictor('age_years')
                               .when('.between(0,0)', p['p_symptomatic_hib_inf_causing_pneumonia'][0])
                               .when('.between(1,1)', p['p_symptomatic_hib_inf_causing_pneumonia'][1])
                               .when('.between(2,4)', p['p_symptomatic_hib_inf_causing_pneumonia'][2])
                               .otherwise(0.0),
                               )
        })
        self.pathogens_causing_pneumonia_as_underlying_condition.update({
            'TB': LinearModel(LinearModelType.MULTIPLICATIVE,
                              1.0,
                              Predictor('ri_last_ALRI_pathogen')
                              .when('TB', 1.0)
                              .otherwise(0.0),
                              Predictor('age_years')
                              .when('.between(0,0)', p['p_symptomatic_TB_inf_causing_pneumonia'][0])
                              .when('.between(1,1)', p['p_symptomatic_TB_inf_causing_pneumonia'][1])
                              .when('.between(2,4)', p['p_symptomatic_TB_inf_causing_pneumonia'][2])
                              .otherwise(0.0),
                              )

        })
        self.pathogens_causing_pneumonia_as_underlying_condition.update({
            'staphylococcus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                          1.0,
                                          Predictor('ri_last_ALRI_pathogen')
                                          .when('staphylococcus', 1.0)
                                          .otherwise(0.0),
                                          Predictor('age_years')
                                          .when('.between(0,0)',
                                                p['p_symptomatic_staphylococcus_inf_causing_pneumonia'][0])
                                          .when('.between(1,1)',
                                                p['p_symptomatic_staphylococcus_inf_causing_pneumonia'][1])
                                          .when('.between(2,4)',
                                                p['p_symptomatic_staphylococcus_inf_causing_pneumonia'][2])
                                          .otherwise(0.0),
                                          )
        })
        self.pathogens_causing_pneumonia_as_underlying_condition.update({
            'influenza': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('ri_last_ALRI_pathogen')
                                     .when('influenza', 1.0)
                                     .otherwise(0.0),
                                     Predictor('age_years')
                                     .when('.between(0,0)', p['p_symptomatic_influenza_inf_causing_pneumonia'][0])
                                     .when('.between(1,1)', p['p_symptomatic_influenza_inf_causing_pneumonia'][1])
                                     .when('.between(2,4)', p['p_symptomatic_influenza_inf_causing_pneumonia'][2])
                                     .otherwise(0.0),
                                     )
        })
        self.pathogens_causing_pneumonia_as_underlying_condition.update({
            'jirovecii': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('ri_last_ALRI_pathogen')
                                     .when('jirovecii', 1.0)
                                     .otherwise(0.0),
                                     Predictor('age_years')
                                     .when('.between(0,0)', p['p_symptomatic_jirovecii_inf_causing_pneumonia'][0])
                                     .when('.between(1,1)', p['p_symptomatic_jirovecii_inf_causing_pneumonia'][1])
                                     .when('.between(2,4)', p['p_symptomatic_jirovecii_inf_causing_pneumonia'][2])
                                     .otherwise(0.0),
                                     )
        })
        self.pathogens_causing_pneumonia_as_underlying_condition.update({
            'other_pathogens': LinearModel(LinearModelType.MULTIPLICATIVE,
                                           1.0,
                                           Predictor('ri_last_ALRI_pathogen')
                                           .when('other_pathogens', 1.0)
                                           .otherwise(0.0),
                                           Predictor('age_years')
                                           .when('.between(0,0)',
                                                 p['p_symptomatic_other_pathogens_inf_causing_pneumonia'][0])
                                           .when('.between(1,1)',
                                                 p['p_symptomatic_other_pathogens_inf_causing_pneumonia'][1])
                                           .when('.between(2,4)',
                                                 p['p_symptomatic_other_pathogens_inf_causing_pneumonia'][2])
                                           .otherwise(0.0),
                                           )
        })

        # check that equations have been declared for each pathogens
        assert self.pathogens == set(list(self.pathogens_causing_pneumonia_as_underlying_condition.keys()))

        # --------------------------------------------------------------------------------------------
        # Create the linear model for the progression to severe pneumonia
        self.progression_to_clinical_severe_penumonia.update({
            'pneumonia': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('age_years')
                                     .when('.between(0,0)', p['r_progress_to_severe_pneumonia'][0])
                                     .when('.between(1,1)', p['r_progress_to_severe_pneumonia'][1])
                                     .when('.between(2,4)', p['r_progress_to_severe_pneumonia'][2])
                                     .otherwise(0.0),
                                     # Predictor('has_hiv').when(True, m.rr_progress_very_sev_pneum_HIV),
                                     # Predictor('malnutrition').when(True, m.rr_progress_very_sev_pneum_SAM),
                                     ) # todo: add pathogens
        })
        self.progression_to_clinical_severe_penumonia.update({
            'bronchiolitis': LinearModel(LinearModelType.MULTIPLICATIVE,
                                         1.0,
                                         Predictor('age_years')
                                         .when('.between(0,0)', p['r_progress_to_severe_bronchiolitis'][0])
                                         .when('.between(1,1)', p['r_progress_to_severe_bronchiolitis'][1])
                                         .when('.between(2,4)', p['r_progress_to_severe_bronchiolitis'][2])
                                         .otherwise(0.0),
                                         # Predictor('has_hiv').when(True, m.rr_progress_very_sev_pneum_HIV),
                                         # Predictor('malnutrition').when(True, m.rr_progress_very_sev_pneum_SAM),
                                         )
        })


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
            'non-severe': {
                'fever': 0.7,
                'cough': 0.8,
                'difficult_breathing': 1,
                'fast_breathing': 0.9,
                'chest_indrawing': 0.5,
                'danger_signs': 0
            },
            'severe': {
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
        assert self.severity == set(list(self.prob_symptoms.keys()))

        # --------------------------------------------------------------------------------------------
        # Create the linear model for the risk of dying due to pneumonia
        self.risk_of_death_clinical_severe_pneumonia.update({
            'pneumonia': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     0.7,
                                     Predictor('ri_last_pneumonia_severity')  ##          <--- fill in
                                     .when('non-severe', 0)  # zero probability of dying if non-severe
                                     .when('severe', 0.9)
                                     #                 # .when('persistent', p['cfr_persistent_diarrhoea']),
                                     #                 # Predictor('age_years')  ##          <--- fill in
                                     #                 # .when('.between(1,2)', p['rr_diarr_death_age12to23mo'])
                                     #                 # .when('.between(2,4)', p['rr_diarr_death_age24to59mo'])
                                     #                 # .otherwise(0.0)
                                     #                 ##          < --- TODO: add in current_severe_dehyration
                                     #                 # # Predictor('hv_inf').
                                     #                 # # when(True, m.rr_diarrhoea_HIV),
                                     #                 # Predictor('malnutrition').
                                     #                 # when(True, m.rr_diarrhoea_SAM)
                                     )
        })
        self.risk_of_death_clinical_severe_pneumonia.update({
            'bronchiolitis': LinearModel(LinearModelType.MULTIPLICATIVE,
                                         0.6,
                                         Predictor('ri_last_pneumonia_severity')  ##          <--- fill in
                                         .when('non-severe', 0)  # zero probability of dying if non-severe
                                         .when('severe', 0.6)
                                         #                 # .when('persistent', p['cfr_persistent_diarrhoea']),
                                         #                 # Predictor('age_years')  ##          <--- fill in
                                         #                 # .when('.between(1,2)', p['rr_diarr_death_age12to23mo'])
                                         #                 # .when('.between(2,4)', p['rr_diarr_death_age24to59mo'])
                                         #                 # .otherwise(0.0)
                                         #                 ##          < --- TODO: add in current_severe_dehyration
                                         #                 # # Predictor('hv_inf').
                                         #                 # # when(True, m.rr_diarrhoea_HIV),
                                         #                 # Predictor('malnutrition').
                                         #                 # when(True, m.rr_diarrhoea_SAM)
                                         )
        })

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
        df['ri_last_ALRI_pathogen'].values[:] = 'none'
        df['ri_last_clinical_severity'].values[:] = 'none'

        # ---- Internal values ----
        df['ri_last_ALRI_date_of_onset'] = pd.NaT
        df['ri_last_ALRI_recovered_date'] = pd.NaT
        df['ri_last_ALRI_death_date'] = pd.NaT

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
        df.at[child_id, 'gi_last_ALRI_pathogen'] = 'none'
        df.at[child_id, 'gi_last_clinical_severity'] = 'none'

        # ---- Internal values ----
        df.at[child_id, 'gi_last_ALRI_date_of_onset'] = pd.NaT
        df.at[child_id, 'gi_last_ALRI_recovered_date'] = pd.NaT
        df.at[child_id, 'gi_last_ALRI_death_date'] = pd.NaT

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
class AcuteLowerRespiratoryInfectionPollingEvent(RegularEvent, PopulationScopeEventMixin):
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
        mask_could_get_new_ALRI_event = \
            df['is_alive'] & (df['age_years'] < 5) & ((df['ri_last_ALRI_recovered_date'] <= self.sim.date) |
                                                      pd.isnull(df['ri_last_ALRI_recovered_date']))

        inc_of_acquiring_ALRI = pd.DataFrame(index=df.loc[mask_could_get_new_ALRI_event].index)

        for pathogen in m.pathogens:
            inc_of_acquiring_ALRI[pathogen] = m.incidence_equations_by_pathogen[pathogen] \
                .predict(df.loc[mask_could_get_new_ALRI_event])

        # Convert the incidence rates that are predicted by the model into risk of an event occurring before the next
        # polling event
        fraction_of_a_year_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(
            1, 'Y')
        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')
        probs_of_acquiring_pathogen = 1 - np.exp(-inc_of_acquiring_ALRI * fraction_of_a_year_until_next_polling_event)

        # Create the probability of getting 'any' pathogen:
        # (Assumes that pathogens are mutually exclusive)
        prob_of_acquiring_any_pathogen = probs_of_acquiring_pathogen.sum(axis=1)
        assert all(prob_of_acquiring_any_pathogen < 1.0)

        # Determine which persons will acquire any pathogen:
        person_id_that_acquire_pathogen = prob_of_acquiring_any_pathogen.index[
            rng.rand(len(prob_of_acquiring_any_pathogen)) < prob_of_acquiring_any_pathogen
            ]

        # # Determine which disease a pathogen will cause:
        # prob_pathogen_causing_pneumonia = pd.DataFrame(index=df.loc[person_id_that_acquire_pathogen].index,
        #                                                columns=['pneumonia', 'bronchiolitis'])
        # for pathogen in m.pathogens:
        #     prob_pathogen_causing_pneumonia['pneumonia'] = \
        #         m.pathogens_causing_pneumonia_as_underlying_condition[pathogen].predict(
        #         df.loc[person_id_that_acquire_pathogen])
        #     prob_pathogen_causing_pneumonia['bronchiolitis'] = 1 - prob_pathogen_causing_pneumonia['pneumonia']
        #
        # # Determine the severity a disease will reach:
        # prob_progress_clinical_severe_case = pd.DataFrame(index=df.loc[person_id_that_acquire_pathogen].index,
        #                                                   columns=['non-severe', 'severe'])
        # for disease in m.diseases:
        #     prob_progress_clinical_severe_case['severe'] = \
        #         m.progression_to_clinical_severe_penumonia[disease].predict(df.loc[person_id_that_acquire_pathogen])
        #     prob_progress_clinical_severe_case['non-severe'] = 1 - prob_progress_clinical_severe_case['severe']

        # Determine which pathogen each person will acquire (among those who will get a pathogen)
        # and create the event for the onset of new infection
        for person_id in person_id_that_acquire_pathogen:
            # ----------------------- Allocate a pathogen to the person ----------------------
            p_by_pathogen = probs_of_acquiring_pathogen.loc[person_id].values
            # print(sum(p_by_pathogen))
            normalised_p_by_pathogen = p_by_pathogen / sum(p_by_pathogen)
            # print(sum(normalised_p_by_pathogen))
            pathogen = rng.choice(probs_of_acquiring_pathogen.columns, p=normalised_p_by_pathogen)

            # ----------------------- Allocate the underlying condition caused by pathogen ----------------------
            # p_disease_by_pathogen = prob_pathogen_causing_pneumonia.loc[person_id].values
            # underlying_condition = rng.choice(prob_pathogen_causing_pneumonia.columns, p=p_disease_by_pathogen)

            # ----------------------- Allocate a date of onset of ALRI ----------------------
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

            # # ----------------------- Determine outcomes for this case ----------------------
            # # severity of the disease ----------------------
            # p_severity_by_disease = prob_progress_clinical_severe_case.loc[person_id].values
            # severity = rng.choice(prob_progress_clinical_severe_case.columns, p=p_severity_by_disease)
            #
            # # death status ----------------------
            # risk_of_death = m.risk_of_death_clinical_severe_pneumonia[underlying_condition].predict(
            #     df.loc[[person_id]]).values[0]
            # will_die = rng.rand() < risk_of_death

            # ----------------------- Create the event for the onset of infection -------------------
            # NB. The symptoms are scheduled by the SymptomManager to 'autoresolve' after the duration
            #       of the pneumonia.
            self.sim.schedule_event(
                event=AcuteLowerRespiratoryInfectionIncidentCase(
                    module=self.module,
                    person_id=person_id,
                    pathogen=pathogen,
                    # disease=underlying_condition,
                    # severity=severity,
                    duration_in_days=duration_in_days_of_pneumonia,
                    # will_die=will_die,
                    symptoms=symptoms_for_this_person
                ),
                date=date_onset
            )


class AcuteLowerRespiratoryInfectionIncidentCase(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of the infection that causes pneumonia.
    """

    def __init__(self, module, person_id, pathogen, duration_in_days, symptoms):
        super().__init__(module, person_id=person_id)
        self.pathogen = pathogen
        # self.severity = severity
        # self.disease = disease
        self.duration_in_days = duration_in_days
        # self.will_die = will_die
        self.symptoms = symptoms

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        rng = self.module.rng

        # The event should not run if the person is not currently alive
        if not df.at[person_id, 'is_alive']:
            return

        # Update the properties in the dataframe:
        df.at[person_id, 'ri_last_ALRI_pathogen'] = self.pathogen
        df.at[person_id, 'ri_last_ALRI_date_of_onset'] = self.sim.date
        df.at[person_id, 'ri_last_clinical_severity'] = 'non-severe'

        # Onset symptoms:
        for symptom in self.symptoms:
            self.module.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove='+',
                disease_module=self.module,
                duration_in_days=self.duration_in_days
            )

        # Determine which disease a pathogen will cause
        prob_pathogen_causing_pneumonia = pd.DataFrame(index=[person_id])
        for pathogen in m.pathogens:
            prob_pathogen_causing_pneumonia = \
                m.pathogens_causing_pneumonia_as_underlying_condition[pathogen].predict((df.loc[[person_id]]).values[0])
        will_have_pneumonia = rng.rand() < prob_pathogen_causing_pneumonia
        if will_have_pneumonia:
            df.at[person_id, 'ri_true_underlying_condition'] = 'pneumonia'
        else:
            df.at[person_id, 'ri_true_underlying_condition'] = 'bronchiolitis'

        # Determine progression to 'severe clinical pneumonia'
        date_of_outcome = self.module.sim.date + DateOffset(days=self.duration_in_days)
        prob_progress_clinical_severe_case = pd.DataFrame(index=[person_id])
        for disease in m.diseases:
            prob_progress_clinical_severe_case = \
                m.progression_to_clinical_severe_penumonia[disease].predict(df.loc[[person_id]]).values[0]
        will_progress_to_severe = rng.rand() < prob_progress_clinical_severe_case

        if will_progress_to_severe:
            df.at[person_id, 'ri_last_ALRI_recovered_date'] = pd.NaT
            df.at[person_id, 'ri_last_ALRI_death_date'] = pd.NaT
            date_onset_clinical_severe = self.module.sim.date + DateOffset(
                days=np.random.randint(2, self.duration_in_days))
            self.sim.schedule_event(SeverePneumoniaEvent(self.module, person_id,
                                                         duration_in_days=self.duration_in_days),
                                    date_onset_clinical_severe)
        else:
            df.at[person_id, 'ri_last_ALRI_recovered_date'] = date_of_outcome
            df.at[person_id, 'ri_last_ALRI_death_date'] = pd.NaT

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
            It sets the property 'ri_last_pneumonia_severity' to 'severe' and schedules the death.
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
                df.at[person_id, 'ri_last_ALRI_date_of_onset'] + DateOffset(days=self.duration_in_days)
            prob_death_by_disease = pd.DataFrame(index=[person_id])
            for disease in m.diseases:
                prob_death_by_disease = \
                    m.risk_of_death_clinical_severe_pneumonia[disease].predict(df.loc[[person_id]]).values[0]
            death_outcome = rng.rand() < prob_death_by_disease

            if death_outcome:
                df.at[person_id, 'ri_last_ALRI_recovered_date'] = pd.NaT
                df.at[person_id, 'ri_last_ALRI_death_date'] = date_of_outcome
                self.sim.schedule_event(PneumoniaDeathEvent(self.module, person_id),
                                        date_of_outcome)
            else:
                df.at[person_id, 'ri_last_ALRI_recovered_date'] = date_of_outcome
                df.at[person_id, 'ri_last_ALRI_death_date'] = pd.NaT


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
        df.at[person_id, 'ri_last_ALRI_recovered_date'] = self.sim.date
        df.at[person_id, 'ri_last_ALRI_death_date'] = pd.NaT

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
            (df.at[person_id, 'ri_last_ALRI_death_date'] == self.sim.date):
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
