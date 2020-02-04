"""
Childhood Diarrhoea Module
Documentation: '04 - Methods Repository/Childhood Disease Methods.docx'
"""
import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Diarrhoea(Module):
    # Declare the pathogens that this module will simulate:
    pathogens = {
        'rotavirus',
        'shigella',
        'adenovirus',
        'cryptosporidium',
        'campylobacter',
        'ST-ETEC',
        'sapovirus',
        'norovirus',
        'astrovirus',
        'tEPEC'}

    PARAMETERS = {
        'base_incidence_diarrhoea_by_rotavirus':
            Parameter(Types.LIST, 'incidence of diarrhoea caused by rotavirus in age groups 0-11, 12-23, 24-59 months '
                      ),
        'base_incidence_diarrhoea_by_shigella':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by shigella spp in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_adenovirus':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by adenovirus 40/41 in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_cryptosporidium':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by cryptosporidium in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_campylobacter':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by campylobacter spp in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_ST-ETEC':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by ST-ETEC in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_sapovirus':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by sapovirus in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_norovirus':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by norovirus in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_astrovirus':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by astrovirus in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_tEPEC':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by tEPEC in age groups 0-11, 12-23, 24-59 months'
                      ),
        'rr_gi_diarrhoea_HHhandwashing':
            Parameter(Types.REAL, 'relative rate of diarrhoea with household handwashing with soap'
                      ),
        'rr_gi_diarrhoea_improved_sanitation':
            Parameter(Types.REAL, 'relative rate of diarrhoea for improved sanitation'
                      ),
        'rr_gi_diarrhoea_clean_water':
            Parameter(Types.REAL, 'relative rate of diarrhoea for access to clean drinking water'
                      ),
        'rr_gi_diarrhoea_HIV':
            Parameter(Types.REAL, 'relative rate of diarrhoea for HIV positive status'
                      ),
        'rr_gi_diarrhoea_SAM':
            Parameter(Types.REAL, 'relative rate of diarrhoea for severe malnutrition'
                      ),
        'rr_gi_diarrhoea_excl_breast':
            Parameter(Types.REAL, 'relative rate of diarrhoea for exclusive breastfeeding upto 6 months'
                      ),
        'rr_gi_diarrhoea_cont_breast':
            Parameter(Types.REAL, 'relative rate of diarrhoea for continued breastfeeding 6 months to 2 years'
                      ),
        'rr_gi_diarrhoea_rotavirus_vaccination':
            Parameter(Types.REAL, 'relative rate of diarrhoea for rotavirus vaccine'
                      ),
        'proportion_AWD_by_rotavirus':
            Parameter(Types.REAL, 'acute diarrhoea type caused by rotavirus'
                      ),
        'proportion_AWD_by_shigella':
            Parameter(Types.REAL, 'acute diarrhoea type caused by shigella'
                      ),
        'proportion_AWD_by_adenovirus':
            Parameter(Types.REAL, 'acute diarrhoea type caused by adenovirus'
                      ),
        'proportion_AWD_by_cryptosporidium':
            Parameter(Types.REAL, 'acute diarrhoea type caused by cryptosporidium'
                      ),
        'proportion_AWD_by_campylobacter':
            Parameter(Types.REAL, 'acute diarrhoea type caused by campylobacter'
                      ),
        'proportion_AWD_by_ST-ETEC':
            Parameter(Types.REAL, 'acute diarrhoea type caused by ST-ETEC'
                      ),
        'proportion_AWD_by_sapovirus':
            Parameter(Types.REAL, 'acute diarrhoea type caused by sapovirus'
                      ),
        'proportion_AWD_by_norovirus':
            Parameter(Types.REAL, 'acute diarrhoea type caused by norovirus'
                      ),
        'proportion_AWD_by_astrovirus':
            Parameter(Types.REAL, 'acute diarrhoea type caused by astrovirus'
                      ),
        'proportion_AWD_by_tEPEC':
            Parameter(Types.REAL, 'acute diarrhoea type caused by tEPEC'
                      ),
        'fever_by_rotavirus':
            Parameter(Types.REAL, 'fever caused by rotavirus'
                      ),
        'fever_by_shigella':
            Parameter(Types.REAL, 'fever caused by shigella'
                      ),
        'fever_by_adenovirus':
            Parameter(Types.REAL, 'fever caused by adenovirus'
                      ),
        'fever_by_cryptosporidium':
            Parameter(Types.REAL, 'fever caused by cryptosporidium'
                      ),
        'fever_by_campylobacter':
            Parameter(Types.REAL, 'fever caused by campylobacter'
                      ),
        'fever_by_ST-ETEC':
            Parameter(Types.REAL, 'fever caused by ST-ETEC'
                      ),
        'fever_by_sapovirus':
            Parameter(Types.REAL, 'fever caused by sapovirus'
                      ),
        'fever_by_norovirus':
            Parameter(Types.REAL, 'fever caused by norovirus'
                      ),
        'fever_by_astrovirus':
            Parameter(Types.REAL, 'fever caused by astrovirus'
                      ),
        'fever_by_tEPEC':
            Parameter(Types.REAL, 'fever caused by tEPEC'
                      ),
        'vomiting_by_rotavirus':
            Parameter(Types.REAL, 'vomiting caused by rotavirus'
                      ),
        'vomiting_by_shigella':
            Parameter(Types.REAL, 'vomiting caused by shigella'
                      ),
        'vomiting_by_adenovirus':
            Parameter(Types.REAL, 'vomiting caused by adenovirus'
                      ),
        'vomiting_by_cryptosporidium':
            Parameter(Types.REAL, 'vomiting caused by cryptosporidium'
                      ),
        'vomiting_by_campylobacter':
            Parameter(Types.REAL, 'vomiting caused by campylobacter'
                      ),
        'vomiting_by_ST-ETEC':
            Parameter(Types.REAL, 'vomiting caused by ST-ETEC'
                      ),
        'vomiting_by_sapovirus':
            Parameter(Types.REAL, 'vomiting caused by sapovirus'
                      ),
        'vomiting_by_norovirus':
            Parameter(Types.REAL, 'vomiting caused by norovirus'
                      ),
        'vomiting_by_astrovirus':
            Parameter(Types.REAL, 'vomiting caused by astrovirus'
                      ),
        'vomiting_by_tEPEC':
            Parameter(Types.REAL, 'vomiting caused by tEPEC'
                      ),
        'dehydration_by_rotavirus':
            Parameter(Types.REAL, 'any dehydration caused by rotavirus'
                      ),
        'dehydration_by_shigella':
            Parameter(Types.REAL, 'any dehydration caused by shigella'
                      ),
        'dehydration_by_adenovirus':
            Parameter(Types.REAL, 'any dehydration caused by adenovirus'
                      ),
        'dehydration_by_cryptosporidium':
            Parameter(Types.REAL, 'any dehydration caused by cryptosporidium'
                      ),
        'dehydration_by_campylobacter':
            Parameter(Types.REAL, 'any dehydration caused by campylobacter'
                      ),
        'dehydration_by_ST-ETEC':
            Parameter(Types.REAL, 'any dehydration caused by ST-ETEC'
                      ),
        'dehydration_by_sapovirus':
            Parameter(Types.REAL, 'any dehydration caused by sapovirus'
                      ),
        'dehydration_by_norovirus':
            Parameter(Types.REAL, 'any dehydration caused by norovirus'
                      ),
        'dehydration_by_astrovirus':
            Parameter(Types.REAL, 'any dehydration caused by astrovirus'
                      ),
        'dehydration_by_tEPEC':
            Parameter(Types.REAL, 'any dehydration caused by tEPEC'
                      ),
        'prolonged_diarr_rotavirus':
            Parameter(Types.REAL, 'prolonged episode by rotavirus'
                      ),
        'prolonged_diarr_shigella':
            Parameter(Types.REAL, 'prolonged episode by shigella'
                      ),
        'prolonged_diarr_adenovirus':
            Parameter(Types.REAL, 'prolonged episode by adenovirus'
                      ),
        'prolonged_diarr_cryptosporidium':
            Parameter(Types.REAL, 'prolonged episode by cryptosporidium'
                      ),
        'prolonged_diarr_campylobacter':
            Parameter(Types.REAL, 'prolonged episode by campylobacter'
                      ),
        'prolonged_diarr_ST-ETEC':
            Parameter(Types.REAL, 'prolonged episode by ST-ETEC'
                      ),
        'prolonged_diarr_sapovirus':
            Parameter(Types.REAL, 'prolonged episode by sapovirus'
                      ),
        'prolonged_diarr_norovirus':
            Parameter(Types.REAL, 'prolonged episode by norovirus'
                      ),
        'prolonged_diarr_astrovirus':
            Parameter(Types.REAL, 'prolonged episode by norovirus'
                      ),
        'prolonged_diarr_tEPEC':
            Parameter(Types.REAL, 'prolonged episode by tEPEC'
                      ),
        'prob_prolonged_to_persistent_diarr':
            Parameter(Types.REAL, 'probability of prolonged diarrhoea becoming persistent diarrhoea'
                      ),
        'prob_dysentery_become_persistent':
            Parameter(Types.REAL, 'probability of dysentery becoming persistent diarrhoea'
                      ),
        'prob_watery_diarr_become_persistent':
            Parameter(Types.REAL, 'probability of acute watery diarrhoea becoming persistent diarrhoea, '
                                  'for children under 11 months, no SAM, no HIV'
                      ),
        'rr_bec_persistent_age12to23':
            Parameter(Types.REAL,
                      'relative rate of acute diarrhoea becoming persistent diarrhoea for age 12 to 23 months'
                      ),
        'rr_bec_persistent_age24to59':
            Parameter(Types.REAL,
                      'relative rate of acute diarrhoea becoming persistent diarrhoea for age 24 to 59 months'
                      ),
        'rr_bec_persistent_HIV':
            Parameter(Types.REAL,
                      'relative rate of acute diarrhoea becoming persistent diarrhoea for HIV positive'
                      ),
        'rr_bec_persistent_SAM':
            Parameter(Types.REAL,
                      'relative rate of acute diarrhoea becoming persistent diarrhoea for severely acute malnutrition'
                      ),
        'rr_bec_persistent_excl_breast':
            Parameter(Types.REAL,
                      'relative rate of acute diarrhoea becoming persistent diarrhoea for exclusive breastfeeding'
                      ),
        'rr_bec_persistent_cont_breast':
            Parameter(Types.REAL,
                      'relative rate of acute diarrhoea becoming persistent diarrhoea for continued breastfeeding'
                      ),
        'case_fatality_rate_AWD':
            Parameter(Types.REAL, 'case fatality rate for acute watery diarrhoea cases'
                      ),
        'case_fatality_rate_dysentery':
            Parameter(Types.REAL, 'case fatality rate for dysentery cases'
                      ),
        'case_fatality_rate_persistent':
            Parameter(Types.REAL, 'case fatality rate for persistent diarrhoea cases'
                      ),
        'rr_diarr_death_age12to23mo':
            Parameter(Types.REAL,
                      'relative rate of diarrhoea death for ages 12 to 23 months'
                      ),
        'rr_diarr_death_age24to59mo':
            Parameter(Types.REAL,
                      'relative rate of diarrhoea death for ages 24 to 59 months'
                      ),
        'rr_diarr_death_dehydration':
            Parameter(Types.REAL, 'relative rate of diarrhoea death for cases with dehyadration'
                      ),
        'rr_diarr_death_HIV':
            Parameter(Types.REAL, 'relative rate of diarrhoea death for HIV'
                      ),
        'rr_diarr_death_SAM':
            Parameter(Types.REAL, 'relative rate of diarrhoea death for severe acute malnutrition'
                      )
    }

    PROPERTIES = {
        # ---- The pathogen which is caused the diarrhoea  ----
        'gi_last_diarrhoea_pathogen': Property(Types.CATEGORICAL,
                                               'Attributable pathogen for the last episode of diarrhoea',
                                               categories=list(pathogens) + ['none']),

        # ---- Classification of the type of diarrhoaea that is caused  ----
        'gi_last_diarrhoea_type': Property(Types.CATEGORICAL,
                                           'Type of the last episode of diarrhoea',
                                           categories=['none',
                                                       'acute',
                                                       'prolonged',
                                                       'persistent',
                                                       'severe_persistent']),

        'gi_last_dehydration_status': Property(Types.CATEGORICAL,
                                               'Dehydration status during the last episode of diarrhoea',
                                               categories=['no dehydration',
                                                           'some dehydration',
                                                           'severe dehydration']),

        # ---- Internal variables to schedule onset and deaths due to diarhoaea  ----
        'gi_last_diarrhoea_date_of_onset': Property(Types.DATE, 'date of onset of last episode of diarrhoea'),
        'gi_last_diarrhoea_recovered_date': Property(Types.DATE, 'date of recovery from last episode of diarrhoea'),
        'gi_last_diarrhoea_death_date': Property(Types.DATE, 'date of death caused by last episode of diarrhoea'),

        # ---- Temporary Variables: To be replaced with the properites of other modules ----
        'tmp_malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'tmp_exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'tmp_continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
    }

    # Declare symptoms
    SYMPTOMS = {'watery_diarrhoea', 'bloody_diarrhoea', 'fever', 'vomiting', 'dehydration', 'prolonged_diarrhoea'}

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # dict to hold equations in for the incidence of pathogens:
        self.incidence_equations_by_pathogen = dict()

        # dict to hold the probability of onset of different types of symptom given a pathgoen:
        self.prob_symptoms = dict()

        # dict to hold the DALY weights
        self.daly_wts = dict()

        # dict to hold counters for the number of episodes of diarrhoea by pathogen-type and age-group
        # (0yrs, 1yrs, 2-4yrs)
        blank_counter = dict(zip(self.pathogens, [0]*len(self.pathogens)))
        self.incident_cases_counter_blank = {
            '0y': blank_counter,
            '1y': blank_counter,
            '2-4y': blank_counter
        }
        self.incident_cases_counter = copy.deepcopy(self.incident_cases_counter_blank)

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module """
        p = self.parameters

        # Read parameters from the resourcefile
        self.load_parameters_from_dataframe(
            pd.read_excel(
                Path(self.resourcefilepath) / 'ResourceFile_Childhood_Diarrhoea.xlsx', sheet_name='Parameter_values')
        )

        # Check that every value has been read-in successfully
        for param_name, type in self.PARAMETERS.items():
            assert param_name in self.parameters, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert isinstance(self.parameters[param_name], type.python_type), f'Parameter "{param_name}" is not read in correctly from the resourcefile.'

        # Get DALY weights
        if 'HealthBurden' in self.sim.modules.keys():
            self.daly_wts['mild_diarrhoea'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=32)
            self.daly_wts['moderate_diarrhoea'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=35)
            self.daly_wts['severe_diarrhoea'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=34)

        # --------------------------------------------------------------------------------------------
        # Make a dict to hold the equations that govern the probability that a person acquires diarrhaoe
        # that is caused (primarily) by a pathogen

        self.incidence_equations_by_pathogen.update({
            'rotavirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('age_years')
                                         .when('.between(0,0)', p['base_incidence_diarrhoea_by_rotavirus'][0])
                                         .when('.between(1,1)', p['base_incidence_diarrhoea_by_rotavirus'][1])
                                         .when('.between(2,4)', p['base_incidence_diarrhoea_by_rotavirus'][2])
                                         .otherwise(0.0),
                                     # Predictor('li_no_access_handwashing')
                                     # .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                     # Predictor('li_no_clean_drinking_water').
                                     # when(False, m.rr_gi_diarrhoea_clean_water),
                                     # Predictor('li_unimproved_sanitation').
                                     # when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                     # # Predictor('hv_inf').
                                     # # when(True, m.rr_gi_diarrhoea_HIV),
                                     # Predictor('malnutrition').
                                     # when(True, m.rr_gi_diarrhoea_SAM),
                                     # Predictor('exclusive_breastfeeding').
                                     # when(False, m.rr_gi_diarrhoea_excl_breast)
                                     )
        })

        self.incidence_equations_by_pathogen.update({
            'shigella': LinearModel(LinearModelType.MULTIPLICATIVE,
                                    1.0,
                                    Predictor('age_years')
                                        .when('.between(0,0)', p['base_incidence_diarrhoea_by_shigella'][0])
                                        .when('.between(1,1)', p['base_incidence_diarrhoea_by_shigella'][1])
                                        .when('.between(2,4)', p['base_incidence_diarrhoea_by_shigella'][2])
                                        .otherwise(0.0),
                                    # Predictor('li_no_access_handwashing')
                                    # .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                    # Predictor('li_no_clean_drinking_water').
                                    # when(False, m.rr_gi_diarrhoea_clean_water),
                                    # Predictor('li_unimproved_sanitation').
                                    # when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                    # # Predictor('hv_inf').
                                    # # when(True, m.rr_gi_diarrhoea_HIV),
                                    # Predictor('malnutrition').
                                    # when(True, m.rr_gi_diarrhoea_SAM),
                                    # Predictor('exclusive_breastfeeding').
                                    # when(False, m.rr_gi_diarrhoea_excl_breast)
                                    )
        })

        self.incidence_equations_by_pathogen.update({
            'adenovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                      1.0,
                                      Predictor('age_years')
                                          .when('.between(0,0)', p['base_incidence_diarrhoea_by_adenovirus'][0])
                                          .when('.between(1,1)', p['base_incidence_diarrhoea_by_adenovirus'][1])
                                          .when('.between(2,4)', p['base_incidence_diarrhoea_by_adenovirus'][2])
                                          .otherwise(0.0),
                                      # Predictor('li_no_access_handwashing')
                                      # .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                      # Predictor('li_no_clean_drinking_water').
                                      # when(False, m.rr_gi_diarrhoea_clean_water),
                                      # Predictor('li_unimproved_sanitation').
                                      # when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                      # # Predictor('hv_inf').
                                      # # when(True, m.rr_gi_diarrhoea_HIV),
                                      # Predictor('malnutrition').
                                      # when(True, m.rr_gi_diarrhoea_SAM),
                                      # Predictor('exclusive_breastfeeding').
                                      # when(False, m.rr_gi_diarrhoea_excl_breast)
                                      )
        })

        self.incidence_equations_by_pathogen.update({
            'cryptosporidium': LinearModel(LinearModelType.MULTIPLICATIVE,
                                           1.0,
                                           Predictor('age_years')
                                               .when('.between(0,0)', p['base_incidence_diarrhoea_by_cryptosporidium'][0])
                                               .when('.between(1,1)', p['base_incidence_diarrhoea_by_cryptosporidium'][1])
                                               .when('.between(2,4)', p['base_incidence_diarrhoea_by_cryptosporidium'][2])
                                               .otherwise(0.0),
                                           # Predictor('li_no_access_handwashing')
                                           # .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                           # Predictor('li_no_clean_drinking_water').
                                           # when(False, m.rr_gi_diarrhoea_clean_water),
                                           # Predictor('li_unimproved_sanitation').
                                           # when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                           # # Predictor('hv_inf').
                                           # # when(True, m.rr_gi_diarrhoea_HIV),
                                           # Predictor('malnutrition').
                                           # when(True, m.rr_gi_diarrhoea_SAM),
                                           # Predictor('exclusive_breastfeeding').
                                           # when(False, m.rr_gi_diarrhoea_excl_breast)
                                           )
        })

        self.incidence_equations_by_pathogen.update({
            'campylobacter': LinearModel(LinearModelType.MULTIPLICATIVE,
                                         1.0,
                                         Predictor('age_years')
                                             .when('.between(0,0)', p['base_incidence_diarrhoea_by_campylobacter'][0])
                                             .when('.between(1,1)', p['base_incidence_diarrhoea_by_campylobacter'][1])
                                             .when('.between(2,4)', p['base_incidence_diarrhoea_by_campylobacter'][2])
                                             .otherwise(0.0),
                                         # Predictor('li_no_access_handwashing')
                                         # .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                         # Predictor('li_no_clean_drinking_water').
                                         # when(False, m.rr_gi_diarrhoea_clean_water),
                                         # Predictor('li_unimproved_sanitation').
                                         # when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                         # # Predictor('hv_inf').
                                         # # when(True, m.rr_gi_diarrhoea_HIV),
                                         # Predictor('malnutrition').
                                         # when(True, m.rr_gi_diarrhoea_SAM),
                                         # Predictor('exclusive_breastfeeding').
                                         # when(False, m.rr_gi_diarrhoea_excl_breast)
                                         )
        })

        self.incidence_equations_by_pathogen.update({
            'ST-ETEC': LinearModel(LinearModelType.MULTIPLICATIVE,
                                   1.0,
                                   Predictor('age_years')
                                       .when('.between(0,0)', p['base_incidence_diarrhoea_by_ST-ETEC'][0])
                                       .when('.between(1,1)', p['base_incidence_diarrhoea_by_ST-ETEC'][1])
                                       .when('.between(2,4)', p['base_incidence_diarrhoea_by_ST-ETEC'][2])
                                       .otherwise(0.0),
                                   # Predictor('li_no_access_handwashing')
                                   # .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                   # Predictor('li_no_clean_drinking_water').
                                   # when(False, m.rr_gi_diarrhoea_clean_water),
                                   # Predictor('li_unimproved_sanitation').
                                   # when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                   # # Predictor('hv_inf').
                                   # # when(True, m.rr_gi_diarrhoea_HIV),
                                   # Predictor('malnutrition').
                                   # when(True, m.rr_gi_diarrhoea_SAM),
                                   # Predictor('exclusive_breastfeeding').
                                   # when(False, m.rr_gi_diarrhoea_excl_breast)
                                   )
        })

        self.incidence_equations_by_pathogen.update({
            'sapovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('age_years')
                                         .when('.between(0,0)', p['base_incidence_diarrhoea_by_sapovirus'][0])
                                         .when('.between(1,1)', p['base_incidence_diarrhoea_by_sapovirus'][1])
                                         .when('.between(2,4)', p['base_incidence_diarrhoea_by_sapovirus'][2])
                                         .otherwise(0.0),
                                     # Predictor('li_no_access_handwashing')
                                     # .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                     # Predictor('li_no_clean_drinking_water').
                                     # when(False, m.rr_gi_diarrhoea_clean_water),
                                     # Predictor('li_unimproved_sanitation').
                                     # when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                     # # Predictor('hv_inf').
                                     # # when(True, m.rr_gi_diarrhoea_HIV),
                                     # Predictor('malnutrition').
                                     # when(True, m.rr_gi_diarrhoea_SAM),
                                     # Predictor('exclusive_breastfeeding').
                                     # when(False, m.rr_gi_diarrhoea_excl_breast)
                                     )
        })

        self.incidence_equations_by_pathogen.update({
            'norovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('age_years')
                                         .when('.between(0,0)', p['base_incidence_diarrhoea_by_norovirus'][0])
                                         .when('.between(1,1)', p['base_incidence_diarrhoea_by_norovirus'][1])
                                         .when('.between(2,4)', p['base_incidence_diarrhoea_by_norovirus'][2])
                                         .otherwise(0.0),
                                     # Predictor('li_no_access_handwashing')
                                     # .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                     # Predictor('li_no_clean_drinking_water').
                                     # when(False, m.rr_gi_diarrhoea_clean_water),
                                     # Predictor('li_unimproved_sanitation').
                                     # when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                     # # Predictor('hv_inf').
                                     # # when(True, m.rr_gi_diarrhoea_HIV),
                                     # Predictor('malnutrition').
                                     # when(True, m.rr_gi_diarrhoea_SAM),
                                     # Predictor('exclusive_breastfeeding').
                                     # when(False, m.rr_gi_diarrhoea_excl_breast)
                                     )
        })

        self.incidence_equations_by_pathogen.update({
            'astrovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                      1.0,
                                      Predictor('age_years')
                                          .when('.between(0,0)', p['base_incidence_diarrhoea_by_astrovirus'][0])
                                          .when('.between(1,1)', p['base_incidence_diarrhoea_by_astrovirus'][1])
                                          .when('.between(2,4)', p['base_incidence_diarrhoea_by_astrovirus'][2])
                                          .otherwise(0.0),
                                      # Predictor('li_no_access_handwashing')
                                      # .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                      # Predictor('li_no_clean_drinking_water').
                                      # when(False, m.rr_gi_diarrhoea_clean_water),
                                      # Predictor('li_unimproved_sanitation').
                                      # when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                      # # Predictor('hv_inf').
                                      # # when(True, m.rr_gi_diarrhoea_HIV),
                                      # Predictor('malnutrition').
                                      # when(True, m.rr_gi_diarrhoea_SAM),
                                      # Predictor('exclusive_breastfeeding').
                                      # when(False, m.rr_gi_diarrhoea_excl_breast)
                                      )
        })

        self.incidence_equations_by_pathogen.update({
            'tEPEC': LinearModel(LinearModelType.MULTIPLICATIVE,
                                 1.0,
                                 Predictor('age_years')
                                     .when('.between(0,0)', p['base_incidence_diarrhoea_by_tEPEC'][0])
                                     .when('.between(1,1)', p['base_incidence_diarrhoea_by_tEPEC'][1])
                                     .when('.between(2,4)', p['base_incidence_diarrhoea_by_tEPEC'][2])
                                     .otherwise(0.0),
                                 # Predictor('li_no_access_handwashing')
                                 # .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                 # Predictor('li_no_clean_drinking_water').
                                 # when(False, m.rr_gi_diarrhoea_clean_water),
                                 # Predictor('li_unimproved_sanitation').
                                 # when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                 # # Predictor('hv_inf').
                                 # # when(True, m.rr_gi_diarrhoea_HIV),
                                 # Predictor('malnutrition').
                                 # when(True, m.rr_gi_diarrhoea_SAM),
                                 # Predictor('exclusive_breastfeeding').
                                 # when(False, m.rr_gi_diarrhoea_excl_breast)
                                 )
        })

        # Check that equations have been declared for each of the pathogens
        assert self.pathogens == set(list(self.incidence_equations_by_pathogen.keys()))

        # --------------------------------------------------------------------------------------------
        # Make a dict containing the probability of symptoms onset given acquistion of diarrhoaea caused
        # by a particular pathogen

        self.prob_symptoms.update({
            'rotavirus': {
                'watery_diarrhoea': p['proportion_AWD_by_rotavirus'],
                'bloody_diarrhoea': 1 - p['proportion_AWD_by_rotavirus'],
                'fever': p['fever_by_rotavirus'],
                'vomiting': p['vomiting_by_rotavirus'],
                'dehydration': p['dehydration_by_rotavirus'],
                'prolonged_diarrhoea': p['prolonged_diarr_rotavirus']
            },

            'shigella': {
                'watery_diarrhoea': p['proportion_AWD_by_shigella'],
                'bloody_diarrhoea': 1 - p['proportion_AWD_by_shigella'],
                'fever': p['fever_by_shigella'],
                'vomiting': p['vomiting_by_shigella'],
                'dehydration': p['dehydration_by_shigella'],
                'prolonged_diarrhoea': p['prolonged_diarr_shigella']
            },

            'adenovirus': {
                'watery_diarrhoea': p['proportion_AWD_by_adenovirus'],
                'bloody_diarrhoea': 1 - p['proportion_AWD_by_adenovirus'],
                'fever': p['fever_by_adenovirus'],
                'vomiting': p['vomiting_by_adenovirus'],
                'dehydration': p['dehydration_by_adenovirus'],
                'prolonged_diarrhoea': p['prolonged_diarr_adenovirus']
            },

            'cryptosporidium': {
                'watery_diarrhoea': p['proportion_AWD_by_cryptosporidium'],
                'bloody_diarrhoea': 1 - p['proportion_AWD_by_cryptosporidium'],
                'fever': p['fever_by_cryptosporidium'],
                'vomiting': p['vomiting_by_cryptosporidium'],
                'dehydration': p['dehydration_by_cryptosporidium'],
                'prolonged_diarrhoea': p['prolonged_diarr_cryptosporidium']
            },

            'campylobacter': {
                'watery_diarrhoea': p['proportion_AWD_by_campylobacter'],
                'bloody_diarrhoea': 1 - p['proportion_AWD_by_campylobacter'],
                'fever': p['fever_by_campylobacter'],
                'vomiting': p['vomiting_by_campylobacter'],
                'dehydration': p['dehydration_by_campylobacter'],
                'prolonged_diarrhoea': p['prolonged_diarr_campylobacter']
            },

            'ST-ETEC': {
                'watery_diarrhoea': p['proportion_AWD_by_ST-ETEC'],
                'bloody_diarrhoea': 1 - p['proportion_AWD_by_ST-ETEC'],
                'fever': p['fever_by_ST-ETEC'],
                'vomiting': p['vomiting_by_ST-ETEC'],
                'dehydration': p['dehydration_by_ST-ETEC'],
                'prolonged_diarrhoea': p['prolonged_diarr_ST-ETEC']
            },

            'sapovirus': {
                'watery_diarrhoea': p['proportion_AWD_by_sapovirus'],
                'bloody_diarrhoea': 1 - p['proportion_AWD_by_sapovirus'],
                'fever': p['fever_by_sapovirus'],
                'vomiting': p['vomiting_by_sapovirus'],
                'dehydration': p['dehydration_by_sapovirus'],
                'prolonged_diarrhoea': p['prolonged_diarr_sapovirus']
            },

            'norovirus': {
                'watery_diarrhoea': p['proportion_AWD_by_norovirus'],
                'bloody_diarrhoea': 1 - p['proportion_AWD_by_norovirus'],
                'fever': p['fever_by_norovirus'],
                'vomiting': p['vomiting_by_norovirus'],
                'dehydration': p['dehydration_by_norovirus'],
                'prolonged_diarrhoea': p['prolonged_diarr_norovirus']
            },

            'astrovirus': {
                'watery_diarrhoea': p['proportion_AWD_by_astrovirus'],
                'bloody_diarrhoea': 1 - p['proportion_AWD_by_astrovirus'],
                'fever': p['fever_by_astrovirus'],
                'vomiting': p['vomiting_by_astrovirus'],
                'dehydration': p['dehydration_by_astrovirus'],
                'prolonged_diarrhoea': p['prolonged_diarr_astrovirus']
            },

            'tEPEC': {
                'watery_diarrhoea': p['proportion_AWD_by_tEPEC'],
                'bloody_diarrhoea': 1 - p['proportion_AWD_by_tEPEC'],
                'fever': p['fever_by_rotavirus'],
                'vomiting': p['vomiting_by_tEPEC'],
                'dehydration': p['dehydration_by_tEPEC'],
                'prolonged_diarrhoea': p['prolonged_diarr_tEPEC']
            },
        })

        # Check that each pathogen has a risk of developing each symptom
        assert self.pathogens == set(list(self.prob_symptoms.keys()))

        assert all(
            [
                self.SYMPTOMS == set(list(self.prob_symptoms[pathogen].keys())) \
                for pathogen in self.prob_symptoms.keys()
            ]
        )

        # --------------------------------------------------------------------------------------------
        # Creat the linear model for the risk of dying due to diarrhoea
        self.risk_of_death_diarrhoea = \
            LinearModel(LinearModelType.MULTIPLICATIVE,
                        1.0,
                        Predictor('gi_last_diarrhoea_type')
                            .when('none', 1.0)  # -- to fill in !
                            .when('acute', 1.0)  # -- to fill in !
                            .when('prolonged', 1.0)  # -- to fill in !
                            .when('persistent', 1.0)  # -- to fill in !
                            .when('severe_persistent', 1.0),  # -- to fill in !
                        Predictor('age_years')
                            .when('.between(1,2)', p['rr_diarr_death_age12to23mo'])
                            .when('.between(2,4)', p['rr_diarr_death_age24to59mo'])
                            .otherwise(0.0)
                        # # Predictor('hv_inf').
                        # # when(True, m.rr_gi_diarrhoea_HIV),
                        # Predictor('malnutrition').
                        # when(True, m.rr_gi_diarrhoea_SAM)
                        )

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_population(self, population):
        """
        Sets that there is no one with diarrahoea at initiation.
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals

        # ---- Key Current Status Classification Properties ----
        df['gi_last_diarrhoea_pathogen'].values[:] = 'none'
        df['gi_last_diarrhoea_type'].values[:] = 'none'
        df['gi_last_dehydration_status'].values[:] = 'no dehydration'

        # ---- Internal values ----
        df['gi_last_diarrhoead_date_of_onset'] = pd.NaT
        df['gi_last_diarrhoea_recovered_date'] = pd.NaT
        df['gi_last_diarrhoea_death_date'] = pd.NaT

        # ---- Temporary values ----
        df['tmp_malnutrition'] = False
        df['tmp_exclusive_breastfeeding'] = False
        df['tmp_continued_breastfeeding'] = False

    def initialise_simulation(self, sim):
        # Schedule the main event:
        sim.schedule_event(DiarrhoeaPollingEvent(self), sim.date + DateOffset(months=0))

        # Schedule logging event to occur at the end of each year of the simulation
        sim.schedule_event(DiarrhoeaLoggingEvent(self), sim.date + DateOffset(months=12))

    def on_birth(self, mother_id, child_id):
        """
        On birth, all children will have no dirrhoea
        """
        df = self.sim.population.props

        # ---- Key Current Status Classification Properties ----
        df.at[child_id, 'gi_last_diarrhoea_pathogen'] = 'none'
        df.at[child_id, 'gi_last_diarrhoea_type'] = 'none'
        df.at[child_id, 'gi_last_dehydration_status'] = 'no dehydration'

        # ---- Internal values ----
        df.at[child_id, 'gi_last_diarrhoea_date_of_onset'] = pd.NaT
        df.at[child_id, 'gi_last_diarrhoea_recovered_date'] = pd.NaT
        df.at[child_id, 'gi_last_diarrhoea_death_date'] = pd.NaT

        # ---- Temporary values ----
        df.at[child_id, 'tmp_malnutrition'] = False
        df.at[child_id, 'tmp_exclusive_breastfeeding'] = False
        df.at[child_id, 'tmp_continued_breastfeeding'] = False

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        logger.debug('This is Diarrhoea, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)
        pass

    def report_daly_values(self):
        """
        This returns DALYS values relating to the current status of persons
        """
        df = self.sim.population.props
        p = self.parameters

        # Map the status during last episode to a daly value and zero-out if the last episode is not current
        total_daly_values = df.loc[df['is_alive'], 'gi_last_diarrhoea_type'].map({
            'none': 0.0,
            'acute': self.daly_wts['mild_diarrhoea'],
            'prolonged':self.daly_wts['moderate_diarrhoea'],
            'persistent': self.daly_wts['moderate_diarrhoea'],
            'severe_persistent': self.daly_wts['severe_diarrhoea']
        })

        mask_currently_has_diarrhoaea = (df['gi_last_diarrhoea_date_of_onset'] <= self.sim.date)\
                                        & (df['gi_last_diarrhoea_recovered_date'] >= self.sim.date)
        total_daly_values.loc[~mask_currently_has_diarrhoaea] = 0.0

        # Split out by pathogen that causes the diarrahoea
        dummies_for_pathogen = pd.get_dummies(df.loc[total_daly_values.index,
                                                     'gi_last_diarrhoea_pathogen'],
                                              dtype='float')

        daly_values_by_pathogen = dummies_for_pathogen.mul(total_daly_values, axis=0).drop(columns='none')

        return daly_values_by_pathogen


class DiarrhoeaPollingEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        """
        This is the main event that runs the acquisition of pathogens that cause Diaarhoea.
        """

        df = population.props
        rng = self.module.rng
        m = self.module

        # Compute the probabilities of each person getting diarrhoea within the next three months
        mask_could_get_new_diarrhoea_episode = df['is_alive'] \
                                              & (df['age_years'] < 5) \
                                              & (
                                                  (df['gi_last_diarrhoea_recovered_date'] <= self.sim.date) |
                                                  pd.isnull(df['gi_last_diarrhoea_recovered_date'])
                                              )

        probs_of_aquiring_pathogen = pd.DataFrame(index=df.loc[mask_could_get_new_diarrhoea_episode].index)
        for pathogen in m.pathogens:
            probs_of_aquiring_pathogen[pathogen] = m.incidence_equations_by_pathogen[pathogen] \
                .predict(df.loc[mask_could_get_new_diarrhoea_episode])

        # Create the probability of getting 'none' pathogen:
        # (Assumes that pathogens are mutually exclusive)
        probs_of_aquiring_pathogen['none'] = 1 - probs_of_aquiring_pathogen.sum(axis=1)
        assert all(1.00 == probs_of_aquiring_pathogen.sum(axis=1))

        # TODO: could vectorize this: perhaps just looping through those that do get a pathogen
        # Determine which pathogen (if any) each person will acquire
        for person_id in probs_of_aquiring_pathogen.index:
            # ----------------------- Allocate a pathogen (or none) to each person ----------------------
            pathogen = rng.choice(probs_of_aquiring_pathogen.columns, p=probs_of_aquiring_pathogen.loc[person_id].values)
            df.at[person_id, 'gi_last_diarrhoea_pathogen'] = pathogen

            if pathogen != 'none':
                # ----------------------- Allocate a date of onset diarrhoaea ----------------------
                date_onset = self.sim.date + DateOffset(days=np.random.randint(0, 90))
                df.at[person_id, 'gi_last_diarrhoea_date_of_onset'] = date_onset

                # ----------------------- Add this incident cases to the counter ----------------------
                age_grp = df.loc[person_id, ['age_years']]\
                    .map({0 : '0y', 1 : '1y', 2 : '2-4y',  3 : '2-4y',  4 : '2-4y'})[0]
                self.module.incident_cases_counter[age_grp][pathogen] += 1

                # ----------------------- Determine outcomes for this case ----------------------
                risk_of_death = m.risk_of_death_diarrhoea.predict(df.loc[[person_id]]).values[0]
                duration_in_days_of_diarrhoea = 2                                               # <--- To Fill In.

                if rng.rand() < risk_of_death:
                    # This person is expected to die
                    date_of_death = self.sim.date + DateOffset(days=duration_in_days_of_diarrhoea)

                    # Set the date of death in the dataframe.
                    # (Nb. This will be reset to pd.NaT if the death should not to occur due to treatment.)
                    df.at[person_id, 'gi_last_diarrhoea_death_date'] = date_of_death
                    df.at[person_id, 'gi_last_diarrhoea_recovered_date'] = pd.NaT

                    # Schedule the death event
                    self.module.sim.schedule_event(DiarrhoeaDeathEvent(self.module, person_id), date_of_death)

                else:
                    df.at[person_id, 'gi_last_diarrhoea_recovered_date'] = \
                        self.sim.date + DateOffset(days=duration_in_days_of_diarrhoea)
                    df.at[person_id, 'gi_last_diarrhoea_death_date'] = pd.NaT

                # ----------------------- Allocate symptoms to onset of diarrhoea ----------------------
                possible_symptoms_for_this_pathogen = m.prob_symptoms[pathogen]
                for symptom, prob in possible_symptoms_for_this_pathogen.items():
                    if rng.rand() < prob:
                        self.sim.modules['SymptomManager'].change_symptom(
                            person_id=person_id,
                            symptom_string=symptom,
                            add_or_remove='+',
                            disease_module=self.module,
                            date_of_onset=date_onset,
                            duration_in_days=duration_in_days_of_diarrhoea
                        )


class DiarrhoeaDeathEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # Check if person should still die of diarahaoea
        if (df.at[person_id, 'is_alive']) and (df.at[person_id, 'gi_last_diarrhoea_death_date'] == self.sim.date):
            self.sim.schedule_event(demography.InstantaneousDeath(self.module,
                                                                  person_id,
                                                                  cause=df.at[person_id, 'gi_last_diarrhoea_pathogen']
                                                                  ),
                                    self.sim.date)



class DiarrhoeaLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    The event runs every 12 months and logs the number of incident cases of dirarrhoea caused by each pathogen in the
    previous 12 months, among the special age-groups (0 years, 1 years, 2-4 years).
    """
    def __init__(self, module):
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # Log the current status of the counters
        logger.info('%s|incidence_count_by_patho|%s',
                    self.sim.date,
                    self.module.incident_cases_counter
                    )

        # Reset the counters
        self.module.incident_cases_counter = self.module.incident_cases_counter_blank
