"""
Childhood diarrhoea module
Documentation: 04 - Methods Repository/Method_Child_EntericInfection.xlsx
"""

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
        'base_incidence_diarrhoea_by_crypto':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by cryptosporidium in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_campylo':
            Parameter(Types.LIST,
                      'incidence of diarrhoea caused by campylobacter spp in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_incidence_diarrhoea_by_ETEC':
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
        'base_incidence_diarrhoea_by_EPEC':
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
        'proportion_AWD_by_crypto':
            Parameter(Types.REAL, 'acute diarrhoea type caused by cryptosporidium'
                      ),
        'proportion_AWD_by_campylo':
            Parameter(Types.REAL, 'acute diarrhoea type caused by campylobacter'
                      ),
        'proportion_AWD_by_ETEC':
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
        'proportion_AWD_by_EPEC':
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
        'fever_by_crypto':
            Parameter(Types.REAL, 'fever caused by cryptosporidium'
                      ),
        'fever_by_campylo':
            Parameter(Types.REAL, 'fever caused by campylobacter'
                      ),
        'fever_by_ETEC':
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
        'fever_by_EPEC':
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
        'vomiting_by_crypto':
            Parameter(Types.REAL, 'vomiting caused by cryptosporidium'
                      ),
        'vomiting_by_campylo':
            Parameter(Types.REAL, 'vomiting caused by campylobacter'
                      ),
        'vomiting_by_ETEC':
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
        'vomiting_by_EPEC':
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
        'dehydration_by_crypto':
            Parameter(Types.REAL, 'any dehydration caused by cryptosporidium'
                      ),
        'dehydration_by_campylo':
            Parameter(Types.REAL, 'any dehydration caused by campylobacter'
                      ),
        'dehydration_by_ETEC':
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
        'dehydration_by_EPEC':
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
        'prolonged_diarr_crypto':
            Parameter(Types.REAL, 'prolonged episode by cryptosporidium'
                      ),
        'prolonged_diarr_campylo':
            Parameter(Types.REAL, 'prolonged episode by campylobacter'
                      ),
        'prolonged_diarr_ETEC':
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
        'prolonged_diarr_EPEC':
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
                      ),
        'daly_wts':
            Parameter(Types.DICT, 'DALY weights for diarrhoea'
                      ),
    }

    PROPERTIES = {
        # ---- The pathogen which is caused the diarrhoea  ----
        'gi_last_diarrhoea_pathogen': Property(Types.CATEGORICAL,
                                               'Attributable pathogen for the last episode of diarrhoea',
                                               categories=list(pathogens) + ['none']),

        # ---- Classification of the type of diarrhoaea that is caused  ----
        'gi_last_diarrhoea_type': Property(Types.CATEGORICAL, 'Type of the last episode of diarrhoaea',
                                           categories=['none',
                                                       'acute',
                                                       'prolonged',
                                                       'persistent',
                                                       'severe_persistent']),

        'gi_last_dehydration_status': Property(Types.CATEGORICAL,
                                               'Dehydration status during the last episode of diarrhoaea',
                                               categories=['no dehydration',
                                                           'some dehydration',
                                                           'severe dehydration']),

        # ---- Internal variables to schedule onset and deaths due to diarhoaea  ----
        'gi_last_diarrhoea_date_of_onset': Property(Types.DATE, 'date of onset of last episode of diarrhoea'),
        'gi_last_diarrhoea_ep_duration': Property(Types.REAL, 'duration of diarrhoea last episode of diarrhoea'),
        'gi_last_diarrhoea_recovered_date': Property(Types.DATE, 'date of recovery from last episode of dirrhoea'),
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

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module """
        p = self.parameters
        #
        # 'progression_persistent_equation':
        #     Parameter(Types.REAL, 'dict that holds the equations governing the risk of progression'
        #                           ' to persistent diarrhoea'
        #               ),

        # Read parameters from the resourcefile
        self.load_parameters_from_dataframe(
            pd.read_excel(
                Path(self.resourcefilepath) / 'ResourceFile_Childhood_Diarrhoea.xlsx', sheet_name='Parameter_values')
        )

        # Get DALY weights
        p['daly_wts'] = dict()
        if 'HealthBurden' in self.sim.modules.keys():
            p['daly_wts']['mild_diarrhoea'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=32)
            p['daly_wts']['moderate_diarrhoea'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=35)
            p['daly_wts']['severe_diarrhoea'] = \
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
                                           .when('.between(0,0)', p['base_incidence_diarrhoea_by_crypto'][0])
                                           .when('.between(1,1)', p['base_incidence_diarrhoea_by_crypto'][1])
                                           .when('.between(2,4)', p['base_incidence_diarrhoea_by_crypto'][2])
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
                                         .when('.between(0,0)', p['base_incidence_diarrhoea_by_campylo'][0])
                                         .when('.between(1,1)', p['base_incidence_diarrhoea_by_campylo'][1])
                                         .when('.between(2,4)', p['base_incidence_diarrhoea_by_campylo'][2])
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
                                   .when('.between(0,0)', p['base_incidence_diarrhoea_by_ETEC'][0])
                                   .when('.between(1,1)', p['base_incidence_diarrhoea_by_ETEC'][1])
                                   .when('.between(2,4)', p['base_incidence_diarrhoea_by_ETEC'][2])
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
                                 .when('.between(0,0)', p['base_incidence_diarrhoea_by_EPEC'][0])
                                 .when('.between(1,1)', p['base_incidence_diarrhoea_by_EPEC'][1])
                                 .when('.between(2,4)', p['base_incidence_diarrhoea_by_EPEC'][2])
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
                'watery_diarrhoea': p['proportion_AWD_by_crypto'],
                'bloody_diarrhoea': 1 - p['proportion_AWD_by_crypto'],
                'fever': p['fever_by_crypto'],
                'vomiting': p['vomiting_by_crypto'],
                'dehydration': p['dehydration_by_crypto'],
                'prolonged_diarrhoea': p['prolonged_diarr_crypto']
            },

            'campylobacter': {
                'watery_diarrhoea': p['proportion_AWD_by_campylo'],
                'bloody_diarrhoea': 1 - p['proportion_AWD_by_campylo'],
                'fever': p['fever_by_campylo'],
                'vomiting': p['vomiting_by_campylo'],
                'dehydration': p['dehydration_by_campylo'],
                'prolonged_diarrhoea': p['prolonged_diarr_campylo']
            },

            'ST-ETEC': {
                'watery_diarrhoea': p['proportion_AWD_by_ETEC'],
                'bloody_diarrhoea': 1 - p['proportion_AWD_by_ETEC'],
                'fever': p['fever_by_ETEC'],
                'vomiting': p['vomiting_by_ETEC'],
                'dehydration': p['dehydration_by_ETEC'],
                'prolonged_diarrhoea': p['prolonged_diarr_ETEC']
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
                'watery_diarrhoea': p['proportion_AWD_by_EPEC'],
                'bloody_diarrhoea': 1 - p['proportion_AWD_by_EPEC'],
                'fever': p['fever_by_rotavirus'],
                'vomiting': p['vomiting_by_EPEC'],
                'dehydration': p['dehydration_by_EPEC'],
                'prolonged_diarrhoea': p['prolonged_diarr_EPEC']
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
        # Asssign the probability of becoming persisent

        # # # # # # # ASSIGN THE PROBABILITY OF BECOMING PERSISTENT (over 14 days) # # # # # #
        # self.parameters['progression_persistent_equation'] = \
        #     LinearModel(LinearModelType.MULTIPLICATIVE,
        #                 0.2,
        #                 Predictor('age_years')
        #                 .when('.between(0,0)', 1)
        #                 .when('.between(1,1)', m.rr_bec_persistent_age12to23)
        #                 .when('.between(2,4)', m.rr_bec_persistent_age24to59)
        #                 .otherwise(0.0),
        #                 # # Predictor('hv_inf')
        #                 # # .when(False, m.rr_bec_persistent_HIV),
        #                 # Predictor('malnutrition').
        #                 # when(False, m.rr_bec_persistent_SAM),
        #                 # Predictor('exclusive_breastfeeding').
        #                 # when(False, m.rr_bec_persistent_excl_breast)
        #                 )

        # # # # # # # # # # # # ASSIGN THE RATE OF DEATH # # # # # # # # # # # #
        self.risk_of_death_diarrhoea = \
            LinearModel(LinearModelType.MULTIPLICATIVE,
                        1.0,
                        Predictor('gi_diarrhoea_type')
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
        :param population:
        :return:
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals

        # ---- Key Current Status Classification Properties ----
        df['gi_last_diarrhoea_pathogen'].values[:] = 'none'
        df['gi_last_diarrhoea_type'].values[:] = 'none'
        df['gi_last_dehydration_status'].values[:] = 'no dehydration'

        # ---- Internal values ----
        df['gi_last_diarrhoead_date_of_onset'] = pd.NaT
        df['gi_last_diarrhoea_ep_duration'] = pd.NaT
        df['gi_last_diarrhoea_recovered_date'] = pd.NaT
        df['gi_last_diarrhoea_death_date'] = pd.NaT

        # ---- Temporary values ----
        df['tmp_malnutrition'] = False
        df['tmp_exclusive_breastfeeding'] = False
        df['tmp_continued_breastfeeding'] = False

    def initialise_simulation(self, sim):
        # Schedule the main event:
        sim.schedule_event(DiarrhoeaPollingEvent(self), sim.date + DateOffset(months=0))

        # Schedule logging event to every now and and then to repeat
        sim.schedule_event(DiarrhoeaLoggingEvent(self), sim.date + DateOffset(months=0))

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
        df.at[child_id, 'gi_last_diarrhoea_ep_duration'] = pd.NaT
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
        daly_values = df.loc[df['is_alive'], 'gi_last_diarrhoea_type'].map({
            'none': 0.0,
            'acute': p['daly_wts']['mild_diarrhoea'],
            'prolonged': p['daly_wts']['moderate_diarrhoea'],
            'persistent': p['daly_wts']['moderate_diarrhoea'],
            'severe_persistent': p['daly_wts']['severe_diarrhoea']
        })

        mask_currently_has_diarrhoaea = (df['gi_last_diarrhoea_date_of_onset'] <= self.sim.date)\
                                        & (df['gi_last_diarrhoea_recovered_date'] >= self.sim.date)
        daly_values.loc[~mask_currently_has_diarrhoaea] = 0.0

        daly_values.name = ''
        return daly_values


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
        now = self.sim.date

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

        # Determine which pathogen (if any) each person will acquire                   # TODO: This should be vectorized
        for person_id in probs_of_aquiring_pathogen.index:
            # ----------------------- Allocate a pathogen (or none) to each person ----------------------
            pathogen = rng.choice(probs_of_aquiring_pathogen.columns, p=probs_of_aquiring_pathogen.loc[i].values)
            df.at[person_id, 'gi_last_diarrhoea_pathogen'] = pathogen

            if pathogen != 'none':
                # ----------------------- Allocate a date of onset and recovery of diarrhoaea ----------------------
                date_onset = self.sim.date + DateOffset(days=np.random.randint(0, 90))
                duration_in_days = 2

                df.at[person_id, 'gi_last_diarrhoea_date_of_onset'] = date_onset
                df.at[person_id, 'gi_last_diarrhoea_recovered_date'] = date_onset + DateOffset(days=duration_in_days)

                # ----------------------- Allocate symptoms to onset of diarrhoaea ----------------------
                possible_symptoms_for_this_pathogen = m.prob_symptoms[pathogen]
                for symptom, prob in possible_symptoms_for_this_pathogen.items():
                    if rng.rand() < prob:
                        self.sim.modules['SymptomManager'].change_symptom(
                            person_id=person_id,
                            symptom_string=symptom,
                            add_or_remove='+',
                            disease_module=self.module,
                            date_of_onset=date_onset,
                            duration_in_days=duration_in_days
                        )

                # ----------------------- Determine if / when the person will die  ----------------------





            # # ----------------------------------------------------------------------------------------
            # # Then work out the symptoms for this person:
            # for symptom_string, prob in m.parameters['prob_symptoms'][outcome_i].items():
            #     if symptom_string == 'watery_diarrhoea':
            #         if rng.rand() < prob:
            #             df.at[i, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
            #         else:
            #             df.at[i, 'gi_diarrhoea_acute_type'] = 'dysentery'
            #     # determine the dehydration status
            #     if symptom_string == 'dehydration':
            #         if rng.rand() < prob:
            #             df.at[i, 'di_dehydration_present'] = True
            #             if rng.rand() < 0.7:
            #                 df.at[i, 'gi_dehydration_status'] = 'some dehydration'
            #             else:
            #                 df.at[i, 'gi_dehydration_status'] = 'severe dehydration'
            #         else:
            #             df.at[i, 'di_dehydration_present'] = False
            #
            #     # determine the which phase in diarrhoea episode
            #     if symptom_string == 'prolonged_diarrhoea':
            #         if rng.rand() < prob:
            #             df.at[i, 'gi_diarrhoea_type'] = 'prolonged'
            #             if rng.rand() < m.parameters['progression_persistent_equation'].predict(df.loc[[i]]).values[
            #                 0]:
            #                 df.at[i, 'gi_diarrhoea_type'] = 'persistent'
            #             if df.at[i, 'di_dehydration_present']:
            #                 df.at[i, 'gi_persistent_diarrhoea'] = 'severe persistent diarrhoea'
            #             else:
            #                 df.at[i, 'gi_persistent_diarrhoea'] = 'persistent diarrhoea'
            #
            #     # determine the duration of the episode
            #     if df.at[i, 'gi_diarrhoea_type'] == 'acute':
            #         duration = rng.randint(3, 7)
            #     if df.at[i, 'gi_diarrhoea_type'] == 'prolonged':
            #         duration = rng.randint(7, 14)
            #     if df.at[i, 'gi_diarrhoea_type'] == 'persistent':
            #         duration = rng.randint(14, 21)
            #
            #     # # Send the symptoms to the SymptomManager
            #     # self.sim.modules['SymptomManager'].change_symptom(symptom_string=symptom_string,
            #     #                                                   person_id=i,
            #     #                                                   add_or_remove='+',
            #     #                                                   disease_module=self.module,
            #     #                                                   date_of_onset=df.at[i, 'date_of_onset_diarrhoea'],
            #     #                                                   duration_in_days=duration
            #     #                                                   )
            #
            #     # # # # # # HEALTH CARE SEEKING BEHAVIOUR - INTERACTION WITH HSB MODULE # # # # #
            #     # # TODO: when you declare the symptoms in the symptom manager, the health care seeking will follow automatically
            #
            #     # # # # # # # # # # # SCHEDULE DEATH OR RECOVERY # # # # # # # # # # #
            #     if rng.rand() < self.module.parameters['rate_death_diarrhoea'].predict(df.iloc[[i]]).values[0]:
            #         if df.at[i, 'gi_diarrhoea_type'] == 'acute':
            #             random_date = rng.randint(low=4, high=6)
            #             random_days = pd.to_timedelta(random_date, unit='d')
            #             death_event = DeathDiarrhoeaEvent(self.module, person_id=i, cause='diarrhoea')
            #             self.sim.schedule_event(death_event, df.at[i, 'date_of_onset_diarrhoea'] + random_days)
            #         if df.at[i, 'gi_diarrhoea_type'] == 'prolonged':
            #             random_date1 = rng.randint(low=7, high=13)
            #             random_days1 = pd.to_timedelta(random_date1, unit='d')
            #             death_event = DeathDiarrhoeaEvent(self.module, person_id=i,
            #                                               cause='diarrhoea')
            #             self.sim.schedule_event(death_event, df.at[i, 'date_of_onset_diarrhoea'] + random_days1)
            #         if df.at[i, 'gi_diarrhoea_type'] == 'persistent':
            #             random_date2 = rng.randint(low=14, high=30)
            #             random_days2 = pd.to_timedelta(random_date2, unit='d')
            #             death_event = DeathDiarrhoeaEvent(self.module, person_id=i,
            #                                               cause='diarrhoea')
            #             self.sim.schedule_event(death_event, df.at[i, 'date_of_onset_diarrhoea'] + random_days2)
            #     else:
            #         if df.at[i, 'gi_diarrhoea_type'] == 'acute':
            #             random_date = rng.randint(low=4, high=6)
            #             random_days = pd.to_timedelta(random_date, unit='d')
            #             self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=i),
            #                                     df.at[i, 'date_of_onset_diarrhoea'] + random_days)
            #         if df.at[i, 'gi_diarrhoea_type'] == 'prolonged':
            #             random_date1 = rng.randint(low=7, high=13)
            #             random_days1 = pd.to_timedelta(random_date1, unit='d')
            #             self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=i),
            #                                     df.at[i, 'date_of_onset_diarrhoea'] + random_days1)
            #         if df.at[i, 'gi_diarrhoea_type'] == 'persistent':
            #             random_date2 = rng.randint(low=14, high=21)
            #             random_days2 = pd.to_timedelta(random_date2, unit='d')
            #             self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=i),
            #                                     df.at[i, 'date_of_onset_diarrhoea'] + random_days2)





class DeathDiarrhoeaEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id, cause):
        super().__init__(module, person_id=person_id)
        self.cause = cause

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        if df.at[person_id, 'is_alive']:
            # # check if person should still die of diarah
            # if df.at[person_id, 'gi_will_die_of_diarh']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id, cause='diarrhoea'),
                                    self.sim.date)
            df.at[person_id, 'gi_diarrhoea_death_date'] = self.sim.date
            df.at[person_id, 'gi_diarrhoea_death'] = True
            logger.info('This is DeathDiarrhoeaEvent determining if person %d on the date %s will die '
                        'from their disease', person_id, self.sim.date)
            # death_count = sum(person_id)
            # # Log the diarrhoea death information
            # logger.info('%s|death_diarrhoea|%s', self.sim.date,
            #             {'death': sum(death_count)
            #              })


class DiarrhoeaLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        # -----------------------------------------------------------------------------------------------------
        # sum all the counters for previous year
        # count_episodes = df['gi_diarrhoea_count'].sum()
        # pop_under5 = len(df[df.is_alive & (df.age_exact_years < 5)])
        # # overall incidence rate in under 5
        # inc_100cy = (count_episodes / pop_under5) * 100
        # logger.info('%s|episodes_counts|%s', now, {'incidence_per100cy': inc_100cy})
        # logger.info('%s|pop_counts|%s', now, {'pop_len': pop_under5})
        #
        # len_under12mo = len(df[df.is_alive & (df.age_exact_years < 1)])
        # len_11to23mo = len(df[df.is_alive & (df.age_exact_years >= 1) & (df.age_exact_years < 2)])
        # len_24to59mo = len(df[df.is_alive & (df.age_exact_years >= 2) & (df.age_exact_years < 5)])
        #
        # # counts for diarrhoeal episodes cause by Rotavirus
        # if df.loc[df.age_years < 1, 'gi_pathogen_count_rota'].sum():
        #     count_rota_episodes0 = df['gi_pathogen_count_rota'].sum()
        #     inc_100cy_rota_under1 = (count_rota_episodes0 / len_under12mo) * 100
        #     logger.info('%s|episodes_pathogen_counts_0to11mo|%s', now,
        #                 {'rotavirus': inc_100cy_rota_under1})
        # if df.loc[(df.age_years >= 1) & (df.age_years < 2), 'gi_pathogen_count_rota'].sum():
        #     count_rota_episodes1 = df['gi_pathogen_count_rota'].sum()
        #     inc_100cy_rota_1to2yo = (count_rota_episodes1 / len_11to23mo) * 100
        # if df.loc[(df.age_years >= 2) & (df.age_years < 5), 'gi_pathogen_count_rota'].sum():
        #     count_rota_episodes2 = df['gi_pathogen_count_rota'].sum()
        #     inc_100cy_rota_2to5yo = (count_rota_episodes2 / len_24to59mo) * 100
        #
        # # counts for diarrhoeal episodes cause by Shigella
        # if df.loc[df.age_years < 1, 'gi_pathogen_count_shig'].sum():
        #     count_shig_episodes0 = df['gi_pathogen_count_shig'].sum()
        #     inc_100cy_shig_under1 = (count_shig_episodes0 / len_under12mo) * 100
        #     logger.info('%s|episodes_pathogen_counts_0to11mo|%s', now,
        #                 {'shigella': inc_100cy_shig_under1})
        # if df.loc[(df.age_years >= 1) & (df.age_years < 2), 'gi_pathogen_count_shig'].sum():
        #     count_shig_episodes1 = df['gi_pathogen_count_shig'].sum()
        #     inc_100cy_shig_1to2yo = (count_shig_episodes1 / len_11to23mo) * 100
        # if df.loc[(df.age_years >= 2) & (df.age_years < 5), 'gi_pathogen_count_shig'].sum():
        #     count_shig_episodes2 = df['gi_pathogen_count_shig'].sum()
        #     inc_100cy_shig_2to5yo = (count_shig_episodes2 / len_24to59mo) * 100
        #
        # # counts for diarrhoeal episodes cause by Adenovirus
        # if df.loc[df.age_years < 1, 'gi_pathogen_count_adeno'].sum():
        #     count_adeno_episodes0 = df['gi_pathogen_count_adeno'].sum()
        #     inc_100cy_adeno_under1 = (count_adeno_episodes0 / len_under12mo) * 100
        #     logger.info('%s|episodes_pathogen_counts_0to11mo|%s', now,
        #                 {'adenovirus': inc_100cy_adeno_under1})
        # if df.loc[(df.age_years >= 1) & (df.age_years < 2), 'gi_pathogen_count_adeno'].sum():
        #     count_adeno_episodes1 = df['gi_pathogen_count_adeno'].sum()
        #     inc_100cy_adeno_1to2yo = (count_adeno_episodes1 / len_11to23mo) * 100
        # if df.loc[(df.age_years >= 2) & (df.age_years < 5), 'gi_pathogen_count_adeno'].sum():
        #     count_adeno_episodes2 = df['gi_pathogen_count_adeno'].sum()
        #     inc_100cy_adeno_2to5yo = (count_adeno_episodes2 / len_24to59mo) * 100
        #
        # # counts for diarrhoeal episodes cause by Cryptosporidium
        # if df.loc[df.age_years < 1, 'gi_pathogen_count_crypto'].sum():
        #     count_crypto_episodes0 = df['gi_pathogen_count_crypto'].sum()
        #     inc_100cy_crypto_under1 = (count_crypto_episodes0 / len_under12mo) * 100
        #     logger.info('%s|episodes_pathogen_counts_0to11mo|%s', now,
        #                 {'cryptosporidium': inc_100cy_crypto_under1})
        # if df.loc[(df.age_years >= 1) & (df.age_years < 2), 'gi_pathogen_count_crypto'].sum():
        #     count_crypto_episodes1 = df['gi_pathogen_count_crypto'].sum()
        #     inc_100cy_crypto_1to2yo = (count_crypto_episodes1 / len_11to23mo) * 100
        # if df.loc[(df.age_years >= 2) & (df.age_years < 5), 'gi_pathogen_count_crypto'].sum():
        #     count_crypto_episodes2 = df['gi_pathogen_count_crypto'].sum()
        #     inc_100cy_crypto_2to5yo = (count_crypto_episodes2 / len_24to59mo) * 100
        #
        # # counts for diarrhoeal episodes cause by Campylobacter
        # if df.loc[df.age_years < 1, 'gi_pathogen_count_campylo'].sum():
        #     count_campylo_episodes0 = df['gi_pathogen_count_campylo'].sum()
        #     inc_100cy_campylo_under1 = (count_campylo_episodes0 / len_under12mo) * 100
        #     logger.info('%s|episodes_pathogen_counts_0to11mo|%s', now,
        #                 {'campylobacter': inc_100cy_campylo_under1})
        # if df.loc[(df.age_years >= 1) & (df.age_years < 2), 'gi_pathogen_count_campylo'].sum():
        #     count_campylo_episodes1 = df['gi_pathogen_count_campylo'].sum()
        #     inc_100cy_campylo_1to2yo = (count_campylo_episodes1 / len_11to23mo) * 100
        # if df.loc[(df.age_years >= 2) & (df.age_years < 5), 'gi_pathogen_count_campylo'].sum():
        #     count_campylo_episodes2 = df['gi_pathogen_count_campylo'].sum()
        #     inc_100cy_campylo_2to5yo = (count_campylo_episodes2 / len_24to59mo) * 100
        #
        # # counts for diarrhoeal episodes cause by ST-ETEC
        # if df.loc[df.age_years < 1, 'gi_pathogen_count_ETEC'].sum():
        #     count_ETEC_episodes0 = df['gi_pathogen_count_ETEC'].sum()
        #     inc_100cy_ETEC_under1 = (count_ETEC_episodes0 / len_under12mo) * 100
        #     logger.info('%s|episodes_pathogen_counts_0to11mo|%s', now,
        #                 {'ETEC': inc_100cy_ETEC_under1})
        # if df.loc[(df.age_years >= 1) & (df.age_years < 2), 'gi_pathogen_count_ETEC'].sum():
        #     count_ETEC_episodes1 = df['gi_pathogen_count_ETEC'].sum()
        #     inc_100cy_ETEC_1to2yo = (count_ETEC_episodes1 / len_11to23mo) * 100
        # if df.loc[(df.age_years >= 2) & (df.age_years < 5), 'gi_pathogen_count_ETEC'].sum():
        #     count_ETEC_episodes2 = df['gi_pathogen_count_ETEC'].sum()
        #     inc_100cy_ETEC_2to5yo = (count_ETEC_episodes2 / len_24to59mo) * 100
        #
        # # counts for diarrhoeal episodes cause by Sapovirus
        # if df.loc[df.age_years < 1, 'gi_pathogen_count_sapo'].sum():
        #     count_sapo_episodes0 = df['gi_pathogen_count_sapo'].sum()
        #     inc_100cy_sapo_under1 = (count_sapo_episodes0 / len_under12mo) * 100
        #     logger.info('%s|episodes_pathogen_counts_0to11mo|%s', now,
        #                 {'sapovirus': inc_100cy_sapo_under1})
        # if df.loc[(df.age_years >= 1) & (df.age_years < 2), 'gi_pathogen_count_sapo'].sum():
        #     count_sapo_episodes1 = df['gi_pathogen_count_sapo'].sum()
        #     inc_100cy_sapo_1to2yo = (count_sapo_episodes1 / len_11to23mo) * 100
        # if df.loc[(df.age_years >= 2) & (df.age_years < 5), 'gi_pathogen_count_sapo'].sum():
        #     count_sapo_episodes2 = df['gi_pathogen_count_sapo'].sum()
        #     inc_100cy_sapo_2to5yo = (count_sapo_episodes2 / len_24to59mo) * 100
        #
        # # counts for diarrhoeal episodes cause by Norovirus
        # if df.loc[df.age_years < 1, 'gi_pathogen_count_noro'].sum():
        #     count_noro_episodes0 = df['gi_pathogen_count_noro'].sum()
        #     inc_100cy_noro_under1 = (count_noro_episodes0 / len_under12mo) * 100
        #     logger.info('%s|episodes_pathogen_counts_0to11mo|%s', now,
        #                 {'norovirus': inc_100cy_noro_under1})
        # if df.loc[(df.age_years >= 1) & (df.age_years < 2), 'gi_pathogen_count_noro'].sum():
        #     count_noro_episodes1 = df['gi_pathogen_count_noro'].sum()
        #     inc_100cy_noro_1to2yo = (count_noro_episodes1 / len_11to23mo) * 100
        # if df.loc[(df.age_years >= 2) & (df.age_years < 5), 'gi_pathogen_count_noro'].sum():
        #     count_noro_episodes2 = df['gi_pathogen_count_noro'].sum()
        #     inc_100cy_noro_2to5yo = (count_noro_episodes2 / len_24to59mo) * 100
        #
        # # counts for diarrhoeal episodes cause by Astrovirus
        # if df.loc[df.age_years < 1, 'gi_pathogen_count_astro'].sum():
        #     count_astro_episodes0 = df['gi_pathogen_count_astro'].sum()
        #     inc_100cy_astro_under1 = (count_astro_episodes0 / len_under12mo) * 100
        #     logger.info('%s|episodes_pathogen_counts_0to11mo|%s', now,
        #                 {'astrovirus': inc_100cy_astro_under1})
        # if df.loc[(df.age_years >= 1) & (df.age_years < 2), 'gi_pathogen_count_astro'].sum():
        #     count_astro_episodes1 = df['gi_pathogen_count_astro'].sum()
        #     inc_100cy_astro_1to2yo = (count_astro_episodes1 / len_11to23mo) * 100
        # if df.loc[(df.age_years >= 2) & (df.age_years < 5), 'gi_pathogen_count_astro'].sum():
        #     count_astro_episodes2 = df['gi_pathogen_count_astro'].sum()
        #     inc_100cy_astro_2to5yo = (count_astro_episodes2 / len_24to59mo) * 100
        #
        # # counts for diarrhoeal episodes cause by tEPEC
        # if df.loc[df.age_years < 1, 'gi_pathogen_count_EPEC'].sum():
        #     count_EPEC_episodes0 = df['gi_pathogen_count_EPEC'].sum()
        #     inc_100cy_EPEC_under1 = (count_EPEC_episodes0 / len_under12mo) * 100
        #     logger.info('%s|episodes_pathogen_counts_0to11mo|%s', now,
        #                 {'EPEC': inc_100cy_EPEC_under1})
        # if df.loc[(df.age_years >= 1) & (df.age_years < 2), 'gi_pathogen_count_EPEC'].sum():
        #     count_EPEC_episodes1 = df['gi_pathogen_count_EPEC'].sum()
        #     inc_100cy_EPEC_1to2yo = (count_EPEC_episodes1 / len_11to23mo) * 100
        # if df.loc[(df.age_years >= 2) & (df.age_years < 5), 'gi_pathogen_count_EPEC'].sum():
        #     count_EPEC_episodes2 = df['gi_pathogen_count_EPEC'].sum()
        #     inc_100cy_EPEC_2to5yo = (count_EPEC_episodes2 / len_24to59mo) * 100
        #
        # # TODO: I think it's easier to output the number of events in the logger and work out the incidence afterwards.
        # # So, I would propose just logging value counts.
        #
        # # log the clinical cases
        # AWD_cases = \
        #     df.loc[df.is_alive & (df.age_exact_years < 5) & df.gi_diarrhoea_status &
        #            (df.gi_diarrhoea_acute_type == 'acute watery diarrhoea') & (df.gi_diarrhoea_type != 'persistent')]
        # dysentery_cases = \
        #     df.loc[df.is_alive & (df.age_exact_years < 5) & df.gi_diarrhoea_status &
        #            (df.gi_diarrhoea_acute_type == 'dysentery') & (df.gi_diarrhoea_type != 'persistent')]
        # persistent_diarr_cases = \
        #     df.loc[df.is_alive & (df.age_exact_years < 5) & df.gi_diarrhoea_status &
        #            (df.gi_diarrhoea_type == 'persistent')]
        #
        # clinical_types = pd.concat([AWD_cases, dysentery_cases, persistent_diarr_cases], axis=0).sort_index()
        #
        # logger.info('%s|clinical_diarrhoea_type|%s', self.sim.date,
        #             {  # 'total': len(clinical_types),
        #                 'AWD': len(AWD_cases),
        #                 'dysentery': len(dysentery_cases),
        #                 'persistent': len(persistent_diarr_cases)
        #             })
        #
        # # log information on attributable pathogens
        # pathogen_count = df[df.is_alive & df.age_years.between(0, 5)].groupby('gi_diarrhoea_pathogen').size()
        #
        # under5 = df[df.is_alive & df.age_years.between(0, 5)]
        # # all_patho_counts = sum(pathogen_count)
        # length_under5 = len(under5)
        # # total_inc = all_patho_counts * 4 * 100 / length_under5
        # rota_inc = (pathogen_count['rotavirus'] / length_under5) * 100 * 4
        # shigella_inc = (pathogen_count['shigella'] / length_under5) * 100 * 4
        # adeno_inc = (pathogen_count['adenovirus'] / length_under5) * 100 * 4
        # crypto_inc = (pathogen_count['cryptosporidium'] * 4 / length_under5) * 100 * 4
        # campylo_inc = (pathogen_count['campylobacter'] * 4 / length_under5) * 100 * 4
        # ETEC_inc = (pathogen_count['ST-ETEC'] / length_under5) * 100 * 4
        # sapo_inc = (pathogen_count['sapovirus'] / length_under5) * 100 * 4
        # noro_inc = (pathogen_count['norovirus'] / length_under5) * 100 * 4
        # astro_inc = (pathogen_count['astrovirus'] / length_under5) * 100 * 4
        # tEPEC_inc = (pathogen_count['tEPEC'] / length_under5) * 100 * 4
        #
        # # incidence rate by pathogen
        # logger.info('%s|diarr_incidence_by_patho|%s', self.sim.date,
        #             {  # 'total': total_inc,
        #                 'rotavirus': rota_inc,
        #                 'shigella': shigella_inc,
        #                 'adenovirus': adeno_inc,
        #                 'cryptosporidium': crypto_inc,
        #                 'campylobacter': campylo_inc,
        #                 'ETEC': ETEC_inc,
        #                 'sapovirus': sapo_inc,
        #                 'norovirus': noro_inc,
        #                 'astrovirus': astro_inc,
        #                 'tEPEC': tEPEC_inc
        #             })
        #
        # # incidence rate per age group by pathogen
        # pathogen_0to11mo = df[df.is_alive & (df.age_years < 1)].groupby('gi_diarrhoea_pathogen').size()
        # len_under12mo = df[df.is_alive & (df.age_years < 1)]
        # pathogen_12to23mo = df[df.is_alive & (df.age_years >= 1) & (df.age_years < 2)].groupby(
        #     'gi_diarrhoea_pathogen').size()
        # len_11to23mo = df[df.is_alive & (df.age_years >= 1) & (df.age_years < 2)]
        # pathogen_24to59mo = df[df.is_alive & (df.age_years >= 2) & (df.age_years < 5)].groupby(
        #     'gi_diarrhoea_pathogen').size()
        # len_24to59mo = df[df.is_alive & (df.age_years >= 2) & (df.age_years < 5)]
        #
        # rota_inc_by_age = [((pathogen_0to11mo['rotavirus'] * 4 * 100) / len(len_under12mo)),
        #                    ((pathogen_12to23mo['rotavirus'] * 4 * 100) / len(len_11to23mo)),
        #                    ((pathogen_24to59mo['rotavirus'] * 4 * 100) / len(len_24to59mo))]
        # shig_inc_by_age = [(pathogen_0to11mo['shigella'] * 4 * 100) / len(len_under12mo),
        #                    (pathogen_12to23mo['shigella'] * 4 * 100) / len(len_11to23mo),
        #                    (pathogen_24to59mo['shigella'] * 4 * 100) / len(len_24to59mo)]
        # adeno_inc_by_age = [(pathogen_0to11mo['adenovirus'] * 4 * 100) / len(len_under12mo),
        #                     (pathogen_12to23mo['adenovirus'] * 4 * 100) / len(len_11to23mo),
        #                     (pathogen_24to59mo['adenovirus'] * 4 * 100) / len(len_24to59mo)]
        # crypto_inc_by_age = [(pathogen_0to11mo['cryptosporidium'] * 4 * 100) / len(len_under12mo),
        #                      (pathogen_12to23mo['cryptosporidium'] * 4 * 100) / len(len_11to23mo),
        #                      (pathogen_24to59mo['cryptosporidium'] * 4 * 100) / len(len_24to59mo)]
        # campylo_inc_by_age = [(pathogen_0to11mo['campylobacter'] * 4 * 100) / len(len_under12mo),
        #                       (pathogen_12to23mo['campylobacter'] * 4 * 100) / len(len_11to23mo),
        #                       (pathogen_24to59mo['campylobacter'] * 4 * 100) / len(len_24to59mo)]
        # etec_inc_by_age = [(pathogen_0to11mo['ST-ETEC'] * 4 * 100) / len(len_under12mo),
        #                    (pathogen_12to23mo['ST-ETEC'] * 4 * 100) / len(len_11to23mo),
        #                    (pathogen_24to59mo['ST-ETEC'] * 4 * 100) / len(len_24to59mo)]
        # sapo_inc_by_age = [(pathogen_0to11mo['sapovirus'] * 4 * 100) / len(len_under12mo),
        #                    (pathogen_12to23mo['sapovirus'] * 4 * 100) / len(len_11to23mo),
        #                    (pathogen_24to59mo['sapovirus'] * 4 * 100) / len(len_24to59mo)]
        # noro_inc_by_age = [(pathogen_0to11mo['norovirus'] * 4 * 100) / len(len_under12mo),
        #                    (pathogen_12to23mo['norovirus'] * 4 * 100) / len(len_11to23mo),
        #                    (pathogen_24to59mo['norovirus'] * 4 * 100) / len(len_24to59mo)]
        # astro_inc_by_age = [(pathogen_0to11mo['astrovirus'] * 4 * 100) / len(len_under12mo),
        #                     (pathogen_12to23mo['astrovirus'] * 4 * 100) / len(len_11to23mo),
        #                     (pathogen_24to59mo['astrovirus'] * 4 * 100) / len(len_24to59mo)]
        # epec_inc_by_age = [(pathogen_0to11mo['tEPEC'] * 4 * 100) / len(len_under12mo),
        #                    (pathogen_12to23mo['tEPEC'] * 4 * 100) / len(len_11to23mo),
        #                    (pathogen_24to59mo['tEPEC'] * 4 * 100) / len(len_24to59mo)]
        #
        # logger.info('%s|diarr_incidence_age0_11|%s', self.sim.date,
        #             {'total': (sum(pathogen_0to11mo) * 4 * 100) / len_under12mo.size,
        #              'rotavirus': rota_inc_by_age[0],
        #              'shigella': shig_inc_by_age[0],
        #              'adenovirus': adeno_inc_by_age[0],
        #              'cryptosporidium': crypto_inc_by_age[0],
        #              'campylobacter': campylo_inc_by_age[0],
        #              'ETEC': etec_inc_by_age[0],
        #              'sapovirus': sapo_inc_by_age[0],
        #              'norovirus': noro_inc_by_age[0],
        #              'astrovirus': astro_inc_by_age[0],
        #              'tEPEC': epec_inc_by_age[0]
        #              })
        # logger.info('%s|diarr_incidence_age12_23|%s', self.sim.date,
        #             {'total': (sum(pathogen_0to11mo) * 4 * 100) / len_11to23mo.size,
        #              'rotavirus': rota_inc_by_age[1],
        #              'shigella': shig_inc_by_age[1],
        #              'adenovirus': adeno_inc_by_age[1],
        #              'cryptosporidium': crypto_inc_by_age[1],
        #              'campylobacter': campylo_inc_by_age[1],
        #              'ETEC': etec_inc_by_age[1],
        #              'sapovirus': sapo_inc_by_age[1],
        #              'norovirus': noro_inc_by_age[1],
        #              'astrovirus': astro_inc_by_age[1],
        #              'tEPEC': epec_inc_by_age[1]
        #              })
        # logger.info('%s|diarr_incidence_age24_59|%s', self.sim.date,
        #             {'total': (sum(pathogen_0to11mo) * 4 * 100) / pathogen_24to59mo.size,
        #              'rotavirus': rota_inc_by_age[2],
        #              'shigella': shig_inc_by_age[2],
        #              'adenovirus': adeno_inc_by_age[2],
        #              'cryptosporidium': crypto_inc_by_age[2],
        #              'campylobacter': campylo_inc_by_age[2],
        #              'ETEC': etec_inc_by_age[2],
        #              'sapovirus': sapo_inc_by_age[2],
        #              'norovirus': noro_inc_by_age[2],
        #              'astrovirus': astro_inc_by_age[2],
        #              'tEPEC': epec_inc_by_age[2]
        #              })
