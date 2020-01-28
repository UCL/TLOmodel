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
logger.setLevel(logging.DEBUG)


class Diarrhoea(Module):

    PARAMETERS = {
        #
        # 'eq_for_alloc_shigella': Parameter(Types.Eq, 'the e.... '),
        # 'eq_for_alloc_rota': Parameter(Types.Eq), 'the XX...')

        # @@ INES --- defining a parameter which is a dict that will hold the equations
        'incidence_equations_by_pathogen':
            Parameter(Types.DICT, 'dict that holds the equations governing the risk of incidence of each'
                                  ' type of pathogen'
                      ),
        'prob_symptoms':
            Parameter(Types.DICT, 'dict that holds the symptoms caused by each pathogen'
                      ),
        'progression_persistent_equation':
            Parameter(Types.REAL, 'dict that holds the equations governing the risk of progression'
                                  ' to persistent diarrhoea'
                      ),

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
            Parameter(Types.REAL, 'relative rate of acute diarrhoea becoming persistent diarrhoea for age 12 to 23 months'
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

        'daly_mild_diarrhoea':
            Parameter(Types.REAL, 'DALY weight for diarrhoea with no dehydration'
                      ),
        'daly_moderate_diarrhoea':
            Parameter(Types.REAL, 'DALY weight for diarrhoea with some dehydration'
                      ),
        'daly_severe_diarrhoea':
            Parameter(Types.REAL, 'DALY weight for diarrhoea with severe dehydration'
                      ),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.

    # TODO: why do some have gi_ and some have di_ as the prefix?
    # TODO: Take out the symptoms from here-- these can tracked in the symptom manager
    PROPERTIES = {
        'gi_diarrhoea_status': Property(Types.BOOL, 'symptomatic infection - diarrhoea disease'),
        'gi_diarrhoea_pathogen': Property(Types.CATEGORICAL, 'attributable pathogen for diarrhoea',
                                          categories=['rotavirus', 'shigella', 'adenovirus', 'cryptosporidium',
                                                      'campylobacter', 'ST-ETEC', 'sapovirus', 'norovirus',
                                                      'astrovirus', 'tEPEC', 'none']),
        'gi_diarrhoea_type': Property(Types.CATEGORICAL, 'progression of diarrhoea type',
                                      categories=['acute', 'prolonged', 'persistent']),
        'gi_diarrhoea_acute_type': Property(Types.CATEGORICAL, 'clinical acute diarrhoea type',
                                            categories=['dysentery', 'acute watery diarrhoea']),
        'gi_dehydration_status': Property(Types.CATEGORICAL, 'dehydration status',
                                          categories=['no dehydration', 'some dehydration', 'severe dehydration']),
        'gi_persistent_diarrhoea': Property(Types.CATEGORICAL,
                                            'diarrhoea episode longer than 14 days with or without dehydration',
                                            categories=['persistent diarrhoea', 'severe persistent diarrhoea']),
        'gi_diarrhoea_death': Property(Types.BOOL, 'death caused by diarrhoea'),
        'date_of_onset_diarrhoea': Property(Types.DATE, 'date of onset of diarrhoea'),
        'diarrhoea_ep_duration': Property(Types.REAL, 'duration of diarrhoea episode'),
        'gi_recovered_date': Property(Types.DATE, 'date of recovery from enteric infection'),
        'gi_diarrhoea_death_date': Property(Types.DATE, 'date of death from enteric infection'),
        'gi_diarrhoea_count': Property(Types.INT, 'annual counter for diarrhoea episodes'),
        'malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        # symptoms of diarrhoea for care seeking
        # 'di_diarrhoea_loose_watery_stools': Property(Types.BOOL, 'diarrhoea symptoms - loose or watery stools'),
        # 'di_blood_in_stools': Property(Types.BOOL, 'dysentery symptoms - blood in the stools'),
        # 'di_dehydration_present': Property(Types.BOOL, 'diarrhoea symptoms - dehydration'),
        # 'di_sympt_fever': Property(Types.BOOL, 'diarrhoea symptoms - associated fever'),
        # 'di_sympt_vomiting': Property(Types.BOOL, 'diarrhoea symptoms - associated vomoting'),
        # 'di_diarrhoea_over14days': Property(Types.BOOL, 'persistent diarrhoea - diarrhoea for 14 days or more'),

    }

    # TODO: as an example, I am declaring some symptoms here that we are going to use in the symptom manager
    # Declares symptoms
    SYMPTOMS = {'watery_diarrhoea', 'bloody_diarrhoea', 'fever', 'vomiting', 'dehydration', 'prolonged_diarrhoea'}

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module """

        p = self.parameters
        m = self
        dfd = pd.read_excel(
            Path(self.resourcefilepath) / 'ResourceFile_Childhood_Diarrhoea.xlsx', sheet_name='Parameter_values')
        # dfd.set_index("parameter_name", inplace=True)
        self.load_parameters_from_dataframe(dfd)

        # Register this disease module with the health system
        # self.sim.modules['HealthSystem'].register_disease_module(self)

        # DALY weights
        if 'HealthBurden' in self.sim.modules.keys():
            p['daly_mild_diarrhoea'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=32)
            p['daly_moderate_diarrhoea'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=35)
            p['daly_severe_diarrhoea'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=34)

        # --------------------------------------------------------------------------------------------
        # Make a dict to hold the equations that govern the probability that a person gets a pathogen

        # @@ INES ---PUTTIG A BLANK DICT INTO THE PARAMETERS FOR THIS MODULE
        self.parameters['incidence_equations_by_pathogen'] = dict()

        # @@ INES --- AND NOW ADDING THE EQUATIONS TO IT

        # @@ Ines -- not the usual formatting conventions here for using the Predictors, which is meant to help readability :-)

        # @@ Also note I how we specify things like 0<x<=1 --> .between(0,0)  for age-years
        # because age_years are whole numbers of years and .between(a,b) is true for a, b and all values inbetween

        self.parameters['incidence_equations_by_pathogen'].update({
            # Define the equation using LinearModel (note that this stage could be done in read_parms)
            'rotavirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('age_years')
                                     .when('.between(0,0)', m.base_incidence_diarrhoea_by_rotavirus[0])
                                     .when('.between(1,1)', m.base_incidence_diarrhoea_by_rotavirus[1])
                                     .when('.between(2,4)', m.base_incidence_diarrhoea_by_rotavirus[2])
                                     .otherwise(0.0),
                                     Predictor('li_no_access_handwashing')
                                     .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                     Predictor('li_no_clean_drinking_water').
                                     when(False, m.rr_gi_diarrhoea_clean_water),
                                     Predictor('li_unimproved_sanitation').
                                     when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                     # Predictor('hv_inf').
                                     # when(True, m.rr_gi_diarrhoea_HIV),
                                     Predictor('malnutrition').
                                     when(True, m.rr_gi_diarrhoea_SAM),
                                     Predictor('exclusive_breastfeeding').
                                     when(False, m.rr_gi_diarrhoea_excl_breast)
                                     )
        })

        self.parameters['incidence_equations_by_pathogen'].update({
            'shigella': LinearModel(LinearModelType.MULTIPLICATIVE,
                                    1.0,
                                    Predictor('age_years')
                                    .when('.between(0,1)', m.base_incidence_diarrhoea_by_shigella[0])
                                    .when('.between(1,2)', m.base_incidence_diarrhoea_by_shigella[1])
                                    .when('.between(2,5)', m.base_incidence_diarrhoea_by_shigella[2])
                                    .otherwise(0.0),
                                    Predictor('li_no_access_handwashing')
                                    .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                    Predictor('li_no_clean_drinking_water').
                                    when(False, m.rr_gi_diarrhoea_clean_water),
                                    Predictor('li_unimproved_sanitation').
                                    when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                    # Predictor('hv_inf').
                                    # when(True, m.rr_gi_diarrhoea_HIV),
                                    Predictor('malnutrition').
                                    when(True, m.rr_gi_diarrhoea_SAM),
                                    Predictor('exclusive_breastfeeding').
                                    when(False, m.rr_gi_diarrhoea_excl_breast)
                                    )
        })

        self.parameters['incidence_equations_by_pathogen'].update({
            'adenovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                      1.0,
                                      Predictor('age_years')
                                      .when('.between(0,1)', m.base_incidence_diarrhoea_by_adenovirus[0])
                                      .when('.between(1,2)', m.base_incidence_diarrhoea_by_adenovirus[1])
                                      .when('.between(2,5)', m.base_incidence_diarrhoea_by_adenovirus[2])
                                      .otherwise(0.0),
                                      Predictor('li_no_access_handwashing')
                                      .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                      Predictor('li_no_clean_drinking_water').
                                      when(False, m.rr_gi_diarrhoea_clean_water),
                                      Predictor('li_unimproved_sanitation').
                                      when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                      # Predictor('hv_inf').
                                      # when(True, m.rr_gi_diarrhoea_HIV),
                                      Predictor('malnutrition').
                                      when(True, m.rr_gi_diarrhoea_SAM),
                                      Predictor('exclusive_breastfeeding').
                                      when(False, m.rr_gi_diarrhoea_excl_breast)
                                      )
        })

        self.parameters['incidence_equations_by_pathogen'].update({
            'cryptosporidium': LinearModel(LinearModelType.MULTIPLICATIVE,
                                           1.0,
                                           Predictor('age_years')
                                           .when('.between(0,1)', m.base_incidence_diarrhoea_by_crypto[0])
                                           .when('.between(1,2)', m.base_incidence_diarrhoea_by_crypto[1])
                                           .when('.between(2,5)', m.base_incidence_diarrhoea_by_crypto[2])
                                           .otherwise(0.0),
                                           Predictor('li_no_access_handwashing')
                                           .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                           Predictor('li_no_clean_drinking_water').
                                           when(False, m.rr_gi_diarrhoea_clean_water),
                                           Predictor('li_unimproved_sanitation').
                                           when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                           # Predictor('hv_inf').
                                           # when(True, m.rr_gi_diarrhoea_HIV),
                                           Predictor('malnutrition').
                                           when(True, m.rr_gi_diarrhoea_SAM),
                                           Predictor('exclusive_breastfeeding').
                                           when(False, m.rr_gi_diarrhoea_excl_breast)
                                           )
        })

        self.parameters['incidence_equations_by_pathogen'].update({
            'campylobacter': LinearModel(LinearModelType.MULTIPLICATIVE,
                                         1.0,
                                         Predictor('age_years')
                                         .when('.between(0,0)', m.base_incidence_diarrhoea_by_campylo[0])
                                         .when('.between(1,1)', m.base_incidence_diarrhoea_by_campylo[1])
                                         .when('.between(2,4)', m.base_incidence_diarrhoea_by_campylo[2])
                                         .otherwise(0.0),
                                         Predictor('li_no_access_handwashing')
                                         .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                         Predictor('li_no_clean_drinking_water').
                                         when(False, m.rr_gi_diarrhoea_clean_water),
                                         Predictor('li_unimproved_sanitation').
                                         when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                         # Predictor('hv_inf').
                                         # when(True, m.rr_gi_diarrhoea_HIV),
                                         Predictor('malnutrition').
                                         when(True, m.rr_gi_diarrhoea_SAM),
                                         Predictor('exclusive_breastfeeding').
                                         when(False, m.rr_gi_diarrhoea_excl_breast)
                                         )
        })

        self.parameters['incidence_equations_by_pathogen'].update({
            'ST-ETEC': LinearModel(LinearModelType.MULTIPLICATIVE,
                                   1.0,
                                   Predictor('age_years')
                                   .when('.between(0,0)', m.base_incidence_diarrhoea_by_ETEC[0])
                                   .when('.between(1,1)', m.base_incidence_diarrhoea_by_ETEC[1])
                                   .when('.between(2,4)', m.base_incidence_diarrhoea_by_ETEC[2])
                                   .otherwise(0.0),
                                   Predictor('li_no_access_handwashing')
                                   .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                   Predictor('li_no_clean_drinking_water').
                                   when(False, m.rr_gi_diarrhoea_clean_water),
                                   Predictor('li_unimproved_sanitation').
                                   when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                   # Predictor('hv_inf').
                                   # when(True, m.rr_gi_diarrhoea_HIV),
                                   Predictor('malnutrition').
                                   when(True, m.rr_gi_diarrhoea_SAM),
                                   Predictor('exclusive_breastfeeding').
                                   when(False, m.rr_gi_diarrhoea_excl_breast)
                                   )
        })

        self.parameters['incidence_equations_by_pathogen'].update({
            'sapovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('age_years')
                                     .when('.between(0,0)', m.base_incidence_diarrhoea_by_sapovirus[0])
                                     .when('.between(1,1)', m.base_incidence_diarrhoea_by_sapovirus[1])
                                     .when('.between(2,4)', m.base_incidence_diarrhoea_by_sapovirus[2])
                                     .otherwise(0.0),
                                     Predictor('li_no_access_handwashing')
                                     .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                     Predictor('li_no_clean_drinking_water').
                                     when(False, m.rr_gi_diarrhoea_clean_water),
                                     Predictor('li_unimproved_sanitation').
                                     when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                     # Predictor('hv_inf').
                                     # when(True, m.rr_gi_diarrhoea_HIV),
                                     Predictor('malnutrition').
                                     when(True, m.rr_gi_diarrhoea_SAM),
                                     Predictor('exclusive_breastfeeding').
                                     when(False, m.rr_gi_diarrhoea_excl_breast)
                                     )
        })

        self.parameters['incidence_equations_by_pathogen'].update({
            'norovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('age_years')
                                     .when('.between(0,0)', m.base_incidence_diarrhoea_by_norovirus[0])
                                     .when('.between(1,1)', m.base_incidence_diarrhoea_by_norovirus[1])
                                     .when('.between(2,4)', m.base_incidence_diarrhoea_by_norovirus[2])
                                     .otherwise(0.0),
                                     Predictor('li_no_access_handwashing')
                                     .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                     Predictor('li_no_clean_drinking_water').
                                     when(False, m.rr_gi_diarrhoea_clean_water),
                                     Predictor('li_unimproved_sanitation').
                                     when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                     # Predictor('hv_inf').
                                     # when(True, m.rr_gi_diarrhoea_HIV),
                                     Predictor('malnutrition').
                                     when(True, m.rr_gi_diarrhoea_SAM),
                                     Predictor('exclusive_breastfeeding').
                                     when(False, m.rr_gi_diarrhoea_excl_breast)
                                     )
        })

        self.parameters['incidence_equations_by_pathogen'].update({
            'astrovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                      1.0,
                                      Predictor('age_years')
                                      .when('.between(0,0)', m.base_incidence_diarrhoea_by_astrovirus[0])
                                      .when('.between(1,1)', m.base_incidence_diarrhoea_by_astrovirus[1])
                                      .when('.between(2,4)', m.base_incidence_diarrhoea_by_astrovirus[2])
                                      .otherwise(0.0),
                                      Predictor('li_no_access_handwashing')
                                      .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                      Predictor('li_no_clean_drinking_water').
                                      when(False, m.rr_gi_diarrhoea_clean_water),
                                      Predictor('li_unimproved_sanitation').
                                      when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                      # Predictor('hv_inf').
                                      # when(True, m.rr_gi_diarrhoea_HIV),
                                      Predictor('malnutrition').
                                      when(True, m.rr_gi_diarrhoea_SAM),
                                      Predictor('exclusive_breastfeeding').
                                      when(False, m.rr_gi_diarrhoea_excl_breast)
                                      )
        })

        self.parameters['incidence_equations_by_pathogen'].update({
            'tEPEC': LinearModel(LinearModelType.MULTIPLICATIVE,
                                 1.0,
                                 Predictor('age_years')
                                 .when('.between(0,0)', m.base_incidence_diarrhoea_by_EPEC[0])
                                 .when('.between(1,1)', m.base_incidence_diarrhoea_by_EPEC[1])
                                 .when('.between(2,4)', m.base_incidence_diarrhoea_by_EPEC[2])
                                 .otherwise(0.0),
                                 Predictor('li_no_access_handwashing')
                                 .when(False, m.rr_gi_diarrhoea_HHhandwashing),
                                 Predictor('li_no_clean_drinking_water').
                                 when(False, m.rr_gi_diarrhoea_clean_water),
                                 Predictor('li_unimproved_sanitation').
                                 when(False, m.rr_gi_diarrhoea_improved_sanitation),
                                 # Predictor('hv_inf').
                                 # when(True, m.rr_gi_diarrhoea_HIV),
                                 Predictor('malnutrition').
                                 when(True, m.rr_gi_diarrhoea_SAM),
                                 Predictor('exclusive_breastfeeding').
                                 when(False, m.rr_gi_diarrhoea_excl_breast)
                                 )
        })

        # @@ INES ___ EVERY SYPTOM HERE MUST BE DECLARED ABOVE IN SYMPTOMS = {}
        # Please double check that all the symptos are declared.

        self.parameters['prob_symptoms'] = {
            'rotavirus': {'watery_diarrhoea': m.proportion_AWD_by_rotavirus,
                          'bloody_diarrhoea': 1 - m.proportion_AWD_by_rotavirus,
                          'fever': m.fever_by_rotavirus, 'vomiting': m.vomiting_by_rotavirus,
                          'dehydration': m.dehydration_by_rotavirus,
                          'prolonged_diarrhoea': m.prolonged_diarr_rotavirus},
            'shigella': {'watery_diarrhoea': m.proportion_AWD_by_shigella,
                         'bloody_diarrhoea': 1 - m.proportion_AWD_by_shigella,
                         'fever': m.fever_by_shigella, 'vomiting': m.vomiting_by_shigella,
                         'dehydration': m.dehydration_by_shigella,
                         'prolonged_diarrhoea': m.prolonged_diarr_shigella},
            'adenovirus': {'watery_diarrhoea': m.proportion_AWD_by_adenovirus,
                           'bloody_diarrhoea': 1 - m.proportion_AWD_by_adenovirus,
                           'fever': m.fever_by_adenovirus, 'vomiting': m.vomiting_by_adenovirus,
                           'dehydration': m.dehydration_by_adenovirus,
                           'prolonged_diarrhoea': m.prolonged_diarr_adenovirus},
            'cryptosporidium': {'watery_diarrhoea': m.proportion_AWD_by_crypto,
                                'bloody_diarrhoea': 1 - m.proportion_AWD_by_crypto,
                                'fever': m.fever_by_crypto, 'vomiting': m.vomiting_by_crypto,
                                'dehydration': m.dehydration_by_crypto,
                                'prolonged_diarrhoea': m.prolonged_diarr_crypto},
            'campylobacter': {'watery_diarrhoea': m.proportion_AWD_by_campylo,
                              'bloody_diarrhoea': 1 - m.proportion_AWD_by_campylo,
                              'fever': m.fever_by_campylo, 'vomiting': m.vomiting_by_campylo,
                              'dehydration': m.dehydration_by_rotavirus,
                              'prolonged_diarrhoea': m.prolonged_diarr_campylo},
            'ST-ETEC': {'watery_diarrhoea': m.proportion_AWD_by_ETEC,
                        'bloody_diarrhoea': 1 - m.proportion_AWD_by_ETEC,
                        'fever': m.fever_by_rotavirus, 'vomiting': m.vomiting_by_ETEC,
                        'dehydration': m.dehydration_by_ETEC, 'prolonged_diarrhoea': m.prolonged_diarr_ETEC},
            'sapovirus': {'watery_diarrhoea': m.proportion_AWD_by_sapovirus,
                          'bloody_diarrhoea': 1 - m.proportion_AWD_by_sapovirus,
                          'fever': m.fever_by_rotavirus, 'vomiting': m.vomiting_by_sapovirus,
                          'dehydration': m.dehydration_by_sapovirus,
                          'prolonged_diarrhoea': m.prolonged_diarr_sapovirus},
            'norovirus': {'watery_diarrhoea': m.proportion_AWD_by_norovirus,
                          'bloody_diarrhoea': 1 - m.proportion_AWD_by_norovirus,
                          'fever': m.fever_by_rotavirus, 'vomiting': m.vomiting_by_norovirus,
                          'dehydration': m.dehydration_by_norovirus,
                          'prolonged_diarrhoea': m.prolonged_diarr_norovirus},
            'astrovirus': {'watery_diarrhoea': m.proportion_AWD_by_astrovirus,
                           'bloody_diarrhoea': 1 - m.proportion_AWD_by_astrovirus,
                           'fever': m.fever_by_rotavirus, 'vomiting': m.vomiting_by_astrovirus,
                           'dehydration': m.dehydration_by_astrovirus,
                           'prolonged_diarrhoea': m.prolonged_diarr_astrovirus},
            'tEPEC': {'watery_diarrhoea': m.proportion_AWD_by_EPEC,
                      'bloody_diarrhoea': 1 - m.proportion_AWD_by_EPEC,
                      'fever': m.fever_by_rotavirus, 'vomiting': m.vomiting_by_EPEC,
                      'dehydration': m.dehydration_by_EPEC, 'prolonged_diarrhoea': m.prolonged_diarr_EPEC},
        }

        # # # # # # ASSIGN THE PROBABILITY OF BECOMING PERSISTENT (over 14 days) # # # # # #
        # THIS EQUATION SHOULD ONLY WORK FOR THOSE WHO ARE IN THE PROLONGED DIARRHOEA PHASE
        self.parameters['progression_persistent_equation'] = \
            LinearModel(LinearModelType.MULTIPLICATIVE,
                        0.2,
                        Predictor('age_years')
                        .when('.between(0,1)', 1)
                        .when('.between(1,2)', m.rr_bec_persistent_age12to23)
                        .when('.between(2,5)', m.rr_bec_persistent_age24to59)
                        .otherwise(0.0),
                        # Predictor('hv_inf')
                        # .when(False, m.rr_bec_persistent_HIV),
                        Predictor('malnutrition').
                        when(False, m.rr_bec_persistent_SAM),
                        Predictor('exclusive_breastfeeding').
                        when(False, m.rr_bec_persistent_excl_breast)
                        )

        # # # # # # # # # # # # ASSIGN THE RATE OF DEATH # # # # # # # # # # # #
        self.parameters['rate_death_diarrhoea'] = \
            LinearModel(LinearModelType.MULTIPLICATIVE,
                        1.0,
                        Predictor('gi_diarrhoea_acute_type')
                        .when('acute watery diarrhoea', 0.0056)
                        .when('dysentery', 0.0427),
                        Predictor('gi_diarrhoea_type').when('persistent', 0.1395),
                        Predictor('age_years')
                        # .when('.between(1,2)', m.rr_diarr_death_age12to23mo)
                        # .when('.between(2,4)', m.rr_diarr_death_age24to59mo)
                        .otherwise(0.0),
                        # Predictor('hv_inf').
                        # when(True, m.rr_gi_diarrhoea_HIV),
                        Predictor('malnutrition').
                        when(True, m.rr_gi_diarrhoea_SAM)
                        )

    def initialise_population(self, population):
        """Set our property values for the initial population.
        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        rng = self.rng

        # DEFAULTS
        df['gi_diarrhoea_status'] = False
        df['gi_diarrhoea_acute_type'] = ''          #@@@@ initliasing as np.nan makes it a float, so then cannot become a string or category
        df['gi_diarrhoea_pathogen'].values[:] = 'none'
        df['gi_diarrhoea_type'] = ''     ## You can't make these nans as they are cateogrical. For now I am putting in str so we can use.
        df['gi_persistent_diarrhoea'] = ''      # same here
        df['gi_dehydration_status'] = 'no dehydration'
        df['date_of_onset_diarrhoea'] = pd.NaT
        df['gi_recovered_date'] = pd.NaT
        df['gi_diarrhoea_death_date'] = pd.NaT
        df['diarrhoea_ep_duration'] = np.nan
        df['gi_diarrhoea_count'] = 0
        df['gi_diarrhoea_death'] = False
        df['malnutrition'] = False
        df['exclusive_breastfeeding'] = False
        df['continued_breastfeeding'] = False

        # TODO: looks like some properties not defined here? - all properties initialised

        # TODO: Assuming zero prevalence initially - fine

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event for acute diarrhoea ---------------------------------------------------
        sim.schedule_event(AcuteDiarrhoeaEvent(self), sim.date + DateOffset(months=0))

        # add an event to log to screen
        sim.schedule_event(DiarrhoeaLoggingEvent(self), sim.date + DateOffset(months=1))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props

        df.at[child_id, 'gi_recovered_date'] = pd.NaT
        df.at[child_id, 'gi_diarrhoea_status'] = False
        df.at[child_id, 'gi_diarrhoea_acute_type'] = np.nan
        df.at[child_id, 'gi_diarrhoea_type'] = ''
        df.at[child_id, 'gi_persistent_diarrhoea'] = np.nan
        df.at[child_id, 'gi_dehydration_status'] = 'no dehydration'
        df.at[child_id, 'date_of_onset_diarrhoea'] = pd.NaT
        df.at[child_id, 'gi_recovered_date'] = pd.NaT
        df.at[child_id, 'gi_diarrhoea_death_date'] = pd.NaT
        df.at[child_id, 'gi_diarrhoea_death'] = False
        df.at[child_id, 'gi_diarrhoea_pathogen'] = 'none'
        df.at[child_id, 'gi_diarrhoea_count'] = 0

        # todo; make sure all properties intiialised for the child

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Diarrhoea, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)
        pass

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is diarrhoea reporting my health values')

        df = self.sim.population.props
        p = self.parameters

        health_values = df.loc[df.is_alive, 'gi_dehydration_status'].map({
            'none': 0,
            'no dehydration': p['daly_mild_diarrhoea'],     # TODO; maybe rename and checkdaly_mild_dehydration_due_to_diarrahea
            'some dehydration': p['daly_moderate_diarrhoea'],
            'severe dehydration': p['daly_severe_diarrhoea']
        })
        health_values.name = 'dehydration'    # label the cause of this disability

        #TODO: is it right that the only thing causing lays from diarrhoa is the dehydration
        #TODO: are these dalys for the episode of diarrhoa of for an amount of time?
        #TODO; nb that this will change when symtoms tracked in SymptomManager

        return health_values.loc[df.is_alive]   # returns the series


class AcuteDiarrhoeaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        # TODO: Say what this event is for and what it will do.
        """Apply this event to the population.
        :param population: the current population
        """

        df = population.props
        m = self.module
        rng = m.rng
        now = self.sim.date

        # and now, this is what goes here in the apply() of the event --->>
        # Compute probabilities and organise in a dataframe
        probs = pd.DataFrame()
        for k, v in m.parameters['incidence_equations_by_pathogen'].items():
            probs[k] = v.predict(df.loc[df.is_alive & df.age_years < 5])

        # Declare that pathogens are mutally exclusive and do the random choice for each person
        probs['none'] = 1 - probs.sum(axis=1)
        outcome = pd.Series(data='none', index=probs.index)

        for i in outcome.index:
            # the outcome - diarrhoea (and which pathogen...) to put in the df
            outcome_i = rng.choice(probs.columns, p=probs.loc[i].values)

            if outcome_i != 'none':
                assert outcome_i in self.module.PROPERTIES['gi_diarrhoea_pathogen'].categories, 'pathogen is being assigned that is not declared in properties'
                df.at[i, 'gi_diarrhoea_pathogen'] = outcome_i
                df.at[i, 'gi_diarrhoea_status'] = True
                df.at[i, 'gi_diarrhoea_type'] = 'acute'
                df.at[i, 'gi_diarrhoea_count'] += 1
                # @@@ INES --- is there a need for an else statement to set these variables to values consistent with
                # no new diarrahea episode? --- @@ TIM: I dont think so, there will be default property values

                # ----------------------- ALLOCATE A RANDOM DATE OF ONSET OF ACUTE DIARRHOEA ----------------------
                df.at[i, 'date_of_onset_diarrhoea'] = self.sim.date + DateOffset(days=np.random.randint(0, 90))
                # does this need to be stored in the df?
                # @@ TIM - not necessarily, but I want to scatter the occurrence of diarrhoea episodes like in real life??

                # ----------------------------------------------------------------------------------------
                # Then work out the symptoms for this person:
                for symptom_string, prob in m.parameters['prob_symptoms'][outcome_i].items():
                    if symptom_string == 'watery_diarrhoea':
                        if rng.rand() < prob:
                            df.at[i, 'gi_diarrhoea_acute_type'] = 'acute watery diarrhoea'
                        else:
                            df.at[i, 'gi_diarrhoea_acute_type'] = 'dysentery'
                    # determine the dehydration status
                    if symptom_string == 'dehydration':
                        if rng.rand() < prob:
                            df.at[i, 'di_dehydration_present'] = True
                            if rng.rand() < 0.7:
                                df.at[i, 'gi_dehydration_status'] = 'some dehydration'
                            else:
                                df.at[i, 'gi_dehydration_status'] = 'severe dehydration'
                        else:
                            df.at[i, 'di_dehydration_present'] = False

                    # determine the which phase in diarrhoea episode
                    if symptom_string == 'prolonged_diarrhoea':
                        if rng.rand() < prob:
                            df.at[i, 'gi_diarrhoea_type'] = 'prolonged'
                            if rng.rand() < m.parameters['progression_persistent_equation'].predict(df.loc[[i]]).values[0]:
                                df.at[i, 'gi_diarrhoea_type'] = 'persistent'
                            if df.at[i, 'di_dehydration_present']:
                                df.at[i, 'gi_persistent_diarrhoea'] = 'severe persistent diarrhoea'
                            else:
                                df.at[i, 'gi_persistent_diarrhoea'] = 'persistent diarrhoea'

                    # determine the duration of the episode
                    if df.at[i, 'gi_diarrhoea_type'] == 'acute':
                        duration = rng.randint(3, 7)
                    if df.at[i, 'gi_diarrhoea_type'] == 'prolonged':
                        duration = rng.randint(7, 14)
                    if df.at[i, 'gi_diarrhoea_type'] == 'persistent':
                        duration = rng.randint(14, 21)

                    # Send the symptoms to the SymptomManager
                    self.sim.modules['SymptomManager'].change_symptom(symptom_string=symptom_string,
                                                                      person_id=i,
                                                                      add_or_remove='+',
                                                                      disease_module=self.module,
                                                                      date_of_onset=df.at[i, 'date_of_onset_diarrhoea'],
                                                                      duration_in_days=duration
                                                                      )

                    # # # # # # HEALTH CARE SEEKING BEHAVIOUR - INTERACTION WITH HSB MODULE # # # # #
                    # # TODO: when you declare the symptoms in the symptom manager, the health care seeking will follow automatically

                # # # # # # # # # # # SCHEDULE DEATH OR RECOVERY # # # # # # # # # # #
                if rng.rand() < self.module.parameters['rate_death_diarrhoea'].predict(df.iloc[[i]]).values[0]:
                    if df.at[i, 'gi_diarrhoea_type'] == 'acute':
                        random_date = rng.randint(low=4, high=6)
                        random_days = pd.to_timedelta(random_date, unit='d')
                        death_event = DeathDiarrhoeaEvent(self.module, person_id=i, cause='diarrhoea')
                        self.sim.schedule_event(death_event, df.at[i, 'date_of_onset_diarrhoea'] + random_days)
                    if df.at[i, 'gi_diarrhoea_type'] == 'prolonged':
                        random_date1 = rng.randint(low=7, high=13)
                        random_days1 = pd.to_timedelta(random_date1, unit='d')
                        death_event = DeathDiarrhoeaEvent(self.module, person_id=i,
                                                          cause='diarrhoea')
                        self.sim.schedule_event(death_event, df.at[i, 'date_of_onset_diarrhoea'] + random_days1)
                    if df.at[i, 'gi_diarrhoea_type'] == 'persistent':
                        random_date2 = rng.randint(low=14, high=30)
                        random_days2 = pd.to_timedelta(random_date2, unit='d')
                        death_event = DeathDiarrhoeaEvent(self.module, person_id=i,
                                                          cause='diarrhoea')
                        self.sim.schedule_event(death_event, df.at[i, 'date_of_onset_diarrhoea'] + random_days2)
                else:
                    if df.at[i, 'gi_diarrhoea_type'] == 'acute':
                        random_date = rng.randint(low=4, high=6)
                        random_days = pd.to_timedelta(random_date, unit='d')
                        self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=i),
                                                df.at[i, 'date_of_onset_diarrhoea'] + random_days)
                    if df.at[i, 'gi_diarrhoea_type'] == 'prolonged':
                        random_date1 = rng.randint(low=7, high=13)
                        random_days1 = pd.to_timedelta(random_date1, unit='d')
                        self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=i),
                                                df.at[i, 'date_of_onset_diarrhoea'] + random_days1)
                    if df.at[i, 'gi_diarrhoea_type'] == 'persistent':
                        random_date2 = rng.randint(low=14, high=21)
                        random_days2 = pd.to_timedelta(random_date2, unit='d')
                        self.sim.schedule_event(SelfRecoverEvent(self.module, person_id=i),
                                                df.at[i, 'date_of_onset_diarrhoea'] + random_days2)


class SelfRecoverEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        # set everything back to default
        df.at[person_id, 'gi_recovered_date'] = pd.NaT
        df.at[person_id, 'gi_diarrhoea_status'] = False
        df.at[person_id, 'gi_diarrhoea_acute_type'] = np.nan
        df.at[person_id, 'gi_diarrhoea_type'] = np.nan
        df.at[person_id, 'gi_persistent_diarrhoea'] = np.nan
        df.at[person_id, 'gi_dehydration_status'] = 'no dehydration'
        df.at[person_id, 'date_of_onset_diarrhoea'] = pd.NaT
        df.at[person_id, 'gi_recovered_date'] = pd.NaT
        df.at[person_id, 'gi_diarrhoea_death_date'] = pd.NaT
        df.at[person_id, 'gi_diarrhoea_death'] = False
        df.at[person_id, 'gi_diarrhoea_pathogen'] = 'none'


class DeathDiarrhoeaEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id, cause):
        super().__init__(module, person_id=person_id)
        self.cause = cause

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        if df.at[person_id, 'is_alive']:

            # check if person should still die of diarah
            if df.at[person_id, 'gi_will_die_of_diarh']:

                self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id, cause='diarrhoea'),
                                        self.sim.date)
                df.at[person_id, 'gi_diarrhoea_death_date'] = self.sim.date
                df.at[person_id, 'gi_diarrhoea_death'] = True
                # logger.info('This is DeathDiarrhoeaEvent determining if person %d on the date %s will die '
                #             'from their disease', person_id, self.sim.date)
                # death_count = sum(person_id)
                # # Log the diarrhoea death information
                # logger.info('%s|death_diarrhoea|%s', self.sim.date,
                #             {'death': sum(death_count)
                #              })


class DiarrhoeaLoggingEvent (RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        # -----------------------------------------------------------------------------------------------------
        # sum all the counters for previous year
        count_episodes = df['gi_diarrhoea_count'].sum()
        pop_under5 = len(df[df.is_alive & (df.age_exact_years < 5)])
        # overall incidence rate in under 5
        inc_100cy = (count_episodes / pop_under5) * 100
        # logger.info('%s|episodes_counts|%s', now, {'incidence_per100cy': inc_100cy})
        # logger.info('%s|pop_counts|%s', now, {'pop_len': pop_under5})

        # TODO: I think it's easier to output the number of events in the logger and work out the incidence afterwards.
        # So, I would propose just logging value counts.

        # log the clinical cases
        AWD_cases = \
            df.loc[df.is_alive & (df.age_exact_years < 5) & df.gi_diarrhoea_status &
                   (df.gi_diarrhoea_acute_type == 'acute watery diarrhoea') & (df.gi_diarrhoea_type != 'persistent')]
        dysentery_cases = \
            df.loc[df.is_alive & (df.age_exact_years < 5) & df.gi_diarrhoea_status &
                   (df.gi_diarrhoea_acute_type == 'dysentery') & (df.gi_diarrhoea_type != 'persistent')]
        persistent_diarr_cases = \
            df.loc[df.is_alive & (df.age_exact_years < 5) & df.gi_diarrhoea_status &
                   (df.gi_diarrhoea_type == 'persistent')]

        # clinical_types = pd.concat([AWD_cases, dysentery_cases, persistent_diarr_cases], axis=0).sort_index()

        logger.info('%s|clinical_diarrhoea_type|%s', self.sim.date,
                    {  # 'total': len(clinical_types),
                        'AWD': len(AWD_cases),
                        'dysentery': len(dysentery_cases),
                        'persistent': len(persistent_diarr_cases)
                    })

        # log information on attributable pathogens
        pathogen_count = df[df.is_alive & df.age_years.between(0, 5)].groupby('gi_diarrhoea_pathogen').size()
        #
        # under5 = df[df.is_alive & df.age_years.between(0, 5)]
        # # all_patho_counts = sum(pathogen_count)
        # length_under5 = len(under5)
        # # total_inc = all_patho_counts * 4 * 100 / length_under5
        # rota_inc = (pathogen_count['rotavirus'] * 4 / length_under5) * 100
        # shigella_inc = (pathogen_count['shigella'] * 4 / length_under5) * 100
        # adeno_inc = (pathogen_count['adenovirus'] * 4 / length_under5) * 100
        # crypto_inc = (pathogen_count['cryptosporidium'] * 4 / length_under5) * 100
        # campylo_inc = (pathogen_count['campylobacter'] * 4 / length_under5) * 100
        # ETEC_inc = (pathogen_count['ST-ETEC'] * 4 / length_under5) * 100
        # sapo_inc = (pathogen_count['sapovirus'] * 4 / length_under5) * 100
        # noro_inc = (pathogen_count['norovirus'] * 4 / length_under5) * 100
        # astro_inc = (pathogen_count['astrovirus'] * 4 / length_under5) * 100
        # tEPEC_inc = (pathogen_count['tEPEC'] * 4 / length_under5) * 100
        #
        # # incidence rate by pathogen
        # logger.info('%s|diarr_incidence_by_patho|%s', self.sim.date,
        #             {#'total': total_inc,
        #              'rotavirus': rota_inc,
        #              'shigella': shigella_inc,
        #              'adenovirus': adeno_inc,
        #              'cryptosporidium': crypto_inc,
        #              'campylobacter': campylo_inc,
        #              'ETEC': ETEC_inc,
        #              'sapovirus': sapo_inc,
        #              'norovirus': noro_inc,
        #              'astrovirus': astro_inc,
        #              'tEPEC': tEPEC_inc
        #              })
        #
        # # incidence rate per age group by pathogen
        # pathogen_0to11mo = df[df.is_alive & (df.age_years < 1)].groupby('gi_diarrhoea_pathogen').size()
        # len_under12mo = df[df.is_alive & df.age_years.between(0, 1)]
        # pathogen_12to23mo = df[df.is_alive & (df.age_years >= 1) & (df.age_years < 2)].groupby(
        #     'gi_diarrhoea_pathogen').size()
        # len_11to23mo = df[df.is_alive & df.age_years.between(1, 2)]
        # pathogen_24to59mo = df[df.is_alive & (df.age_years >= 2) & (df.age_years < 5)].groupby(
        #     'gi_diarrhoea_pathogen').size()
        # len_24to59mo = df[df.is_alive & df.age_years.between(2, 5)]
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
        #
