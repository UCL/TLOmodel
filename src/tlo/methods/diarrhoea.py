"""
Childhood Diarrhoea Module
Documentation: '04 - Methods Repository/Childhood Disease Methods.docx'
Reading parameters from TLOmodel/resources/ResourceFile_Diarrhoea.xlsx
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

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
        'tEPEC'
    }

    PARAMETERS = {
        'base_inc_rate_diarrhoea_by_rotavirus':
            Parameter(Types.LIST,
                      'incidence rate (per person-year)'
                      ' of diarrhoea caused by rotavirus in age groups 0-11, 12-23, 24-59 months '
                      ),
        'base_inc_rate_diarrhoea_by_shigella':
            Parameter(Types.LIST,
                      'incidence rate (per person-year) '
                      'of diarrhoea caused by shigella spp in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_diarrhoea_by_adenovirus':
            Parameter(Types.LIST,
                      'incidence rate (per person-year) '
                      'of diarrhoea caused by adenovirus 40/41 in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_diarrhoea_by_cryptosporidium':
            Parameter(Types.LIST,
                      'incidence rate (per person-year) '
                      'of diarrhoea caused by cryptosporidium in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_diarrhoea_by_campylobacter':
            Parameter(Types.LIST,
                      'incidence rate (per person-year) '
                      'of diarrhoea caused by campylobacter spp in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_diarrhoea_by_ST-ETEC':
            Parameter(Types.LIST,
                      'incidence rate (per person-year) '
                      'of diarrhoea caused by ST-ETEC in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_diarrhoea_by_sapovirus':
            Parameter(Types.LIST,
                      'incidence rate (per person-year) '
                      'of diarrhoea caused by sapovirus in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_diarrhoea_by_norovirus':
            Parameter(Types.LIST,
                      'incidence rate (per person-year) '
                      'of diarrhoea caused by norovirus in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_diarrhoea_by_astrovirus':
            Parameter(Types.LIST,
                      'incidence rate (per person-year) '
                      'of diarrhoea caused by astrovirus in age groups 0-11, 12-23, 24-59 months'
                      ),
        'base_inc_rate_diarrhoea_by_tEPEC':
            Parameter(Types.LIST,
                      'incidence rate (per person-year) '
                      'of diarrhoea caused by tEPEC in age groups 0-11, 12-23, 24-59 months'
                      ),
        'rr_diarrhoea_HHhandwashing':
            Parameter(Types.REAL, 'relative rate of diarrhoea with household handwashing with soap'
                      ),
        'rr_diarrhoea_improved_sanitation':
            Parameter(Types.REAL, 'relative rate of diarrhoea for improved sanitation'
                      ),
        'rr_diarrhoea_clean_water':
            Parameter(Types.REAL, 'relative rate of diarrhoea for access to clean drinking water'
                      ),
        'rr_diarrhoea_HIV':
            Parameter(Types.REAL, 'relative rate of diarrhoea for HIV positive status'
                      ),
        'rr_diarrhoea_SAM':
            Parameter(Types.REAL, 'relative rate of diarrhoea for severe malnutrition'
                      ),
        'rr_diarrhoea_excl_breast':
            Parameter(Types.REAL, 'relative rate of diarrhoea for exclusive breastfeeding upto 6 months'
                      ),
        'rr_diarrhoea_cont_breast':
            Parameter(Types.REAL, 'relative rate of diarrhoea for continued breastfeeding 6 months to 2 years'
                      ),
        'rr_diarrhoea_rotavirus_vaccination':
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
                      ),
        'days_onset_severe_dehydration_before_death':
            Parameter(Types.INT, 'number of days before a death (in the untreated case) that dehydration would be '
                                 'classified as severe and child ought to be classified as positive for the danger'
                                 'signs')
    }

    PROPERTIES = {
        # ---- The pathogen which is caused the diarrhoea  ----
        'gi_last_diarrhoea_pathogen': Property(Types.CATEGORICAL,
                                               'Attributable pathogen for the last episode of diarrhoea',
                                               categories=list(pathogens) + ['none']),

        # ---- Classification of the type of diarrhoea that is caused  ----
        'gi_last_diarrhoea_type': Property(Types.CATEGORICAL,
                                           'Type of the last episode of diarrhoea',
                                           categories=['none',
                                                       'watery',
                                                       'bloody']),

        # ---- Classification of severity of the dehydration caused ----
        'gi_last_diarrhoea_dehydration': Property(Types.CATEGORICAL,
                                                  'Severity of dehydration of last episode of diarrhoea',
                                                  categories=['none',
                                                              'some',
                                                              'severe']),

        # ---- Internal variables to schedule onset and deaths due to diarrhoea  ----
        'gi_last_diarrhoea_date_of_onset': Property(Types.DATE, 'date of onset of last episode of diarrhoea'),
        'gi_last_diarrhoea_duration': Property(Types.REAL, 'number of days of last episode of diarrhoea'),
        'gi_last_diarrhoea_recovered_date': Property(Types.DATE, 'date of recovery from last episode of diarrhoea'),
        'gi_last_diarrhoea_death_date': Property(Types.DATE, 'date of death caused by last episode of diarrhoea'),

        # ---- Temporary Variables: To be replaced with the properties of other modules ----
        'tmp_malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'tmp_exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'tmp_continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'tmp_hv_inf': Property(Types.BOOL, 'Temporary property - current HIV infection')
    }

    # Declare symptoms that this module will cause:
    SYMPTOMS = {
        'diarrhoea',
        'bloody_stool',
        'fever',
        'vomiting',
        'dehydration'  # NB. This is non-severe dehydration
    }


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
        blank_counter = dict(zip(self.pathogens, [list() for _ in self.pathogens]))
        # dict(zip(keys, values)) covert two lists into a dictionary, is the same as:
        # blank_counter = dict()
        # for pathogen in self.pathogens:
        #     blank_counter[pathogen] = list()

        self.incident_case_tracker_blank = {
            '0y': copy.deepcopy(blank_counter),
            '1y': copy.deepcopy(blank_counter),
            '2-4y': copy.deepcopy(blank_counter),
            '5+y': copy.deepcopy(blank_counter)
        }
        self.incident_case_tracker = copy.deepcopy(self.incident_case_tracker_blank)

        zeros_counter = dict(zip(self.pathogens, [0] * len(self.pathogens)))
        self.incident_case_tracker_zeros = {
            '0y': copy.deepcopy(zeros_counter),
            '1y': copy.deepcopy(zeros_counter),
            '2-4y': copy.deepcopy(zeros_counter),
            '5+y': copy.deepcopy(zeros_counter)
        }

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module """
        p = self.parameters

        # Read parameters from the resourcefile
        self.load_parameters_from_dataframe(
            pd.read_excel(
                Path(self.resourcefilepath) / 'ResourceFile_Diarrhoea.xlsx', sheet_name='Parameter_values')
        )

        # Check that every value has been read-in successfully
        for param_name, type in self.PARAMETERS.items():
            assert param_name in self.parameters, f'Parameter "{param_name}" ' \
                f'is not read in correctly from the resourcefile.'
            assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert isinstance(self.parameters[param_name],
                              type.python_type), f'Parameter "{param_name}" ' \
                f'is not read in correctly from the resourcefile.'

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_population(self, population):
        """
        Sets that there is no one with diarrahoea at initiation.
        """
        #TODO - loc on is_alive -- HERE AND THROUGHOUT
        df = population.props  # a shortcut to the data-frame storing data for individuals

        # ---- Key Current Status Classification Properties ----
        df['gi_last_diarrhoea_pathogen'].values[:] = 'none'
        df['gi_last_diarrhoea_type'].values[:] = 'none'
        df['gi_last_diarrhoea_dehydration'].values[:] = 'none'

        # ---- Internal values ----
        df['gi_last_diarrhoea_date_of_onset'] = pd.NaT
        df['gi_last_diarrhoea_duration'] = np.nan
        df['gi_last_diarrhoea_recovered_date'] = pd.NaT
        df['gi_last_diarrhoea_death_date'] = pd.NaT

        # ---- Temporary values ----
        df['tmp_malnutrition'] = False
        df['tmp_exclusive_breastfeeding'] = False
        df['tmp_continued_breastfeeding'] = False

    def initialise_simulation(self, sim):
        """Prepares for simulation:
        * Schedules the main polling event
        * Schedules the main logging event
        * Estbalishes the linear models and other data sturctures using the parameters that have been read-in
        """

        # Schedule the main polling event (to first occur immidiately)
        sim.schedule_event(DiarrhoeaPollingEvent(self), sim.date + DateOffset(months=0))

        # Schedule the main logging event (to first occur in one year)
        sim.schedule_event(DiarrhoeaLoggingEvent(self), sim.date + DateOffset(years=1))

        # Get DALY weights
        if 'HealthBurden' in self.sim.modules.keys():
            self.daly_wts['mild_diarrhoea'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=32)
            self.daly_wts['moderate_diarrhoea'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=35)
            self.daly_wts['severe_diarrhoea'] = \
                self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=34)

        # --------------------------------------------------------------------------------------------
        # Make a dict to hold the equations that govern the probability that a person acquires diarrhoea
        # that is caused (primarily) by a pathogen
        p = self.parameters
        self.incidence_equations_by_pathogen.update({
            'rotavirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('age_years')
                                         .when('.between(0,0)', p['base_inc_rate_diarrhoea_by_rotavirus'][0])
                                         .when('.between(1,1)', p['base_inc_rate_diarrhoea_by_rotavirus'][1])
                                         .when('.between(2,4)', p['base_inc_rate_diarrhoea_by_rotavirus'][2])
                                         .otherwise(0.0),
                                     Predictor('li_no_access_handwashing')
                                        .when(False, p['rr_diarrhoea_HHhandwashing']),
                                     Predictor('li_no_clean_drinking_water')
                                        .when(False, p['rr_diarrhoea_clean_water']),
                                     Predictor('li_unimproved_sanitation')
                                        .when(False, p['rr_diarrhoea_improved_sanitation']),
                                     Predictor('tmp_hv_inf')
                                        .when(True, p['rr_diarrhoea_HIV']),
                                     Predictor('tmp_malnutrition')
                                        .when(True, p['rr_diarrhoea_SAM']),
                                     Predictor('tmp_exclusive_breastfeeding')
                                        .when(False, p['rr_diarrhoea_excl_breast'])
                                     )
        })

        self.incidence_equations_by_pathogen.update({
            'shigella': LinearModel(LinearModelType.MULTIPLICATIVE,
                                    1.0,
                                    Predictor('age_years')
                                        .when('.between(0,0)', p['base_inc_rate_diarrhoea_by_shigella'][0])
                                        .when('.between(1,1)', p['base_inc_rate_diarrhoea_by_shigella'][1])
                                        .when('.between(2,4)', p['base_inc_rate_diarrhoea_by_shigella'][2])
                                        .otherwise(0.0),
                                    Predictor('li_no_access_handwashing')
                                        .when(False, p['rr_diarrhoea_HHhandwashing']),
                                    Predictor('li_no_clean_drinking_water')
                                        .when(False, p['rr_diarrhoea_clean_water']),
                                    Predictor('li_unimproved_sanitation')
                                        .when(False, p['rr_diarrhoea_improved_sanitation']),
                                    Predictor('tmp_hv_inf')
                                        .when(True, p['rr_diarrhoea_HIV']),
                                    Predictor('tmp_malnutrition')
                                        .when(True, p['rr_diarrhoea_SAM']),
                                    Predictor('tmp_exclusive_breastfeeding')
                                        .when(False, p['rr_diarrhoea_excl_breast'])
                                    )
        })

        self.incidence_equations_by_pathogen.update({
            'adenovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                      1.0,
                                      Predictor('age_years')
                                          .when('.between(0,0)', p['base_inc_rate_diarrhoea_by_adenovirus'][0])
                                          .when('.between(1,1)', p['base_inc_rate_diarrhoea_by_adenovirus'][1])
                                          .when('.between(2,4)', p['base_inc_rate_diarrhoea_by_adenovirus'][2])
                                          .otherwise(0.0),
                                      Predictor('li_no_access_handwashing')
                                            .when(False, p['rr_diarrhoea_HHhandwashing']),
                                      Predictor('li_no_clean_drinking_water')
                                            .when(False, p['rr_diarrhoea_clean_water']),
                                      Predictor('li_unimproved_sanitation')
                                            .when(False, p['rr_diarrhoea_improved_sanitation']),
                                      Predictor('tmp_hv_inf')
                                            .when(True, p['rr_diarrhoea_HIV']),
                                      Predictor('tmp_malnutrition')
                                            .when(True, p['rr_diarrhoea_SAM']),
                                      Predictor('tmp_exclusive_breastfeeding')
                                            .when(False, p['rr_diarrhoea_excl_breast'])
                                      )
        })

        self.incidence_equations_by_pathogen.update({
            'cryptosporidium': LinearModel(LinearModelType.MULTIPLICATIVE,
                                           1.0,
                                           Predictor('age_years')
                                               .when('.between(0,0)', p['base_inc_rate_diarrhoea_by_cryptosporidium'][0])
                                               .when('.between(1,1)', p['base_inc_rate_diarrhoea_by_cryptosporidium'][1])
                                               .when('.between(2,4)', p['base_inc_rate_diarrhoea_by_cryptosporidium'][2])
                                               .otherwise(0.0),
                                           Predictor('li_no_access_handwashing')
                                                .when(False, p['rr_diarrhoea_HHhandwashing']),
                                           Predictor('li_no_clean_drinking_water')
                                                .when(False, p['rr_diarrhoea_clean_water']),
                                           Predictor('li_unimproved_sanitation')
                                                .when(False, p['rr_diarrhoea_improved_sanitation']),
                                           Predictor('tmp_hv_inf')
                                                .when(True, p['rr_diarrhoea_HIV']),
                                           Predictor('tmp_malnutrition')
                                                .when(True, p['rr_diarrhoea_SAM']),
                                           Predictor('tmp_exclusive_breastfeeding')
                                                .when(False, p['rr_diarrhoea_excl_breast'])
                                           )
        })

        self.incidence_equations_by_pathogen.update({
            'campylobacter': LinearModel(LinearModelType.MULTIPLICATIVE,
                                         1.0,
                                         Predictor('age_years')
                                             .when('.between(0,0)', p['base_inc_rate_diarrhoea_by_campylobacter'][0])
                                             .when('.between(1,1)', p['base_inc_rate_diarrhoea_by_campylobacter'][1])
                                             .when('.between(2,4)', p['base_inc_rate_diarrhoea_by_campylobacter'][2])
                                             .otherwise(0.0),
                                         Predictor('li_no_access_handwashing')
                                            .when(False, p['rr_diarrhoea_HHhandwashing']),
                                         Predictor('li_no_clean_drinking_water')
                                            .when(False, p['rr_diarrhoea_clean_water']),
                                         Predictor('li_unimproved_sanitation')
                                            .when(False, p['rr_diarrhoea_improved_sanitation']),
                                         Predictor('tmp_hv_inf')
                                            .when(True, p['rr_diarrhoea_HIV']),
                                         Predictor('tmp_malnutrition')
                                            .when(True, p['rr_diarrhoea_SAM']),
                                         Predictor('tmp_exclusive_breastfeeding')
                                            .when(False, p['rr_diarrhoea_excl_breast'])
                                         )
        })

        self.incidence_equations_by_pathogen.update({
            'ST-ETEC': LinearModel(LinearModelType.MULTIPLICATIVE,
                                   1.0,
                                   Predictor('age_years')
                                       .when('.between(0,0)', p['base_inc_rate_diarrhoea_by_ST-ETEC'][0])
                                       .when('.between(1,1)', p['base_inc_rate_diarrhoea_by_ST-ETEC'][1])
                                       .when('.between(2,4)', p['base_inc_rate_diarrhoea_by_ST-ETEC'][2])
                                       .otherwise(0.0),
                                   Predictor('li_no_access_handwashing')
                                        .when(False, p['rr_diarrhoea_HHhandwashing']),
                                   Predictor('li_no_clean_drinking_water')
                                        .when(False, p['rr_diarrhoea_clean_water']),
                                   Predictor('li_unimproved_sanitation')
                                        .when(False, p['rr_diarrhoea_improved_sanitation']),
                                   Predictor('tmp_hv_inf')
                                        .when(True, p['rr_diarrhoea_HIV']),
                                   Predictor('tmp_malnutrition')
                                        .when(True, p['rr_diarrhoea_SAM']),
                                   Predictor('tmp_exclusive_breastfeeding')
                                        .when(False, p['rr_diarrhoea_excl_breast'])
                                   )
        })

        self.incidence_equations_by_pathogen.update({
            'sapovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('age_years')
                                         .when('.between(0,0)', p['base_inc_rate_diarrhoea_by_sapovirus'][0])
                                         .when('.between(1,1)', p['base_inc_rate_diarrhoea_by_sapovirus'][1])
                                         .when('.between(2,4)', p['base_inc_rate_diarrhoea_by_sapovirus'][2])
                                         .otherwise(0.0),
                                     Predictor('li_no_access_handwashing')
                                        .when(False, p['rr_diarrhoea_HHhandwashing']),
                                     Predictor('li_no_clean_drinking_water')
                                        .when(False, p['rr_diarrhoea_clean_water']),
                                     Predictor('li_unimproved_sanitation')
                                        .when(False, p['rr_diarrhoea_improved_sanitation']),
                                     Predictor('tmp_hv_inf')
                                        .when(True, p['rr_diarrhoea_HIV']),
                                     Predictor('tmp_malnutrition')
                                        .when(True, p['rr_diarrhoea_SAM']),
                                     Predictor('tmp_exclusive_breastfeeding')
                                        .when(False, p['rr_diarrhoea_excl_breast'])
                                     )
        })

        self.incidence_equations_by_pathogen.update({
            'norovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                     1.0,
                                     Predictor('age_years')
                                         .when('.between(0,0)', p['base_inc_rate_diarrhoea_by_norovirus'][0])
                                         .when('.between(1,1)', p['base_inc_rate_diarrhoea_by_norovirus'][1])
                                         .when('.between(2,4)', p['base_inc_rate_diarrhoea_by_norovirus'][2])
                                         .otherwise(0.0),
                                     Predictor('li_no_access_handwashing')
                                        .when(False, p['rr_diarrhoea_HHhandwashing']),
                                     Predictor('li_no_clean_drinking_water')
                                        .when(False, p['rr_diarrhoea_clean_water']),
                                     Predictor('li_unimproved_sanitation')
                                        .when(False, p['rr_diarrhoea_improved_sanitation']),
                                     Predictor('tmp_hv_inf')
                                        .when(True, p['rr_diarrhoea_HIV']),
                                     Predictor('tmp_malnutrition')
                                        .when(True, p['rr_diarrhoea_SAM']),
                                     Predictor('tmp_exclusive_breastfeeding')
                                        .when(False, p['rr_diarrhoea_excl_breast'])
                                     )
        })

        self.incidence_equations_by_pathogen.update({
            'astrovirus': LinearModel(LinearModelType.MULTIPLICATIVE,
                                      1.0,
                                      Predictor('age_years')
                                        .when('.between(0,0)', p['base_inc_rate_diarrhoea_by_astrovirus'][0])
                                        .when('.between(1,1)', p['base_inc_rate_diarrhoea_by_astrovirus'][1])
                                        .when('.between(2,4)', p['base_inc_rate_diarrhoea_by_astrovirus'][2])
                                        .otherwise(0.0),
                                      Predictor('li_no_access_handwashing')
                                        .when(False, p['rr_diarrhoea_HHhandwashing']),
                                      Predictor('li_no_clean_drinking_water')
                                        .when(False, p['rr_diarrhoea_clean_water']),
                                      Predictor('li_unimproved_sanitation')
                                        .when(False, p['rr_diarrhoea_improved_sanitation']),
                                      Predictor('tmp_hv_inf')
                                        .when(True, p['rr_diarrhoea_HIV']),
                                      Predictor('tmp_malnutrition')
                                        .when(True, p['rr_diarrhoea_SAM']),
                                      Predictor('tmp_exclusive_breastfeeding')
                                        .when(False, p['rr_diarrhoea_excl_breast'])
                                      )
        })

        self.incidence_equations_by_pathogen.update({
            'tEPEC': LinearModel(LinearModelType.MULTIPLICATIVE,
                                 1.0,
                                 Predictor('age_years')
                                     .when('.between(0,0)', p['base_inc_rate_diarrhoea_by_tEPEC'][0])
                                     .when('.between(1,1)', p['base_inc_rate_diarrhoea_by_tEPEC'][1])
                                     .when('.between(2,4)', p['base_inc_rate_diarrhoea_by_tEPEC'][2])
                                     .otherwise(0.0),
                                 Predictor('li_no_access_handwashing')
                                    .when(False, p['rr_diarrhoea_HHhandwashing']),
                                 Predictor('li_no_clean_drinking_water')
                                    .when(False, p['rr_diarrhoea_clean_water']),
                                 Predictor('li_unimproved_sanitation')
                                    .when(False, p['rr_diarrhoea_improved_sanitation']),
                                 Predictor('tmp_hv_inf')
                                    .when(True, p['rr_diarrhoea_HIV']),
                                 Predictor('tmp_malnutrition')
                                    .when(True, p['rr_diarrhoea_SAM']),
                                 Predictor('tmp_exclusive_breastfeeding')
                                    .when(False, p['rr_diarrhoea_excl_breast'])
                                 )
        })

        # Check that equations have been declared for each of the pathogens
        assert self.pathogens == set(list(self.incidence_equations_by_pathogen.keys()))

        # Scale the models of risk of acquiring pathogens so that they give the desired means (has to be done at
        # run-time because it will depend on frequency of other properties.

        for k, v in self.incidence_equations_by_pathogen.items():
            self.incidence_equations_by_pathogen[k] = self.rescale_equ(v)


        # --------------------------------------------------------------------------------------------
        # Make a dict containing the probability of symptoms onset given acquisition of diarrhoeaa caused
        # by a particular pathogen

        self.prob_symptoms.update({
            'rotavirus': {
                'diarrhoea': 1.0,
                'bloody_stool': 1 - p['proportion_AWD_by_rotavirus'],
                'fever': p['fever_by_rotavirus'],
                'vomiting': p['vomiting_by_rotavirus'],
                'dehydration': p['dehydration_by_rotavirus'],
            },

            'shigella': {
                'diarrhoea': 1.0,
                'bloody_stool': 1 - p['proportion_AWD_by_shigella'],
                'fever': p['fever_by_shigella'],
                'vomiting': p['vomiting_by_shigella'],
                'dehydration': p['dehydration_by_shigella'],
            },

            'adenovirus': {
                'diarrhoea': 1.0,
                'bloody_stool': 1 - p['proportion_AWD_by_adenovirus'],
                'fever': p['fever_by_adenovirus'],
                'vomiting': p['vomiting_by_adenovirus'],
                'dehydration': p['dehydration_by_adenovirus'],
            },

            'cryptosporidium': {
                'diarrhoea': 1.0,
                'bloody_stool': 1 - p['proportion_AWD_by_cryptosporidium'],
                'fever': p['fever_by_cryptosporidium'],
                'vomiting': p['vomiting_by_cryptosporidium'],
                'dehydration': p['dehydration_by_cryptosporidium'],
            },

            'campylobacter': {
                'diarrhoea': 1.0,
                'bloody_stool': 1 - p['proportion_AWD_by_campylobacter'],
                'fever': p['fever_by_campylobacter'],
                'vomiting': p['vomiting_by_campylobacter'],
                'dehydration': p['dehydration_by_campylobacter'],
            },

            'ST-ETEC': {
                'diarrhoea': 1.0,
                'bloody_stool': 1 - p['proportion_AWD_by_ST-ETEC'],
                'fever': p['fever_by_ST-ETEC'],
                'vomiting': p['vomiting_by_ST-ETEC'],
                'dehydration': p['dehydration_by_ST-ETEC'],
            },

            'sapovirus': {
                'diarrhoea': 1.0,
                'bloody_stool': 1 - p['proportion_AWD_by_sapovirus'],
                'fever': p['fever_by_sapovirus'],
                'vomiting': p['vomiting_by_sapovirus'],
                'dehydration': p['dehydration_by_sapovirus'],
            },

            'norovirus': {
                'diarrhoea': 1.0,
                'bloody_stool': 1 - p['proportion_AWD_by_norovirus'],
                'fever': p['fever_by_norovirus'],
                'vomiting': p['vomiting_by_norovirus'],
                'dehydration': p['dehydration_by_norovirus'],
            },

            'astrovirus': {
                'diarrhoea': 1.0,
                'bloody_stool': 1 - p['proportion_AWD_by_astrovirus'],
                'fever': p['fever_by_astrovirus'],
                'vomiting': p['vomiting_by_astrovirus'],
                'dehydration': p['dehydration_by_astrovirus'],
            },

            'tEPEC': {
                'diarrhoea': 1.0,
                'bloody_stool': 1 - p['proportion_AWD_by_tEPEC'],
                'fever': p['fever_by_rotavirus'],
                'vomiting': p['vomiting_by_tEPEC'],
                'dehydration': p['dehydration_by_tEPEC'],
            },
        })

        # Check that each pathogen has a risk of developing each symptom
        assert self.pathogens == set(list(self.prob_symptoms.keys()))

        assert all(
            [
                self.SYMPTOMS == set(list(self.prob_symptoms[pathogen].keys()))
                for pathogen in self.prob_symptoms.keys()
            ]
        )

        # --------------------------------------------------------------------------------------------
        # Creat the linear model for the risk of dying due to diarrhoea
        # todo -- no magic numbers
        self.risk_of_death_diarrhoea = \
            LinearModel(LinearModelType.MULTIPLICATIVE,
                        1.0,
                        Predictor('gi_last_diarrhoea_type')
                            .when('watery', p['case_fatality_rate_AWD'])
                            .when('bloody', p['case_fatality_rate_dysentery']),
                        Predictor('gi_last_diarrhoea_duration')
                            .when('>13', 0.0035),
                        Predictor('gi_last_diarrhoea_dehydration')
                            .when('severe', 0.05)
                            .when('persistent', 0.05),
                        Predictor('age_years')
                            .when('.between(1,2)', p['rr_diarr_death_age12to23mo'])
                            .when('.between(2,4)', p['rr_diarr_death_age24to59mo'])
                            .otherwise(0.0),
                        Predictor('tmp_hv_inf')
                            .when(True, p['rr_diarrhoea_HIV']),
                        Predictor('tmp_malnutrition')
                            .when(True, p['rr_diarrhoea_SAM'])
                        )
        # --------------------------------------------------------------------------------------------
        # Create the linear model for the duration of the episode of diarrhoea
        # TODO; add these add parameters
        self.mean_duration_in_days_of_diarrhoea = \
            LinearModel(LinearModelType.ADDITIVE,
                        0.0,
                        Predictor('gi_last_diarrhoea_pathogen')
                            .when('rotavirus', 7)
                            .when('shigella', 7)
                            .when('adenovirus', 7)
                            .when('cryptosporidium', 7)
                            .when('campylobacter', 7)
                            .when('ST-ETEC', 7)
                            .when('sapovirus', 7)
                            .when('norovirus', 7)
                            .when('astrovirus', 7)
                            .when('tEPEC', 7)
                        )

        # --------------------------------------------------------------------------------------------
        # Create the linear model for the probability that the diarrhoea is 'watery' (rather than 'bloody')
        self.prob_diarrhoea_is_watery = \
            LinearModel(LinearModelType.ADDITIVE,
                        0.0,
                        Predictor('gi_last_diarrhoea_pathogen')
                            .when('rotavirus', p['proportion_AWD_by_rotavirus'])
                            .when('shigella', p['proportion_AWD_by_shigella'])
                            .when('adenovirus', p['proportion_AWD_by_adenovirus'])
                            .when('cryptosporidium', p['proportion_AWD_by_cryptosporidium'])
                            .when('campylobacter', p['proportion_AWD_by_campylobacter'])
                            .when('ST-ETEC', p['proportion_AWD_by_ST-ETEC'])
                            .when('sapovirus', p['proportion_AWD_by_sapovirus'])
                            .when('norovirus', p['proportion_AWD_by_norovirus'])
                            .when('astrovirus', p['proportion_AWD_by_astrovirus'])
                            .when('tEPEC', p['proportion_AWD_by_tEPEC'])
                        )
        # --------------------------------------------------------------------------------------------

    def on_birth(self, mother_id, child_id):
        """
        On birth, all children will have no diarrhoea
        """
        df = self.sim.population.props

        # ---- Key Current Status Classification Properties ----
        df.at[child_id, 'gi_last_diarrhoea_pathogen'] = 'none'
        df.at[child_id, 'gi_last_diarrhoea_type'] = 'none'
        df.at[child_id, 'gi_last_diarrhoea_dehydration'] = 'none'

        # ---- Internal values ----
        df.at[child_id, 'gi_last_diarrhoea_date_of_onset'] = pd.NaT
        df.at[child_id, 'gi_last_diarrhoea_recovered_date'] = pd.NaT
        df.at[child_id, 'gi_last_diarrhoea_death_date'] = pd.NaT

        # ---- Temporary values ----
        df.at[child_id, 'tmp_malnutrition'] = False
        df.at[child_id, 'tmp_exclusive_breastfeeding'] = False
        df.at[child_id, 'tmp_continued_breastfeeding'] = False

    def rescale_equ(self, lm):
        """
        Accepts a linear model, and adjust its intercept value such that the mean incidence rate of 0 year-olds matches
        the specified marginal value for 0 year-olds in the model.
        NB. Have to pluck out the value from the linear model rather than run because that is how this model is
        parameterised.
        """
        # mean when lm is applied
        df = self.sim.population.props
        actual_mean = lm.predict(df.loc[df.is_alive & (df.age_years == 0)]).mean()

        # target mean value as specified in the lm for a 0 year-old with no other conditions
        assert lm.predictors[0].property_name == 'age_years'

        # check that the first conditions is for 0 year-olds
        assert lm.predictors[0].conditions[0][0] == '(age_years.between(0,0))'

        target_mean = lm.predictors[0].conditions[0][1]

        # adjust value of intercept
        lm.intercept *= target_mean / actual_mean

        # check by applying the model to the under 5 population
        assert (target_mean - lm.predict(df.loc[df.is_alive & (df.age_years == 0)]).mean()) < 1e-10

        return lm

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

        # Assume that the current DALY loading is linked to current symptomatic experience of
        # symptoms diarrhoea and dehyrdration
        # Check this!

        total_daly_values = pd.Series(data=0.0, index=df.loc[df['is_alive']].index)
        total_daly_values.loc[
            self.sim.modules['SymptomManager'].who_has('diarrhoea')
        ] \
            = self.daly_wts['mild_diarrhoea']
        total_daly_values.loc[
            self.sim.modules['SymptomManager'].who_has(['diarrhoea', 'dehydration'])
        ] \
            = self.daly_wts['moderate_diarrhoea']

        total_daly_values.loc[
            self.sim.population.props['gi_last_diarrhoea_dehydration'] == 'severe'
        ] \
            = self.daly_wts['severe_diarrhoea']

        # Split out by pathogen that causes the diarrhoea
        dummies_for_pathogen = pd.get_dummies(df.loc[total_daly_values.index,
                                                     'gi_last_diarrhoea_pathogen'],
                                              dtype='float')
        daly_values_by_pathogen = dummies_for_pathogen.mul(total_daly_values, axis=0).drop(columns='none')

        return daly_values_by_pathogen


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
#
# ---------------------------------------------------------------------------------------------------------

class DiarrhoeaPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This is the main event that runs the acquisition of pathogens that cause Diarrhoea.
    It determines who is infected and when and schedules individual IncidentCase events to represent onset.

    A known issue is that diarrhoea events are scheduled based on the risk of current age but occur a short time
    later when the children have aged. This means that when comparing the
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        # NB. The frequency of the occurrences of this event can be edited safely.

    def apply(self, population):
        df = population.props
        rng = self.module.rng
        m = self.module

        # Compute the incidence rate for each person getting diarrhoea and then convert into a probability
        # getting all children that do not have diarrhoea currently
        mask_could_get_new_diarrhoea_episode = \
            df['is_alive'] & (df['age_years'] < 5) & ((df['gi_last_diarrhoea_recovered_date'] <= self.sim.date) |
                                                      pd.isnull(df['gi_last_diarrhoea_recovered_date']))

        inc_of_aquiring_pathogen = pd.DataFrame(index=df.loc[mask_could_get_new_diarrhoea_episode].index)

        for pathogen in m.pathogens:
            inc_of_aquiring_pathogen[pathogen] = m.incidence_equations_by_pathogen[pathogen] \
                .predict(df.loc[mask_could_get_new_diarrhoea_episode])

        # Convert the incidence rates that are predicted by the model into risk of an event occurring before the next
        # polling event
        fraction_of_a_year_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(
            1, 'Y')
        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')
        probs_of_aquiring_pathogen = 1 - np.exp(-inc_of_aquiring_pathogen * fraction_of_a_year_until_next_polling_event)

        # Create the probability of getting 'any' pathogen:
        # (Assumes that pathogens are mutually exclusive)
        prob_of_acquiring_any_pathogen = probs_of_aquiring_pathogen.sum(axis=1)
        assert all(prob_of_acquiring_any_pathogen < 1.0)

        # Determine which persons will acquire any pathogen:
        person_id_that_acquire_pathogen = prob_of_acquiring_any_pathogen.index[
            rng.rand(len(prob_of_acquiring_any_pathogen)) < prob_of_acquiring_any_pathogen
            ]

        # Determine which pathogen each person will acquire (among those who will get a pathogen)
        # and create the event for the onset of new infection
        for person_id in person_id_that_acquire_pathogen:
            # ----------------------- Allocate a pathogen to the person ----------------------
            p_by_pathogen = probs_of_aquiring_pathogen.loc[person_id].values
            normalised_p_by_pathogen = p_by_pathogen / sum(p_by_pathogen)  # don't understand this normalised
            pathogen = rng.choice(probs_of_aquiring_pathogen.columns,
                                  p=normalised_p_by_pathogen)

            # ----------------------- Allocate a date of onset diarrhoea ----------------------
            date_onset = self.sim.date + DateOffset(days=np.random.randint(0, days_until_next_polling_event))

            # TODO - the rest of this to go into the Incident Event
            # ----------------------- Determine outcomes for this case ----------------------
            duration_in_days_of_diarrhoea = max(1, int(
                m.mean_duration_in_days_of_diarrhoea.predict(df.loc[[person_id]]).values[0] +
                (-2 + 7 * rng.rand())  # assumes uniform interval around mean duration
            ))


            # ----------------------- Allocate symptoms to onset of diarrhoea ----------------------
            possible_symptoms_for_this_pathogen = m.prob_symptoms[pathogen]
            symptoms_for_this_person = list()
            for symptom, prob in possible_symptoms_for_this_pathogen.items():
                if rng.rand() < prob:
                    symptoms_for_this_person.append(symptom)

            # ----------------------- Create the event for the onset of infection -------------------
            # NB. The symptoms are scheduled by the SymptomManager to 'autoresolve' after the duration
            #       of the diarrhoea.
            self.sim.schedule_event(
                event=DiarrhoeaIncidentCase(
                    module=self.module,
                    person_id=person_id,
                    pathogen=pathogen,
                    duration_in_days=duration_in_days_of_diarrhoea,
                    symptoms=symptoms_for_this_person
                ),
                date=date_onset
            )


class DiarrhoeaIncidentCase(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of the infection that causes diarrhoea.
    """

    def __init__(self, module, person_id, pathogen, duration_in_days, symptoms):
        super().__init__(module, person_id=person_id)
        self.pathogen = pathogen
        self.duration_in_days = duration_in_days
        self.symptoms = symptoms

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module

        # The event should not run if the person is not currently alive
        if not df.at[person_id, 'is_alive']:
            return

        # Update the properties in the dataframe:
        df.at[person_id, 'gi_last_diarrhoea_pathogen'] = self.pathogen
        df.at[person_id, 'gi_last_diarrhoea_date_of_onset'] = self.sim.date
        df.at[person_id, 'gi_last_diarrhoea_duration'] = self.duration_in_days

        if 'bloody_stool' in self.symptoms:
            df.at[person_id, 'gi_last_diarrhoea_type'] = 'bloody'
        else:
            df.at[person_id, 'gi_last_diarrhoea_type'] = 'watery'

        if 'dehydration' in self.symptoms:
            df.at[person_id, 'gi_last_diarrhoea_dehydration'] = 'some'
        else:
            df.at[person_id, 'gi_last_diarrhoea_dehydration'] = 'none'

        # Onset symptoms:
        for symptom in self.symptoms:
            self.module.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove='+',
                disease_module=self.module,
                duration_in_days=self.duration_in_days
            )

        # Determine the progress to severe dehydration
        date_of_outcome = self.module.sim.date + DateOffset(days=self.duration_in_days)
        if df.at[person_id, 'gi_last_diarrhoea_dehydration'] == 'some':
            prob_severe_dehydration = self.module.rng.rand() < 0.3    #todo - make into parameter
            if prob_severe_dehydration:
                self.sim.schedule_event(DiarrhoeaSevereDehydrationEvent(self.module, person_id, symptoms=self.symptoms),
                                        max(self.sim.date, date_of_outcome - DateOffset(
                                            days=(3 * m.rng.rand()))))  # todo - make into parameters

        # Determine timing of outcome (either recovery or death)
        death_outcome = m.risk_of_death_diarrhoea.predict(df.loc[[person_id]], m.rng)

        if death_outcome:
            df.at[person_id, 'gi_last_diarrhoea_recovered_date'] = pd.NaT
            df.at[person_id, 'gi_last_diarrhoea_death_date'] = date_of_outcome
            self.sim.schedule_event(DiarrhoeaDeathEvent(self.module, person_id),
                                    date_of_outcome)
        else:
            df.at[person_id, 'gi_last_diarrhoea_recovered_date'] = date_of_outcome
            df.at[person_id, 'gi_last_diarrhoea_death_date'] = pd.NaT

        # Add this incident case to the tracker
        age = df.loc[person_id, ['age_years']]
        if age.values[0] < 5:
            age_grp = age.map({0: '0y', 1: '1y', 2: '2-4y', 3: '2-4y', 4: '2-4y'}).values[0]
        else:
            age_grp = '5+y'
        self.module.incident_case_tracker[age_grp][self.pathogen].append(self.sim.date)


class DiarrhoeaSevereDehydrationEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id, symptoms):
        super().__init__(module, person_id=person_id)
        self.symptoms = symptoms

    def apply(self, person_id):
        logger.debug("DiarrhoeaSevereDehydrationEvent: Severe dehydration in person %d", person_id)
        df = self.sim.population.props

        # terminate the event if the person has already died.
        if not df.at[person_id, 'is_alive']:
            return

        df.at[person_id, 'gi_last_diarrhoea_dehydration'] = 'severe'


class DiarrhoeaCureEvent(Event, IndividualScopeEventMixin):
    """
    #This is the cure event. It is called by HSI events that have implemented a cure.
    It does the following:
        * "cancels" the death caused by diarrhoea
        * sets the date of recovery
        * resolves all symptoms caused by diarrhoea
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("DiarrhoeaCureEvent: Stopping diarrhoea treatment and curing person %d", person_id)
        df = self.sim.population.props

        # terminate the event if the person has already died.
        if not df.at[person_id, 'is_alive']:
            return

        # Stop the person from dying of Diarrhoea (if they were going to die)
        df.at[person_id, 'gi_last_diarrhoea_recovered_date'] = self.sim.date
        df.at[person_id, 'gi_last_diarrhoea_death_date'] = pd.NaT

        # Resolve all the symptoms immediately
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id,
                                                          disease_module=self.sim.modules['Diarrhoea'])

class DiarrhoeaDeathEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the death of someone that is caused by the infection with a pathogen that causes diarrhoea.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # Check if person should still die of diarrhoea
        if (df.at[person_id, 'is_alive']) and (df.at[person_id, 'gi_last_diarrhoea_death_date'] == self.sim.date):
            self.sim.schedule_event(demography.InstantaneousDeath(self.module,
                                                                  person_id,
                                                                  cause='Diarrhoea_' + df.at[
                                                                      person_id, 'gi_last_diarrhoea_pathogen']
                                                                  ),
                                    self.sim.date
                                    )


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class DiarrhoeaLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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

# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------

class HSI_Diarrhoea_Treatment_PlanA(HSI_Event, IndividualScopeEventMixin):
    """
    This is a treatment for uncomplicated diarrhoea administered at outpatient setting through IMCI
    """
    # no dehydration - PLAN A
    # advise on follow up in 5 days

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient
        self.TREATMENT_ID = 'Diarrhoea_Treatment_PlanA'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug('Provide Treatment Plan A for uncomplicated Diarrhoea')

        # Stop the person from dying of Diarrhoea (if they were going to die)
        df = self.sim.population.props
        # Get consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_uncomplic_diarrhoea = \
            pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'ORS',
                                'Intervention_Pkg_Code'])[0]

        # give the mother 2 packets of ORS
        # give zinc (2 mo up to 5 years) - <6 mo 1/2 tablet (10mg) for 10 days, >6 mo 1 tab (20mg) for 10 days
        # follow up in 5 days if not improving
        # continue feeding

        the_consumables_needed = {
            'Intervention_Package_Code': {pkg_code_uncomplic_diarrhoea: 1},
            'Item_Code': {}
        }

        # Request the treatment
        is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_consumables_needed)
        logger.warning(f"is_cons_available ({is_cons_available}) should be used in this method")

        if is_cons_available['Intervention_Package_Code']['pkg_code_uncomplic_diarrhoea']:
            logger.debug('HSI_Diarrhoea_Dysentery: giving uncomplicated diarrhoea treatment for child %d',
                         person_id)
            if df.at[person_id, 'is_alive']:
                self.sim.schedule_event(DiarrhoeaCureEvent(self.module, person_id),
                                        self.sim.date + DateOffset(weeks=1))


class HSI_Diarrhoea_Treatment_PlanB(HSI_Event, IndividualScopeEventMixin):
    """
    This is a treatment for diarrhoea with some dehydration at outpatient setting through IMCI
    """
    # some dehydration with no other severe classification - PLAN B
    # if child has a severe classification -
    # refer urgently to hospital and gic=ving ORS on the way, advise on breastfeeding
    # advise on follow up in 5 days

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient
        self.TREATMENT_ID = 'Diarrhoea_Treatment_PlanB'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug('Provide Treatment Plan B for Diarrhoea with some dehydartion')

        df = self.sim.population.props

        # Get consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_uncomplic_diarrhoea = \
            pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'ORS',
                                'Intervention_Pkg_Code'])[0]
        # Give ORS for the first 4 hours and reassess.

        the_consumables_needed = {
            'Intervention_Package_Code': {pkg_code_uncomplic_diarrhoea: 1},
            'Item_Code': {}
        }

        # Request the treatment
        is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_consumables_needed)
        logger.warning(f"is_cons_available ({is_cons_available}) should be used in this method")

        # TODO; be careful here --- you need to check the availability of the consumable--- copy this change for everytime you use request_consumables.
        if is_cons_available['Intervention_Package_Code']['pkg_code_uncomplic_diarrhoea']:
            logger.debug('HSI_Diarrhoea_Dysentery: giving uncomplicated diarrhoea treatment for child %d',
                         person_id)
            if df.at[person_id, 'is_alive']:
                self.sim.schedule_event(DiarrhoeaCureEvent(self.module, person_id),
                                        self.sim.date + DateOffset(weeks=1))


class HSI_Diarrhoea_Treatment_PlanC(HSI_Event, IndividualScopeEventMixin):
    """
    This is a treatment for diarrhoea with severe dehydration administered at outpatient setting through IMCI
    """
# if child has no other severe classification: PLAN C
# Or if child has another severe classification:
    # refer urgently to hospital with mother giving frequent ORS on the way, advise on breastfeeding
# if cholera is in your area, give antibiotic for cholera

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient
        self.TREATMENT_ID = 'Treatment_PlanC'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug('Provide the treatment for Diarrhoea')

        # Stop the person from dying of Diarrhoea (if they were going to die)
        df = self.sim.population.props
        # Get consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_diarrhoea_severe_dehydration = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Treatment of severe diarrhea',
                            'Intervention_Pkg_Code'])[0]

        the_consumables_needed = {
            'Intervention_Package_Code': {pkg_code_diarrhoea_severe_dehydration: 1},
            'Item_Code': {}
        }

        # Outcome of the request for treatment
        is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_consumables_needed)
        logger.warning(f"is_cons_available ({is_cons_available}) should be used in this method")

        if is_cons_available:   # TODO - this is now how to do a check on the availability?
            logger.debug('HSI_Diarrhoea_Dysentery: giving dysentery treatment for child %d',
                         person_id)
            if df.at[person_id, 'is_alive']:
                # schedule cure date
                self.sim.schedule_event(DiarrhoeaCureEvent(self.module, person_id),
                                        self.sim.date + DateOffset(weeks=1))


class HSI_Diarrhoea_Severe_Persistent_Diarrhoea(HSI_Event, IndividualScopeEventMixin):
    """
    This is a treatment for Severe_Dehydration administered at FacilityLevel=1
    """
    # treat the dehydration and refer to the hospital

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient
        self.TREATMENT_ID = 'Diarrhoea_Severe_Persistent_Diarrhoea'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug('Provide the treatment for Diarrhoea')

        df = self.sim.population.props

        # Get consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = \
            pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'ORS',
                                'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': {pkg_code1: 1},
            'Item_Code': {}
        }
        # if bloody stool and fever in persistent diarroea - give Nalidixic Acid 50mg/kg divided in 4 doses per day for
        # 5 days if malnourished give cotrimoxazole 24mg/kg every 12 hours for 5 days and supplemental feeding and supplements

        # Request the treatment
        is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint)
        logger.warning(f"is_cons_available ({is_cons_available}) should be used in this method")

        if is_cons_available:
            logger.debug('HSI_Diarrhoea_Dysentery: giving dysentery treatment for child %d',
                         person_id)
            if df.at[person_id, 'is_alive']:
                self.sim.schedule_event(DiarrhoeaCureEvent(self.module, person_id),
                                        self.sim.date + DateOffset(weeks=1))


class HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea(HSI_Event, IndividualScopeEventMixin):
    """
    This is a treatment for Severe_Dehydration administered at FacilityLevel=1
    """
    # give multivitamins for 14 days
    # give zinc for 10 days
    # follow up in 5 days

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient
        self.TREATMENT_ID = 'Diarrhoea_Non_Severe_Persistent_Diarrhoea'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug('Provide the treatment for Diarrhoea')

        # Stop the person from dying of Diarrhoea (if they were going to die)
        df = self.sim.population.props

        # Get consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if df.at[person_id, df.age_exact_years < 0.5]:
            pkg_code1 = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Zinc for Children 0-6 months',
                                'Intervention_Pkg_Code'])[0]
            item_code_zinc1 = pd.unique(
                consumables.loc[consumables['Items'] == 'Zinc, tablet, 20 mg', 'Item_Code'])[0]
            the_consumables_needed1 = {
                'Intervention_Package_Code': {pkg_code1: 1},
                'Item_Code': {item_code_zinc1: 5}
            }
            # Request the treatment
            is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=the_consumables_needed1)
            logger.warning(f"is_cons_available ({is_cons_available}) should be used in this method")
            if is_cons_available:
                logger.debug('HSI_Diarrhoea_Dysentery: giving dysentery treatment for child %d',
                             person_id)
                if df.at[person_id, 'is_alive']:
                    self.sim.schedule_event(DiarrhoeaCureEvent(self.module, person_id),
                                            self.sim.date + DateOffset(weeks=1))

        if df.at[person_id, df.age_exact_years > 0.5 & df.age_exact_years < 5]:
            pkg_code2 = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Zinc for Children 6-59 months',
                                'Intervention_Pkg_Code'])[0]
            item_code_zinc2 = pd.unique(
                consumables.loc[consumables['Items'] == 'Zinc, tablet, 20 mg', 'Item_Code'])[0]
            the_consumables_needed2 = {
                'Intervention_Package_Code': {pkg_code2: 1},
                'Item_Code': {item_code_zinc2: 5}
            }
            # Request the treatment
            is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=the_consumables_needed2)
            logger.warning(f"is_cons_available ({is_cons_available}) should be used in this method")
            if is_cons_available:
                logger.debug('HSI_Diarrhoea_Dysentery: giving dysentery treatment for child %d',
                             person_id)
                if df.at[person_id, 'is_alive']:
                    self.sim.schedule_event(DiarrhoeaCureEvent(self.module, person_id),
                                            self.sim.date + DateOffset(weeks=1))

# todo - add in effectiveness = 1.0 & make it linear model.

class HSI_Diarrhoea_Dysentery(HSI_Event, IndividualScopeEventMixin):
    """
    This is a treatment for Severe_Dehydration administered at FacilityLevel=1
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient
        self.TREATMENT_ID = 'Diarrhoea_Dysentery'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug('Provide the treatment for Diarrhoea')

        df = self.sim.population.props

        # Get consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = \
            pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Antibiotics for treatment of dysentery',
                                'Intervention_Pkg_Code'])[0]
        item_code_cipro = pd.unique(
            consumables.loc[consumables['Items'] == 'Ciprofloxacin 250mg_100_CMST', 'Item_Code'])[0]

        # <6 mo - 250mg 1/2 tab x2 daily for 3 days
        # >6 mo upto 5 yo - 250mg 1 tab x2 daily for 3 days
        # follow up in 3 days

        the_cons_footprint = {
            'Intervention_Package_Code': {pkg_code1: 1},
            'Item_Code': {item_code_cipro: 6}
        }

        # Request the treatment
        is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=the_cons_footprint)
        logger.warning(f"is_cons_available ({is_cons_available}) should be used in this method")

        if is_cons_available:
            logger.debug('HSI_Diarrhoea_Dysentery: giving dysentery treatment for child %d',
                         person_id)
            if df.at[person_id, 'is_alive']:
                self.sim.schedule_event(DiarrhoeaCureEvent(self.module, person_id),
                                        self.sim.date + DateOffset(weeks=1))
