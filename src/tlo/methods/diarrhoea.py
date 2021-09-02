"""
Childhood Diarrhoea Module

Overview
=======
Individuals are exposed to the risk of onset of diarrhoea. They can have diarrhoea caused by one pathogen at a time.
 During an episode (prior to recovery- either natural or cured), the symptom of diarrhoea is present in addition to
 other possible symptoms. Diarrhoea may cause dehydration and this may progress to become 'severe' during an episode.
 The individual may recovery naturally or die.

Health care seeking is prompted by the onset of the symptom diarrhoea. The individual can be treated; if successful the
 risk of death is removed and they are cured (symptom resolved) some days later.

Outstanding issues:
==================
* Onset of severe dehydration has no relationship to the risk of death (only the treatment that is provided)
* Risk of death is linked to duration of episode - but this is assigned randomly, so stricly  is not necessary.
* Follow-up appointments for initial HSI events.
* Double check parameters and consumables codes for the HSI events.

"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

class Diarrhoea(Module):
    # Declare the pathogens that this module will simulate:
    pathogens = [
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
    ]

    INIT_DEPENDENCIES = {'Demography', 'Lifestyle', 'HealthSystem', 'SymptomManager'}

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        f'Diarrhoea_{path}': Cause(gbd_causes='Diarrheal diseases', label='Childhood Diarrhoea')
        for path in pathogens
    }

    # Declare Causes of Death and Disability
    CAUSES_OF_DISABILITY = {
        f'Diarrhoea_{path}': Cause(gbd_causes='Diarrheal diseases', label='Childhood Diarrhoea')
        for path in pathogens
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
                      'relative risk of diarrhoea death for ages 12 to 23 months'
                      ),
        'rr_diarr_death_age24to59mo':
            Parameter(Types.REAL,
                      'relative risk of diarrhoea death for ages 24 to 59 months'
                      ),
        'rr_diarr_death_if_duration_longer_than_13_days':
            Parameter(Types.REAL,
                      'relative risk of diarrhoea death if the duration episode is 13 days or longer'
                      ),
        'rr_diarr_death_dehydration':
            Parameter(Types.REAL, 'relative risk of diarrhoea death for cases with (some) dehyadration'
                      ),
        'rr_diarr_death_HIV':
            Parameter(Types.REAL, 'relative risk of diarrhoea death for HIV'
                      ),
        'rr_diarr_death_SAM':
            Parameter(Types.REAL, 'relative risk of diarrhoea death for severe acute malnutrition'
                      ),
        'days_onset_severe_dehydration_before_death':
            Parameter(Types.INT, 'number of days before a death (in the untreated case) that dehydration would be '
                                 'classified as severe and child ought to be classified as positive for the danger'
                                 'signs'),
        'mean_days_duration_with_rotavirus':
            Parameter(Types.INT, 'mean number of days duration with diarrhoea caused by rotavirus'),
        'mean_days_duration_with_shigella':
            Parameter(Types.INT, 'mean number of days duration with diarrhoea caused by shigella'),
        'mean_days_duration_with_adenovirus':
            Parameter(Types.INT, 'mean number of days duration with diarrhoea caused by adenovirus'),
        'mean_days_duration_with_cryptosporidium':
            Parameter(Types.INT, 'mean number of days duration with diarrhoea caused by cryptosporidium'),
        'mean_days_duration_with_campylobacter':
            Parameter(Types.INT, 'mean number of days duration with diarrhoea caused by campylobacter'),
        'mean_days_duration_with_ST-ETEC':
            Parameter(Types.INT, 'mean number of days duration with diarrhoea caused by ST-ETEC'),
        'mean_days_duration_with_sapovirus':
            Parameter(Types.INT, 'mean number of days duration with diarrhoea caused by sapovirus'),
        'mean_days_duration_with_norovirus':
            Parameter(Types.INT, 'mean number of days duration with diarrhoea caused by norovirus'),
        'mean_days_duration_with_astrovirus':
            Parameter(Types.INT, 'mean number of days duration with diarrhoea caused by astrovirus'),
        'mean_days_duration_with_tEPEC':
            Parameter(Types.INT, 'mean number of days duration with diarrhoea caused by tEPEC'),
        'prob_of_cure_given_Treatment_PlanA':
            Parameter(Types.REAL, 'probability of the person being cured if is provided with Treatment Plan A'),
        'prob_of_cure_given_Treatment_PlanB':
            Parameter(Types.REAL, 'probability of the person being cured if is provided with Treatment Plan B'),
        'prob_of_cure_given_Treatment_PlanC':
            Parameter(Types.REAL, 'probability of the person being cured if is provided with Treatment Plan C'),
        'max_number_of_days_for_onset_of_severe_dehydration_before_end_of_episode':
            Parameter(Types.INT, 'if severe dehydration occurs, it onsets a number of days before the end of the '
                                 'episode of diarrhoea; that number of days is chosen from a uniform distribution '
                                 'between 0 and this number'),
        'probability_of_severe_dehydration_if_some_dehydration':
            Parameter(Types.REAL, 'probability that someone with diarrhoea and some dehydration develops severe '
                                  'dehydration'),
        'min_days_duration_of_episode':
            Parameter(Types.INT, 'the shortest duration of any episode of diarrhoea'),
        'range_in_days_duration_of_episode':
            Parameter(Types.INT, 'the duration of an episode of diarrhoea is a uniform distribution around the mean '
                                 'with a range equal by this number.'),

        # TODO check the below parmaeters for those HSI are relevant
        'prob_of_cure_given_HSI_Diarrhoea_Severe_Persistent_Diarrhoea':
            Parameter(Types.REAL,
                      'probability of the person being cured if is provided with '
                      'HSI_Diarrhoea_Severe_Persistent_Diarrhoea.'),
        'prob_of_cure_given_HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea':
            Parameter(Types.REAL,
                      'probability of the person being cured if is provided with '
                      'HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea.'),
        'prob_of_cure_given_HSI_Diarrhoea_Dysentery':
            Parameter(Types.REAL,
                      'probability of the person being cured if is provided with HSI_Diarrhoea_Dysentery.'),
        'days_between_treatment_and_cure':
            Parameter(Types.INT, 'number of days between any treatment being given in an HSI and the cure occurring.')
    }

    PROPERTIES = {
        # ---- The pathogen which is caused the diarrhoea  ----
        'gi_ever_had_diarrhoea': Property(Types.BOOL,
                                          'Whether or not the person has ever had an episode of diarrhoea.'
                                          ),

        'gi_last_diarrhoea_pathogen': Property(Types.CATEGORICAL,
                                               'Attributable pathogen for the last episode of diarrhoea.'
                                               'not_applicable is used if the person has never had an episode of '
                                               'diarrhoea',
                                               categories=list(pathogens) + ['not_applicable']),

        # ---- Classification of the type of diarrhoea that is caused  ----
        'gi_last_diarrhoea_type': Property(Types.CATEGORICAL,
                                           'Type of the last episode of diarrhoea: either watery or bloody.'
                                           'not_applicable is used if the person has never had an episode of '
                                           'diarrhoea.',
                                           categories=['not_applicable',  # (never has had diarrhoea)
                                                       'watery',
                                                       'bloody']),

        # ---- Classification of severity of the dehydration caused ----
        'gi_last_diarrhoea_dehydration': Property(Types.CATEGORICAL,
                                                  'Severity of dehydration of last episode of diarrhoea.'
                                                  'not_applicable is used if the person has never had an episode of '
                                                  'diarrhoea',
                                                  categories=['not_applicable',  # (never has had diarrhoea)
                                                              'none',       # <-- this can be assigned at onset
                                                              'some',       # <-- this can be assigned at onset
                                                              'severe'      # <-- this may develop during the episode
                                                              ]),

        # ---- Internal variables to schedule onset and deaths due to diarrhoea  ----
        'gi_last_diarrhoea_date_of_onset': Property(Types.DATE, 'date of onset of last episode of diarrhoea. '
                                                                'pd.NaT if never had diarrhoea'),
        'gi_last_diarrhoea_duration': Property(Types.REAL, 'number of days of last episode of diarrhoea. '
                                                           'pd.nan if never had diarrhoea'),
        'gi_last_diarrhoea_recovered_date': Property(Types.DATE,
                                                     'date of recovery from last episode of diarrhoea. '
                                                     'pd.NaT if never had an episode or if the last episode caused'
                                                     'death.'
                                                     'This is scheduled when the episode is onset and may be revised'
                                                     'subsequently if a cure is enacted by a treatment.'),
        'gi_last_diarrhoea_death_date': Property(Types.DATE, 'date of death caused by last episode of diarrhoea.'
                                                             'pd.NaT if never had an episode or if the last episode did'
                                                             ' not cause death.'
                                                             'This is scheduled when the episode is onset and may be '
                                                             'revised to pd.NaT if a cure is enacted.'),
        'gi_last_diarrhoea_treatment_date': Property(Types.DATE,
                                                     'date of first treatment on the last episode of diarrhoea.'
                                                     'pd.NaT is never had an episode or if the last episode did not '
                                                     'receive treatment.'
                                                     'It is set to pd.NaT at onset of the episode and may be revised if'
                                                     'treatment is received.'),
        'gi_end_of_last_episode': Property(Types.DATE,
                                           'date on which the last episode of diarrhoea is resolved, (including '
                                           'allowing for the possibility that a cure is scheduled following onset). '
                                           'This is used to determine when a new episode can begin. '
                                           'This stops successive episodes interfering with one another.'),

        # ---- Temporary Variables: To be replaced with the properties of other modules ----
        'tmp_malnutrition': Property(Types.BOOL, 'temporary property - malnutrition status'),
        'tmp_exclusive_breastfeeding': Property(Types.BOOL, 'temporary property - exclusive breastfeeding upto 6 mo'),
        'tmp_continued_breastfeeding': Property(Types.BOOL, 'temporary property - continued breastfeeding 6mo-2years'),
        'tmp_hv_inf': Property(Types.BOOL, 'Temporary property - current HIV infection')
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

        self.consumables_used_in_hsi = dict()

        # dict to hold counters for the number of episodes of diarrhoea by pathogen-type and age-group
        # (0yrs, 1yrs, 2-4yrs)
        blank_counter = dict(zip(self.pathogens, [list() for _ in self.pathogens]))
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

        self.risk_of_death_diarrhoea = None
        self.mean_duration_in_days_of_diarrhoea = None
        self.mean_duration_in_days_of_diarrhoea_lookup = None
        self.prob_diarrhoea_is_watery = None

        # Store the symptoms that this module will use:
        self.symptoms = {
            'diarrhoea',
            'bloody_stool',
            'fever',
            'vomiting',
            'dehydration'  # NB. This is non-severe dehydration
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
        for param_name, param_type in self.PARAMETERS.items():
            assert param_name in p, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert isinstance(p[param_name],
                              param_type.python_type), f'Parameter "{param_name}" is not read in correctly from the ' \
                                                       f'resourcefile.'

        # Declare symptoms that this module will cause and which are not included in the generic symptoms:
        generic_symptoms = self.sim.modules['SymptomManager'].generic_symptoms
        for symptom_name in self.symptoms:
            if symptom_name not in generic_symptoms:
                self.sim.modules['SymptomManager'].register_symptom(
                    Symptom(name=symptom_name)  # (give non-generic symptom 'average' healthcare seeking)
                )

    def initialise_population(self, population):
        """
        Sets that there is no one with diarrhoea at initiation.
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals

        # ---- Key Current Status Classification Properties ----
        df.loc[df.is_alive, 'gi_ever_had_diarrhoea'] = False
        df.loc[df.is_alive, 'gi_last_diarrhoea_pathogen'] = 'not_applicable'
        df.loc[df.is_alive, 'gi_last_diarrhoea_type'] = 'not_applicable'
        df.loc[df.is_alive, 'gi_last_diarrhoea_dehydration'] = 'not_applicable'

        # ---- Internal values ----
        df.loc[df.is_alive, 'gi_last_diarrhoea_date_of_onset'] = pd.NaT
        df.loc[df.is_alive, 'gi_last_diarrhoea_duration'] = np.nan
        df.loc[df.is_alive, 'gi_last_diarrhoea_recovered_date'] = pd.NaT
        df.loc[df.is_alive, 'gi_last_diarrhoea_death_date'] = pd.NaT
        df.loc[df.is_alive, 'gi_last_diarrhoea_treatment_date'] = pd.NaT
        df.loc[df.is_alive, 'gi_end_of_last_episode'] = pd.NaT

        # ---- Temporary values ----
        df.loc[df.is_alive, 'tmp_malnutrition'] = False
        df.loc[df.is_alive, 'tmp_exclusive_breastfeeding'] = False
        df.loc[df.is_alive, 'tmp_continued_breastfeeding'] = False

    def initialise_simulation(self, sim):
        """Prepares for simulation:
        * Schedules the main polling event
        * Schedules the main logging event
        * Establishes the linear models and other data structures using the parameters that have been read-in
        * Store the consumables that are required in each of the HSI
        """

        # Schedule the main polling event (to first occur immediately)
        sim.schedule_event(DiarrhoeaPollingEvent(self), sim.date)

        # Schedule the main logging event (to first occur in one year)
        sim.schedule_event(DiarrhoeaLoggingEvent(self), sim.date + DateOffset(years=1))

        # Get DALY weights
        if 'HealthBurden' in self.sim.modules.keys():
            get_daly_weight = self.sim.modules['HealthBurden'].get_daly_weight
            self.daly_wts['mild_diarrhoea'] = get_daly_weight(sequlae_code=32)
            self.daly_wts['moderate_diarrhoea'] = get_daly_weight(sequlae_code=35)
            self.daly_wts['severe_diarrhoea'] = get_daly_weight(sequlae_code=34)

        # --------------------------------------------------------------------------------------------
        # Make a dict to hold the equations that govern the probability that a person acquires diarrhoea
        # that is caused (primarily) by a pathogen
        p = self.parameters

        def make_scaled_linear_model(patho):
            """Makes the unscaled linear model with default intercept of 1. Calculates the mean incidents rate for
            0-year-olds and then creates a new linear model with adjusted intercept so incidents in 0-year-olds
            matches the specified value in the model when averaged across the population
            """
            def make_linear_model(patho, intercept=1.0):
                base_inc_rate = f'base_inc_rate_diarrhoea_by_{patho}'
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
                    Predictor('li_no_access_handwashing').when(False, p['rr_diarrhoea_HHhandwashing']),
                    Predictor('li_no_clean_drinking_water').when(False, p['rr_diarrhoea_clean_water']),
                    Predictor('li_unimproved_sanitation').when(False, p['rr_diarrhoea_improved_sanitation']),
                    Predictor('tmp_hv_inf').when(True, p['rr_diarrhoea_HIV']),
                    Predictor('tmp_malnutrition').when(True, p['rr_diarrhoea_SAM']),
                    Predictor('tmp_exclusive_breastfeeding').when(False, p['rr_diarrhoea_excl_breast'])
                )

            df = self.sim.population.props
            unscaled_lm = make_linear_model(patho)
            target_mean = p[f'base_inc_rate_diarrhoea_by_{patho}'][0]
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 0)]).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean) \
                if (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else 1.0
            scaled_lm = make_linear_model(patho, intercept=scaled_intercept)
            # check by applying the model to mean incidence of 0-year-olds
            if (df.is_alive & (df.age_years == 0)).sum() > 0:
                assert (target_mean - scaled_lm.predict(df.loc[df.is_alive & (df.age_years == 0)]).mean()) < 1e-10
            return scaled_lm

        for pathogen in Diarrhoea.pathogens:
            self.incidence_equations_by_pathogen[pathogen] = make_scaled_linear_model(pathogen)

        # --------------------------------------------------------------------------------------------
        # Make a dict containing the probability of symptoms onset given acquisition of diarrhoea caused
        # by a particular pathogen.
        # Note that the type
        def make_symptom_probs(patho):
            return {
                'diarrhoea': 1.0,
                'bloody_stool': 1 - p[f'proportion_AWD_by_{patho}'],
                'fever': p[f'fever_by_{patho}'],
                'vomiting': p[f'vomiting_by_{patho}'],
                'dehydration': p[f'dehydration_by_{patho}'],
            }

        for pathogen in Diarrhoea.pathogens:
            self.prob_symptoms[pathogen] = make_symptom_probs(pathogen)

        # Check that each pathogen has a risk of developing each symptom
        assert set(self.pathogens) == set(self.prob_symptoms.keys())

        assert all(
            [
                set(self.symptoms) == set(self.prob_symptoms[pathogen].keys())
                for pathogen in self.prob_symptoms.keys()
            ]
        )

        # --------------------------------------------------------------------------------------------
        # Create the linear model for the risk of dying due to diarrhoea
        self.risk_of_death_diarrhoea = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1.0,
            Predictor('gi_last_diarrhoea_type', conditions_are_mutually_exclusive=True)
            .when('watery', p['case_fatality_rate_AWD'])
            .when('bloody', p['case_fatality_rate_dysentery']),
            Predictor('gi_last_diarrhoea_duration').when('>13', p['rr_diarr_death_if_duration_longer_than_13_days']),
            Predictor('gi_last_diarrhoea_dehydration').when('some', p['rr_diarr_death_dehydration']),
            Predictor('age_years', conditions_are_mutually_exclusive=True)
            .when('.between(1, 2, inclusive="left")', p['rr_diarr_death_age12to23mo'])
            .when('.between(2, 4, inclusive="left")', p['rr_diarr_death_age24to59mo'])
            .otherwise(0.0),
            Predictor('tmp_hv_inf').when(True, p['rr_diarrhoea_HIV']),
            Predictor('tmp_malnutrition').when(True, p['rr_diarrhoea_SAM'])
        )

        # --------------------------------------------------------------------------------------------
        # Create the linear model for the duration of the episode of diarrhoea
        self.mean_duration_in_days_of_diarrhoea = LinearModel(
            LinearModelType.ADDITIVE,
            0.0,
            Predictor(
                'gi_last_diarrhoea_pathogen',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
            .when('rotavirus', p['mean_days_duration_with_rotavirus'])
            .when('shigella', p['mean_days_duration_with_shigella'])
            .when('adenovirus', p['mean_days_duration_with_adenovirus'])
            .when('cryptosporidium', p['mean_days_duration_with_cryptosporidium'])
            .when('campylobacter', p['mean_days_duration_with_campylobacter'])
            .when('ST-ETEC', p['mean_days_duration_with_ST-ETEC'])
            .when('sapovirus', p['mean_days_duration_with_sapovirus'])
            .when('norovirus', p['mean_days_duration_with_norovirus'])
            .when('astrovirus', p['mean_days_duration_with_astrovirus'])
            .when('tEPEC', p['mean_days_duration_with_tEPEC'])
            .when('not_applicable', 0)
        )

        self.mean_duration_in_days_of_diarrhoea_lookup = {
            'rotavirus': p['mean_days_duration_with_rotavirus'],
            'shigella': p['mean_days_duration_with_shigella'],
            'adenovirus': p['mean_days_duration_with_adenovirus'],
            'cryptosporidium': p['mean_days_duration_with_cryptosporidium'],
            'campylobacter': p['mean_days_duration_with_campylobacter'],
            'ST-ETEC': p['mean_days_duration_with_ST-ETEC'],
            'sapovirus': p['mean_days_duration_with_sapovirus'],
            'norovirus': p['mean_days_duration_with_norovirus'],
            'astrovirus': p['mean_days_duration_with_astrovirus'],
            'tEPEC': p['mean_days_duration_with_tEPEC'],
        }

        # --------------------------------------------------------------------------------------------
        # Create the linear model for the probability that the diarrhoea is 'watery' (rather than 'bloody')
        self.prob_diarrhoea_is_watery = LinearModel(
            LinearModelType.ADDITIVE,
            0.0,
            Predictor(
                'gi_last_diarrhoea_pathogen',
                conditions_are_mutually_exclusive=True,
                conditions_are_exhaustive=True,
            )
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
            .when('not_applicable', 0)
        )

        # --------------------------------------------------------------------------------------------
        # Look-up and store the consumables that are required for each HSI
        self.look_up_consumables()

    def on_birth(self, mother_id, child_id):
        """
        On birth, all children will have no diarrhoea
        """
        df = self.sim.population.props

        # ---- Key Current Status Classification Properties ----
        df.at[child_id, 'gi_ever_had_diarrhoea'] = False
        df.at[child_id, 'gi_last_diarrhoea_pathogen'] = 'not_applicable'
        df.at[child_id, 'gi_last_diarrhoea_type'] = 'not_applicable'
        df.at[child_id, 'gi_last_diarrhoea_dehydration'] = 'not_applicable'

        # ---- Internal values ----
        df.at[child_id, 'gi_last_diarrhoea_date_of_onset'] = pd.NaT
        df.at[child_id, 'gi_last_diarrhoea_recovered_date'] = pd.NaT
        df.at[child_id, 'gi_last_diarrhoea_death_date'] = pd.NaT
        df.at[child_id, 'gi_last_diarrhoea_treatment_date'] = pd.NaT
        df.at[child_id, 'gi_end_of_last_episode'] = pd.NaT

        # ---- Temporary values ----
        df.at[child_id, 'tmp_malnutrition'] = False
        df.at[child_id, 'tmp_exclusive_breastfeeding'] = False
        df.at[child_id, 'tmp_continued_breastfeeding'] = False

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        pass

    def report_daly_values(self):
        """
        This returns DALYS values relating to the current status of persons.
        Assumes that the current DALY loading is linked to currently having the symptom of diarrhoea and the status of
        dehyrdration.
        """
        df = self.sim.population.props
        who_has_symptoms = self.sim.modules['SymptomManager'].who_has

        total_daly_values = pd.Series(data=0.0, index=df.index[df.is_alive])

        total_daly_values.loc[who_has_symptoms('diarrhoea')] = self.daly_wts['mild_diarrhoea']
        total_daly_values.loc[who_has_symptoms(['diarrhoea', 'dehydration'])] = self.daly_wts['moderate_diarrhoea']
        total_daly_values.loc[
            df.is_alive & (df.gi_last_diarrhoea_dehydration == 'severe')
        ] = self.daly_wts['severe_diarrhoea']

        # Split out by pathogen that causes the diarrhoea
        dummies_for_pathogen = pd.get_dummies(df.loc[total_daly_values.index,
                                                     'gi_last_diarrhoea_pathogen'],
                                              dtype='float').drop(columns='not_applicable')
        daly_values_by_pathogen = dummies_for_pathogen.mul(total_daly_values, axis=0)

        # return with columns labelled to match the declare CAUSES_OF_DISABILITY
        return daly_values_by_pathogen.add_prefix('Diarrhoea_')

    def look_up_consumables(self):
        """Look up and store the consumables used in each of the HSI."""

        def get_code(package=None, item=None):
            consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
            if package is not None:
                condition = consumables['Intervention_Pkg'] == package
                column = 'Intervention_Pkg_Code'
            else:
                condition = consumables['Items'] == item
                column = 'Item_Code'
            return pd.unique(consumables.loc[condition, column])[0]

        def add_consumable(_event, _package, _item):
            self.consumables_used_in_hsi[_event] = {
                'Intervention_Package_Code': _package,
                'Item_Code': _item
            }

        ors_code = get_code(package='ORS')
        severe_diarrhoea_code = get_code(package='Treatment of severe diarrhea')
        zinc_tablet_code = get_code(item='Zinc, tablet, 20 mg')
        zinc_under_6m_code = get_code(package='Zinc for Children 0-6 months')
        zinc_over_6m_code = get_code(package='Zinc for Children 6-59 months')
        antibiotics_code = get_code(package='Antibiotics for treatment of dysentery')
        cipro_code = get_code(item='Ciprofloxacin 250mg_100_CMST')

        # -- Assemble the footprints for each HSI:
        add_consumable('HSI_Diarrhoea_Treatment_PlanA', {ors_code: 1}, {})
        add_consumable('HSI_Diarrhoea_Treatment_PlanB', {ors_code: 1}, {})
        add_consumable('HSI_Diarrhoea_Treatment_PlanC', {severe_diarrhoea_code: 1}, {})
        add_consumable('HSI_Diarrhoea_Severe_Persistent_Diarrhoea', {ors_code: 1}, {})
        add_consumable('HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea_Under6mo',
                       {zinc_under_6m_code: 1},
                       {zinc_tablet_code: 5})
        add_consumable('HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea_6moPlus',
                       {zinc_over_6m_code: 1},
                       {zinc_tablet_code: 5})
        add_consumable('HSI_Diarrhoea_Dysentery', {antibiotics_code: 1}, {cipro_code: 6})

    def do_treatment(self, person_id, prob_of_cure):
        """Helper function that enacts the effects of a treatment to diarrhoea caused by a pathogen.
        It will only do something if the diarrhoea is caused by a pathogen (this module). It will not allow any effect
         if the diarrhoea is caused by another module.

        * Log the treatment date
        * Prevents this episode of diarrhoea from causing a death
        * Schedules the cure event, at which symptoms are alleviated.

        """
        df = self.sim.population.props
        person = df.loc[person_id]

        if not person.is_alive:
            return

        # Do nothing if the diarrhoea has not been caused by a pathogen or has otherwise resolved already
        if not (
            (person.gi_last_diarrhoea_pathogen != 'not_applicable') &
            (person.gi_last_diarrhoea_date_of_onset <= self.sim.date <= person.gi_end_of_last_episode)
        ) or (
            'Diarrhoea' not in self.sim.modules['SymptomManager'].causes_of(person_id, 'diarrhoea')
        ):
            return

        # Log that the treatment is provided:
        df.at[person_id, 'gi_last_diarrhoea_treatment_date'] = self.sim.date

        # Determine if the treatment is effective
        if prob_of_cure > self.rng.rand():
            # If treatment is successful: cancel death and schedule cure event
            self.cancel_death_date(person_id)
            self.sim.schedule_event(DiarrhoeaCureEvent(self, person_id),
                                    self.sim.date + DateOffset(
                                        days=self.parameters['days_between_treatment_and_cure']
                                    ))

    def cancel_death_date(self, person_id):
        """
        Cancels a scheduled date of death due to diarrhoea for a person. This is called prior to the scheduling the
        CureEvent to prevent deaths happening in the time between a treatment being given and the cure event occurring.

        :param person_id:
        :return:
        """
        df = self.sim.population.props
        df.at[person_id, 'gi_last_diarrhoea_death_date'] = pd.NaT

# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
#
# ---------------------------------------------------------------------------------------------------------


class DiarrhoeaPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This is the main event that runs the acquisition of pathogens that cause Diarrhoea.
    It determines who is infected and schedules individual IncidentCase events to represent onset.

    A known issue is that diarrhoea events are scheduled based on the risk of current age but occur a short time
     later when the children will be slightly older. This means that when comparing the model output with data, the
     model slightly under-represents incidence among younger age-groups and over-represents incidence among older
     age-groups. This is a small effect when the frequency of the polling event is high.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))
        # NB. The frequency of the occurrences of this event can be edited safely.

    def apply(self, population):
        df = population.props
        rng = self.module.rng
        m = self.module

        # Those susceptible are children that do not currently have an episode (never had an episode or last episode
        # resolved) and who do not have diarrhoea as a 'spurious symptom' of diarrhoea already.
        mask_could_get_new_diarrhoea_episode = (
            df.is_alive &
            (df.age_years < 5) &
            ((df.gi_end_of_last_episode < self.sim.date) | pd.isnull(df.gi_end_of_last_episode)) &
            ~df.index.isin(self.sim.modules['SymptomManager'].who_has('diarrhoea'))
        )

        # Compute the incidence rate for each person getting diarrhoea
        inc_of_acquiring_pathogen = pd.DataFrame(index=df.loc[mask_could_get_new_diarrhoea_episode].index)
        for pathogen in m.pathogens:
            inc_of_acquiring_pathogen[pathogen] = m.incidence_equations_by_pathogen[pathogen] \
                .predict(df.loc[mask_could_get_new_diarrhoea_episode])

        # Convert the incidence rates into risk of an event occurring before the next polling event
        fraction_of_a_year_until_next_polling_event = \
            (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'Y')
        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')
        probs_of_aquiring_pathogen = 1 - np.exp(
            -inc_of_acquiring_pathogen * fraction_of_a_year_until_next_polling_event
        )

        # Compute the probability of getting 'any' pathogen:
        # (Assumes that pathogens are mutually exclusive. Prevents probability being greater than 1.0.
        prob_of_acquiring_any_pathogen = probs_of_aquiring_pathogen.sum(axis=1).clip(upper=1.0)
        assert all(prob_of_acquiring_any_pathogen <= 1.0)

        # Determine which persons will acquire any pathogen:
        person_id_that_acquire_pathogen = prob_of_acquiring_any_pathogen.index[
            rng.rand(len(prob_of_acquiring_any_pathogen)) < prob_of_acquiring_any_pathogen
            ]

        # Determine which pathogen each person will acquire (among those who will get a pathogen)
        # and create the event for the onset of new infection:
        for person_id in person_id_that_acquire_pathogen:
            # Allocate a pathogen to the person
            p_by_pathogen = probs_of_aquiring_pathogen.loc[person_id].values
            normalised_p_by_pathogen = p_by_pathogen / sum(p_by_pathogen)  # don't understand this normalised
            pathogen = rng.choice(probs_of_aquiring_pathogen.columns,
                                  p=normalised_p_by_pathogen)

            # Allocate a date of onset diarrhoea
            date_onset = self.sim.date + DateOffset(days=rng.randint(0, days_until_next_polling_event))

            # Create the event for the onset of infection
            self.sim.schedule_event(
                event=DiarrhoeaIncidentCase(module=self.module, person_id=person_id, pathogen=pathogen),
                date=date_onset
            )


class DiarrhoeaIncidentCase(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of the infection that causes diarrhoea.
     * Refreshes all the properties so that they pertain to this current episode of diarrhoea
     * Imposes the symptoms
     * Schedules relevant natural history event {(DiarrhoeaSevereDehydrationEvent) and
       (either DiarrhoeaNaturalRecoveryEvent or DiarrhoeaDeathEvent)}
     * Updates a counter for incident cases
    """
    AGE_GROUPS = {0: '0y', 1: '1y', 2: '2-4y', 3: '2-4y', 4: '2-4y'}

    def __init__(self, module, person_id, pathogen):
        super().__init__(module, person_id=person_id)
        self.pathogen = pathogen

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters
        rng = m.rng

        person = df.loc[person_id]

        # The event should not run if the person is not currently alive
        if not person.is_alive:
            return

        # ----------------------- Determine duration for this episode ----------------------
        mean_duration = m.mean_duration_in_days_of_diarrhoea_lookup[self.pathogen]
        half_range = p['range_in_days_duration_of_episode'] / 2
        actual_duration = mean_duration + rng.randint(-half_range, half_range)
        duration_in_days_of_episode = int(max(p['min_days_duration_of_episode'], actual_duration))

        date_of_outcome = self.sim.date + DateOffset(days=duration_in_days_of_episode)

        # Collect the updated properties of this person
        props_new = {
            'gi_ever_had_diarrhoea': True,
            'gi_last_diarrhoea_pathogen': self.pathogen,
            'gi_last_diarrhoea_date_of_onset': self.sim.date,
            'gi_last_diarrhoea_duration': duration_in_days_of_episode,
            'gi_last_diarrhoea_treatment_date': pd.NaT,
            'gi_last_diarrhoea_type': 'watery',
            'gi_last_diarrhoea_dehydration': 'none',
            'gi_last_diarrhoea_recovered_date': pd.NaT,
            'gi_last_diarrhoea_death_date': pd.NaT
        }

        # ----------------------- Determine symptoms for this episode ----------------------
        possible_symptoms_for_this_pathogen = m.prob_symptoms[self.pathogen]
        for symptom, prob in possible_symptoms_for_this_pathogen.items():
            if rng.random_sample() < prob:
                self.sim.modules['SymptomManager'].change_symptom(
                    person_id=person_id,
                    symptom_string=symptom,
                    add_or_remove='+',
                    disease_module=self.module
                )
                if symptom == 'bloody_stool':
                    props_new['gi_last_diarrhoea_type'] = 'bloody'
                elif symptom == 'dehydration':
                    props_new['gi_last_diarrhoea_dehydration'] = 'some'

        # ----------------------- Determine the progress to severe dehydration for this episode ----------------------
        # Progress to severe dehydration may or may not happen. If it does, it occurs some number of days before
        # the resolution of the episode. It does not affect the risk of death or anything else in the natural history -
        # - it can just be considered to be a marker of the how close the individual is to the episode ending (which may
        # or may not result in death).
        if props_new['gi_last_diarrhoea_dehydration'] == 'some':
            if rng.random_sample() < p['probability_of_severe_dehydration_if_some_dehydration']:
                # schedule the onset of severe dehydration:
                days = rng.randint(0, p['max_number_of_days_for_onset_of_severe_dehydration_before_end_of_episode'])
                date_of_onset_severe_dehydration = max(self.sim.date, date_of_outcome - DateOffset(days=days))
                self.sim.schedule_event(
                    event=DiarrhoeaSevereDehydrationEvent(m, person_id),
                    date=date_of_onset_severe_dehydration,
                )

        # ----------------------- Determine outcome (recovery or death) of this episode ----------------------
        # Determine if episode will result in death
        prob_death = 1.0
        if person['age_exact_years'] >= 4:
            prob_death = 0
        else:
            if person['gi_last_diarrhoea_type'] == 'watery':
                prob_death *= p['case_fatality_rate_AWD']
            if person['gi_last_diarrhoea_type'] == 'bloody':
                prob_death *= p['case_fatality_rate_dysentery']
            if person['gi_last_diarrhoea_duration'] > 13:
                prob_death *= p['rr_diarr_death_if_duration_longer_than_13_days']
            if person['gi_last_diarrhoea_dehydration'] == 'some':
                prob_death *= p['rr_diarr_death_dehydration']
            if 1 <= person['age_exact_years'] < 2:
                prob_death *= p['rr_diarr_death_age12to23mo']
            elif 2 <= person['age_exact_years'] < 4:
                prob_death *= p['rr_diarr_death_age24to59mo']
            if person['tmp_hv_inf']:
                prob_death *= p['rr_diarrhoea_HIV']
            if person['tmp_malnutrition']:
                prob_death *= p['rr_diarrhoea_SAM']

        if rng.random_sample() < prob_death:
            props_new['gi_last_diarrhoea_death_date'] = date_of_outcome
            self.sim.schedule_event(DiarrhoeaDeathEvent(m, person_id), date_of_outcome)
        else:
            props_new['gi_last_diarrhoea_recovered_date'] = date_of_outcome
            self.sim.schedule_event(DiarrhoeaNaturalRecoveryEvent(m, person_id), date_of_outcome)

        # Record 'episode end' data. This the date when this episode ends. It is the last possible data that any HSI
        # could affect this episode.
        props_new['gi_end_of_last_episode'] = date_of_outcome + DateOffset(days=p['days_between_treatment_and_cure'])

        # -------------------------------------------------------------------------------------------
        # Add this incident case to the tracker
        age_group = DiarrhoeaIncidentCase.AGE_GROUPS.get(person.age_years, '5+y')
        m.incident_case_tracker[age_group][self.pathogen].append(self.sim.date)
        # -------------------------------------------------------------------------------------------

        # Update the entry in the population dataframe
        df.loc[person_id, props_new.keys()] = props_new.values()


class DiarrhoeaNaturalRecoveryEvent(Event, IndividualScopeEventMixin):
    """
    #This is the Natural Recovery event. It is part of the natural history and represents the end of an episode of
    Diarrhoea.
    It does the following:
        * resolves all symptoms caused by diarrhoea
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]

        # The event should not run if the person is not currently alive
        if not person.is_alive:
            return

        # Do nothing if the recovery is not expected for today
        # (Because the person has already recovered, through being cured).
        if not (self.sim.date == person.gi_last_diarrhoea_recovered_date):
            return

        # Confirm that this is event is occurring during a current episode of diarrhoea
        assert person.gi_last_diarrhoea_date_of_onset <= self.sim.date <= person.gi_end_of_last_episode

        # Check that the person is not scheduled to die in this episode
        assert pd.isnull(person.gi_last_diarrhoea_death_date)

        # Resolve all the symptoms immediately
        df.at[person_id, 'gi_last_diarrhoea_dehydration'] = 'none'
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id,
                                                          disease_module=self.sim.modules['Diarrhoea'])


class DiarrhoeaSevereDehydrationEvent(Event, IndividualScopeEventMixin):
    """
    #This is the Severe Dehydration. It is part of the natural history and represents a change in the status of the
    person's level of dehydration.
    It does the following:
        * changes the property 'gi_last_diarrhoea_dehydration' to severe
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        # The event should not run if the person is not currently alive
        if not df.at[person_id, 'is_alive']:
            return

        # Confirm that this is event is occurring during a current episode of diarrhoea
        assert (
            (df.at[person_id, 'gi_last_diarrhoea_date_of_onset']) <=
            self.sim.date <=
            (df.at[person_id, 'gi_end_of_last_episode'])
        )

        # Do nothing if the person has recovered already (after having been cured)
        if df.at[person_id, 'gi_last_diarrhoea_recovered_date'] <= self.sim.date:
            return

        # Change the status:
        df.at[person_id, 'gi_last_diarrhoea_dehydration'] = 'severe'


class DiarrhoeaDeathEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the death of someone that is caused by the infection with a pathogen that causes diarrhoea.
    The person dies if the 'diarrhoea_death_date' that was set at onset of the episode is equal to the current date.
    (It will have been reset to pd.NaT if there has been a treatment in the intervening time).
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # The event should not run if the person is not currently alive
        if not df.at[person_id, 'is_alive']:
            return

        # Confirm that this is event is occurring during a current episode of diarrhoea
        assert (
            (df.at[person_id, 'gi_last_diarrhoea_date_of_onset']) <=
            self.sim.date <=
            (df.at[person_id, 'gi_end_of_last_episode'])
        )

        # Check if person should still die of diarrhoea:
        if (
            (df.at[person_id, 'gi_last_diarrhoea_death_date'] == self.sim.date) and
            pd.isnull(df.at[person_id, 'gi_last_diarrhoea_recovered_date'])
        ):
            # Implement the death:
            self.sim.schedule_event(
                demography.InstantaneousDeath(
                    self.module,
                    person_id,
                    cause='Diarrhoea_' + df.at[person_id, 'gi_last_diarrhoea_pathogen']
                ),
                self.sim.date)


class DiarrhoeaCureEvent(Event, IndividualScopeEventMixin):
    """
    This is the cure event. It is scheduled by an HSI treatment event.
    It enacts the actual "cure" of the person that is caused (after some delay) by the treatment administered.
    It does the following:
        * Sets the date of recovery to today's date
        * Resolves all symptoms caused by diarrhoea

    NB. This is the event that would be called if another module has caused the symptom of diarrhoea and care is sought.

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]

        # The event should not run if the person is not currently alive
        if not person.is_alive:
            return

        # Confirm that this is event is occurring during a current episode of diarrhoea
        if ~(
            person.gi_last_diarrhoea_date_of_onset <= self.sim.date <= person.gi_end_of_last_episode
        )\
            or (
            'Diarrhoea' not in self.sim.modules['SymptomManager'].causes_of(person_id, 'diarrhoea')
        ):
            # If not, then the event has been caused by another cause of diarrhoea (which may has resolved by now)
            return

        # Cure should not happen if the person has already recovered
        if person.gi_last_diarrhoea_recovered_date <= self.sim.date:
            return

        # Stop the person from dying of Diarrhoea (if they were going to die) and record date of recovery
        df.at[person_id, 'gi_last_diarrhoea_recovered_date'] = self.sim.date

        # Resolve all the symptoms immediately
        df.at[person_id, 'gi_last_diarrhoea_dehydration'] = 'none'
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id,
                                                          disease_module=self.sim.modules['Diarrhoea'])


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

        logger.info(key='incidence_count_by_pathogen', data=counts)

        # Reset the counters and the date_last_run
        self.module.incident_case_tracker = copy.deepcopy(self.module.incident_case_tracker_blank)
        self.date_last_run = self.sim.date


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------

class HSI_Diarrhoea_Treatment_PlanA(HSI_Event, IndividualScopeEventMixin):
    """
    This is a treatment for uncomplicated diarrhoea administered at outpatient setting through IMCI.
    "PLAN A": for children no dehydration

    NB. This will be called when a child presents with Diarrhoea that is caused by another module/Symptom Manager.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient
        self.TREATMENT_ID = 'Diarrhoea_Treatment_PlanA'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug(key='debug', data='Provide Treatment Plan A for uncomplicated Diarrhoea')

        # Stop the person from dying of Diarrhoea (if they were going to die)
        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            return

        # Get consumables required
        cons_footprint = self.module.consumables_used_in_hsi['HSI_Diarrhoea_Treatment_PlanA']
        # give the mother 2 packets of ORS
        # give zinc (2 mo up to 5 years) - <6 mo 1/2 tablet (10mg) for 10 days, >6 mo 1 tab (20mg) for 10 days
        # follow up in 5 days if not improving
        # continue feeding

        rtn_from_health_system = self.sim.modules['HealthSystem'].request_consumables(self, cons_footprint)
        cons_available = all(
            rtn_from_health_system['Intervention_Package_Code'].values()
        ) and all(
            rtn_from_health_system['Item_Code'].values()
        )
        # todo use self.get_all_consumables when this is updated in master

        if cons_available:
            self.module.do_treatment(
                person_id=person_id,
                prob_of_cure=self.module.parameters['prob_of_cure_given_Treatment_PlanA']
            )


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
        logger.debug(key='debug', data='Provide Treatment Plan B for Diarrhoea with some non-severe dehydration')

        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            return

        # Get consumables required
        cons_footprint = self.module.consumables_used_in_hsi['HSI_Diarrhoea_Treatment_PlanB']
        # Give ORS for the first 4 hours and reassess. todo - this is not happening curently

        rtn_from_health_system = self.sim.modules['HealthSystem'].request_consumables(self, cons_footprint)
        cons_available = all(
            rtn_from_health_system['Intervention_Package_Code'].values()
        ) and all(
            rtn_from_health_system['Item_Code'].values()
        )
        # todo use self.get_all_consumables when this is updated in master

        if cons_available:
            self.module.do_treatment(
                person_id=person_id,
                prob_of_cure=self.module.parameters['prob_of_cure_given_Treatment_PlanB']
            )


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
        logger.debug(key='debug', data='Provide Treatment Plan C for Diarrhoea with severe dehydration')

        # Stop the person from dying of Diarrhoea (if they were going to die)
        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            return

        cons_footprint = self.module.consumables_used_in_hsi['HSI_Diarrhoea_Treatment_PlanC']

        # Request the treatment
        rtn_from_health_system = self.sim.modules['HealthSystem'].request_consumables(self, cons_footprint)
        cons_available = all(
            rtn_from_health_system['Intervention_Package_Code'].values()
        ) and all(
            rtn_from_health_system['Item_Code'].values()
        )
        # todo use self.get_all_consumables when this is updated in master

        if cons_available:
            self.module.do_treatment(
                person_id=person_id,
                prob_of_cure=self.module.parameters['prob_of_cure_given_Treatment_PlanC']
            )


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
        logger.debug(key='debug', data='Provide the treatment for Diarrhoea')

        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            return

        # Get consumables
        # if bloody stool and fever in persistent diarroea - give Nalidixic Acid 50mg/kg divided in 4 doses per day for
        # 5 days if malnourished give cotrimoxazole 24mg/kg every 12 hours for 5 days and supplemental feeding and
        # supplements
        cons_footprint = self.module.consumables_used_in_hsi['HSI_Diarrhoea_Severe_Persistent_Diarrhoea']

        rtn_from_health_system = self.sim.modules['HealthSystem'].request_consumables(self, cons_footprint)
        cons_available = all(
            rtn_from_health_system['Intervention_Package_Code'].values()
        ) and all(
            rtn_from_health_system['Item_Code'].values()
        )
        # todo use self.get_all_consumables when this is updated in master

        if cons_available:
            self.module.do_treatment(
                person_id=person_id,
                prob_of_cure=self.module.parameters['prob_of_cure_given_HSI_Diarrhoea_Severe_Persistent_Diarrhoea']
            )


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
        logger.debug(key='debug', data='Provide the treatment for Diarrhoea')

        person = self.sim.population.props.loc[person_id]

        if not person.is_alive:
            return

        if person.age_exact_years < 0.5:
            cons_footprint = self.module.consumables_used_in_hsi[
                'HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea_Under6mo']
        else:
            cons_footprint = self.module.consumables_used_in_hsi[
                'HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea_6moPlus']

        # Request the treatment
        rtn_from_health_system = self.sim.modules['HealthSystem'].request_consumables(self, cons_footprint)
        cons_available = all(
            rtn_from_health_system['Intervention_Package_Code'].values()
        ) and all(
            rtn_from_health_system['Item_Code'].values()
        )
        # todo use self.get_all_consumables when this is updated in master

        if cons_available:
            self.module.do_treatment(
                person_id=person_id,
                prob_of_cure=self.module.parameters['prob_of_cure_given_HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea']
            )


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
        logger.debug(key='debug', data='Provide the treatment for Diarrhoea')

        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            return

        cons_footprint = self.module.consumables_used_in_hsi['HSI_Diarrhoea_Dysentery']

        # Get consumables required
        # <6 mo - 250mg 1/2 tab x2 daily for 3 days
        # >6 mo upto 5 yo - 250mg 1 tab x2 daily for 3 days
        # follow up in 3 days # todo - there are not follow-up events currently

        rtn_from_health_system = self.sim.modules['HealthSystem'].request_consumables(self, cons_footprint)
        cons_available = all(
            rtn_from_health_system['Intervention_Package_Code'].values()
        ) and all(
            rtn_from_health_system['Item_Code'].values()
        )
        # todo use self.get_all_consumables when this is updated in master

        if cons_available:
            self.module.do_treatment(
                person_id=person_id,
                prob_of_cure=self.module.parameters['prob_of_cure_given_HSI_Diarrhoea_Dysentery']
            )
