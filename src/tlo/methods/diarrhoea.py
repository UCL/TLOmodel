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

from tlo.methods.dxmanager import DxTest

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
        'ETEC',
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
        'base_inc_rate_diarrhoea_by_ETEC':
            Parameter(Types.LIST,
                      'incidence rate (per person-year) '
                      'of diarrhoea caused by ETEC in age groups 0-11, 12-23, 24-59 months'
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
        'rr_diarrhoea_untreated_HIV':
            Parameter(Types.REAL, 'relative rate of diarrhoea for HIV positive status'
                      ),
        'rr_diarrhoea_SAM':
            Parameter(Types.REAL, 'relative rate of diarrhoea for severe malnutrition'
                      ),
        'rr_diarrhoea_exclusive_vs_no_breastfeeding_<6mo':
            Parameter(Types.REAL, 'relative rate of diarrhoea for no breastfeeding in <6 months old, '
                                  'compared to exclusive breastfeeding'
                      ),
        'rr_diarrhoea_exclusive_vs_partial_breastfeeding_<6mo':
            Parameter(Types.REAL, 'relative rate of diarrhoea for partial breastfeeding in <6 months old, '
                                  'compared to exclusive breastfeeding'
                      ),
        'rr_diarrhoea_any_vs_no_breastfeeding_6_11mo':
            Parameter(Types.REAL, 'relative rate of diarrhoea for no breastfeeding in 6 months old to 1 years old, '
                                  'compared to any breastfeeding at this age group'
                      ),
        'rr_severe_diarrhoea_RV1':
            Parameter(Types.REAL, 'relative rate of severe diarrhoea for rotavirus vaccine'
                      ),
        'proportion_AWD_in_rotavirus':
            Parameter(Types.REAL, 'acute diarrhoea type caused in rotavirus-attributed diarrhoea'
                      ),
        'proportion_AWD_in_shigella':
            Parameter(Types.REAL, 'acute diarrhoea type caused in shigella-attributed diarrhoea'
                      ),
        'proportion_AWD_in_adenovirus':
            Parameter(Types.REAL, 'acute diarrhoea type caused in adenovirus-attributed diarrhoea'
                      ),
        'proportion_AWD_in_cryptosporidium':
            Parameter(Types.REAL, 'acute diarrhoea type caused in cryptosporidium-attributed diarrhoea'
                      ),
        'proportion_AWD_in_campylobacter':
            Parameter(Types.REAL, 'acute diarrhoea type caused in campylobacter-attributed diarrhoea'
                      ),
        'proportion_AWD_in_ETEC':
            Parameter(Types.REAL, 'acute diarrhoea type caused in ETEC-attributed diarrhoea'
                      ),
        'proportion_AWD_in_sapovirus':
            Parameter(Types.REAL, 'acute diarrhoea type caused in sapovirus-attributed diarrhoea'
                      ),
        'proportion_AWD_in_norovirus':
            Parameter(Types.REAL, 'acute diarrhoea type caused in norovirus-attributed diarrhoea'
                      ),
        'proportion_AWD_in_astrovirus':
            Parameter(Types.REAL, 'acute diarrhoea type caused in astrovirus-attributed diarrhoea'
                      ),
        'proportion_AWD_in_tEPEC':
            Parameter(Types.REAL, 'acute diarrhoea type in tEPEC-attributed diarrhoea'
                      ),
        'prob_fever_by_rotavirus':
            Parameter(Types.REAL, 'probability of fever caused by rotavirus'
                      ),
        'prob_fever_by_shigella':
            Parameter(Types.REAL, 'probability of fever caused by shigella'
                      ),
        'prob_fever_by_adenovirus':
            Parameter(Types.REAL, 'probability of fever caused by adenovirus'
                      ),
        'prob_fever_by_cryptosporidium':
            Parameter(Types.REAL, 'probability of fever caused by cryptosporidium'
                      ),
        'prob_fever_by_campylobacter':
            Parameter(Types.REAL, 'probability of fever caused by campylobacter'
                      ),
        'prob_fever_by_ETEC':
            Parameter(Types.REAL, 'probability of fever caused by ETEC'
                      ),
        'prob_fever_by_sapovirus':
            Parameter(Types.REAL, 'probability of fever caused by sapovirus'
                      ),
        'prob_fever_by_norovirus':
            Parameter(Types.REAL, 'probability of fever caused by norovirus'
                      ),
        'prob_fever_by_astrovirus':
            Parameter(Types.REAL, 'probability of fever caused by astrovirus'
                      ),
        'prob_fever_by_tEPEC':
            Parameter(Types.REAL, 'probability of fever caused by tEPEC'
                      ),
        'prob_vomiting_by_rotavirus':
            Parameter(Types.REAL, 'probability of vomiting caused by rotavirus'
                      ),
        'prob_vomiting_by_shigella':
            Parameter(Types.REAL, 'probability of vomiting caused by shigella'
                      ),
        'prob_vomiting_by_adenovirus':
            Parameter(Types.REAL, 'probability of vomiting caused by adenovirus'
                      ),
        'prob_vomiting_by_cryptosporidium':
            Parameter(Types.REAL, 'probability of vomiting caused by cryptosporidium'
                      ),
        'prob_vomiting_by_campylobacter':
            Parameter(Types.REAL, 'probability of vomiting caused by campylobacter'
                      ),
        'prob_vomiting_by_ETEC':
            Parameter(Types.REAL, 'probability of vomiting caused by ETEC'
                      ),
        'prob_vomiting_by_sapovirus':
            Parameter(Types.REAL, 'probability of vomiting caused by sapovirus'
                      ),
        'prob_vomiting_by_norovirus':
            Parameter(Types.REAL, 'probability of vomiting caused by norovirus'
                      ),
        'prob_vomiting_by_astrovirus':
            Parameter(Types.REAL, 'probability of vomiting caused by astrovirus'
                      ),
        'prob_vomiting_by_tEPEC':
            Parameter(Types.REAL, 'probability of vomiting caused by tEPEC'
                      ),
        'prob_dehydration_by_rotavirus':
            Parameter(Types.REAL, 'probability of any dehydration caused by rotavirus'
                      ),
        'prob_dehydration_by_shigella':
            Parameter(Types.REAL, 'probability of any dehydration caused by shigella'
                      ),
        'prob_dehydration_by_adenovirus':
            Parameter(Types.REAL, 'probability of any dehydration caused by adenovirus'
                      ),
        'prob_dehydration_by_cryptosporidium':
            Parameter(Types.REAL, 'probability of any dehydration caused by cryptosporidium'
                      ),
        'prob_dehydration_by_campylobacter':
            Parameter(Types.REAL, 'probability of any dehydration caused by campylobacter'
                      ),
        'prob_dehydration_by_ETEC':
            Parameter(Types.REAL, 'probability of any dehydration caused by ETEC'
                      ),
        'prob_dehydration_by_sapovirus':
            Parameter(Types.REAL, 'probability of any dehydration caused by sapovirus'
                      ),
        'prob_dehydration_by_norovirus':
            Parameter(Types.REAL, 'probability of any dehydration caused by norovirus'
                      ),
        'prob_dehydration_by_astrovirus':
            Parameter(Types.REAL, 'probability of any dehydration caused by astrovirus'
                      ),
        'prob_dehydration_by_tEPEC':
            Parameter(Types.REAL, 'probability of any dehydration caused by tEPEC'
                      ),
        'prob_prolonged_diarr_rotavirus':
            Parameter(Types.REAL, 'probability of prolonged episode in rotavirus-attributed diarrhoea'
                      ),
        'prob_prolonged_diarr_shigella':
            Parameter(Types.REAL, 'probability of prolonged episode by shigella-attributed diarrhoea'
                      ),
        'prob_prolonged_diarr_adenovirus':
            Parameter(Types.REAL, 'probability of prolonged episode by adenovirus-attributed diarrhoea'
                      ),
        'prob_prolonged_diarr_cryptosporidium':
            Parameter(Types.REAL, 'probability of prolonged episode by cryptosporidium-attributed diarrhoea'
                      ),
        'prob_prolonged_diarr_campylobacter':
            Parameter(Types.REAL, 'probability of prolonged episode by campylobacter-attributed diarrhoea'
                      ),
        'prob_prolonged_diarr_ETEC':
            Parameter(Types.REAL, 'probability of prolonged episode by ETEC-attributed diarrhoea'
                      ),
        'prob_prolonged_diarr_sapovirus':
            Parameter(Types.REAL, 'probability of prolonged episode by sapovirus-attributed diarrhoea'
                      ),
        'prob_prolonged_diarr_norovirus':
            Parameter(Types.REAL, 'probability of prolonged episode by norovirus-attributed diarrhoea'
                      ),
        'prob_prolonged_diarr_astrovirus':
            Parameter(Types.REAL, 'probability of prolonged episode by norovirus-attributed diarrhoea'
                      ),
        'prob_prolonged_diarr_tEPEC':
            Parameter(Types.REAL, 'probability of prolonged episode by tEPEC-attributed diarrhoea'
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
        'rr_bec_persistent_stunted':
            Parameter(Types.REAL,
                      'relative rate of acute diarrhoea becoming persistent diarrhoea for stunted children'
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
        'rr_diarr_death_untreated_HIV':
            Parameter(Types.REAL, 'relative risk of diarrhoea death for untreated HIV'
                      ),
        'rr_diarr_death_SAM':
            Parameter(Types.REAL, 'relative risk of diarrhoea death for severe acute malnutrition'
                      ),
        'rr_diarr_death_alri':
            Parameter(Types.REAL, 'relative risk of diarrhoea death if concurrent lower respiratory infection'
                      ),
        'rr_diarr_death_cryptosporidium':
            Parameter(Types.REAL, 'relative risk of diarrhoea death if caused by cryptosporidium'
                      ),
        'rr_diarr_death_shigella':
            Parameter(Types.REAL, 'relative risk of diarrhoea death if caused by shigella'
                      ),

        # todo --- all these mean_days_duration_with_PATHOGEN are not used!
        'mean_days_duration_with_rotavirus':
            Parameter(Types.LIST, 'mean, std, min, max number of days duration with diarrhoea caused by rotavirus'),
        'mean_days_duration_with_shigella':
            Parameter(Types.LIST, 'mean, std, min, max number of days duration with diarrhoea caused by shigella'),
        'mean_days_duration_with_adenovirus':
            Parameter(Types.LIST, 'mean, std, min, max number of days duration with diarrhoea caused by adenovirus'),
        'mean_days_duration_with_cryptosporidium':
            Parameter(Types.LIST, 'mean, std, min, max number of days duration with diarrhoea caused by cryptosporidium'
                      ),
        'mean_days_duration_with_campylobacter':
            Parameter(Types.LIST, 'mean, std, min, max number of days duration with diarrhoea caused by campylobacter'),
        'mean_days_duration_with_ETEC':
            Parameter(Types.LIST, 'mean, std, min, max number of days duration with diarrhoea caused by ETEC'),
        'mean_days_duration_with_sapovirus':
            Parameter(Types.LIST, 'mean, std, min, max number of days duration with diarrhoea caused by sapovirus'),
        'mean_days_duration_with_norovirus':
            Parameter(Types.LIST, 'mean, std, min, max number of days duration with diarrhoea caused by norovirus'),
        'mean_days_duration_with_astrovirus':
            Parameter(Types.LIST, 'mean, std, min, max number of days duration with diarrhoea caused by astrovirus'),
        'mean_days_duration_with_tEPEC':
            Parameter(Types.LIST, 'mean, std, min, max number of days duration with diarrhoea caused by tEPEC'),
        'prob_of_cure_given_WHO_PlanC':
            Parameter(Types.REAL, 'probability of the person being cured if is provided with Treatment Plan C'),

        'probability_of_severe_dehydration_if_some_dehydration':
            Parameter(Types.REAL, 'probability that someone with diarrhoea and some dehydration develops severe '
                                  'dehydration'),
        'range_in_days_duration_of_episode':
            Parameter(Types.INT, 'the duration of an episode of diarrhoea is a uniform distribution around the mean '
                                 'with a range equal by this number.'),

        # TODO check the below parmaeters for those HSI are relevant
        'ors_effectiveness_against_severe_dehydration':
            Parameter(Types.REAL, 'effectiveness of ORS in treating severe dehydration'),
        'number_of_days_reduced_duration_with_zinc':
            Parameter(Types.INT, 'number of days reduced duration with zinc'),
        'days_between_treatment_and_cure':
            Parameter(Types.INT, 'number of days between any treatment being given in an HSI and the cure occurring.'),
        'ors_effectiveness_on_diarrhoea_mortality':
            Parameter(Types.REAL,
                      'effectiveness of ORS in treating acute diarrhoea'),
        'antibiotic_effectiveness_for_dysentery':
            Parameter(Types.REAL,
                      'probability of cure of dysentery when treated with antibiotics'),
        'rr_diarr_death_vitaminA_supplementation':
            Parameter(Types.REAL,
                      'relative risk of death with vitamin A supplementation'),
        'mean_days_reduced_with_zinc_supplementation_in_acute_diarrhoea':
            Parameter(Types.REAL,
                      'mean duration in days reduced when managed with zinc supplementation, '
                      'in acute diarrhoea of > 6 months old'),
        'mean_days_reduced_with_zinc_supplementation_in_malnourished_children':
            Parameter(Types.REAL,
                      'mean duration in days reduced when managed with zinc supplementation, '
                      'in malnourished children of > 6 months old'),

        # Parameters describing the treatment of diarrhoea: todo --  redefine as necc.
        'prob_recommended_treatment_given_by_hw':
            Parameter(Types.REAL,
                      'probability of recommended treatment given by health care worker'
                      ),
        'prob_at_least_ors_given_by_hw':
            Parameter(Types.REAL,
                      'probability of ORS given by health care worker, with or without zinc'
                      ),
        'prob_antibiotic_given_for_dysentery_by_hw':
            Parameter(Types.REAL,
                      'probability of antibiotics given by health care worker, for dysentery'
                      ),
        'prob_multivitamins_given_for_persistent_diarrhoea_by_hw':
            Parameter(Types.REAL,
                      'probability of multivitamins given by health care worker, for persistent diarrhoea'
                      ),
        'prob_hospitalization_referral_for_severe_diarrhoea':
            Parameter(Types.REAL,
                      'probability of hospitalisation of severe diarrhoea'
                      ),
        'sensitivity_danger_signs_visual_inspection':
            Parameter(Types.REAL,
                      'sensitivity of health care workers visual inspection of danger signs'
                      ),
        'specificity_danger_signs_visual_inspection':
            Parameter(Types.REAL,
                      'specificity of health care workers visual inspection of danger signs'
                      ),
    }

    PROPERTIES = {
        # ---- Core Properties of Actual Status and Intrinsic Properties of A Current Episode  ----
        'gi_ever_had_diarrhoea': Property(Types.BOOL,
                                          'Whether or not the person has ever had an episode of diarrhoea.'
                                          ),
        'gi_last_diarrhoea_pathogen': Property(Types.CATEGORICAL,
                                               'Attributable pathogen for the last episode of diarrhoea.'
                                               'not_applicable is used if the person has never had an episode of '
                                               'diarrhoea',
                                               categories=list(pathogens) + ['not_applicable']),
        'gi_last_diarrhoea_type': Property(Types.CATEGORICAL,
                                           'Type of the last episode of diarrhoea: either watery or bloody.'
                                           'not_applicable is used if the person has never had an episode of '
                                           'diarrhoea.',
                                           categories=['not_applicable',  # (never has had diarrhoea)
                                                       'watery',
                                                       'bloody']),
        'gi_last_diarrhoea_dehydration': Property(Types.CATEGORICAL,
                                                  'Severity of dehydration of last episode of diarrhoea.'
                                                  'not_applicable is used if the person has never had an episode of '
                                                  'diarrhoea',
                                                  categories=['not_applicable',  # (never has had diarrhoea)
                                                              'none',       # <-- this can be assigned at onset
                                                              'some',       # <-- this can be assigned at onset
                                                              'severe'      # <-- this may develop during the episode
                                                              ]),
        'gi_last_diarrhoea_duration_longer_than_13days': Property(Types.BOOL,
                                                                  'Whether the duration of the _untreated_ episode'
                                                                  'would last longer than 13 days.'),

        # ---- Internal variables storing dates of scheduled events  ----
        'gi_last_diarrhoea_date_of_onset': Property(Types.DATE, 'date of onset of last episode of diarrhoea. '
                                                                'pd.NaT if never had diarrhoea'),
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
    }

    def __init__(self, name=None, resourcefilepath=None, do_checks=False):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.models = None
        self.daly_wts = dict()
        self.consumables_used_in_hsi = dict()
        self.do_checks = do_checks

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

        # Parameters for diagnostics/treatment todo - put as parameters in resourcefile
        self.parameters['prob_hospitalization_referral_for_severe_diarrhoea'] = 0.059
        self.parameters['prob_at_least_ors_given_by_hw'] = 0.633  # for all with uncomplicated diarrhoea
        self.parameters['prob_recommended_treatment_given_by_hw'] = 0.423  # for all with uncomplicated diarrhoea
        self.parameters['prob_antibiotic_given_for_dysentery_by_hw'] = 0.8  # dummy
        self.parameters['prob_multivitamins_given_for_persistent_diarrhoea_by_hw'] = 0.7  # dummy
        self.parameters['sensitivity_danger_signs_visual_inspection'] = 0.9  # dummy
        self.parameters['specificity_danger_signs_visual_inspection'] = 0.8  # dummy

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
        df.loc[df.is_alive, 'gi_last_diarrhoea_duration_longer_than_13days'] = False

        # ---- Internal values ----
        df.loc[df.is_alive, 'gi_last_diarrhoea_date_of_onset'] = pd.NaT
        df.loc[df.is_alive, 'gi_last_diarrhoea_recovered_date'] = pd.NaT
        df.loc[df.is_alive, 'gi_last_diarrhoea_death_date'] = pd.NaT
        df.loc[df.is_alive, 'gi_last_diarrhoea_treatment_date'] = pd.NaT
        df.loc[df.is_alive, 'gi_end_of_last_episode'] = pd.NaT

    def initialise_simulation(self, sim):
        """Prepares for simulation:
        * Schedules the main polling event
        * Schedules the main logging event
        * Schedule the check properties event (if needed)
        * Establishes the linear models and other data structures using the parameters that have been read-in
        * Store the consumables that are required in each of the HSI
        * Register test for the determination of 'danger signs'
        """

        # Schedule the main polling event (to first occur immediately)
        sim.schedule_event(DiarrhoeaPollingEvent(self), sim.date)

        # Schedule the main logging event (to first occur in one year)
        sim.schedule_event(DiarrhoeaLoggingEvent(self), sim.date + DateOffset(years=1))

        if self.do_checks:
            # Schedule the event that does checking every day (with time-offset to ensure it's the last event done):
            sim.schedule_event(DiarrhoeaCheckPropertiesEvent(self),
                               sim.date + pd.Timedelta(hours=23, minutes=59))

        # Create and store the models needed
        self.models = Models(self)

        # Get DALY weights
        if 'HealthBurden' in self.sim.modules.keys():
            get_daly_weight = self.sim.modules['HealthBurden'].get_daly_weight
            self.daly_wts['mild_diarrhoea'] = get_daly_weight(sequlae_code=32)
            self.daly_wts['moderate_diarrhoea'] = get_daly_weight(sequlae_code=35)
            self.daly_wts['severe_diarrhoea'] = get_daly_weight(sequlae_code=34)

        # Look-up and store the consumables that are required for each HSI
        self.look_up_consumables()

        # Define test to determine danger signs
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            danger_signs_visual_inspection=DxTest(
                property='gi_last_diarrhoea_dehydration',
                target_categories=['severe'],
                sensitivity=self.parameters['sensitivity_danger_signs_visual_inspection'],
                specificity=self.parameters['specificity_danger_signs_visual_inspection']
            )
        )

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
        df.at[child_id, 'gi_last_diarrhoea_duration_longer_than_13days'] = False

        # ---- Internal values ----
        df.at[child_id, 'gi_last_diarrhoea_date_of_onset'] = pd.NaT
        df.at[child_id, 'gi_last_diarrhoea_recovered_date'] = pd.NaT
        df.at[child_id, 'gi_last_diarrhoea_death_date'] = pd.NaT
        df.at[child_id, 'gi_last_diarrhoea_treatment_date'] = pd.NaT
        df.at[child_id, 'gi_end_of_last_episode'] = pd.NaT

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

        def add_consumable(_condition, _package, _item):
            self.consumables_used_in_hsi[_condition] = {
                'Intervention_Package_Code': _package,
                'Item_Code': _item
            }

        ors_code = get_code(package='ORS')
        severe_diarrhoea_code = get_code(package='Treatment of severe diarrhea')
        zinc_under_6m_code = get_code(package='Zinc for Children 0-6 months')
        zinc_over_6m_code = get_code(package='Zinc for Children 6-59 months')
        zinc_tablet_code = get_code(item='Zinc, tablet, 20 mg')
        antibiotics_code = get_code(package='Antibiotics for treatment of dysentery')
        cipro_code = get_code(item='Ciprofloxacin 250mg_100_CMST')

        # -- Assemble the footprints for each diarrhoea-related condition or plan:
        # -- todo only a few of these are being used.
        add_consumable('ORS', {ors_code: 1}, {})
        add_consumable('Dehydration_Plan_A', {ors_code: 1}, {})
        add_consumable('Dehydration_Plan_B', {ors_code: 1}, {})
        add_consumable('Dehydration_Plan_C', {severe_diarrhoea_code: 1}, {})
        add_consumable('Zinc_Under6mo', {zinc_under_6m_code: 1}, {zinc_tablet_code: 5})
        add_consumable('Zinc_Over6mo', {zinc_over_6m_code: 1}, {zinc_tablet_code: 5})
        add_consumable('Antibiotics_for_Dysentery', {antibiotics_code: 1}, {cipro_code: 6})
        add_consumable('Multivitamins_for_Persistent', {zinc_under_6m_code: 1}, {zinc_tablet_code: 5})

        # TODO: multivitamins consumables

    def do_when_presentation_with_diarrhoea(self, person_id, hsi_event):
        """This routine is called when Diarrhoea is a symptom for a child attending a Generic HSI Appointment. It
        checks for danger signs and schedules HSI Events appropriately."""

        # 1) Assessment of danger signs
        danger_signs = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run="danger_signs_visual_inspection", hsi_event=hsi_event)

        # 2) Determine which HSI to use:
        if danger_signs:
            # Danger signs --> In-patient
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Diarrhoea_Treatment_Inpatient(
                    person_id=person_id,
                    module=self.sim.modules['Diarrhoea']),
                priority=0,
                topen=self.sim.date,
                tclose=None)

        else:
            # No danger signs --> Out-patient
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Diarrhoea_Treatment_Outpatient(
                    person_id=person_id,
                    module=self.sim.modules['Diarrhoea']),
                priority=0,
                topen=self.sim.date,
                tclose=None)

    def do_treatment(self, person_id, hsi_event):
        """Method that enacts decisions about a treatment and its effect for diarrhoea caused by a pathogen.
        (It will do anything if the diarrhoea is caused by another module.)
        Actions:
        * Log the treatment date
        * Prevents this episode of diarrhoea from causing a death if the treatment is succesful
        * Schedules the cure event, at which symptoms are alleviated.

        See this report:
          https://apps.who.int/iris/bitstream/handle/10665/104772/9789241506823_Chartbook_eng.pdf (page 3).
        NB:
        * Provisions for cholera are not included
        * The danger signs are classified collectively and are based on the result of a DxTest representing the
          ability of the clinician to correctly determine the true value of the
          property 'gi_last_diarrhoea_dehydration' being equal to 'severe'

          # todo - @ines and @timc - check logic here: multiplication/division of probabilities!?!
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

        # *** Implement the treatment algorithm ***
        # Check the child's condition
        blood_in_stool = df.at[person_id, 'gi_last_diarrhoea_type'] == 'bloody'
        dehydration = df.at[person_id, 'gi_last_diarrhoea_dehydration']

        prob_cure = 0.0
        if (dehydration == 'some'):   # todo: <-- should this be 'some' or 'not severe' ....?
            # If some dehydration...

            # Provide ORS (if available):
            if hsi_event.get_all_consumables(footprint=self.consumables_used_in_hsi['ORS']):
                # Set probaility of succesful treatment:
                prob_cure = 1.0 - self.parameters['ors_effectiveness_on_diarrhoea_mortality']

        else:
            # If Severe dehyrdation...

            # Provide Package of medicines for Severe Dehyrdation (if available);
            if hsi_event.get_all_consumables(footprint=self.consumables_used_in_hsi['Dehydration_Plan_C']):
                # Set probability of succesful treatment:
                prob_cure = 1.0 - self.parameters['ors_effectiveness_against_severe_dehydration']

        # todo: Log the use of multivitamins -- in which case?
        # _ = hsi_event.get_all_consumables(self.consumables_used_in_hsi['Multivitamins_for_Persistent'])


        # If blood_in_stool (i.e., dysentery):
        if blood_in_stool:
            # Provide antibiotics (if available) #todo: how should these effects be combined; currently makes no sense.
            if hsi_event.get_all_consumables(footprint=self.consumables_used_in_hsi['Antibiotics_for_Dysentery']):
                pass
            else:
                prob_cure *=  (1.0 - self.parameters['antibiotic_effectiveness_for_dysentery'])


        # -------------------------------------
        # Log that the treatment is provided:
        df.at[person_id, 'gi_last_diarrhoea_treatment_date'] = self.sim.date

        # Determine if the treatment is effective
        if prob_cure > self.rng.rand():
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
        """
        self.sim.population.props.at[person_id, 'gi_last_diarrhoea_death_date'] = pd.NaT

    def check_properties(self):
        """Check that the properties are ok: for use in testing only"""

        df = self.sim.population.props

        # Those that have never had diarrhoea, should have not_applicable/null values for all the other properties:
        assert (df.loc[~df.gi_ever_had_diarrhoea & ~df.date_of_birth.isna(), [
            'gi_last_diarrhoea_pathogen',
            'gi_last_diarrhoea_type',
            'gi_last_diarrhoea_dehydration']
        ] == 'not_applicable').all().all()

        assert pd.isnull(df.loc[~df.date_of_birth.isna() & ~df['gi_ever_had_diarrhoea'], [
            'gi_last_diarrhoea_date_of_onset',
            'gi_last_diarrhoea_recovered_date',
            'gi_last_diarrhoea_death_date',
            'gi_last_diarrhoea_treatment_date']
        ]).all().all()

        # Those that have had diarrhoea, should have a pathogen and a number of days duration
        assert (df.loc[df.gi_ever_had_diarrhoea, 'gi_last_diarrhoea_pathogen'] != 'none').all()

        # Those that have had diarrhoea and no treatment, should have either a recovery date or a death_date (but not both)
        has_recovery_date = ~pd.isnull(df.loc[df.gi_ever_had_diarrhoea & pd.isnull(df.gi_last_diarrhoea_treatment_date),
                                              'gi_last_diarrhoea_recovered_date'])
        has_death_date = ~pd.isnull(df.loc[df.gi_ever_had_diarrhoea & pd.isnull(df.gi_last_diarrhoea_treatment_date),
                                           'gi_last_diarrhoea_death_date'])
        has_recovery_date_or_death_date = has_recovery_date | has_death_date
        has_both_recovery_date_and_death_date = has_recovery_date & has_death_date
        assert has_recovery_date_or_death_date.all()
        assert not has_both_recovery_date_and_death_date.any()

        # Those for whom the death date has past should be dead
        assert not df.loc[df.gi_ever_had_diarrhoea & (df['gi_last_diarrhoea_death_date'] < self.sim.date), 'is_alive'].any()

        # Check that those in a current episode have symptoms of diarrhoea [caused by the diarrhoea module]
        #  but not others (among those who are alive)
        has_symptoms_of_diar = set(self.sim.modules['SymptomManager'].who_has('diarrhoea'))
        has_symptoms = set([p for p in has_symptoms_of_diar if
                            'Diarrhoea' in self.sim.modules['SymptomManager'].causes_of(p, 'diarrhoea')
                            ])

        in_current_episode_before_recovery = \
            df.is_alive & \
            df.gi_ever_had_diarrhoea & \
            (df.gi_last_diarrhoea_date_of_onset <= self.sim.date) & \
            (self.sim.date <= df.gi_last_diarrhoea_recovered_date)
        set_of_person_id_in_current_episode_before_recovery = set(
            in_current_episode_before_recovery[in_current_episode_before_recovery].index
        )

        in_current_episode_before_death = \
            df.is_alive & \
            df.gi_ever_had_diarrhoea & \
            (df.gi_last_diarrhoea_date_of_onset <= self.sim.date) & \
            (self.sim.date <= df.gi_last_diarrhoea_death_date)
        set_of_person_id_in_current_episode_before_death = set(
            in_current_episode_before_death[in_current_episode_before_death].index
        )

        in_current_episode_before_cure = \
            df.is_alive & \
            df.gi_ever_had_diarrhoea & \
            (df.gi_last_diarrhoea_date_of_onset <= self.sim.date) & \
            (df.gi_last_diarrhoea_treatment_date <= self.sim.date) & \
            pd.isnull(df.gi_last_diarrhoea_recovered_date) & \
            pd.isnull(df.gi_last_diarrhoea_death_date)
        set_of_person_id_in_current_episode_before_cure = set(
            in_current_episode_before_cure[in_current_episode_before_cure].index
        )

        assert set() == set_of_person_id_in_current_episode_before_recovery.intersection(
            set_of_person_id_in_current_episode_before_death
        )

        set_of_person_id_in_current_episode = set().union(
            set_of_person_id_in_current_episode_before_recovery,
            set_of_person_id_in_current_episode_before_death,
            set_of_person_id_in_current_episode_before_cure,
        )
        assert set_of_person_id_in_current_episode == has_symptoms

class Models:
    """Helper-class to store all the models that specify the natural history of the Alri disease"""
    # todo work to do here on these models

    def __init__(self, module):
        self.module = module
        self.p = module.parameters
        self.rng = module.rng

        # Models:
        self.incidence_equations_by_pathogen = None
        self.prob_symptoms = None
        self.prob_diarrhoea_is_persistent_if_prolonged = None
        self.risk_of_death_diarrhoea = None

        # Prepare linear models / lookups
        self.write_incidence_equations_by_pathogen()
        self.write_prob_symptoms()
        self.write_lm_prob_diarrhoea_is_persistent_if_prolonged()
        self.write_lm_risk_of_death_diarrhoea()

    def write_lm_prob_diarrhoea_is_persistent_if_prolonged(self):
        """Create the linear model for the probability that the diarrhoea is 'persistent', given that it is prolonged"""
        self.prob_diarrhoea_is_persistent_if_prolonged = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.p['prob_prolonged_to_persistent_diarr'],
            Predictor('age_exact_years').when('.between(1,1.9999)', self.p['rr_bec_persistent_age12to23'])
                                        .when('.between(2,4.9999)', self.p['rr_diarr_death_age24to59mo']),
            Predictor('un_HAZ_category').when('HAZ<-3', self.p['rr_bec_persistent_stunted']),
            Predictor('un_clinical_acute_malnutrition').when('SAM', self.p['rr_bec_persistent_SAM']),
            Predictor().when('(hv_inf == True) & (hv_art == "not")', self.p['rr_bec_persistent_HIV']),
            # todo: add exclusive breastfeeding
        )

    def write_prob_symptoms(self):
        """Make a dict containing the probability of symptoms onset given acquisition of diarrhoea caused
        # by a particular pathogen"""

        def make_symptom_probs(patho):
            return {
                'diarrhoea': 1.0,
                'bloody_stool': 1 - self.p[f'proportion_AWD_in_{patho}'],
                'fever': self.p[f'prob_fever_by_{patho}'],
                'vomiting': self.p[f'prob_vomiting_by_{patho}'],
                'dehydration': self.p[f'prob_dehydration_by_{patho}'],
            }

        self.prob_symptoms = dict()
        for pathogen in self.module.pathogens:
            self.prob_symptoms[pathogen] = make_symptom_probs(pathogen)

        # Check that each pathogen has a risk of developing each symptom
        assert set(self.module.pathogens) == set(self.prob_symptoms.keys())

        assert all(
            [
                set(self.module.symptoms) == set(self.prob_symptoms[pathogen].keys())
                for pathogen in self.prob_symptoms.keys()
            ]
        )

    def get_duration_of_new_episode(self, pathogen, person_id):
        """Determine the duration of an episode of diarrhoea, based on pathogen and characteristics of person"""
        # todo bring the linear models into this and do in one go!

        df_slice = self.module.sim.population.props.loc[[person_id]]

        # todo - put these as parameters
        min_dur_acute = 1
        min_dur_prolonged = 5
        min_dur_persistent = 13
        max_dur_persistent = 30

        # --------------------------------------------------------------------------------------------
        # # Get mean duration
        # # todo: these are not being used!!!
        # mean_duration_in_days_of_diarrhoea = p[f"mean_days_duration_with_{pathogen}"]

        if self.p[f"prob_prolonged_diarr_{pathogen}"] > self.rng.rand():

            if self.prob_diarrhoea_is_persistent_if_prolonged.predict(df_slice, self.rng):
                # Persistent dirarrhoa
                duration = self.rng.randint(min_dur_persistent, max_dur_persistent)
            else:
                # "Prolonged" but not "persistnet"
                duration = self.rng.randint(min_dur_prolonged, min_dur_persistent)
        else:
            # If not prolonged, the episode is acute: duration of 4 days
            duration = self.rng.randint(min_dur_acute, min_dur_prolonged)

        return duration

    def write_lm_risk_of_death_diarrhoea(self):
        """Makes the unscaled linear model with default intercept of 1. Calculates the mean CFR for
        0-year-olds and then creates a new linear model with adjusted intercept so incidents in 0-year-olds
        matches the specified value in the model when averaged across the population
        """

        # todo - effect of dehydration is not included here!
        def make_lm_death(intercept=1.0):
            return LinearModel(
                LinearModelType.MULTIPLICATIVE,
                intercept,
                Predictor('gi_last_diarrhoea_pathogen').when('cryptosporidium', self.p['rr_diarr_death_cryptosporidium'])
                    .when('shigella', self.p['rr_diarr_death_shigella']),
                Predictor('gi_last_diarrhoea_type').when('watery', self.p['case_fatality_rate_AWD'])
                                                   .when('bloody', self.p['case_fatality_rate_dysentery']),
                Predictor('gi_last_diarrhoea_duration_longer_than_13days').when(True,
                                                             self.p['rr_diarr_death_if_duration_longer_than_13_days']),

                Predictor('age_exact_years').when('.between(1,1.9999)', self.p['rr_diarr_death_age12to23mo'])
                                            .when('.between(2,4.9999)', self.p['rr_diarr_death_age24to59mo'])
                                            .when('.between(0,0.9999)', 1.0).otherwise(0.0),
                Predictor('ri_current_ALRI_status').when(True, self.p['rr_diarr_death_alri']),
                Predictor().when('(hv_inf == True) & (hv_art == "not")', self.p['rr_diarr_death_untreated_HIV']),
                Predictor('un_clinical_acute_malnutrition').when('SAM', self.p['rr_diarrhoea_SAM'])
            )

        # todo - does this logic actually work???? what kinds of cases are being developed?
        df = self.module.sim.population.props
        unscaled_lm = make_lm_death()
        target_mean = 5.306/10000  # target CFR: calculated with no. death / no. episodes GBD 2016 diarrhoea paper
        actual_mean = unscaled_lm.predict(df.loc[df.is_alive &
                                                 (df.gi_last_diarrhoea_pathogen == any(list(self.module.pathogens))) &
                                                 (df.age_years < 5)]).mean()
        scaled_intercept = 1.0 * (target_mean / actual_mean) \
            if (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else 1.0
        self.risk_of_death_diarrhoea = make_lm_death(intercept=scaled_intercept)

    def write_incidence_equations_by_pathogen(self):
        """Make a dict to hold the equations that govern the probability that a person acquires diarrhoea that is
        caused (primarily) by a pathogen"""

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
                    Predictor('age_years').when('.between(0,0)', self.p[base_inc_rate][0])
                                          .when('.between(1,1)', self.p[base_inc_rate][1])
                                          .when('.between(2,4)', self.p[base_inc_rate][2])
                                          .otherwise(0.0),
                    Predictor('li_no_access_handwashing').when(False, self.p['rr_diarrhoea_HHhandwashing']),
                    Predictor('li_no_clean_drinking_water').when(False, self.p['rr_diarrhoea_clean_water']),
                    Predictor('li_unimproved_sanitation').when(False, self.p['rr_diarrhoea_improved_sanitation']),
                    Predictor().when('(hv_inf == True) & (hv_art == "not")', self.p['rr_diarrhoea_untreated_HIV']),
                    Predictor('un_clinical_acute_malnutrition').when('SAM', self.p['rr_diarrhoea_SAM']),
                    Predictor().when('(nb_breastfeeding_status == "none") & (age_exact_years < 0.5)',
                                     self.p['rr_diarrhoea_exclusive_vs_no_breastfeeding_<6mo']),
                    Predictor().when('(nb_breastfeeding_status == "non_exclusive") & (age_exact_years < 0.5)',
                                     self.p['rr_diarrhoea_exclusive_vs_partial_breastfeeding_<6mo']),
                    Predictor().when('(nb_breastfeeding_status == "none") & (0.5 < age_exact_years < 1)',
                                     self.p['rr_diarrhoea_any_vs_no_breastfeeding_6_11mo']),

                )

            df = self.module.sim.population.props
            unscaled_lm = make_linear_model(patho)
            target_mean = self.p[f'base_inc_rate_diarrhoea_by_{patho}'][0]
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 0)]).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean) \
                if (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else 1.0
            scaled_lm = make_linear_model(patho, intercept=scaled_intercept)
            # check by applying the model to mean incidence of 0-year-olds
            if (df.is_alive & (df.age_years == 0)).sum() > 0:
                assert (target_mean - scaled_lm.predict(df.loc[df.is_alive & (df.age_years == 0)]).mean()) < 1e-10
            return scaled_lm

        self.incidence_equations_by_pathogen = dict()
        for pathogen in self.module.pathogens:
            self.incidence_equations_by_pathogen[pathogen] = make_scaled_linear_model(pathogen)


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
            inc_of_acquiring_pathogen[pathogen] = m.models.incidence_equations_by_pathogen[pathogen] \
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

            # Allocate a date of onset diarrhoea ----------------------------
            date_onset = self.sim.date + DateOffset(days=rng.randint(0, days_until_next_polling_event))

            # ----------------------------------------------------------------------------------------------
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

        # Collect the updated properties of this person
        props_new = {
            'gi_ever_had_diarrhoea': True,
            'gi_last_diarrhoea_pathogen': self.pathogen,
            'gi_last_diarrhoea_date_of_onset': self.sim.date,
            'gi_last_diarrhoea_treatment_date': pd.NaT,
            'gi_last_diarrhoea_type': 'watery',
            'gi_last_diarrhoea_dehydration': 'none',
            'gi_last_diarrhoea_recovered_date': pd.NaT,
            'gi_last_diarrhoea_death_date': pd.NaT,

        }
        # Update the entry in the population dataframe
        df.loc[person_id, props_new.keys()] = props_new.values()

        # Determine the duration of the dirarrhoea, the date of outcome and the end of episode (the date when this
        # episode ends. It is the last possible data that any HSI could affect this episode.)
        duration_in_days = m.models.get_duration_of_new_episode(pathogen=self.pathogen, person_id=person_id)
        date_of_outcome = self.sim.date + DateOffset(days=duration_in_days)

        props_new['gi_last_diarrhoea_duration_longer_than_13days'] = duration_in_days >= 13  # todo rename property
        props_new['gi_end_of_last_episode'] = date_of_outcome + DateOffset(days=p['days_between_treatment_and_cure'])

        # ----------------------- Determine symptoms for this episode ----------------------
        possible_symptoms_for_this_pathogen = m.models.prob_symptoms[self.pathogen]
        for symptom, prob in possible_symptoms_for_this_pathogen.items():
            if rng.rand() < prob:
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

        # Determine the progress to severe dehydration for this episode ----------------------
        if props_new['gi_last_diarrhoea_dehydration'] == 'some':
            if rng.rand() < p['probability_of_severe_dehydration_if_some_dehydration']:
                # Change the status:
                df.at[person_id, 'gi_last_diarrhoea_dehydration'] = 'severe'

        # ----------------------- Determine outcome (recovery or death) of this episode ----------------------
        if m.models.risk_of_death_diarrhoea.predict(df.loc[[person_id]], rng):
            # todo --- this equation is not looking at props_news but the properties in the df
            # person will die (unless treated)
            props_new['gi_last_diarrhoea_death_date'] = date_of_outcome
            self.sim.schedule_event(DiarrhoeaDeathEvent(m, person_id), date_of_outcome)
        else:
            # person will recover
            props_new['gi_last_diarrhoea_recovered_date'] = date_of_outcome
            self.sim.schedule_event(DiarrhoeaNaturalRecoveryEvent(m, person_id), date_of_outcome)

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
        df.at[person_id, 'gi_last_diarrhoea_type'] = 'not_applicable'
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id,
                                                          disease_module=self.sim.modules['Diarrhoea'])


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
            # Implement the death immidiately:
            self.sim.modules['Demography'].do_death(
                individual_id=person_id,
                cause='Diarrhoea_' + df.at[person_id, 'gi_last_diarrhoea_pathogen'],
                originating_module=self.module)


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

class HSI_Diarrhoea_Treatment_Outpatient(HSI_Event, IndividualScopeEventMixin):
    """
    This is a treatment for diarrhoea administered at outpatient.
    NB. This can be called when a child presents with Diarrhoea that is caused by another module/Symptom Manager.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Diarrhoea_Treatment_Outpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Run `do_treatment` for this person from an out-potient setting."""

        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        self.module.do_treatment(person_id=person_id, hsi_event=self)

        # todo - should this be implemtned now??!?!?
        #   3) continue feeding
        #   4) follow up in 5 days if not improving


class HSI_Diarrhoea_Treatment_Inpatient(HSI_Event, IndividualScopeEventMixin):
    """
    This is a treatment for acute diarrhoea with severe dehydration administered at inpatient when danger_signs have
    been diagnosed.

    # todo - is this out-patient (descripton seems to suggest it should be in-patient?
        # commment is written but not implemented...>!!??!
        # if child has no other severe classification: PLAN C
        # Or if child has another severe classification:
        # refer urgently to hospital with mother giving frequent ORS on the way, advise on breastfeeding
        # if cholera is in your area, give antibiotic for cholera?
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Diarrhoea_Treatment_Inpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD': 1})  # todo - should this be in patient?
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Run `do_treatment` for this person from an in-potient setting."""

        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        self.module.do_treatment(person_id=person_id, hsi_event=self)


# ---------------------------------------------------------------------------------------------------------
#   HELPER MODULES FOR USE IN TESTING
# ---------------------------------------------------------------------------------------------------------

class PropertiesOfOtherModules(Module):
    """For the purpose of the testing, this module generates the properties upon which the Alri module relies"""

    PROPERTIES = {
        'hv_inf': Property(Types.BOOL, 'temporary property for HIV infection status'),
        'hv_art': Property(Types.CATEGORICAL, 'temporary property for ART status',
                           categories=["not", "on_VL_suppressed", "on_not_VL_suppressed"]),
        'ri_current_ALRI_status': Property(Types.BOOL, 'temporary property'),
        'nb_low_birth_weight_status': Property(Types.CATEGORICAL, 'temporary property',
                                               categories=['extremely_low_birth_weight', 'very_low_birth_weight',
                                                           'low_birth_weight', 'normal_birth_weight']),

        'nb_breastfeeding_status': Property(Types.CATEGORICAL, 'temporary property',
                                            categories=['none', 'non_exclusive', 'exclusive']),
        'un_clinical_acute_malnutrition': Property(Types.CATEGORICAL, 'temporary property',
                                                   categories=['MAM', 'SAM', 'well']),
        'un_HAZ_category': Property(Types.CATEGORICAL, 'temporary property',
                                    categories=['HAZ<-3', '-3<=HAZ<-2', 'HAZ>=-2']),

    }

    def __init__(self, name=None):
        super().__init__(name)

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        df = population.props
        df.loc[df.is_alive, 'hv_inf'] = False
        df.loc[df.is_alive, 'hv_art'] = 'not'
        df.loc[df.is_alive, 'ri_current_ALRI_status'] = False
        df.loc[df.is_alive, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.loc[df.is_alive, 'nb_breastfeeding_status'] = 'non_exclusive'
        df.loc[df.is_alive, 'un_clinical_acute_malnutrition'] = 'well'
        df.loc[df.is_alive, 'un_HAZ_category'] = 'HAZ>=-2'

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        df = self.sim.population.props
        df.at[child, 'hv_inf'] = False
        df.at[child, 'hv_art'] = 'not'
        df.at[child, 'ri_current_ALRI_status'] = False
        df.at[child, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.at[child, 'nb_breastfeeding_status'] = 'non_exclusive'
        df.at[child, 'un_clinical_acute_malnutrition'] = 'well'
        df.at[child, 'un_HAZ_category'] = 'HAZ>=-2'

# ---------------------------------------------------------------------------------------------------------
#   DEBUGGING / TESTING EVENTS
# ---------------------------------------------------------------------------------------------------------

class DiarrhoeaCheckPropertiesEvent(RegularEvent, PopulationScopeEventMixin):
    """This event runs daily and checks properties are in the right configuration. Only use whilst debugging!
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=1))

    def apply(self, population):
        self.module.check_properties()
