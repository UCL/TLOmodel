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

 Outstanding Issues
 * To include rotavirus vaccine
 * See todo

"""
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.util import random_date, sample_outcome

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

    INIT_DEPENDENCIES = {
        'Demography',
        'HealthSystem',
        'Hiv',
        'Lifestyle',
        'NewbornOutcomes',
        'SymptomManager',
        'Wasting',
    }

    ADDITIONAL_DEPENDENCIES = {'Alri', 'Epi', 'Stunting'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

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
        'Diarrhoea': Cause(gbd_causes='Diarrheal diseases', label='Childhood Diarrhoea')
        for path in pathogens
    }

    PARAMETERS = {
        # Parameters governing the incidence of infection with a pathogen that causes diarrhoea
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
        'rr_diarrhoea_exclusive_vs_partial_breastfeeding_<6mo':
            Parameter(Types.REAL, 'relative rate of diarrhoea for partial breastfeeding in <6 months old, '
                                  'compared to exclusive breastfeeding'
                      ),
        'rr_diarrhoea_exclusive_vs_no_breastfeeding_<6mo':
            Parameter(Types.REAL, 'relative rate of diarrhoea for no breastfeeding in <6 months old, '
                                  'compared to exclusive breastfeeding'
                      ),
        'rr_diarrhoea_any_vs_no_breastfeeding_6_11mo':
            Parameter(Types.REAL, 'relative rate of diarrhoea for no breastfeeding in 6 months old to 1 years old, '
                                  'compared to any breastfeeding at this age group'
                      ),
        'rr_diarrhoea_untreated_HIV':
            Parameter(Types.REAL, 'relative rate of diarrhoea for HIV positive status'
                      ),
        'rr_diarrhoea_SAM':
            Parameter(Types.REAL, 'relative rate of diarrhoea for severe malnutrition'
                      ),

        # Parameters governing the type of diarrhoea (Watery [AWD: Acute Watery Diarrhoea ]or bloody)
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

        # Parameters governing the extent of dehydration
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
        'probability_of_severe_dehydration_if_some_dehydration':
            Parameter(Types.REAL, 'probability that someone with diarrhoea and some dehydration develops severe '
                                  'dehydration'),

        # Parameters governing the duration of the episode
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
        'rr_bec_persistent_age>6mo':
            Parameter(Types.REAL,
                      'relative rate of acute diarrhoea becoming persistent diarrhoea for children over 6 months'
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
        'min_dur_acute':
            Parameter(Types.INT,
                      'The minimum duration (in days) for an episode that is classified as an acute episode'
                      ),
        'min_dur_prolonged':
            Parameter(Types.INT,
                      'The minimum duration (in days) for an episode that is classified as a prolonged episode'
                      ),
        'min_dur_persistent':
            Parameter(Types.INT,
                      'The minimum duration (in days) for an episode that is classified as a persistent episode'
                      ),
        'max_dur_persistent':
            Parameter(Types.INT,
                      'The maximum duration (in days) for an episode that is classified as a persistent episode'
                      ),

        # Parameters governing the occurence of other symptoms during the episode (fever, vomiting)
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

        # Parameters governing the risk of death due to Diarrhoea
        'adjustment_factor_on_cfr':
            Parameter(Types.REAL,
                      'Factor by which fatality probabilities are multiplied (to be used in calibration)'
                      ),
        'case_fatality_rate_AWD':
            Parameter(Types.REAL, 'case fatality rate for acute watery diarrhoea cases'
                      ),
        'rr_diarr_death_bloody':
            Parameter(Types.REAL, 'relative risk of diarrhoea death (compared to `case_fatality_rate_AWD` if the '
                                  'diarrhoea is of type "bloody" (i.e. dyssentry).'
                      ),
        'rr_diarr_death_age24to59mo':
            Parameter(Types.REAL,
                      'relative risk of diarrhoea death for ages 24 to 59 months'
                      ),
        'rr_diarr_death_if_duration_longer_than_13_days':
            Parameter(Types.REAL,
                      'relative risk of diarrhoea death if the duration episode is 13 days or longer'
                      ),
        'rr_diarr_death_severe_dehydration':
            Parameter(Types.REAL, 'relative risk of diarrhoea death for cases with severe dehyadration'
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

        # Parameters governing the care provided to those that present with Diarrhoea
        'sensitivity_severe_dehydration_visual_inspection':
            Parameter(Types.REAL,
                      'sensitivity of IMCI severe dehydration algorithm for dehydration >9% loss of body weight'
                      ),
        'specificity_severe_dehydration_visual_inspection':
            Parameter(Types.REAL,
                      'specificity of IMCI severe dehydration algorithm for dehydration >9% loss of body weight'
                      ),
        'prob_hospitalization_on_danger_signs':
            Parameter(Types.REAL,
                      'probability of hospitalisation when danger signs are detected in an initial consultation'
                      ),

        # Parameters governing the efficacy of treatments
        'prob_WHOPlanC_cures_dehydration_if_severe_dehydration':
            Parameter(Types.REAL,
                      'probability that severe dehydration is cured by Treatment Plan C'),
        'prob_ORS_cures_dehydration_if_severe_dehydration':
            Parameter(Types.REAL,
                      'probability that severe dehydration is cured by ORS'),
        'prob_ORS_cures_dehydration_if_non_severe_dehydration':
            Parameter(Types.REAL,
                      'probability that non-severe dehydration is cured by ORS'),
        'prob_antibiotic_cures_dysentery':
            Parameter(Types.REAL,
                      'probability that blood-in-stool is removed by the use of antibiotics'),
        'number_of_days_reduced_duration_with_zinc':
            Parameter(Types.INT, 'number of days reduced duration with zinc'),
        'days_between_treatment_and_cure':
            Parameter(Types.INT, 'number of days between any treatment being given in an HSI and the cure occurring.'),

        # Parameters describing efficacy of the monovalent rotavirus vaccine (R1)
        'rr_severe_dehydration_due_to_rotavirus_with_R1_under1yo':
            Parameter(Types.REAL,
                      'relative risk of severe dehydration with rotavirus vaccine, for under 1 years old.'),
        'rr_severe_dehydration_due_to_rotavirus_with_R1_over1yo':
            Parameter(Types.REAL,
                      'relative risk of severe dehydration with rotavirus vaccine, for those aged 1 year and older.'),
    }

    PROPERTIES = {
        # ---- Core Properties of Actual Status and Intrinsic Properties of A Current Episode  ----
        'gi_has_diarrhoea': Property(Types.BOOL,
                                     'Whether or not the person currently has an episode of diarrhoea.'
                                     ),
        'gi_pathogen': Property(Types.CATEGORICAL,
                                'The attributable pathogen for the current episode of diarrhoea '
                                '(np.nan if the person does not currently have diarrhoea).',
                                categories=list(pathogens)),
        'gi_type': Property(Types.CATEGORICAL,
                            'Type (watery or blood) of the current episode of diarrhoea '
                            '(np.nan if the person does not currently have diarrhoea).',
                            categories=['watery',
                                        'bloody']),
        'gi_dehydration': Property(Types.CATEGORICAL,
                                   'Severity of dehydration for the current episode of diarrhoea '
                                   '(np.nan if the person does not currently have diarrhoea).',
                                   categories=['none',
                                               'some',  # <-- this level is not used currently.
                                               'severe'
                                               ]),
        'gi_duration_longer_than_13days': Property(Types.BOOL,
                                                   'Whether the duration of the current episode would last longer than '
                                                   '13 days if untreated. (False if does not have current episode)'),
        'gi_number_of_episodes': Property(Types.INT,
                                          "Number of episodes of diarrhoea caused by a pathogen"),

        # ---- Internal variables storing dates of scheduled events  ----
        'gi_date_of_onset': Property(Types.DATE, 'Date of onset of current episode of diarrhoea (pd.NaT if does not '
                                                 'have current episode of diarrhoea).'),
        'gi_scheduled_date_recovery': Property(Types.DATE,
                                               'Scheduled date of recovery from current episode of diarrhoea '
                                               '(pd.NaT if does not have current episode or current episode '
                                               'is scheduled to result in death). This is scheduled when the '
                                               'episode is onset and may be revised subsequently if the episode '
                                               'is cured by a treatment'),
        'gi_scheduled_date_death': Property(Types.DATE, 'Scheduled date of death caused by current episode of diarrhoea'
                                                        ' (pd.NaT if does not have current episode or if current '
                                                        'episode will not result in death). This is scheduled when the '
                                                        'episode is onset and may be revised subsequently if the '
                                                        'episode is cured by a treatment.'),
        'gi_date_end_of_last_episode': Property(Types.DATE,
                                                'The date on which the last episode of diarrhoea is fully resolved, '
                                                'including allowing for the possibility of HSI events (pd.NaT if has '
                                                'never had an episode). This is used to determine when a new episode '
                                                'can begin and stops successive episodes interfering with one another.'
                                                'This is notnull when the person has ever had an episode of diarrhoea.'
                                                ),
        'gi_treatment_date': Property(Types.DATE,
                                      'The actual date on which treatment is first administered for the current episode'
                                      ' (pd.NaT if does not have current episode or if no treatment has yet been '
                                      'provided in the current episode).')
    }

    def __init__(self, name=None, resourcefilepath=None, do_checks=False):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.models = None
        self.daly_wts = dict()
        self.unreported_dalys = list()
        self.consumables_used_in_hsi = dict()
        self.do_checks = do_checks

    def read_parameters(self, data_folder):
        """ Setup parameters values used by the module"""
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

    def initialise_population(self, population):
        """
        Sets that there is no one with diarrhoea at initiation.
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals

        # ---- Key Current Status Classification Properties ----
        df.loc[df.is_alive, 'gi_has_diarrhoea'] = False
        df.loc[df.is_alive, 'gi_pathogen'] = np.nan
        df.loc[df.is_alive, 'gi_type'] = np.nan
        df.loc[df.is_alive, 'gi_dehydration'] = np.nan
        df.loc[df.is_alive, 'gi_duration_longer_than_13days'] = False
        df.loc[df.is_alive, 'gi_number_of_episodes'] = 0

        # ---- Internal values ----
        df.loc[df.is_alive, 'gi_date_of_onset'] = pd.NaT
        df.loc[df.is_alive, 'gi_scheduled_date_recovery'] = pd.NaT
        df.loc[df.is_alive, 'gi_scheduled_date_death'] = pd.NaT
        df.loc[df.is_alive, 'gi_treatment_date'] = pd.NaT
        df.loc[df.is_alive, 'gi_date_end_of_last_episode'] = pd.NaT

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

        if self.do_checks:
            # Schedule the event that does checking every day (with time-offset to ensure it's the last event done):
            sim.schedule_event(DiarrhoeaCheckPropertiesEvent(self),
                               sim.date + pd.Timedelta(hours=23, minutes=59))

        # Create and store the models needed
        self.models = Models(self)

        # Get DALY weights
        if 'HealthBurden' in self.sim.modules:
            get_daly_weight = self.sim.modules['HealthBurden'].get_daly_weight
            self.daly_wts['dehydration=none'] = get_daly_weight(sequlae_code=32)
            self.daly_wts['dehydration=some'] = get_daly_weight(sequlae_code=35)
            self.daly_wts['dehydration=severe'] = get_daly_weight(sequlae_code=34)

        # Look-up and store the consumables that are required for each HSI
        self.look_up_consumables()

        # Define test to determine danger signs
        # The danger signs are classified collectively and are based on the result of a DxTest representing the ability
        # of the clinician to correctly determine the true value of the property 'gi_dehydration' being 'severe'.
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            imci_severe_dehydration_visual_inspection=DxTest(
                property='gi_dehydration',
                target_categories=['severe'],
                sensitivity=self.parameters['sensitivity_severe_dehydration_visual_inspection'],
                specificity=self.parameters['specificity_severe_dehydration_visual_inspection']
            )
        )

    def on_birth(self, mother_id, child_id):
        """
        On birth, all children will have no diarrhoea
        """
        df = self.sim.population.props

        # ---- Key Current Status Classification Properties ----
        df.at[child_id, 'gi_has_diarrhoea'] = False
        df.at[child_id, 'gi_pathogen'] = np.nan
        df.at[child_id, 'gi_type'] = np.nan
        df.at[child_id, 'gi_dehydration'] = np.nan
        df.at[child_id, 'gi_duration_longer_than_13days'] = False
        df.at[child_id, 'gi_number_of_episodes'] = 0

        # ---- Internal values ----
        df.at[child_id, 'gi_date_of_onset'] = pd.NaT
        df.at[child_id, 'gi_scheduled_date_recovery'] = pd.NaT
        df.at[child_id, 'gi_scheduled_date_death'] = pd.NaT
        df.at[child_id, 'gi_treatment_date'] = pd.NaT
        df.at[child_id, 'gi_date_end_of_last_episode'] = pd.NaT

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        pass

    def report_daly_values(self):
        """
        This returns person-time live with disability values to the HealthBurden module.
        At the end of each episode, a record is made in a list, `self.unreported_dalys`, of the form
        (person_id, duration_in_days_of_episode * average_daly_weight). This record is used to create the pd.Series
        that is returned (index: person_id for all those alive, value: the average disability weight during the
        preceding month). The list is then cleared.
        """
        df = self.sim.population.props

        if len(self.unreported_dalys) == 0:
            return pd.Series(index=df.loc[df.is_alive].index, data=0.0)

        # Count number of days last month (knowing that this function is called on the first day of the month).
        days_last_month = (
            (self.sim.date - pd.DateOffset(days=1)) - (
                self.sim.date - pd.DateOffset(days=1) - pd.DateOffset(months=1))
        ).days

        # Get the person_id and the values from the list, and clear the list.
        idx, values = zip(*self.unreported_dalys)
        self.unreported_dalys = list()  # <-- clear list

        average_daly_weight_in_last_month = pd.Series(values, idx) / days_last_month
        return average_daly_weight_in_last_month.reindex(index=df.loc[df.is_alive].index, fill_value=0.0)

    def look_up_consumables(self):
        """Look up and store the consumables item codes used in each of the HSI."""
        get_item_codes_from_package_name = self.sim.modules['HealthSystem'].get_item_codes_from_package_name

        self.consumables_used_in_hsi['ORS'] = get_item_codes_from_package_name(
            package='ORS')
        self.consumables_used_in_hsi['Treatment_Severe_Dehydration'] = get_item_codes_from_package_name(
            package='Treatment of severe diarrhea')
        self.consumables_used_in_hsi['Zinc_Under6mo'] = get_item_codes_from_package_name(
            package='Zinc for Children 0-6 months')
        self.consumables_used_in_hsi['Zinc_Over6mo'] = get_item_codes_from_package_name(
            package='Zinc for Children 6-59 months')
        self.consumables_used_in_hsi['Antibiotics_for_Dysentery'] = get_item_codes_from_package_name(
            package='Antibiotics for treatment of dysentery')

    def do_when_presentation_with_diarrhoea(self, person_id, hsi_event):
        """This routine is called when Diarrhoea is a symptom for a child attending a Generic HSI Appointment. It
        checks for danger signs and schedules HSI Events appropriately."""

        # 1) Assessment of danger signs
        danger_signs = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run="imci_severe_dehydration_visual_inspection", hsi_event=hsi_event)

        # 2) Determine which HSI to use:
        if danger_signs and (self.rng.rand() < self.parameters['prob_hospitalization_on_danger_signs']):
            # Danger signs and hospitalized --> In-patient
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Diarrhoea_Treatment_Inpatient(
                    person_id=person_id,
                    module=self),
                priority=0,
                topen=self.sim.date,
                tclose=None)

        else:
            # No danger signs or otherwise not hospitalized --> Out-patient
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Diarrhoea_Treatment_Outpatient(
                    person_id=person_id,
                    module=self),
                priority=0,
                topen=self.sim.date,
                tclose=None)

    def do_treatment(self, person_id, hsi_event):
        """Method called by the HSI that enacts decisions about a treatment and its effect for diarrhoea caused by a
        pathogen. (It will do nothing if the diarrhoea is caused by another module.)
        Actions:
        * If the episode will cause death: if the treatment is successful, prevents this episode of diarrhoea from
         causing a death and schedules Cure Event
        * If the episode will not cause death: if treatment is succesful, schedules a CureEvent that will occur earlier
         than the `NaturalRecovery` event.
        * Records that treatment is provided.

        NB. Provisions for cholera are not included
        See this report:
          https://apps.who.int/iris/bitstream/handle/10665/104772/9789241506823_Chartbook_eng.pdf (page 3).
        """

        df = self.sim.population.props
        person = df.loc[person_id]
        p = self.parameters

        if not person.is_alive:
            return

        # Do nothing if the diarrhoea has not been caused by a pathogen
        if 'Diarrhoea' not in self.sim.modules['SymptomManager'].causes_of(person_id, 'diarrhoea'):
            return

        # Do nothing if the episode is not on-going
        if not (
            person.gi_has_diarrhoea and
            (person.gi_date_of_onset <= self.sim.date <= person.gi_date_end_of_last_episode)
        ):
            return

        # Check the child's condition
        type_of_diarrhoea_is_bloody = person.gi_type == 'bloody'
        dehydration_is_severe = person.gi_dehydration == 'severe'
        # days_elapsed_with_diarrhoea = (self.sim.date - person.gi_date_of_onset).days
        will_die = pd.notnull(person.gi_scheduled_date_death)
        is_in_patient = isinstance(hsi_event, HSI_Diarrhoea_Treatment_Inpatient)

        # ** Implement the procedure for treatment **
        # STEP ZERO: Get the Zinc consumable (happens irrespective of whether child will die or not)
        gets_zinc = hsi_event.get_consumables(
            item_codes=self.consumables_used_in_hsi[
                'Zinc_Under6mo' if person.age_exact_years < 0.5 else 'Zinc_Over6mo']
        )

        # STEP ONE: Aim to alleviate dehydration:
        prob_remove_dehydration = 0.0
        if is_in_patient:
            if hsi_event.get_consumables(item_codes=self.consumables_used_in_hsi['Treatment_Severe_Dehydration']):
                # In-patient receiving IV fluids (WHO Plan C)
                prob_remove_dehydration = \
                    p['prob_WHOPlanC_cures_dehydration_if_severe_dehydration'] if dehydration_is_severe \
                    else self.parameters['prob_ORS_cures_dehydration_if_non_severe_dehydration']

        else:
            if hsi_event.get_consumables(item_codes=self.consumables_used_in_hsi['ORS']):
                # Out-patient receving ORS
                prob_remove_dehydration = \
                    self.parameters['prob_ORS_cures_dehydration_if_severe_dehydration'] if dehydration_is_severe \
                    else self.parameters['prob_ORS_cures_dehydration_if_non_severe_dehydration']

        # Determine dehydration after treatment
        dehydration_after_treatment = 'none' if self.rng.rand() < prob_remove_dehydration else person.gi_dehydration

        # STEP TWO: If has bloody diarrhoea (i.e., dysentry), then aim to clear bacterial infection
        if type_of_diarrhoea_is_bloody and hsi_event.get_consumables(
            item_codes=self.consumables_used_in_hsi['Antibiotics_for_Dysentery']
        ):
            prob_clear_bacterial_infection = self.parameters['prob_antibiotic_cures_dysentery']
        else:
            prob_clear_bacterial_infection = 0.0

        # Determine type after treatment
        type_after_treatment = 'watery' if self.rng.rand() < prob_clear_bacterial_infection else person.gi_type

        # Determine if the changes in dehydration or type (if any) will cause the treatment to block the death:
        if will_die:
            if self.models.does_treatment_prevent_death(
                pathogen=person.gi_pathogen,
                type=(person.gi_type, type_after_treatment),  # <-- type may have changed
                duration_longer_than_13days=person.gi_duration_longer_than_13days,
                dehydration=(person.gi_dehydration, dehydration_after_treatment),  # <-- dehydration may have changed
                age_exact_years=person.age_exact_years,
                ri_current_infection_status=person.ri_current_infection_status,
                untreated_hiv=person.hv_inf and (person.hv_art != "on_VL_suppressed"),
                un_clinical_acute_malnutrition=person.un_clinical_acute_malnutrition
            ):
                # Treatment is successful: cancel death and schedule cure event
                self.cancel_death_date(person_id)
                self.sim.schedule_event(
                    DiarrhoeaCureEvent(self, person_id),
                    self.sim.date + pd.DateOffset(
                        days=self.parameters['days_between_treatment_and_cure'] - (
                            p['number_of_days_reduced_duration_with_zinc'] if gets_zinc else 0)
                    )
                )
        else:
            # The child would not die without treatment, but treatment can lead to an earlier end to the episode if they
            # get zinc.
            if gets_zinc:
                # Schedule the Cure Event to happen earlier that the scheduled recovery event
                self.sim.schedule_event(
                    DiarrhoeaCureEvent(self, person_id),
                    max(self.sim.date,
                        person.gi_scheduled_date_recovery - pd.DateOffset(
                            days=p['number_of_days_reduced_duration_with_zinc'])
                        )
                )

        # -------------------------------------
        # Log that the treatment is provided:
        df.at[person_id, 'gi_treatment_date'] = self.sim.date

    def cancel_death_date(self, person_id):
        """
        Cancels a scheduled date of death due to diarrhoea for a person. This is called prior to the scheduling the
        CureEvent to prevent deaths happening in the time between a treatment being given and the cure event occurring.
        """
        self.sim.population.props.at[person_id, 'gi_scheduled_date_death'] = pd.NaT

    def end_episode(self, person_id, outcome):
        """Helper function that enacts the end of the episode of diarrhoea (either by natural recovery, death or  cure)
        * Logs that the episode has ended
        * Stores the total time-lived-with-disability during the episode
        * Enacts the death (if the outcome=='death'); Otherwise, removes symptoms and resets properties
        """
        assert outcome in ['recovery', 'cure', 'death']

        df = self.sim.population.props
        person = df.loc[person_id]

        # Log that the episode has ended
        logger.info(
            key='end_of_case',
            data={
                'person_id': person_id,
                'date_of_onset': person.gi_date_of_onset,
                'outcome': outcome
            },
            description='information when a case is ended by recovery, cure or death.'
        )

        # Store the totals of days * daly_weight incurred during the episode
        if 'HealthBurden' in self.sim.modules:
            days_duration = (self.sim.date - person.gi_date_of_onset).days
            daly_wt = self.daly_wts[f'dehydration={person.gi_dehydration}']
            self.unreported_dalys.append((person_id, daly_wt * days_duration))

        # Enacts the death (if the outcome is 'death'); Otherwise, removes symptoms and resets properties
        if outcome == 'death':
            # If outcome is death, implement the death immidiately:
            self.sim.modules['Demography'].do_death(
                individual_id=person_id,
                cause='Diarrhoea_' + df.at[person_id, 'gi_pathogen'],
                originating_module=self)

        else:
            # If outcome is not death, then remove all symptoms and reset properties:
            self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id,
                                                              disease_module=self)

            df.loc[person_id, [
                'gi_has_diarrhoea',
                'gi_pathogen',
                'gi_type',
                'gi_dehydration',
                'gi_duration_longer_than_13days',
                'gi_date_of_onset',
                'gi_scheduled_date_recovery',
                'gi_scheduled_date_death',
                'gi_treatment_date']
            ] = (
                False,
                np.nan,
                np.nan,
                np.nan,
                False,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT
            )
            # NB. `gi_date_end_of_last_episode` is not reset at the end of the episode.

    def check_properties(self):
        """Check that the properties are ok: for use in testing only"""

        df = self.sim.population.props

        # Those that do not currently have diarrhoea, should have not_applicable/null values for all the other
        # properties:
        not_in_current_episode = df.is_alive & ~df.gi_has_diarrhoea
        assert pd.isnull(df.loc[not_in_current_episode, [
            'gi_pathogen',
            'gi_type',
            'gi_dehydration']
                         ]).all().all()
        assert not df.loc[not_in_current_episode, 'gi_duration_longer_than_13days'].any()
        assert pd.isnull(df.loc[not_in_current_episode, [
            'gi_date_of_onset',
            'gi_scheduled_date_recovery',
            'gi_scheduled_date_death',
            'gi_treatment_date']
                         ]).all().all()

        # Those that do currently have diarrhoea, should have a pathogen
        in_current_episode = df.is_alive & df.gi_has_diarrhoea
        assert not pd.isnull(df.loc[in_current_episode, 'gi_pathogen']).any()

        # Those that do currently have diarrhoea, should have a non-zero value for the total number of episodes
        assert (df.loc[in_current_episode, 'gi_number_of_episodes'] > 0).all()

        # Those that currently have diarrhoea and have not had a treatment, should have either a recovery date or
        # a death_date (but not both)
        has_recovery_date = ~pd.isnull(df.loc[in_current_episode & pd.isnull(df.gi_treatment_date),
                                              'gi_scheduled_date_recovery'])
        has_death_date = ~pd.isnull(df.loc[in_current_episode & pd.isnull(df.gi_treatment_date),
                                           'gi_scheduled_date_death'])
        has_recovery_date_or_death_date = has_recovery_date | has_death_date
        has_both_recovery_date_and_death_date = has_recovery_date & has_death_date
        assert has_recovery_date_or_death_date.all()
        assert not has_both_recovery_date_and_death_date.any()

        # Those for whom the death date has past should be dead
        assert not df.loc[df['gi_scheduled_date_death'] < self.sim.date, 'is_alive'].any()

        # Check that those in a current episode have symptoms of diarrhoea [caused by the diarrhoea module]
        #  but not others (among those who are alive)
        has_symptoms_of_diar = set(self.sim.modules['SymptomManager'].who_has('diarrhoea'))
        has_symptoms = {
            p for p in has_symptoms_of_diar
            if 'Diarrhoea' in self.sim.modules['SymptomManager'].causes_of(p, 'diarrhoea')
        }

        in_current_episode_before_recovery = \
            df.is_alive & \
            df.gi_has_diarrhoea & \
            (df.gi_date_of_onset <= self.sim.date) & \
            (self.sim.date <= df.gi_scheduled_date_recovery)
        set_of_person_id_in_current_episode_before_recovery = set(
            in_current_episode_before_recovery[in_current_episode_before_recovery].index
        )

        in_current_episode_before_death = \
            df.is_alive & \
            df.gi_has_diarrhoea & \
            (df.gi_date_of_onset <= self.sim.date) & \
            (self.sim.date <= df.gi_scheduled_date_death)
        set_of_person_id_in_current_episode_before_death = set(
            in_current_episode_before_death[in_current_episode_before_death].index
        )

        in_current_episode_before_cure = \
            df.is_alive & \
            df.gi_has_diarrhoea & \
            (df.gi_date_of_onset <= self.sim.date) & \
            (df.gi_treatment_date <= self.sim.date) & \
            pd.isnull(df.gi_scheduled_date_recovery) & \
            pd.isnull(df.gi_scheduled_date_death)
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

    def __init__(self, module):
        self.module = module
        self.p = module.parameters
        self.rng = module.rng

        # Write model for incidence of each pathogen
        self.linear_model_for_incidence_by_pathogen = self.write_linear_model_for_incidence_of_pathogens()

    def write_linear_model_for_incidence_of_pathogens(self):
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
                    Predictor('age_years',
                              conditions_are_mutually_exclusive=True,
                              conditions_are_exhaustive=True,
                              ) .when(0, self.p[base_inc_rate][0])
                                .when(1, self.p[base_inc_rate][1])
                                .when('.between(2, 4)', self.p[base_inc_rate][2])
                                .when('> 4', 0.0),
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

        _incidence_equations_by_pathogen = dict()
        for pathogen in self.module.pathogens:
            _incidence_equations_by_pathogen[pathogen] = make_scaled_linear_model(pathogen)

        return _incidence_equations_by_pathogen

    def get_prob_persisent_if_prolonged(self,
                                        age_exact_years,
                                        un_HAZ_category,
                                        un_clinical_acute_malnutrition,
                                        untreated_hiv,
                                        ):
        # Baseline prob:
        prob_persistent_if_prolonged = self.p['prob_prolonged_to_persistent_diarr']

        if age_exact_years > 0.5:
            prob_persistent_if_prolonged *= self.p['rr_bec_persistent_age>6mo']

        if un_HAZ_category == 'HAZ<-3':
            prob_persistent_if_prolonged *= self.p['rr_bec_persistent_stunted']

        if un_clinical_acute_malnutrition == 'SAM':
            prob_persistent_if_prolonged *= self.p['rr_bec_persistent_SAM']

        if untreated_hiv:
            prob_persistent_if_prolonged *= self.p['rr_bec_persistent_HIV']

        return prob_persistent_if_prolonged

    def get_duration(self,
                     pathogen,
                     age_exact_years,
                     un_HAZ_category,
                     un_clinical_acute_malnutrition,
                     untreated_hiv,
                     ):
        """For new incident case of diarrhoea, determine its duration.
        1) Determine if this will be acute or prolonged; and if prolonged, will it be persistent.
        2) Randomly select a duration from with a range defined for each classification."""

        # Get probability of this episode being "prolonged"
        prob_prolonged = self.p[f"prob_prolonged_diarr_{pathogen}"]

        # Get probability of this episode being "persistent" if it is "prolonged"
        prob_persistent_if_prolonged = self.get_prob_persisent_if_prolonged(
            age_exact_years=age_exact_years,
            un_HAZ_category=un_HAZ_category,
            un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
            untreated_hiv=untreated_hiv,
        )

        if self.rng.rand() < prob_prolonged:
            if self.rng.rand() < prob_persistent_if_prolonged:
                # "Persistent" diarrhoea
                return self.rng.randint(self.p['min_dur_persistent'], self.p['max_dur_persistent'])
            # "Prolonged" but not "persistent"
            return self.rng.randint(self.p['min_dur_prolonged'], self.p['min_dur_persistent'])
        # If not prolonged, the episode is acute: duration of 4 days
        return self.rng.randint(self.p['min_dur_acute'], self.p['min_dur_prolonged'])

    def get_type(self, pathogen):
        """For new incident case of diarrhoea, determine the type - 'watery' or 'bloody'"""
        return 'watery' if self.rng.rand() < self.p[f'proportion_AWD_in_{pathogen}'] else 'bloody'

    def get_dehydration(self, pathogen, va_rota_all_doses, age_years):
        """For new incident case of diarrhoea, determine the degree of dehydration - 'none', 'some' or 'severe'.
        The effect of the R1 vaccine is to reduce the probability of severe dehydration."""

        if (pathogen == "rotavirus") and va_rota_all_doses:
            relative_prob_severe_dehydration_due_to_vaccine = \
                self.p['rr_severe_dehydration_due_to_rotavirus_with_R1_under1yo'] if age_years < 1 \
                else self.p['rr_severe_dehydration_due_to_rotavirus_with_R1_over1yo']
        else:
            relative_prob_severe_dehydration_due_to_vaccine = 1.0

        if self.rng.rand() < self.p[f'prob_dehydration_by_{pathogen}']:
            if self.rng.rand() < (
                self.p['probability_of_severe_dehydration_if_some_dehydration']
                * relative_prob_severe_dehydration_due_to_vaccine
            ):
                return 'severe'
            return 'some'
        return 'none'

    def prob_death(self,
                   pathogen,
                   type,
                   duration_longer_than_13days,
                   dehydration,
                   age_exact_years,
                   ri_current_infection_status,
                   untreated_hiv,
                   un_clinical_acute_malnutrition
                   ):
        """Compute probability will die given a set of conditions"""

        # Baseline risks for diarrhoea of type 'watery', including 'adjustment factor'
        risk = self.p['case_fatality_rate_AWD'] * self.p['adjustment_factor_on_cfr']

        # Factors that adjust risk up or down:
        if type == 'bloody':
            risk *= self.p['rr_diarr_death_bloody']

        if pathogen == 'cryptosporidium':
            risk *= self.p['rr_diarr_death_cryptosporidium']
        elif pathogen == 'shigella':
            risk *= self.p['rr_diarr_death_shigella']

        if duration_longer_than_13days:
            risk *= self.p['rr_diarr_death_if_duration_longer_than_13_days']

        if dehydration == 'severe':
            risk *= self.p['rr_diarr_death_severe_dehydration']

        if age_exact_years < 2.0:
            pass
        elif (2.0 <= age_exact_years < 5.0):
            risk *= self.p['rr_diarr_death_age24to59mo']
        else:
            risk *= 0.0

        if ri_current_infection_status:
            risk *= self.p['rr_diarr_death_alri']

        if untreated_hiv:
            risk *= self.p['rr_diarr_death_untreated_HIV']

        if un_clinical_acute_malnutrition == 'SAM':
            risk *= self.p['rr_diarr_death_SAM']

        return risk

    def will_die(self,
                 pathogen,
                 type,
                 duration_longer_than_13days,
                 dehydration,
                 age_exact_years,
                 ri_current_infection_status,
                 untreated_hiv,
                 un_clinical_acute_malnutrition
                 ):
        """For new incident case of dirarrhoea, determine whether death will result.
        This gets the probability of death and determines outcome probabilistically. """

        prob = self.prob_death(
            pathogen,
            type,
            duration_longer_than_13days,
            dehydration,
            age_exact_years,
            ri_current_infection_status,
            untreated_hiv,
            un_clinical_acute_malnutrition
        )

        # Return bool following coin-flip to determine outcome
        return self.rng.rand() < prob

    def does_treatment_prevent_death(self, **kwargs):
        """For a case of diarrhoea that will cause death if untreated, that is now being treated, determine if the
        treatment will prevent a death from occuring.
        Each parameter can be provided as a single value or as an itterable representing the status as that
        ("before_treatment", "after_treatment").
        This method is called for a person for whom a death is scheduled and is receiving treatment. If the outcome is
        True then the death should be cancalled as the change in circumstances (treatment of diarrhoea type or
        dehydration) have reduced the probabilty of death such that there it will not occur."""

        def is_iterable_and_not_string(x):
            return isinstance(x, Iterable) and not isinstance(x, str)

        prob_death = dict()
        for i, case in enumerate(['before_treatment', 'after_treatment']):
            prob_death[case] = (
                self.prob_death(
                    **{
                        k: v[i] if is_iterable_and_not_string(v) else v for k, v in kwargs.items()
                    }
                )
            )

        # Probability that treatment "blocks" the death for someone that would have died.
        prob_treatment_blocks_death = 1 - prob_death['after_treatment']/prob_death['before_treatment']

        # Return outcome, determine probabilstically
        return self.rng.rand() < prob_treatment_blocks_death

    def get_symptoms(self, pathogen):
        """For new incident case of diarrhoea, determine the symptoms that onset."""

        probs = {
            'fever': self.p[f'prob_fever_by_{pathogen}'],
            'vomiting': self.p[f'prob_vomiting_by_{pathogen}'],
        }

        symptoms_that_onset = ['diarrhoea']
        for symptom, prob in probs.items():
            if self.rng.rand() < prob:
                symptoms_that_onset.append(symptom)
        return symptoms_that_onset


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
        self.fraction_of_a_year_until_next_polling_event = self.compute_fraction_of_year_between_polling_event()

    def compute_fraction_of_year_between_polling_event(self):
        """Compute fraction of a year that elapses between polling event. This is used to adjust the risk of
        infection"""
        return (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'Y')

    def apply(self, population):
        df = population.props
        m = self.module
        models = m.models

        # Define who is at risk of onset of a episode of Diarrhoea.
        # Those susceptible are children that do not currently have an episode (never had an episode or last episode
        # resolved) and who do not have diarrhoea as a 'spurious symptom' of diarrhoea already.
        mask_could_get_new_diarrhoea_episode = (
            df.is_alive &
            (df.age_years < 5) &
            ~df['gi_has_diarrhoea'] &
            ((df.gi_date_end_of_last_episode < self.sim.date) | pd.isnull(df.gi_date_end_of_last_episode)) &
            ~df.index.isin(self.sim.modules['SymptomManager'].who_has('diarrhoea'))
        )

        # Compute the incidence rate for each person getting diarrhoea
        inc_of_acquiring_pathogen = pd.DataFrame(index=df.loc[mask_could_get_new_diarrhoea_episode].index)
        for pathogen in m.pathogens:
            inc_of_acquiring_pathogen[pathogen] = models.linear_model_for_incidence_by_pathogen[pathogen] \
                .predict(df.loc[mask_could_get_new_diarrhoea_episode])

        # Convert the incidence rates into risk of an event occurring before the next polling event
        probs_of_aquiring_pathogen = 1 - np.exp(
            -inc_of_acquiring_pathogen * self.fraction_of_a_year_until_next_polling_event
        )

        # Sample to find outcomes (which pathogen, if any, caused Diarrhoea among persons at-risk):
        outcome = sample_outcome(probs=probs_of_aquiring_pathogen, rng=self.module.rng)

        for person_id, pathogen in outcome.items():
            # Create and schedule the event for the onset of infection
            self.sim.schedule_event(
                event=DiarrhoeaIncidentCase(module=self.module, person_id=person_id, pathogen=pathogen),
                date=random_date(self.sim.date, self.sim.date + self.frequency - pd.DateOffset(days=1), m.rng)
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

    def __init__(self, module, person_id, pathogen):
        super().__init__(module, person_id=person_id)
        self.pathogen = pathogen

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters

        person = df.loc[person_id]
        untreated_hiv = person.hv_inf and (person.hv_art != "on_VL_suppressed")

        # The event will not run if the person is not currently alive
        if not person.is_alive:
            return

        # The event will not run if the child is now older than five year-olds
        if not person.age_years < 5:
            return

        # Determine the duration of the dirarrhoea, the date of outcome and the end of episode (the date when this
        # episode ends. It is the last possible data that any HSI could affect this episode.)
        duration_in_days = m.models.get_duration(pathogen=self.pathogen,
                                                 age_exact_years=person.age_exact_years,
                                                 un_HAZ_category=person.un_HAZ_category,
                                                 un_clinical_acute_malnutrition=person.un_clinical_acute_malnutrition,
                                                 untreated_hiv=untreated_hiv,
                                                 )
        date_of_outcome = self.sim.date + DateOffset(days=duration_in_days)

        # Collected updated properties of this person
        props_new = {
            'gi_has_diarrhoea': True,
            'gi_pathogen': self.pathogen,
            'gi_type': m.models.get_type(self.pathogen),
            'gi_dehydration': m.models.get_dehydration(pathogen=self.pathogen,
                                                       va_rota_all_doses=person.va_rota_all_doses,
                                                       age_years=person.age_years),
            'gi_duration_longer_than_13days': duration_in_days >= 13,
            'gi_date_of_onset': self.sim.date,
            'gi_date_end_of_last_episode': date_of_outcome + DateOffset(days=p['days_between_treatment_and_cure']),
            'gi_scheduled_date_recovery': pd.NaT,   # <-- determined below
            'gi_scheduled_date_death': pd.NaT,      # <-- determined below
            'gi_treatment_date': pd.NaT,            # <-- pd.NaT until treatment is provided.
        }

        # Determine outcome (recovery or death) of this episode
        if m.models.will_die(
            pathogen=props_new['gi_pathogen'],
            type=props_new['gi_type'],
            duration_longer_than_13days=props_new['gi_duration_longer_than_13days'],
            dehydration=props_new['gi_dehydration'],
            age_exact_years=person['age_exact_years'],
            ri_current_infection_status=person['ri_current_infection_status'],
            untreated_hiv=untreated_hiv,
            un_clinical_acute_malnutrition=person['un_clinical_acute_malnutrition']
        ):
            # person will die (unless treated)
            props_new['gi_scheduled_date_death'] = date_of_outcome
            self.sim.schedule_event(DiarrhoeaDeathEvent(m, person_id), date_of_outcome)
        else:
            # person will recover
            props_new['gi_scheduled_date_recovery'] = date_of_outcome
            self.sim.schedule_event(DiarrhoeaNaturalRecoveryEvent(m, person_id), date_of_outcome)

        # Update the entry in the population dataframe
        df.loc[person_id, props_new.keys()] = props_new.values()

        # Apply symptoms for this episode (these do not affect the course of disease)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=person_id,
            symptom_string=m.models.get_symptoms(self.pathogen),
            add_or_remove='+',
            disease_module=self.module
        )

        # Log this incident case:
        logger.info(
            key='incident_case',
            data={
                'person_id': person_id,
                'age_years': person.age_years,
                'pathogen': props_new['gi_pathogen'],
                'type': props_new['gi_type'],
                'dehydration': props_new['gi_dehydration'],
                'duration_longer_than_13days': props_new['gi_duration_longer_than_13days'],
                'date_of_outcome': date_of_outcome,
                'will_die': pd.isnull(props_new['gi_scheduled_date_recovery'])
            },
            description='each incicdent case of diarrhoea'
        )

        # Increment the counter for the number of cases of diarrhoea had
        df.at[person_id, 'gi_number_of_episodes'] += 1


class DiarrhoeaNaturalRecoveryEvent(Event, IndividualScopeEventMixin):
    """
    This is the Natural Recovery event. It is part of the natural history and represents the end of an episode of
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
        if not self.sim.date == person.gi_scheduled_date_recovery:
            return

        # Confirm that this is event is occurring during a current episode of diarrhoea
        assert person.gi_date_of_onset <= self.sim.date <= person.gi_date_end_of_last_episode
        assert person.gi_has_diarrhoea

        # Check that the person is not scheduled to die in this episode
        assert pd.isnull(person.gi_scheduled_date_death)

        # Resolve all the symptoms and reset the properties
        self.module.end_episode(person_id=person_id, outcome='recovery')


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
        person = df.loc[person_id]

        # The event should not run if the person is not currently alive
        if not person.is_alive:
            return

        # Do nothing if the death is not scheduled for today (i.e. if it has been cancalled)
        if not (
            (self.sim.date == person.gi_scheduled_date_death) and
            pd.isnull(person.gi_scheduled_date_recovery)
        ):
            return

        # Confirm that this is event is occurring during a current episode of diarrhoea
        assert person.gi_date_of_onset <= self.sim.date <= person.gi_date_end_of_last_episode
        assert person.gi_has_diarrhoea

        # End the episode with a death:
        self.module.end_episode(person_id=person_id, outcome='death')


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

        # Do nothing if this is the person is not in a current episode of diarrhoea
        if not (
            person.gi_has_diarrhoea and (person.gi_date_of_onset <= self.sim.date <= person.gi_date_end_of_last_episode)
        ):
            return

        # Do nothing if the episode of diarrhoea is not caused by this module (e.g., spurious symptoms)
        if 'Diarrhoea' not in self.sim.modules['SymptomManager'].causes_of(person_id, 'diarrhoea'):
            return

        # Resolve all the symptoms and reset the properties
        self.module.end_episode(person_id=person_id, outcome='cure')


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
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Run `do_treatment` for this person from an out-potient setting."""

        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        self.module.do_treatment(person_id=person_id, hsi_event=self)


class HSI_Diarrhoea_Treatment_Inpatient(HSI_Event, IndividualScopeEventMixin):
    """
    This is a treatment for acute diarrhoea with severe dehydration administered at inpatient when danger_signs have
    been diagnosed.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Diarrhoea_Treatment_Inpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'InpatientDays': 2, 'IPAdmission': 1})
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 2})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Run `do_treatment` for this person from an in-potient setting."""

        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return

        self.module.do_treatment(person_id=person_id, hsi_event=self)


# ---------------------------------------------------------------------------------------------------------
#   HELPER MODULES AND METHODS FOR USE IN TESTING
# ---------------------------------------------------------------------------------------------------------

class DiarrhoeaPropertiesOfOtherModules(Module):
    """For the purpose of the testing, this module generates the properties upon which the Alri module relies"""

    INIT_DEPENDENCIES = {'Demography'}

    # Though this module provides some properties from NewbornOutcomes we do not list
    # NewbornOutcomes in the ALTERNATIVE_TO set to allow using in conjunction with
    # SimplifiedBirths which can also be used as an alternative to NewbornOutcomes
    ALTERNATIVE_TO = {'Alri', 'Epi', 'Hiv', 'Stunting', 'Wasting'}

    PROPERTIES = {
        'hv_inf': Property(Types.BOOL, 'temporary property for HIV infection status'),
        'hv_art': Property(Types.CATEGORICAL, 'temporary property for ART status',
                           categories=["not", "on_VL_suppressed", "on_not_VL_suppressed"]),
        'ri_current_infection_status': Property(Types.BOOL, 'temporary property'),
        'nb_breastfeeding_status': Property(Types.CATEGORICAL, 'temporary property',
                                            categories=['none', 'non_exclusive', 'exclusive']),
        'un_clinical_acute_malnutrition': Property(Types.CATEGORICAL, 'temporary property',
                                                   categories=['MAM', 'SAM', 'well']),
        'un_HAZ_category': Property(Types.CATEGORICAL, 'temporary property',
                                    categories=['HAZ<-3', '-3<=HAZ<-2', 'HAZ>=-2']),
        'va_rota_all_doses': Property(Types.BOOL, 'temporary property')
    }

    def __init__(self, name=None):
        super().__init__(name)

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        df = population.props
        df.loc[df.is_alive, 'hv_inf'] = False
        df.loc[df.is_alive, 'hv_art'] = 'not'
        df.loc[df.is_alive, 'ri_current_infection_status'] = False
        df.loc[df.is_alive, 'nb_breastfeeding_status'] = 'non_exclusive'
        df.loc[df.is_alive, 'un_clinical_acute_malnutrition'] = 'well'
        df.loc[df.is_alive, 'un_HAZ_category'] = 'HAZ>=-2'
        df.loc[df.is_alive, 'va_rota_all_doses'] = False

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        df = self.sim.population.props
        df.at[child, 'hv_inf'] = False
        df.at[child, 'hv_art'] = 'not'
        df.at[child, 'ri_current_infection_status'] = False
        df.at[child, 'nb_breastfeeding_status'] = 'non_exclusive'
        df.at[child, 'un_clinical_acute_malnutrition'] = 'well'
        df.at[child, 'un_HAZ_category'] = 'HAZ>=-2'
        df.at[child, 'va_rota_all_doses'] = False


class DiarrhoeaCheckPropertiesEvent(RegularEvent, PopulationScopeEventMixin):
    """This event runs daily and checks properties are in the right configuration. Only use whilst debugging!
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=1))

    def apply(self, population):
        self.module.check_properties()


def increase_incidence_of_pathogens(diarrhoea_module):
    """Helper function to increase the incidence of pathogens and symptoms onset with Diarrhoea."""

    # Increase incidence of pathogens (such that almost certain to get at least one pathogen each year)
    pathogens = diarrhoea_module.pathogens
    for pathogen in pathogens:
        diarrhoea_module.parameters[f"base_inc_rate_diarrhoea_by_{pathogen}"] = \
            [0.95 / len(diarrhoea_module.pathogens)] * 3

    probs = pd.DataFrame(
        [diarrhoea_module.parameters[f"base_inc_rate_diarrhoea_by_{pathogen}"] for pathogen in pathogens]
    )
    assert np.isclose(probs.sum(axis=0), 0.95).all()

    # Increase symptoms so that everyone gets symptoms:
    for param_name in diarrhoea_module.parameters:
        if param_name.startswith('proportion_AWD_by_'):
            diarrhoea_module.parameters[param_name] = 1.0
        if param_name.startswith('fever_by_'):
            diarrhoea_module.parameters[param_name] = 1.0
        if param_name.startswith('vomiting_by_'):
            diarrhoea_module.parameters[param_name] = 1.0
        if param_name.startswith('dehydration_by_'):
            diarrhoea_module.parameters[param_name] = 1.0


def increase_risk_of_death(diarrhoea_module):
    """Helper function to increase death and make it dependent on dehydration and blood-in-diarrhoea that are cured by
     treatment"""

    diarrhoea_module.parameters['case_fatality_rate_AWD'] = 0.0001
    diarrhoea_module.parameters['rr_diarr_death_bloody'] = 1000
    diarrhoea_module.parameters['rr_diarr_death_severe_dehydration'] = 1000
    diarrhoea_module.parameters['rr_diarr_death_age24to59mo'] = 1.0
    diarrhoea_module.parameters['rr_diarr_death_if_duration_longer_than_13_days'] = 1.0
    diarrhoea_module.parameters['rr_diarr_death_untreated_HIV'] = 1.0
    diarrhoea_module.parameters['rr_diarr_death_SAM'] = 1.0
    diarrhoea_module.parameters['rr_diarr_death_alri'] = 1.0
    diarrhoea_module.parameters['rr_diarr_death_cryptosporidium'] = 1.0
    diarrhoea_module.parameters['rr_diarr_death_shigella'] = 1.0


def make_treatment_perfect(diarrhoea_module):
    """Apply perfect efficacy for treatments"""
    diarrhoea_module.parameters['prob_WHOPlanC_cures_dehydration_if_severe_dehydration'] = 1.0
    diarrhoea_module.parameters['prob_ORS_cures_dehydration_if_severe_dehydration'] = 1.0
    diarrhoea_module.parameters['prob_ORS_cures_dehydration_if_non_severe_dehydration'] = 1.0
    diarrhoea_module.parameters['prob_antibiotic_cures_dysentery'] = 1.0

    # Apply perfect assessment and referral
    diarrhoea_module.parameters['prob_hospitalization_referral_for_severe_diarrhoea'] = 1.0
    diarrhoea_module.parameters['sensitivity_severe_dehydration_visual_inspection'] = 1.0
    diarrhoea_module.parameters['specificity_severe_dehydration_visual_inspection'] = 1.0
