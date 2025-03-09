"""Childhood wasting module"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.methods.symptommanager import Symptom

if TYPE_CHECKING:
    from tlo.methods.hsi_generic_first_appts import HSIEventScheduler
    from tlo.population import IndividualProperties

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------
class Wasting(Module, GenericFirstAppointmentsMixin):
    """
    This module applies the prevalence of wasting at the population-level, based on the Malawi DHS Survey 2015-2016.
    The definitions:
    - moderate wasting: weight_for_height Z-score (WHZ) < -2 SD from the reference mean
    - severe wasting: weight_for_height Z-score (WHZ) < -3 SD from the reference mean

    """

    INIT_DEPENDENCIES = {'Demography', 'SymptomManager', 'NewbornOutcomes', 'HealthBurden'}

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'Severe Acute Malnutrition': Cause(gbd_causes='Protein-energy malnutrition',
                                           label='Childhood Undernutrition')
    }

    # Declare Causes of Death and Disability
    CAUSES_OF_DISABILITY = {
        'Severe Acute Malnutrition': Cause(gbd_causes='Protein-energy malnutrition',
                                           label='Childhood Undernutrition')
    }

    PARAMETERS = {
        # initial prevalence of wasting by age group
        'prev_init_any_by_agegp': Parameter(
            Types.LIST, 'initial any wasting (WHZ < -2) prevalence in 2010 by age groups'),
        'prev_init_sev_by_agegp': Parameter(
            Types.LIST, 'initial severe wasting (WHZ < -3) prevalence in 2010 by age groups'),
        # effect of risk factors on wasting prevalence
        'or_wasting_hhwealth_Q5': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is poorest Q5, ref group Q1'),
        'or_wasting_hhwealth_Q4': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is poorer Q4, ref group Q1'),
        'or_wasting_hhwealth_Q3': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is middle Q3, ref group Q1'),
        'or_wasting_hhwealth_Q2': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is richer Q2, ref group Q1'),
        'or_wasting_AGA_and_preterm': Parameter(
            Types.REAL, 'odds ratio of wasting if born adequate for gestational age and preterm'),
        'or_wasting_SGA_and_term': Parameter(
            Types.REAL, 'odds ratio of wasting if born small for gestational age and term'),
        'or_wasting_SGA_and_preterm': Parameter(
            Types.REAL, 'odds ratio of wasting if born small for gestational age and preterm'),
        # incidence
        'base_overall_inc_rate_wasting': Parameter(
            Types.REAL, 'base overall monthly moderate wasting incidence rate '
                        '(ref age group <0.5 years old)'),
        'rr_inc_rate_wasting_by_agegp': Parameter(
            Types.LIST, 'relative risk of moderate wasting incidence rate by age group'),
        'rr_wasting_AGA_and_preterm': Parameter(
            Types.REAL, 'relative risk of moderate wasting incidence if born adequate for gestational age '
                        'and preterm'),
        'rr_wasting_SGA_and_term': Parameter(
            Types.REAL, 'relative risk of moderate wasting incidence if born small for gestational age '
                        'and term'),
        'rr_wasting_SGA_and_preterm': Parameter(
            Types.REAL, 'relative risk of moderate wasting incidence if born small for gestational age '
                        'and preterm'),
        'rr_wasting_wealth_level': Parameter(
            Types.REAL, 'relative risk of moderate wasting incidence per 1 unit decrease in wealth level'),
        # progression
        'min_days_duration_of_wasting': Parameter(
            Types.REAL, 'minimum limit for moderate and severe wasting duration episode (days)'),
        'duration_of_untreated_mod_wasting': Parameter(
            Types.REAL, 'duration of untreated moderate wasting (days)'),
        'duration_of_untreated_sev_wasting': Parameter(
            Types.REAL, 'duration of untreated severe wasting (days)'),
        'progression_severe_wasting_monthly_by_agegp': Parameter(
            Types.LIST, 'monthly progression rate to severe wasting by age group'),
        'prob_complications_in_SAM': Parameter(
            Types.REAL, 'probability of medical complications in a SAM episode'),
        'duration_sam_to_death': Parameter(
            Types.REAL, 'duration of SAM till death if supposed to die due to untreated SAM (days)'),
        'base_death_rate_untreated_SAM': Parameter(
            Types.REAL, 'base death rate due to untreated SAM (reference age group <0.5 months old)'),
        'rr_death_rate_by_agegp': Parameter(
            Types.LIST, 'list with relative risks of death due to untreated SAM by age group, reference '
                        'group <0.5 months old'),
        # MUAC distributions
        'proportion_WHZ<-3_with_MUAC<115mm': Parameter(
            Types.REAL, 'proportion of individuals with severe wasting who have MUAC < 115 mm'),
        'proportion_-3<=WHZ<-2_with_MUAC<115mm': Parameter(
            Types.REAL, 'proportion of individuals with moderate wasting who have MUAC < 115 mm'),
        'proportion_-3<=WHZ<-2_with_MUAC_[115-125)mm': Parameter(
            Types.REAL, 'proportion of individuals with moderate wasting who have 115 mm ≤ MUAC < 125 mm'),
        'proportion_mam_with_MUAC_[115-125)mm_and_normal_whz': Parameter(
            Types.REAL, 'proportion of individuals with MAM who have 115 mm ≤ MUAC < 125 mm and normal/mild'
                        ' WHZ'),
        'proportion_mam_with_MUAC_[115-125)mm_and_-3<=WHZ<-2': Parameter(
            Types.REAL, 'proportion of individuals with MAM who have both 115 mm ≤ MUAC < 125 mm and moderate'
                        ' wasting'),
        'proportion_mam_with_-3<=WHZ<-2_and_normal_MUAC': Parameter(
            Types.REAL, 'proportion of individuals with MAM who have moderate wasting and normal MUAC'),
        'MUAC_distribution_WHZ<-3': Parameter(
            Types.LIST,
            'mean and standard deviation of a normal distribution of MUAC measurements for WHZ < -3'),
        'MUAC_distribution_-3<=WHZ<-2': Parameter(
            Types.LIST,
            'mean and standard deviation of a normal distribution of MUAC measurements for -3 <= WHZ < -2'),
        'MUAC_distribution_WHZ>=-2': Parameter(
            Types.LIST,
            'mean and standard deviation of a normal distribution of MUAC measurements for WHZ >= -2'),
        # nutritional oedema
        'prevalence_nutritional_oedema': Parameter(
            Types.REAL, 'prevalence of nutritional oedema'),
        'proportion_WHZ<-2_with_oedema': Parameter(
            Types.REAL, 'proportion of individuals with wasting (moderate or severe) who have oedema'),
        'proportion_oedema_with_WHZ<-2': Parameter(
            Types.REAL, 'proportion of individuals with oedema who are wasted (moderately or severely)'),
        # detection
        'growth_monitoring_frequency_days_agecat': Parameter(
            Types.LIST, 'growth monitoring frequency (days) for age categories '),
        'growth_monitoring_attendance_prob_agecat': Parameter(
            Types.LIST, 'probability to attend the growth monitoring for age categories'),
        # recovery due to treatment/interventions
        'recovery_rate_with_soy_RUSF': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with soy RUSF'),
        'recovery_rate_with_CSB++': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with CSB++'),
        'recovery_rate_with_standard_RUTF': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with standard RUTF'),
        'recovery_rate_with_inpatient_care': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with inpatient care'),
        'tx_length_weeks_SuppFeedingMAM': Parameter(
            Types.REAL, 'number of weeks the patient receives treatment in the Supplementary Feeding '
                        'Programme for MAM before being discharged'),
        'tx_length_weeks_OutpatientSAM': Parameter(
            Types.REAL, 'number of weeks the patient receives treatment in the Outpatient Therapeutic '
                        'Programme for SAM before being discharged if they do not die beforehand'),
        'tx_length_weeks_InpatientSAM': Parameter(
            Types.REAL, 'number of weeks the patient receives treatment in the Inpatient Care for complicated'
                        ' SAM before being discharged if they do not die beforehand'),
        # treatment/intervention outcomes
        'prob_death_after_SAMcare': Parameter(
            Types.REAL, 'probability of dying from SAM after receiving care'),
    }

    PROPERTIES = {
        # Properties related to wasting
        'un_ever_wasted': Property(Types.BOOL, 'ever had an episode of wasting (WHZ < -2)'),
        'un_WHZ_category': Property(Types.CATEGORICAL, 'weight-for-height Z-score category',
                                    categories=['WHZ<-3', '-3<=WHZ<-2', 'WHZ>=-2']),
        'un_last_wasting_date_of_onset': Property(Types.DATE, 'date of onset of last episode of wasting'),

        # Properties related to clinical acute malnutrition
        'un_clinical_acute_malnutrition': Property(Types.CATEGORICAL, 'clinical acute malnutrition state '
                                                                      'based on WHZ and/or MUAC and/or nutritional '
                                                                      'oedema',
                                                   categories=['MAM', 'SAM', 'well']),
        'un_am_nutritional_oedema': Property(Types.BOOL, 'bilateral pitting oedema present in wasting '
                                                         'episode'),
        'un_am_MUAC_category': Property(Types.CATEGORICAL, 'MUAC measurement categories, based on WHO '
                                                           'cut-offs',
                                        categories=['<115mm', '[115-125)mm', '>=125mm']),
        'un_sam_with_complications': Property(Types.BOOL, 'medical complications in SAM episode'),
        'un_sam_death_date': Property(Types.DATE, 'if alive, scheduled death date from SAM if not recovers '
                                                  'with treatment; if not alive, date of death due to SAM'),
        'un_am_recovery_date': Property(Types.DATE, 'recovery date from last acute malnutrition episode'),
        # Properties related to treatment
        'un_am_tx_start_date': Property(Types.DATE, 'treatment start date, if currently on treatment'),
        'un_am_treatment_type': Property(Types.CATEGORICAL, 'treatment type for acute malnutrition the person'
                                         ' is currently on; set to not_applicable if well hence no treatment required',
                                         categories=['standard_RUTF', 'soy_RUSF', 'CSB++', 'inpatient_care'] + [
                                             'none', 'not_applicable']),
        'un_am_discharge_date': Property(Types.DATE, 'planned discharge date from current treatment '
                                                     'when recovery will happen if not yet recovered'),
        # Properties to help cancel events
        'un_recov_to_mam_to_cancel': Property(Types.LIST, 'list of dates of scheduled natural recovery to be '
                                                       'canceled for the person'),
        'un_full_recov_to_cancel': Property(Types.LIST, 'list of dates of scheduled recovery with tx '
                                                        'to be canceled for the person'),
        'un_progression_to_cancel': Property(Types.LIST, 'list of dates of scheduled progression to severe '
                                                         'wasting to be canceled for the person'),
        # Property to avoid double non-emergency appt on the same date
        'un_last_nonemergency_appt_date': Property(Types.DATE, 'last date of generic non-emergency first '
                                                               'appointment'),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.prob_normal_whz = None
        self.wasting_models = None
        self.resourcefilepath = resourcefilepath
        # wasting states
        self.wasting_states = self.PROPERTIES["un_WHZ_category"].categories
        # wasting symptom
        self.wasting_symptom = 'weight_loss'

        # dict to hold counters for the number of episodes by wasting-type and age-group
        blank_inc_counter = dict(
            zip(self.wasting_states, [list() for _ in self.wasting_states]))
        self.wasting_incident_case_tracker_blank = {
            _agrp: copy.deepcopy(blank_inc_counter) for _agrp in ['0y', '1y', '2y', '3y', '4y', '5+y']}
        self.wasting_incident_case_tracker = copy.deepcopy(self.wasting_incident_case_tracker_blank)

        self.recovery_options = ['mod_MAM_nat_full_recov',
                                 'mod_SAM_nat_recov_to_MAM',
                                 'sev_SAM_nat_recov_to_MAM',
                                 'mod_MAM_tx_full_recov',
                                 'mod_SAM_tx_full_recov', 'mod_SAM_tx_recov_to_MAM',
                                 'sev_SAM_tx_full_recov', 'sev_SAM_tx_recov_to_MAM',
                                 'mod_not_yet_recovered',
                                 'sev_not_yet_recovered']
        blank_length_counter = dict(
            zip(self.recovery_options, [list() for _ in self.recovery_options]))
        self.wasting_length_tracker_blank = {
            _agrp: copy.deepcopy(blank_length_counter) for _agrp in ['0y', '1y', '2y', '3y', '4y', '5+y']}
        self.wasting_length_tracker = copy.deepcopy(self.wasting_length_tracker_blank)

        self.person_of_interest_id = 4118 # debugging

        # define age groups
        self.age_grps = {0: '0y', 1: '1y', 2: '2y', 3: '3y', 4: '4y'}
        self.age_gps_range_mo = []
        for low_bound_mos, high_bound_mos in [(0, 5), (6, 11), (12, 23), (24, 35), (36, 47), (48, 59)]:  # in months
            agegp_i = f'{low_bound_mos}_{high_bound_mos}mo'
            self.age_gps_range_mo.append(agegp_i)

    def read_parameters(self, data_folder):
        """
        :param data_folder: path of a folder supplied to the Simulation containing data files. Typically,
        modules would read a particular file within here.
        """
        # Read parameters from the resource file
        self.load_parameters_from_dataframe(
            pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_Wasting.csv')
        )

        # Register wasting symptom (weight loss) in Symptoms Manager with high odds of seeking care
        self.sim.modules["SymptomManager"].register_symptom(
            Symptom(
                name=self.wasting_symptom,
                odds_ratio_health_seeking_in_children=20.0,
            )
        )

    def initialise_population(self, population):
        """
        Set our property values for the initial population. This method is called by the simulation when creating
        the initial population, and is responsible for assigning initial values, for every individual,
        of those properties 'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population:
        """
        df = population.props
        p = self.parameters
        print("\nPARAMETERS:")
        print(f"{p['base_death_rate_untreated_SAM']=}")
        print(f"mod_wast_incidence__coef={p['base_overall_inc_rate_wasting']/0.019}")
        print(f"progression_to_sev_wast__coef={p['progression_severe_wasting_monthly_by_agegp'][0]/0.0144}")
        if p['base_death_rate_untreated_SAM'] != 0:
            print("prob_death_after_SAMcare__as_prop_of_death_rate_untreated_sam="
                  f"{p['prob_death_after_SAMcare']*(1-0.738)/p['base_death_rate_untreated_SAM']}")
        print("-----------")
        print(f"{p['base_overall_inc_rate_wasting']=}")
        print("base inc rates by age group: "
              f"{[s * p['base_overall_inc_rate_wasting'] for s in p['rr_inc_rate_wasting_by_agegp']]}")
        print(f"{p['progression_severe_wasting_monthly_by_agegp']=}")
        print(f"{p['prob_death_after_SAMcare']=}")

        # Adjust monthly severe wasting incidence to the duration of untreated moderate wasting
        p['progression_severe_wasting_by_agegp'] = \
            [s/30.4375*p['duration_of_untreated_mod_wasting'] for s in p['progression_severe_wasting_monthly_by_agegp']]
        print(f"{p['progression_severe_wasting_by_agegp']=}")

        # Set initial properties
        df.loc[df.is_alive, 'un_ever_wasted'] = False
        df.loc[df.is_alive, 'un_WHZ_category'] = 'WHZ>=-2'  # not undernourished
        # df.loc[df.is_alive, 'un_last_wasting_date_of_onset'] = pd.NaT
        df.loc[df.is_alive, 'un_clinical_acute_malnutrition'] = 'well'
        df.loc[df.is_alive, 'un_am_nutritional_oedema'] = False
        df.loc[df.is_alive, 'un_am_MUAC_category'] = '>=125mm'
        # df.loc[df.is_alive, 'un_sam_death_date'] = pd.NaT
        # df.loc[df.is_alive, 'un_am_recovery_date'] = pd.NaT
        # df.loc[df.is_alive, 'un_am_discharge_date'] = pd.NaT
        # df.loc[df.is_alive, 'un_am_tx_start_date'] = pd.NaT
        df.loc[df.is_alive, 'un_am_treatment_type'] = 'not_applicable'
        df.loc[df.is_alive, 'un_recov_to_mam_to_cancel'] = \
            df.loc[df.is_alive, 'un_recov_to_mam_to_cancel'].apply(lambda x: [])
        df.loc[df.is_alive, 'un_full_recov_to_cancel'] = \
            df.loc[df.is_alive, 'un_full_recov_to_cancel'].apply(lambda x: [])
        df.loc[df.is_alive, 'un_progression_to_cancel'] = \
            df.loc[df.is_alive, 'un_progression_to_cancel'].apply(lambda x: [])
        # df.loc[df.is_alive, 'un_last_nonemergency_appt_date']= pd.NaT

        # initialise wasting linear models.
        self.wasting_models = WastingModels(self)

        # Assign wasting categories in young children at initiation
        for low_bound_mos, high_bound_mos in [(0, 5), (6, 11), (12, 23), (24, 35), (36, 47), (48, 59)]:  # in months
            low_bound_age_in_years = low_bound_mos / 12.0
            high_bound_age_in_years = (1 + high_bound_mos) / 12.0
            # linear model external variables
            agegp = f'{low_bound_mos}_{high_bound_mos}mo'
            children_of_agegp = df.loc[df.is_alive & df.age_exact_years.between(
                low_bound_age_in_years, high_bound_age_in_years, inclusive='left'
            )]

            # apply prevalence of wasting and categorise into moderate (-3 <= WHZ < -2) or severe (WHZ < -3) wasting
            wasted = \
                self.wasting_models.get_wasting_prevalence(agegp=agegp, children_of_agegp=children_of_agegp).predict(
                    children_of_agegp, self.rng, False
                )
            probability_of_severe = self.get_prob_severe_wasting_among_wasted(agegp=agegp)
            for idx in children_of_agegp.index[wasted]:
                wasted_category = self.rng.choice(['WHZ<-3', '-3<=WHZ<-2'], p=[probability_of_severe,
                                                                               1 - probability_of_severe])
                df.at[idx, 'un_WHZ_category'] = wasted_category
                df.at[idx, 'un_last_wasting_date_of_onset'] = self.sim.date
                df.at[idx, 'un_ever_wasted'] = True

        index_under5 = df.index[df.is_alive & (df.age_exact_years < 5)]
        # calculate approximation of probability of having normal WHZ in children under 5 to be used later
        self.prob_normal_whz = \
            len(index_under5.intersection(df.index[df.un_WHZ_category == 'WHZ>=-2'])) / len(index_under5)
        # ----------------------------------------------------------------------------------------------------- #
        # # # #    Give MUAC category, presence of oedema, and determine acute malnutrition state         # # # #
        # # # #    and, in SAM cases, determine presence of complications and eventually schedule death   # # # #
        self.clinical_signs_acute_malnutrition(index_under5)

        print(f"{self.person_of_interest_id=}")
        print("--------------------------------------")

    def initialise_simulation(self, sim):
        """Prepares for simulation. Schedules:
        * the first growth monitoring to happen straight away, scheduled monthly to detect new cases for treatment.
        * the main incidence polling event.
        * the main logging event.
        """

        sim.schedule_event(Wasting_InitLoggingEvent(self), sim.date)
        sim.schedule_event(Wasting_InitiateGrowthMonitoring(self), sim.date)
        sim.schedule_event(Wasting_IncidencePoll(self), sim.date + DateOffset(months=3))
        sim.schedule_event(Wasting_LoggingEvent(self), sim.date + DateOffset(years=1) - DateOffset(days=1))
        # sim.schedule_event(PrintPersonPropertiesEventIfUpdated(self, self.person_of_interest_id),
        #                    sim.date + DateOffset(days=1))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        # Set initial properties
        df.at[child_id, 'un_ever_wasted'] = False
        df.at[child_id, 'un_WHZ_category'] = 'WHZ>=-2'  # not undernourished
        # df.at[child_id, 'un_last_wasting_date_of_onset'] = pd.NaT
        df.at[child_id, 'un_clinical_acute_malnutrition'] = 'well'
        df.at[child_id, 'un_am_nutritional_oedema'] = False
        df.at[child_id, 'un_am_MUAC_category'] = '>=125mm'
        # df.loc[df.is_alive, 'un_sam_death_date'] = pd.NaT
        # df.loc[df.is_alive, 'un_am_recovery_date'] = pd.NaT
        # df.loc[df.is_alive, 'un_am_discharge_date'] = pd.NaT
        # df.loc[df.is_alive, 'un_am_tx_start_date'] = pd.NaT
        df.at[child_id, 'un_am_treatment_type'] = 'not_applicable'
        df.at[child_id, 'un_recov_to_mam_to_cancel'] = []
        df.at[child_id, 'un_full_recov_to_cancel'] = []
        df.at[child_id, 'un_progression_to_cancel'] = []
        # df.at[child_id, 'un_last_nonemergency_appt_date']= pd.NaT

        # initiate growth monitoring from day 1
        self.sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_event=HSI_Wasting_GrowthMonitoring(module=self, person_id=child_id),
            priority=2, topen=self.sim.date + pd.DateOffset(days=1)
        )

    def get_prob_severe_wasting_among_wasted(self, agegp: str) -> Union[float, int]:
        """
        This function will return the probability of severe wasting for those with wasting status within the agegp
        :param agegp: age group defined be range in months
        :return: probability of severe wasting among all wasting cases in the agegp
        """
        # determine age group
        i = self.age_gps_range_mo.index(agegp)

        # get probability of any wasting (WHZ < -2) for the age group
        probability_less_than_minus2sd = self.parameters['prev_init_any_by_agegp'][i]

        # get probability of severe wasting: WHZ < -3
        probability_less_than_minus3sd = self.parameters['prev_init_sev_by_agegp'][i]

        # make WHZ < -2 as the 100% and get the adjusted probability of severe wasting within overall wasting
        # return the probability of severe wasting among all wasting cases for the agegp
        return probability_less_than_minus3sd / probability_less_than_minus2sd


    def get_odds_wasting(self, agegp: str) -> Union[float, int]:
        """
        This function will calculate the WHZ scores by categories and return odds of wasting
        :param agegp: age grouped in months
        :return: odds of wasting among all children of the agegp
        """
        # determine age group
        i = self.age_gps_range_mo.index(agegp)

        # get probability of any wasting (WHZ < -2) for the age group
        probability_less_than_minus2sd = self.parameters['prev_init_any_by_agegp'][i]

        # convert probability of wasting to odds and return the odds of wasting
        return probability_less_than_minus2sd / (1 - probability_less_than_minus2sd)

    def muac_cutoff_by_WHZ(self, idx, whz):
        """
        Proportion of MUAC < 115 mm in WHZ < -3 and -3 <= WHZ < -2,
        and proportion of wasted children with oedematous malnutrition (kwashiorkor, marasmic kwashiorkor)

        :param idx: index of children ages 6-59 months or person_id
        :param whz: weight for height category
        """
        df = self.sim.population.props
        p = self.parameters

        # ----- MUAC distribution for severe wasting (WHZ < -3) ------
        if whz == 'WHZ<-3':
            # for severe wasting assumed no MUAC >= 125mm
            prop_severe_wasting_with_muac_between_115and125mm = 1 - p['proportion_WHZ<-3_with_MUAC<115mm']

            df.loc[idx, 'un_am_MUAC_category'] = df.loc[idx].apply(
                lambda x: self.rng.choice(['<115mm', '[115-125)mm'],
                                          p=[p['proportion_WHZ<-3_with_MUAC<115mm'],
                                             prop_severe_wasting_with_muac_between_115and125mm]),
                axis=1
            )

        # ----- MUAC distribution for moderate wasting (-3 <= WHZ < -2) ------
        if whz == '-3<=WHZ<-2':
            prop_moderate_wasting_with_muac_over_125mm = \
                1 - p['proportion_-3<=WHZ<-2_with_MUAC<115mm'] - p['proportion_-3<=WHZ<-2_with_MUAC_[115-125)mm']

            df.loc[idx, 'un_am_MUAC_category'] = df.loc[idx].apply(
                lambda x: self.rng.choice(['<115mm', '[115-125)mm', '>=125mm'],
                                          p=[p['proportion_-3<=WHZ<-2_with_MUAC<115mm'],
                                             p['proportion_-3<=WHZ<-2_with_MUAC_[115-125)mm'],
                                             prop_moderate_wasting_with_muac_over_125mm]),
                axis=1
            )

        # ----- MUAC distribution for WHZ >= -2 -----
        if whz == 'WHZ>=-2':

            muac_distribution_in_well_group = norm(loc=p['MUAC_distribution_WHZ>=-2'][0],
                                                   scale=p['MUAC_distribution_WHZ>=-2'][1])
            # get probabilities of MUAC
            prob_normal_whz_with_muac_over_115mm = muac_distribution_in_well_group.sf(11.5)
            prob_normal_whz_with_muac_over_125mm = muac_distribution_in_well_group.sf(12.5)

            prob_normal_whz_with_muac_less_than_115mm = 1 - prob_normal_whz_with_muac_over_115mm
            prob_normal_whz_with_muac_between_115and125mm = \
                prob_normal_whz_with_muac_over_115mm - prob_normal_whz_with_muac_over_125mm

            df.loc[idx, 'un_am_MUAC_category'] = df.loc[idx].apply(
                lambda x: self.rng.choice(['<115mm', '[115-125)mm', '>=125mm'],
                                          p=[prob_normal_whz_with_muac_less_than_115mm,
                                             prob_normal_whz_with_muac_between_115and125mm,
                                             prob_normal_whz_with_muac_over_125mm]),
                axis=1
            )

    def nutritional_oedema_present(self, idx):
        """
        This function applies the probability of nutritional oedema present in wasting and non-wasted cases
        :param idx: index of children under 5, or person_id
        """
        if len(idx) == 0:
            return
        df = self.sim.population.props
        p = self.parameters

        # Knowing the prevalence of nutritional oedema in under 5 population,
        # apply the probability of oedema in WHZ < -2
        # get those children with wasting
        children_with_wasting = idx.intersection(df.index[df.un_WHZ_category != 'WHZ>=-2'])
        children_without_wasting = idx.intersection(df.index[df.un_WHZ_category == 'WHZ>=-2'])

        # oedema among wasted children
        oedema_in_wasted_children = self.rng.random_sample(size=len(
            children_with_wasting)) < p['proportion_WHZ<-2_with_oedema']
        df.loc[children_with_wasting, 'un_am_nutritional_oedema'] = oedema_in_wasted_children

        # oedema among non-wasted children
        if len(children_without_wasting) == 0:
            return
        # proportion_normalWHZ_with_oedema: P(oedema|WHZ>=-2) =
        # P(oedema & WHZ>=-2) / P(WHZ>=-2) = P(oedema) * [1 - P(WHZ<-2|oedema)] / P(WHZ>=-2)
        proportion_normal_whz_with_oedema = \
            p['prevalence_nutritional_oedema'] * (1 - p['proportion_oedema_with_WHZ<-2']) / self.prob_normal_whz
        oedema_in_non_wasted = self.rng.random_sample(size=len(
            children_without_wasting)) < proportion_normal_whz_with_oedema
        df.loc[children_without_wasting, 'un_am_nutritional_oedema'] = oedema_in_non_wasted

    def clinical_acute_malnutrition_state(self, person_id, pop_dataframe):
        """
        This function will determine the clinical acute malnutrition status (MAM, SAM) based on anthropometric indices
        and presence of nutritional oedema (Kwashiorkor); And help determine whether the individual will have medical
        complications, applicable to SAM cases only, requiring inpatient care.
        :param person_id: individual id
        :param pop_dataframe: population dataframe
        """
        df = pop_dataframe
        p = self.parameters

        whz = df.at[person_id, 'un_WHZ_category']
        muac = df.at[person_id, 'un_am_MUAC_category']
        oedema_presence = df.at[person_id, 'un_am_nutritional_oedema']

        # if person well
        if (whz == 'WHZ>=-2') and (muac == '>=125mm') and (not oedema_presence):
            df.at[person_id, 'un_clinical_acute_malnutrition'] = 'well'
        # if person not well
        else:
            # start without treatment
            df.at[person_id, 'un_am_treatment_type'] = 'none'
            # reset recovery date
            df.at[person_id, 'un_am_recovery_date'] = pd.NaT

            # severe acute malnutrition (SAM): MUAC < 115 mm and/or WHZ < -3 and/or nutritional oedema
            if (muac == '<115mm') or (whz == 'WHZ<-3') or oedema_presence:
                df.at[person_id, 'un_clinical_acute_malnutrition'] = 'SAM'
                # apply symptoms to all SAM cases
                self.wasting_clinical_symptoms(person_id=person_id)

            # otherwise moderate acute malnutrition (MAM)
            else:
                df.at[person_id, 'un_clinical_acute_malnutrition'] = 'MAM'

        if df.at[person_id, 'un_clinical_acute_malnutrition'] == 'SAM':
            # Determine if SAM episode has complications
            if self.rng.random_sample() < p['prob_complications_in_SAM']:
                df.at[person_id, 'un_sam_with_complications'] = True
            else:
                df.at[person_id, 'un_sam_with_complications'] = False
            # Determine whether the SAM leads to death
            death_due_to_sam = self.wasting_models.death_due_to_sam_lm.predict(
                df.loc[[person_id]], rng=self.rng
            )
            if death_due_to_sam:
                outcome_date = self.date_of_death_for_untreated_sam()
                self.sim.schedule_event(
                    event=Wasting_SevereAcuteMalnutritionDeath_Event(module=self, person_id=person_id),
                    date=outcome_date
                )
                df.at[person_id, 'un_sam_death_date'] = outcome_date

        else:
            df.at[person_id, 'un_sam_with_complications'] = False
            # clear all wasting symptoms
            self.sim.modules["SymptomManager"].clear_symptoms(
                person_id=person_id, disease_module=self
            )

        assert not ((df.at[person_id, 'un_clinical_acute_malnutrition'] == 'MAM')
                    and (df.at[person_id, 'un_sam_with_complications'])), f'{person_id=} has MAM with complications.'

    def date_of_outcome_for_untreated_wasting(self, whz_category):
        """
        Helper function to determine the duration of the wasting episode to get date of outcome (recovery, progression,
        or death).
        :param whz_category: 'WHZ<-3', or '-3<=WHZ<-2'
        :return: date of outcome, which can be recovery to no wasting, progression to severe wasting, or death due to
        SAM in cases of moderate wasting; or recovery to moderate wasting or death due to SAM in cases of severe wasting
        """
        p = self.parameters

        # moderate wasting (for progression to severe, or recovery to no wasting) -----
        if whz_category == '-3<=WHZ<-2':
            # Allocate the duration of the moderate wasting episode
            duration_mod_wasting = int(max(p['min_days_duration_of_wasting'], p['duration_of_untreated_mod_wasting']))
            # Allocate a date of outcome (death, progression, or recovery)
            date_of_outcome = self.sim.date + DateOffset(days=duration_mod_wasting)
            return date_of_outcome

        # severe wasting (recovery to moderate wasting) -----
        elif whz_category == 'WHZ<-3':
            # determine the duration of severe wasting episode
            duration_sev_wasting = int(max(p['min_days_duration_of_wasting'], p['duration_of_untreated_sev_wasting']))
            # Allocate a date of outcome (death, progression, or recovery)
            date_of_outcome = self.sim.date + DateOffset(days=duration_sev_wasting)
            return date_of_outcome

    def date_of_death_for_untreated_sam(self):
        """
        Helper function to determine date of death, assuming it occurs earlier than the progression/recovery from any
        wasting, moderate or severe.
        :return: date of death
        """
        p = self.parameters

        duration_sam_to_death = int(min(p['duration_of_untreated_mod_wasting'] - 1,
                                        p['duration_of_untreated_sev_wasting'] - 1,
                                        p['duration_sam_to_death']))
        date_of_death = self.sim.date + DateOffset(days=duration_sam_to_death)
        return date_of_death


    def clinical_signs_acute_malnutrition(self, idx):
        """
        When WHZ changed, update other anthropometric indices and clinical signs (MUAC, oedema) that determine the
        clinical state of acute malnutrition. If SAM, update medical complications. If not SAM, clear symptoms.
        This will include both wasted and non-wasted children with other signs of acute malnutrition.
        :param idx: index of children or person_id less than 5 years old
        """
        df = self.sim.population.props

        # if idx only person_id, transform into an Index object
        if not isinstance(idx, pd.Index):
            idx = pd.Index([idx])

        # give MUAC measurement category for all WHZ, including normal WHZ -----
        for whz in ['WHZ<-3', '-3<=WHZ<-2', 'WHZ>=-2']:
            index_6_59mo_by_whz = idx.intersection(df.index[df.age_exact_years.between(0.5, 5, inclusive='left')
                                                            & (df.un_WHZ_category == whz)])
            self.muac_cutoff_by_WHZ(idx=index_6_59mo_by_whz, whz=whz)

        # determine the presence of nutritional oedema (oedematous malnutrition)
        self.nutritional_oedema_present(idx=idx)

        # determine the clinical acute malnutrition state -----
        for person_id in idx:
            self.clinical_acute_malnutrition_state(person_id=person_id, pop_dataframe=df)

    def report_daly_values(self):
        """
        This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        experienced by persons in the previous month. Only rows for alive-persons must be returned. The names of the
        series of columns is taken to be the label of the cause of this disability. It will be recorded by the
        healthburden module as <ModuleName>_<Cause>.
        :return: current daly values for everyone alive
        """
        # Dict to hold the DALY weights
        daly_wts = dict()

        df = self.sim.population.props
        # Get DALY weights
        get_daly_weight = self.sim.modules['HealthBurden'].get_daly_weight

        daly_wts['mod_wasting_with_oedema'] = get_daly_weight(sequlae_code=461)
        daly_wts['sev_wasting_w/o_oedema'] = get_daly_weight(sequlae_code=462)
        daly_wts['sev_wasting_with_oedema'] = get_daly_weight(sequlae_code=463)

        total_daly_values = pd.Series(data=0.0,
                                      index=df.index[df.is_alive])
        total_daly_values.loc[df.is_alive & (df.un_WHZ_category == 'WHZ<-3') &
                              df.un_am_nutritional_oedema] = daly_wts['sev_wasting_with_oedema']
        total_daly_values.loc[df.is_alive & (df.un_WHZ_category == 'WHZ<-3') &
                              (~df.un_am_nutritional_oedema)] = daly_wts['sev_wasting_w/o_oedema']
        total_daly_values.loc[df.is_alive & (df.un_WHZ_category == '-3<=WHZ<-2') &
                              df.un_am_nutritional_oedema] = daly_wts['mod_wasting_with_oedema']
        return total_daly_values

    def wasting_clinical_symptoms(self, person_id) -> None:
        """
        assign clinical symptoms to new acute malnutrition cases
        :param person_id:
        """
        df = self.sim.population.props
        if df.at[person_id, 'un_clinical_acute_malnutrition'] != 'SAM':
            return

        # apply wasting symptoms to all SAM cases
        self.sim.modules["SymptomManager"].change_symptom(
            person_id=person_id,
            symptom_string=self.wasting_symptom,
            add_or_remove="+",
            disease_module=self
        )

    def do_at_generic_first_appt(
        self,
        person_id: int,
        symptoms: List[str],
        individual_properties: IndividualProperties,
        schedule_hsi_event: HSIEventScheduler,
        **kwargs,
    ) -> None:

        df = self.sim.population.props
        # p = self.parameters

        do_prints = False
        if person_id == self.person_of_interest_id:
            do_prints = True
            print(f"NON-EMERGENCY APPT on {self.sim.date=}")
            print(f"{symptoms=}")

        if (individual_properties['age_years'] >= 5) or \
            (individual_properties['un_last_wasting_date_of_onset'] < individual_properties['un_am_tx_start_date'] <
             self.sim.date) or \
            (self.sim.date == individual_properties['un_last_nonemergency_appt_date']):
            if do_prints:
                print("not going through because")
                if individual_properties["age_years"] >= 5:
                    print(f"person not under 5, {individual_properties['age_years']=}")
                if (individual_properties['un_last_wasting_date_of_onset'] < individual_properties['un_am_tx_start_date'] <
             self.sim.date):
                    print(f"person currently treated, {individual_properties['un_am_treatment_type']=}")
                if self.sim.date == individual_properties['un_last_nonemergency_appt_date']:
                    print("the non-emerg. appt did went through today already")
                print("----------------------------------")
            if self.sim.date == individual_properties['un_last_nonemergency_appt_date']:
                logger.debug(
                    key="non-emergency",
                    data=f"A non-emerg. appt runs again on the same date {self.sim.date=} for the {person_id=}. "
                         "All DOs related to wasting are cancelled."
                )
            return

        # or if HSI was already scheduled due to growth monitoring, won't be checked for acute malnutrition again
        hsi_event_scheduled = [
            ev
            for ev in self.sim.modules["HealthSystem"].find_events_for_person(person_id)
            if isinstance(ev[1], (HSI_Wasting_SupplementaryFeedingProgramme_MAM,
                                  HSI_Wasting_OutpatientTherapeuticProgramme_SAM,
                                  HSI_Wasting_InpatientTherapeuticCare_ComplicatedSAM))
        ]
        if hsi_event_scheduled:
            if do_prints:
                print(f"not going through because {hsi_event_scheduled=} via growth monitoring")
            return

        # track the date of the last non-emergency appt
        df.at[person_id, 'un_last_nonemergency_appt_date'] = self.sim.date

        # get the clinical states
        clinical_am = individual_properties['un_clinical_acute_malnutrition']
        complications = individual_properties['un_sam_with_complications']
        if do_prints:
            print(f"{clinical_am=}, {complications=}")

        # No interventions if well
        if clinical_am == 'well':
            if do_prints:
                print("person is well, hence no outcomes from the appt")
                print("---------------------------------------------")
            return

        # Interventions for MAM
        elif clinical_am == 'MAM':
            # schedule HSI for supplementary feeding program for MAM
            if do_prints:
                print("SFP for MAM scheduled for today")
            schedule_hsi_event(
                hsi_event=HSI_Wasting_SupplementaryFeedingProgramme_MAM(module=self, person_id=person_id),
                priority=0, topen=self.sim.date)

        elif clinical_am == 'SAM':

            # Interventions for uncomplicated SAM
            if not complications:
                # schedule HSI for supplementary feeding program for MAM
                if do_prints:
                    print("OTP for SAM w\out complications scheduled for today")
                schedule_hsi_event(
                    hsi_event=HSI_Wasting_OutpatientTherapeuticProgramme_SAM(module=self, person_id=person_id),
                    priority=0, topen=self.sim.date)

            # Interventions for complicated SAM
            if complications:
                # schedule HSI for supplementary feeding program for MAM
                if do_prints:
                    print("ITC for SAM w\ complications scheduled for today")
                schedule_hsi_event(
                    hsi_event=HSI_Wasting_InpatientTherapeuticCare_ComplicatedSAM(module=self, person_id=person_id),
                    priority=0, topen=self.sim.date)

        if do_prints:
            print("----------------end of non-ermerg appt-----------------------------")

    def do_when_am_treatment(self, person_id, intervention) -> None:
        """
        This function will apply the linear model of recovery based on intervention given
        :param person_id:
        :param intervention:
        """
        df = self.sim.population.props
        p = self.parameters

        do_prints = False
        if person_id == self.person_of_interest_id:
            do_prints = True
            print(f"{self.person_of_interest_id=} RECEIVING TX on {self.sim.date=}")

        # natural progression or recovery is cancelled with the tx and the outcome is fully driven by tx
        self.cancel_future_event(person_id, event_type=Wasting_ProgressionToSevere_Event, do_prints=do_prints)
        self.cancel_future_event(person_id, event_type=Wasting_FullRecovery_Event, do_prints=do_prints)
        self.cancel_future_event(person_id, event_type=Wasting_RecoveryToMAM_Event, do_prints=do_prints)

        # Set the date when the treatment is provided:
        df.at[person_id, 'un_am_tx_start_date'] = self.sim.date
        # Reset tx discharge date
        df.at[person_id, 'un_am_discharge_date'] = pd.NaT
        # Cancel natural death due to SAM with tx
        df.at[person_id, 'un_sam_death_date'] = pd.NaT

        if intervention == 'SFP':
            outcome_date = self.sim.date + DateOffset(weeks=p['tx_length_weeks_SuppFeedingMAM'])

            mam_full_recovery = self.wasting_models.acute_malnutrition_recovery_mam_lm.predict(
                df.loc[[person_id]], self.rng
            )

            if mam_full_recovery:
                # set discharge date and schedule recovery
                df.at[person_id, 'un_am_discharge_date'] = outcome_date
                if do_prints:
                    print(f"scheduled full recovery from MAM with SFP at {df.at[person_id, 'un_am_discharge_date']=}")
                self.sim.schedule_event(
                    event=Wasting_FullRecovery_Event(module=self, person_id=person_id),
                    date=(df.at[person_id, 'un_am_discharge_date'])
                )
            else:
                # remained MAM, send for another SFP
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_Wasting_SupplementaryFeedingProgramme_MAM(module=self, person_id=person_id),
                    priority=0, topen=outcome_date)
                if do_prints:
                    print("remained MAM with SFP")
                    print(f"sent for another SFP on {outcome_date=}")
                return

        elif intervention in ['OTP', 'ITC']:
            if intervention == 'OTP':
                outcome_date = (self.sim.date + DateOffset(weeks=p['tx_length_weeks_OutpatientSAM']))
            else:
                outcome_date = (self.sim.date + DateOffset(weeks=p['tx_length_weeks_InpatientSAM']))

            sam_full_recovery = self.wasting_models.acute_malnutrition_recovery_sam_lm.predict(
                df.loc[[person_id]], self.rng
            )
            if sam_full_recovery:
                df.at[person_id, 'un_am_discharge_date'] = outcome_date
                # schedule full recovery
                self.sim.schedule_event(
                    event=Wasting_FullRecovery_Event(module=self, person_id=person_id),
                    date=outcome_date
                )
                # send for follow-up treatment for MAM
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_Wasting_SupplementaryFeedingProgramme_MAM(module=self, person_id=person_id),
                    priority=0, topen=outcome_date
                )
                if do_prints:
                    print(f"scheduled full recovery from SAM with {intervention=} at {outcome_date=} and sent for"
                          "follow-up MAM tx")

            else:
                outcome = self.rng.choice(['recovery_to_mam', 'death'],
                                          p=[
                                              1-self.parameters['prob_death_after_SAMcare'],
                                              self.parameters['prob_death_after_SAMcare']
                                          ])
                if outcome == 'death':
                    if do_prints:
                        print(f"death due to SAM with {intervention=} at {outcome_date=}")
                    self.sim.schedule_event(
                        event=Wasting_SevereAcuteMalnutritionDeath_Event(module=self, person_id=person_id),
                        date=outcome_date
                    )
                    df.at[person_id, 'un_sam_death_date'] = outcome_date
                else:  # recovery to MAM and send for treatment for MAM
                    df.at[person_id, 'un_am_discharge_date'] = outcome_date
                    if do_prints:
                        print(f"recovery to MAM with {intervention=} scheduled at {outcome_date=} "
                              "and sent for MAM tx")
                    self.sim.schedule_event(event=Wasting_RecoveryToMAM_Event(module=self, person_id=person_id),
                                            date=outcome_date)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        hsi_event=HSI_Wasting_SupplementaryFeedingProgramme_MAM(module=self, person_id=person_id),
                        priority=0, topen=outcome_date)

    def cancel_future_event(self, person_id, event_type, do_prints: bool) -> None:
        """
        This function will add dates of recovery and/or progression events that need to be canceled.
        :param person_id:
        :param event_type: which event type to cancel
        :param do_prints: prints for person_of_interest only
        """

        df = self.sim.population.props

        event_tuples = [event_tuple for event_tuple in self.sim.find_events_for_person(person_id)
                        if isinstance(event_tuple[1], event_type)]
        if event_tuples:
            dates = [event_tuple[0] for event_tuple in event_tuples]
            event_type_map = {
                Wasting_RecoveryToMAM_Event: 'un_recov_to_mam_to_cancel',
                Wasting_FullRecovery_Event: 'un_full_recov_to_cancel',
                Wasting_ProgressionToSevere_Event: 'un_progression_to_cancel'
            }

            for date in dates:
                df.at[person_id, event_type_map[event_type]].append(date)
                if do_prints:
                    print(f"a natural history {event_type=} for "
                          f"clinical_am={df.at[person_id, 'un_clinical_acute_malnutrition']}, "
                          f"complications={df.at[person_id, 'un_sam_with_complications']} on {date=}\n"
                          " is cancelled due to tx, the health outcome will be driven by the tx")
                    print(f"the {event_type_map[event_type]}: {df.at[person_id, event_type_map[event_type]]}")
        else:
            if do_prints:
                print(f"{event_type=} not scheduled, hence no need to cancel any")


class PrintPersonPropertiesEventIfUpdated(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, frequency=DateOffset(months=1))
        self.person_id = person_id
        self.old_person_properties = None
        self.old_person_scheduled_events = None
        self.old_person_scheduled_hs_events = None

    def apply(self, population):
        df = population.props

        print(f"{df.at[self.person_id, 'age_exact_years']=}, {df.at[self.person_id, 'is_alive']=}")

        # new_person_properties = df.loc[self.person_id]
        # new_person_scheduled_events = self.sim.find_events_for_person(self.person_id)
        # new_person_scheduled_hs_events = self.sim.modules['HealthSystem'].find_events_for_person(self.person_id)
        #
        # print(f"{self.sim.date=}")
        # if self.sim.date == Date(year=2010, month=1, day=1):
        #     pd.set_option('display.max_columns', None)
        #     print(f"Properties of person {self.person_id} at initiation:\n {new_person_properties.to_string()}")
        #     self.old_person_properties = new_person_properties
        #     print(f"Scheduled events for person {self.person_id}:\n {new_person_scheduled_events}")
        #     self.old_person_scheduled_events = new_person_scheduled_events
        #     print(f"HealthSystem events for person {self.person_id}:\n {new_person_scheduled_hs_events}")
        #     self.old_person_scheduled_hs_events = new_person_scheduled_hs_events
        #
        # else:
        #     if not self.old_person_properties.equals(new_person_properties):
        #         changed_columns = new_person_properties[
        #             (new_person_properties != self.old_person_properties) &
        #             ~(new_person_properties.isna() & self.old_person_properties.isna())
        #         ]
        #         print(f"Properties of person {self.person_id} that have changed:\n {changed_columns.to_string()}")
        #         self.old_person_properties = new_person_properties
        #     if self.old_person_scheduled_events != new_person_scheduled_events:
        #         print(f"Scheduled events for person {self.person_id} changed to:\n {new_person_scheduled_events}")
        #         self.old_person_scheduled_events = new_person_scheduled_events
        #     if self.old_person_scheduled_hs_events != new_person_scheduled_hs_events:
        #         print(f"Scheduled events for person {self.person_id} changed to:\n {new_person_scheduled_hs_events}")
        #         self.old_person_scheduled_hs_events = new_person_scheduled_hs_events

class Wasting_IncidencePoll(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that determines new cases of wasting (WHZ < -2) to the under-5 population, and schedules
    individual incident cases to represent onset. It determines those who will progress to severe wasting
    (WHZ < -3) and schedules the event to update on properties. These are events occurring without the input
    of interventions, these events reflect the natural history only.
    """
    AGE_GROUPS = {0: '0y', 1: '1y', 2: '2y', 3: '3y', 4: '4y'}

    def __init__(self, module):
        """schedule to run every month
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, Wasting)

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        rng = self.module.rng

        # # # INCIDENCE OF MODERATE WASTING # # # # # # # # # # # # # # # # # # # # #
        # Determine who will be onset with wasting among those who are not currently wasted -------------
        not_am_or_treated = df.loc[df.is_alive & (df.age_exact_years < 5) &
                                   (df.un_clinical_acute_malnutrition == 'well') & (df.un_am_tx_start_date.isna())]
        incidence_of_wasting = self.module.wasting_models.wasting_incidence_lm.predict(not_am_or_treated, rng=rng)
        mod_wasting_new_cases = not_am_or_treated.loc[incidence_of_wasting]
        mod_wasting_new_cases_idx = mod_wasting_new_cases.index
        # update the properties for new cases of wasted children
        df.loc[mod_wasting_new_cases_idx, 'un_ever_wasted'] = True
        df.loc[mod_wasting_new_cases_idx, 'un_last_wasting_date_of_onset'] = self.sim.date
        # initiate moderate wasting
        df.loc[mod_wasting_new_cases_idx, 'un_WHZ_category'] = '-3<=WHZ<-2'
        # -------------------------------------------------------------------------------------------
        # Add these incident cases to the tracker
        do_prints = False
        for person_id in mod_wasting_new_cases_idx:
            age_group = Wasting_IncidencePoll.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
            self.module.wasting_incident_case_tracker[age_group]['-3<=WHZ<-2'].append(self.sim.date)
            if person_id == self.module.person_of_interest_id:
                do_prints = True
                print(f"WASTING INCIDENCE on {self.sim.date=}")
                print(f"{mod_wasting_new_cases_idx=}, {age_group=}")
        # Update properties related to clinical acute malnutrition
        # (MUAC, oedema, clinical state of acute malnutrition and if SAM complications and death;
        # clear symptoms if not SAM)
        self.module.clinical_signs_acute_malnutrition(mod_wasting_new_cases_idx)
        if do_prints:
            print(f"assigned am indicators:\n"
                  f" {df.at[self.module.person_of_interest_id, 'un_WHZ_category']=}, "
                  f"{df.at[self.module.person_of_interest_id, 'un_am_nutritional_oedema']=},\n"
                  f" {df.at[self.module.person_of_interest_id, 'un_am_MUAC_category']=}")
            print("am status determined and if SAM, complications and death determined:\n"
                  f" {df.at[self.module.person_of_interest_id, 'un_clinical_acute_malnutrition']=}, "
                  f"{df.at[self.module.person_of_interest_id, 'un_sam_with_complications']=},\n"
                  f" {df.at[self.module.person_of_interest_id, 'un_sam_death_date']=}")
        # -------------------------------------------------------------------------------------------

        outcome_date = self.module.date_of_outcome_for_untreated_wasting(whz_category='-3<=WHZ<-2')

        # # # PROGRESS TO SEVERE WASTING # # # # # # # # # # # # # # # # # #
        # Determine those that will progress to severe wasting (WHZ < -3) and schedule progression event ---------
        progression_severe_wasting = self.module.wasting_models.severe_wasting_progression_lm.predict(
            df.loc[mod_wasting_new_cases_idx], rng=rng, squeeze_single_row_output=False
        )
        if do_prints:
            print(f"{outcome_date=},\n {progression_severe_wasting=}")

        for person_id in mod_wasting_new_cases_idx[progression_severe_wasting]:
            # schedule severe wasting WHZ < -3 onset after duration of untreated moderate wasting
            self.sim.schedule_event(
                event=Wasting_ProgressionToSevere_Event(module=self.module, person_id=person_id), date=outcome_date
            )
            if person_id == self.module.person_of_interest_id:
                print(f"scheduled progression to sev wast at {outcome_date=}")

        # # # MODERATE WASTING NATURAL RECOVERY # # # # # # # # # # # # # #
        # Schedule recovery for those not progressing to severe wasting ---------
        for person_id in mod_wasting_new_cases_idx[~progression_severe_wasting]:
            if df.at[person_id, 'un_clinical_acute_malnutrition'] == 'MAM':
                # schedule full recovery after duration of moderate wasting
                self.sim.schedule_event(event=Wasting_FullRecovery_Event(
                    module=self.module, person_id=person_id), date=outcome_date)
                if person_id == self.module.person_of_interest_id:
                      print(f"scheduled natural full recovery at {outcome_date=} as the person has MAM: "
                          f"{df.at[person_id, 'un_clinical_acute_malnutrition']=}")
            else: # == SAM
                # schedule recovery to MAM after duration of moderate wasting
                self.sim.schedule_event(event=Wasting_RecoveryToMAM_Event(
                    module=self.module, person_id=person_id), date=outcome_date)
                if person_id == self.module.person_of_interest_id:
                    print(f"scheduled recovery to MAM at {outcome_date=} as the person has SAM: "
                          f"{df.at[person_id, 'un_clinical_acute_malnutrition']=}")

        if do_prints:
            print("---------------------------------")


class Wasting_ProgressionToSevere_Event(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of severe wasting (WHZ < -3).
     * Refreshes all the properties so that they pertain to this current episode of wasting
     * Imposes wasting symptom
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Wasting)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        do_prints = False
        if person_id == self.module.person_of_interest_id:
            do_prints = True
            print(f"PROGRESSION TO SEV WAST on {self.sim.date=}")

        if (
            (not df.at[person_id, 'is_alive']) or
            (df.at[person_id, 'age_exact_years'] >= 5) or
            (df.at[person_id, 'un_WHZ_category'] != '-3<=WHZ<-2') or
            (df.at[person_id, 'un_last_wasting_date_of_onset'] < df.at[person_id, 'un_am_tx_start_date'] <
                self.sim.date)
        ):
            if do_prints:
                print("not going through because")
                if not df.at[person_id, 'is_alive']:
                    print("is already dead")
                if df.at[person_id, 'age_exact_years'] >= 5:
                    print(f"is not under 5, {df.at[person_id, 'age_exact_years']=}")
                if df.at[person_id, 'un_WHZ_category'] != '-3<=WHZ<-2':
                    print(f"not moderately wasted, {df.at[person_id, 'un_WHZ_category']=} ")
                if (df.at[person_id, 'un_last_wasting_date_of_onset'] < df.at[person_id, 'un_am_tx_start_date'] <
                self.sim.date):
                    print("is currently treated")
                print("----------------------------------")
            return

        if self.sim.date in df.at[person_id, 'un_progression_to_cancel']:
            df.at[person_id, 'un_progression_to_cancel'].remove(self.sim.date)
            if do_prints:
                print("Natural progression to severe wasting cancelled as person received tx.")
                print("----------------------------------")
            return

        # # # INCIDENCE OF SEVERE WASTING # # # # # # # # # # # # # # # # # # # # #
        # Continue with progression to severe if not treated/recovered
        # update properties
        # - WHZ
        df.at[person_id, 'un_WHZ_category'] = 'WHZ<-3'
        # - MUAC, oedema, clinical state of acute malnutrition, complications, death
        self.module.clinical_signs_acute_malnutrition(person_id)
        if do_prints:
            print("assigned am indicators:\n"
                  f" {df.at[self.module.person_of_interest_id, 'un_WHZ_category']=}, "
                  f"{df.at[self.module.person_of_interest_id, 'un_am_nutritional_oedema']=},\n"
                  f" {df.at[self.module.person_of_interest_id, 'un_am_MUAC_category']=}")
            print("determined am status and if SAM complications and death:\n"
                  f" {df.at[self.module.person_of_interest_id, 'un_clinical_acute_malnutrition']=}, "
                  f"{df.at[self.module.person_of_interest_id, 'un_sam_with_complications']=},\n"
                  f" {df.at[self.module.person_of_interest_id, 'un_sam_death_date']=}")

        # -------------------------------------------------------------------------------------------
        # Add this severe wasting incident case to the tracker
        age_group = Wasting_IncidencePoll.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
        self.module.wasting_incident_case_tracker[age_group]['WHZ<-3'].append(self.sim.date)
        if do_prints:
            print(f"{age_group=}")

        if pd.isnull(df.at[person_id, 'un_sam_death_date']):
            # # # SEVERE WASTING NATURAL RECOVERY # # # # # # # # # # # # # # # #
            # Schedule recovery tom MAM from SAM for those not dying due to SAM
            outcome_date = self.module.date_of_outcome_for_untreated_wasting(whz_category='WHZ<-3')
            self.sim.schedule_event(event=Wasting_RecoveryToMAM_Event(
                module=self.module, person_id=person_id), date=outcome_date)
            if do_prints:
                print(f"natural recovery to MAM scheduled on {outcome_date=}")
        else:
            if do_prints:
                print("death due to SAM scheduled earlier")

        if do_prints:
            print("---------------------------------------------")


class Wasting_SevereAcuteMalnutritionDeath_Event(Event, IndividualScopeEventMixin):
    """
    This event applies the death function
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Wasting)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        do_prints = False
        if person_id == self.module.person_of_interest_id:
            do_prints = True
            print(f"DEATH DUE TO SAM on {self.sim.date=}")

        # The event should not run if the person is not currently alive or doesn't have SAM
        if not df.at[person_id, 'is_alive']:
            if do_prints:
                print("not going through as the person is already dead")
                print("----------------------------------")
            return

        # # Check if this person should still die from SAM and that it should happen now not in the future:
        if (
            pd.isnull(df.at[person_id, 'un_am_recovery_date']) and
            not (df.at[person_id, 'un_am_discharge_date'] > df.at[person_id, 'un_am_tx_start_date']) and
            not pd.isnull(df.at[person_id, 'un_sam_death_date']) and
            df.at[person_id, 'un_sam_death_date'] <= self.sim.date
        ):
            assert df.at[person_id, 'un_clinical_acute_malnutrition'] == 'SAM',\
                f"{person_id=} dying due to SAM while \n{df.at[person_id, 'un_clinical_acute_malnutrition']=}"
            if do_prints:
                print("death still happening,\n ie recovery date = NaT & not (discharge_date > tx_start_date) "
                      "& death_date != NaT and is <= sim.date)")
            # Cause the death to happen immediately
            df.at[person_id, 'un_sam_death_date'] = self.sim.date
            self.sim.modules['Demography'].do_death(
                individual_id=person_id,
                cause='Severe Acute Malnutrition',
                originating_module=self.module)
        else:
            df.at[person_id, 'un_sam_death_date'] = pd.NaT
            if do_prints:
                print("death is not happening because")
                if not pd.isnull(df.at[person_id, 'un_am_recovery_date']):
                    print("the person already recovered and didn't get wasted again since")
                if df.at[person_id, 'un_am_discharge_date'] > df.at[person_id, 'un_am_tx_start_date']:
                    print("discharge_date is set, hence the person should recover due to tx, not to die")
                if pd.isnull(df.at[person_id, 'un_sam_death_date']):
                    print("the death was canceled due to tx")
                if df.at[person_id, 'un_sam_death_date'] > self.sim.date:
                    print("the death was canceled due to tx, but scheduled for later as will die with tx anyway")

        if do_prints:
            print(f"{df.at[person_id, 'un_am_recovery_date']=}, {df.at[person_id, 'un_am_discharge_date']=},\n"
                  f"{df.at[person_id, 'un_am_tx_start_date']=}, {pd.isnull(df.at[person_id, 'un_sam_death_date'])=}")
            print("------------------------------------------------------")

class Wasting_FullRecovery_Event(Event, IndividualScopeEventMixin):
    """
    This event sets acute malnutrition signs back to well state.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Wasting)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        p = self.module.parameters

        if pd.isnull(df.at[person_id, 'un_am_tx_start_date']):
            recov_how = 'nat'
        else:
            recov_how = 'tx'

        do_prints = False
        if person_id == self.module.person_of_interest_id:
            do_prints = True
            print(f"FULL RECOVERY {recov_how=} on {self.sim.date=}")
            print(f"{df.at[person_id, 'un_am_tx_start_date']=}")

        if not df.at[person_id, 'is_alive']:
            if do_prints:
                print("not going through, because the person is already dead")
                print("----------------------------------")
            return

        if self.sim.date in df.at[person_id, 'un_full_recov_to_cancel']:
            df.at[person_id, 'un_full_recov_to_cancel'].remove(self.sim.date)
            if do_prints:
                print("not going through, because the natural recovery was cancelled, the outcome will be driven by tx")
                print("----------------------------------")
            return

        # if not well (i.e. NOT already fully recovered with SAM tx, and send here from follow-up MAM tx)
        if df.at[person_id, 'un_WHZ_category'] != 'WHZ>=-2':
            if df.at[person_id, 'un_WHZ_category'] == '-3<=WHZ<-2':
                recov_opt = f"mod_{df.at[person_id, 'un_clinical_acute_malnutrition']}_{recov_how}_full_recov"
            else: # df.at[person_id, 'un_WHZ_category'] == 'WHZ<-3':
                recov_opt = f"sev_{df.at[person_id, 'un_clinical_acute_malnutrition']}_{recov_how}_full_recov"
            age_group = self.module.age_grps.get(df.loc[person_id].age_years, '5+y')
            wasted_days = (self.sim.date - df.at[person_id, 'un_last_wasting_date_of_onset']).days

            def get_min_length(in_recov_how, in_person_id, in_whz):
                if in_whz == '-3<=WHZ<-2':
                    min_length_nat = p['duration_of_untreated_mod_wasting']
                else:  # in_whz == 'WHZ<=-3'
                    min_length_nat = p['duration_of_untreated_mod_wasting'] + p['duration_of_untreated_sev_wasting'] - 1

                # MAM
                if df.at[in_person_id, 'un_clinical_acute_malnutrition'] == 'MAM':
                    min_length_tx =  p['tx_length_weeks_SuppFeedingMAM'] * 7
                # SAM with complications
                elif df.at[in_person_id, 'un_sam_with_complications']:
                    min_length_tx = p['tx_length_weeks_InpatientSAM'] * 7
                # SAM without complications
                else:
                    min_length_tx = p['tx_length_weeks_OutpatientSAM'] * 7

                if in_recov_how == 'tx':
                    min_length = min_length_tx
                else:  # in_recov_how == 'nat'
                    min_length = min_length_nat
                return min_length - 1

            whz = df.at[person_id, 'un_WHZ_category']
            assert wasted_days >= get_min_length(recov_how, person_id, whz),\
                (f" The {person_id=} is {wasted_days=} < minimal expected length= {get_min_length(recov_how, person_id, whz)} days "
                 f"when {recov_opt=}.")
            self.module.wasting_length_tracker[age_group][recov_opt].append(wasted_days)

            if do_prints:
                print(f"{recov_opt=}, {age_group=}, {wasted_days=} >= min_length= "
                      f"{get_min_length(recov_how, person_id, whz)} days")

        df.at[person_id, 'un_am_recovery_date'] = self.sim.date
        df.at[person_id, 'un_WHZ_category'] = 'WHZ>=-2'  # normal WHZ
        df.at[person_id, 'un_clinical_acute_malnutrition'] = 'well' # well-nourished
        df.at[person_id, 'un_am_nutritional_oedema'] = False # no oedema
        df.at[person_id, 'un_am_MUAC_category'] = '>=125mm' # normal MUAC
        df.at[person_id, 'un_sam_with_complications'] = False
        df.at[person_id, 'un_sam_death_date'] = pd.NaT
        df.at[person_id, 'un_am_tx_start_date'] = pd.NaT
        df.at[person_id, 'un_am_treatment_type'] = 'not_applicable'

        # this will clear all wasting symptoms
        self.sim.modules["SymptomManager"].clear_symptoms(
            person_id=person_id, disease_module=self.module
        )

        if do_prints:
            print("recovered to well with all properties being set and SAM symptoms removed")
            print("-------------------------------------------------")

class Wasting_RecoveryToMAM_Event(Event, IndividualScopeEventMixin):
    """
    This event updates the clinical signs of acute malnutrition for those cases that improved from SAM to MAM.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Wasting)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        rng = self.module.rng
        p = self.module.parameters

        if pd.isnull(df.at[person_id, 'un_am_tx_start_date']):
            recov_how = 'nat'
        else:
            recov_how = 'tx'

        do_prints = False
        if person_id == self.module.person_of_interest_id:
            do_prints = True
            print(f"RECOVERY TO MAM {recov_how=} on {self.sim.date=}")

        # if died or recovered in between, should not update to MAM
        if (not df.at[person_id, 'is_alive']) or (df.at[person_id, 'un_clinical_acute_malnutrition'] != 'SAM'):
            if do_prints:
                print("not going through because")
                if not df.at[person_id, 'is_alive']:
                    print("is already dead")
                if df.at[person_id, 'un_clinical_acute_malnutrition'] != 'SAM':
                    print(f"not having SAM, {df.at[person_id, 'un_clinical_acute_malnutrition']=}")
                print("----------------------------------")
            return

        if self.sim.date in df.at[person_id, 'un_recov_to_mam_to_cancel']:
            df.at[person_id, 'un_recov_to_mam_to_cancel'].remove(self.sim.date)
            if do_prints:
                print("not going through, because the natural recovery was cancelled, the outcome will be driven by tx")
                print("----------------------------------")
            return

        # For cases with normal WHZ and other acute malnutrition signs:
        # oedema, or low MUAC - do not change the WHZ
        whz = df.at[person_id, 'un_WHZ_category']
        if whz == 'WHZ>=-2':
            # MAM by MUAC only
            df.at[person_id, 'un_am_MUAC_category'] = '[115-125)mm'
            # TODO: I think this changes the proportions below as some of the cases will be issued here

        else:
            # using the probability of mam classification by anthropometric indices
            mam_classification = rng.choice(['mam_by_muac_only', 'mam_by_muac_and_whz', 'mam_by_whz_only'],
                                            p=[p['proportion_mam_with_MUAC_[115-125)mm_and_normal_whz'],
                                               p['proportion_mam_with_MUAC_[115-125)mm_and_-3<=WHZ<-2'],
                                               p['proportion_mam_with_-3<=WHZ<-2_and_normal_MUAC']])

            if mam_classification == 'mam_by_muac_only':
                if whz == '-3<=WHZ<-2':
                    recov_opt = f"mod_SAM_{recov_how}_recov_to_MAM"
                else: # whz == 'WHZ<-3':
                    recov_opt = f"sev_SAM_{recov_how}_recov_to_MAM"
                age_group = self.module.age_grps.get(df.loc[person_id].age_years, '5+y')
                wasted_days = (self.sim.date - df.at[person_id, 'un_last_wasting_date_of_onset']).days

                def get_min_length(in_recov_how, in_person_id, in_whz):
                    if in_recov_how == 'tx':
                        if df.at[in_person_id, 'un_sam_with_complications']:
                            return (p['tx_length_weeks_InpatientSAM'] * 7) - 1
                        else:  # SAM without complications
                            return (p['tx_length_weeks_OutpatientSAM'] * 7) - 1
                    else: # in_recov_how == 'nat'
                        if in_whz == '-3<=WHZ<-2':
                            return p['duration_of_untreated_mod_wasting'] - 1
                        else:  # in_whz == 'WHZ<=-3'
                            return p['duration_of_untreated_mod_wasting'] + p['duration_of_untreated_sev_wasting'] - 2

                assert wasted_days >= get_min_length(recov_how, person_id, whz), \
                    (f" The {person_id=} is wasted {wasted_days=} < minimal expected length= "
                     f"{get_min_length(recov_how, person_id, whz)} days when {recov_opt=}.")
                self.module.wasting_length_tracker[age_group][recov_opt].append(wasted_days)

                df.at[person_id, 'un_WHZ_category'] = 'WHZ>=-2'
                df.at[person_id, 'un_am_MUAC_category'] = '[115-125)mm'

                if do_prints:
                    print(f"{recov_opt=}, {age_group=}, {wasted_days=} >= min_length= "
                          f"{get_min_length(recov_how, person_id, whz)} days")

            if mam_classification == 'mam_by_muac_and_whz':
                df.at[person_id, 'un_WHZ_category'] = '-3<=WHZ<-2'
                df.at[person_id, 'un_am_MUAC_category'] = '[115-125)mm'

            if mam_classification == 'mam_by_whz_only':
                df.at[person_id, 'un_WHZ_category'] = '-3<=WHZ<-2'
                df.at[person_id, 'un_am_MUAC_category'] = '>=125mm'

        # Update all other properties equally
        df.at[person_id, 'un_clinical_acute_malnutrition'] = 'MAM'
        df.at[person_id, 'un_am_nutritional_oedema'] = False
        df.at[person_id, 'un_sam_with_complications'] = False
        df.at[person_id, 'un_sam_death_date'] = pd.NaT # death is cancelled if was scheduled
        df.at[person_id, 'un_am_tx_start_date'] = pd.NaT
        # Start without treatment, treatment will be applied with HSI if care sought
        df.at[person_id, 'un_am_treatment_type'] = 'none'

        # this will clear all wasting symptoms (applicable for SAM, not MAM)
        self.sim.modules["SymptomManager"].clear_symptoms(
            person_id=person_id, disease_module=self.module
        )

        if do_prints:
            print(f"wast indicators updated to {df.at[person_id, 'un_WHZ_category']=},"
                  f"{df.at[person_id, 'un_am_MUAC_category']=}, and no oedema => MAM")
            print("------------------------------------------------")


class Wasting_InitiateGrowthMonitoring(Event, PopulationScopeEventMixin):
    # TODO: maybe will be updated to integrate monitoring of < 1y old in epi module, and on birth schedule to be
    #  monitored within wasting module only when > 1y old
    """
    Event that schedules HSI_Wasting_GrowthMonitoring for all under-5 children for a random day within the age-dependent
    frequency.
    """

    def __init__(self, module):
        """Runs only once, when simulation is initiated.
        :param module: the module that created this event
        """
        super().__init__(module)
        assert isinstance(module, Wasting)

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """

        df = population.props
        rng = self.module.rng
        p = self.module.parameters

        # TODO: including treated children? (until there is growth monitoring with tx and scheduled post-tx, yes)
        index_under5 = df.index[df.is_alive & (df.age_exact_years < 5)]
        # and ~df.un_am_treatment_type.isin(['standard_RUTF', 'soy_RUSF', 'CSB++', 'inpatient_care'])

        def get_monitoring_frequency_days(age):
            # TODO: maybe in future 0-1 to be dealt with within epi module
            if age < 1:
                return p['growth_monitoring_frequency_days_agecat'][0]
            elif age <= 2:
                return p['growth_monitoring_frequency_days_agecat'][1]
            else:
                return p['growth_monitoring_frequency_days_agecat'][2]

        # schedule monitoring within age-dependent frequency
        for person_id in index_under5:
            next_event_days = \
                rng.randint(0, max(0, (get_monitoring_frequency_days(df.at[person_id, 'age_exact_years']) - 2)))
            if (df.at[person_id, 'age_exact_years'] + (next_event_days / 365.25)) < 5:
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_Wasting_GrowthMonitoring(module=self.module, person_id=person_id),
                    priority=2, topen=self.sim.date + pd.DateOffset(days=next_event_days)
                )


class HSI_Wasting_GrowthMonitoring(HSI_Event, IndividualScopeEventMixin):
    """ Attendance is determined for the HSI. If the child attends, measurements with available equipment are performed
    for that child. Based on these measurements, the child can be diagnosed as well/MAM/(un)complicated SAM and
    eventually scheduled for the appropriate treatment. If the child (attending or not) is still under 5 at the time of
    the next growth monitoring, the next event is scheduled with age-dependent frequency.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Wasting)

        self.attendance = None

        self.TREATMENT_ID = "Undernutrition_GrowthMonitoring"
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    @property
    def EXPECTED_APPT_FOOTPRINT(self):
        """Return the expected appointment footprint based on attendance at the HSI event."""
        rng = self.module.rng
        p = self.module.parameters
        person_age = self.sim.population.props.loc[self.target].age_exact_years

        # TODO: for now assumed general attendance prob for <1 y old,
        #  later may be excluded from here and be dealt with within epi module
        def get_attendance_prob(age):
            if age < 1:
                return p['growth_monitoring_attendance_prob_agecat'][0]
            if age <= 2:
                return p['growth_monitoring_attendance_prob_agecat'][1]
            else:
                return p['growth_monitoring_attendance_prob_agecat'][2]

        # perform growth monitoring if attending
        self.attendance = rng.random_sample() < get_attendance_prob(person_age)
        if self.attendance:
            return self.make_appt_footprint({'Under5OPD': 1})
        else:
            return self.make_appt_footprint({})

    def apply(self, person_id, squeeze_factor):
        logger.debug(key='debug', data='This is HSI_Wasting_GrowthMonitoring')

        df = self.sim.population.props
        rng = self.module.rng
        p = self.module.parameters

        do_prints = False
        if person_id == self.module.person_of_interest_id:
            do_prints = True
            print(f"GROWTH MONITORING on {self.sim.date=}")

        # TODO: Will they be monitored during the treatment? Can we assume, that after the treatment they will be
        #  always properly checked (all measurements and oedema checked), or should be the assumed "treatment outcome"
        #  be also based on equipment availability and probability of checking oedema? Maybe they should be sent for
        #  after treatment monitoring, where the assumed "treatment outcome" will be determined and follow-up treatment
        #  based on that? - The easiest way (currently coded) is assuming that after treatment all measurements are
        #  done, hence correctly diagnosed. The growth monitoring is scheduled for them as usual, ie, for instance, for
        #  a child 2-5 old, if they were sent for treatment via growth monitoring, they will be on tx for adequate nmb
        #  of weeks, but next monitoring will be done in ~5 months after the treatment. - Or we could schedule for the
        #  treated children a monitoring sooner after the treatment.
        if (not df.at[person_id, 'is_alive']) or (df.at[person_id, 'age_exact_years'] >= 5):
            # or
            # df.at[person_id, 'un_am_treatment_type'].isin(['standard_RUTF', 'soy_RUSF', 'CSB++', 'inpatient_care']):
            if do_prints:
                print("not going through and no more monitoring scheduled because")
                if not df.at[person_id, 'is_alive']:
                    print("already dead")
                if df.at[person_id, 'age_exact_years'] >= 5:
                    print("not under 5")
                print("----------------------------------")
            return

        def schedule_next_monitoring():
            def get_monitoring_frequency_days(age):
                # TODO: in future maybe 0-1 to be dealt with within epi module
                if age < 1:
                    return p['growth_monitoring_frequency_days_agecat'][0]
                elif age <= 2:
                    return p['growth_monitoring_frequency_days_agecat'][1]
                else:
                    return p['growth_monitoring_frequency_days_agecat'][2]

            person_monitoring_frequency = get_monitoring_frequency_days(df.at[person_id, 'age_exact_years'])
            if do_prints:
                print(f"{df.at[person_id, 'age_exact_years']=}, {person_monitoring_frequency=}")
            if (df.at[person_id, 'age_exact_years'] + (person_monitoring_frequency / 365.25)) < 5:
                # schedule next growth monitoring
                if do_prints:
                    print("next growth monitoring scheduled at "
                          f"{(self.sim.date + pd.DateOffset(days=person_monitoring_frequency))=}")
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_Wasting_GrowthMonitoring(module=self.module, person_id=person_id),
                    topen=self.sim.date + pd.DateOffset(days=person_monitoring_frequency),
                    tclose=None,
                    priority=2
                )
            else:
                if do_prints:
                    print("no more growth monitoring scheduled as the age will be above 5")

        # TODO: as stated above, for now we schedule next monitoring for all children, even those sent for treatment
        schedule_next_monitoring()

        # but if they are currently treated, the growth monitoring will not go through
        # TODO: later could be scheduled for monitoring within the tx to use the resources
        if (df.at[person_id, 'un_last_wasting_date_of_onset'] < df.at[person_id, 'un_am_tx_start_date'] <
                self.sim.date):
            if do_prints:
                print("not going through because is currently treated")
                print("-----------------------------------------")
            return
        # or if HSI was already scheduled due to care-seeking, no need to attend the growth monitoring
        hsi_event_scheduled = [
            ev
            for ev in self.sim.modules["HealthSystem"].find_events_for_person(person_id)
            if isinstance(ev[1], (HSI_Wasting_SupplementaryFeedingProgramme_MAM,
                                  HSI_Wasting_OutpatientTherapeuticProgramme_SAM,
                                  HSI_Wasting_InpatientTherapeuticCare_ComplicatedSAM))
        ]
        if hsi_event_scheduled:
            if do_prints:
                print(f"not going through because {hsi_event_scheduled=} via care-seeking")
            return

        if not self.attendance:
            if do_prints:
                print("does not attend to this growth monitoring appt")
                print("-----------------------------------------")
            return

        available_equipment = []
        for equip in ['Height Pole (Stadiometer)', 'Weighing scale', 'MUAC tape']:
            available = rng.random_sample() < HSI_Event.probability_all_equipment_available(self, equip)
            if available:
                available_equipment.append(equip)
        self.add_equipment(set(available_equipment))
        if do_prints:
            print(f"{available_equipment=}")

        def schedule_tx_by_diagnosis(hsi_event):
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=hsi_event(module=self.module, person_id=person_id),
                priority=0, topen=self.sim.date
            )

        complications = df.at[person_id, 'un_sam_with_complications']

        # DIAGNOSIS
        # oedema is assumed to be quite obvious if present
        # in addition, based on performed measurements (depending on what equipment is available)
        if df.at[person_id, 'un_am_nutritional_oedema']:
            diagnosis = 'SAM'
            if do_prints:
                print(f"oedema present => {diagnosis=}")
        else:
            # all equip available and used for diagnosis
            if all(item in available_equipment for item in
                   ['Height Pole (Stadiometer)', 'Weighing scale']):
                diagnosis = df.at[person_id, 'un_clinical_acute_malnutrition']
                if do_prints:
                    print(f"oedema not present, all equip available, hence {diagnosis=} is the actual am state: "
                          f"{df.at[person_id, 'un_clinical_acute_malnutrition']=}")
            # MUAC measurement is solely used for diagnosis, as Height Pole and/or Weighing scale not available
            elif 'MUAC tape' in available_equipment:
                # TODO: rm print
                print("debugging-WARNING: full availability of equip assumed, we should have never get here")
                muac = df.at[person_id, 'un_am_MUAC_category']
                if muac == '>=125mm':
                    diagnosis = 'well'
                elif muac == '<115mm':
                    diagnosis = 'SAM'
                else:
                    diagnosis = 'MAM'
                if do_prints:
                    print("oedema not present, MUAC tape available but heigh pole and/pr weighing scale not, hence "
                          f"{diagnosis=} based on {muac=}, not on actual state "
                          f"{df.at[person_id, 'un_clinical_acute_malnutrition']=}")
            # WHZ score is solely used for diagnosis
            elif all(item in available_equipment for item in
                       ['Height Pole (Stadiometer)', 'Weighing scale']):
                # TODO: rm print
                print("debugging-WARNING: full availability of equip assumed, we should have never get here")
                whz = df.at[person_id, 'un_WHZ_category']
                if whz == 'WHZ>=-2':
                    diagnosis = 'well'
                elif whz == 'WHZ<-3':
                    diagnosis = 'SAM'
                else:
                    diagnosis = 'MAM'
            # WHZ score nor MUAC measurement available, hence diagnosis based solely on presence of oedema and oedema is
            # not present
            else:
                diagnosis = 'well'

        if diagnosis == 'well':
            if do_prints:
                print("diagnosed as being well, hence ntg else going on")
                print("---------------------------------")
            return
        elif diagnosis == 'MAM':
            if do_prints:
                print("MAM diagnosed, send for SFP")
            schedule_tx_by_diagnosis(HSI_Wasting_SupplementaryFeedingProgramme_MAM)
        elif (diagnosis == 'SAM') and (not complications):
            if do_prints:
                print("SAM w\out complications diagnosed, send for OTP")
            schedule_tx_by_diagnosis(HSI_Wasting_OutpatientTherapeuticProgramme_SAM)
        else:  # (diagnosis == 'SAM') and complications:
            if do_prints:
                print("SAM w\ complications diagnosed, send for ITC")
            schedule_tx_by_diagnosis(HSI_Wasting_InpatientTherapeuticCare_ComplicatedSAM)

        if do_prints:
            print("-------------------------------------------")

    def did_not_run(self):
        logger.debug(key="HSI_Wasting_GrowthMonitoring",
                     data="HSI_Wasting_GrowthMonitoring: did not run"
                     )
        pass


class HSI_Wasting_SupplementaryFeedingProgramme_MAM(HSI_Event, IndividualScopeEventMixin):
    """
    This is the supplementary feeding programme for MAM without complications
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources
        # of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one outpatient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Undernutrition_Feeding_Supplementary'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        assert isinstance(self.module, Wasting)

        df = self.sim.population.props
        # p = self.module.parameters

        do_prints = False
        if person_id == self.module.person_of_interest_id:
            do_prints = True
            print(f"SFP APPT on {self.sim.date=}")

        if not df.at[person_id, 'is_alive']:
            if do_prints:
                print("is dead, hence not going through")
                print("------------------------------")
            return

        # Do here whatever happens to an individual during the admission for the treatment
        # ~~~~~~~~~~~~~~~~~~~~~~
        # Perform measurements (height/length), weight, MUAC
        self.add_equipment({'Height Pole (Stadiometer)', 'Weighing scale', 'MUAC tape'})

        # Make request for consumables
        consumables = self.sim.modules['HealthSystem'].parameters['item_and_package_code_lookups']
        # individual items
        item_code1 = pd.unique(consumables.loc[consumables['Items'] ==
                                               'Corn Soya Blend (or Supercereal - CSB++)', 'Item_Code'])[0]

        # check availability of consumables
        if self.get_consumables([item_code1]):
            logger.debug(key='debug', data='consumables are available')
            # Log that the treatment is provided:
            df.at[person_id, 'un_am_treatment_type'] = 'CSB++'
            if do_prints:
                print("consumables available")
            self.module.do_when_am_treatment(person_id, intervention='SFP')
        else:
            logger.debug(key='debug',
                         data=f"Consumable(s) not available, hence {self.TREATMENT_ID} cannot be provided.")
            if do_prints:
                print("consumables not available, SFP tx not scheduled, should be picked up with next\n"
                      "growth monitoring if not naturally recovered in between or could be picked up\n"
                      "with non-emergency appt")

        if do_prints:
            print("-----------------------------------------------")

    def did_not_run(self):
        logger.debug(key='debug', data=f'{self.TREATMENT_ID}: did not run')
        pass


class HSI_Wasting_OutpatientTherapeuticProgramme_SAM(HSI_Event, IndividualScopeEventMixin):
    """
    This is the outpatient management of SAM without any medical complications
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources
        # of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint['U5Malnutr'] = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Undernutrition_Feeding_Outpatient'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        assert isinstance(self.module, Wasting)

        df = self.sim.population.props
        # p = self.module.parameters

        do_prints = False
        if person_id == self.module.person_of_interest_id:
            do_prints = True
            print(f"OTP APPT on {self.sim.date=}")

        if not df.at[person_id, 'is_alive']:
            if do_prints:
                print("dead already, appt not going through")
                print("--------------OTP end1-------------------")
            return

        # Do here whatever happens to an individual during the admission for the treatment
        # ~~~~~~~~~~~~~~~~~~~~~~
        # Perform measurements (height/length), weight, MUAC
        self.add_equipment({'Height Pole (Stadiometer)', 'Weighing scale', 'MUAC tape'})

        # Make request for consumables
        consumables = self.sim.modules['HealthSystem'].parameters[
            'item_and_package_code_lookups']

        # individual items
        item_code1 = pd.unique(consumables.loc[consumables['Items'] ==
                                               'SAM theraputic foods', 'Item_Code'])[0]
        item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'SAM medicines', 'Item_Code'])[0]

        # check availability of consumables
        if self.get_consumables(item_code1) and self.get_consumables(item_code2):
            logger.debug(key='debug', data='consumables are available.')
            # Log that the treatment is provided:
            df.at[person_id, 'un_am_treatment_type'] = 'standard_RUTF'
            if do_prints:
                print("consumables available")
            self.module.do_when_am_treatment(person_id, intervention='OTP')
        else:
            logger.debug(key='debug',
                         data=f"Consumable(s) not available, hence {self.TREATMENT_ID} cannot be provided.")
            if do_prints:
                print("consumables not available, OTP tx not scheduled, should be picked up with next\n"
                      "growth monitoring or non-emergency appt if not naturally recovered or died in\n"
                      "between")

        if do_prints:
            print("----------------------OTP end2-------------------------")

    def did_not_run(self):
        logger.debug(key='debug', data=f'{self.TREATMENT_ID}: did not run')
        pass


class HSI_Wasting_InpatientTherapeuticCare_ComplicatedSAM(HSI_Event, IndividualScopeEventMixin):
    """
    This is the inpatient management of SAM with medical complications
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Undernutrition_Feeding_Inpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"U5Malnutr": 1})
        self.ACCEPTED_FACILITY_LEVEL = '2'
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 7})

    def apply(self, person_id, squeeze_factor):
        assert isinstance(self.module, Wasting)

        df = self.sim.population.props
        # p = self.module.parameters

        do_prints = False
        if person_id == self.module.person_of_interest_id:
            do_prints = True
            print(f"ITC APPT on {self.sim.date=}")

        # Stop the person from dying of acute malnutrition (if they were going to die)
        if not df.at[person_id, 'is_alive']:
            if do_prints:
                print("not going through because is already dead")
                print("----------------------------------")
            return

        # Do here whatever happens to an individual during the admission for the treatment
        # ~~~~~~~~~~~~~~~~~~~~~~
        # Perform measurements (height/length, weight, MUAC)
        self.add_equipment({'Height Pole (Stadiometer)', 'Weighing scale', 'MUAC tape'})

        # Make request for consumables
        consumables = self.sim.modules['HealthSystem'].parameters['item_and_package_code_lookups']

        # individual items
        item_code1 = pd.unique(
            consumables.loc[consumables['Items'] == 'SAM theraputic foods', 'Item_Code'])[0]
        item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'SAM medicines', 'Item_Code'])[0]

        # check availability of consumables
        if self.get_consumables(item_code1) and self.get_consumables(item_code2):
            logger.debug(key='debug', data='consumables available, so use it.')
            # Log that the treatment is provided:
            df.at[person_id, 'un_am_treatment_type'] = 'inpatient_care'
            if do_prints:
                print("consumables available")
            self.module.do_when_am_treatment(person_id, intervention='ITC')
        else:
            logger.debug(key='debug',
                         data=f"Consumable(s) not available, hence {self.TREATMENT_ID} cannot be provided.")
            if do_prints:
                print("consumables not available, ITC tx not scheduled, should be picked up with next\n"
                      "growth monitoring or non-emergency appt if not naturally recovered or died in\n"
                      "between")

        if do_prints:
            print("-----------------------------------------------")

    def did_not_run(self):
        logger.debug(key='debug', data=f'{self.TREATMENT_ID}: did not run')
        pass


class WastingModels:
    """ houses all wasting linear models """

    def __init__(self, module):
        self.module = module
        self.rng = module.rng
        self.params = module.parameters

        # a linear model to predict the probability of individual's recovery from moderate acute malnutrition
        self.acute_malnutrition_recovery_mam_lm = LinearModel.multiplicative(
            Predictor('un_am_treatment_type',
                      conditions_are_mutually_exclusive=True, conditions_are_exhaustive=True)
            .when('soy_RUSF', self.params['recovery_rate_with_soy_RUSF'])
            .when('CSB++', self.params['recovery_rate_with_CSB++'])
        )

        # a linear model to predict the probability of individual's recovery from severe acute malnutrition
        self.acute_malnutrition_recovery_sam_lm = LinearModel.multiplicative(
            Predictor('un_am_treatment_type',
                      conditions_are_mutually_exclusive=True, conditions_are_exhaustive=True)
            .when('standard_RUTF', self.params['recovery_rate_with_standard_RUTF'])
            .when('inpatient_care', self.params['recovery_rate_with_inpatient_care'])
        )

        # Linear model for the probability of progression to severe wasting (age-dependent only)
        # (natural history only, no interventions)
        self.severe_wasting_progression_lm = LinearModel.multiplicative(
            Predictor('age_exact_years',
                      conditions_are_mutually_exclusive=True, conditions_are_exhaustive=True)
            .when('<0.5', self.params['progression_severe_wasting_by_agegp'][0])
            .when('.between(0.5,1, inclusive="left")', self.params['progression_severe_wasting_by_agegp'][1])
            .when('.between(1,2, inclusive="left")', self.params['progression_severe_wasting_by_agegp'][2])
            .when('.between(2,3, inclusive="left")', self.params['progression_severe_wasting_by_agegp'][3])
            .when('.between(3,4, inclusive="left")', self.params['progression_severe_wasting_by_agegp'][4])
            .when('.between(4,5, inclusive="left")', self.params['progression_severe_wasting_by_agegp'][5])
            .when('>=5', 0)
        )

        # get wasting incidence linear model
        self.wasting_incidence_lm = self.get_wasting_incidence()

        # Linear model for the probability of death due to SAM
        self.death_due_to_sam_lm =  LinearModel(
            LinearModelType.MULTIPLICATIVE,
            self.params['base_death_rate_untreated_SAM'],
            Predictor('age_exact_years',
                      conditions_are_mutually_exclusive=True, conditions_are_exhaustive=False)
            .when('<0.5', self.params['rr_death_rate_by_agegp'][0])
            .when('.between(0.5,1, inclusive="left")', self.params['rr_death_rate_by_agegp'][1])
            .when('.between(1,2, inclusive="left")', self.params['rr_death_rate_by_agegp'][2])
            .when('.between(2,3, inclusive="left")', self.params['rr_death_rate_by_agegp'][3])
            .when('.between(3,4, inclusive="left")', self.params['rr_death_rate_by_agegp'][4])
            .when('.between(4,5, inclusive="left")', self.params['rr_death_rate_by_agegp'][5]),
            Predictor().when('un_clinical_acute_malnutrition != "SAM"', 0),
        )

    def get_wasting_incidence(self) -> LinearModel:
        """
        :return: the scaled multiplicative linear model of wasting incidence rate
        """
        df = self.module.sim.population.props

        def unscaled_wasting_incidence_lm(intercept: Union[float, int] = 1.0) -> LinearModel:
            # linear model to predict the incidence of wasting
            return LinearModel(
                LinearModelType.MULTIPLICATIVE,
                intercept,  # base_overall_inc_rate_wasting
                Predictor('age_exact_years',
                          conditions_are_mutually_exclusive=True, conditions_are_exhaustive=False)
                .when('<0.5', self.params['rr_inc_rate_wasting_by_agegp'][0])
                .when('.between(0.5,1, inclusive="left")', self.params['rr_inc_rate_wasting_by_agegp'][1])
                .when('.between(1,2, inclusive="left")', self.params['rr_inc_rate_wasting_by_agegp'][2])
                .when('.between(2,3, inclusive="left")', self.params['rr_inc_rate_wasting_by_agegp'][3])
                .when('.between(3,4, inclusive="left")', self.params['rr_inc_rate_wasting_by_agegp'][4])
                .when('.between(4,5, inclusive="left")', self.params['rr_inc_rate_wasting_by_agegp'][5]),
                Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                 '& (nb_late_preterm == False) & (nb_early_preterm == False)',
                                 self.params['rr_wasting_SGA_and_term']),
                Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                 '& ((nb_late_preterm == True) | (nb_early_preterm == True))',
                                 self.params['rr_wasting_SGA_and_preterm']),
                Predictor().when('(nb_size_for_gestational_age == "average_for_gestational_age") '
                                 '& ((nb_late_preterm == True) | (nb_early_preterm == True))',
                                 self.params['rr_wasting_AGA_and_preterm']),
                Predictor('li_wealth').apply(
                    lambda x: 1 if x == 1 else (x - 1) ** (self.params['rr_wasting_wealth_level'])),
            )

        # target: base inc rate (i.e. inc rate of reference group 1-5mo old)
        target_mean = self.params['base_overall_inc_rate_wasting']
        unscaled_lm = unscaled_wasting_incidence_lm(intercept=target_mean)
        not_am_or_treated_ref_agegp = df.loc[df.is_alive & (df.un_clinical_acute_malnutrition == 'well') &
                                             (df.un_am_tx_start_date.isna()) & (df.age_exact_years < 0.5)]
        actual_mean = unscaled_lm.predict(not_am_or_treated_ref_agegp).mean()
        scaled_intercept = target_mean * (target_mean / actual_mean) if \
            (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else target_mean
        scaled_wasting_incidence_lm = unscaled_wasting_incidence_lm(intercept=scaled_intercept)

        return scaled_wasting_incidence_lm

    def get_wasting_prevalence(self, agegp: str, children_of_agegp: pd.DataFrame) -> LinearModel:
        """
        :param agegp: children's age group
        :param children_of_agegp: df with children in the age group
        :return: the scaled logistic linear model of wasting prevalence, i.e., odds of wasting among the specific agegp
        """

        def unscaled_wasting_prevalence_lm(intercept: Union[float, int]) -> LinearModel:
            return LinearModel(
                LinearModelType.LOGISTIC,
                intercept,  # baseline odds: get_odds_wasting(agegp=agegp)
                Predictor('li_wealth',
                          conditions_are_mutually_exclusive=True, conditions_are_exhaustive=False)
                .when(2, self.params['or_wasting_hhwealth_Q2'])
                .when(3, self.params['or_wasting_hhwealth_Q3'])
                .when(4, self.params['or_wasting_hhwealth_Q4'])
                .when(5, self.params['or_wasting_hhwealth_Q5']),
                Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                 '& (nb_late_preterm == False) & (nb_early_preterm == False)',
                                 self.params['or_wasting_SGA_and_term']),
                Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                 '& ((nb_late_preterm == True) | (nb_early_preterm == True))',
                                 self.params['or_wasting_SGA_and_preterm']),
                Predictor().when('(nb_size_for_gestational_age == "average_for_gestational_age") '
                                 '& ((nb_late_preterm == True) | (nb_early_preterm == True))',
                                 self.params['or_wasting_AGA_and_preterm'])
            )

        target_mean = self.module.get_odds_wasting(agegp=agegp)
        unscaled_lm = unscaled_wasting_prevalence_lm(intercept=target_mean)
        actual_mean = unscaled_lm.predict(children_of_agegp).mean()
        scaled_intercept = target_mean * (target_mean / actual_mean) if \
            (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else target_mean
        scaled_wasting_prevalence_lm = unscaled_wasting_prevalence_lm(intercept=scaled_intercept)

        return scaled_wasting_prevalence_lm


class Wasting_LoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This Event logs the number of incident cases that have occurred since the previous logging event, the average length
    of wasting cases at recovery point since the previous logging, the prevalence proportions at the time of logging,
    and population sizes at the time of logging.
    Analysis scripts expect that the frequency of this logging event is once per year. Logs are expected to happen on
    the last day of each year.
    """

    def __init__(self, module):
        # This event to occur every year
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, Wasting)

        self.date_last_run = self.sim.date

    def apply(self, population):
        df = self.sim.population.props
        p = self.module.parameters

        # ----- INCIDENCE LOG ----------------
        # Convert the list of timestamps into a number of timestamps
        # and check that all the dates have occurred since self.date_last_run
        inc_df = pd.DataFrame(index=self.module.wasting_incident_case_tracker.keys(),
                              columns=self.module.wasting_states)
        for age_grp in self.module.wasting_incident_case_tracker.keys():
            for state in self.module.wasting_states:
                inc_df.loc[age_grp, state] = len(self.module.wasting_incident_case_tracker[age_grp][state])
                assert all(date >= self.date_last_run for
                           date in self.module.wasting_incident_case_tracker[age_grp][state]), \
                    f"Some incident cases trying to be logged on {self.sim.date=} from the day of last log "\
                    f"{self.date_last_run=} or before."

        logger.info(key='wasting_incidence_count', data=inc_df.to_dict())

        # Reset the incidence tracker and the date_last_run
        self.module.wasting_incident_case_tracker = copy.deepcopy(self.module.wasting_incident_case_tracker_blank)
        self.date_last_run = self.sim.date

        # ----- LENGTH LOG ----------------
        # Convert the list of lengths to an avg length
        # and check that all the lengths are positive
        length_df = pd.DataFrame(index=self.module.wasting_length_tracker.keys(),
                                 columns=self.module.recovery_options)
        for age_grp in self.module.wasting_length_tracker.keys():
            for recov_opt in self.module.recovery_options:
                if self.module.wasting_length_tracker[age_grp][recov_opt]:
                    length_df.loc[age_grp, recov_opt] = (sum(self.module.wasting_length_tracker[age_grp][recov_opt]) /
                                                         len(self.module.wasting_length_tracker[age_grp][recov_opt]))
                else:
                    length_df.loc[age_grp, recov_opt] = 0
                assert not np.isnan(length_df.loc[age_grp, recov_opt]),\
                    f'There is an empty length for {age_grp=}, {recov_opt=}.'

                if recov_opt in ['mod_MAM_nat_full_recov', 'mod_SAM_nat_recov_to_MAM']:
                    assert all(length >= (p['duration_of_untreated_mod_wasting'] - 1) for length in
                           self.module.wasting_length_tracker[age_grp][recov_opt]),\
                        f"{self.module.wasting_length_tracker[age_grp][recov_opt]=} contains length(s) < "\
                        f"{(p['duration_of_untreated_mod_wasting'] - 1)=}; {age_grp=}, {recov_opt=}"
                elif recov_opt in ['sev_SAM_nat_recov_to_MAM']:
                    assert all(length >=
                               (p['duration_of_untreated_mod_wasting'] + p['duration_of_untreated_sev_wasting'] - 2) for
                               length in self.module.wasting_length_tracker[age_grp][recov_opt]),\
                        (f"{self.module.wasting_length_tracker[age_grp][recov_opt]=} contains length(s) < duration of "
                         "(mod + sev wast - 2): "
                         f"{(p['duration_of_untreated_mod_wasting'] + p['duration_of_untreated_sev_wasting'] - 2)=} "
                         f"days; {age_grp=}, {recov_opt=}")
                elif recov_opt in ['mod_MAM_tx_full_recov', 'mod_SAM_tx_full_recov', 'mod_SAM_tx_recov_to_MAM',
                                   'sev_SAM_tx_full_recov', 'sev_SAM_tx_recov_to_MAM']:
                    if recov_opt == 'mod_MAM_tx_full_recov':
                        min_length = (p['tx_length_weeks_SuppFeedingMAM'] * 7) - 1
                    else:
                        min_length = \
                            (min(p['tx_length_weeks_OutpatientSAM'], p['tx_length_weeks_InpatientSAM']) * 7) - 1

                    assert all(length >= min_length for length in
                               self.module.wasting_length_tracker[age_grp][recov_opt]), \
                        f'{self.module.wasting_length_tracker[age_grp][recov_opt]=} contains length(s) < ' \
                        f'{min_length=} days; {age_grp=}, {recov_opt=}'

                assert recov_opt in self.module.recovery_options, f'\nInvalid {recov_opt=}.'

        # Reset the length tracker
        self.module.wasting_length_tracker = copy.deepcopy(self.module.wasting_length_tracker_blank)

        under5s = df.loc[df.is_alive & (df.age_exact_years < 5)]
        above5s = df.loc[df.is_alive & (df.age_exact_years >= 5)]

        for age_ys in range(6):
            age_grp = self.module.age_grps.get(age_ys, '5+y')

            # get those children who are wasted
            if age_ys < 5:
                mod_wasted_whole_ys_agegrp = under5s[(
                    under5s.age_years.between(age_ys, age_ys + 1, inclusive='left') &
                    (under5s.un_WHZ_category == '-3<=WHZ<-2')
                )]
                sev_wasted_whole_ys_agegrp = under5s[(
                    under5s.age_years.between(age_ys, age_ys + 1, inclusive='left') &
                    (under5s.un_WHZ_category == 'WHZ<-3')
                )]
            else:
                mod_wasted_whole_ys_agegrp = above5s[(
                    above5s.un_WHZ_category == '-3<=WHZ<-2'
                )]
                sev_wasted_whole_ys_agegrp = above5s[(
                    above5s.un_WHZ_category == 'WHZ<-3'
                )]
            mod_wasted_whole_ys_agegrp = mod_wasted_whole_ys_agegrp.copy()
            mod_wasted_whole_ys_agegrp.loc[:, 'wasting_length'] = \
                (self.sim.date - mod_wasted_whole_ys_agegrp['un_last_wasting_date_of_onset']).dt.days
            sev_wasted_whole_ys_agegrp = sev_wasted_whole_ys_agegrp.copy()
            sev_wasted_whole_ys_agegrp.loc[:, 'wasting_length'] = \
                (self.sim.date - sev_wasted_whole_ys_agegrp['un_last_wasting_date_of_onset']).dt.days
            if len(mod_wasted_whole_ys_agegrp) > 0:
                assert not np.isnan(mod_wasted_whole_ys_agegrp['wasting_length']).any(),\
                    ("There is at least one NaN length.\n"
                     f"{mod_wasted_whole_ys_agegrp['wasting_length']=} for {age_grp=}")
                assert all(length > 0 for length in mod_wasted_whole_ys_agegrp['wasting_length']),\
                    ("There is at least one zero length.\n"
                     f"{mod_wasted_whole_ys_agegrp['wasting_length']=} for {age_grp=}")
                length_df.loc[age_grp, 'mod_not_yet_recovered'] = (
                    sum(mod_wasted_whole_ys_agegrp['wasting_length']) /
                    len(mod_wasted_whole_ys_agegrp['wasting_length'])
                )
            else:
                length_df.loc[age_grp, 'mod_not_yet_recovered'] = 0
            assert not np.isnan(length_df.loc[age_grp, 'mod_not_yet_recovered']), \
                f"The avg {length_df.loc[age_grp, 'mod_not_yet_recovered']=} for {age_grp=} is empty."

            if len(sev_wasted_whole_ys_agegrp) > 0:
                assert not np.isnan(sev_wasted_whole_ys_agegrp['wasting_length']).any(), \
                    ("There is at least one NaN length.\n"
                     f"{mod_wasted_whole_ys_agegrp['wasting_length']=} for {age_grp=}")

                assert all(length > 0 for length in sev_wasted_whole_ys_agegrp['wasting_length']), \
                    ("There is at least one zero length.\n"
                     f"{mod_wasted_whole_ys_agegrp['wasting_length']=} for {age_grp=}")
                length_df.loc[age_grp, 'sev_not_yet_recovered'] = (
                    sum(sev_wasted_whole_ys_agegrp['wasting_length']) /
                    len(sev_wasted_whole_ys_agegrp['wasting_length'])
                )
            else:
                length_df.loc[age_grp, 'sev_not_yet_recovered'] = 0
            assert not np.isnan(length_df.loc[age_grp, 'sev_not_yet_recovered']), \
                f"The avg {length_df.loc[age_grp, 'sev_not_yet_recovered']=} for {age_grp=} is empty."

        logger.info(key='wasting_length_avg', data=length_df.to_dict())

        # ----- PREVALENCE LOG ----------------
        # Wasting totals (prevalence & pop size at logging time)
        # declare a dictionary that will hold proportions of wasting prevalence per each age group
        wasting_prev_dict: Dict[str, Any] = dict()
        # declare a dictionary that will hold pop sizes
        pop_sizes_dict: Dict[str, Any] = dict()

        # loop through different age groups and get proportions of wasting prevalence per each age group
        for low_bound_mos, high_bound_mos in [(0, 5), (6, 11), (12, 23), (24, 35), (36, 47), (48, 59)]:  # in months
            low_bound_age_in_years = low_bound_mos / 12.0
            high_bound_age_in_years = (1 + high_bound_mos) / 12.0
            total_per_agegrp_nmb = (under5s.age_exact_years.between(low_bound_age_in_years, high_bound_age_in_years,
                                                                    inclusive='left')).sum()
            if total_per_agegrp_nmb > 0:
                # get those children who are wasted
                mod_wasted_agegrp_nmb = (under5s.age_exact_years.between(low_bound_age_in_years, high_bound_age_in_years,
                                                                         inclusive='left') & (under5s.un_WHZ_category
                                                                                              == '-3<=WHZ<-2')).sum()
                sev_wasted_agegrp_nmb = (under5s.age_exact_years.between(low_bound_age_in_years, high_bound_age_in_years,
                                                                         inclusive='left') & (under5s.un_WHZ_category
                                                                                           == 'WHZ<-3')).sum()
                # add moderate and severe wasting prevalence to the dictionary
                wasting_prev_dict[f'mod__{low_bound_mos}_{high_bound_mos}mo'] = \
                    mod_wasted_agegrp_nmb / total_per_agegrp_nmb
                wasting_prev_dict[f'sev__{low_bound_mos}_{high_bound_mos}mo'] = \
                    sev_wasted_agegrp_nmb / total_per_agegrp_nmb
            else:
                # add zero moderate and severe wasting prevalence to the dictionary
                wasting_prev_dict[f'mod__{low_bound_mos}_{high_bound_mos}mo'] = 0
                wasting_prev_dict[f'sev__{low_bound_mos}_{high_bound_mos}mo'] = 0

            # add pop sizes to the dataframe
            pop_sizes_dict[f'mod__{low_bound_mos}_{high_bound_mos}mo'] = mod_wasted_agegrp_nmb
            pop_sizes_dict[f'sev__{low_bound_mos}_{high_bound_mos}mo'] = sev_wasted_agegrp_nmb
            pop_sizes_dict[f'total__{low_bound_mos}_{high_bound_mos}mo'] = total_per_agegrp_nmb
        # log prevalence & pop size for children above 5y
        above5s = df.loc[df.is_alive & (df.age_exact_years >= 5)]
        assert (len(under5s) + len(above5s)) == len(df.loc[df.is_alive]), \
            ("The numbers of persons under and above 5 don't sum to all alive person, when logging on"
             f"{self.sim.date=}.")
        mod_wasted_above5_nmb = (above5s.un_WHZ_category == '-3<=WHZ<-2').sum()
        sev_wasted_above5_nmb = (above5s.un_WHZ_category == 'WHZ<-3').sum()
        wasting_prev_dict['mod__5y+'] = mod_wasted_above5_nmb / len(above5s)
        wasting_prev_dict['sev__5y+'] = sev_wasted_above5_nmb / len(above5s)
        pop_sizes_dict['mod__5y+'] = mod_wasted_above5_nmb
        pop_sizes_dict['sev__5y+'] = sev_wasted_above5_nmb
        pop_sizes_dict['total__5y+'] = len(above5s)

        # add to dictionary proportion of all moderately/severely wasted children under 5 years
        mod_under5_nmb = (under5s.un_WHZ_category == '-3<=WHZ<-2').sum()
        sev_under5_nmb = (under5s.un_WHZ_category == 'WHZ<-3').sum()
        wasting_prev_dict['total_mod_under5_prop'] = mod_under5_nmb / len(under5s)
        wasting_prev_dict['total_sev_under5_prop'] = sev_under5_nmb / len(under5s)
        pop_sizes_dict['mod__under5'] = mod_under5_nmb
        pop_sizes_dict['sev__under5'] = sev_under5_nmb
        pop_sizes_dict['total__under5'] = len(under5s)

        # log wasting prevalence
        logger.info(key='wasting_prevalence_props', data=wasting_prev_dict)

        # log pop sizes
        logger.info(key='pop sizes', data=pop_sizes_dict)


class Wasting_InitLoggingEvent(Event, PopulationScopeEventMixin):
    """
    This Event logs initial prevalence proportions of moderate and severe wasting, and initial population sizes.
    """

    def __init__(self, module):
        # This event to occur every year
        super().__init__(module)
        assert isinstance(module, Wasting)

    def apply(self, population):
        df = self.sim.population.props

        # ----- PREVALENCE LOG ----------------
        # Wasting totals (prevalence & pop size at logging time)
        # declare a dictionary that will hold proportions of wasting prevalence per each age group
        wasting_prev_dict: Dict[str, Any] = dict()
        # declare a dictionary that will hold pop sizes
        pop_sizes_dict: Dict[str, Any] = dict()

        under5s = df.loc[df.is_alive & (df.age_exact_years < 5)]
        # loop through different age groups and get proportions of wasting prevalence per each age group
        for low_bound_mos, high_bound_mos in [(0, 5), (6, 11), (12, 23), (24, 35), (36, 47), (48, 59)]:  # in months
            low_bound_age_in_years = low_bound_mos / 12.0
            high_bound_age_in_years = (1 + high_bound_mos) / 12.0
            # get those children who are wasted
            mod_wasted_agegrp_nmb = (under5s.age_exact_years.between(low_bound_age_in_years, high_bound_age_in_years,
                                                                     inclusive='left') & (under5s.un_WHZ_category
                                                                                          == '-3<=WHZ<-2')).sum()
            sev_wasted_agegrp_nmb = (under5s.age_exact_years.between(low_bound_age_in_years, high_bound_age_in_years,
                                                                     inclusive='left') & (under5s.un_WHZ_category
                                                                                          == 'WHZ<-3')).sum()
            total_per_agegrp_nmb = (under5s.age_exact_years.between(low_bound_age_in_years, high_bound_age_in_years,
                                                                    inclusive='left')).sum()
            # add moderate and severe wasting prevalence to the dictionary
            wasting_prev_dict[f'mod__{low_bound_mos}_{high_bound_mos}mo'] = mod_wasted_agegrp_nmb / total_per_agegrp_nmb
            wasting_prev_dict[f'sev__{low_bound_mos}_{high_bound_mos}mo'] = sev_wasted_agegrp_nmb / total_per_agegrp_nmb
            # add pop sizes to the dataframe
            pop_sizes_dict[f'mod__{low_bound_mos}_{high_bound_mos}mo'] = mod_wasted_agegrp_nmb
            pop_sizes_dict[f'sev__{low_bound_mos}_{high_bound_mos}mo'] = sev_wasted_agegrp_nmb
            pop_sizes_dict[f'total__{low_bound_mos}_{high_bound_mos}mo'] = total_per_agegrp_nmb
        # log prevalence & pop size for children above 5y
        above5s = df.loc[df.is_alive & (df.age_exact_years >= 5)]
        assert (len(under5s) + len(above5s)) == len(df.loc[df.is_alive]), \
            ("The numbers of persons under and above 5 don't sum to all alive person, when logging at"
             "sim initiation.")
        mod_wasted_above5_nmb = (above5s.un_WHZ_category == '-3<=WHZ<-2').sum()
        sev_wasted_above5_nmb = (above5s.un_WHZ_category == 'WHZ<-3').sum()
        wasting_prev_dict['mod__5y+'] = mod_wasted_above5_nmb / len(above5s)
        wasting_prev_dict['sev__5y+'] = sev_wasted_above5_nmb / len(above5s)
        pop_sizes_dict['mod__5y+'] = mod_wasted_above5_nmb
        pop_sizes_dict['sev__5y+'] = sev_wasted_above5_nmb
        pop_sizes_dict['total__5y+'] = len(above5s)

        # add to dictionary proportion of all moderately/severely wasted children under 5 years
        mod_under5_nmb = (under5s.un_WHZ_category == '-3<=WHZ<-2').sum()
        sev_under5_nmb = (under5s.un_WHZ_category == 'WHZ<-3').sum()
        wasting_prev_dict['total_mod_under5_prop'] = mod_under5_nmb / len(under5s)
        wasting_prev_dict['total_sev_under5_prop'] = sev_under5_nmb / len(under5s)
        pop_sizes_dict['mod__under5'] = mod_under5_nmb
        pop_sizes_dict['sev__under5'] = sev_under5_nmb
        pop_sizes_dict['total__under5'] = len(under5s)

        # log wasting prevalence
        logger.info(key='wasting_init_prevalence_props', data=wasting_prev_dict)

        # log pop sizes
        logger.info(key='init pop sizes', data=pop_sizes_dict)

