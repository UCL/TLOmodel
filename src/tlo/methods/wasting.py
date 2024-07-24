"""Childhood wasting module"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Union

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
        # prevalence of wasting by age group
        'prev_WHZ_distribution_age_0_5mo': Parameter(
            Types.LIST, 'distribution of WHZ among less than 6 months of age in 2015'),
        'prev_WHZ_distribution_age_6_11mo': Parameter(
            Types.LIST, 'distribution of WHZ among 6 months and 1 year of age in 2015'),
        'prev_WHZ_distribution_age_12_23mo': Parameter(
            Types.LIST, 'distribution of WHZ among 1 year olds in 2015'),
        'prev_WHZ_distribution_age_24_35mo': Parameter(
            Types.LIST, 'distribution of WHZ among 2 year olds in 2015'),
        'prev_WHZ_distribution_age_36_47mo': Parameter(
            Types.LIST, 'distribution of WHZ among 3 year olds in 2015'),
        'prev_WHZ_distribution_age_48_59mo': Parameter(
            Types.LIST, 'distribution of WHZ among 4 year olds  in 2015'),
        # effect of risk factors on wasting prevalence
        'or_wasting_hhwealth_Q5': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is poorest Q5, ref group Q1'),
        'or_wasting_hhwealth_Q4': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is poorer Q4, ref group Q1'),
        'or_wasting_hhwealth_Q3': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is middle Q3, ref group Q1'),
        'or_wasting_hhwealth_Q2': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is richer Q2, ref group Q1'),
        'or_wasting_preterm_and_AGA': Parameter(
            Types.REAL, 'odds ratio of wasting if born preterm and adequate for gestational age'),
        'or_wasting_SGA_and_term': Parameter(
            Types.REAL, 'odds ratio of wasting if born term and small for gestational age'),
        'or_wasting_SGA_and_preterm': Parameter(
            Types.REAL, 'odds ratio of wasting if born preterm and small for gestational age'),
        # incidence
        'base_inc_rate_wasting_by_agegp': Parameter(
            Types.LIST, 'List with baseline incidence of wasting by age group'),
        'rr_wasting_preterm_and_AGA': Parameter(
            Types.REAL, 'relative risk of wasting if born preterm and adequate for gestational age'),
        'rr_wasting_SGA_and_term': Parameter(
            Types.REAL, 'relative risk of wasting if born term and small for gestational age'),
        'rr_wasting_SGA_and_preterm': Parameter(
            Types.REAL, 'relative risk of wasting if born preterm and small for gestational age'),
        'rr_wasting_wealth_level': Parameter(
            Types.REAL, 'relative risk of wasting per 1 unit decrease in wealth level'),
        # progression
        'min_days_duration_of_wasting': Parameter(
            Types.REAL, 'minimum duration in days of wasting (MAM and SAM)'),
        'average_duration_of_untreated_MAM': Parameter(
            Types.REAL, 'average duration of untreated MAM'),
        'average_duration_of_untreated_SAM': Parameter(
            Types.REAL, 'average duration of untreated SAM'),
        'progression_severe_wasting_by_agegp': Parameter(
            Types.LIST, 'List with progression rates to severe wasting by age group'),
        'prob_complications_in_SAM': Parameter(
            Types.REAL, 'probability of medical complications in SAM '),
        # MUAC distributions
        'MUAC_distribution_WHZ<-3': Parameter(
            Types.LIST,
            'mean and standard deviation of a normal distribution of MUAC measurements for WHZ < -3'),
        'MUAC_distribution_-3<=WHZ<-2': Parameter(
            Types.LIST,
            'mean and standard deviation of a normal distribution of MUAC measurements for -3 <= WHZ < -2'),
        'MUAC_distribution_WHZ>=-2': Parameter(
            Types.LIST,
            'mean and standard deviation of a normal distribution of MUAC measurements for WHZ >= -2'),
        'proportion_WHZ<-3_with_MUAC<115mm': Parameter(
            Types.REAL, 'proportion of severe wasting with MUAC < 115mm'),
        'proportion_-3<=WHZ<-2_with_MUAC<115mm': Parameter(
            Types.REAL, 'proportion of moderate wasting with MUAC < 115mm'),
        'proportion_-3<=WHZ<-2_with_MUAC_[115-125)mm': Parameter(
            Types.REAL, 'proportion of moderate wasting with 115 mm ≤ MUAC < 125mm'),
        'proportion_mam_with_MUAC_[115-125)mm_and_normal_whz': Parameter(
            Types.REAL, 'proportion of MAM cases with 115 mm ≤ MUAC < 125 mm and normal/mild WHZ'),
        'proportion_mam_with_MUAC_[115-125)mm_and_-3<=WHZ<-2': Parameter(
            Types.REAL, 'proportion of MAM cases with both 115 mm ≤ MUAC < 125 mm and moderate wasting'),
        'proportion_mam_with_-3<=WHZ<-2_and_normal_MUAC': Parameter(
            Types.REAL, 'proportion of MAM cases with moderate wasting and normal MUAC'),
        # bilateral oedema
        'prevalence_nutritional_oedema': Parameter(
            Types.REAL, 'prevalence of nutritional oedema in children under 5 in Malawi'),
        'proportion_oedema_with_WHZ<-2': Parameter(
            Types.REAL, 'proportion of oedematous malnutrition with concurrent wasting'),
        # treatment
        'coverage_supplementary_feeding_program': Parameter(
            Types.REAL, 'coverage of supplementary feeding program for MAM in health centres'),
        'coverage_outpatient_therapeutic_care': Parameter(
            Types.REAL, 'coverage of outpatient therapeutic care for SAM in health centres'),
        'coverage_inpatient_care': Parameter(
            Types.REAL, 'coverage of inpatient care for complicated SAM in hospitals'),
        'prob_mam_after_care': Parameter(
            Types.REAL, 'probability of returning to MAM after seeking care'),
        'prob_death_after_care': Parameter(
            Types.REAL, 'probability of dying after seeking care'),
        # recovery
        'recovery_rate_with_standard_RUTF': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with standard RUTF'),
        'recovery_rate_with_soy_RUSF': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with soy RUSF'),
        'recovery_rate_with_CSB++': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with CSB++'),
        'recovery_rate_with_inpatient_care': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with inpatient care'),
    }

    PROPERTIES = {
        # Properties related to wasting
        'un_ever_wasted': Property(Types.BOOL, 'ever had an episode of wasting (WHZ < -2)'),
        'un_WHZ_category': Property(Types.CATEGORICAL, 'weight-for-height Z-score category',
                                    categories=['WHZ<-3', '-3<=WHZ<-2', 'WHZ>=-2']),
        'un_last_wasting_date_of_onset': Property(Types.DATE, 'date of onset of last episode of wasting'),

        # Properties related to clinical acute malnutrition
        'un_clinical_acute_malnutrition': Property(Types.CATEGORICAL, 'clinical acute malnutrition state based'
                                                                      ' on WHZ and/or MUAC and/or oedema',
                                                   categories=['MAM', 'SAM'] + ['well']),
        'un_am_bilateral_oedema': Property(Types.BOOL, 'bilateral pitting oedema present in wasting episode'),
        'un_am_MUAC_category': Property(Types.CATEGORICAL, 'MUAC measurement categories, based on WHO '
                                                           'cut-offs',
                                        categories=['<115mm', '[115-125)mm', '>=125mm']),
        'un_sam_with_complications': Property(Types.BOOL, 'medical complications in SAM episode'),
        'un_sam_death_date': Property(Types.DATE, 'death date from severe acute malnutrition'),
        'un_am_recovery_date': Property(Types.DATE, 'recovery date from acute malnutrition'),
        'un_am_discharge_date': Property(Types.DATE, 'discharge date from treatment of MAM/SAM'),
        'un_am_tx_start_date': Property(Types.DATE, 'intervention start date'),
        'un_am_treatment_type': Property(Types.CATEGORICAL, 'treatment types for acute malnutrition',
                                         categories=['standard_RUTF', 'soy_RUSF', 'CSB++', 'inpatient_care'] + [
                                             'none', 'not_applicable']),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.wasting_models = None
        self.resourcefilepath = resourcefilepath
        # wasting states
        self.wasting_states = self.PROPERTIES["un_WHZ_category"].categories
        # wasting symptom
        self.wasting_symptom = 'weight_loss'

        # dict to hold counters for the number of episodes by wasting-type and age-group
        blank_counter = dict(
            zip(self.wasting_states, [list() for _ in self.wasting_states]))
        self.wasting_incident_case_tracker_blank = {
            _agrp: copy.deepcopy(blank_counter) for _agrp in ['0y', '1y', '2y', '3y', '4y', '5+y']}

        self.wasting_incident_case_tracker = copy.deepcopy(self.wasting_incident_case_tracker_blank)

    def read_parameters(self, data_folder):
        """
        :param data_folder: path of a folder supplied to the Simulation containing data files. Typically,
        modules would read a particular file within here.
        :return:
        """
        # Read parameters from the resource file
        self.load_parameters_from_dataframe(
            pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_Wasting.csv')
        )

        # Register wasting symptom(weight loss) in Symptoms Manager
        self.sim.modules['SymptomManager'].register_symptom(Symptom(name=self.wasting_symptom))

    def initialise_population(self, population):
        """
        Set our property values for the initial population. This method is called by the simulation when creating
        the initial population, and is responsible for assigning initial values, for every individual,
        of those properties 'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population:
        :return:
        """
        df = population.props

        # Set initial properties
        df.loc[df.is_alive, 'un_ever_wasted'] = False
        df.loc[df.is_alive, 'un_WHZ_category'] = 'WHZ>=-2'  # not undernourished
        df.loc[df.is_alive, 'un_last_wasting_date_of_onset'] = pd.NaT
        df.loc[df.is_alive, 'un_clinical_acute_malnutrition'] = 'well'
        df.loc[df.is_alive, 'un_am_bilateral_oedema'] = False
        df.loc[df.is_alive, 'un_am_MUAC_category'] = '>=125mm'
        df.loc[df.is_alive, 'un_sam_death_date'] = pd.NaT
        df.loc[df.is_alive, 'un_am_tx_start_date'] = pd.NaT
        df.loc[df.is_alive, 'un_am_treatment_type'] = 'not_applicable'

        # initialise wasting linear models.
        self.wasting_models = WastingModels(self)

        # Assign wasting categories in young children at initiation
        for low_bound_mos, high_bound_mos in [(0, 5), (6, 11), (12, 23), (24, 35), (36, 47), (48, 59)]:  # in months
            low_bound_age_in_years = low_bound_mos / 12.0
            high_bound_age_in_years = (1 + high_bound_mos) / 12.0
            # linear model external variables
            agegp = f'{low_bound_mos}_{high_bound_mos}mo'
            mask = (df.is_alive & df.age_exact_years.between(low_bound_age_in_years, high_bound_age_in_years,
                                                             inclusive='left'))
            prevalence_of_wasting = self.wasting_models.get_wasting_prevalence(agegp=agegp).predict(df.loc[mask])

            # apply prevalence of wasting and categorise into moderate (-3 <= WHZ < -2) or severe (WHZ < -3) wasting
            wasted = self.rng.random_sample(len(prevalence_of_wasting)) < prevalence_of_wasting
            for idx in prevalence_of_wasting.index[wasted]:
                probability_of_severe = self.get_prob_severe_wasting_or_odds_wasting(agegp=agegp)
                wasted_category = self.rng.choice(['WHZ<-3', '-3<=WHZ<-2'], p=[probability_of_severe,
                                                                               1 - probability_of_severe])
                df.at[idx, 'un_WHZ_category'] = wasted_category
                df.at[idx, 'un_last_wasting_date_of_onset'] = self.sim.date
                df.at[idx, 'un_ever_wasted'] = True
                # start without treatment
                df.at[idx, 'un_am_treatment_type'] = 'none'

        # ------------------------------------------------------------------
        # # # # # # Give MUAC category, presence of oedema, and determine acute malnutrition state # # # # #
        self.population_poll_clinical_am(df)

    def initialise_simulation(self, sim):
        """Prepares for simulation:
        * Schedules the main polling event
        * Schedules the main logging event
        """

        # schedule wasting pool event
        sim.schedule_event(WastingPollingEvent(self), sim.date + DateOffset(months=3))

        # schedule wasting logging event
        sim.schedule_event(WastingLoggingEvent(self), sim.date + DateOffset(months=12))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        # Set initial properties
        df.at[child_id, 'un_ever_wasted'] = False
        df.at[child_id, 'un_WHZ_category'] = 'WHZ>=-2'  # not undernourished
        df.at[child_id, 'un_clinical_acute_malnutrition'] = 'well'
        df.at[child_id, 'un_last_wasting_date_of_onset'] = pd.NaT
        df.at[child_id, 'un_am_tx_start_date'] = pd.NaT
        df.at[child_id, 'un_sam_death_date'] = pd.NaT
        df.at[child_id, 'un_am_bilateral_oedema'] = False
        df.at[child_id, 'un_am_MUAC_category'] = '>=125mm'
        df.at[child_id, 'un_am_treatment_type'] = 'not_applicable'

    def get_prob_severe_wasting_or_odds_wasting(self, agegp: str, get_odds: bool = False) -> Union[float, int]:
        """
        This function will calculate the WHZ scores by categories and return probability of severe wasting
        for those with wasting status, or odds of wasting
        :param agegp: age grouped in months
        :param get_odds: when set to True, this argument will cause this method return the odds of wasting to be used
                for scaling wasting prevalence linear model
        :return: probability of severe wasting among all wasting cases (if 'get_odds' == False),
                or odds of wasting among all children under 5 (if 'get_odds' == True)
        """
        # generate random numbers from N(mean, sd)
        mean, stdev = self.parameters[f'prev_WHZ_distribution_age_{agegp}']
        whz_normal_distribution = norm(loc=mean, scale=stdev)

        # get probability of any wasting: WHZ < -2
        probability_less_than_minus2sd = 1 - whz_normal_distribution.sf(-2)

        if get_odds:
            # convert probability of wasting to odds and return the odds of wasting
            return probability_less_than_minus2sd / (1 - probability_less_than_minus2sd)

        # get probability of severe wasting: WHZ < -3
        probability_less_than_minus3sd = 1 - whz_normal_distribution.sf(-3)

        # make WHZ < -2 as the 100% and get the adjusted probability of severe wasting within overall wasting
        # return the probability of severe wasting among all wasting cases
        return probability_less_than_minus3sd / probability_less_than_minus2sd

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
            # apply probability of MUAC < 115 mm in severe wasting
            low_muac_in_severe_wasting = self.rng.random_sample(size=len(idx)) < p['proportion_WHZ<-3_with_MUAC<115mm']

            df.loc[idx[low_muac_in_severe_wasting], 'un_am_MUAC_category'] = '<115mm'
            # other with severe wasting will have MUAC within [115-125)mm
            df.loc[idx[~low_muac_in_severe_wasting], 'un_am_MUAC_category'] = '[115-125)mm'

        # ----- MUAC distribution for moderate wasting (-3 <= WHZ < -2) ------
        if whz == '-3<=WHZ<-2':
            # apply probability of MUAC < 115 mm in moderate wasting
            low_muac_in_moderate_wasting = self.rng.random_sample(size=len(idx)) < \
                                           p['proportion_-3<=WHZ<-2_with_MUAC<115mm']
            df.loc[idx[low_muac_in_moderate_wasting], 'un_am_MUAC_category'] = '<115mm'

            # apply probability of MUAC within [115-125)mm in moderate wasting
            moderate_low_muac_in_moderate_wasting = \
                self.rng.random_sample(size=len(idx[~low_muac_in_moderate_wasting])) < \
                p['proportion_-3<=WHZ<-2_with_MUAC_[115-125)mm']
            df.loc[idx[~low_muac_in_moderate_wasting][moderate_low_muac_in_moderate_wasting], 'un_am_MUAC_category'] \
                = '[115-125)mm'
            # other with moderate wasting will have normal MUAC
            df.loc[idx[~low_muac_in_moderate_wasting][~moderate_low_muac_in_moderate_wasting], 'un_am_MUAC_category'] \
                = '>=125mm'

        # ----- MUAC distribution for WHZ >= -2 -----
        if whz == 'WHZ>=-2':

            muac_distribution_in_well_group = norm(loc=p['MUAC_distribution_WHZ>=-2'][0],
                                                   scale=p['MUAC_distribution_WHZ>=-2'][1])
            # get probability of MUAC < 115 mm
            probability_over_or_equal_115 = muac_distribution_in_well_group.sf(11.5)
            probability_over_or_equal_125 = muac_distribution_in_well_group.sf(12.5)

            prob_less_than_115 = 1 - probability_over_or_equal_115
            pro_between_115_125 = probability_over_or_equal_115 - probability_over_or_equal_125

            for pid in idx:
                muac_cat = self.rng.choice(['<115mm', '[115-125)mm', '>=125mm'],
                                           p=[prob_less_than_115, pro_between_115_125, probability_over_or_equal_125])
                df.at[pid, 'un_am_MUAC_category'] = muac_cat

    def nutritional_oedema_present(self, idx):
        """
        This function applies the probability of bilateral oedema present in wasting and non-wasted cases
        :param idx: index of children under 5, or person_id
        :return:
        """
        df = self.sim.population.props
        p = self.parameters

        # Knowing the prevalence of nutritional oedema in under 5 population,
        # apply the probability of oedema in WHZ < -2
        # get those children with wasting
        children_with_wasting = idx.intersection(df.index[df.un_WHZ_category != 'WHZ>=-2'])
        children_without_wasting = idx.intersection(df.index[df.un_WHZ_category == 'WHZ>=-2'])

        # oedema among wasted children
        oedema_in_wasted_children = self.rng.random_sample(size=len(
            children_with_wasting)) < p['prevalence_nutritional_oedema'] * p['proportion_oedema_with_WHZ<-2']
        df.loc[children_with_wasting[oedema_in_wasted_children], 'un_am_bilateral_oedema'] = True
        df.loc[children_with_wasting[~oedema_in_wasted_children], 'un_am_bilateral_oedema'] = False

        # oedema among non-wasted children
        oedema_in_non_wasted = self.rng.random_sample(size=len(
            children_without_wasting)) < p['prevalence_nutritional_oedema'] * (1 - p['proportion_oedema_with_WHZ<-2'])
        df.loc[children_without_wasting[oedema_in_non_wasted], 'un_am_bilateral_oedema'] = True
        df.loc[children_without_wasting[~oedema_in_non_wasted], 'un_am_bilateral_oedema'] = False

    def clinical_acute_malnutrition_state(self, person_id, pop_dataframe):
        """
        This function will determine the clinical acute malnutrition status (MAM, SAM) based on anthropometric indices
        and presence of bilateral oedema (Kwashiorkor); And help determine whether the individual will have medical
        complications, applicable to SAM cases only, requiring inpatient care.
        :param person_id: individual id
        :param pop_dataframe: population dataframe
        :return:
        """
        df = pop_dataframe
        p = self.parameters

        # check if person is not wasted
        if ((df.at[person_id, 'un_WHZ_category'] == 'WHZ>=-2') &
                (df.at[person_id, 'un_am_MUAC_category'] == '>=125mm') & (~df.at[person_id, 'un_am_bilateral_oedema'])):
            df.at[person_id, 'un_clinical_acute_malnutrition'] = 'well'

        # severe acute malnutrition -- MUAC < 115 mm and/or WHZ < -3 and/or bilateral oedema
        elif ((df.at[person_id, 'un_am_MUAC_category'] == '<115mm') | (df.at[person_id, 'un_WHZ_category'] == 'WHZ<-3')
              | (df.at[person_id, 'un_am_bilateral_oedema'])):
            df.at[person_id, 'un_clinical_acute_malnutrition'] = 'SAM'
            # apply symptoms to all SAM cases
            self.wasting_clinical_symptoms(person_id=person_id)

        else:
            df.at[person_id, 'un_clinical_acute_malnutrition'] = 'MAM'

        # Determine if SAM episode has complications
        if df.at[person_id, 'un_clinical_acute_malnutrition'] == 'SAM':
            if self.rng.random_sample() < p['prob_complications_in_SAM']:
                df.at[person_id, 'un_sam_with_complications'] = True
            else:
                df.at[person_id, 'un_sam_with_complications'] = False
        else:
            df.at[person_id, 'un_sam_with_complications'] = False

        assert not (df.at[person_id, 'un_clinical_acute_malnutrition'] == 'MAM') & \
                   (df.at[person_id, 'un_sam_with_complications'])

    def date_of_outcome_for_untreated_am(self, person_id, duration_am):
        """
        helper function to get the duration, the wasting episode and date of outcome (recovery, progression, or death)
        :param person_id:
        :param duration_am:
        :return:
        """
        df = self.sim.population.props
        p = self.parameters

        # moderate wasting (for progression to severe, or recovery from MAM) -----
        if duration_am == 'MAM':
            # Allocate the duration of the moderate wasting episode
            duration_mam = int(max(p['min_days_duration_of_wasting'], p['average_duration_of_untreated_MAM']))
            # Allocate a date of outcome (progression, recovery or death)
            date_of_outcome = df.at[person_id, 'un_last_wasting_date_of_onset'] + DateOffset(days=duration_mam)
            return date_of_outcome

        # severe wasting (for death, or recovery to moderate wasting) -----
        if duration_am == 'SAM':
            # determine the duration of SAM episode
            duration_sam = int(max(p['min_days_duration_of_wasting'], p['average_duration_of_untreated_MAM']
                                   + p['average_duration_of_untreated_SAM']))
            # Allocate a date of outcome (progression, recovery or death)
            date_of_outcome = df.at[person_id, 'un_last_wasting_date_of_onset'] + DateOffset(days=duration_sam)
            return date_of_outcome

    def population_poll_clinical_am(self, population):
        """
        Update at the population level other anthropometric indices  and clinical signs (MUAC, oedema,
        medical complications) that determine the clinical state of acute malnutrition This will include both wasted
        and non-wasted children with other signs of acute malnutrition
        :param population: population dataframe
        :return:
        """
        df = population

        # give MUAC measurement category for all WHZ, including well
        # nourished children -----
        for whz in ['WHZ<-3', '-3<=WHZ<-2', 'WHZ>=-2']:
            index_6_59mo_by_whz = df.index[df.is_alive & (df.age_exact_years.between(0.5, 5, inclusive='left'))
                                           & (df.un_WHZ_category == whz)]
            self.muac_cutoff_by_WHZ(idx=index_6_59mo_by_whz, whz=whz)

        # determine the presence of bilateral oedema / oedematous malnutrition
        index_under5 = df.index[df.is_alive & (df.age_exact_years < 5)]
        self.nutritional_oedema_present(idx=index_under5)

        # determine the clinical acute malnutrition state -----
        for person_id in index_under5:
            self.clinical_acute_malnutrition_state(person_id=person_id, pop_dataframe=df)

    def report_daly_values(self):
        """
        This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        experienced by persons in the previous month. Only rows for alive-persons must be returned. The names of the
        series of columns is taken to be the label of the cause of this disability. It will be recorded by the
        healthburden module as <ModuleName>_<Cause>.
        """
        # Dict to hold the DALY weights
        daly_wts = dict()

        df = self.sim.population.props
        # Get DALY weights
        get_daly_weight = self.sim.modules['HealthBurden'].get_daly_weight

        daly_wts['MAM_with_oedema'] = get_daly_weight(sequlae_code=461)
        daly_wts['SAM_w/o_oedema'] = get_daly_weight(sequlae_code=462)
        daly_wts['SAM_with_oedema'] = get_daly_weight(sequlae_code=463)

        total_daly_values = pd.Series(data=0.0,
                                      index=df.index[df.is_alive])
        total_daly_values.loc[df.is_alive & (df.un_clinical_acute_malnutrition == 'SAM') &
                              df.un_am_bilateral_oedema] = daly_wts['SAM_with_oedema']
        total_daly_values.loc[df.is_alive & (df.un_clinical_acute_malnutrition == 'SAM') &
                              (~df.un_am_bilateral_oedema)] = daly_wts['SAM_w/o_oedema']
        total_daly_values.loc[df.is_alive & (
                ((df.un_WHZ_category == '-3<=WHZ<-2') & (df.un_am_MUAC_category != "<115mm")) | (
                    (df.un_WHZ_category != 'WHZ<-3') & (
                        df.un_am_MUAC_category != "[115-125)mm"))) & df.un_am_bilateral_oedema] = daly_wts[
            'MAM_with_oedema']
        return total_daly_values

    def wasting_clinical_symptoms(self, person_id):
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
        individual_properties: IndividualProperties,
        schedule_hsi_event: HSIEventScheduler,
        **kwargs,
    ) -> None:
        if individual_properties["age_years"] > 5:
            return
        p = self.parameters

        # get the clinical states
        clinical_am = individual_properties['un_clinical_acute_malnutrition']
        complications = individual_properties['un_sam_with_complications']

        # Interventions for MAM
        if clinical_am == 'MAM':
            # Check for coverage of supplementary feeding
            if self.rng.random_sample() < p['coverage_supplementary_feeding_program']:
                # schedule HSI for supplementary feeding program for MAM
                schedule_hsi_event(
                    hsi_event=HSI_Wasting_SupplementaryFeedingProgramme_MAM(module=self, person_id=person_id),
                    priority=0, topen=self.sim.date)
            else:
                return
        # Interventions for uncomplicated SAM
        if clinical_am == 'SAM':
            if not complications:
                # Check for coverage of outpatient therapeutic care
                if self.rng.random_sample() < p['coverage_outpatient_therapeutic_care']:
                    # schedule HSI for supplementary feeding program for MAM
                    schedule_hsi_event(
                        hsi_event=HSI_Wasting_OutpatientTherapeuticProgramme_SAM(
                            module=self, person_id=person_id), priority=0, topen=self.sim.date)
                else:
                    return
            # Interventions for complicated SAM
            if complications:
                # Check for coverage of outpatient therapeutic care
                if self.rng.random_sample() < p['coverage_inpatient_care']:
                    # schedule HSI for supplementary feeding program for MAM
                    schedule_hsi_event(
                        hsi_event=HSI_Wasting_InpatientCareForComplicated_SAM(
                            module=self, person_id=person_id), priority=0, topen=self.sim.date)
                else:
                    return

    def do_when_am_treatment(self, person_id, intervention):
        """
        This function will apply the linear model of recovery based on intervention given
        :param person_id:
        :param intervention:
        :return:
        """
        df = self.sim.population.props
        # Log that the treatment is provided:
        df.at[person_id, 'un_am_tx_start_date'] = self.sim.date

        if intervention == 'SFP':
            mam_recovery = self.wasting_models.acute_malnutrition_recovery_mam_lm.predict(
                df.loc[[person_id]], self.rng)

            if mam_recovery:
                # schedule recovery date
                self.sim.schedule_event(event=ClinicalAcuteMalnutritionRecoveryEvent(
                    module=self, person_id=person_id),
                    date=df.at[person_id, 'un_am_tx_start_date'] + DateOffset(weeks=3))
                # cancel progression date (in ProgressionEvent)
            else:
                # remained MAM
                return

        if intervention == 'OTC':
            sam_recovery = self.wasting_models.acute_malnutrition_recovery_sam_lm.predict(
                df.loc[[person_id]], self.rng)
            if sam_recovery:
                # schedule recovery date
                self.sim.schedule_event(event=ClinicalAcuteMalnutritionRecoveryEvent(
                    module=self, person_id=person_id),
                    date=df.at[person_id, 'un_am_tx_start_date'] + DateOffset(weeks=3))
                # cancel death date
                df.at[person_id, 'un_sam_death_date'] = pd.NaT
            else:
                outcome = self.rng.choice(['remained_mam', 'death'], p=[self.parameters['prob_mam_after_care'],
                                                                        self.parameters['prob_death_after_care']])
                if outcome == 'death':
                    self.sim.schedule_event(event=SevereAcuteMalnutritionDeathEvent(
                        module=self, person_id=person_id),
                        date=df.at[person_id, 'un_am_tx_start_date'] + DateOffset(weeks=3))

                else:
                    self.sim.schedule_event(event=UpdateToMAM(module=self, person_id=person_id),
                                            date=df.at[person_id, 'un_am_tx_start_date'] +
                                                 DateOffset(weeks=3))

        if intervention == 'ITC':
            sam_recovery = self.wasting_models.acute_malnutrition_recovery_sam_lm.predict(
                df.loc[[person_id]], self.rng)
            if sam_recovery:
                # schedule recovery date
                self.sim.schedule_event(event=ClinicalAcuteMalnutritionRecoveryEvent(
                    module=self, person_id=person_id),
                    date=df.at[person_id, 'un_am_tx_start_date'] + DateOffset(weeks=4))
                # cancel death date
                df.at[person_id, 'un_sam_death_date'] = pd.NaT
            else:
                outcome = self.rng.choice(['remained_mam', 'death'],
                                          p=[self.parameters['prob_mam_after_care'], self.parameters[
                                              'prob_death_after_care']])
                if outcome == 'death':
                    self.sim.schedule_event(event=SevereAcuteMalnutritionDeathEvent(
                        module=self, person_id=person_id),
                        date=df.at[person_id, 'un_am_tx_start_date'] + DateOffset(weeks=4))
                else:
                    self.sim.schedule_event(
                        event=UpdateToMAM(module=self, person_id=person_id),
                        date=df.at[person_id, 'un_am_tx_start_date'] + DateOffset(weeks=4))


class WastingPollingEvent(RegularEvent, PopulationScopeEventMixin):
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
        self.repeat_months = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat_months))
        assert isinstance(module, Wasting)

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        rng = self.module.rng

        # # # INCIDENCE OF WASTING # # # # # # # # # # # # # # # # # # # # #
        # Determine who will be onset with wasting among those who are not currently wasted -------------
        inc_wasting = df.loc[df.is_alive & (df.age_exact_years < 5) & (
                df.un_WHZ_category == 'WHZ>=-2')]
        incidence_of_wasting = self.module.wasting_models.wasting_incidence_lm.predict(inc_wasting,
                                                                                       rng=rng
                                                                                       )
        wasting_idx = inc_wasting.index
        # update the properties for wasted children
        df.loc[wasting_idx[incidence_of_wasting], 'un_ever_wasted'] = True
        df.loc[wasting_idx[incidence_of_wasting], 'un_last_wasting_date_of_onset'] = self.sim.date
        # start as moderate wasting
        df.loc[wasting_idx[incidence_of_wasting], 'un_WHZ_category'] = '-3<=WHZ<-2'
        # start without treatment
        df.loc[wasting_idx[incidence_of_wasting], 'un_am_treatment_type'] = 'none'
        # --------------------------------------------------------------------
        # Add these incident cases to the tracker
        for person in wasting_idx:
            wasting_severity = df.at[person, 'un_WHZ_category']
            age_group = WastingPollingEvent.AGE_GROUPS.get(df.loc[person].age_years, '5+y')
            # if wasting_severity != 'WHZ>=-2':
            self.module.wasting_incident_case_tracker[age_group][wasting_severity].append(self.sim.date)

        # ---------------------------------------------------------------------

        # # # PROGRESS TO SEVERE WASTING # # # # # # # # # # # # # # # # # #
        # Determine those that will progress to severe wasting (WHZ < -3) and schedule progression event ---------
        progression_sev_wasting = df.loc[df.is_alive & (df.age_exact_years < 5) &
                                         (df.un_WHZ_category == '-3<=WHZ<-2')]
        progression_severe_wasting = self.module.wasting_models.severe_wasting_progression_lm.predict(
            progression_sev_wasting, rng=rng, squeeze_single_row_output=False)

        # determine those individuals who will progress to severe wasting and time of progression
        for person in progression_sev_wasting.index[progression_severe_wasting]:
            outcome_date = self.module.date_of_outcome_for_untreated_am(person_id=person, duration_am='MAM')
            # schedule severe wasting WHZ < -3 onset
            if outcome_date <= self.sim.date:
                # schedule severe wasting (WHZ < -3) onset today
                self.sim.schedule_event(event=ProgressionSevereWastingEvent(
                    module=self.module, person_id=person), date=self.sim.date)
            else:
                # schedule severe wasting WHZ < -3 onset according to duration
                self.sim.schedule_event(
                    event=ProgressionSevereWastingEvent(
                        module=self.module, person_id=person), date=outcome_date)

        # # # MODERATE WASTING NATURAL RECOVERY # # # # # # # # # # # # # #
        # Schedule recovery from moderate wasting for those not progressing to severe wasting ---------
        for person in progression_sev_wasting.index[~progression_severe_wasting]:
            outcome_date = self.module.date_of_outcome_for_untreated_am(person_id=person, duration_am='MAM')
            if outcome_date <= self.sim.date:
                # schedule recovery for today
                self.sim.schedule_event(event=WastingNaturalRecoveryEvent(
                    module=self.module, person_id=person), date=self.sim.date)
            else:
                # schedule recovery according to duration
                self.sim.schedule_event(event=WastingNaturalRecoveryEvent(
                    module=self.module, person_id=person), date=outcome_date)

        # ------------------------------------------------------------------------------------------
        # ## UPDATE PROPERTIES RELATED TO CLINICAL ACUTE MALNUTRITION # # # #
        # ------------------------------------------------------------------------------------------
        # This applies to all children under 5

        # give MUAC measurement category for all WHZ, including well nourished children -----
        # determine the presence of bilateral oedema / oedematous malnutrition
        # determine the clinical state of acute malnutrition, and check complications if SAM
        self.module.population_poll_clinical_am(df)


class ProgressionSevereWastingEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of severe wasting (WHZ < -3).
     * Refreshes all the properties so that they pertain to this current episode of wasting
     * Imposes wasting symptom
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module

        # before progression to severe wasting, check those who started
        # supplementary feeding programme before today
        if df.at[person_id, 'un_last_wasting_date_of_onset'] < \
                df.at[person_id, 'un_am_tx_start_date'] < self.sim.date:
            return

        # continue with progression to severe if not treated/recovered
        else:
            # update properties
            df.at[person_id, 'un_WHZ_category'] = 'WHZ<-3'

            # Give MUAC measurement category for WHZ < -3
            if df.at[person_id, 'age_exact_years'] > 0.5:
                m.muac_cutoff_by_WHZ(idx=df.loc[[person_id]].index, whz='WHZ<-3')

            # update the clinical state of acute malnutrition, and check
            # complications if SAM
            m.clinical_acute_malnutrition_state(person_id=person_id, pop_dataframe=df)

            # -------------------------------------------------------------------------------------------
            # Add this incident case to the tracker
            wasting_severity = df.at[person_id, 'un_WHZ_category']
            age_group = WastingPollingEvent.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
            if wasting_severity != 'WHZ>=-2':
                m.wasting_incident_case_tracker[age_group][wasting_severity].append(self.sim.date)


class SevereAcuteMalnutritionDeathEvent(Event, IndividualScopeEventMixin):
    """
    This event applies the death function
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # The event should not run if the person is not currently alive
        if not df.at[person_id, 'is_alive']:
            return

        # # Check if this person should still die from SAM:
        if pd.isnull(df.at[person_id, 'un_am_recovery_date']) & \
                (df.at[person_id, 'un_clinical_acute_malnutrition'] == 'SAM'):
            # Cause the death to happen immediately
            df.at[person_id, 'un_sam_death_date'] = self.sim.date
            self.sim.modules['Demography'].do_death(
                individual_id=person_id,
                cause='Severe Acute Malnutrition',
                originating_module=self.module)


class WastingNaturalRecoveryEvent(Event, IndividualScopeEventMixin):
    """
    This event sets wasting properties back to normal state, based on home care/ improvement without
    interventions, low-moderate MUAC categories oedema may or may not be present
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module

        if not df.at[person_id, 'is_alive']:
            return

        df.at[person_id, 'un_am_recovery_date'] = self.sim.date
        df.at[person_id, 'un_WHZ_category'] = 'WHZ>=-2'  # not undernourished

        # For cases with normal WHZ, attribute probability of MUAC category
        if df.at[person_id, 'age_exact_years'] > 0.5:
            m.muac_cutoff_by_WHZ(idx=df.loc[[person_id]].index, whz='WHZ>=-2')

        # Note assumption: prob of oedema remained the same as applied in
        # wasting onset

        # update the clinical acute malnutrition state
        m.clinical_acute_malnutrition_state(person_id=person_id, pop_dataframe=df)
        if df.at[person_id, 'un_clinical_acute_malnutrition'] != 'SAM':
            # this will clear all wasting symptoms
            self.sim.modules["SymptomManager"].clear_symptoms(
                person_id=person_id, disease_module=self.module
            )


class ClinicalAcuteMalnutritionRecoveryEvent(Event, IndividualScopeEventMixin):
    """
    This event sets wasting properties back to normal state.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        if not df.at[person_id, 'is_alive']:
            return

        df.at[person_id, 'un_am_recovery_date'] = self.sim.date
        df.at[person_id, 'un_WHZ_category'] = 'WHZ>=-2'  # not undernourished
        df.at[person_id, 'un_clinical_acute_malnutrition'] = 'well'
        df.at[person_id, 'un_sam_death_date'] = pd.NaT
        df.at[person_id, 'un_am_bilateral_oedema'] = False
        df.at[person_id, 'un_am_MUAC_category'] = '>=125mm'
        df.at[person_id, 'un_sam_with_complications'] = False
        df.at[person_id, 'un_am_tx_start_date'] = pd.NaT
        df.at[person_id, 'un_am_treatment_type'] = 'not_applicable'

        # this will clear all wasting symptoms
        self.sim.modules["SymptomManager"].clear_symptoms(
            person_id=person_id, disease_module=self.module
        )


class UpdateToMAM(Event, IndividualScopeEventMixin):
    """
    This event updates the properties for those cases that remained/improved from SAM to MAM following
    treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        rng = m.rng
        p = m.parameters

        if not df.at[person_id, 'is_alive']:
            return

        # For cases with normal WHZ and other acute malnutrition signs:
        # oedema, or low muac - do not change the WHZ
        if df.at[person_id, 'un_WHZ_category'] == 'WHZ>=-2':
            # mam by muac only
            df.at[person_id, 'un_am_MUAC_category'] = '[115-125)mm'

        else:
            # using the probability of mam classification by anthropometric
            # indices
            mam_classification = rng.choice(['mam_by_muac_only', 'mam_by_muac_and_whz', 'mam_by_whz_only'],
                                            p=[p['proportion_mam_with_MUAC_[115-125)mm_and_normal_whz'],
                                               p['proportion_mam_with_MUAC_[115-125)mm_and_-3<=WHZ<-2'],
                                               p['proportion_mam_with_-3<=WHZ<-2_and_normal_MUAC']])

            if mam_classification == 'mam_by_muac_only':
                df.at[person_id, 'un_WHZ_category'] = 'WHZ>=-2'
                df.at[person_id, 'un_am_MUAC_category'] = '[115-125)mm'

            if mam_classification == 'mam_by_muac_and_whz':
                df.at[person_id, 'un_WHZ_category'] = '-3<=WHZ<-2'
                df.at[person_id, 'un_am_MUAC_category'] = '[115-125)mm'

            if mam_classification == 'mam_by_whz_only':
                df.at[person_id, 'un_WHZ_category'] = '-3<=WHZ<-2'
                df.at[person_id, 'un_am_MUAC_category'] = '>=125mm'

        # Update all other properties equally
        df.at[person_id, 'un_clinical_acute_malnutrition'] = 'MAM'
        df.at[person_id, 'un_am_bilateral_oedema'] = False
        df.at[person_id, 'un_sam_with_complications'] = False
        df.at[person_id, 'un_am_tx_start_date'] = pd.NaT
        df.at[person_id, 'un_am_recovery_date'] = pd.NaT
        df.at[person_id, 'un_am_discharge_date'] = pd.NaT
        # will start the process again
        df.at[person_id, 'un_am_treatment_type'] = 'not_applicable'

        # this will clear all wasting symptoms (applicable for SAM, not MAM)
        self.sim.modules["SymptomManager"].clear_symptoms(
            person_id=person_id, disease_module=self.module
        )


class HSI_Wasting_SupplementaryFeedingProgramme_MAM(HSI_Event, IndividualScopeEventMixin):
    """
    This is the supplementary feeding programme for MAM without complications
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Wasting)

        # Get a blank footprint and then edit to define call on resources
        # of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Undernutrition_Feeding_Supplementary'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        # Stop the person from dying of acute malnutrition (if they were
        # going to die)
        if not df.at[person_id, 'is_alive']:
            return

        # Do here whatever happens to an individual during this health
        # system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~
        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['item_and_package_code_lookups']
        # individual items
        item_code1 = pd.unique(consumables.loc[consumables['Items'] ==
                                               'Corn Soya Blend (or Supercereal - CSB++)', 'Item_Code'])[0]

        # check availability of consumables
        if self.get_consumables([item_code1]):
            logger.debug(key='debug', data='consumables are available')
            # Log that the treatment is provided:
            df.at[person_id, 'un_am_tx_start_date'] = self.sim.date
            df.at[person_id, 'un_am_discharge_date'] = self.sim.date + DateOffset(weeks=3)
            df.at[person_id, 'un_am_treatment_type'] = 'CSB++'
            self.module.do_when_am_treatment(person_id, intervention='SFP')
        else:
            logger.debug(key='debug', data="PkgCode1 is not available, so can't use it.")

    def did_not_run(self):
        logger.debug("Undernutrition_Feeding_Supplementary: did not run")
        pass


class HSI_Wasting_OutpatientTherapeuticProgramme_SAM(HSI_Event, IndividualScopeEventMixin):
    """
    This is the outpatient management of SAM without any medical complications
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Wasting)

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
        df = self.sim.population.props

        # Stop the person from dying of acute malnutrition (if they were
        # going to die)
        if not df.at[person_id, 'is_alive']:
            return

        # Do here whatever happens to an individual during this health
        # system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~
        # Make request for some consumables
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
            df.at[person_id, 'un_am_tx_start_date'] = self.sim.date
            df.at[person_id, 'un_am_discharge_date'] = self.sim.date + DateOffset(weeks=3)
            df.at[person_id, 'un_am_treatment_type'] = 'standard_RUTF'
            self.module.do_when_am_treatment(person_id, intervention='OTC')
        else:
            logger.debug(key='debug', data="consumables not available, so can't use it.")

    def did_not_run(self):
        logger.debug("HSI_Undernutrition_Feeding_Outpatient: did not run")
        pass


class HSI_Wasting_InpatientCareForComplicated_SAM(HSI_Event, IndividualScopeEventMixin):
    """
    This is the inpatient management of SAM with medical complications
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Wasting)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint['U5Malnutr'] = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Undernutrition_Feeding_Inpatient'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '2'
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 7})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        # Stop the person from dying of acute malnutrition (if they were going to die)
        if not df.at[person_id, 'is_alive']:
            return

        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['item_and_package_code_lookups']

        # individual items
        item_code1 = pd.unique(
            consumables.loc[consumables['Items'] == 'SAM theraputic foods', 'Item_Code'])[0]
        item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'SAM medicines', 'Item_Code'])[0]

        # # check availability of consumables
        if self.get_consumables(item_code1) and self.get_consumables(item_code2):
            logger.debug(key='debug', data='consumables available, so use it.')
            # Log that the treatment is provided:
            df.at[person_id, 'un_am_tx_start_date'] = self.sim.date
            df.at[person_id, 'un_am_discharge_date'] = self.sim.date + DateOffset(weeks=4)
            df.at[person_id, 'un_am_treatment_type'] = 'inpatient_care'
            self.module.do_when_am_treatment(person_id, intervention='ITC')
        else:
            logger.debug(key='debug', data="consumables not available, so can't use it.")

    def did_not_run(self):
        logger.debug("HSI_inpatient_care_for_complicated_SAM: did not run")
        pass


class WastingModels:
    """ houses all wasting linear models """

    def __init__(self, module):
        self.module = module
        self.params = module.parameters
        self.rng = module.rng

        # a linear model to predict the probability of individual's recovery from moderate acute malnutrition
        self.acute_malnutrition_recovery_mam_lm = LinearModel.multiplicative(
            Predictor('un_am_treatment_type', conditions_are_mutually_exclusive=True, conditions_are_exhaustive=True)
            .when('soy_RUSF', self.params['recovery_rate_with_soy_RUSF'])
            .when('CSB++', self.params['recovery_rate_with_CSB++'])
        )

        # a linear model to predict the probability of individual's recovery from severe acute malnutrition
        self.acute_malnutrition_recovery_sam_lm = LinearModel.multiplicative(
            Predictor('un_am_treatment_type', conditions_are_mutually_exclusive=True, conditions_are_exhaustive=True)
            .when('standard_RUTF', self.params['recovery_rate_with_standard_RUTF'])
            .when('inpatient_care', self.params['recovery_rate_with_inpatient_care'])
        )

        # Linear model for the probability of progression to severe wasting (age-dependent only)
        # (natural history only, no interventions)
        self.severe_wasting_progression_lm = LinearModel.multiplicative(
            Predictor('age_exact_years', conditions_are_mutually_exclusive=True, conditions_are_exhaustive=False)
            .when('<0.5', self.params['progression_severe_wasting_by_agegp'][0])
            .when('.between(0.5,1, inclusive="left")', self.params['progression_severe_wasting_by_agegp'][1])
            .when('.between(1,2, inclusive="left")', self.params['progression_severe_wasting_by_agegp'][2])
            .when('.between(2,3, inclusive="left")', self.params['progression_severe_wasting_by_agegp'][3])
            .when('.between(3,4, inclusive="left")', self.params['progression_severe_wasting_by_agegp'][4])
            .when('.between(4,5, inclusive="left")', self.params['progression_severe_wasting_by_agegp'][5])
        )

        # get wasting incidence linear model
        self.wasting_incidence_lm = self.get_wasting_incidence()

    def get_wasting_incidence(self) -> LinearModel:
        """ return a scaled wasting incidence linear model amongst young children
        :params df: population dataframe """
        df = self.module.sim.population.props

        def unscaled_wasting_lm(intercept: Union[float, int] = 1.0) -> LinearModel:
            # linear model to predict the incidence of wasting
            return LinearModel(
                LinearModelType.MULTIPLICATIVE,
                intercept,
                Predictor('age_exact_years', conditions_are_mutually_exclusive=True, conditions_are_exhaustive=False)
                .when('<0.5', self.params['base_inc_rate_wasting_by_agegp'][0])
                .when('.between(0.5,1, inclusive="left")', self.params['base_inc_rate_wasting_by_agegp'][1])
                .when('.between(1,2, inclusive="left")', self.params['base_inc_rate_wasting_by_agegp'][2])
                .when('.between(2,3, inclusive="left")', self.params['base_inc_rate_wasting_by_agegp'][3])
                .when('.between(3,4, inclusive="left")', self.params['base_inc_rate_wasting_by_agegp'][4])
                .when('.between(4,5, inclusive="left")', self.params['base_inc_rate_wasting_by_agegp'][5]),
                Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                 '& (nb_late_preterm == False) & (nb_early_preterm == False)',
                                 self.params['rr_wasting_SGA_and_term']),
                Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                 '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                                 self.params['rr_wasting_SGA_and_preterm']),
                Predictor().when('(nb_size_for_gestational_age == "average_for_gestational_age") '
                                 '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                                 self.params['rr_wasting_preterm_and_AGA']),
                Predictor('li_wealth').apply(
                    lambda x: 1 if x == 1 else (x - 1) ** (self.params['rr_wasting_wealth_level'])),
            )

        unscaled_lm = unscaled_wasting_lm()
        target_mean = self.params['base_inc_rate_wasting_by_agegp'][2]  # base inc rate for 12-23mo old
        actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 1) &
                                                 (df.un_WHZ_category == 'WHZ>=-2')]).mean()

        scaled_intercept = 1.0 * (target_mean / actual_mean) \
            if (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else 1.0
        scaled_wasting_incidence_lm = unscaled_wasting_lm(intercept=scaled_intercept)
        return scaled_wasting_incidence_lm

    def get_wasting_prevalence(self, agegp: str) -> LinearModel:
        """ return a scaled wasting prevalence linear model amongst young children less than 5 years
        :params df: population dataframe
        :param agegp: children's age group
        """
        df = self.module.sim.population.props

        def make_linear_model_wasting(intercept: Union[float, int]) -> LinearModel:
            return LinearModel(
                LinearModelType.LOGISTIC,
                intercept,  # baseline odds: get_odds_wasting(agegp=agegp)
                Predictor('li_wealth', conditions_are_mutually_exclusive=True, conditions_are_exhaustive=False)
                .when(2, self.params['or_wasting_hhwealth_Q2'])
                .when(3, self.params['or_wasting_hhwealth_Q3'])
                .when(4, self.params['or_wasting_hhwealth_Q4'])
                .when(5, self.params['or_wasting_hhwealth_Q5']),
                Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                 '& (nb_late_preterm == False) & (nb_early_preterm == False)',
                                 self.params['or_wasting_SGA_and_term']),
                Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                 '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                                 self.params['or_wasting_SGA_and_preterm']),
                Predictor().when('(nb_size_for_gestational_age == "average_for_gestational_age") '
                                 '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                                 self.params['or_wasting_preterm_and_AGA'])
            )

        get_odds_wasting = self.module.get_prob_severe_wasting_or_odds_wasting(agegp=agegp, get_odds=True)
        unscaled_lm = make_linear_model_wasting(intercept=get_odds_wasting)
        target_mean = self.module.get_prob_severe_wasting_or_odds_wasting(agegp='12_23mo', get_odds=True)
        actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 1)]).mean()
        scaled_intercept = get_odds_wasting * (target_mean / actual_mean) if \
            (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else get_odds_wasting
        scaled_lm = make_linear_model_wasting(intercept=scaled_intercept)

        return scaled_lm


class WastingLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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
        inc_df = pd.DataFrame(index=self.module.wasting_incident_case_tracker.keys(),
                              columns=self.module.wasting_states)
        for age_grp in self.module.wasting_incident_case_tracker.keys():
            for state in self.module.wasting_states:
                inc_df.loc[age_grp, state] = len(self.module.wasting_incident_case_tracker[age_grp][state])

        logger.info(key='wasting_incidence_count', data=inc_df.to_dict())

        # Reset the counters and the date_last_run
        self.module.wasting_incident_case_tracker = copy.deepcopy(
            self.module.wasting_incident_case_tracker_blank)
        self.date_last_run = self.sim.date

        # Wasting totals (prevalence at logging time)
        under5s = df.loc[df.is_alive & df.age_exact_years < 5]
        # declare a dictionary that will hold proportions of wasting prevalence per each age group
        wasting_prev_dict: Dict[str, Any] = dict()
        # loop through different age groups and get proportions of wasting prevalence per each age group
        for low_bound_mos, high_bound_mos in [(0, 5), (6, 11), (12, 23), (24, 35), (36, 47), (48, 59)]:  # in months
            low_bound_age_in_years = low_bound_mos / 12.0
            high_bound_age_in_years = (1 + high_bound_mos) / 12.0
            # get those children who are wasted
            wasted_agegrp = (under5s.age_exact_years.between(low_bound_age_in_years, high_bound_age_in_years,
                                                             inclusive='left') & (under5s.un_WHZ_category
                                                                                  != 'WHZ>=-2')).sum()
            total_per_agegrp = (under5s.age_exact_years < high_bound_age_in_years).sum()
            # add proportions to the dictionary
            wasting_prev_dict[f'{low_bound_mos}_{high_bound_mos}mo'] = wasted_agegrp / total_per_agegrp

        # add to dictionary proportion of all wasted children under 5 years
        wasting_prev_dict['total_under5_prop'] = (under5s.un_WHZ_category != 'WHZ>=-2').sum() / len(under5s)
        # log wasting prevalence
        logger.info(key='wasting_prevalence_count', data=wasting_prev_dict)
