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
        'duration_of_untreated_mod_wasting': Parameter(
            Types.REAL, 'duration of untreated moderate wasting (days)'),
        'duration_of_untreated_sev_wasting': Parameter(
            Types.REAL, 'duration of untreated severe wasting (days)'),
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
        # bilateral oedema
        'prevalence_nutritional_oedema': Parameter(
            Types.REAL, 'prevalence of nutritional oedema in children under 5 in Malawi'),
        'proportion_WHZ<-2_with_oedema': Parameter(
            Types.REAL, 'proportion of individuals with wasting (moderate or severe) who have oedema'),
        'proportion_oedema_with_WHZ<-2': Parameter(
            Types.REAL, 'proportion of individuals with oedema who are wasted (moderately or severely)'),
        # treatment/interventions
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
        # recovery due to treatment/interventions
        'recovery_rate_with_standard_RUTF': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with standard RUTF'),
        'recovery_rate_with_soy_RUSF': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with soy RUSF'),
        'recovery_rate_with_CSB++': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with CSB++'),
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
        self.prob_normal_whz = None
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
        """
        # Read parameters from the resource file
        self.load_parameters_from_dataframe(
            pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_Wasting.csv')
        )

        # Register wasting symptom (weight loss) in Symptoms Manager
        self.sim.modules['SymptomManager'].register_symptom(Symptom(name=self.wasting_symptom))

    def initialise_population(self, population):
        """
        Set our property values for the initial population. This method is called by the simulation when creating
        the initial population, and is responsible for assigning initial values, for every individual,
        of those properties 'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population:
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
            children_of_agegp = df.loc[df.is_alive & df.age_exact_years.between(
                low_bound_age_in_years, high_bound_age_in_years, inclusive='left'
            )]

            # apply prevalence of wasting and categorise into moderate (-3 <= WHZ < -2) or severe (WHZ < -3) wasting
            wasted = self.wasting_models.get_wasting_prevalence(agegp=agegp).predict(
                children_of_agegp, self.rng, False
            )
            probability_of_severe = self.get_prob_severe_wasting_among_wasted(agegp=agegp)
            for idx in children_of_agegp.index[wasted]:
                wasted_category = self.rng.choice(['WHZ<-3', '-3<=WHZ<-2'], p=[probability_of_severe,
                                                                               1 - probability_of_severe])
                df.at[idx, 'un_WHZ_category'] = wasted_category
                df.at[idx, 'un_last_wasting_date_of_onset'] = self.sim.date
                df.at[idx, 'un_ever_wasted'] = True
                # start without treatment
                df.at[idx, 'un_am_treatment_type'] = 'none'

        index_under5 = df.index[df.is_alive & (df.age_exact_years < 5)]
        # calculate approximation of probability of having normal WHZ in children under 5 to be used later
        self.prob_normal_whz = \
            len(index_under5.intersection(df.index[df.un_WHZ_category == 'WHZ>=-2'])) / len(index_under5)
        # -------------------------------------------------------------------------------------------------- #
        # # # #    Give MUAC category, presence of oedema, and determine acute malnutrition state      # # # #
        # # # #    and, in SAM cases, determine presence of complications                             # # # #
        self.clinical_signs_acute_malnutrition(index_under5)

    def initialise_simulation(self, sim):
        """Prepares for simulation:
        * Schedules the main polling event
        * Schedules the main logging event
        """

        # schedule wasting pool event
        sim.schedule_event(WastingIncidencePollingEvent(self), sim.date + DateOffset(months=3))

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

    def get_prob_severe_wasting_among_wasted(self, agegp: str) -> Union[float, int]:
        """
        This function will calculate the WHZ scores by categories and return probability of severe wasting
        for those with wasting status
        :param agegp: age grouped in months
        :return: probability of severe wasting among all wasting cases
        """
        # generate random numbers from N(mean, sd)
        mean, stdev = self.parameters[f'prev_WHZ_distribution_age_{agegp}']
        whz_normal_distribution = norm(loc=mean, scale=stdev)

        # get probability of any wasting: WHZ < -2
        probability_less_than_minus2sd = 1 - whz_normal_distribution.sf(-2)

        # get probability of severe wasting: WHZ < -3
        probability_less_than_minus3sd = 1 - whz_normal_distribution.sf(-3)

        # make WHZ < -2 as the 100% and get the adjusted probability of severe wasting within overall wasting
        # return the probability of severe wasting among all wasting cases
        return probability_less_than_minus3sd / probability_less_than_minus2sd

    def get_odds_wasting(self, agegp: str) -> Union[float, int]:
        """
        This function will calculate the WHZ scores by categories and return odds of wasting
        :param agegp: age grouped in months
        :return: odds of wasting among all children under 5
        """
        # generate random numbers from N(mean, sd)
        mean, stdev = self.parameters[f'prev_WHZ_distribution_age_{agegp}']
        whz_normal_distribution = norm(loc=mean, scale=stdev)

        # get probability of any wasting: WHZ < -2
        probability_less_than_minus2sd = 1 - whz_normal_distribution.sf(-2)

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
        This function applies the probability of bilateral oedema present in wasting and non-wasted cases
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
        df.loc[children_with_wasting, 'un_am_bilateral_oedema'] = oedema_in_wasted_children

        # oedema among non-wasted children
        if len(children_without_wasting) == 0:
            return
        # proportion_normalWHZ_with_oedema: P(oedema|WHZ>=-2) =
        # P(oedema & WHZ>=-2) / P(WHZ>=-2) = P(oedema) * [1 - P(WHZ<-2|oedema)] / P(WHZ>=-2)
        print(f"{self.prob_normal_whz=}")
        proportion_normal_whz_with_oedema = \
            p['prevalence_nutritional_oedema'] * (1 - p['proportion_oedema_with_WHZ<-2']) / self.prob_normal_whz
        oedema_in_non_wasted = self.rng.random_sample(size=len(
            children_without_wasting)) < proportion_normal_whz_with_oedema
        df.loc[children_without_wasting, 'un_am_bilateral_oedema'] = oedema_in_non_wasted

    def clinical_acute_malnutrition_state(self, person_id, pop_dataframe):
        """
        This function will determine the clinical acute malnutrition status (MAM, SAM) based on anthropometric indices
        and presence of bilateral oedema (Kwashiorkor); And help determine whether the individual will have medical
        complications, applicable to SAM cases only, requiring inpatient care.
        :param person_id: individual id
        :param pop_dataframe: population dataframe
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
            # clear all wasting symptoms
            self.sim.modules["SymptomManager"].clear_symptoms(
                person_id=person_id, disease_module=self
            )

        assert not (df.at[person_id, 'un_clinical_acute_malnutrition'] == 'MAM') & \
                   (df.at[person_id, 'un_sam_with_complications'])

    def date_of_outcome_for_untreated_wasting(self, person_id):
        """
        Helper function to use the duration of the wasting episode to get date of outcome (recovery, progression,
        or death)
        :param person_id:
        :return: date of outcome, which can be recovery to no wasting or progression to severe wasting from moderate
        wasting; or recovery to moderate wasting or death due to severe wasting
        """
        df = self.sim.population.props
        p = self.parameters
        whz_category = df.at[person_id, 'un_WHZ_category']

        # moderate wasting (for progression to severe, or recovery to no wasting) -----
        if whz_category == '-3<=WHZ<-2':
            # Allocate the duration of the moderate wasting episode
            duration_mod_wasting = int(max(p['min_days_duration_of_wasting'], p['duration_of_untreated_mod_wasting']))
            # Allocate a date of outcome (progression, or recovery)
            date_of_outcome = df.at[person_id, 'un_last_wasting_date_of_onset'] + DateOffset(days=duration_mod_wasting)
            return date_of_outcome

        # severe wasting (recovery to moderate wasting) -----
        if whz_category == 'WHZ<-3':
            # determine the duration of severe wasting episode
            duration_sev_wasting = int(max(p['min_days_duration_of_wasting'], p['duration_of_untreated_mod_wasting']
                                           + p['duration_of_untreated_sev_wasting']))
            # Allocate a date of outcome (recovery)
            date_of_outcome = df.at[person_id, 'un_last_wasting_date_of_onset'] + DateOffset(days=duration_sev_wasting)
            return date_of_outcome

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

        # determine the presence of bilateral oedema / oedematous malnutrition
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
        :return:
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
                              df.un_am_bilateral_oedema] = daly_wts['sev_wasting_with_oedema']
        total_daly_values.loc[df.is_alive & (df.un_WHZ_category == 'WHZ<-3') &
                              (~df.un_am_bilateral_oedema)] = daly_wts['sev_wasting_w/o_oedema']
        total_daly_values.loc[df.is_alive & (df.un_WHZ_category == '-3<=WHZ<-2') &
                              df.un_am_bilateral_oedema] = daly_wts['mod_wasting_with_oedema']
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
        """
        df = self.sim.population.props
        p = self.parameters
        # Set the date when the treatment is provided:
        df.at[person_id, 'un_am_tx_start_date'] = self.sim.date

        if intervention == 'SFP':
            mam_full_recovery = self.wasting_models.acute_malnutrition_recovery_mam_lm.predict(
                df.loc[[person_id]], self.rng
            )

            if mam_full_recovery:
                # schedule recovery date
                self.sim.schedule_event(
                    event=WastingClinicalAcuteMalnutritionRecoveryEvent(module=self, person_id=person_id),
                    date=(df.at[person_id, 'un_am_tx_start_date'] +
                          DateOffset(weeks=p['tx_length_weeks_SuppFeedingMAM']))
                )
                # cancel progression date (in ProgressionEvent)
            else:
                # remained MAM
                return

        elif (intervention == 'OTC') or (intervention == 'ITC'):
            if intervention == 'OTC':
                outcome_date = (df.at[person_id, 'un_am_tx_start_date'] +
                                DateOffset(weeks=p['tx_length_weeks_OutpatientSAM']))
            else:
                outcome_date = (df.at[person_id, 'un_am_tx_start_date'] +
                                DateOffset(weeks=p['tx_length_weeks_InpatientSAM']))

            sam_full_recovery = self.wasting_models.acute_malnutrition_recovery_sam_lm.predict(
                df.loc[[person_id]], self.rng
            )
            if sam_full_recovery:
                # schedule full recovery
                self.sim.schedule_event(
                    event=WastingClinicalAcuteMalnutritionRecoveryEvent(module=self, person_id=person_id),
                    date=outcome_date
                )
                # cancel death date
                df.at[person_id, 'un_sam_death_date'] = pd.NaT
            else:
                outcome = self.rng.choice(['recovery_to_mam', 'death'], p=[self.parameters['prob_mam_after_care'],
                                                                           self.parameters['prob_death_after_care']])
                if outcome == 'death':
                    self.sim.schedule_event(
                        event=WastingSevereAcuteMalnutritionDeathEvent(module=self, person_id=person_id),
                        date=outcome_date
                    )
                else:
                    self.sim.schedule_event(event=WastingUpdateToMAM(module=self, person_id=person_id),
                                            date=outcome_date)
                    # cancel death date
                    df.at[person_id, 'un_sam_death_date'] = pd.NaT


class WastingIncidencePollingEvent(RegularEvent, PopulationScopeEventMixin):
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

        # # # INCIDENCE OF MODERATE WASTING # # # # # # # # # # # # # # # # # # # # #
        # Determine who will be onset with wasting among those who are not currently wasted -------------
        not_wasted = df.loc[df.is_alive & (df.age_exact_years < 5) & (
            df.un_WHZ_category == 'WHZ>=-2')]
        incidence_of_wasting = self.module.wasting_models.wasting_incidence_lm.predict(not_wasted, rng=rng)
        mod_wasting_new_cases_idx = not_wasted.index[incidence_of_wasting]
        # update the properties for new cases of wasted children
        df.loc[mod_wasting_new_cases_idx, 'un_ever_wasted'] = True
        df.loc[mod_wasting_new_cases_idx, 'un_last_wasting_date_of_onset'] = self.sim.date
        # initiate moderate wasting
        df.loc[mod_wasting_new_cases_idx, 'un_WHZ_category'] = '-3<=WHZ<-2'
        # start without treatment
        df.loc[mod_wasting_new_cases_idx, 'un_am_treatment_type'] = 'none'
        # -------------------------------------------------------------------------------------------
        # Add these incident cases to the tracker
        for person_id in mod_wasting_new_cases_idx:
            age_group = WastingIncidencePollingEvent.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
            self.module.wasting_incident_case_tracker[age_group]['-3<=WHZ<-2'].append(self.sim.date)
        # Update properties related to clinical acute malnutrition
        # (MUAC, oedema, clinical state of acute malnutrition and if SAM complications; clear symptoms if not SAM)
        self.module.clinical_signs_acute_malnutrition(mod_wasting_new_cases_idx)
        # -------------------------------------------------------------------------------------------

        # # # PROGRESS TO SEVERE WASTING # # # # # # # # # # # # # # # # # #
        # Determine those that will progress to severe wasting (WHZ < -3) and schedule progression event ---------
        progression_severe_wasting = self.module.wasting_models.severe_wasting_progression_lm.predict(
            df.loc[mod_wasting_new_cases_idx], rng=rng, squeeze_single_row_output=False
        )

        for person in mod_wasting_new_cases_idx[progression_severe_wasting]:
            outcome_date = self.module.date_of_outcome_for_untreated_wasting(person_id=person)
            # schedule severe wasting WHZ < -3 onset
            if outcome_date <= self.sim.date:
                # schedule severe wasting (WHZ < -3) onset today
                self.sim.schedule_event(event=WastingProgressionToSevereEvent(
                    module=self.module, person_id=person), date=self.sim.date)
            else:
                # schedule severe wasting WHZ < -3 onset according to duration
                self.sim.schedule_event(
                    event=WastingProgressionToSevereEvent(
                        module=self.module, person_id=person), date=outcome_date)

        # # # MODERATE WASTING NATURAL RECOVERY # # # # # # # # # # # # # #
        # Schedule recovery from moderate wasting for those not progressing to severe wasting ---------
        for person in mod_wasting_new_cases_idx[~progression_severe_wasting]:
            outcome_date = self.module.date_of_outcome_for_untreated_wasting(person_id=person)
            if outcome_date <= self.sim.date:
                # schedule recovery for today
                self.sim.schedule_event(event=WastingNaturalRecoveryEvent(
                    module=self.module, person_id=person), date=self.sim.date)
            else:
                # schedule recovery according to duration
                self.sim.schedule_event(event=WastingNaturalRecoveryEvent(
                    module=self.module, person_id=person), date=outcome_date)


class WastingProgressionToSevereEvent(Event, IndividualScopeEventMixin):
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

        if ((not df.at[person_id, 'is_alive']) or (df.at[person_id, 'age_exact_years'] >= 5)
                or df.at[person_id, 'un_WHZ_category'] != '-3<=WHZ<-2'):
            return

        # before progression to severe wasting, check those who started
        # supplementary feeding programme before today
        if df.at[person_id, 'un_last_wasting_date_of_onset'] < \
                df.at[person_id, 'un_am_tx_start_date'] < self.sim.date:
            return
        # # # INCIDENCE OF SEVERE WASTING # # # # # # # # # # # # # # # # # # # # #
        # continue with progression to severe if not treated/recovered
        else:
            # update properties
            # - WHZ
            df.at[person_id, 'un_WHZ_category'] = 'WHZ<-3'
            # - MUAC, oedema, clinical state of acute malnutrition, complications
            self.module.clinical_signs_acute_malnutrition(person_id)

            # -------------------------------------------------------------------------------------------
            # Add this severe wasting incident case to the tracker
            age_group = WastingIncidencePollingEvent.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
            m.wasting_incident_case_tracker[age_group]['WHZ<-3'].append(self.sim.date)


class WastingSevereAcuteMalnutritionDeathEvent(Event, IndividualScopeEventMixin):
    """
    This event applies the death function
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # The event should not run if the person is not currently alive
        if ((not df.at[person_id, 'is_alive'])
                or df.at[person_id, 'un_clinical_acute_malnutrition'] != 'SAM'):
            return

        # # Check if this person should still die from SAM:
        if pd.isnull(df.at[person_id, 'un_am_recovery_date']):
            # Cause the death to happen immediately
            df.at[person_id, 'un_sam_death_date'] = self.sim.date
            self.sim.modules['Demography'].do_death(
                individual_id=person_id,
                cause='Severe Acute Malnutrition',
                originating_module=self.module)


class WastingNaturalRecoveryEvent(Event, IndividualScopeEventMixin):
    """
    This event improves wasting by 1 SD, based on home care/improvement without interventions.
    MUAC, oedema, clinical state of acute malnutrition, and if SAM complications are updated,
    and symptoms cleared if not SAM.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        if (not df.at[person_id, 'is_alive']) or (df.at[person_id, 'un_WHZ_category'] == 'WHZ>=-2'):
            return

        whz = df.at[person_id, 'un_WHZ_category']
        if whz == '-3<=WHZ<-2':
            # improve WHZ
            df.at[person_id, 'un_WHZ_category'] = 'WHZ>=-2'  # not undernourished
        else:
            # whz == 'WHZ<-3'
            # improve WHZ
            df.at[person_id, 'un_WHZ_category'] = '-3<=WHZ<-2'  # moderate wasting

        # update MUAC, oedema, clinical state of acute malnutrition and if SAM complications,
        # clear symptoms if not SAM
        self.module.clinical_signs_acute_malnutrition(person_id)
        # set recovery date if recovered
        if df.at[person_id, 'un_clinical_acute_malnutrition'] == 'well':
            df.at[person_id, 'un_am_recovery_date'] = self.sim.date


class WastingClinicalAcuteMalnutritionRecoveryEvent(Event, IndividualScopeEventMixin):
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


class WastingUpdateToMAM(Event, IndividualScopeEventMixin):
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

        if (not df.at[person_id, 'is_alive']) or (df.at[person_id, 'un_clinical_acute_malnutrition'] != 'SAM'):
            return

        # For cases with normal WHZ and other acute malnutrition signs:
        # oedema, or low MUAC - do not change the WHZ
        if df.at[person_id, 'un_WHZ_category'] == 'WHZ>=-2':
            # MAM by MUAC only
            df.at[person_id, 'un_am_MUAC_category'] = '[115-125)mm'
            # TODO: I think this changes the proportions below as some of the cases will be issued here

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
        the_appt_footprint['Under5OPD'] = 1  # This requires one outpatient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Undernutrition_Feeding_Supplementary'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            return

        # Do here whatever happens to an individual during this health system interaction event
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
            logger.debug(key='debug',
                         data=f"Consumable(s) not available, hence {self.TREATMENT_ID} cannot be provided.")

    def did_not_run(self):
        logger.debug(key='debug', data=f'{self.TREATMENT_ID}: did not run')
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
            logger.debug(key='debug',
                         data=f"Consumable(s) not available, hence {self.TREATMENT_ID} cannot be provided.")

    def did_not_run(self):
        logger.debug(key='debug', data=f'{self.TREATMENT_ID}: did not run')
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
            logger.debug(key='debug',
                         data=f"Consumable(s) not available, hence {self.TREATMENT_ID} cannot be provided.")

    def did_not_run(self):
        logger.debug(key='debug', data=f'{self.TREATMENT_ID}: did not run')
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
        """
        :return: a scaled wasting incidence linear model amongst young children
        """
        df = self.module.sim.population.props

        def unscaled_wasting_incidence_lm(intercept: Union[float, int] = 1.0) -> LinearModel:
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

        unscaled_lm = unscaled_wasting_incidence_lm()
        target_mean = self.params['base_inc_rate_wasting_by_agegp'][2]  # base inc rate for 12-23mo old
        actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 1) &
                                                 (df.un_WHZ_category != 'WHZ>=-2')]).mean()

        scaled_intercept = 1.0 * (target_mean / actual_mean) \
            if (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else 1.0
        scaled_wasting_incidence_lm = unscaled_wasting_incidence_lm(intercept=scaled_intercept)
        return scaled_wasting_incidence_lm

    def get_wasting_prevalence(self, agegp: str) -> LinearModel:
        """
        :param agegp: children's age group
        :return: a scaled wasting prevalence linear model amongst young children less than 5 years
        """
        df = self.module.sim.population.props

        def unscaled_wasting_prevalence_lm(intercept: Union[float, int]) -> LinearModel:
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

        get_odds_wasting = self.module.get_odds_wasting(agegp=agegp)
        unscaled_lm = unscaled_wasting_prevalence_lm(intercept=get_odds_wasting)
        target_mean = self.module.get_odds_wasting(agegp='12_23mo')
        actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 1)]).mean()
        scaled_intercept = get_odds_wasting * (target_mean / actual_mean) if \
            (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else get_odds_wasting
        scaled_wasting_prevalence_lm = unscaled_wasting_prevalence_lm(intercept=scaled_intercept)

        return scaled_wasting_prevalence_lm


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
                assert all(date >= self.date_last_run for
                           date in self.module.wasting_incident_case_tracker[age_grp][state])

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
            total_per_agegrp = (under5s.age_exact_years.between(low_bound_age_in_years, high_bound_age_in_years,
                                                                inclusive='left')).sum()
            # add proportions to the dictionary
            wasting_prev_dict[f'{low_bound_mos}_{high_bound_mos}mo'] = wasted_agegrp / total_per_agegrp

        # add to dictionary proportion of all wasted children under 5 years
        wasting_prev_dict['total_under5_prop'] = (under5s.un_WHZ_category != 'WHZ>=-2').sum() / len(under5s)
        # log wasting prevalence
        logger.info(key='wasting_prevalence_props', data=wasting_prev_dict)
