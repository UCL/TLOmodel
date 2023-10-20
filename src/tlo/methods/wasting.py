"""Childhood wasting module"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------


class Wasting(Module):
    """
    This module applies the prevalence of wasting at the population-level,
    based on the Malawi DHS Survey 2015-2016.
    The definitions:
    - moderate wasting: weight_for_height Z-score (WHZ) <-2 SD from the reference mean
    - severe wasting: weight_for_height Z-score (WHZ) <-3 SD from the reference mean

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
        'SAM': Cause(gbd_causes='Protein-energy malnutrition', label='Childhood Wasting')
    }

    # Declare Causes of Death and Disability
    CAUSES_OF_DISABILITY = {
        'SAM': Cause(gbd_causes='Protein-energy malnutrition', label='Childhood Wasting')
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
            Types.REAL, 'odds ratio of wasting if born term and small for geatational age'),
        'or_wasting_SGA_and_preterm': Parameter(
            Types.REAL, 'odds ratio of wasting if born preterm and small for gestational age'),
        # incidence parameters
        'base_inc_rate_wasting_by_agegp': Parameter(
            Types.LIST, 'List with baseline incidence of wasting by age group'),
        'rr_wasting_preterm_and_AGA': Parameter(
            Types.REAL, 'relative risk of wasting if born preterm and adequate for gestational age'),
        'rr_wasting_SGA_and_term': Parameter(
            Types.REAL, 'relative risk of wasting if born term and small for geatational age'),
        'rr_wasting_SGA_and_preterm': Parameter(
            Types.REAL, 'relative risk of wasting if born preterm and small for gestational age'),
        'rr_wasting_wealth_level': Parameter(
            Types.REAL, 'relative risk of wasting per 1 unit decrease in wealth level'),
        'min_days_duration_of_wasting': Parameter(
            Types.REAL, 'minimum duration in days of wasting (MAM and SAM)'),
        'average_duration_of_untreated_MAM': Parameter(
            Types.REAL, 'average duration of untreated MAM'),
        # progression to severe parameters
        'progression_severe_wasting_by_agegp': Parameter(
            Types.LIST, 'List with progression rates to severe wasting by age group'),
        'average_duration_of_untreated_SAM': Parameter(
            Types.REAL, 'average duration of untreated SAM'),
        'prob_complications_in_SAM': Parameter(
            Types.REAL, 'probability of medical complications in SAM '),
        # recovery parameters
        'recovery_rate_with_standard_RUTF': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with standard RUTF'),
        'recovery_rate_with_soy_RUSF': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with soy RUSF'),
        'recovery_rate_with_CSB++': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with CSB++'),
        'recovery_rate_with_inpatient_care': Parameter(
            Types.REAL, 'probability of recovery from wasting following treatment with inpatient care'),
        # MUAC distributions
        'MUAC_distribution_WHZ<-3': Parameter(
            Types.LIST, 'mean and standard deviation of a normal distribution of MUAC measurements for WHZ<-3'),
        'MUAC_distribution_-3<=WHZ<-2': Parameter(
            Types.LIST, 'mean and standard deviation of a normal distribution of MUAC measurements for -3<=WHZ<-2'),
        'MUAC_distribution_WHZ>=-2': Parameter(
            Types.LIST, 'mean and standard deviation of a normal distribution of MUAC measurements for WHZ>=-2'),

        'proportion_WHZ<-3_with_MUAC<115mm': Parameter(
            Types.REAL, 'proportion of severe weight-for-height Z-score with MUAC<115mm'),
        'proportion_-3<=WHZ<-2_with_MUAC<115mm': Parameter(
            Types.REAL, 'proportion of moderate weight-for-height Z-score with MUAC<115mm'),
        'proportion_-3<=WHZ<-2_with_MUAC_115-<125mm': Parameter(
            Types.REAL, 'proportion of moderate weight-for-height Z-score with MUAC between 115mm and 125mm'),
        'proportion_mam_with_MUAC_115-<125mm_and_normal_whz': Parameter(
            Types.REAL, 'proportion of mam cases with MUAC between 115mm and 125mm and normal/mild WHZ'),
        'proportion_mam_with_MUAC_115-<125mm_and_-3<=WHZ<-2': Parameter(
            Types.REAL, 'proportion of mam cases with both MUAC between 115mm and 125mm and moderate wasting'),
        'proportion_mam_with_-3<=WHZ<-2_and_normal_MUAC': Parameter(
            Types.REAL, 'proportion of mam cases with moderate wasting and normal MUAC'),

        # bilateral oedema
        'prevalence_nutritional_oedema': Parameter(
            Types.REAL, 'prevalence of nutritional oedema in children under 5 in Malawi'),
        'proportion_oedema_with_WHZ<-2': Parameter(
            Types.REAL, 'proportion of oedematous malnutrition with concurrent wasting'),
        # death CFR, risk factors
        'base_death_rate_untreated_SAM': Parameter(
            Types.REAL, 'baseline death rate of untreated SAM'),
        'rr_SAM_death_with_complications': Parameter(
            Types.REAL, 'relative rate of death for complicated SAM'),
        'rr_SAM_death_WHZ<-3_only': Parameter(
            Types.REAL, 'relative risk of death from SAM if indices of WHZ<-3, compared to MUAC<115mm'),
        'rr_SAM_death_both_WHZ<-3_&_MUAC<115mm': Parameter(
            Types.REAL, 'relative risk of death from SAM if both indices WHZ<-3 & MUAC<115mm are present, '
                        'compared to MUAC<115mm alone'),
        'rr_SAM_death_kwashiorkor_only': Parameter(
            Types.REAL, 'relative risk of death from SAM if bilateral oedema present (kwashiorkor), '
                        'compared to MUAC<115mm alone'),
        'rr_SAM_death_kwashiorkor_MUAC<115mm_only': Parameter(
            Types.REAL, 'relative risk of death from SAM if bilateral oedema present and MUAC<115mm, '
                        'compared to MUAC<115mm alone'),
        'rr_SAM_death_kwashiorkor_WHZ<-3_only': Parameter(
            Types.REAL, 'relative risk of death from SAM if bilateral oedema present and WHZ<-3, '
                        'compared to MUAC<115mm alone'),
        'rr_SAM_death_kwashiorkor_both_WHZ<-3_&_MUAC<115mm': Parameter(
            Types.REAL, 'relative risk of death from SAM if bilateral oedema present, WHZ<-3 and MUAC<115mm, '
                        'compared to MUAC<115mm alone'),
        # treatment parameters
        'coverage_supplementary_feeding_program': Parameter(
            Types.REAL, 'coverage of supplementary feeding program for MAM in health centres'),
        'coverage_outpatient_therapeutic_care': Parameter(
            Types.REAL, 'coverage of outpatient therapeutic care for SAM in health centres'),
        'coverage_inpatient_care': Parameter(
            Types.REAL, 'coverage of inpatient care for complicated SAM in hospitals'),
    }

    PROPERTIES = {
        # Properties related to wasting
        'un_ever_wasted': Property(Types.BOOL, 'had wasting before WHZ <-2'),
        'un_WHZ_category': Property(Types.CATEGORICAL, 'weight-for-height z-score group',
                                    categories=['WHZ<-3', '-3<=WHZ<-2', 'WHZ>=-2']),
        'un_last_wasting_date_of_onset': Property(Types.DATE, 'date of onset of latest wasting episode'),

        # Properties related to clinical acute malnutrition
        'un_clinical_acute_malnutrition': Property(Types.CATEGORICAL, 'clinical acute malnutrition state based on WHZ',
                                                   categories=['MAM', 'SAM'] + ['well']),
        'un_am_bilateral_oedema': Property(Types.BOOL, 'bilateral oedema present in wasting'),
        'un_am_MUAC_category': Property(Types.CATEGORICAL, 'MUAC measurement categories',
                                        categories=['<115mm', '115-<125mm', '>=125mm']),
        'un_sam_with_complications': Property(Types.BOOL, 'medical complications in SAM'),
        'un_sam_death_date': Property(Types.DATE, 'death date from severe acute malnutrition'),
        'un_am_recovery_date': Property(Types.DATE, 'recovery date from acute malnutrition'),
        'un_am_discharge_date': Property(Types.DATE, 'discharge date from treatment of MAM/ SAM'),
        'un_acute_malnutrition_tx_start_date': Property(Types.DATE, 'intervention start date'),
        'un_am_treatment_type': Property(Types.CATEGORICAL, 'treatment types for acute malnutrition',
                                         categories=['standard_RUTF', 'soy_RUSF', 'CSB++', 'inpatient_care'] +
                                                    ['none', 'not_applicable']),
    }

    wasting_states = ['WHZ<-3', '-3<=WHZ<-2', 'WHZ>=-2']

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # Store the symptoms that this module will use:
        self.symptoms = {
            'palmar_pallor',
            'weight_loss',
            'poor_appetite',
            'lethargic',
            # 'dehydration'
        }

        # dict to hold counters for the number of episodes by wasting-type and age-group
        blank_counter = dict(zip(self.wasting_states, [list() for _ in self.wasting_states]))
        self.wasting_incident_case_tracker_blank = {
            _agrp: blank_counter for _agrp in ['0y', '1y', '2y', '3y', '4y', '5+y']}

        self.wasting_incident_case_tracker = dict(self.wasting_incident_case_tracker_blank)

        zeros_counter = dict(zip(self.wasting_states, [0] * len(self.wasting_states)))

        self.wasting_incident_case_tracker_zeros = {
            _agrp: zeros_counter for _agrp in ['0y', '1y', '2y', '3y', '4y', '5+y']}

        # dict to hold the DALY weights
        self.daly_wts = dict()

        # --------------------- linear models of the natural history --------------------- #

        # set the linear model equations for prevalence and incidence
        self.prevalence_equations_by_age = dict()
        self.wasting_incidence_equation = dict()

        # set the linear model for progression to severe wasting
        self.severe_wasting_progression_equation = dict()

        # set the linear model for death from severe acute malnutrition
        self.sam_death_equation = dict()

        # --------------------- linear models following HSI interventions --------------------- #

        # set the linear models for MAM and SAM recovery by intervention
        self.acute_malnutrition_recovery_based_on_interventions = dict()

    def read_parameters(self, data_folder):
        """
        :param data_folder: path of a folder supplied to the Simulation containing data files.
              Typically, modules would read a particular file within here.
        :return:
        """
        # Update parameters from the resource dataframe
        # Read parameters from the resourcefile
        self.load_parameters_from_dataframe(
            pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Wasting.xlsx',
                          sheet_name='Parameter_values_AM'))

        # Declare symptoms that this module will cause and which are not included in the generic symptoms:
        generic_symptoms = self.sim.modules['SymptomManager'].generic_symptoms
        for symptom_name in self.symptoms:
            if symptom_name not in generic_symptoms:
                self.sim.modules['SymptomManager'].register_symptom(
                    Symptom(name=symptom_name)  # (give non-generic symptom 'average' healthcare seeking)
                )

    def initialise_population(self, population):
        """
        Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population:
        :return:
        """
        df = population.props
        p = self.parameters

        # Set initial properties
        df.loc[df.is_alive, 'un_ever_wasted'] = False
        df.loc[df.is_alive, 'un_WHZ_category'] = 'WHZ>=-2'  # not undernourished
        df.loc[df.is_alive, 'un_clinical_acute_malnutrition'] = 'well'
        df.loc[df.is_alive, 'un_last_wasting_date_of_onset'] = pd.NaT
        df.loc[df.is_alive, 'un_acute_malnutrition_tx_start_date'] = pd.NaT
        df.loc[df.is_alive, 'un_sam_death_date'] = pd.NaT
        df.loc[df.is_alive, 'un_am_bilateral_oedema'] = False
        df.loc[df.is_alive, 'un_am_MUAC_category'] = '>=125mm'
        df.loc[df.is_alive, 'un_am_treatment_type'] = 'not_applicable'

        # -----------------------------------------------------------------------------------------------------
        # # # # # allocate initial prevalence of wasting at the start of the simulation # # # # #

        def make_scaled_linear_model_wasting(agegp):
            """ Makes the unscaled linear model with intercept of baseline odds of wasting (WHZ <-2).
            Calculates the mean odds of wasting by age group and then creates a new linear model
            with adjusted intercept so odds in 1-year-olds matches the specified value in the model
            when averaged across the population
            """

            def get_odds_wasting(agegp):
                """
                This function will calculate the WHZ scores by categories and return the odds of wasting
                :param agegp: age grouped in months
                :return:
                """
                # generate random numbers from N(meean, sd)
                baseline_WHZ_prevalence_by_agegp = f'prev_WHZ_distribution_age_{agegp}'
                WHZ_normal_distribution = norm(loc=p[baseline_WHZ_prevalence_by_agegp][0],
                                               scale=p[baseline_WHZ_prevalence_by_agegp][1])

                # get all wasting: WHZ <-2
                probability_over_or_equal_minus2sd = WHZ_normal_distribution.sf(-2)
                probability_less_than_minus2sd = 1 - probability_over_or_equal_minus2sd

                # convert probability to odds
                base_odds_of_wasting = probability_less_than_minus2sd / (1 - probability_less_than_minus2sd)
                return base_odds_of_wasting

            def make_linear_model_wasting(intercept=get_odds_wasting(agegp=agegp)):
                return LinearModel(
                    LinearModelType.LOGISTIC,
                    intercept,  # baseline odds: get_odds_wasting(agegp=agegp)
                    Predictor('li_wealth').when(2, p['or_wasting_hhwealth_Q2'])
                    .when(3, p['or_wasting_hhwealth_Q3'])
                    .when(4, p['or_wasting_hhwealth_Q4'])
                    .when(5, p['or_wasting_hhwealth_Q5'])
                    .otherwise(1.0),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& (nb_late_preterm == False) & (nb_early_preterm == False)',
                                     p['or_wasting_SGA_and_term']),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                                     p['or_wasting_SGA_and_preterm']),
                    Predictor().when('(nb_size_for_gestational_age == "average_for_gestational_age") '
                                     '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                                     p['or_wasting_preterm_and_AGA'])
                )

            unscaled_lm = make_linear_model_wasting(intercept=get_odds_wasting(agegp=agegp))

            target_mean = get_odds_wasting(agegp='12_23mo')
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 1)]).mean()
            scaled_intercept = get_odds_wasting(agegp) * (target_mean / actual_mean) if \
                (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else get_odds_wasting(agegp)
            scaled_lm = make_linear_model_wasting(intercept=scaled_intercept)
            return scaled_lm

        # the linear model returns the probability that is implied by the model prob = odds / (1 + odds)
        for agegp in ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo']:
            self.prevalence_equations_by_age[agegp] = make_scaled_linear_model_wasting(agegp)

        # get the initial prevalence values for each age group using the lm equation (scaled)
        prevalence_of_wasting = pd.DataFrame(index=df.index[df.is_alive & (df.age_exact_years < 5)])

        prevalence_of_wasting['0_5mo'] = self.prevalence_equations_by_age['0_5mo'] \
            .predict(df.loc[df.is_alive & (df.age_exact_years < 0.5)])
        prevalence_of_wasting['6_11mo'] = self.prevalence_equations_by_age['6_11mo'] \
            .predict(df.loc[df.is_alive & (df.age_exact_years.between(0.5, 1, inclusive='left'))])
        prevalence_of_wasting['12_23mo'] = self.prevalence_equations_by_age['12_23mo'] \
            .predict(df.loc[df.is_alive & (df.age_exact_years.between(1, 2, inclusive='left'))])
        prevalence_of_wasting['24_35mo'] = self.prevalence_equations_by_age['24_35mo'] \
            .predict(df.loc[df.is_alive & (df.age_exact_years.between(2, 3, inclusive='left'))])
        prevalence_of_wasting['36_47mo'] = self.prevalence_equations_by_age['36_47mo'] \
            .predict(df.loc[df.is_alive & (df.age_exact_years.between(3, 4, inclusive='left'))])
        prevalence_of_wasting['48_59mo'] = self.prevalence_equations_by_age['48_59mo'] \
            .predict(df.loc[df.is_alive & (df.age_exact_years.between(4, 5, inclusive='left'))])

        # -----------------------------------------------------------------------------------------------------
        # # # # # # further categorize into moderate (-3<=WHZ<-2) or severe (WHZ<-3) wasting # # # # #
        def get_prob_severe_in_overall_wasting(agegp):
            """
            This function will calculate the WHZ scores by categories and return probability of severe wasting
            for those with wasting status
            :param agegp: age grouped in months
            :return:
            """
            # generate random numbers from N(meean, sd)
            baseline_WHZ_prevalence_by_agegp = f'prev_WHZ_distribution_age_{agegp}'
            WHZ_normal_distribution = norm(loc=p[baseline_WHZ_prevalence_by_agegp][0],
                                           scale=p[baseline_WHZ_prevalence_by_agegp][1])

            # get all wasting: WHZ <-2
            probability_over_or_equal_minus2sd = WHZ_normal_distribution.sf(-2)
            probability_less_than_minus2sd = 1 - probability_over_or_equal_minus2sd
            # get severe wasting zcores: WHZ <-3
            probability_over_or_equal_minus3sd = WHZ_normal_distribution.sf(-3)
            probability_less_than_minus3sd = 1 - probability_over_or_equal_minus3sd

            # make WHZ <-2 as the 100% and get the adjusted probability of severe wasting within overall wasting
            proportion_severe_in_overall_wasting = probability_less_than_minus3sd * probability_less_than_minus2sd

            # get the probability of severe wasting
            return proportion_severe_in_overall_wasting

        prev_wasting_idx = prevalence_of_wasting.index  # get index of prevalence of wasting dataframe
        # differentiate into severe wasting and moderate wasting, by age group
        for agegp in ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo']:
            wasted = self.rng.random_sample(len(prevalence_of_wasting[agegp])) < prevalence_of_wasting[agegp]
            for idx in prev_wasting_idx[wasted]:
                probability_of_severe = get_prob_severe_in_overall_wasting(agegp)
                wasted_category = self.rng.choice(['WHZ<-3', '-3<=WHZ<-2'],
                                                  p=[probability_of_severe, 1 - probability_of_severe])
                df.at[idx, 'un_WHZ_category'] = wasted_category
                df.at[idx, 'un_last_wasting_date_of_onset'] = self.sim.date
                df.at[idx, 'un_ever_wasted'] = True
                df.at[idx, 'un_am_treatment_type'] = 'none'  # start without treatment
                # update clinical symptoms for severe wasting
                self.wasting_clinical_symptoms(person_id=idx)

            df.loc[prev_wasting_idx[~wasted], 'un_WHZ_category'] = 'WHZ>=-2'

        # -----------------------------------------------------------------------------------------------------
        # # # # # # Give MUAC category, presence of oedema, and determine acute malnutrition state # # # # #
        self.population_poll_clinical_am(df)

        # -----------------------------------------------------------------------------------------------------
        # # # # # Treatment coverage and cure rates at initiation # # # # #

        # inpatient care
        sam_requiring_inpatient_care = df.index[df.is_alive & (df.age_exact_years < 5) & (
            df.un_clinical_acute_malnutrition == 'SAM') & df.un_sam_with_complications]
        recovered_complic_sam = self.rng.random_sample(len(sam_requiring_inpatient_care)) < (
            p['coverage_inpatient_care'] * p['recovery_rate_with_inpatient_care'])
        # schedule recovery, and reset properties
        for person in sam_requiring_inpatient_care[recovered_complic_sam]:
            self.sim.schedule_event(
                event=ClinicalAcuteMalnutritionRecoveryEvent(module=self, person_id=person),
                date=self.sim.date + DateOffset(days=self.rng.randint(0, 90)))  # in the next 3 months

        for person in sam_requiring_inpatient_care[~recovered_complic_sam]:
            self.sim.schedule_event(
                event=SevereAcuteMalnutritionDeathEvent(module=self, person_id=person),
                date=self.sim.date + DateOffset(days=self.rng.randint(0, 90)))  # in the next 3 months

        # outpatient care
        uncomplicated_sam = df.index[df.is_alive & (df.age_exact_years < 5) & (
            df.un_clinical_acute_malnutrition == 'SAM') & ~df.un_sam_with_complications]
        recovered_uncompl_sam = self.rng.random_sample(len(uncomplicated_sam)) > (
            p['coverage_outpatient_therapeutic_care'] * p['recovery_rate_with_standard_RUTF'])
        for person in uncomplicated_sam[recovered_uncompl_sam]:
            self.sim.schedule_event(
                event=ClinicalAcuteMalnutritionRecoveryEvent(module=self, person_id=person),
                date=self.sim.date + DateOffset(months=3))  # in the next 3 months
        for person in uncomplicated_sam[~recovered_uncompl_sam]:
            self.sim.schedule_event(
                event=SevereAcuteMalnutritionDeathEvent(module=self, person_id=person),
                date=self.sim.date + DateOffset(months=3))  # in the next 3 months

        # supplementary feeding for MAM
        children_with_mam = df.index[df.is_alive & (df.age_exact_years < 5) & (
            df.un_clinical_acute_malnutrition == 'MAM') & ~df.un_sam_with_complications]
        recovered_mam = self.rng.random_sample(len(children_with_mam)) < (
            p['coverage_supplementary_feeding_program'] * p['recovery_rate_with_CSB++'])
        for person in children_with_mam[recovered_mam]:
            self.sim.schedule_event(
                event=ClinicalAcuteMalnutritionRecoveryEvent(module=self, person_id=person),
                date=self.sim.date + DateOffset(days=self.rng.randint(0, 90)))  # in the next 3 months
        # we suggest progressing to severe wasting if not recovered
        for person in children_with_mam[~recovered_mam]:
            self.sim.schedule_event(
                event=ProgressionSevereWastingEvent(module=self, person_id=person),
                date=self.sim.date + DateOffset(months=3))  # in the next 3 months

    def initialise_simulation(self, sim):
        """Prepares for simulation:
        * Schedules the main polling event
        * Schedules the main logging event
        * Establishes the incidence linear models and other data structures using the parameters that have been read-in
        * Store the consumables that are required in each of the HSI
        """
        df = self.sim.population.props
        p = self.parameters

        # schedule wasting pool event
        sim.schedule_event(WastingPollingEvent(self), sim.date + DateOffset(months=3))

        # schedule acute malnutrition event
        sim.schedule_event(AcuteMalnutritionDeathPollingEvent(self), sim.date + DateOffset(months=3))

        # schedule wasting logging event
        sim.schedule_event(WastingLoggingEvent(self), sim.date + DateOffset(months=12))

        # Get DALY weights
        get_daly_weight = self.sim.modules['HealthBurden'].get_daly_weight
        if 'HealthBurden' in self.sim.modules.keys():
            # self.daly_wts['MAM_w/o_oedema'] = get_daly_weight(sequlae_code=460)  ## no value given
            self.daly_wts['MAM_with_oedema'] = get_daly_weight(sequlae_code=461)
            self.daly_wts['SAM_w/o_oedema'] = get_daly_weight(sequlae_code=462)
            self.daly_wts['SAM_with_oedema'] = get_daly_weight(sequlae_code=463)

        # --------------------------------------------------------------------------------------------
        # Make a linear model equation that govern the probability that a person becomes wasted WHZ<-2
        def make_scaled_lm_wasting_incidence():
            """
            Makes the unscaled linear model with default intercept of 1. Calculates the mean incidents rate for
            1-year-olds and then creates a new linear model with adjusted intercept so incidents in 1-year-olds
            matches the specified value in the model when averaged across the population
            """

            def make_lm_wasting_incidence(intercept=1.0):
                return LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept,
                    Predictor('age_exact_years').when('<0.5', p['base_inc_rate_wasting_by_agegp'][0])
                    .when('<1.0', p['base_inc_rate_wasting_by_agegp'][1])
                    .when('.between(1,1.9999)', p['base_inc_rate_wasting_by_agegp'][2])
                    .when('.between(2,2.9999)', p['base_inc_rate_wasting_by_agegp'][3])
                    .when('.between(3,3.9999)', p['base_inc_rate_wasting_by_agegp'][4])
                    .when('.between(4,4.9999)', p['base_inc_rate_wasting_by_agegp'][5])
                    .otherwise(0.0),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& (nb_late_preterm == False) & (nb_early_preterm == False)',
                                     p['rr_wasting_SGA_and_term']),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                                     p['rr_wasting_SGA_and_preterm']),
                    Predictor().when('(nb_size_for_gestational_age == "average_for_gestational_age") '
                                     '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                                     p['rr_wasting_preterm_and_AGA']),
                    Predictor('li_wealth').apply(lambda x: 1 if x == 1 else (x - 1) ** (p['rr_wasting_wealth_level'])),
                )

            unscaled_lm = make_lm_wasting_incidence()
            target_mean = p['base_inc_rate_wasting_by_agegp'][2]
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 1) &
                                                     (df.un_WHZ_category == 'WHZ>=-2')]).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean) \
                if (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else 1.0
            scaled_lm = make_lm_wasting_incidence(intercept=scaled_intercept)

            return scaled_lm

        self.wasting_incidence_equation = make_scaled_lm_wasting_incidence()

        # --------------------------------------------------------------------------------------------
        # Linear model for the probability of progression to severe wasting (age-dependent only)
        # (natural history only, no interventions)
        self.severe_wasting_progression_equation = \
            LinearModel(LinearModelType.MULTIPLICATIVE,
                        1.0,
                        Predictor('age_exact_years')
                        .when('<0.5', p['progression_severe_wasting_by_agegp'][0])
                        .when('.between(0.5,0.9999)', p['progression_severe_wasting_by_agegp'][1])
                        .when('.between(1,1.9999)', p['progression_severe_wasting_by_agegp'][2])
                        .when('.between(2,2.9999)', p['progression_severe_wasting_by_agegp'][3])
                        .when('.between(3,3.9999)', p['progression_severe_wasting_by_agegp'][4])
                        .when('.between(4,4.9999)', p['progression_severe_wasting_by_agegp'][5])
                        .otherwise(0.0),
                        )

        # --------------------------------------------------------------------------------------------
        # Linear model for the probability of recovery based on interventions
        self.acute_malnutrition_recovery_based_on_interventions.update({
            'MAM':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('un_am_treatment_type').when('soy_RUSF', p['recovery_rate_with_soy_RUSF'])
                            .when('CSB++', p['recovery_rate_with_CSB++'])
                            .otherwise(0.0),
                            ),
            'SAM':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('un_am_treatment_type')
                            .when('standard_RUTF', p['recovery_rate_with_standard_RUTF'])
                            .when('inpatient_care', p['recovery_rate_with_inpatient_care'])
                            .otherwise(0.0),
                            )
        })

        # --------------------------------------------------------------------------------------------
        # Make a linear model equation of death from severe acute malnutrition
        def make_scaled_lm_sam_death():
            """
            Makes the unscaled linear model with default intercept of 1. Calculates the mean death rate for
            1-year-olds and then creates a new linear model with adjusted intercept so incidents in 1-year-olds
            matches the specified value in the model when averaged across the population
            """

            def make_lm_sam_death(intercept=1.0):
                return LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept,
                    Predictor('un_am_treatment_type').when('none', p['base_death_rate_untreated_SAM']).otherwise(0.0),
                    Predictor('un_sam_with_complications').when(True,
                                                                p['rr_SAM_death_with_complications']),
                    Predictor().when('(un_am_bilateral_oedema == False) & '
                                     '(un_WHZ_category == "WHZ<-3") & (un_am_MUAC_category == "115-<125mm")',
                                     p['rr_SAM_death_WHZ<-3_only']),
                    Predictor().when('(un_am_bilateral_oedema == False) & '
                                     '(un_WHZ_category == "WHZ<-3") & (un_am_MUAC_category == "<115mm")',
                                     p['rr_SAM_death_both_WHZ<-3_&_MUAC<115mm']),
                    Predictor().when('(un_am_bilateral_oedema == True) & '
                                     '(un_WHZ_category == "-3<=WHZ<-2") & (un_am_MUAC_category == "<115mm")',
                                     p['rr_SAM_death_kwashiorkor_only']),
                    Predictor().when('(un_am_bilateral_oedema == True) & '
                                     '(un_WHZ_category == "WHZ<-3") & (un_am_MUAC_category == "115-<125mm")',
                                     p['rr_SAM_death_kwashiorkor_WHZ<-3_only']),
                    Predictor().when('(un_am_bilateral_oedema == True) & '
                                     '(un_WHZ_category == "WHZ<-3") & (un_am_MUAC_category == "<115mm")',
                                     p['rr_SAM_death_kwashiorkor_both_WHZ<-3_&_MUAC<115mm']),
                    Predictor().when('(un_am_bilateral_oedema == True) & '
                                     '(un_WHZ_category == "-3<=WHZ<-2") & (un_am_MUAC_category == "<115mm")',
                                     p['rr_SAM_death_kwashiorkor_MUAC<115mm_only']),
                )

            unscaled_lm = make_lm_sam_death()
            target_mean = p['base_death_rate_untreated_SAM']
            actual_mean = unscaled_lm.predict(
                df.loc[df.is_alive & ((df.age_exact_years > 0.5) & (df.age_exact_years < 5)) &
                       (df.un_clinical_acute_malnutrition == 'SAM')]).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean) \
                if (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else 1.0
            scaled_lm = make_lm_sam_death(intercept=scaled_intercept)
            return scaled_lm

        self.sam_death_equation = make_scaled_lm_sam_death()

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'un_WHZ_category'] = 'WHZ>=-2'

    def muac_cutoff_by_WHZ(self, idx, whz):
        """
        Proportion of MUAC<115mm in WHZ<-3 and -3<=WHZ<-2,
        and proportion of wasted children with oedematous malnutrition (Kwashiokor, marasmic-kwashiorkor)
        :param idx: index of children ages 6-59 months or person_id
        :param whz:
        :return:
        """
        df = self.sim.population.props
        p = self.parameters

        # ---------- MUAC <115mm in severe wasting (WHZ<-3) and moderate (-3<=WHZ<-2) ----------
        if whz == 'WHZ<-3':
            # apply probability of MUAC<115mm in severe wasting
            low_muac_in_severe_wasting = self.rng.random_sample(size=len(idx)) < p['proportion_WHZ<-3_with_MUAC<115mm']

            df.loc[idx[low_muac_in_severe_wasting], 'un_am_MUAC_category'] = '<115mm'
            # other severe wasting will have MUAC between 115-<125mm
            df.loc[idx[~low_muac_in_severe_wasting], 'un_am_MUAC_category'] = '115-<125mm'

        if whz == '-3<=WHZ<-2':
            # apply probability of MUAC<115mm in moderate wasting
            low_muac_in_moderate_wasting = self.rng.random_sample(size=len(idx)) < p[
                'proportion_-3<=WHZ<-2_with_MUAC<115mm']
            df.loc[idx[low_muac_in_moderate_wasting], 'un_am_MUAC_category'] = '<115mm'

            # apply probability of MUAC between 115-<125mm in moderate wasting
            moderate_low_muac_in_moderate_wasting = self.rng.random_sample(size=len(
                idx[~low_muac_in_moderate_wasting])) < p['proportion_-3<=WHZ<-2_with_MUAC_115-<125mm']
            df.loc[idx[~low_muac_in_moderate_wasting][moderate_low_muac_in_moderate_wasting], 'un_am_MUAC_category'] = \
                '115-<125mm'
            # other moderate wasting will have normal MUAC
            df.loc[idx[~low_muac_in_moderate_wasting][~moderate_low_muac_in_moderate_wasting], 'un_am_MUAC_category'] \
                = '>=125mm'

        if whz == 'WHZ>=-2':
            # Give MUAC distribution for WHZ>=-2 ('well' group) ---------
            muac_distribution_in_well_group = norm(loc=p['MUAC_distribution_WHZ>=-2'][0],
                                                   scale=p['MUAC_distribution_WHZ>=-2'][1])
            # get probability of MUAC <115mm
            probability_over_or_equal_115 = muac_distribution_in_well_group.sf(11.5)
            probability_over_or_equal_125 = muac_distribution_in_well_group.sf(12.5)

            prob_less_than_115 = 1 - probability_over_or_equal_115
            pro_between_115_125 = probability_over_or_equal_115 - probability_over_or_equal_125

            for id in idx:
                muac_cat = self.rng.choice(['<115mm', '115-<125mm', '>=125mm'],
                                           p=[prob_less_than_115, pro_between_115_125, probability_over_or_equal_125])
                df.at[id, 'un_am_MUAC_category'] = muac_cat

    def nutritional_oedema_present(self, idx):
        """
        This function applies the probability of bilateral oedema present in wasting and non-wasted cases
        :param idx: index of children under 5, or person_id
        :return:
        """
        df = self.sim.population.props
        p = self.parameters

        # Knowing the prevalence of nutritional oedema in under 5 population, apply the probability of oedema in WHZ<-2
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
        This fuction will determine the clinical acute malnutrition status (MAM, SAM) based on anthropometric indices
        and presence of bilateral oedema (Kwashiorkor);
        And help determine whether the individual will have medical complications, applicable to SAM cases only,
        requiring inpatient care.
        :param person_id: individual id
        :param pop_dataframe: population dataframe
        :return:
        """
        df = pop_dataframe
        p = self.parameters

        # check if person is not wasted
        if (
            (df.at[person_id, 'un_WHZ_category'] == 'WHZ>=-2') &
            (df.at[person_id, 'un_am_MUAC_category'] == '>=125mm') &
            (~df.at[person_id, 'un_am_bilateral_oedema'])
        ):
            df.at[person_id, 'un_clinical_acute_malnutrition'] = 'well'

        # severe acute malnutrition - MUAC<115mm and/or WHZ<-3 and/or bilateral oedema
        elif (
            (df.at[person_id, 'un_am_MUAC_category'] == '<115mm') |
            (df.at[person_id, 'un_WHZ_category'] == 'WHZ<-3') |
            (df.at[person_id, 'un_am_bilateral_oedema'])
        ):
            df.at[person_id, 'un_clinical_acute_malnutrition'] = 'SAM'

        else:
            df.at[person_id, 'un_clinical_acute_malnutrition'] = 'MAM'

        # Determine if SAM episode has complications
        if df.at[person_id, 'un_clinical_acute_malnutrition'] == 'SAM':
            if p['prob_complications_in_SAM'] > self.rng.random_sample():
                df.at[person_id, 'un_sam_with_complications'] = True
            else:
                df.at[person_id, 'un_sam_with_complications'] = False
        else:
            df.at[person_id, 'un_sam_with_complications'] = False

        assert not (df.at[person_id, 'un_clinical_acute_malnutrition'] == 'MAM') & \
                   (df.at[person_id, 'un_sam_with_complications'])

    def date_of_outcome_for_untreated_am(self, person_id, duration_am):
        """
        helper funtion to get the duration and the wasting episode and date of outcome (recovery, progression, or death)
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
            duration_sam = int(max(p['min_days_duration_of_wasting'], p['average_duration_of_untreated_MAM'] +
                                   p['average_duration_of_untreated_SAM']))
            # Allocate a date of outcome (progression, recovery or death)
            date_of_outcome = df.at[person_id, 'un_last_wasting_date_of_onset'] + DateOffset(days=duration_sam)
            return date_of_outcome

    def population_poll_clinical_am(self, population):
        """
        Update at the population level other anthropometric indices and clinical signs
        (MUAC, oedema, medical complications) that determine the clinical state of acute malnutrition
        This will include both wasted and non-wasted children with other signs of acute malnutrition
        :param population:
        :return:
        """
        df = population

        # give MUAC measurement category for all WHZ, including well nourished children -----
        for whz in ['WHZ<-3', '-3<=WHZ<-2', 'WHZ>=-2']:
            index_6_59mo_by_whz = df.index[df.is_alive & (df.age_exact_years.between(0.5, 5, inclusive='left'))
                                           & (df.un_WHZ_category == whz)]
            self.muac_cutoff_by_WHZ(idx=index_6_59mo_by_whz, whz=whz)

        # determine the presence of bilateral oedema / oedematous malnutrition -----
        index_under5 = df.index[df.is_alive & (df.age_exact_years < 5)]
        self.nutritional_oedema_present(idx=index_under5)

        # determine the clinical acute malnutrition state -----
        df = self.sim.population.props
        for person_id in index_under5:
            self.clinical_acute_malnutrition_state(person_id=person_id, pop_dataframe=df)

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        logger.debug(key='message',
                     data=f'This is Wasting, being alerted about a health system interaction for person'
                          f'{person_id} and treatment {treatment_id}')

    def report_daly_values(self):
        """
        This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        experienced by persons in the previous month. Only rows for alive-persons must be returned.
        The names of the series of columns is taken to be the label of the cause of this disability.
        It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        """
        df = self.sim.population.props

        total_daly_values = pd.Series(data=0.0, index=df.index[df.is_alive])
        total_daly_values.loc[df.is_alive & (df.un_clinical_acute_malnutrition == 'SAM') &
                              df.un_am_bilateral_oedema] = self.daly_wts['SAM_with_oedema']
        total_daly_values.loc[df.is_alive & (df.un_clinical_acute_malnutrition == 'SAM') &
                              (~df.un_am_bilateral_oedema)] = self.daly_wts['SAM_w/o_oedema']
        total_daly_values.loc[df.is_alive & (
            ((df.un_WHZ_category == '-3<=WHZ<-2') & (df.un_am_MUAC_category != "<115mm")) |
            ((df.un_WHZ_category != 'WHZ<-3') & (df.un_am_MUAC_category != "115-<125mm"))
        ) & df.un_am_bilateral_oedema] = self.daly_wts['MAM_with_oedema']
        # total_daly_values.loc[df.is_alive & (df.un_clinical_acute_malnutrition == 'MAM')] = \
        #     self.daly_wts['MAM_w/o_oedema']

        return total_daly_values

    def wasting_clinical_symptoms(self, person_id):
        """
        assign clinical symptoms to new acute malnutrition cases
        :param person_id:
        """
        df = self.sim.population.props
        if df.at[person_id, 'un_clinical_acute_malnutrition'] != 'SAM':
            return

        # currently symptoms list is applied to all (who are SAM)
        for symptom in self.symptoms:
            self.sim.modules["SymptomManager"].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove="+",
                disease_module=self,
                # duration_in_days=None,
            )

    def do_when_acute_malnutrition_assessment(self, person_id):
        """
        This is called by the generic HSI event when acute malnutrition is checked.
        :param person_id:
        :return:
        """
        df = self.sim.population.props
        p = self.parameters

        # get the clinical states
        clinical_am = df.at[person_id, 'un_clinical_acute_malnutrition']
        complications = df.at[person_id, 'un_sam_with_complications']

        # Interventions for MAM
        if clinical_am == 'MAM':
            # Check for coverage of supplementary feeding
            if p['coverage_supplementary_feeding_program'] > self.rng.random_sample():
                # schedule HSI for supplementary feeding program for MAM
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_Wasting_SupplementaryFeedingProgramme_MAM
                    (module=self,
                     person_id=person_id),
                    priority=0,
                    topen=self.sim.date
                )
            else:
                return
        # Interventions for uncomplicated SAM
        if clinical_am == 'SAM':
            if not complications:
                # Check for coverage of outpatient therapeutic care
                if p['coverage_outpatient_therapeutic_care'] > self.rng.random_sample():
                    # schedule HSI for supplementary feeding program for MAM
                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        hsi_event=HSI_Wasting_OutpatientTherapeuticProgramme_SAM
                        (module=self,
                         person_id=person_id),
                        priority=0,
                        topen=self.sim.date
                    )
                else:
                    return
            # Interventions for complicated SAM
            if complications:
                # Check for coverage of outpatient therapeutic care
                if p['coverage_inpatient_care'] > self.rng.random_sample():
                    # schedule HSI for supplementary feeding program for MAM
                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        hsi_event=HSI_Wasting_InpatientCareForComplicated_SAM
                        (module=self,
                         person_id=person_id),
                        priority=0,
                        topen=self.sim.date
                    )
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
        df.at[person_id, 'un_acute_malnutrition_tx_start_date'] = self.sim.date

        if intervention == 'SFP':
            mam_recovery = self.acute_malnutrition_recovery_based_on_interventions['MAM'].predict(
                df.loc[[person_id]]).values[0]
            if self.rng.random_sample() < mam_recovery:
                # schedule recovery date
                self.sim.schedule_event(
                    event=ClinicalAcuteMalnutritionRecoveryEvent(module=self, person_id=person_id),
                    date=df.at[person_id, 'un_acute_malnutrition_tx_start_date'] + DateOffset(weeks=3))
                # cancel progression date (in ProgressionEvent)
            else:
                # remained MAM
                return

        if intervention == 'OTC':
            sam_recovery = self.acute_malnutrition_recovery_based_on_interventions['SAM'].predict(
                df.loc[[person_id]]).values[0]
            if self.rng.random_sample() < sam_recovery:
                # schedule recovery date
                self.sim.schedule_event(
                    event=ClinicalAcuteMalnutritionRecoveryEvent(module=self, person_id=person_id),
                    date=df.at[person_id, 'un_acute_malnutrition_tx_start_date'] + DateOffset(weeks=3))
                # cancel death date
                df.at[person_id, 'un_sam_death_date'] = pd.NaT
            else:
                # remained MAM or death
                outcome = self.rng.choice(['remained_mam', 'death'],
                                          p=[0.5, 0.5])
                if outcome == 'remained_mam':
                    self.sim.schedule_event(
                        event=UpdateToMAM(module=self, person_id=person_id),
                        date=df.at[person_id, 'un_acute_malnutrition_tx_start_date'] + DateOffset(weeks=3))
                if outcome == 'death':
                    self.sim.schedule_event(
                        event=SevereAcuteMalnutritionDeathEvent(module=self, person_id=person_id),
                        date=df.at[person_id, 'un_acute_malnutrition_tx_start_date'] + DateOffset(weeks=3))

        if intervention == 'ITC':
            sam_recovery = self.acute_malnutrition_recovery_based_on_interventions['SAM'].predict(
                df.loc[[person_id]]).values[0]
            if self.rng.random_sample() < sam_recovery:
                # schedule recovery date
                self.sim.schedule_event(
                    event=ClinicalAcuteMalnutritionRecoveryEvent(module=self, person_id=person_id),
                    date=df.at[person_id, 'un_acute_malnutrition_tx_start_date'] + DateOffset(weeks=4))
                # cancel death date
                df.at[person_id, 'un_sam_death_date'] = pd.NaT
            else:
                # remained MAM or death
                outcome = self.rng.choice(['remained_mam', 'death'],
                                          p=[0.5, 0.5])
                if outcome == 'remained_mam':
                    self.sim.schedule_event(
                        event=UpdateToMAM(module=self, person_id=person_id),
                        date=df.at[person_id, 'un_acute_malnutrition_tx_start_date'] + DateOffset(weeks=4))
                if outcome == 'death':
                    self.sim.schedule_event(
                        event=SevereAcuteMalnutritionDeathEvent(module=self, person_id=person_id),
                        date=df.at[person_id, 'un_acute_malnutrition_tx_start_date'] + DateOffset(weeks=4))


class WastingPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that determines new cases of wasting (WHZ<-2) to the under-5 population,
    and schedules individual incident cases to represent onset.
    It determines those who will progress to severe wasting (WHZ<-3) and schedules the event to update on properties.
    These are events occurring without the input of interventions, these events reflect the natural history only.
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

        # # # # # # # # # # # # # # # # # # # # # INCIDENCE OF WASTING # # # # # # # # # # # # # # # # # # # # #
        # Determine who will be onset with wasting among those who are not currently wasted -------------
        inc_wasting = df.loc[df.is_alive & (df.age_exact_years < 5) & (df.un_WHZ_category == 'WHZ>=-2')]
        incidence_of_wasting = self.module.wasting_incidence_equation.predict(inc_wasting, rng)

        wasting_idx = inc_wasting.index
        # update the properties for wasted children
        df.loc[wasting_idx[incidence_of_wasting], 'un_ever_wasted'] = True
        df.loc[wasting_idx[incidence_of_wasting], 'un_last_wasting_date_of_onset'] = self.sim.date
        df.loc[wasting_idx[incidence_of_wasting], 'un_WHZ_category'] = '-3<=WHZ<-2'  # start as moderate wasting
        df.loc[wasting_idx[incidence_of_wasting], 'un_am_treatment_type'] = 'none'  # start without treatment

        # -------------------------------------------------------------------------------------------
        # Add this incident case to the tracker
        for person in wasting_idx[incidence_of_wasting]:
            wasting_severity = df.at[person, 'un_WHZ_category']
            age_group = WastingPollingEvent.AGE_GROUPS.get(df.loc[person].age_years, '5+y')
            if wasting_severity != 'WHZ>=-2':
                self.module.wasting_incident_case_tracker[age_group][wasting_severity].append(self.sim.date)
        # -------------------------------------------------------------------------------------------

        # # # # # # # # # # # # # # # # # # # # # PROGRESS TO SEVERE WASTING # # # # # # # # # # # # # # # # # # # # #

        # Determine those that will progress to severe wasting ( WHZ<-3) and schedule progression event ---------
        progression_sev_wasting = df.loc[df.is_alive & (df.age_exact_years < 5) & (df.un_WHZ_category == '-3<=WHZ<-2')]
        progression_severe_wasting = self.module.severe_wasting_progression_equation.predict(progression_sev_wasting,
                                                                                             rng)
        progression_sev_wasting_idx = progression_sev_wasting.index
        # determine those individuals who will progress to severe wasting and time of progression
        for person in progression_sev_wasting_idx[progression_severe_wasting]:
            outcome_date = self.module.date_of_outcome_for_untreated_am(person_id=person, duration_am='MAM')
            # schedule severe wasting WHZ<-3 onset
            if outcome_date <= self.sim.date:
                # schedule severe wasting WHZ<-3 onset today
                self.sim.schedule_event(
                    event=ProgressionSevereWastingEvent(module=self.module, person_id=person),
                    date=self.sim.date)
            else:
                # schedule severe wasting WHZ<-3 onset according to duration
                self.sim.schedule_event(
                    event=ProgressionSevereWastingEvent(module=self.module, person_id=person),
                    date=outcome_date)

        # # # # # # # # # # # # # # # # # MODERATE WASTING NATURAL RECOVERY # # # # # # # # # # # # # # # # #

        # moderate wasting not progressed to severe, schedule recovery
        for person in progression_sev_wasting_idx[~progression_severe_wasting]:
            outcome_date = self.module.date_of_outcome_for_untreated_am(person_id=person, duration_am='MAM')
            if outcome_date <= self.sim.date:
                # schedule recovery for today
                self.sim.schedule_event(
                    event=WastingNaturalRecoveryEvent(module=self.module, person_id=person),
                    date=self.sim.date)
            else:
                # schedule recovery according to duration
                self.sim.schedule_event(
                    event=WastingNaturalRecoveryEvent(module=self.module, person_id=person),
                    date=outcome_date)

        # ------------------------------------------------------------------------------------------
        # # # # # # # # # UPDATE PROPERTIES RELATED TO CLINICAL ACUTE MALNUTRITION # # # # # # # # #
        # ------------------------------------------------------------------------------------------
        # This applies to all children under 5

        # give MUAC measurement category for all WHZ, including well nourished children -----
        # determine the presence of bilateral oedema / oedematous malnutrition -----
        # determine the clinical state of acute malnutrition, and check complications if SAM
        self.module.population_poll_clinical_am(df)

        # then, update clinical symptoms for those with severe acute malnutrition
        children_with_sam = df.loc[df.is_alive & (df.age_exact_years < 5) &
                                   (df.un_clinical_acute_malnutrition == 'SAM')]
        for person in children_with_sam.index:
            self.module.wasting_clinical_symptoms(person_id=person)


class ProgressionSevereWastingEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of severe wasting (WHZ <-3).
     * Refreshes all the properties so that they pertain to this current episode of wasting
     * Imposes the symptoms
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module

        # before progression to severe wasting, check those who started supplementary feeding programme before today
        if df.at[person_id, 'un_last_wasting_date_of_onset'] < df.at[person_id, 'un_acute_malnutrition_tx_start_date'] \
                < self.sim.date:
            return

        # continue with progression to severe if not treated/recovered
        else:
            # update properties
            df.at[person_id, 'un_WHZ_category'] = 'WHZ<-3'

            # Give MUAC measurement category for WHZ<-3
            if df.at[person_id, 'age_exact_years'] > 0.5:
                m.muac_cutoff_by_WHZ(idx=df.loc[[person_id]].index, whz='WHZ<-3')

            # update the clinical state of acute malnutrition, and check complications if SAM
            m.clinical_acute_malnutrition_state(person_id=person_id, pop_dataframe=df)

            # update clinical symptoms for severe wasting
            m.wasting_clinical_symptoms(person_id=person_id)

            # -------------------------------------------------------------------------------------------
            # Add this incident case to the tracker
            wasting_severity = df.at[person_id, 'un_WHZ_category']
            age_group = WastingPollingEvent.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
            if wasting_severity != 'WHZ>=-2':
                m.wasting_incident_case_tracker[age_group][wasting_severity].append(self.sim.date)


class AcuteMalnutritionDeathPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that determines death due to acute malnutrition to the under-5 population,
    and schedules individual death events.
    It also determines those who will improve nutritional state.
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

        # # # # # # # # # # # # # # # # # # # # # DEATH # # # # # # # # # # # # # # # # # # # # #

        # Determine those that will die -----------------------------------------------
        _to_die = df.loc[df.is_alive & (df.age_exact_years < 5) & (df.un_clinical_acute_malnutrition != 'well')]
        will_die = self.module.sam_death_equation.predict(_to_die, rng)
        _to_die_idx = _to_die.index
        # schedule death date
        for person in _to_die_idx[will_die]:
            death_date = self.module.date_of_outcome_for_untreated_am(person_id=person, duration_am='SAM')
            if death_date <= self.sim.date:
                # schedule death for today
                self.sim.schedule_event(
                    event=SevereAcuteMalnutritionDeathEvent(module=self.module, person_id=person),
                    date=self.sim.date)
                # df.at[person, 'un_sam_death_date'] = self.sim.date
            else:
                # schedule death according to duration
                self.sim.schedule_event(
                    event=SevereAcuteMalnutritionDeathEvent(module=self.module, person_id=person),
                    date=death_date)
                # df.at[person, 'un_sam_death_date'] = death_date

        # # # # # # # # # # # # # # # # # # # # # IMPROVEMENT FROM SAM TO MAM # # # # # # # # # # # # # # # # # # # # #
        # SAM = severe wasting (WHZ<-3) and/or MUAC <115mm, or nutritional oedema
        # Those not scheduled to die, will have improved WHZ status by 1sd (-3<=WHZ<-2)

        # schedule improvement to MAM for those not scheduled to die
        for person in will_die[~will_die].index:
            outcome_date = self.module.date_of_outcome_for_untreated_am(person_id=person, duration_am='SAM')
            if outcome_date <= self.sim.date:
                # schedule improvement to moderate wasting date for today
                self.sim.schedule_event(
                    event=UpdateToMAM(module=self.module, person_id=person),
                    date=self.sim.date)
            if outcome_date > self.sim.date:
                # schedule improvement to moderate wasting date according to duration
                self.sim.schedule_event(
                    event=UpdateToMAM(module=self.module, person_id=person),
                    date=outcome_date)


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

        if df.at[person_id, 'un_acute_malnutrition_tx_start_date'] < self.sim.date:
            return

        # Check if this person should still die from SAM:
        if pd.isnull(df.at[person_id, 'un_am_recovery_date']) &\
                (df.at[person_id, 'un_clinical_acute_malnutrition'] == 'SAM'):
            # person_ids = df.at[person_id, 'un_am_recovery_date'])]
            # Cause the death to happen immediately
            df.at[person_id, 'un_sam_death_date'] = self.sim.date
            self.sim.modules['Demography'].do_death(
                individual_id=person_id,
                cause='SAM',
                originating_module=self.module
            )


class WastingNaturalRecoveryEvent(Event, IndividualScopeEventMixin):
    """
    This event sets wasting properties back to normal state, based on home care/ improvement without interventions,
    low-moderate MUAC categories oedema may or may not be present
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

        # Note assumption: prob of oedema remained the same as applied in wasting onset

        # update the clinical acute malnutrition state
        m.clinical_acute_malnutrition_state(person_id=person_id, pop_dataframe=df)

        if df.at[person_id, 'un_clinical_acute_malnutrition'] == 'well':
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
        df.at[person_id, 'un_acute_malnutrition_tx_start_date'] = pd.NaT
        df.at[person_id, 'un_am_treatment_type'] = 'not_applicable'

        # this will clear all wasting symptoms
        self.sim.modules["SymptomManager"].clear_symptoms(
            person_id=person_id, disease_module=self.module
        )


class UpdateToMAM(Event, IndividualScopeEventMixin):
    """
    This event updates the properties for those cases that remained/ improved from SAM to MAM following treatment
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

        # For cases with normal WHZ and other acute malnutrition signs: oedema, or low muac - do not change the WHZ
        if df.at[person_id, 'un_WHZ_category'] == 'WHZ>=-2':
            df.at[person_id, 'un_am_MUAC_category'] = '115-<125mm'  # mam by muac only

        else:
            # using the probability of mam classification by anthropometric indices
            mam_classification = rng.choice(['mam_by_muac_only', 'mam_by_muac_and_whz', 'mam_by_whz_only'],
                                            p=[p['proportion_mam_with_MUAC_115-<125mm_and_normal_whz'],
                                               p['proportion_mam_with_MUAC_115-<125mm_and_-3<=WHZ<-2'],
                                               p['proportion_mam_with_-3<=WHZ<-2_and_normal_MUAC']])

            if mam_classification == 'mam_by_muac_only':
                df.at[person_id, 'un_WHZ_category'] = 'WHZ>=-2'
                df.at[person_id, 'un_am_MUAC_category'] = '115-<125mm'

            if mam_classification == 'mam_by_muac_and_whz':
                df.at[person_id, 'un_WHZ_category'] = '-3<=WHZ<-2'
                df.at[person_id, 'un_am_MUAC_category'] = '115-<125mm'

            if mam_classification == 'mam_by_whz_only':
                df.at[person_id, 'un_WHZ_category'] = '-3<=WHZ<-2'
                df.at[person_id, 'un_am_MUAC_category'] = '>=125mm'

        # Update all other properties equally
        df.at[person_id, 'un_clinical_acute_malnutrition'] = 'MAM'
        df.at[person_id, 'un_am_bilateral_oedema'] = False
        df.at[person_id, 'un_sam_with_complications'] = False
        df.at[person_id, 'un_acute_malnutrition_tx_start_date'] = pd.NaT
        df.at[person_id, 'un_am_recovery_date'] = pd.NaT
        df.at[person_id, 'un_am_discharge_date'] = pd.NaT
        df.at[person_id, 'un_am_treatment_type'] = 'not_applicable'  # will start the process again

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

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Supplementary_feeding_programme_for_MAM'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        # Stop the person from dying of acute malnutrition (if they were going to die)
        if not df.at[person_id, 'is_alive']:
            return

        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~
        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['item_and_package_code_lookups']
        # whole package of interventions
        # pkg_code_mam = pd.unique(
        #     consumables.loc[consumables['Intervention_Pkg'] == 'Management of moderate acute malnutrition (children)',
        #                     'Intervention_Pkg_Code'])[0]  # This package includes only CSB(or supercereal or CSB++)
        # individual items
        item_code1 = pd.unique(
            consumables.loc[consumables['Items'] == 'Corn Soya Blend (or Supercereal - CSB++)', 'Item_Code'])[0]

        # consumables_needed = {'Intervention_Package_Code': {pkg_code_mam: 1}, 'Item_Code': {item_code1: 1}}

        # check availability of consumables
        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=self, cons_req_as_footprint=consumables_needed)
        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if self.get_consumables([item_code1]):
            logger.debug(key='debug', data='consumables are available')
            # Log that the treatment is provided:
            df.at[person_id, 'un_acute_malnutrition_tx_start_date'] = self.sim.date
            df.at[person_id, 'un_am_discharge_date'] = self.sim.date + DateOffset(weeks=3)
            df.at[person_id, 'un_am_treatment_type'] = 'CSB++'
            self.module.do_when_am_treatment(person_id, intervention='SFP')
        else:
            logger.debug(key='debug', data="PkgCode1 is not available, so can't use it.")

        # --------------------------------------------------------------------------------------------------
        # # check to see if all consumables returned (for demonstration purposes):
        # all_available = (outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_mam]) and \
        #                 (outcome_of_request_for_consumables['Item_Code'][item_code1])
        # # use helper function instead (for demonstration purposes)
        # all_available_using_helper_function = self.get_all_consumables(
        #     item_codes=[item_code1],
        #     pkg_codes=[pkg_code_mam]
        # )
        # # Demonstrate equivalence
        # assert all_available == all_available_using_helper_function

    def did_not_run(self):
        logger.debug("supplementary_feeding_programme_for_MAM: did not run")
        pass


class HSI_Wasting_OutpatientTherapeuticProgramme_SAM(HSI_Event, IndividualScopeEventMixin):
    """
    This is the outpatient management of SAM without any medical complications
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Wasting)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint['U5Malnutr'] = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Outpatient_therapeutic_programme_for_SAM'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        # Stop the person from dying of acute malnutrition (if they were going to die)
        if not df.at[person_id, 'is_alive']:
            return

        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~
        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['item_and_package_code_lookups']
        # whole package of interventions
        # pkg_code_sam = pd.unique(
        #     consumables.loc[consumables['Intervention_Pkg'] == 'Management of severe malnutrition (children)',
        #                     'Intervention_Pkg_Code'])[0]
        # individual items
        item_code1 = pd.unique(
            consumables.loc[consumables['Items'] == 'SAM theraputic foods', 'Item_Code'])[0]
        item_code2 = pd.unique(
            consumables.loc[consumables['Items'] == 'SAM medicines', 'Item_Code'])[0]

        # consumables_needed = {'Intervention_Package_Code': {pkg_code_sam: 1}, 'Item_Code': {item_code1: 1,
        #                                                                   item_code2: 1}}

        # check availability of consumables
        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=self, cons_req_as_footprint=consumables_needed)
        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if self.get_consumables(item_code1) and self.get_consumables(item_code2):
            logger.debug(key='debug', data='consumables are available.')
            # Log that the treatment is provided:
            df.at[person_id, 'un_acute_malnutrition_tx_start_date'] = self.sim.date
            df.at[person_id, 'un_am_discharge_date'] = self.sim.date + DateOffset(weeks=3)
            df.at[person_id, 'un_am_treatment_type'] = 'standard_RUTF'
            self.module.do_when_am_treatment(person_id, intervention='OTC')
        else:
            logger.debug(key='debug', data="consumables not available, so can't use it.")
        # --------------------------------------------------------------------------------------------------
        # # check to see if all consumables returned (for demonstration purposes):
        # all_available = (outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sam]) and \
        #                 (outcome_of_request_for_consumables['Item_Code'][item_code1][item_code2])
        #
        # # use helper function instead (for demonstration purposes)
        # all_available_using_helper_function = self.get_all_consumables(
        #     item_codes=[item_code1, item_code2],
        #     pkg_codes=[pkg_code_sam]
        # )
        # # Demonstrate equivalence
        # assert all_available == all_available_using_helper_function

    def did_not_run(self):
        logger.debug("HSI_outpatient_therapeutic_programme_for_SAM: did not run")
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
        self.TREATMENT_ID = 'Inpatient_care_for_complicated_SAM'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '2'
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 7})

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        # Stop the person from dying of acute malnutrition (if they were going to die)
        if not df.at[person_id, 'is_alive']:
            return

        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~
        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['item_and_package_code_lookups']
        # whole package of interventions
        # pkg_code_sam = pd.unique(
        #     consumables.loc[consumables['Intervention_Pkg'] == 'Management of severe malnutrition (children)',
        #                     'Intervention_Pkg_Code'])[0]
        # individual items
        item_code1 = pd.unique(
            consumables.loc[consumables['Items'] == 'SAM theraputic foods', 'Item_Code'])[0]
        item_code2 = pd.unique(
            consumables.loc[consumables['Items'] == 'SAM medicines', 'Item_Code'])[0]

        # pkg_codes = self.sim.modules['HealthSystem'].get_item_codes_from_package_name
        # pkg_codes_num = pkg_codes('Management of severe malnutrition (children)')
        #
        # consumables_needed = {'Intervention_Package_Code': {pkg_code_sam: 1}, 'Item_Code': {item_code1: 1,
        #                                                                                     item_code2: 1}}

        # # check availability of consumables
        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=self, cons_req_as_footprint=consumables_needed)
        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if self.get_consumables(item_code1) and self.get_consumables(item_code2):
            logger.debug(key='debug', data='consumables available, so use it.')
            # Log that the treatment is provided:
            df.at[person_id, 'un_acute_malnutrition_tx_start_date'] = self.sim.date
            df.at[person_id, 'un_am_discharge_date'] = self.sim.date + DateOffset(weeks=4)
            df.at[person_id, 'un_am_treatment_type'] = 'inpatient_care'
            self.module.do_when_am_treatment(person_id, intervention='ITC')
        else:
            logger.debug(key='debug', data="consumables not available, so can't use it.")
        # --------------------------------------------------------------------------------------------------
        # # check to see if all consumables returned (for demonstration purposes):
        # all_available = (outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sam]) and \
        #                 (outcome_of_request_for_consumables['Item_Code'][item_code1][item_code2])
        # # use helper function instead (for demonstration purposes)
        # all_available_using_helper_function = self.get_all_consumables(
        #     item_codes=[item_code1, item_code2],
        #     pkg_codes=[pkg_code_sam]
        # )
        # # Demonstrate equivalence
        # assert all_available == all_available_using_helper_function

    def did_not_run(self):
        logger.debug("HSI_inpatient_care_for_complicated_SAM: did not run")
        pass


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
        counts_am = self.module.wasting_incident_case_tracker_zeros

        for age_grp in self.module.wasting_incident_case_tracker.keys():
            for state in self.module.wasting_states:
                list_of_times = self.module.wasting_incident_case_tracker[age_grp][state]
                counts_am[age_grp][state] = len(list_of_times)
                for t in list_of_times:
                    assert self.date_last_run <= t <= self.sim.date

        logger.info(key='wasting_incidence_count', data=counts_am)

        # Reset the counters and the date_last_run
        self.module.wasting_incident_case_tracker = self.module.wasting_incident_case_tracker_blank

        self.date_last_run = self.sim.date

        # Wasting totals (prevalence at logging time)
        currently_wasted_age_0_5mo = (df.is_alive & (df.age_exact_years < 0.5) &
                                      (df.un_WHZ_category != 'WHZ>=-2')).sum()
        currently_wasted_age_6_11mo = (df.is_alive & ((df.age_exact_years >= 0.5) & (
            df.age_exact_years < 1)) & (df.un_WHZ_category != 'WHZ>=-2')).sum()
        currently_wasted_age_12_23mo = (df.is_alive & ((df.age_exact_years >= 1) & (
            df.age_exact_years < 2)) & (df.un_WHZ_category != 'WHZ>=-2')).sum()
        currently_wasted_age_24_35mo = (df.is_alive & ((df.age_exact_years >= 2) & (
            df.age_exact_years < 3)) & (df.un_WHZ_category != 'WHZ>=-2')).sum()
        currently_wasted_age_36_47mo = (df.is_alive & ((df.age_exact_years >= 3) & (
            df.age_exact_years < 4)) & (df.un_WHZ_category != 'WHZ>=-2')).sum()
        currently_wasted_age_48_59mo = (df.is_alive & ((df.age_exact_years >= 4) & (
            df.age_exact_years < 5)) & (df.un_WHZ_category != 'WHZ>=-2')).sum()

        currently_wasted = {'0_5mo': currently_wasted_age_0_5mo,
                            '6_11mo': currently_wasted_age_6_11mo,
                            '12_23mo': currently_wasted_age_12_23mo,
                            '24_35mo': currently_wasted_age_24_35mo,
                            '36_47mo': currently_wasted_age_36_47mo,
                            '48_59mo': currently_wasted_age_48_59mo}

        logger.info(key='wasting_prevalence_count', data=currently_wasted)
