"""
Childhood stunting module
Documentation: '04 - Methods Repository/Undernutrition module - Description.docx'

Overview
=======
This module applies the prevalence of stunting at the population-level, and schedules new incidences of stunting

"""
import copy
from pathlib import Path
from scipy.stats import norm

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------


class Stunting(Module):
    """
    This module applies the prevalence of stunting at the population-level,
    based on the Malawi DHS Survey 2015-2016.
    The definitions:
    - moderate stunting: height-for-age Z-score (HAZ) <-2 SD from the reference mean
    - severe stunting: height-for-age Z-score (HAZ) <-3 SD from the reference mean

    """

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    stunting_states = ['moderate_stunting', 'severe_stunting']

    PARAMETERS = {
        # prevalence of stunting by age group
        'prev_HAZ_distribution_age_0_5mo': Parameter(
            Types.LIST, 'distribution of HAZ among less than 6 months of age in 2015'),
        'prev_HAZ_distribution_age_6_11mo': Parameter(
            Types.LIST, 'distribution of HAZ among 6 months and 1 year of age in 2015'),
        'prev_HAZ_distribution_age_12_23mo': Parameter(
            Types.LIST, 'distribution of HAZ among 1 year olds in 2015'),
        'prev_HAZ_distribution_age_24_35mo': Parameter(
            Types.LIST, 'distribution of HAZ among 2 year olds in 2015'),
        'prev_HAZ_distribution_age_36_47mo': Parameter(
            Types.LIST, 'distribution of HAZ among 3 year olds in 2015'),
        'prev_HAZ_distribution_age_48_59mo': Parameter(
            Types.LIST, 'distribution of HAZ among 4 year olds  in 2015'),
        # effect of risk factors on stunting prevalence
        'or_stunting_male': Parameter(
            Types.REAL, 'odds ratio of stunting if male gender'),
        'or_stunting_no_recent_diarrhoea': Parameter(
            Types.REAL, 'odds ratio of stunting if no recent diarrhoea in past 2 weeks, compared to recent episode'),
        'or_stunting_single_birth': Parameter(
            Types.REAL, 'odds ratio of stunting if single birth, ref group multiple birth (twins)'),
        'or_stunting_mother_no_education': Parameter(
            Types.REAL, 'odds ratio of stunting if mother has no formal education, ref group secondary education'),
        'or_stunting_mother_primary_education': Parameter(
            Types.REAL, 'odds ratio of stunting if mother has primary education, ref group secondary education'),
        'or_stunting_motherBMI_underweight': Parameter(
            Types.REAL, 'odds ratio of stunting if mother has low BMI, ref group high BMI (overweight)'),
        'or_stunting_motherBMI_normal': Parameter(
            Types.REAL, 'odds ratio of stunting if mother has normal BMI, ref group high BMI (overweight)'),
        'or_stunting_hhwealth_Q5': Parameter(
            Types.REAL, 'odds ratio of stunting if household wealth is poorest Q5, ref group Q1'),
        'or_stunting_hhwealth_Q4': Parameter(
            Types.REAL, 'odds ratio of stunting if household wealth is poorer Q4, ref group Q1'),
        'or_stunting_hhwealth_Q3': Parameter(
            Types.REAL, 'odds ratio of stunting if household wealth is middle Q3, ref group Q1'),
        'or_stunting_hhwealth_Q2': Parameter(
            Types.REAL, 'odds ratio of stunting if household wealth is richer Q2, ref group Q1'),
        # incidence parameters
        'base_inc_rate_stunting_by_agegp': Parameter(
            Types.LIST, 'List with baseline incidence of stunting by age group'),
        'rr_stunting_preterm_and_AGA': Parameter(
            Types.REAL, 'relative risk of stunting if born preterm and adequate for gestational age'),
        'rr_stunting_SGA_and_term': Parameter(
            Types.REAL, 'relative risk of stunting if born term and small for gestational age'),
        'rr_stunting_SGA_and_preterm': Parameter(
            Types.REAL, 'relative risk of stunting if born preterm and small for gestational age'),
        'rr_stunting_untreated_HIV': Parameter(
            Types.REAL, 'relative risk of stunting for untreated HIV+'),
        'rr_stunting_wealth_level': Parameter(
            Types.REAL, 'relative risk of stunting by increase in wealth level'),
        'rr_stunting_no_exclusive_breastfeeding': Parameter(
            Types.REAL, 'relative risk of stunting for not exclusively breastfed babies < 6 months'),
        'rr_stunting_no_continued_breastfeeding': Parameter(
            Types.REAL, 'relative risk of stunting for not continued breasfed infants 6-24 months'),
        'rr_stunting_per_diarrhoeal_episode': Parameter(
            Types.REAL, 'relative risk of stunting for recent diarrhoea episode'),

        # progression parameters
        'r_progression_severe_stunting_by_agegp': Parameter(
            Types.LIST, 'list with rates of progression to severe stunting by age group'),
        'rr_progress_severe_stunting_preterm_and_AGA': Parameter(
            Types.REAL, 'relative risk of severe stunting if born preterm and adequate for gestational age'),
        'rr_progress_severe_stunting_SGA_and_term': Parameter(
            Types.REAL, 'relative risk of severe stunting if born term and small for gestational age'),
        'rr_progress_severe_stunting_SGA_and_preterm': Parameter(
            Types.REAL, 'relative risk of severe stunting if born preterm and small for gestational age'),
        'rr_progress_severe_stunting_untreated_HIV': Parameter(
            Types.REAL, 'relative risk of severe stunting for untreated HIV+'),
        'rr_progress_severe_stunting_previous_wasting': Parameter(
            Types.REAL, 'relative risk of severe stunting if previously wasted'),

        'baseline_rate_of_HAZ_improvement_by_1sd': Parameter(
            Types.REAL, 'baseline rate or natural recovery rate for HAZ improvement by 1 standard deviation'),
        'rr_stunting_improvement_with_continued_breastfeeding': Parameter(
            Types.REAL, 'relative rate of improvement in stunting HAZ with continued breastfeeding'),

    }

    PROPERTIES = {
        'un_HAZ_score': Property(Types.REAL, 'height-for-age z-score'),
        'un_ever_stunted': Property(Types.BOOL, 'had stunting before (HAZ <-2)'),
        'un_HAZ_category': Property(Types.CATEGORICAL, 'height-for-age z-score group',
                                    categories=['HAZ<-3', '-3<=HAZ<-2', 'HAZ>=-2']),
        'un_clinical_chronic_malnutrition':
            Property(Types.CATEGORICAL, 'clinical acute malnutrition state based on HAZ',
                     categories=['moderate_stunting', 'severe_stunting']),
        'un_last_stunting_date_of_onset': Property(Types.DATE, 'date of onset of lastest stunting episode'),
        'un_CM_treatment_type': Property(Types.CATEGORICAL, 'treatment types for of chronic malnutrition',
                                         categories=['continued breastfeeding']),

    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # set the linear model equations for prevalence and incidence
        self.prevalence_equations_by_age = dict()
        self.stunting_incidence_equation = dict()

        # helper function for recent diarrhoeal episodes
        self.diarrhoea_in_last_6months = None

        # set the linear model equation for progression to severe stunting state
        self.severe_stunting_progression_equation = dict()

        # set the linear model equation for recovery of stunting
        self.stunting_improvement_rate = dict()

        # dict to hold counters for the number of episodes by stunting-type and age-group
        blank_counter = dict(zip(self.stunting_states, [list() for _ in self.stunting_states]))
        self.stunting_incident_case_tracker_blank = {
            '0y': copy.deepcopy(blank_counter),
            '1y': copy.deepcopy(blank_counter),
            '2y': copy.deepcopy(blank_counter),
            '3y': copy.deepcopy(blank_counter),
            '4y': copy.deepcopy(blank_counter),
            '5+y': copy.deepcopy(blank_counter)
        }
        self.stunting_incident_case_tracker = copy.deepcopy(self.stunting_incident_case_tracker_blank)

        zeros_counter = dict(zip(self.stunting_states, [0] * len(self.stunting_states)))
        self.stunting_incident_case_tracker_zeros = {
            '0y': copy.deepcopy(zeros_counter),
            '1y': copy.deepcopy(zeros_counter),
            '2y': copy.deepcopy(zeros_counter),
            '3y': copy.deepcopy(zeros_counter),
            '4y': copy.deepcopy(zeros_counter),
            '5+y': copy.deepcopy(zeros_counter)
        }

        # dict to hold the DALY weights
        self.daly_wts = dict()

    def read_parameters(self, data_folder):
        """
        :param data_folder: path of a folder supplied to the Simulation containing data files.
              Typically modules would read a particular file within here.
        :return:
        """
        # Update parameters from the resource dataframe
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Undernutrition.xlsx', sheet_name='Parameter_values_CM')
        self.load_parameters_from_dataframe(dfd)

        p = self.parameters

        # Check that every value has been read-in successfully
        for param_name, param_type in self.PARAMETERS.items():
            assert param_name in p, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert isinstance(p[param_name],
                              param_type.python_type), f'Parameter "{param_name}" is not read in correctly from the ' \
                                                       f'resourcefile.'

        # Stunting does not have specific symptoms, but a consequence of poor nutrition and repeated infections

        # no DALYs for stunting in the TLO daly weights

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

        # # # # # allocate initial prevalence of stunting at the start of the simulation # # # # #

        def make_scaled_linear_model_stunting(agegp):
            """Makes the unscaled linear model with intercept of baseline odds of stunting (HAZ <-2).
            Calculates the mean odds of stunting by age group and then creates a new linear model
            with adjusted intercept so odds in 0-year-olds matches the specified value in the model
            when averaged across the population
            """
            def get_odds_stunting(agegp):
                """
                This function will calculate the HAZ scores by categories and return the odds of stunting
                :param agegp: age grouped in months
                :return:
                """
                # generate random numbers from N(meean, sd)
                baseline_HAZ_prevalence_by_agegp = f'prev_HAZ_distribution_age_{agegp}'
                HAZ_normal_distribution = norm(loc=p[baseline_HAZ_prevalence_by_agegp][0],
                                               scale=p[baseline_HAZ_prevalence_by_agegp][1])

                # get all stunting: HAZ <-2
                probability_over_or_equal_minus2sd = HAZ_normal_distribution.sf(-2)
                probability_less_than_minus2sd = 1 - probability_over_or_equal_minus2sd

                # convert probability to odds
                base_odds_of_stunting = probability_less_than_minus2sd / (1-probability_less_than_minus2sd)

                return base_odds_of_stunting

            def make_linear_model_stunting(agegp, intercept=1.0):
                return LinearModel(
                    LinearModelType.LOGISTIC,
                    get_odds_stunting(agegp=agegp),  # base odds
                    # Predictor('gi_last_diarrhoea_date_of_onset').when(range(self.sim.date - DateOffset(weeks=2)),
                    #                                                   p['or_stunting_no_recent_diarrhoea']),
                    Predictor('sex').when('M', p['or_stunting_male']),
                    Predictor('li_ed_lev').when(1, p['or_stunting_mother_no_education'])
                        .when(2, p['or_stunting_mother_primary_education']),
                    Predictor('li_wealth').when(2, p['or_stunting_hhwealth_Q2'])
                        .when(3, p['or_stunting_hhwealth_Q3'])
                        .when(4, p['or_stunting_hhwealth_Q4'])
                        .when(5, p['or_stunting_hhwealth_Q5']),
                )

            unscaled_lm = make_linear_model_stunting(agegp)  # intercept=get_odds_stunting(agegp)
            target_mean = get_odds_stunting(agegp='12_23mo')
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 1)]).mean()
            scaled_intercept = get_odds_stunting(agegp) * (target_mean / actual_mean)
            scaled_lm = make_linear_model_stunting(agegp, intercept=scaled_intercept)
            return scaled_lm

        # the linear model returns the probability that is implied by the model prob = odds / (1 + odds)
        for agegp in ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo']:
            self.prevalence_equations_by_age[agegp] = make_scaled_linear_model_stunting(agegp)

        prevalence_of_stunting = pd.DataFrame(index=df.loc[df.is_alive & (df.age_exact_years < 5)].index)

        prevalence_of_stunting['0_5mo'] = self.prevalence_equations_by_age['0_5mo']\
            .predict(df.loc[df.is_alive & (df.age_exact_years < 0.5)])
        prevalence_of_stunting['6_11mo'] = self.prevalence_equations_by_age['6_11mo']\
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 0.5) & (df.age_exact_years < 1))])
        prevalence_of_stunting['12_23mo'] = self.prevalence_equations_by_age['12_23mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 1) & (df.age_exact_years < 2))])
        prevalence_of_stunting['24_35mo'] = self.prevalence_equations_by_age['24_35mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 2) & (df.age_exact_years < 3))])
        prevalence_of_stunting['36_47mo'] = self.prevalence_equations_by_age['36_47mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 3) & (df.age_exact_years < 4))])
        prevalence_of_stunting['48_59mo'] = self.prevalence_equations_by_age['48_59mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 4) & (df.age_exact_years < 5))])

        def get_prob_severe_in_overall_stunting(agegp):
            """
            This function will calculate the HAZ scores by categories and return probability of severe stunting
            :param agegp: age grouped in months
            :return:
            """
            # generate random numbers from N(meean, sd)
            baseline_HAZ_prevalence_by_agegp = f'prev_HAZ_distribution_age_{agegp}'
            HAZ_normal_distribution = norm(loc=p[baseline_HAZ_prevalence_by_agegp][0],
                                           scale=p[baseline_HAZ_prevalence_by_agegp][1])

            # get all stunting: HAZ <-2
            probability_over_or_equal_minus2sd = HAZ_normal_distribution.sf(-2)
            probability_less_than_minus2sd = 1 - probability_over_or_equal_minus2sd

            # get severe stunting zcores: HAZ <-3
            probability_over_or_equal_minus3sd = HAZ_normal_distribution.sf(-3)
            probability_less_than_minus3sd = 1 - probability_over_or_equal_minus3sd

            # get moderate stunting zcores: <=-3 HAZ <-2
            probability_between_minus3_minus2sd =\
                probability_over_or_equal_minus3sd - probability_over_or_equal_minus2sd

            # make HAZ <-2 as the 100% and get the adjusted probability of severe stunting
            proportion_severe_in_overall_stunting = probability_less_than_minus3sd * probability_less_than_minus2sd

            # get a list with probability of severe stunting, and moderate stunting
            return proportion_severe_in_overall_stunting

        # # # # # allocate initial prevalence of stunting at the start of the simulation # # # # #

        # further differentiate between severe stunting and moderate stunting, and normal HAZ
        for agegp in ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo']:
            stunted = self.rng.random_sample(len(prevalence_of_stunting[agegp])) < prevalence_of_stunting[agegp]
            for id in stunted[stunted].index:
                probability_of_severe = get_prob_severe_in_overall_stunting(agegp)
                stunted_category = self.rng.choice(['HAZ<-3', '-3<=HAZ<-2'],
                                                   p=[probability_of_severe, 1 - probability_of_severe])
                df.at[id, 'un_HAZ_category'] = stunted_category
            df.loc[stunted[stunted==False].index, 'un_HAZ_category'] = 'HAZ>=-2'

    def count_all_previous_diarrhoea_episodes(self, today, index):
        """
        Get all diarrhoea episodes since birth prior to today's date
        :param today:
        :param index:
        :return:
        """
        df = self.sim.population.props
        # delta_dates = today - (today - DateOffset(months=3))
        list_dates = []

        for person in index:
            delta_dates = df.at[person, 'date_of_birth'] - today
            for i in range(delta_dates.days):
                day = today - DateOffset(days=i)
                while df.gi_last_diarrhoea_date_of_onset[person] == day:
                    list_dates.append(day)

        total_diarrhoea_count_to_date = len(list_dates)

        return total_diarrhoea_count_to_date

    def initialise_simulation(self, sim):
        """Prepares for simulation:
        * Schedules the main polling event
        * Schedules the main logging event
        * Establishes the linear models and other data structures using the parameters that have been read-in
        * Store the consumables that are required in each of the HSI
        """
        df = self.sim.population.props
        p = self.parameters

        # Schedule the main polling event
        sim.schedule_event(StuntingPollingEvent(self), sim.date + DateOffset(months=6))

        # Schedule progression to severe stunting
        sim.schedule_event(SevereStuntingPollingEvent(self), sim.date + DateOffset(months=9))

        # Schedule the main logging event (to first occur in one year)
        # sim.schedule_event(StuntingLoggingEvent(self), sim.date + DateOffset(years=1))

        # Get DALY weights
        # no DALYs for stunting directly, but cognitive impairment should be added later

        # --------------------------------------------------------------------------------------------
        # # # # # # # # # # INCIDENCE # # # # # # # # # #
        # Make a linear model equation that govern the probability that a person becomes stunted HAZ<-2
        def make_scaled_lm_stunting_incidence():
            """
            Makes the unscaled linear model with default intercept of 1. Calculates the mean incidents rate for
            1-year-olds and then creates a new linear model with adjusted intercept so incidents in 1-year-olds
            matches the specified value in the model when averaged across the population
            """
            def make_lm_stunting_incidence(intercept=1.0):
                return LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept,
                    Predictor('age_exact_years')
                        .when('<0.5', p['base_inc_rate_stunting_by_agegp'][0])
                        .when('.between(0.5,0.9999)', p['base_inc_rate_stunting_by_agegp'][1]),
                    Predictor('age_years')
                        .when('.between(1,1)', p['base_inc_rate_stunting_by_agegp'][2])
                        .when('.between(2,2)', p['base_inc_rate_stunting_by_agegp'][3])
                        .when('.between(3,3)', p['base_inc_rate_stunting_by_agegp'][4])
                        .when('.between(4,4)', p['base_inc_rate_stunting_by_agegp'][5])
                        .otherwise(0.0),
                    Predictor('nb_size_for_gestational_age').when('small_for_gestational_age',
                                                                  p['rr_stunting_SGA_and_term']),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& (nb_late_preterm == False) & (nb_early_preterm == False)',
                                     p['rr_stunting_SGA_and_term']),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& ((nb_late_preterm == True) | (nb_early_preterm == True))',
                                     p['rr_stunting_SGA_and_preterm']),
                    Predictor().when('(nb_size_for_gestational_age == "average_for_gestational_age") '
                                     '& ((nb_late_preterm == True) | (nb_early_preterm == True))',
                                     p['rr_stunting_preterm_and_AGA']),
                    Predictor().when('(hv_inf == True) & (hv_art == "not")', p['rr_stunting_untreated_HIV']),
                    Predictor('li_wealth').apply(lambda x: 1 if x == 1 else (x - 1) ** (p['rr_stunting_wealth_level'])),
                    Predictor('nb_breastfeeding_status').when('non_exclusive | none',
                                                              p['rr_stunting_no_exclusive_breastfeeding']),
                    Predictor().when('((nb_breastfeeding_status == "non_exclusive") | '
                                     '(nb_breastfeeding_status == "none")) & (age_exact_years < 0.5)',
                                     p['rr_stunting_no_exclusive_breastfeeding']),
                    Predictor().when('(nb_breastfeeding_status == "none") & (age_exact_years.between(0.5,2))',
                                     p['rr_stunting_no_continued_breastfeeding']),
                    Predictor('previous_diarrhoea_episodes', external=True).apply(
                        lambda x: x ** (p['rr_stunting_per_diarrhoeal_episode'])),
                )

            unscaled_lm = make_lm_stunting_incidence()
            target_mean = p[f'base_inc_rate_stunting_by_agegp'][2]
            actual_mean = unscaled_lm.predict(
                df.loc[df.is_alive & (df.age_years == 1) & (df.un_HAZ_category == 'HAZ>=-2')],
                previous_diarrhoea_episodes=
                self.count_all_previous_diarrhoea_episodes(
                    today=sim.date, index=df.loc[df.is_alive & (df.age_years == 1) &
                                                 (df.un_HAZ_category == 'HAZ>=-2')].index)).mean()

            scaled_intercept = 1.0 * (target_mean / actual_mean)
            scaled_lm = make_lm_stunting_incidence(intercept=scaled_intercept)
            return scaled_lm

        self.stunting_incidence_equation = make_scaled_lm_stunting_incidence()

        # --------------------------------------------------------------------------------------------
        # # # # # # # # # # SEVERITY PROGRESSION # # # # # # # # # #
        # Make a linear model equation that govern the probability that a person becomes severely stunted HAZ<-3
        def make_scaled_lm_severe_stunting():
            """
            Makes the unscaled linear model with default intercept of 1. Calculates the mean progression rate for
            1-year-olds and then creates a new linear model with adjusted intercept so progression in 1-year-olds
            matches the specified value in the model when averaged across the population
            """
            def make_lm_severe_stunting(intercept=1.0):
                return LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept,
                    Predictor('age_exact_years')
                        .when('<0.5', p['r_progression_severe_stunting_by_agegp'][0])
                        .when('.between(0.5,0.9999)', p['r_progression_severe_stunting_by_agegp'][1])
                        .otherwise(0.0),
                    Predictor('age_years')
                        .when('.between(1,1)', p['r_progression_severe_stunting_by_agegp'][2])
                        .when('.between(2,2)', p['r_progression_severe_stunting_by_agegp'][3])
                        .when('.between(3,3)', p['r_progression_severe_stunting_by_agegp'][4])
                        .when('.between(4,4)', p['r_progression_severe_stunting_by_agegp'][5])
                        .otherwise(0.0),
                    Predictor('nb_size_for_gestational_age').when('small_for_gestational_age',
                                                                  p['rr_stunting_SGA_and_term']),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& (nb_late_preterm == False) & (nb_early_preterm == False)',
                                     p['rr_progress_severe_stunting_SGA_and_term']),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& ((nb_late_preterm == True) | (nb_early_preterm == True))',
                                     p['rr_progress_severe_stunting_SGA_and_preterm']),
                    Predictor().when('(nb_size_for_gestational_age == "average_for_gestational_age") '
                                     '& ((nb_late_preterm == True) | (nb_early_preterm == True))',
                                     p['rr_progress_severe_stunting_preterm_and_AGA']),
                    # Predictor('un_ever_wasted').when(True, p['rr_progress_severe_stunting_previous_wasting']),
                    Predictor().when('(hv_inf == True) & (hv_art == "not")', p['rr_stunting_untreated_HIV']),
                )

            unscaled_lm = make_lm_severe_stunting()
            target_mean = p[f'base_inc_rate_stunting_by_agegp'][2]
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 1) &
                                                     (df.un_HAZ_category == '-3<=HAZ<-2')]).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean)
            scaled_lm = make_lm_severe_stunting(intercept=scaled_intercept)
            return scaled_lm

        self.severe_stunting_progression_equation = make_scaled_lm_severe_stunting()

        # --------------------------------------------------------------------------------------------
        # # # # # # # # # # RECOVERY # # # # # # # # # #
        # Make a linear model equation that govern the probability that a person improves in stunting state
        self.stunting_improvement_rate.update({
            'moderate_stunting':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            p['baseline_rate_of_HAZ_improvement_by_1sd'],
                            Predictor('un_CM_treatment_type')
                            .when('continued_breastfeeding', p['rr_stunting_improvement_with_continued_breastfeeding'])
                            .otherwise(0.0)
                            ),
            'severe_stunting':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            p['baseline_rate_of_HAZ_improvement_by_1sd'],
                            Predictor('un_CM_treatment_type')
                            .when('continued_breastfeeding', p['rr_stunting_improvement_with_continued_breastfeeding'])
                            .otherwise(0.0)
                            )
        })

    def on_birth(self, mother_id, child_id):
        pass

    def report_daly_values(self):
        df = self.sim.population.props

        total_daly_values = pd.Series(data=0.0, index=df.index[df.is_alive])

        return total_daly_values


class StuntingPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all stunting properties for the population
    It determines who will be stunted and schedules individual incident cases to represent onset.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=6))
        assert isinstance(module, Stunting)

    def apply(self, population):
        df = population.props
        rng = self.module.rng
        p = self.module.parameters

        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')

        # Determine who will be onset with stunting among those who are not currently stunted
        incidence_of_stunting = self.module.stunting_incidence_equation.predict(
            df.loc[df.is_alive & (df.age_exact_years < 5) & (df.un_HAZ_category == 'HAZ>=-2')],
            previous_diarrhoea_episodes=self.module.count_all_previous_diarrhoea_episodes(
                today=self.sim.date, index=df.loc[df.is_alive & (df.age_exact_years < 5) &
                                                  (df.un_HAZ_category == 'HAZ>=-2')].index))
        stunted = rng.random_sample(len(incidence_of_stunting)) < incidence_of_stunting

        # determine the time of onset and other disease characteristics for each individual
        for person_id in stunted[stunted].index:
            # Allocate a date of onset for stunting episode
            date_onset = self.sim.date + DateOffset(days=rng.randint(0, days_until_next_polling_event))

            # Create the event for the onset of stunting (start with mild/moderate stunting)
            self.sim.schedule_event(
                event=StuntingOnsetEvent(module=self.module,
                                         person_id=person_id,
                                         am_state='moderate_stunting'), date=date_onset
            )


class SevereStuntingPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that determines those that will progress to severe stunting
     and schedules individual incident cases to represent onset od severe stunting.
     First run to occur 3 months after the first StuntingPollingEvent
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=6))
        assert isinstance(module, Stunting)

    def apply(self, population):
        df = population.props
        rng = self.module.rng
        p = self.module.parameters

        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')

        # determine those individuals that will progress to severe stunting
        progression_to_severe_stunting = self.module.severe_stunting_progression_equation.predict(
            df.loc[df.is_alive & (df.age_exact_years < 5) & (df.un_HAZ_category == '-3<=HAZ<-2')])
        severely_stunted = rng.random_sample(len(progression_to_severe_stunting)) < progression_to_severe_stunting

        # determine the onset date of severe stunting and schedule event
        for person_id in severely_stunted[severely_stunted].index:
            # Allocate a date of onset for stunting episode
            date_onset_severe_stunting = self.sim.date + DateOffset(days=rng.randint(0, days_until_next_polling_event))

            # Create the event for the onset of severe stunting
            self.sim.schedule_event(
                event=ProgressionSevereStuntingEvent(module=self.module,
                                                     person_id=person_id,
                                                     cm_state='severe_stunting'), date=date_onset_severe_stunting
            )


class StuntingRecoveryPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that determines those that will improve their stunting state
     and schedules individual recoveries.
    """
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=6))
        assert isinstance(module, Stunting)

    def apply(self, population):
        df = population.props
        rng = self.module.rng
        p = self.module.parameters

        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')

        # determine those individuals that will improve stunting state
        improvement_of_stunting_state = self.module.improvement_in_stunting_status_equation.predict(
            df.loc[df.is_alive & (df.age_exact_years < 5) & (df.un_HAZ_category == '-3<=HAZ<-2')])
        severely_stunted = rng.random_sample(len(improvement_of_stunting_state)) < improvement_of_stunting_state

        # determine the onset date of severe stunting and schedule event
        for person_id in severely_stunted[severely_stunted].index:
            # Allocate a date of onset for stunting episode
            date_onset_severe_stunting = self.sim.date + DateOffset(days=rng.randint(0, days_until_next_polling_event))

            # Create the event for the onset of severe stunting
            self.sim.schedule_event(
                event=ProgressionSevereStuntingEvent(module=self.module,
                                                     person_id=person_id,
                                                     cm_state='severe_stunting'), date=date_onset_severe_stunting
            )


class StuntingOnsetEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of stunting (stunting with HAZ <-2).
     * Refreshes all the properties so that they pertain to this current episode of stunting
     * Imposes the symptoms
     * Schedules relevant natural history event {(ProgressionSevereStuntingEvent) and
       (either StuntingRecoveryEvent or StuntingDeathEvent)}
    """

    AGE_GROUPS = {0: '0y', 1: '1y', 2: '2y', 3: '3y', 4: '4y'}

    def __init__(self, module, person_id, am_state):
        super().__init__(module, person_id=person_id)
        self.stunting_state = am_state

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters
        rng = m.rng

        df.at[person_id, 'un_ever_stunted'] = True
        df.at[person_id, 'un_currently_stunted'] = True
        df.at[person_id, 'un_HAZ_category'] = '-3<=HAZ<-2'  # start as moderate stunting
        df.at[person_id, 'un_clinical_acute_malnutrition'] = self.stunting_state
        df.at[person_id, 'un_last_stunting_date_of_onset'] = self.sim.date

        # -------------------------------------------------------------------------------------------
        # Add this incident case to the tracker
        age_group = StuntingOnsetEvent.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
        m.stunting_incident_case_tracker[age_group][self.stunting_state].append(self.sim.date)


class ProgressionSevereStuntingEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of severe stunting (with HAZ <-3).
     * Refreshes all the properties so that they pertain to this current episode of stunting
     * Imposes the symptoms
     * Schedules relevant natural history event {(either WastingRecoveryEvent or WastingDeathEvent)}
    """

    def __init__(self, module, person_id, cm_state):
        super().__init__(module, person_id=person_id)
        self.stunting_state = cm_state

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters
        rng = m.rng

        # update properties
        df.at[person_id, 'un_HAZ_category'] = 'HAZ<-3'
        df.at[person_id, 'un_clinical_acute_malnutrition'] = self.stunting_state

        # -------------------------------------------------------------------------------------------
        # Add this incident case to the tracker
        age_group = StuntingOnsetEvent.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
        m.stunting_incident_case_tracker[age_group][self.stunting_state].append(self.sim.date)


class StuntingRecoveryEvent(Event, IndividualScopeEventMixin):
    """
    This event sets the properties back to normal state
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters
        rng = m.rng

        if not df.at[person_id, 'is_alive']:
            return

        df.at[person_id, 'un_HAZ_category'] = 'HAZ>=-2'  # not undernourished
        df.at[person_id, 'un_clinical_acute_malnutrition'] = None

        # # set the indexes to apply by age group
        # index_children_aged_0_5mo = df.index[df.is_alive & df.age_exact_years < 0.5]
        # index_children_aged_6_11mo = df.index[df.is_alive & ((df.age_exact_years >= 0.5) & (df.age_exact_years < 1))]
        # index_children_aged_12_23mo = df.index[df.is_alive & ((df.age_exact_years >= 1) & (df.age_exact_years < 2))]
        # index_children_aged_24_35mo = df.index[df.is_alive & ((df.age_exact_years >= 2) & (df.age_exact_years < 3))]
        # index_children_aged_36_47mo = df.index[df.is_alive & ((df.age_exact_years >= 3) & (df.age_exact_years < 4))]
        # index_children_aged_48_59mo = df.index[df.is_alive & ((df.age_exact_years >= 4) & (df.age_exact_years < 5))]
        #
        # # # # Random draw of HAZ scores from a normal distribution # # #
        # # HAZ scores for under 6 months old, update the df
        # df.loc[index_children_aged_0_5mo, 'un_HAZ_score'] = \
        #     np.random.normal(loc=p['prev_HAZ_distribution_age_0_5mo'][0],
        #                      scale=p['prev_HAZ_distribution_age_0_5mo'][1])
        #
        # # HAZ scores for 6 to 11 months
        # df.loc[index_children_aged_6_11mo, 'un_HAZ_score'] = \
        #     np.random.normal(loc=p['prev_HAZ_distribution_age_6_11mo'][0],
        #                      scale=p['prev_HAZ_distribution_age_6_11mo'][1])
        #
        # # HAZ scores for 12 to 23 months
        # df.loc[index_children_aged_12_23mo, 'un_HAZ_score'] = \
        #     np.random.normal(loc=p['prev_HAZ_distribution_age_12_23mo'][0],
        #                      scale=p['prev_HAZ_distribution_age_12_23mo'][1])
        #
        # # HAZ scores for 24 to 35 months
        # df.loc[index_children_aged_24_35mo, 'un_HAZ_score'] = \
        #     np.random.normal(loc=p['prev_HAZ_distribution_age_24_35mo'][0],
        #                      scale=p['prev_HAZ_distribution_age_24_35mo'][1])
        #
        # # HAZ scores for 36 to 47 months
        # df.loc[index_children_aged_36_47mo, 'un_HAZ_score'] = \
        #     np.random.normal(loc=p['prev_HAZ_distribution_age_36_47mo'][0],
        #                      scale=p['prev_HAZ_distribution_age_36_47mo'][1])
        #
        # # HAZ scores for 48 to 59 months
        # df.loc[index_children_aged_48_59mo, 'un_HAZ_score'] = \
        #     np.random.normal(loc=p['prev_HAZ_distribution_age_48_59mo'][0],
        #                      scale=p['prev_HAZ_distribution_age_48_59mo'][1])
        #
        # # HAZ category of under-5, update df
        # under_5_index = df.index[df.is_alive & df.age_exact_years < 5]
        # severe_stunting = df.loc[under_5_index, 'un_HAZ_score'] < -3.0
        # moderate_stunting = df.loc[df.is_alive & df.age_exact_years < 5 & (df['un_HAZ_score'] >= -3.0) & (df['un_HAZ_score'] < -2.0)]
        # no_stunting = df.loc[under_5_index, 'un_HAZ_score'] >= -2.0
        #
        # df.loc[severe_stunting.index, 'un_HAZ_category'] = 'HAZ<-3'
        # df.loc[moderate_stunting.index, 'un_HAZ_category'] = '-3<=HAZ<-2'
        # df.loc[no_stunting.index, 'un_HAZ_category'] = 'HAZ>=-2'
