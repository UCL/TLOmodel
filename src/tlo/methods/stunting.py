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
        'base_inc_rate_stunting': Parameter(
            Types.REAL, 'baseline incidence of stunting'),
        'rr_stunting_preterm_and_AGA': Parameter(
            Types.REAL, 'relative risk of stunting if born preterm and adequate for gestational age'),
        'rr_stunting_SGA_and_term': Parameter(
            Types.REAL, 'relative risk of stunting if born term and small for geatational age'),
        'rr_stunting_SGA_and_preterm': Parameter(
            Types.REAL, 'relative risk of stunting if born preterm and small for gestational age'),

    }

    PROPERTIES = {
        'un_HAZ_score': Property(Types.REAL, 'height-for-age z-score'),
        'un_HAZ_category': Property(Types.CATEGORICAL, 'height-for-age z-score group',
                                    categories=['HAZ<-3', '-3<=HAZ<-2', 'HAZ>=-2']),

    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.prevalence_equations_by_age = dict()

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

        # if 'HealthBurden' in self.sim.modules.keys():
        #     #get the DALY weight - 860-862 are the sequale codes for epilepsy
            # p['daly_wt_epilepsy_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=860)
            # p['daly_wt_epilepsy_less_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=861)
            # p['daly_wt_epilepsy_seizure_free'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=862)

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
        AGE_GROUPS = {0: '0y', 1: '1y', 2: '2y', 3: '3y', 4: '4y'}

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

                # # get severe stunting zcores: HAZ <-3
                # probability_over_or_equal_minus3sd = HAZ_normal_distribution.sf(-3)
                # probability_less_than_minus3sd = 1 - probability_over_or_equal_minus3sd
                #
                # # get moderate stunting zcores: <=-3 HAZ <-2
                # probability_between_minus3_minus2sd =\
                #     probability_over_or_equal_minus3sd - probability_over_or_equal_minus2sd

                # convert probability to odds
                base_odds_of_stunting = probability_less_than_minus2sd / (1-probability_less_than_minus2sd)

                return base_odds_of_stunting

            def make_linear_model_stunting(agegp, intercept=1.0):
                return LinearModel(
                    LinearModelType.LOGISTIC,
                    get_odds_stunting(agegp=agegp),  # base odds
                    # Predictor('gi_last_diarrhoea_date_of_onset').when(range(self.sim.date - DateOffset(weeks=2)),
                    #                                                   p['or_stunting_no_recent_diarrhoea']),
                    Predictor('li_ed_lev').when(1, p['or_stunting_mother_no_education'])
                        .when(2, p['or_stunting_mother_primary_education']),
                    Predictor('li_wealth').when(2, p['or_stunting_hhwealth_Q2'])
                        .when(3, p['or_stunting_hhwealth_Q3'])
                        .when(4, p['or_stunting_hhwealth_Q4'])
                        .when(5, p['or_stunting_hhwealth_Q5']),
                )

            unscaled_lm = make_linear_model_stunting(agegp)  # intercept=get_odds_stunting(agegp)
            target_mean = get_odds_stunting(agegp)
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 0)]).mean()
            scaled_intercept = get_odds_stunting(agegp) * (target_mean / actual_mean)
            scaled_lm = make_linear_model_stunting(agegp, intercept=scaled_intercept)
            # check by applying the model to mean incidence of 0-year-olds
            return scaled_lm

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

        print(prevalence_of_stunting)

        # allocate initial prevalence of stunting at the start of the simulation
        # set the indexes to apply by age group
        index_children_aged_0_5mo = df.index[df.is_alive & df.age_exact_years < 0.5]
        index_children_aged_6_11mo = df.index[df.is_alive & ((df.age_exact_years >= 0.5) & (df.age_exact_years < 1))]
        index_children_aged_12_23mo = df.index[df.is_alive & ((df.age_exact_years >= 1) & (df.age_exact_years < 2))]
        index_children_aged_24_35mo = df.index[df.is_alive & ((df.age_exact_years >= 2) & (df.age_exact_years < 3))]
        index_children_aged_36_47mo = df.index[df.is_alive & ((df.age_exact_years >= 3) & (df.age_exact_years < 4))]
        index_children_aged_48_59mo = df.index[df.is_alive & ((df.age_exact_years >= 4) & (df.age_exact_years < 5))]

        # # # Random draw of HAZ scores from a normal distribution # # #
        # HAZ scores for under 6 months old, update the df
        df.loc[index_children_aged_0_5mo, 'un_HAZ_score'] = \
            np.random.normal(loc=p['prev_HAZ_distribution_age_0_5mo'][0],
                             scale=p['prev_HAZ_distribution_age_0_5mo'][1])

        # HAZ scores for 6 to 11 months
        df.loc[index_children_aged_6_11mo, 'un_HAZ_score'] = \
            np.random.normal(loc=p['prev_HAZ_distribution_age_6_11mo'][0],
                             scale=p['prev_HAZ_distribution_age_6_11mo'][1])

        # HAZ scores for 12 to 23 months
        df.loc[index_children_aged_12_23mo, 'un_HAZ_score'] = \
            np.random.normal(loc=p['prev_HAZ_distribution_age_12_23mo'][0],
                             scale=p['prev_HAZ_distribution_age_12_23mo'][1])

        # HAZ scores for 24 to 35 months
        df.loc[index_children_aged_24_35mo, 'un_HAZ_score'] = \
            np.random.normal(loc=p['prev_HAZ_distribution_age_24_35mo'][0],
                             scale=p['prev_HAZ_distribution_age_24_35mo'][1])

        # HAZ scores for 36 to 47 months
        df.loc[index_children_aged_36_47mo, 'un_HAZ_score'] = \
            np.random.normal(loc=p['prev_HAZ_distribution_age_36_47mo'][0],
                             scale=p['prev_HAZ_distribution_age_36_47mo'][1])

        # HAZ scores for 48 to 59 months
        df.loc[index_children_aged_48_59mo, 'un_HAZ_score'] = \
            np.random.normal(loc=p['prev_HAZ_distribution_age_48_59mo'][0],
                             scale=p['prev_HAZ_distribution_age_48_59mo'][1])

        # HAZ category of under-5, update df
        under_5_index = df.index[df.is_alive & df.age_exact_years < 5]
        severe_stunting = df.loc[under_5_index, 'un_HAZ_score'] < -3.0
        moderate_stunting = df.loc[df.is_alive & df.age_exact_years < 5 & (df['un_HAZ_score'] >= -3.0) & (df['un_HAZ_score'] < -2.0)]
        no_stunting = df.loc[under_5_index, 'un_HAZ_score'] >= -2.0

        df.loc[severe_stunting.index, 'un_HAZ_category'] = 'HAZ<-3'
        df.loc[moderate_stunting.index, 'un_HAZ_category'] = '-3<=HAZ<-2'
        df.loc[no_stunting.index, 'un_HAZ_category'] = 'HAZ>=-2'

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother_id, child_id):
        pass
