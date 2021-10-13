"""
todo's
* Reference in resource file
* No DALY?
"""

import copy
from pathlib import Path
from scipy.stats import norm
from collections import namedtuple
import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

class Stunting(Module):
    """The Stunting module determines the prevalence of stunting.
    * Schedules new incidences of stunting, progression to severe stunting, ....
    *
    """

    INIT_DEPENDENCIES = {'Demography', 'Diarrhoea'}

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
    }

    stunting_states = ['HAZ<-3', '-3<=HAZ<-2', 'HAZ>=-2']

    PARAMETERS = {
        # Pevalence of stunting by age group at initiation
        'prev_HAZ_distribution_age_0_5mo': Parameter(
            Types.LIST, 'Distribution of HAZ among less than 6 months of age in 2015 (mean, standard deviation)'),
        'prev_HAZ_distribution_age_6_11mo': Parameter(
            Types.LIST, 'Distribution of HAZ among 6 months and 1 year of age in 2015 (mean, standard deviation)'),
        'prev_HAZ_distribution_age_12_23mo': Parameter(
            Types.LIST, 'Distribution of HAZ among 1 year olds in 2015 (mean, standard deviation)'),
        'prev_HAZ_distribution_age_24_35mo': Parameter(
            Types.LIST, 'Distribution of HAZ among 2 year olds in 2015 (mean, standard deviation)'),
        'prev_HAZ_distribution_age_36_47mo': Parameter(
            Types.LIST, 'Distribution of HAZ among 3 year olds in 2015 (mean, standard deviation)'),
        'prev_HAZ_distribution_age_48_59mo': Parameter(
            Types.LIST, 'Distribution of HAZ among 4 year olds  in 2015 (mean, standard deviation)'),

        # Risk factors for stunting prevalence at initiation
        'or_stunting_male': Parameter(
            Types.REAL, 'Odds ratio of stunting if male gender'),
        'or_stunting_preterm_and_AGA': Parameter(
            Types.REAL, 'Odds ratio of stunting if born preterm and adequate for gestational age'),
        'or_stunting_SGA_and_term': Parameter(
            Types.REAL, 'Odds ratio of stunting if born term and small for geatational age'),
        'or_stunting_SGA_and_preterm': Parameter(
            Types.REAL, 'Odds ratio of stunting if born preterm and small for gestational age'),
        'or_stunting_hhwealth_Q5': Parameter(
            Types.REAL, 'Odds ratio of stunting if household wealth is poorest Q5, ref group Q1'),
        'or_stunting_hhwealth_Q4': Parameter(
            Types.REAL, 'Odds ratio of stunting if household wealth is poorer Q4, ref group Q1'),
        'or_stunting_hhwealth_Q3': Parameter(
            Types.REAL, 'Odds ratio of stunting if household wealth is middle Q3, ref group Q1'),
        'or_stunting_hhwealth_Q2': Parameter(
            Types.REAL, 'Odds ratio of stunting if household wealth is richer Q2, ref group Q1'),

        # The incidence of stunting
        'base_inc_rate_stunting_by_agegp': Parameter(
            Types.LIST, 'List with baseline incidence of stunting by age group (1-5, 6-11, 12-23, 24-35, 36-47, 48-59mo'),
        'rr_stunting_preterm_and_AGA': Parameter(
            Types.REAL, 'Relative risk of stunting if born preterm and adequate for gestational age'),
        'rr_stunting_SGA_and_term': Parameter(
            Types.REAL, 'Relative risk of stunting if born term and small for gestational age'),
        'rr_stunting_SGA_and_preterm': Parameter(
            Types.REAL, 'Relative risk of stunting if born preterm and small for gestational age'),
        'rr_stunting_prior_wasting': Parameter(
            Types.REAL, 'Relative risk of stunting if prior wasting in the last 3 months'),
        'rr_stunting_untreated_HIV': Parameter(
            Types.REAL, 'Relative risk of stunting for untreated HIV+'),
        'rr_stunting_wealth_level': Parameter(
            Types.REAL, 'Relative risk of stunting by increase in wealth level'),
        'rr_stunting_no_exclusive_breastfeeding': Parameter(
            Types.REAL, 'Relative risk of stunting for not exclusively breastfed babies < 6 months'),
        'rr_stunting_no_continued_breastfeeding': Parameter(
            Types.REAL, 'Relative risk of stunting for not continued breasfed infants 6-24 months'),
        'rr_stunting_per_diarrhoeal_episode': Parameter(
            Types.REAL, 'Relative risk of stunting for recent diarrhoea episode'),

        # Progression to severe stunting
        'r_progression_severe_stunting_by_agegp': Parameter(
            Types.LIST, 'List with rates of progression to severe stunting by age group (1-5, 6-11, 12-23, 24-35, 36-47, 48-59mo'),
        'rr_progress_severe_stunting_untreated_HIV': Parameter(
            Types.REAL, 'Relative risk of severe stunting for untreated HIV+'),
        'rr_progress_severe_stunting_previous_wasting': Parameter(
            Types.REAL, 'Relative risk of severe stunting if previously wasted'),
        'prob_remained_stunted_in_the_next_3months': Parameter(
            Types.REAL, 'Probability of stunted remained stunted in the next 3 month period'),

        # The effect of treatment (todo - check definitions)
        'un_effectiveness_complementary_feeding_promo_education_only_in_stunting_reduction': Parameter(
            Types.REAL, 'Effectiveness of complementary feeding promotion (education only) in reducing stunting'),
        'un_effectiveness_complementary_feeding_promo_with_food_supplementation_in_stunting_reduction': Parameter(
            Types.REAL,
            'Effectiveness of complementary feeding promotion with food supplementation in reducing stunting'),
        'un_effectiveness_zinc_supplementation_in_stunting_reduction': Parameter(
            Types.REAL, 'Effectiveness of zinc supplementation in reducing stunting'),
    }

    PROPERTIES = {
        'un_ever_stunted': Property(Types.BOOL, 'had stunting before (HAZ <-2)'),
        'un_HAZ_category': Property(Types.CATEGORICAL, 'height-for-age z-score group',
                                    categories=['HAZ<-3', '-3<=HAZ<-2', 'HAZ>=-2']),
        'un_last_stunting_date_of_onset': Property(Types.DATE, 'date of onset of lastest stunting episode'),
        'un_stunting_recovery_date': Property(Types.DATE, 'recovery date, when HAZ>=-2'),
        'un_cm_treatment_type': Property(Types.CATEGORICAL, 'treatment types for of chronic malnutrition',
                                         categories=['education_on_complementary_feeding',
                                                     'complementary_feeding_with_food_supplementation'] +
                                                    ['none'] + ['not_applicable']),
        'un_stunting_tx_start_date': Property(Types.DATE, 'start date of treatment for stunting')
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.models = None  # will store the models used in the module

    def read_parameters(self, data_folder):
        # Load parameters from resource file
        self.load_parameters_from_dataframe(
            pd.read_excel(
                Path(self.resourcefilepath) / 'ResourceFile_Stunting.xlsx', sheet_name='Parameter_values')
        )

        # Check that every value has been read-in successfully
        p = self.parameters
        for param_name, param_type in self.PARAMETERS.items():
            assert param_name in p, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert isinstance(p[param_name],
                              param_type.python_type), f'Parameter "{param_name}" is not read in correctly from the ' \
                                                       f'resourcefile.'

    def initialise_population(self, population):
        """Set initial prevalence of stunting according to distributions providec in parameters"""
        df = population.props
        p = self.parameters

        # Set default properties
        df.loc[df.is_alive, 'un_ever_stunted'] = False
        df.loc[df.is_alive, 'un_HAZ_category'] = 'HAZ>=-2'  # not undernourished
        df.loc[df.is_alive, 'un_last_stunting_date_of_onset'] = pd.NaT
        df.loc[df.is_alive, 'un_stunting_recovery_date'] = pd.NaT
        df.loc[df.is_alive, 'un_cm_treatment_type'] = np.nan
        df.loc[df.is_alive, 'un_stunting_tx_start_date'] = np.nan

        def get_probs_stunting(agegp):
            """For the a given HAZ distribution (specified in the parameters by age-group), find the odds of
            a value <-2 (= 'stunted') and the odds of a value <-3 given a value <-2 (severely stunted)."""

            mean, stdev = p[f'prev_HAZ_distribution_age_{agegp}']
            haz_distribution = norm(loc=mean, scale=stdev)

            # Compute proportion "stunted" (HAZ <-2)
            p_stunted = haz_distribution.cdf(-2.0)

            # Compute proportion "severely stunted" given "stunted" (HAZ <-3 given HAZ <-2)
            p_severely_stunted_given_stunted = haz_distribution.cdf(-3.0) / haz_distribution.cdf(-2.0)

            # Return results needed as named tuple:
            result = namedtuple('probs', ['prob_stunting', 'prob_severe_given_stunting'])

            return result(
                p_stunted,
                p_severely_stunted_given_stunted
            )

        def make_scaled_linear_model_stunting(target_prob, mask):

            def make_linear_model_stunting(intercept=1.0):
                return LinearModel(
                    LinearModelType.LOGISTIC,
                    intercept,
                    Predictor('sex').when('M', p['or_stunting_male']),
                    Predictor('li_wealth')  .when(2, p['or_stunting_hhwealth_Q2'])
                                            .when(3, p['or_stunting_hhwealth_Q3'])
                                            .when(4, p['or_stunting_hhwealth_Q4'])
                                            .when(5, p['or_stunting_hhwealth_Q5']),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& (nb_late_preterm == False) & (nb_early_preterm == False)',
                                     p['or_stunting_SGA_and_term']),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& ((nb_late_preterm == True) | (nb_early_preterm == True))',
                                     p['or_stunting_SGA_and_preterm']),
                    Predictor().when('(nb_size_for_gestational_age == "average_for_gestational_age") '
                                     '& ((nb_late_preterm == True) | (nb_early_preterm == True))',
                                     p['or_stunting_preterm_and_AGA'])
                )

            unscaled_lm = make_linear_model_stunting()

            actual_prob = unscaled_lm.predict(df.loc[mask]).mean()
            actual_odds = actual_prob / (1.0 - actual_prob)
            target_odds = target_prob / (1.0 - target_prob)

            return make_linear_model_stunting(
                intercept=target_odds / actual_odds if not np.isnan(actual_prob) else target_odds
            )

        for agegp in ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo']:

            p_stunting = get_probs_stunting(agegp)

            low_bound_age_in_years = int(agegp.split('_')[0]) / 12.0
            high_bound_age_in_years = (1 + int(agegp.split('_')[1].split('mo')[0])) / 12.0

            mask = df.is_alive & df.age_exact_years.between(low_bound_age_in_years, high_bound_age_in_years, inclusive='left')

            stunted = make_scaled_linear_model_stunting(target_prob=p_stunting.prob_stunting, mask=mask).predict(df.loc[mask], self.rng)

            severely_stunted_idx = stunted.loc[stunted & (self.rng.rand(len(stunted)) < p_stunting.prob_severe_given_stunting)].index
            stunted_but_not_severe_idx = set(stunted[stunted].index) - set(severely_stunted_idx)

            df.loc[stunted_but_not_severe_idx, "un_HAZ_category"] = '-3<=HAZ<-2'
            df.loc[severely_stunted_idx, "un_HAZ_category"] = 'HAZ<-3'
            df.loc[stunted_but_not_severe_idx.union(severely_stunted_idx), 'un_last_stunting_date_of_onset'] = self.sim.date
            df.loc[stunted_but_not_severe_idx.union(severely_stunted_idx), 'un_ever_stunted'] = True

    def count_all_previous_diarrhoea_episodes(self, today, index):
        """
        Get all diarrhoea episodes since birth prior to today's date for non-stunted children;
        for already moderately stunted children, get all diarrhoea episodes since the onset of stunting
        :param today:
        :param index:
        :return:
        """
        # todo - don;' think this is working! remove
        df = self.sim.population.props
        list_dates = []

        for person in index:
            if df.at[person, 'un_HAZ_category'] == 'HAZ>=-2':
                delta_dates = df.at[person, 'date_of_birth'] - today
                for i in range(delta_dates.days):
                    day = today - DateOffset(days=i)
                    while df.gi_last_diarrhoea_date_of_onset[person] == day:
                        list_dates.append(day)

            if df.at[person, 'un_HAZ_category'] == '-3<=HAZ<-2':
                delta_dates = df.at[person, 'un_last_stunting_date_of_onset'] - today
                for i in range(delta_dates.days):
                    day = today - DateOffset(days=i)
                    while df.gi_last_diarrhoea_date_of_onset[person] == day:
                        list_dates.append(day)

        total_diarrhoea_count_to_date = len(list_dates)

        return total_diarrhoea_count_to_date

    def initialise_simulation(self, sim):
        """Prepares for simulation:
        * Schedules the main polling events
        * Establishes the models
        """
        # Schedule the main polling events
        sim.schedule_event(StuntingPollingEvent(self), sim.date)
        sim.schedule_event(StuntingRecoveryPollingEvent(self), sim.date)

        # Establish Models:
        self.models = Models(self)

    def on_birth(self, mother_id, child_id):
        """Set that on birth there is no stunting"""
        self.sim.population.props.loc[child_id, [
            'un_ever_stunted',
            'un_HAZ_category',
            'un_last_stunting_date_of_onset',
            'un_stunting_recovery_date',
            'un_cm_treatment_type',
            'un_stunting_tx_start_date'
        ]] = (
            False,
            'HAZ>=-2',
            pd.NaT,
            pd.NaT,
            'not_applicable',
            pd.NaT
        )

    def do_when_chronic_malnutrition_assessment(self, person_id):
        """
        This is called by the a generic HSI event when chronic malnutrition/ stunting is checked.
        :param person_id:
        :param hsi_event: The HSI event that has called this event
        :return:
        """
        # Interventions for stunting

        # Check for coverage of complementary feeding, by assuming
        # these interventions are given in supplementary feeding programmes (in wasting)
        if self.sim.modules['Wasting'].parameters['coverage_supplementary_feeding_program'] > self.rng.rand():
            # schedule HSI for complementary feeding program
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_complementary_feeding_with_supplementary_foods
                (module=self,
                 person_id=person_id),
                priority=0,
                topen=self.sim.date
            )
        else:
            # if not in supplementary feeding program, education only will be provided in outpatient health centres
            # schedule HSI for complementary feeding program
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_complementary_feeding_education_only
                (module=self,
                 person_id=person_id),
                priority=0,
                topen=self.sim.date
            )
            # ASSUMPTION : ALL PATIENTS WILL GET ADVICE/COUNSELLING AT OUTPATIENT VISITS

    def do_when_cm_treatment(self, person_id):
        """
        This function will apply the linear model of recovery based on intervention given
        :param person_id:
        :param intervention:
        :return:
        """
        df = self.sim.population.props

        stunting_improvement = self.stunting_improvement_based_on_interventions.predict(
            df.loc[[person_id]]).values[0]
        if self.rng.rand() < stunting_improvement:
            # schedule recovery date
            self.sim.schedule_event(
                event=StuntingRecoveryEvent(module=self, person_id=person_id),
                date=df.at[person_id, 'un_stunting_tx_start_date'] + DateOffset(weeks=4))
            # cancel progression to severe stunting date (in ProgressionEvent)
        else:
            # remained stunted or severe stunted
            return


class Models:
    def __init__(self, module):
        self.module = module
        self.p = module.parameters

        # set the linear model equations for prevalence and incidence
        self.prevalence_equations_by_age = dict()

        # Make a linear model equation that govern the probability that a person becomes stunted (i.e., HAZ<-2)
        self.stunting_incidence_equation = self.make_scaled_lm_stunting_incidence()

        # set the linear model equation for progression to severe stunting state
        self.severe_stunting_progression_equation = self.make_scaled_lm_severe_stunting()

        # --------------------- linear models following HSI interventions --------------------- #
        # set the linear models for stunting improvement by intervention
        #     # Make a linear model equation that govern the probability that a person improves in stunting state
        self.stunting_improvement_based_on_interventions = LinearModel(LinearModelType.MULTIPLICATIVE,
                    1.0,
                    Predictor('un_cm_treatment_type')
                    .when('complementary_feeding_with_food_supplementation',
                          self.p['un_effectiveness_complementary_feeding_promo_'
                            'with_food_supplementation_in_stunting_reduction'])
                    .when('education_on_complementary_feeding',
                          self.p['un_effectiveness_complementary_feeding_promo_'
                            'education_only_in_stunting_reduction'])
                    .otherwise(0.0)
                    )

    # --------------------------------------------------------------------------------------------

    def make_scaled_lm_stunting_incidence(self):
        """
        Makes the unscaled linear model with default intercept of 1. Calculates the mean incidents rate for
        1-year-olds and then creates a new linear model with adjusted intercept so incidents in 1-year-olds
        matches the specified value in the model when averaged across the population
        """
        p = self.p
        df = self.module.sim.population.props

        def make_lm_stunting_incidence(intercept=1.0):
            return LinearModel(
                LinearModelType.MULTIPLICATIVE,
                intercept,
                Predictor('age_exact_years').when('<0.5', p['base_inc_rate_stunting_by_agegp'][0])
                    .when('.between(0.5,0.9999)', p['base_inc_rate_stunting_by_agegp'][1])
                    .when('.between(1,1.9999)', p['base_inc_rate_stunting_by_agegp'][2])
                    .when('.between(2,2.9999)', p['base_inc_rate_stunting_by_agegp'][3])
                    .when('.between(3,3.9999)', p['base_inc_rate_stunting_by_agegp'][4])
                    .when('.between(4,4.9999)', p['base_inc_rate_stunting_by_agegp'][5])
                    .otherwise(0.0),
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
                Predictor('un_ever_wasted').when(True, p['rr_stunting_prior_wasting']),
                Predictor('nb_breastfeeding_status').when('non_exclusive | none',
                                                          p['rr_stunting_no_exclusive_breastfeeding']),
                Predictor().when('((nb_breastfeeding_status == "non_exclusive") | '
                                 '(nb_breastfeeding_status == "none")) & (age_exact_years < 0.5)',
                                 p['rr_stunting_no_exclusive_breastfeeding']),
                Predictor().when('(nb_breastfeeding_status == "none") & (age_exact_years.between(0.5,2))',
                                 p['rr_stunting_no_continued_breastfeeding']),
                # Predictor('previous_diarrhoea_episodes', external=True).apply(
                #     lambda x: x ** (p['rr_stunting_per_diarrhoeal_episode'])),
            )

        unscaled_lm = make_lm_stunting_incidence()
        target_mean = p[f'base_inc_rate_stunting_by_agegp'][2]
        actual_mean = unscaled_lm.predict(
            df.loc[df.is_alive & (df.age_years == 1) & (df.un_HAZ_category == 'HAZ>=-2')]).mean()
        # actual_mean = unscaled_lm.predict(
        #     df.loc[df.is_alive & (df.age_years == 1) & (df.un_HAZ_category == 'HAZ>=-2')],
        #     previous_diarrhoea_episodes=self.count_all_previous_diarrhoea_episodes(
        #         today=sim.date, index=df.loc[df.is_alive & (df.age_years == 1) &
        #                                      (df.un_HAZ_category == 'HAZ>=-2')].index)).mean()
        scaled_intercept = 1.0 * (target_mean / actual_mean) \
            if (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else 1.0
        scaled_lm = make_lm_stunting_incidence(intercept=scaled_intercept)
        return scaled_lm

    def make_scaled_lm_severe_stunting(self):
        # --------------------------------------------------------------------------------------------
        # Make a linear model equation that govern the probability that a person becomes severely stunted HAZ<-3
        # (natural history only, no interventions)
        """
        Makes the unscaled linear model with default intercept of 1. Calculates the mean progression rate for
        1-year-olds and then creates a new linear model with adjusted intercept so progression in 1-year-olds
        matches the specified value in the model when averaged across the population
        """
        p = self.p
        df = self.module.sim.population.props

        def make_lm_severe_stunting(intercept=1.0):
            return LinearModel(
                LinearModelType.MULTIPLICATIVE,
                intercept,
                Predictor('age_exact_years').when('<0.5', p['r_progression_severe_stunting_by_agegp'][0])
                    .when('.between(0.5,0.9999)',
                          p['r_progression_severe_stunting_by_agegp'][1])
                    .when('.between(1,1.9999)',
                          p['r_progression_severe_stunting_by_agegp'][2])
                    .when('.between(2,2.9999)',
                          p['r_progression_severe_stunting_by_agegp'][3])
                    .when('.between(3,3.9999)',
                          p['r_progression_severe_stunting_by_agegp'][4])
                    .when('.between(4,4.9999)',
                          p['r_progression_severe_stunting_by_agegp'][5])
                    .otherwise(0.0),
                Predictor('un_ever_wasted').when(True, p['rr_progress_severe_stunting_previous_wasting']),
                Predictor().when('(hv_inf == True) & (hv_art == "not")', p['rr_stunting_untreated_HIV']),
                # Predictor('previous_diarrhoea_episodes', external=True).apply(
                #     lambda x: x ** (p['rr_stunting_per_diarrhoeal_episode'])),
            )

        unscaled_lm = make_lm_severe_stunting()
        target_mean = p[f'base_inc_rate_stunting_by_agegp'][2]
        actual_mean = unscaled_lm.predict(
            df.loc[df.is_alive & (df.age_years == 1) & (df.un_HAZ_category == '-3<=HAZ<-2')]).mean()
        # actual_mean = unscaled_lm.predict(
        #     df.loc[df.is_alive & (df.age_years == 1) & (df.un_HAZ_category == '-3<=HAZ<-2')],
        #     previous_diarrhoea_episodes=self.count_all_previous_diarrhoea_episodes(
        #         today=sim.date, index=df.loc[df.is_alive & (df.age_years == 1) &
        #                                      (df.un_HAZ_category == '-3<=HAZ<-2')].index)).mean()
        scaled_intercept = 1.0 * (target_mean / actual_mean) \
            if (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else 1.0
        scaled_lm = make_lm_severe_stunting(intercept=scaled_intercept)
        return scaled_lm


class StuntingPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all stunting properties for the population
    It determines who will be stunted and schedules individual incident cases to represent onset.
    """

    AGE_GROUPS = {0: '0y', 1: '1y', 2: '2y', 3: '3y', 4: '4y'}

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, Stunting)

    def apply(self, population):
        df = population.props
        rng = self.module.rng

        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')

        # Determine who will be onset with stunting among those who are not currently stunted
        incidence_of_stunting = self.module.stunting_incidence_equation.predict(
            df.loc[df.is_alive & (df.age_exact_years < 5) & (df.un_HAZ_category == 'HAZ>=-2')])
        # incidence_of_stunting = self.module.stunting_incidence_equation.predict(
        #     df.loc[df.is_alive & (df.age_exact_years < 5) & (df.un_HAZ_category == 'HAZ>=-2')],
        #     previous_diarrhoea_episodes=self.module.count_all_previous_diarrhoea_episodes(
        #         today=self.sim.date, index=df.loc[df.is_alive & (df.age_exact_years < 5) &
        #                                           (df.un_HAZ_category == 'HAZ>=-2')].index))
        stunted = rng.random_sample(len(incidence_of_stunting)) < incidence_of_stunting
        stunted_idx = stunted[stunted].index

        # determine the time of onset and other disease characteristics for each individual
        for person_id in stunted_idx:
            # Allocate a date of onset for stunting episode
            date_onset = self.sim.date + DateOffset(days=rng.randint(0, days_until_next_polling_event))

            # Create the event for the onset of stunting (start with moderate stunting)
            self.sim.schedule_event(
                event=StuntingOnsetEvent(module=self.module,
                                         person_id=person_id), date=date_onset)


class StuntingRecoveryPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that determines those that will improve their stunting state
     and schedules individual recoveries, these are based on interventions
    """
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))
        assert isinstance(module, Stunting)

    def apply(self, population):
        df = population.props
        rng = self.module.rng

        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')

        # determine those individuals that will improve stunting state
        improvement_of_stunting_state = self.module.stunting_improvement_based_on_interventions.predict(
            df.loc[df.is_alive & (df.age_exact_years < 5) & ((df.un_HAZ_category == '-3<=HAZ<-2') |
                                                             (df.un_HAZ_category == 'HAZ<-3'))])
        improved_stunting_state = rng.random_sample(len(improvement_of_stunting_state)) < improvement_of_stunting_state

        # determine the onset date of severe stunting and schedule event
        for person_id in improved_stunting_state[improved_stunting_state].index:
            # Allocate a date of onset for stunting episode
            if df.at[person_id,  'un_stunting_tx_start_date'] > self.sim.date - DateOffset(months=3):
                date_recovery_stunting = df.at[person_id,  'un_stunting_tx_start_date'] + \
                                         DateOffset(days=rng.randint(0, days_until_next_polling_event))
            else:
                date_recovery_stunting = self.sim.date + \
                                         DateOffset(days=rng.randint(0, days_until_next_polling_event))
            # Create the event for the onset of stunting recovery by 1sd in HAZ
            self.sim.schedule_event(
                event=StuntingRecoveryEvent(module=self.module,
                                            person_id=person_id), date=date_recovery_stunting)


class StuntingOnsetEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of stunting (stunting with HAZ <-2).
     * Refreshes all the properties so that they pertain to this current episode of stunting
     * Imposes the symptoms
     * Schedules relevant natural history event {(ProgressionSevereStuntingEvent) and
       (either StuntingRecoveryEvent or StuntingDeathEvent)}
    """

    AGE_GROUPS = {0: '0y', 1: '1y', 2: '2y', 3: '3y', 4: '4y'}

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters
        rng = m.rng

        df.at[person_id, 'un_ever_stunted'] = True
        df.at[person_id, 'un_HAZ_category'] = '-3<=HAZ<-2'  # start as moderate stunting
        df.at[person_id, 'un_last_stunting_date_of_onset'] = self.sim.date

        # -------------------------------------------------------------------------------------------
        # Add this incident case to the tracker
        stunting_state = df.at[person_id, 'un_HAZ_category']
        age_group = StuntingOnsetEvent.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
        m.stunting_incident_case_tracker[age_group][stunting_state].append(self.sim.date)
        # -------------------------------------------------------------------------------------------

        # determine if this person will progress to severe stunting # # # # # # # # # # #
        progression_to_severe_stunting = self.module.severe_stunting_progression_equation.predict(
            df.loc[[person_id]]).values[0]

        # progression_to_severe_stunting = self.module.severe_stunting_progression_equation.predict(
        #     df.loc[[person_id]],
        #     previous_diarrhoea_episodes=self.module.count_all_previous_diarrhoea_episodes(
        #         today=self.sim.date, index=df.loc[[person_id]].index))

        if rng.rand() < progression_to_severe_stunting:
            # Allocate a date of onset for stunting episode
            date_onset_severe_stunting = self.sim.date + DateOffset(months=3)

            # Create the event for the onset of severe stunting
            self.sim.schedule_event(
                event=ProgressionSevereStuntingEvent(module=self.module,
                                                     person_id=person_id), date=date_onset_severe_stunting
            )

        else:
            # determine if this person will improve stunting state without interventions # # # # # # # # # # #
            improved_stunting_state = 1 - p['prob_remained_stunted_in_the_next_3months']
            if rng.rand() < improved_stunting_state:
                # Allocate a date of onset for improvement of stunting episode
                date_recovery_stunting = self.sim.date + DateOffset(months=3)

                # Create the event for the onset of stunting recovery by 1sd in HAZ
                self.sim.schedule_event(
                    event=StuntingRecoveryEvent(module=self.module,
                                                person_id=person_id), date=date_recovery_stunting)


class ProgressionSevereStuntingEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of severe stunting (with HAZ <-3).
     * Refreshes all the properties so that they pertain to this current episode of stunting
     * Imposes the symptoms
     * Schedules relevant natural history event {(either WastingRecoveryEvent or WastingDeathEvent)}
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

        # before progression to severe stunting, check those who started complementary feeding interventions
        if df.at[person_id,
                 'un_last_stunting_date_of_onset'] < df.at[person_id,
                                                           'un_stunting_tx_start_date'] < self.sim.date:
            return

        # update properties
        df.at[person_id, 'un_HAZ_category'] = 'HAZ<-3'

        # -------------------------------------------------------------------------------------------
        # Add this incident case to the tracker
        stunting_state = df.at[person_id, 'un_HAZ_category']
        age_group = StuntingOnsetEvent.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
        m.stunting_incident_case_tracker[age_group][stunting_state].append(self.sim.date)
        # -------------------------------------------------------------------------------------------

        # determine if this person will improve stunting state # # # # # # # # # # #
        improved_stunting_state = 1 - p['prob_remained_stunted_in_the_next_3months']
        if rng.rand() < improved_stunting_state:
            # Allocate a date of onset for improvement of stunting episode
            date_recovery_stunting = self.sim.date + DateOffset(months=3)

            # Create the event for the onset of stunting recovery by 1sd in HAZ
            self.sim.schedule_event(
                event=StuntingRecoveryEvent(module=self.module,
                                            person_id=person_id), date=date_recovery_stunting)


class StuntingRecoveryEvent(Event, IndividualScopeEventMixin):
    """
    This event sets the properties back to normal state
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        if not df.at[person_id, 'is_alive']:
            return

        if df.at[person_id, 'un_HAZ_category'] == '-3<=HAZ<-2':
            df.at[person_id, 'un_HAZ_category'] = 'HAZ>=-2'
            df.at[person_id, 'un_stunting_recovery_date'] = self.sim.date
        if df.at[person_id, 'un_HAZ_category'] == 'HAZ<-3':
            df.at[person_id, 'un_HAZ_category'] = '-3<=HAZ<-2'


class HSI_complementary_feeding_education_only(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI is for education (only) of complementary feeding / without provision of supplementary foods
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Stunting)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint['GrowthMon'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'complementary_feeding_education_only'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        # Stop the person from dying of acute malnutrition (if they were going to die)
        if not df.at[person_id, 'is_alive']:
            return

        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~
        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # individual items
        item_code_complementary_feeding_education = pd.unique(
            consumables.loc[consumables['Items'] ==
                            'Complementary feeding--education only drugs/supplies to service a client', 'Item_Code'])[0]

        # check availability of consumables
        if self.get_all_consumables(item_codes=item_code_complementary_feeding_education):
            logger.debug(key='debug', data='item_code_complementary_feeding_education is available, so use it.')
            # Update properties
            df.at[person_id, 'un_stunting_tx_start_date'] = self.sim.date
            df.at[person_id, 'un_cm_treatment_type'] = 'education_on_complementary_feeding'
            self.module.do_when_cm_treatment(person_id)
        else:
            logger.debug(key='debug', data="item_code_complementary_feeding_education is not available, "
                                           "so can't use it.")

        # --------------------------------------------------------------------------------------------------
        # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        actual_appt_footprint['GrowthMon'] = actual_appt_footprint['GrowthMon'] * 2
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug("HSI_complementary_feeding_education_only: did not run")
        pass


class HSI_complementary_feeding_with_supplementary_foods(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI is for complementary feeding with provision of supplementary foods
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Stunting)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint['GrowthMon'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'complementary_feeding_with_supplementary_foods'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        # Stop the person from dying of acute malnutrition (if they were going to die)
        if not df.at[person_id, 'is_alive']:
            return

        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~
        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # individual items
        item_code_complementary_feeding_with_supplements = pd.unique(
            consumables.loc[consumables['Items'] ==
                            'Supplementary spread, sachet 92g/CAR-150', 'Item_Code'])[0]

        # check availability of consumables
        if self.get_all_consumables(item_codes=item_code_complementary_feeding_with_supplements):
            logger.debug(key='debug', data='item_code_complementary_feeding_with_supplements is available, so use it.')
            # Update properties
            df.at[person_id, 'un_stunting_tx_start_date'] = self.sim.date
            df.at[person_id, 'un_cm_treatment_type'] = 'complementary_feeding_with_food_supplementation'
            self.module.do_when_cm_treatment(person_id)
        else:
            logger.debug(key='debug', data="item_code_complementary_feeding_with_supplements is not available, "
                                           "so can't use it.")

        # --------------------------------------------------------------------------------------------------
        # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        actual_appt_footprint['GrowthMon'] = actual_appt_footprint['GrowthMon'] * 2
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug("HSI_complementary_feeding_with_supplementary_foods: did not run")
        pass


class StuntingPropertiesOfOtherModules(Module):
    """For the purpose of the testing, this module generates the properties upon which the Stunting module relies"""

    ALTERNATE_TO = {'Hiv', 'NewbornOutcomes'}

    PROPERTIES = {
        'hv_inf': Property(Types.BOOL, 'temporary property'),
        'hv_art': Property(Types.CATEGORICAL, 'temporary property',
                           categories=['not', 'on_VL_suppressed', 'on_not_VL_suppressed']),
        'nb_low_birth_weight_status': Property(Types.CATEGORICAL, 'temporary property',
                                               categories=['extremely_low_birth_weight', 'very_low_birth_weight',
                                                           'low_birth_weight', 'normal_birth_weight']),
        'nb_size_for_gestational_age': Property(Types.CATEGORICAL, 'temporary property',
                                                categories=['small_for_gestational_age',
                                                            'average_for_gestational_age']),
        'nb_late_preterm': Property(Types.BOOL, 'temporary property'),
        'nb_early_preterm': Property(Types.BOOL, 'temporary property'),

        'nb_breastfeeding_status': Property(Types.CATEGORICAL, 'temporary property',
                                            categories=['none', 'non_exclusive', 'exclusive']),

    }

    def __init__(self, name=None):
        super().__init__(name)

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        df = population.props
        df.loc[df.is_alive, 'hv_inf'] = False
        df.loc[df.is_alive, 'hv_art'] = 'not'
        df.loc[df.is_alive, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.loc[df.is_alive, 'nb_size_for_gestational_age'] = 'average_for_gestational_age'
        df.loc[df.is_alive, 'nb_late_preterm'] = False
        df.loc[df.is_alive, 'nb_early_preterm'] = False
        df.loc[df.is_alive, 'nb_breastfeeding_status'] = 'exclusive'

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        df = self.sim.population.props
        df.at[child, 'hv_inf'] = False
        df.at[child, 'hv_art'] = 'not'
        df.at[child, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.at[child, 'nb_size_for_gestational_age'] = 'average_for_gestational_age'
        df.at[child, 'nb_late_preterm'] = False
        df.at[child, 'nb_early_preterm'] = False
        df.at[child, 'nb_breastfeeding_status'] = 'exclusive'
