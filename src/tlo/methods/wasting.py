"""
Childhood wasting module
Documentation: '04 - Methods Repository/Undernutrition module - Description.docx'

Overview
=======
This module applies the prevalence of wasting at the population-level, and schedules new incidences of wasting

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
from tlo.util import create_age_range_lookup

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
    - moderate wasting: height-for-age Z-score (WHZ) <-2 SD from the reference mean
    - severe wasting: height-for-age Z-score (WHZ) <-3 SD from the reference mean

    """

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    wasting_states = ['MAM', 'SAM']

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
        'or_wasting_motherBMI_underweight': Parameter(
            Types.REAL, 'odds ratio of wasting if mother has low BMI, ref group high BMI (overweight)'),
        'or_wasting_hhwealth_Q5': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is poorest Q5, ref group Q1'),
        'or_wasting_hhwealth_Q4': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is poorer Q4, ref group Q1'),
        'or_wasting_hhwealth_Q3': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is middle Q3, ref group Q1'),
        'or_wasting_hhwealth_Q2': Parameter(
            Types.REAL, 'odds ratio of wasting if household wealth is richer Q2, ref group Q1'),
        'base_inc_rate_MAM_by_agegp': Parameter(
            Types.LIST, 'List with baseline incidence of wasting by age group'),
        'rr_MAM_preterm_and_AGA': Parameter(
            Types.REAL, 'relative risk of wasting if born preterm and adequate for gestational age'),
        'rr_MAM_SGA_and_term': Parameter(
            Types.REAL, 'relative risk of wasting if born term and small for geatational age'),
        'rr_MAM_SGA_and_preterm': Parameter(
            Types.REAL, 'relative risk of wasting if born preterm and small for gestational age'),
        'min_days_duration_of_wasting': Parameter(
            Types.REAL, 'minimum duration in days of wasting (MAM and SAM)'),
        'average_duration_of_untreated_MAM': Parameter(
            Types.REAL, 'average duration of untreated MAM'),
        'prob_progress_to_SAM_without_tx': Parameter(
            Types.REAL, 'probability of MAM progressing to SAM without treatment'),
        'average_duration_of_untreated_SAM': Parameter(
            Types.REAL, 'average duration of untreated SAM'),
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
        'un_WHZ_score': Property(Types.REAL, 'height-for-age z-score'),
        'un_ever_wasted': Property(Types.BOOL, 'had wasting before WHZ <-2'),
        'un_WHZ_category': Property(Types.CATEGORICAL, 'height-for-age z-score group',
                                    categories=['WHZ<-3', '-3<=WHZ<-2', 'WHZ>=-2']),
        'un_clinical_acute_malnutrition': Property(Types.CATEGORICAL, 'clinical acute malnutrition state based on WHZ',
                                                   categories=['MAM', 'SAM']),
        'un_wasting_death_date': Property(Types.DATE, 'death date from wasting'),

        'un_wasting_oedema': Property(Types.BOOL, 'oedema present in wasting'),
        'un_wasting_MUAC_measure': Property(Types.REAL, 'MUAC measurement'),
        'un_AM_treatment_type': Property(Types.CATEGORICAL, 'treatment types for of acute malnutrition',
                                         categories=['standard_RUTF', 'soy_RUSF', 'CSB++', 'inpatient_care']),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # set the linear model equations for prevalence and incidence
        self.prevalence_equations_by_age = dict()
        self.wasting_incidence_equation = dict()

        # dict to hold the probability of onset of different symptoms:
        self.prob_symptoms = dict()

        # Store the symptoms that this module will use:
        self.symptoms = {
            'palmar_pallor',
            'oedema',
            'lack_of_appetite',
            'lethargic',
            'dehydration'
        }

        # set the linear model for recovery
        self.uncomplicated_acute_malnutrition_recovery_rate = dict()
        self.acute_malnutrition_with_complications_recovery_rate = dict()

        # dict to hold counters for the number of episodes by wasting-type and age-group
        blank_counter = dict(zip(self.wasting_states, [list() for _ in self.wasting_states]))
        self.wasting_incident_case_tracker_blank = {
            '0y': copy.deepcopy(blank_counter),
            '1y': copy.deepcopy(blank_counter),
            '2y': copy.deepcopy(blank_counter),
            '3y': copy.deepcopy(blank_counter),
            '4y': copy.deepcopy(blank_counter),
            '5+y': copy.deepcopy(blank_counter)
        }
        self.wasting_incident_case_tracker = copy.deepcopy(self.wasting_incident_case_tracker_blank)

        zeros_counter = dict(zip(self.wasting_states, [0] * len(self.wasting_states)))
        self.wasting_incident_case_tracker_zeros = {
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
        # Read parameters from the resourcefile
        self.load_parameters_from_dataframe(
            pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Undernutrition.xlsx',
                          sheet_name='Parameter_values_AM'))
        p = self.parameters

        # Check that every value has been read-in successfully
        for param_name, param_type in self.PARAMETERS.items():
            assert param_name in p, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert isinstance(p[param_name],
                              param_type.python_type), f'Parameter "{param_name}" is not read in correctly from the ' \
                                                       f'resourcefile.'

        # Declare symptoms that this module will cause and which are not included in the generic symptoms:
        generic_symptoms = self.sim.modules['SymptomManager'].parameters['generic_symptoms']
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

            def make_linear_model_wasting(agegp, intercept=get_odds_wasting(agegp=agegp)):
                return LinearModel(
                    LinearModelType.LOGISTIC,
                    intercept,  # baseline odds: get_odds_wasting(agegp=agegp)
                    Predictor('li_bmi').when(1, p['or_wasting_motherBMI_underweight']),
                    Predictor('li_wealth').when(2, p['or_wasting_hhwealth_Q2'])
                        .when(3, p['or_wasting_hhwealth_Q3'])
                        .when(4, p['or_wasting_hhwealth_Q4'])
                        .when(5, p['or_wasting_hhwealth_Q5']),
                )

            unscaled_lm = make_linear_model_wasting(agegp, intercept=get_odds_wasting(agegp=agegp))
            target_mean = get_odds_wasting(agegp='12_23mo')
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 1)]).mean()
            scaled_intercept = get_odds_wasting(agegp) * (target_mean / actual_mean)
            scaled_lm = make_linear_model_wasting(agegp, intercept=scaled_intercept)
            return scaled_lm

        # the linear model returns the probability that is implied by the model prob = odds / (1 + odds)
        for agegp in ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo']:
            self.prevalence_equations_by_age[agegp] = make_scaled_linear_model_wasting(agegp)

        # get the initial prevalence values for each age group using the lm equation (scaled)
        prevalence_of_wasting = pd.DataFrame(index=df.loc[df.is_alive & (df.age_exact_years < 5)].index)

        prevalence_of_wasting['0_5mo'] = self.prevalence_equations_by_age['0_5mo'] \
            .predict(df.loc[df.is_alive & (df.age_exact_years < 0.5)])
        prevalence_of_wasting['6_11mo'] = self.prevalence_equations_by_age['6_11mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 0.5) & (df.age_exact_years < 1))])
        prevalence_of_wasting['12_23mo'] = self.prevalence_equations_by_age['12_23mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 1) & (df.age_exact_years < 2))])
        prevalence_of_wasting['24_35mo'] = self.prevalence_equations_by_age['24_35mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 2) & (df.age_exact_years < 3))])
        prevalence_of_wasting['36_47mo'] = self.prevalence_equations_by_age['36_47mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 3) & (df.age_exact_years < 4))])
        prevalence_of_wasting['48_59mo'] = self.prevalence_equations_by_age['48_59mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 4) & (df.age_exact_years < 5))])

        # further categorize into MAM or SAM
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

            # get moderate wasting zcores: <=-3 WHZ <-2
            probability_between_minus3_minus2sd = \
                probability_over_or_equal_minus3sd - probability_over_or_equal_minus2sd

            # make WHZ <-2 as the 100% and get the adjusted probability of severe wasting within overall wasting
            proportion_severe_in_overall_wasting = probability_less_than_minus3sd * probability_less_than_minus2sd

            # get the probability of severe wasting
            return proportion_severe_in_overall_wasting

        # further differentiate between severe wasting and moderate wasting, and normal WHZ
        for agegp in ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo']:
            wasted = self.rng.random_sample(len(prevalence_of_wasting[agegp])) < prevalence_of_wasting[agegp]
            for id in wasted[wasted].index:
                probability_of_severe = get_prob_severe_in_overall_wasting(agegp)
                wasted_category = self.rng.choice(['WHZ<-3', '-3<=WHZ<-2'],
                                                  p=[probability_of_severe, 1 - probability_of_severe])
                df.at[id, 'un_WHZ_category'] = wasted_category
            df.loc[wasted[wasted == False].index, 'un_WHZ_category'] = 'WHZ>=-2'

    def initialise_simulation(self, sim):
        """Prepares for simulation:
        * Schedules the main polling event
        * Schedules the main logging event
        * Establishes the incidence linear models and other data structures using the parameters that have been read-in
        * Store the consumables that are required in each of the HSI
        """
        df = self.sim.population.props
        p = self.parameters

        event = WastingPollingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=6))

        event = WastingLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=12))

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
                    Predictor('age_exact_years')
                        .when('<0.5', p['base_inc_rate_MAM_by_agegp'][0])
                        .when('.between(0.5,0.9999)', p['base_inc_rate_MAM_by_agegp'][1]),
                    Predictor('age_years')
                        .when('.between(1,1)', p['base_inc_rate_MAM_by_agegp'][2])
                        .when('.between(2,2)', p['base_inc_rate_MAM_by_agegp'][3])
                        .when('.between(3,3)', p['base_inc_rate_MAM_by_agegp'][4])
                        .when('.between(4,4)', p['base_inc_rate_MAM_by_agegp'][5]),
                    Predictor('nb_size_for_gestational_age').when('small_for_gestational_age',
                                                                  p['rr_MAM_SGA_and_term']),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& (nb_late_preterm == False) & (nb_early_preterm == False)',
                                     p['rr_MAM_SGA_and_term']),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                                     p['rr_MAM_SGA_and_preterm']),
                    Predictor().when('(nb_size_for_gestational_age == "average_for_gestational_age") '
                                     '& (nb_late_preterm == True) | (nb_early_preterm == True)',
                                     p['rr_MAM_preterm_and_AGA'])
                )

            unscaled_lm = make_lm_wasting_incidence()
            target_mean = p[f'base_inc_rate_MAM_by_agegp'][2]
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 1)]).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean)
            scaled_lm = make_lm_wasting_incidence(intercept=scaled_intercept)
            return scaled_lm

        self.wasting_incidence_equation = make_scaled_lm_wasting_incidence()

        # --------------------------------------------------------------------------------------------
        # Make a linear model equations that govern the probability of recovery following treatment
        # lm with probability of recovery following treatment for uncomplicated cases
        self.uncomplicated_acute_malnutrition_recovery_rate.update({
            'MAM':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('un_AM_treatment_type').when('soy_RUSF', p['recovery_rate_with_soy_RUSF'])
                            .otherwise(0.0),
                            Predictor('un_AM_treatment_type').when('CSB++', p['recovery_rate_with_CSB++'])
                            .otherwise(0.0),
                            ),
            'SAM':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('un_AM_treatment_type')
                            .when('standard_RUTF', p['recovery_rate_with_standard_RUTF'])
                            .otherwise(0.0),
                            )
        })

        # lm with probability of recovery following treatment for complicated cases
        self.acute_malnutrition_with_complications_recovery_rate.update({
            'SAM':
                LinearModel(LinearModelType.MULTIPLICATIVE,
                            1.0,
                            Predictor('un_AM_treatment_type').when('inpatient_care',
                                                                    p['recovery_rate_with_inpatient_care'])
                            .otherwise(0.0),

                            ),
        })

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'un_WHZ_category'] = 'WHZ>=-2'

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
        total_daly_values.loc[df.is_alive & (df.un_WHZ_category == 'WHZ<-3') &
                              (df.un_wasting_oedema == True)] = self.daly_wts['SAM_with_oedema']
        total_daly_values.loc[df.is_alive & (df.un_WHZ_category == 'WHZ<-3') &
                              (df.un_wasting_oedema == False)] = self.daly_wts['SAM_w/o_oedema']
        total_daly_values.loc[df.is_alive & (df.un_WHZ_category == '-3<=WHZ<-2') &
                              (df.un_wasting_oedema == True)] = self.daly_wts['MAM_with_oedema']
        # total_daly_values.loc[df.is_alive & (df.un_WHZ_category == '-3<=WHZ<-2') &
        #                       (df.un_wasting_oedema == False)] = self.daly_wts['MAM_w/o_oedema']

        return total_daly_values

    def wasting_clinical_symptoms(self, population, clinical_index):
        """
        assign clinical symptoms to new acute malnutrition cases

        :param population:
        :param clinical_index:
        """
        df = population
        p = self.parameters
        rng = self.rng
        now = self.sim.date

        # currently symptoms list is applied to all
        for symptom in self.symptoms:
            # this also schedules symptom resolution in 5 days
            self.sim.modules["SymptomManager"].change_symptom(
                person_id=list(clinical_index),
                symptom_string=symptom,
                add_or_remove="+",
                disease_module=self,
                duration_in_days=None,
            )

    # def severe_symptoms(self, population, severe_index, child=False):
    #     """assign clinical symptoms to new severe malaria cases. Symptoms can only be resolved by treatment
    #     handles both adult and child (using the child parameter) symptoms
    #
    #     :param population: the population dataframe
    #     :param severe_index: the indices of new clinical cases
    #     :param child: to apply severe symptoms to children (otherwise applied to adults)
    #     """
    #     df = population
    #     p = self.parameters
    #     rng = self.rng
    #     now = self.sim.date
    #
    #     # currently symptoms list is applied to all
    #     for symptom in self.symptoms:
    #         # this also schedules symptom resolution in 5 days
    #         self.sim.modules["SymptomManager"].change_symptom(
    #             person_id=list(severe_index),
    #             symptom_string=symptom,
    #             add_or_remove="+",
    #             disease_module=self,
    #             duration_in_days=None,
    #         )

    def do_wasting_treatment(self, person_id, wasting_severity, treatment_id):
        """Helper function that enacts the effects of a treatment to acute malnutrition.
        * Log the treatment date
        * Prevents this episode of wasting from causing a death
        * Schedules the cure event, at which symptoms are alleviated.
        """
        df = self.sim.population.props
        p = self.module.parameters
        person = df.loc[person_id]
        if not person.is_alive:
            return

        # Log that the treatment is provided:
        df.at[person_id, 'un_wasting_tx_start_date'] = self.sim.date

        # Determine the outcome of treatment
        recovery_from_wasting = self.module.uncomplicated_acute_malnutrition_recovery_rate[wasting_severity].predict(
            df.loc[person_id])
        if recovery_from_wasting > self.rng.rand():
            # If treatment is successful: cancel death and schedule cure event
            self.cancel_death_date(person_id)
            self.sim.schedule_event(WastingRecoveryEvent(self, person_id),
                                    self.sim.date + DateOffset(
                                        days=self.parameters['days_between_treatment_and_cure']
                                    ))
        # else:  # not improving seek care or death
        #     self.do_when_not_improving(person_id)

    def cancel_death_date(self, person_id):
        """
        Cancels a scheduled date of death due to wasting for a person. This is called prior to the scheduling the
        CureEvent to prevent deaths happening in the time between a treatment being given and the cure event occurring.

        :param person_id:
        :return:
        """
        df = self.sim.population.props
        df.at[person_id, 'un_wasting_death_date'] = pd.NaT


class WastingPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all wasting properties for the population
    """

    def __init__(self, module):
        """schedule to run every 6 months
        :param module: the module that created this event
        """
        self.repeat_months = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat_months))
        assert isinstance(module, Wasting)

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng
        p = m.parameters

        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')

        # Determine who will be onset with wasting among those who are not currently wasted
        incidence_of_wasting = self.module.wasting_incidence_equation.predict(
            df.loc[df.is_alive & (df.age_exact_years < 5)])
        wasted = rng.random_sample(len(incidence_of_wasting)) < incidence_of_wasting

        # update clinical symptoms for all new clinical infections
        self.module.wasting_clinical_symptoms(df, wasted[wasted].index)

        # determine the time of onset and other disease characteristics for each individual
        for person_id in wasted[wasted].index:
            # Allocate a date of onset for wasting episode
            date_onset = self.sim.date + DateOffset(days=rng.randint(0, days_until_next_polling_event))

            # Create the event for the onset of wasting
            self.sim.schedule_event(
                event=WastingOnsetEvent(module=self.module,
                                        person_id=person_id,
                                        am_state='MAM'), date=date_onset
            )


class WastingOnsetEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of wasting (MAM with WHZ <-2).
     * Refreshes all the properties so that they pertain to this current episode of wasting
     * Imposes the symptoms
     * Schedules relevant natural history event {(ProgressionSAMEvent) and
       (either WastingRecoveryEvent or WastingDeathEvent)}
    """

    AGE_GROUPS = {0: '0y', 1: '1y', 2: '2y', 3: '3y', 4: '4y'}

    def __init__(self, module, person_id, am_state):
        super().__init__(module, person_id=person_id)
        self.wasting_state = am_state

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters
        rng = m.rng

        df.at[person_id, 'un_ever_wasted'] = True
        df.at[person_id, 'un_currently_wasted'] = True
        df.at[person_id, 'un_WHZ_category'] = '-3<=WHZ<-2'  # start as MAM
        df.at[person_id, 'un_clinical_acute_malnutrition'] = self.wasting_state

        # Allocate the duration of the wasting episode (as MAM)
        duration_in_days = int(max(p['min_days_duration_of_wasting'], p['average_duration_of_untreated_MAM']))

        # Determine those that will progress to SAM -----------------------------------------------

        # after reaching the last day of the total duration of MAM, the child can progress to SAM,
        # naturally recover or die (due to other causes). Here we just allocate them into recovery or SAM
        outcome_from_MAM = rng.choice(['SAM', 'recovery'],
                                      p=[p['prob_progress_to_SAM_without_tx'],
                                         1 - p['prob_progress_to_SAM_without_tx']])
        if outcome_from_MAM == 'SAM':
            # schedule SAM onset
            self.sim.schedule_event(
                event=ProgressionSevereWastingEvent(module=self.module, person_id=person_id, am_state='SAM'),
                date=self.sim.date + DateOffset(duration_in_days)
            )
        else:
            # schedule natural recovery
            self.sim.schedule_event(
                event=WastingRecoveryEvent(module=self.module, person_id=person_id),
                date=self.sim.date + DateOffset(duration_in_days)
            )

        # -------------------------------------------------------------------------------------------
        # Add this incident case to the tracker
        age_group = WastingOnsetEvent.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
        m.wasting_incident_case_tracker[age_group][self.wasting_state].append(self.sim.date)


class ProgressionSevereWastingEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of wasting (SAM with WHZ <-3).
     * Refreshes all the properties so that they pertain to this current episode of wasting
     * Imposes the symptoms
     * Schedules relevant natural history event {(either WastingRecoveryEvent or WastingDeathEvent)}
    """

    def __init__(self, module, person_id, am_state):
        super().__init__(module, person_id=person_id)
        self.wasting_state = am_state

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters
        rng = m.rng

        # update properties
        df.at[person_id, 'un_WHZ_category'] = 'WHZ<-3'
        df.at[person_id, 'un_clinical_acute_malnutrition'] = self.wasting_state

        # determine the duration of SAM episode
        duration_in_days = int(max(p['min_days_duration_of_wasting'], p['average_duration_of_untreated_SAM']))

        # Determine progression to death or natural recovery
        outcome_from_SAM = rng.choice(['death', 'recovery'], p=[p['prob_progress_to_SAM_without_tx'],
                                                                1 - p['prob_progress_to_SAM_without_tx']])
        if outcome_from_SAM == 'death':
            # schedule death event
            self.sim.schedule_event(
                event=SevereWastingDeathEvent(module=self.module, person_id=person_id),
                date=self.sim.date + DateOffset(duration_in_days)
            )
        else:
            # schedule natural recovery
            self.sim.schedule_event(
                event=WastingRecoveryEvent(module=self.module, person_id=person_id),
                date=self.sim.date + DateOffset(duration_in_days)
            )  # TODO: add improvement to MAM before full recovery

        # -------------------------------------------------------------------------------------------
        # Add this incident case to the tracker
        age_group = WastingOnsetEvent.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
        m.wasting_incident_case_tracker[age_group][self.wasting_state].append(self.sim.date)


class SevereWastingDeathEvent(Event, IndividualScopeEventMixin):
    """
    This event applies the death function
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters
        rng = m.rng

        # Implement the death:
        self.sim.schedule_event(
            demography.InstantaneousDeath(
                self.module,
                person_id,
                cause='Wasting'
            ),
            self.sim.date)


class WastingRecoveryEvent(Event, IndividualScopeEventMixin):
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

        df.at[person_id, 'un_WHZ_category'] = 'WHZ>=-2'  # not undernourished
        df.at[person_id, 'un_clinical_acute_malnutrition'] = None
        df.at[person_id, 'un_wasting_death_date'] = pd.NaT
        df.at[person_id, 'un_wasting_oedema'] = False

        # this will clear all wasting symptoms
        self.sim.modules["SymptomManager"].clear_symptoms(
            person_id=person_id, disease_module=self.module
        )


class HSI_uncomplicated_MAM_treatment(HSI_Event, IndividualScopeEventMixin):
    """
    this is the treatment for moderate acute malnutrition without complications
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Wasting)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'uncomplicated_MAM_treatment'
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
        # whole package of interventions
        pkg_code_mam = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Management of moderate acute malnutrition (children)',
                            'Intervention_Pkg_Code'])[0]  # This package includes only CSB(or supercereal or CSB++)
        # individual items
        item_code1 = pd.unique(
            consumables.loc[consumables['Items'] == 'Corn Soya Blend (or Supercereal - CSB++)', 'Item_Code'])[0]

        consumables_needed = {'Intervention_Package_Code': {pkg_code_mam: 1}, 'Item_Code': {item_code1: 1}}

        # check availability of consumables
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed)
        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_mam]:
            logger.debug(key='debug', data='PkgCode1 is available, so use it.')
            self.module.do_wasting_treatment(
                person_id=person_id,
                wasting_severity='MAM',
                treatment_id='CSB++'
            )
        else:
            logger.debug(key='debug', data="PkgCode1 is not available, so can't use it.")

        # --------------------------------------------------------------------------------------------------
        # check to see if all consumables returned (for demonstration purposes):
        all_available = (outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_mam]) and \
                        (outcome_of_request_for_consumables['Item_Code'][item_code1])
        # use helper function instead (for demonstration purposes)
        all_available_using_helper_function = self.get_all_consumables(
            item_codes=[item_code1],
            pkg_codes=[pkg_code_mam]
        )
        # Demonstrate equivalence
        assert all_available == all_available_using_helper_function
        # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug("HSI_Malaria_tx_compl_adult: did not run")
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
        # Convert the list of timestamps into a number of timestamps
        # and check that all the dates have occurred since self.date_last_run
        counts_am = copy.deepcopy(self.module.wasting_incident_case_tracker_zeros)

        for age_grp in self.module.wasting_incident_case_tracker.keys():
            for state in self.module.wasting_states:
                list_of_times = self.module.wasting_incident_case_tracker[age_grp][state]
                counts_am[age_grp][state] = len(list_of_times)
                for t in list_of_times:
                    assert self.date_last_run <= t <= self.sim.date

        logger.info(key='wasting_incidence_count', data=counts_am)

        # Reset the counters and the date_last_run
        self.module.wasting_incident_case_tracker = copy.deepcopy(self.module.wasting_incident_case_tracker_blank)
        self.date_last_run = self.sim.date


