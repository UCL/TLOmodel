"""
Stunting Module

Overview
--------
The Stunting module determines the prevalence of stunting for children under 5 years old. A polling event runs
every month and determines the risk of onset of non-severe stunting, progression to severe stunting or natural
recovery. The Generic HSI calls `do_routine_assessment_for_chronic_undernutrition` for any HSI with a child under
5 years old: if they have any stunting they are provided with an intervention - `HSI_Stunting_ComplementaryFeeding`.

"""

from collections import namedtuple
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITION
# ---------------------------------------------------------------------------------------------------------

class Stunting(Module):
    """This is the disease module for Stunting"""

    INIT_DEPENDENCIES = {'Demography', 'Wasting', 'NewbornOutcomes', 'Diarrhoea', 'Hiv'}

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_HEALTHSYSTEM,
    }

    PARAMETERS = {
        # Prevalence of stunting by age group at initiation
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
            Types.LIST,
            'Baseline incidence rate per year of stunting by age group (1-5, 6-11, 12-23, 24-35, 36-47, 48-59mo'),
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
            Types.REAL, 'Relative risk of stunting if wealth-level is greater than 1 compared 1'),
        'rr_stunting_no_exclusive_breastfeeding': Parameter(
            Types.REAL, 'Relative risk of stunting for not exclusively breastfed babies < 6 months'),
        'rr_stunting_no_continued_breastfeeding': Parameter(
            Types.REAL, 'Relative risk of stunting for not continued breastfed infants 6-24 months'),
        'rr_stunting_per_diarrhoeal_episode': Parameter(
            Types.REAL, 'Relative risk of stunting for recent diarrhoea episode'),

        # Progression to severe stunting
        'r_progression_severe_stunting_by_agegp': Parameter(
            Types.LIST,
            'Rates per year of progression to severe stunting by age group (1-5, 6-11, 12-23, 24-35, 36-47, 48-59mo'),
        'rr_progress_severe_stunting_if_prior_wasting': Parameter(
            Types.REAL, 'Relative risk of severe stunting if previously wasted'),
        'rr_progress_severe_stunting_untreated_HIV': Parameter(
            Types.REAL, 'Relative risk of severe stunting for untreated HIV+'),

        # Natural recovery from stunting
        'mean_years_to_1stdev_natural_improvement_in_stunting': Parameter(
            Types.REAL,
            'Mean time (in years) to a one standard deviation improvement in stunting without any treatment.'),

        # The effect of treatment
        'effectiveness_of_complementary_feeding_education_in_stunting_reduction': Parameter(
            Types.REAL,
            'Probability of stunting being reduced by one standard deviation (category) by education about '
            'supplementary feeding (but not supplying supplementary feeding consmables).'),
        'effectiveness_of_food_supplementation_in_stunting_reduction': Parameter(
            Types.REAL,
            'Probability of stunting being reduced by one standard deviation (category) by supplementary feeding.'),

        # The probability of a (severe) stunting person being checked and correctly diagnosed
        'prob_stunting_diagnosed_at_generic_appt': Parameter(
            Types.REAL,
            'Probability of a stunted or severely stunted person being checked and correctly diagnosed'),
    }

    PROPERTIES = {
        'un_HAZ_category': Property(Types.CATEGORICAL,
                                    'Indicator of current stunting status - the height-for-age z-score category:'
                                    '"HAZ>=-2" == No Stunting (within 2 standard deviations of mean); '
                                    '"-3<=HAZ<-2" == Non-Severe Stunting (2-3 standard deviations from mean); '
                                    '"HAZ<-3 == Severe Stunting (more than 3 standard deviations from mean)',
                                    categories=['HAZ<-3', '-3<=HAZ<-2', 'HAZ>=-2']),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.models = None  # (Will store the models used in the module)
        self.cons_item_codes = None  # (Will store consumable item codes)

    def read_parameters(self, data_folder):
        self.load_parameters_from_dataframe(
            pd.read_excel(
                Path(self.resourcefilepath) / 'ResourceFile_Stunting.xlsx', sheet_name='Parameter_values')
        )

    def initialise_population(self, population):
        """Set initial prevalence of stunting according to distributions provided in parameters"""
        df = population.props
        p = self.parameters

        # Set default for property
        df.loc[df.is_alive, 'un_HAZ_category'] = 'HAZ>=-2'

        def get_probs_stunting(_agegp):
            """For the a given HAZ distribution (specified in the parameters by age-group), find the odds of
            a value <-2 (= 'stunted') and the odds of a value <-3 given a value <-2 (= 'severely stunted')."""

            mean, stdev = p[f'prev_HAZ_distribution_age_{_agegp[0]}_{_agegp[1]}mo']
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
                    Predictor('li_wealth').when(2, p['or_stunting_hhwealth_Q2'])
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

        for agegp in [(0, 5), (6, 11), (12, 23), (24, 35), (36, 47), (48, 59)]:  # in months
            p_stunting = get_probs_stunting(agegp)

            low_bound_age_in_years = agegp[0] / 12.0
            high_bound_age_in_years = (1 + agegp[1]) / 12.0

            mask = df.is_alive & df.age_exact_years.between(low_bound_age_in_years, high_bound_age_in_years,
                                                            inclusive='left')

            stunted = make_scaled_linear_model_stunting(target_prob=p_stunting.prob_stunting, mask=mask).predict(
                df.loc[mask], self.rng, squeeze_single_row_output=False)

            severely_stunted_idx = stunted.loc[
                stunted & (self.rng.random_sample(len(stunted)) < p_stunting.prob_severe_given_stunting)].index
            stunted_but_not_severe_idx = set(stunted[stunted].index) - set(severely_stunted_idx)

            df.loc[stunted_but_not_severe_idx, "un_HAZ_category"] = '-3<=HAZ<-2'
            df.loc[severely_stunted_idx, "un_HAZ_category"] = 'HAZ<-3'

    def initialise_simulation(self, sim):
        """Prepare for simulation to start"""
        # Establish the models
        self.models = Models(self)

        # Schedule the main polling event to begin on the first day of the simulation
        sim.schedule_event(StuntingPollingEvent(self), sim.date)

        # Schedule the logging event to begin on the first day of the simulation
        sim.schedule_event(StuntingLoggingEvent(self), sim.date)

        # Look-up consumable item codes
        self.look_up_consumable_item_codes()

    def on_birth(self, mother_id, child_id):
        """Set that on birth there is no stunting"""
        self.sim.population.props.loc[child_id, 'un_HAZ_category'] = 'HAZ>=-2'

    def look_up_consumable_item_codes(self):
        """Look up the item codes that used in the HSI in the module"""
        get_item_codes = self.sim.modules['HealthSystem'].get_item_code_from_item_name

        self.cons_item_codes = dict()
        self.cons_item_codes['supplementary_feeding'] = get_item_codes('Supplementary spread, sachet 92g/CAR-150')
        self.cons_item_codes['education_for_supplementary_feeding'] = get_item_codes(
            'Complementary feeding--education only drugs/supplies to service a client')

    def do_onset(self, idx: pd.Index):
        """Represent the onset of stunting for the person_id given in `idx`"""
        df = self.sim.population.props
        df.loc[idx, 'un_HAZ_category'] = '-3<=HAZ<-2'

    def do_progression(self, idx: pd.Index):
        """Represent the progression to severe stunting for the person_id given in `idx`"""
        df = self.sim.population.props
        df.loc[idx, 'un_HAZ_category'] = 'HAZ<-3'

    def do_recovery(self, idx: Union[list, pd.Index]):
        """Represent the recovery from stunting for the person_id given in `idx`. Recovery causes the person to move
        'up' one level: i.e. 'HAZ<-3' --> '-3<=HAZ<-2' or '-3<=HAZ<-2' --> 'HAZ>=-2'"""
        df = self.sim.population.props
        df.loc[idx, 'un_HAZ_category'] = df.loc[idx, 'un_HAZ_category'].map({
            'HAZ<-3': '-3<=HAZ<-2',
            '-3<=HAZ<-2': 'HAZ>=-2'
        })

    def do_routine_assessment_for_chronic_undernutrition(self, person_id):
        """This is called by the a generic HSI event for every child aged less than 5 years. It assesses stunting
        and schedules an HSI as needed."""

        df = self.sim.population.props
        person = df.loc[person_id]
        is_stunted = person.un_HAZ_category in ('HAZ<-3', '-3<=HAZ<-2')
        p_stunting_diagnosed = self.parameters['prob_stunting_diagnosed_at_generic_appt']

        if not is_stunted:
            return

        # Schedule the HSI for provision of treatment based on the probability of stunting diagnosis
        if p_stunting_diagnosed > self.rng.random_sample():
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Stunting_ComplementaryFeeding(module=self, person_id=person_id),
                priority=2,  # <-- lower priority that for wasting and most other HSI
                topen=self.sim.date)

    def do_treatment(self, person_id, prob_success):
        """Represent the treatment with supplementary feeding. If treatment is successful, effect the recovery
        of the person immediately."""
        if prob_success > self.rng.random_sample():
            self.do_recovery([person_id])


class Models:
    def __init__(self, module):
        self.module = module
        self.p = module.parameters

        self.lm_prob_becomes_stunted = self.make_lm_prob_becomes_stunted()
        self.lm_prob_progression_to_severe_stunting = self.make_lm_prob_progression_to_severe_stunting()
        self.lm_prob_natural_recovery = self.make_lm_prob_natural_recovery()

    def make_lm_prob_becomes_stunted(self):
        """Returns LinearModel for the probability per year of becoming stunted."""
        p = self.p

        return LinearModel.multiplicative(
            Predictor('age_exact_years',
                      conditions_are_exhaustive=True,
                      conditions_are_mutually_exclusive=True).when('< 0.5',
                                                                   p['base_inc_rate_stunting_by_agegp'][0])
                                                             .when('.between(0.5, 1.0, inclusive="left")',
                                                                   p['base_inc_rate_stunting_by_agegp'][1])
                                                             .when('.between(1.0, 2.0, inclusive="left")',
                                                                   p['base_inc_rate_stunting_by_agegp'][2])
                                                             .when('.between(2.0, 3.0, inclusive="left")',
                                                                   p['base_inc_rate_stunting_by_agegp'][3])
                                                             .when('.between(3.0, 4.0, inclusive="left")',
                                                                   p['base_inc_rate_stunting_by_agegp'][4])
                                                             .when('.between(4.0, 5.0, inclusive="left")',
                                                                   p['base_inc_rate_stunting_by_agegp'][5])
                                                             .when('> 5.0', 0.0),
            Predictor('li_wealth',
                      conditions_are_mutually_exclusive=True).when(1, 1.0)
                                                             .otherwise(p['rr_stunting_wealth_level']),
            Predictor('gi_number_of_episodes').apply(lambda x: p['rr_stunting_per_diarrhoeal_episode'] ** x),
            Predictor('un_ever_wasted',
                      conditions_are_exhaustive=True,
                      conditions_are_mutually_exclusive=True).when(True, p['rr_stunting_prior_wasting'])
                                                             .when(False, 1.0),
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
            Predictor().when('((nb_breastfeeding_status == "non_exclusive") | '
                             '(nb_breastfeeding_status == "none")) & (age_exact_years < 0.5)',
                             p['rr_stunting_no_exclusive_breastfeeding']),
            Predictor().when(
                '(nb_breastfeeding_status == "none") & (age_exact_years.between(0.5, 2.0, inclusive="left"))',
                p['rr_stunting_no_continued_breastfeeding']),
        )

    def make_lm_prob_progression_to_severe_stunting(self):
        """Return LinearModel for the probability per year of progressing from a state of being Stunted (but not severe)
        (-3 < HAZ <= -2) to a state of Severe Stunting (HAZ <= -3)"""
        p = self.p

        return LinearModel.multiplicative(
            Predictor('age_exact_years',
                      conditions_are_exhaustive=True,
                      conditions_are_mutually_exclusive=True).when('< 0.5',
                                                                   p['r_progression_severe_stunting_by_agegp'][0])
                                                             .when('.between(0.5, 1.0, inclusive="left")',
                                                                   p['r_progression_severe_stunting_by_agegp'][1])
                                                             .when('.between(1.0, 2.0, inclusive="left")',
                                                                   p['r_progression_severe_stunting_by_agegp'][2])
                                                             .when('.between(2.0, 3.0, inclusive="left")',
                                                                   p['r_progression_severe_stunting_by_agegp'][3])
                                                             .when('.between(3.0, 4.0, inclusive="left")',
                                                                   p['r_progression_severe_stunting_by_agegp'][4])
                                                             .when('.between(4.0, 5.0, inclusive="left")',
                                                                   p['r_progression_severe_stunting_by_agegp'][5])
                                                             .when('> 5.0', 1.0),
            Predictor('un_ever_wasted',
                      conditions_are_exhaustive=True,
                      conditions_are_mutually_exclusive=True
                      ).when(True, p['rr_progress_severe_stunting_if_prior_wasting'])
                       .when(False, 1.0),
            Predictor().when('(hv_inf == True) & (hv_art == "not")', p['rr_stunting_untreated_HIV']),
        )

    def make_lm_prob_natural_recovery(self):
        """Return LinearModel for the probability per year of improving by one category (one HAZ standard deviation)
        i.e. from being severely stunted (HAZ<-3) to non-severely stunted (-3<HAZ<=-2); and from being non-severely
        stunted to not Stunted (HAZ>-2)."""
        p = self.p

        mean_years = p['mean_years_to_1stdev_natural_improvement_in_stunting']
        prob_recovery_per_year = 1.0 - np.exp(-1.0 / mean_years)

        return LinearModel(
            LinearModelType.MULTIPLICATIVE,
            intercept=prob_recovery_per_year
        )


# ---------------------------------------------------------------------------------------------------------
#   NATURAL HISTORY EVENTS
# ---------------------------------------------------------------------------------------------------------

class StuntingPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """Regular event that controls the onset of stunting, progression to severe stunting and natural recovery."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, Stunting)

    def apply(self, population):
        """
        * Determine who will be onset for stunting (among those not stunted) and effect that change;
        * Determine who will be onset for recovery (among those stunted) and effect that change.
        * Determine who will be onset for progression (among those stunted but not severely) and effect that change;
        """
        df = population.props
        models = self.module.models
        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')

        # Onset of Stunting
        eligible_for_stunting = (df.is_alive &
                                 (df.age_exact_years < 5.0) &
                                 (df.un_HAZ_category == 'HAZ>=-2'))
        idx_will_be_stunted = self.apply_model(
            model=models.lm_prob_becomes_stunted,
            mask=eligible_for_stunting,
            days_exposed_to_risk=days_until_next_polling_event
        )
        self.module.do_onset(idx_will_be_stunted)

        # Recovery from Stunting
        eligible_for_recovery = (df.is_alive &
                                 (df.age_exact_years < 5.0) &
                                 (df.un_HAZ_category != 'HAZ>=-2') &
                                 ~df.index.isin(idx_will_be_stunted))
        idx_will_recover = self.apply_model(
            model=models.lm_prob_natural_recovery,
            mask=eligible_for_recovery,
            days_exposed_to_risk=days_until_next_polling_event
        )
        self.module.do_recovery(idx_will_recover)

        # Progression to Severe Stunting
        eligible_for_progression = (df.is_alive &
                                    (df.age_exact_years < 5.0) &
                                    (df.un_HAZ_category == '-3<=HAZ<-2') & ~df.index.isin(idx_will_be_stunted)
                                    & ~df.index.isin(idx_will_recover))
        idx_will_progress = self.apply_model(
            model=models.lm_prob_progression_to_severe_stunting,
            mask=eligible_for_progression,
            days_exposed_to_risk=days_until_next_polling_event
        )
        self.module.do_progression(idx_will_progress)

    def apply_model(self, model, mask, days_exposed_to_risk):
        """Return the person_ids selected for an event of occur to in a given period (in days), as specified by a
        LinearModel that provides the annual risk of the event.
        * Looks-up annual probability of the event using a linear model supplied in `model` to a population masked with
         `mask`
        * Converts the annual probability to a probability of the event occurring during the number of
        `days_exposed_to_risk`
        * Randomly selects which individuals will have the events
        """
        df = self.sim.population.props
        rng = self.module.rng

        annual_prob = model.predict(df.loc[mask]).clip(upper=1.0)
        cum_prob_over_days_exposed = 1.0 - np.exp(np.log(1.0 - annual_prob) * days_exposed_to_risk / 365.25)

        assert pd.notnull(cum_prob_over_days_exposed).all()
        return mask[mask].index[cum_prob_over_days_exposed > rng.random_sample(mask.sum())]


# ---------------------------------------------------------------------------------------------------------
#   Logging
# ---------------------------------------------------------------------------------------------------------


class StuntingLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Logging event occurring every year"""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(years=1))
        assert isinstance(module, Stunting)

    def apply(self, population):
        """Log the current distribution of stunting classification by age"""
        df = population.props

        d_to_log = df.loc[df.is_alive & (df.age_years < 5)].groupby(
            by=['age_years', 'un_HAZ_category']).size().sort_index().to_dict()

        def convert_keys_to_string(d):
            return {str(k): v for k, v in d.items()}

        logger.info(
            key='prevalence',
            data=convert_keys_to_string(d_to_log),
            description='Current number of children in each stunting category by single year of age.'
        )


# ---------------------------------------------------------------------------------------------------------
#   HSI
# ---------------------------------------------------------------------------------------------------------

class HSI_Stunting_ComplementaryFeeding(HSI_Event, IndividualScopeEventMixin):
    """HSI for complementary feeding, either with the provision of supplementary foods or with education only."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = 'Undernutrition_Feeding'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'U5Malnutr': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        person = df.loc[person_id]

        if not person.is_alive:
            return

        # Only do anything if child is under 5 and remains stunted
        if (person.age_years < 5) and (person.un_HAZ_category in ['HAZ<-3', '-3<=HAZ<-2']):

            # Provide supplementary feeding if consumable available, otherwise provide 'education only' (which has a
            # different probability of success).
            if self.get_consumables(item_codes=self.module.cons_item_codes['supplementary_feeding']):
                self.module.do_treatment(person_id, prob_success=self.module.parameters[
                    'effectiveness_of_food_supplementation_in_stunting_reduction'])

            else:
                # Request consumables for provision of education for supplementary feeding, but do not let
                # non-availability prevent provision of the intervention.
                _ = self.get_consumables(
                    item_codes=self.module.cons_item_codes['education_for_supplementary_feeding'])
                self.module.do_treatment(person_id, prob_success=self.module.parameters[
                    'effectiveness_of_complementary_feeding_education_in_stunting_reduction'])

            # Schedule a further instance of this HSI for monitoring and resupply of consumables.
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=self,
                priority=2,  # <-- lower priority that for wasting and most other HSI
                topen=self.sim.date + pd.DateOffset(months=6)
            )


# ---------------------------------------------------------------------------------------------------------
#   ACCESSORIES FOR TESTING
# ---------------------------------------------------------------------------------------------------------

class StuntingPropertiesOfOtherModules(Module):
    """For the purpose of the testing, this module generates the properties upon which the Stunting module relies"""

    INIT_DEPENDENCIES = {'Demography'}

    # Though this module provides some properties from NewbornOutcomes we do not list
    # NewbornOutcomes in the ALTERNATIVE_TO set to allow using in conjunction with
    # SimplifiedBirths which can also be used as an alternative to NewbornOutcomes
    ALTERNATIVE_TO = {'Hiv', 'Wasting', 'Diarrhoea'}

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
        'un_ever_wasted': Property(Types.BOOL, 'temporary property'),
        'gi_number_of_episodes': Property(Types.INT, 'temporary property')
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
        df.loc[df.is_alive, 'un_ever_wasted'] = False
        df.loc[df.is_alive, 'gi_number_of_episodes'] = 0

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
        df.at[child, 'un_ever_wasted'] = False
        df.at[child, 'gi_number_of_episodes'] = 0
