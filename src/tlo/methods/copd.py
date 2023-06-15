from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from tlo import Module, Parameter, Property, Types, logging
from tlo.analysis.utils import flatten_multi_index_series_into_dict_for_logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, Predictor
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.util import random_date

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch_lungfunction_cats = list(range(7))


class Copd(Module):
    """The module responsible for determining Chronic Obstructive Pulmonary Diseases (COPD) status and outcomes.
     and initialises parameters and properties associated with COPD plus functions and events related to COPD."""

    INIT_DEPENDENCIES = {'SymptomManager', 'Lifestyle'}
    ADDITIONAL_DEPENDENCIES = set()

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
    }

    gbd_causes_of_copd_represented_in_this_module = {
        'Other chronic respiratory diseases',
    }

    CAUSES_OF_DEATH = {
        # Chronic Obstructive Pulmonary Diseases
        'COPD': Cause(gbd_causes=sorted(gbd_causes_of_copd_represented_in_this_module),
                      label='COPD')
    }

    CAUSES_OF_DISABILITY = {
        # Chronic Obstructive Pulmonary Diseases
        'COPD': Cause(gbd_causes=sorted(gbd_causes_of_copd_represented_in_this_module),
                      label='COPD')
    }

    PARAMETERS = {
        'prob_progress_to_next_cat': Parameter(
            Types.REAL, 'probability of changing from a lower lung function category to a higher lung function category'
        ),
        'prob_mod_exacerb': Parameter(
            Types.LIST, 'probability of moderate exacerbation given lung function, for each lungfunction category. '
        ),
        'prob_sev_exacerb': Parameter(
            Types.LIST, 'probability of severe exacerbation given lung function, for each lungfunction category.'
        ),
        'prob_will_die_sev_exacerbation': Parameter(
            Types.REAL, 'probability that a person will die of severe exacerbation '
        ),
        'prob_will_survive_given_oxygen': Parameter(
            Types.REAL, 'probability that an individual with severe copd will not die when given oxygen '
        ),
        'rel_risk_progress_per_higher_cat': Parameter(
            Types.REAL, 'relative risk of progression to the next higher level of lung function for each higher level '
        ),
        'rel_risk_tob': Parameter(
            Types.REAL, 'relative effect of current tobacco smoking on the rate of lung function progression'
        ),
        'rel_risk_wood_burn_stove': Parameter(
            Types.REAL, 'relative risk of wood burn stove on the rate of lung function progression'
        ),

    }

    PROPERTIES = {
        'ch_lungfunction': Property(
            Types.CATEGORICAL, 'Lung function of the person.'
                               'NaN for those under 15; on a 7-point scale for others, from 0 (Perfect) to 6 (End-Stage'
                               ' COPD).', categories=ch_lungfunction_cats, ordered=True,
        ),
        'ch_will_die_this_episode': Property(
            Types.BOOL, 'Whether this person will die during a current severe exacerbation'
        ),
        'ch_has_inhaler': Property(
            Types.BOOL, 'Whether the person is currently using an inhaler'
        ),
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = Path(resourcefilepath)
        self.models = None  # Will hold pointer to helper class containing models
        self.item_codes = None  # Will hold dict containing the item_codes for consumables needed in the HSI Events.

    def read_parameters(self, data_folder):
        """ Read all parameters and define symptoms (if any)"""
        self.load_parameters_from_dataframe(pd.read_csv(self.resourcefilepath / 'ResourceFile_Copd.csv'))
        self.define_symptoms()

    def pre_initialise_population(self):
        """Things to do before processing the population:
         * Generate the models
         """
        self.models = CopdModels(self.parameters, self.rng)

    def initialise_population(self, population):
        """ Set initial values of properties values for all individuals"""
        df = population.props
        df.loc[df.is_alive, 'ch_lungfunction'] = self.models.init_lung_function(df.loc[df.is_alive])
        df.loc[df.is_alive, 'ch_will_die_this_episode'] = False
        df.loc[df.is_alive, 'ch_has_inhaler'] = False

    def initialise_simulation(self, sim):
        """ Get ready for simulation start:
         * Schedule the main polling event
         * Look-up item codes for consumables
        """
        sim.schedule_event(CopdPollEvent(self), sim.date)
        self.lookup_item_codes()

    def on_birth(self, mother_id, child_id):
        """ Initialise COPD properties for a newborn individual."""
        props = {
            'ch_lungfunction': self.models.at_birth_lungfunction(child_id),
            'ch_will_die_this_episode': False,
            'ch_has_inhaler': False
        }
        self.sim.population.props.loc[child_id, props.keys()] = props.values()

    def report_daly_values(self):
        """Return disability weight for alive persons, based on the current status of ch_lungfunction."""
        df = self.sim.population.props
        return df.loc[df.is_alive, 'ch_lungfunction'].map(self.models.disability_weight_given_lungfunction)

    def define_symptoms(self):
        """Define and register Symptoms"""
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom('breathless_moderate'),
            Symptom.emergency('breathless_severe'),
        )

    def lookup_item_codes(self):
        """Look-up the item-codes for the consumables needed in the HSI Events for this module."""
        # todo: Need to look-up these item-codes.
        self.item_codes = {
            'bronchodilater_inhaler': 293,
            'steroid_inhaler': 294,
            'oxygen': 127,
            'aminophylline': 292,
            'amoxycillin': 125,
            'prednisolone': 291
        }

    def do_logging(self):
        """Log current states."""
        df = self.sim.population.props
        counts = df.loc[df.is_alive].groupby(by=['sex', 'age_range', 'li_tob', 'ch_lungfunction']).size()
        # proportions = counts.unstack().apply(lambda row: row / row.sum(), axis=1).stack()
        logger.info(
            key='copd_prevalence',
            description='Proportion of alive persons in each COPD category currently (by age and sex)',
            data=flatten_multi_index_series_into_dict_for_logging(counts)
        )

        _ex_tobacco_smokers = df.loc[df.is_alive].groupby(by=['ch_lungfunction']).size()
        logger.info(
            key='ex_smokers',
            description='Proportion of alive persons in each COPD category ever smoked tobacco',
            data=flatten_multi_index_series_into_dict_for_logging(_ex_tobacco_smokers)
        )

    def give_inhaler(self, person_id: int, hsi_event: HSI_Event):
        """Give inhaler if person does not already have one"""
        df = self.sim.population.props
        has_inhaler = df.at[person_id, 'ch_has_inhaler']
        if not has_inhaler:
            if hsi_event.get_consumables(self.item_codes['bronchodilater_inhaler']):
                df.at[person_id, 'ch_has_inhaler'] = True

    def do_when_present_with_breathless(self, person_id: int, hsi_event: HSI_Event):
        """What to do when a person presents at the generic first appt HSI with a symptom of `breathless_severe` or
        `breathless_moderate`.
        * If severe --> give the inhaler and schedule the HSI for Treatment
        * Otherwise --> just give inhaler.
        """
        self.give_inhaler(hsi_event=hsi_event, person_id=person_id)

        if 'breathless_severe' in self.sim.modules['SymptomManager'].has_what(person_id):
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Copd_TreatmentOnSevereExacerbation(module=self, person_id=person_id),
                priority=0,
                topen=self.sim.date,
                tclose=None,
            )


class CopdModels:
    """Helper class containing the models needed for the Copd module."""

    def __init__(self, params, rng):
        self.params = params
        self.rng = rng

        # The chance (in a 3-month period) of progressing to the next (greater) category of ch_lungfunction
        self.__Prob_Progress_LungFunction__ = LinearModel.multiplicative(
            Predictor(
                'ch_lungfunction',
                conditions_are_exhaustive=True,
                conditions_are_mutually_exclusive=True
            ).when(0, self.params['prob_progress_to_next_cat'] *
                   (self.params['rel_risk_progress_per_higher_cat'] ** 0))
            .when(1, self.params['prob_progress_to_next_cat'] *
                  (self.params['rel_risk_progress_per_higher_cat'] ** 1))
            .when(2, self.params['prob_progress_to_next_cat'] *
                  (self.params['rel_risk_progress_per_higher_cat'] ** 2))
            .when(3, self.params['prob_progress_to_next_cat'] *
                  (self.params['rel_risk_progress_per_higher_cat'] ** 3))
            .when(4, self.params['prob_progress_to_next_cat'] *
                  (self.params['rel_risk_progress_per_higher_cat'] ** 4))
            .when(5, self.params['prob_progress_to_next_cat'] *
                  (self.params['rel_risk_progress_per_higher_cat'] ** 5)),
            Predictor(
                'li_tob',
                conditions_are_exhaustive=True,
                conditions_are_mutually_exclusive=True
            ).when(True, self.params['rel_risk_tob']),
            Predictor(
                'li_wood_burn_stove',
                conditions_are_exhaustive=True,
                conditions_are_mutually_exclusive=True
            ).when(True, self.params['rel_risk_wood_burn_stove'])
        )
        # The probability (in a 3-month period) of having a Moderate Exacerbation
        self.__Prob_ModerateExacerbation__ = LinearModel.multiplicative(
            Predictor(
                'ch_lungfunction',
                conditions_are_exhaustive=True,
                conditions_are_mutually_exclusive=True
            ).when(0, self.params['prob_mod_exacerb'][0])
            .when(1, self.params['prob_mod_exacerb'][1])
            .when(2, self.params['prob_mod_exacerb'][2])
            .when(3, self.params['prob_mod_exacerb'][3])
            .when(4, self.params['prob_mod_exacerb'][4])
            .when(5, self.params['prob_mod_exacerb'][5])
            .when(6, self.params['prob_mod_exacerb'][6])
        )

        # The probability (in a 3-month period) of having a Severe Exacerbation
        self.__Prob_SevereExacerbation__ = LinearModel.multiplicative(
            Predictor(
                'ch_lungfunction',
                conditions_are_exhaustive=True,
                conditions_are_mutually_exclusive=True
            ).when(0, self.params['prob_sev_exacerb'][0])
            .when(1, self.params['prob_sev_exacerb'][1])
            .when(2, self.params['prob_sev_exacerb'][2])
            .when(3, self.params['prob_sev_exacerb'][3])
            .when(4, self.params['prob_sev_exacerb'][4])
            .when(5, self.params['prob_sev_exacerb'][5])
            .when(6, self.params['prob_sev_exacerb'][6])
        )

    def init_lung_function(self, df: pd.DataFrame) -> pd.Series:
        """Returns the values for ch_lungfunction for an initial population described in `df`."""
        # For individuals over-15s, assign a lung function of either 2 or 3
        cats = ch_lungfunction_cats[2:4]
        idx_over15_no_tob = df.index[(df.age_years >= 15) & ~df.li_tob]
        probs = np.ones(len(cats)) / len(cats)
        cats_for_over15s_no_tob = dict(zip(idx_over15_no_tob, self.rng.choice(cats, p=probs,
                                                                              size=len(idx_over15_no_tob))))

        # For individuals over-15s and do smoke tobacco, assign a lung function of either 4 or 5
        cats = ch_lungfunction_cats[4:6]
        idx_over15_tob = df.index[(df.age_years >= 15) & df.li_tob]
        cats_for_over15s_tob = dict(zip(idx_over15_tob, self.rng.choice(cats, p=probs,
                                                                        size=len(idx_over15_tob))))
        # For under-15s, assign the category that would be given at birth
        idx_notover15 = df.index[df.age_years < 15]
        cats_for_under15s = {idx: self.at_birth_lungfunction(idx) for idx in idx_notover15}

        return pd.Series(index=df.index, data={**cats_for_over15s_no_tob, **cats_for_over15s_tob, **cats_for_under15s})

    def at_birth_lungfunction(self, person_id: int) -> int:
        """Returns value for ch_lungfunction for the person at birth."""
        return 0

    def prob_livesaved_given_treatment(self, oxygen: bool, amino_phylline: bool):
        """Returns the probability that a treatment prevents death during an exacerbation, according to the treatment
        provided (oxygen and/or amino_phylline)"""
        if oxygen and amino_phylline:
            return self.params['prob_will_survive_given_oxygen']
        elif oxygen:
            return self.params['prob_will_survive_given_oxygen']
        else:
            return 0.0

    @property
    def disability_weight_given_lungfunction(self) -> Dict:
        """Returns `dict` with the mapping between a lung_function and a disability weight"""
        return {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 0.2,
            4: 0.3,
            5: 0.6,
            6: 0.7,
        }

    def will_progres_to_next_cat_of_lungfunction(self, df: pd.DataFrame) -> List:
        """Returns the indices (corresponding to the person_id) that will progress to the next category up of
         `ch_lungfunction`, based on the probability of progression in a 3-month period."""
        will_progress_to_next_category = self.__Prob_Progress_LungFunction__.predict(df, self.rng,
                                                                                     squeeze_single_row_output=False)
        return will_progress_to_next_category[will_progress_to_next_category].index.to_list()

    def will_get_moderate_exacerbation(self, df: pd.DataFrame) -> List:
        """Returns the indices (corresponding to the person_id) that will have a Moderate Exacerbation, based on the
        probability of having a Severe Exacerbation in a 3-month period"""
        will_get_ex = self.__Prob_ModerateExacerbation__.predict(df, self.rng, squeeze_single_row_output=False)
        return will_get_ex[will_get_ex].index.to_list()

    def will_get_severe_exacerbation(self, df: pd.DataFrame) -> List:
        """Returns the indices (corresponding to the person_id) that will have a Severe Exacerbation, based on the
        probability of having a Severe Exacerbation in a 3-month period"""
        will_get_ex = self.__Prob_SevereExacerbation__.predict(df, self.rng, squeeze_single_row_output=False)
        return will_get_ex[will_get_ex].index.to_list()

    def will_die_given_severe_exacerbation(self) -> bool:
        """Return bool indicating if a person will die due to a severe exacerbation."""
        death_rate_prob = self.params['prob_will_die_sev_exacerbation']
        return self.rng.random_sample() < death_rate_prob


def eligible_to_progress_to_next_lung_function(df: pd.DataFrame) -> pd.Series:
    """ Returns a pd.Series with the same index as `df` and with value `True` where individuals are eligible to progress
     to the next level of ch_lungfunction (i.e., alive, aged 15+, and not in the highest category already).
    :param df: an individual population dataframe """
    return (
        df.is_alive & (df.age_years >= 15) & (df['ch_lungfunction'] != ch_lungfunction_cats[-1])
    )


class CopdPollEvent(RegularEvent, PopulationScopeEventMixin):
    """An event that controls the COPD infection process and logs current states. It repeats every 3 months."""

    def __init__(self, module):
        super().__init__(module, frequency=pd.DateOffset(months=3))

    def apply(self, population):
        """
         * Progress persons to higher categories of ch_lungfunction
         * Schedules Exacerbation (Moderate / Severe) events.
         * Log current states.
        """
        df = population.props

        # call a function to make individuals progress to a next higher lung function
        self.progress_to_next_lung_function(df)

        # Schedule Moderate Exacerbation
        for idx in self.module.models.will_get_moderate_exacerbation(df.loc[df.is_alive]):
            self.sim.schedule_event(CopdExacerbationEvent(self.module, idx, severe=False),
                                    self.gen_random_date_in_next_three_months())

        # Schedule Severe Exacerbation (persons can have a moderate and severe exacerbation in the same 3-month period)
        for idx in self.module.models.will_get_severe_exacerbation(df.loc[df.is_alive]):
            self.sim.schedule_event(CopdExacerbationEvent(self.module, idx, severe=True),
                                    self.gen_random_date_in_next_three_months())

        # Logging
        self.module.do_logging()

    def gen_random_date_in_next_three_months(self):
        """Returns a datetime for a day that is chosen randomly to be within the next 3 months."""
        return random_date(self.sim.date, self.sim.date + pd.DateOffset(months=3), self.module.rng)

    @staticmethod
    def increment_category(ser: pd.Series) -> pd.Series:
        """Returns a pd.Series with same index as `ser` but with the categories shifted to next higher one."""
        new_codes = ser.cat.codes + 1
        ser.values[:] = new_codes
        return ser

    def progress_to_next_lung_function(self, df):
        """ make individuals progress to a next higher lung function """
        idx_will_progress_to_next_category = self.module.models.will_progres_to_next_cat_of_lungfunction(
            df.loc[eligible_to_progress_to_next_lung_function(df)])
        df.loc[idx_will_progress_to_next_category, 'ch_lungfunction'] = self.increment_category(
            df.loc[idx_will_progress_to_next_category, 'ch_lungfunction'])


class CopdExacerbationEvent(Event, IndividualScopeEventMixin):
    """An Exacerbation, which may be 'Severe' (severe=True) or 'Moderate' (severe=False).
     * A Moderate Exacerbation will not cause death but may cause non-emergency healthcare seeking;
     * A Severe Exacerbation may cause death and may cause emergency healthcare seeking.
    """

    def __init__(self, module, person_id, severe: bool = False):
        super().__init__(module, person_id=person_id)
        self.severe = severe

    def apply(self, person_id):

        if not self.sim.population.props.at[person_id, 'is_alive']:
            return

        # Onset symptom (that will auto-resolve in two days)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=person_id,
            symptom_string='breathless_severe' if self.severe else 'breathless_moderate',
            add_or_remove='+',
            disease_module=self.module,
            duration_in_days=2,
        )

        if self.severe:
            # Work out if the person will die of this exacerbation (if not treated). If they die, they die the next day.
            if self.module.models.will_die_given_severe_exacerbation():
                self.sim.population.props.at[person_id, "ch_will_die_this_episode"] = True
                self.sim.schedule_event(CopdDeath(self.module, person_id), self.sim.date + pd.DateOffset(days=1))


class CopdDeath(Event, IndividualScopeEventMixin):
    """This event will cause the person to die to 'COPD' if the person's property `ch_will_die_this_episode` is True.
    It is scheduled when a Severe Exacerbation is lethal, but can be cancelled if a subsequent treatment is successful.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id, ['is_alive', 'ch_will_die_this_episode']]
        # Check if they should still die and, if so, cause the death
        if person.is_alive and person.ch_will_die_this_episode:
            self.sim.modules['Demography'].do_death(
                individual_id=person_id,
                cause='COPD',
                originating_module=self.module,
            )
            # log deaths by lung function
            log_deaths_by_ch_lungfunction = {
                'person_id': person_id,
                'ch_lungfunction': df.loc[person_id, 'ch_lungfunction']
            }
            logger.info(
                key="deaths_by_lung_function",
                data=log_deaths_by_ch_lungfunction
            )


class HSI_Copd_TreatmentOnSevereExacerbation(HSI_Event, IndividualScopeEventMixin):
    """ HSI event for issuing treatment to all individuals with severe exacerbation. We first check the availability of
    oxygen at the default facility(1a). If no oxygen is found we refer individuals to the next higher level facility
    until either we find oxygen or we are at the highest facility level. Here facility levels are ["1a", "1b", "2"]"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.facility_levels_index = 0  # an index to facility levels
        self.all_facility_levels = ["1a", "1b", "2"]  # available facility levels

        self.TREATMENT_ID = "Copd_Treatment"
        self.ACCEPTED_FACILITY_LEVEL = self.all_facility_levels[self.facility_levels_index]
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 2})

    def apply(self, person_id, squeeze_factor):
        """What to do when someone presents for care with an exacerbation.
         * Provide treatment: whatever is available at this facility at this time (no referral).
        """
        df = self.sim.population.props
        if not self.get_consumables(self.module.item_codes['oxygen']):
            # refer to the next higher facility if the current facility has no oxygen
            self.facility_levels_index += 1
            if self.facility_levels_index >= len(self.all_facility_levels):
                return
            self.ACCEPTED_FACILITY_LEVEL = self.all_facility_levels[self.facility_levels_index]
            self.sim.modules['HealthSystem'].schedule_hsi_event(self, topen=self.sim.date, priority=0)

        else:
            # Give oxygen and AminoPhylline, if possible, ... and cancel death if the treatment is successful.
            prob_treatment_success = self.module.models.prob_livesaved_given_treatment(
                oxygen=self.get_consumables(self.module.item_codes['oxygen']),
                amino_phylline=self.get_consumables(self.module.item_codes['aminophylline'])
            )
            if self.module.rng.random_sample() < prob_treatment_success:
                df.at[person_id, 'ch_will_die_this_episode'] = False
