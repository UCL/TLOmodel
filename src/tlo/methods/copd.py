from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from tlo import Module, Property, Types, logging
from tlo.analysis.utils import flatten_multi_index_series_into_dict_for_logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.util import random_date

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch_lungfunction_cats = list(range(7))


class Copd(Module):
    """ The module responsible for infecting individuals with Chronic Obstructive Pulmonary Diseases (COPD). It defines
     and initialises parameters and properties associated with COPD plus functions and events related to COPD."""

    INIT_DEPENDENCIES = {'SymptomManager', }
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
        'COPD':  Cause(gbd_causes=sorted(gbd_causes_of_copd_represented_in_this_module),
                       label='COPD')
    }

    CAUSES_OF_DISABILITY = {
        # Chronic Obstructive Pulmonary Diseases
        'COPD': Cause(gbd_causes=sorted(gbd_causes_of_copd_represented_in_this_module),
                      label='COPD')
    }

    PARAMETERS = {

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
        sim.schedule_event(Copd_PollEvent(self), sim.date)
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
            Symptom('breathless_severe', emergency_in_adults=True, emergency_in_children=True),
            #   todo The line above can be updated to `Symptom.emergency('breathless_severe')` when new PRs are merged.
        )

    def lookup_item_codes(self):
        """Look-up the item-codes for the consumables needed in the HSI Events for this module."""
        # todo: Need to look-up these item-codes.
        self.item_codes = {
            'inhaler': 0,
            'oxygen': 0,
            'amino_phylline': 0
        }

    def do_logging(self):
        """Log current states."""
        df = self.sim.population.props
        counts = df.loc[df.is_alive].groupby(by=['sex', 'age_range', 'ch_lungfunction']).size()
        proportions = counts.unstack().apply(lambda row: row / row.sum(), axis=1).stack()
        logger.info(
            key='copd_prevalence',
            description='Proportion of alive persons in each COPD category currently (by age and sex)',
            data=flatten_multi_index_series_into_dict_for_logging(proportions)
        )

    def give_inhaler(self, person_id: int, hsi_event: HSI_Event):
        """Give inhaler if person does not already have one"""
        df = self.sim.population.props
        has_inhaler = df.at[person_id, 'ch_has_inhaler']
        if not has_inhaler:
            if hsi_event.get_consumables(self.item_codes['inhaler']):
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
                hsi_event=HSI_Copd_Treatment_OnSevereExcaberbation(module=self, person_id=person_id),
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
        self.Prob_Progress_LungFunction = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0.02,
        )

        # The probability (in a 3-month period) of having a Moderate Exacerbation
        self.__Prob_ModerateExacerbation__ = LinearModel.multiplicative(
            Predictor(
                'ch_lungfunction',
                conditions_are_exhaustive=True,
                conditions_are_mutually_exclusive=True
            ).when(0, 0.0)
            .when(1, 0.0)
            .when(2, 0.0)
            .when(3, 0.01)
            .when(4, 0.05)
            .when(5, 0.10)
            .when(6, 0.20)
        )

        # The probability (in a 3-month period) of having a Moderate Exacerbation
        self.__Prob_SevereExacerbation__ = LinearModel.multiplicative(
            Predictor(
                'ch_lungfunction',
                conditions_are_exhaustive=True,
                conditions_are_mutually_exclusive=True
            ).when(0, 0.0)
            .when(1, 0.0)
            .when(2, 0.0)
            .when(3, 0.001)
            .when(4, 0.005)
            .when(5, 0.010)
            .when(6, 0.020)
        )

    def init_lung_function(self, df: pd.DataFrame) -> pd.Series:
        """Returns the values for ch_lungfunction for an initial population described in `df`."""
        # todo Persons are assigned a random category - this should be updated
        cats = ch_lungfunction_cats
        probs = np.ones(len(cats)) / len(cats)
        return pd.Series(index=df.index, data=self.rng.choice(cats, p=probs, size=len(df)))

    def at_birth_lungfunction(self, person_id: int) -> int:
        """Returns value for ch_lungfunction for the person at birth."""
        return 0  # todo This might need to be associated with birth weight.

    def prob_livesaved_given_treatment(self, oxygen: bool, amino_phylline: bool):
        """Returns the probability that a treatment prevents death during an exacerbation, according to the treatment
        provided (oxygen and/or amino_phylline)"""
        if oxygen and amino_phylline:
            return 1.0
        elif oxygen:
            return 0.9
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

    def will_get_moderate_exacerbation(self, df: pd.DataFrame) -> List:
        """Returns the indices (corresponding to the person_id) that will have a Moderate Exacerbation, based on the
        probability of having a Severe Exacerbation in a 3-month period"""
        prob = self.__Prob_ModerateExacerbation__.predict(df, self.rng)
        return prob.index[self.rng.random_sample(len(df)) < prob.values].to_list()

    def will_get_severe_exacerbation(self, df: pd.DataFrame) -> List:
        """Returns the indices (corresponding to the person_id) that will have a Severe Exacerbation, based on the
        probability of having a Severe Exacerbation in a 3-month period"""
        prob = self.__Prob_SevereExacerbation__.predict(df, self.rng)
        return prob.index[self.rng.random_sample(len(df)) < prob.values].to_list()

    def will_die_Given_severe_exacerbation(self) -> bool:
        """Return bool indicating if a person will die due to a severe exacerbation."""
        prob = 0.5
        return prob < self.rng.random_sample()


class Copd_PollEvent(RegularEvent, PopulationScopeEventMixin):
    """An event that controls the COPD infection process and logs current states. It repeats every 3 months."""

    def __init__(self, module):
        super().__init__(module, frequency=pd.DateOffset(months=3))

    def apply(self, population):
        """
         * Progress persons to higher categories of ch_lungfunction
         * Schedules Exacerbation (Moderate / Severe) events.
         * Log current states.
        """

        def gen_random_date_in_next_three_months():
            """Returns a datetime for a day that is chosen randomly to be within the next 3 months."""
            return random_date(self.sim.date, self.sim.date + pd.DateOffset(months=3), self.module.rng)

        df = population.props

        # Progres the ch_lungfunction property (alive, aged 15+, and not in the highest category already)
        eligible_to_progress_category = (
            df.is_alive
            & (df.age_years >= 15)
            & (df['ch_lungfunction'] != ch_lungfunction_cats[-1])
        )

        def increment_category(ser):
            """Returns ser (a pd.Series with categorical variable) with the categories shifted to next higher one."""
            new_codes = ser.cat.codes + 1
            ser.values[:] = new_codes
            return ser

        will_progress_to_next_category = self.module.models.Prob_Progress_LungFunction.predict(
            df.loc[eligible_to_progress_category], self.module.rng)
        idx_will_progress_to_next_category = will_progress_to_next_category[will_progress_to_next_category].index
        df.loc[idx_will_progress_to_next_category, 'ch_lungfunction'] = increment_category(
            df.loc[idx_will_progress_to_next_category, 'ch_lungfunction'])

        # Schedule Moderate Exacerbation
        for id in self.module.models.will_get_moderate_exacerbation(df):
            self.sim.schedule_event(Copd_ExacerbationEvent(self.module, id, severe=False),
                                    gen_random_date_in_next_three_months())

        # Schedule Severe Exacerbation (persons can have a moderate and severe exacerbation in the same 3-month period)
        for id in self.module.models.will_get_severe_exacerbation(df):
            self.sim.schedule_event(Copd_ExacerbationEvent(self.module, id, severe=True),
                                    gen_random_date_in_next_three_months())

        # Logging
        self.module.do_logging()


class Copd_ExacerbationEvent(Event, IndividualScopeEventMixin):
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
            if self.module.models.will_die_Given_severe_exacerbation():
                self.sim.schedule_event(Copd_Death(self.module, person_id), self.sim.date + pd.DateOffset(days=1))


class Copd_Death(Event, IndividualScopeEventMixin):
    """This event will cause the person to die to 'COPD' if the person's property `ch_will_die_this_episode` is True.
    It is scheduled when a Severe Exacerbation is lethal, but can be cancelled if a subsequent treatment is successful.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        person = self.sim.population.props.loc[person_id, ['is_alive', 'ch_will_die_this_episode']]

        # Check if they should still die and, if so, cause the death
        if person.is_alive and person.ch_will_die_this_episode:
            self.sim.modules['Demography'].do_death(
                individual_id=person_id,
                cause='COPD',
                originating_module=self.module,
            )


class HSI_Copd_Treatment_OnSevereExcaberbation(HSI_Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "Copd_Treatment"
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 2})

    def apply(self, person_id, squeeze_factor):
        """What to do when someone presents for care with an exacerbation.
         * Give an inhaler if they do not already have one (it is assumed that once they have one, they always have one)
         * Provide treatment: whatever is available at this facility at this time (no referral).
        """
        # todo: Consider whether person should be referred to higher level.

        df = self.sim.population.props

        # Give oxygen and AminoPhylline, if possible, ... and cancel death if the treatment is successful.
        prob_treatment_success = self.module.models.prob_livesaved_given_treatment(
            oxygen=self.get_consumables(self.module.item_codes['oxygen']),
            amino_phylline=self.get_consumables(self.module.item_codes['amino_phylline'])
        )
        if self.module.rng.random_sample() < prob_treatment_success:
            df.at[person_id, 'ch_will_die_this_episode'] = False
