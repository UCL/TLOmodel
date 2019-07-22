"""
A skeleton template for disease methods.
"""
import logging

import pandas as pd

import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SandBox(Module):
    """ This module handles all assisted vaginal deliveries and associated complications
    """

    PARAMETERS = {
    }

    PROPERTIES = {
        'birth_history': Property(Types.LIST, 'Description of property a'),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """
        df = population.props
        m = self
        rng = m.rng
        params = self.parameters

        birth_history_one = [{'due_date': 'n/a', 'labour_type': 'n/a', 'delivery_type': 'n/a'},
                                {'due_date': 'n/a', 'labour_type': 'n/a', 'delivery_type': 'n/a'}]


#        df.loc[df.sex == 'F', 'birth_history'] = birth_history_one

        # DUMMY TASKS
#        women_idx = df.index[df.is_alive & (df.sex == 'F')]

#        baseline_cs1 = pd.Series(0, index=women_idx)

        # A weighted random choice is used to determine whether women who are para1 had delivered via caesarean
#        random_draw1 = pd.Series(self.rng.choice(range(0, 2), p=[0.91, 0.09], size=len(women_idx)),
#                                 index=women_idx)

#        dfx = pd.concat([baseline_cs1, random_draw1], axis=1)
#        dfx.columns = ['baseline_cs1', 'random_draw1']
#        idx_prev_cs = dfx.index[dfx.random_draw1 >= 1]
#        df.loc[idx_prev_cs, 'birth_history'][0]["due_date"] = 1

 #       women_idx = df.index[df.is_alive & df.sex == 'F']

       #  birth_history['due_date'] = 7
      #  for idx_prev_cs


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        event = LabourLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))


    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        pass

class SkeletonEvent(RegularEvent, PopulationScopeEventMixin):
    """A skeleton class for an event

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """One line summary here

        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.

        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        pass

class LabourLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""
    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(days=self.repeat))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props

        logger.debug('%s|person_one|%s',
                          self.sim.date, df.loc[0].to_dict())
