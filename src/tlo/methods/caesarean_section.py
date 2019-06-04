
import logging

import pandas as pd

import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, Labour, eclampsia_treatment


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CaesareanSection(Module):
    """ This module manages both emergency and planned deliveries via caesarean section
    """

    PARAMETERS = {
        'parameter_a': Parameter(
            Types.REAL, 'Description of parameter a'),
    }

    PROPERTIES = {
        'property_a': Property(Types.BOOL, 'Description of property a'),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        pass

    def initialise_population(self, population):

       pass

    def initialise_simulation(self, sim):

        event = CaesareanLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        pass


class EmergencyCaesareanSection(Event, IndividualScopeEventMixin):

    """Event handling deliveries for women requiring an emergency caesarean section for any indication
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self


class CaesareanLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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
