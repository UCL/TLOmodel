"""
A skeleton template for disease methods.
"""

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
import numpy as np
import pandas as pd

class Lifestyle(Module):
    """
    One line summary goes here...

    All disease modules need to be implemented as a class inheriting from Module.
    They need to provide several methods which will be called by the simulation
    framework:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)`
    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'r_urban': Parameter(Types.REAL, 'rate of change from rural to urban'),
        'r_rural': Parameter(Types.REAL, 'rate of change from urban to rural'),
        'initial_urban': Parameter(Types.REAL, 'proportion urban at baseline'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'li_urban': Property(Types.BOOL, 'Currently urban'),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        self.parameters['r_urban'] = 0.05
        self.parameters['r_rural'] = 0.01
        self.parameters['initial_urban'] = 0.17

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props  # a shortcut to the dataframe storing data for individiuals
        df['li_urban'] = False  # default: all individuals urban

        # randomly selected some individuals as urban
        initial_urban = self.parameters['initial_urban']
        initial_rural = 1 - initial_urban
        df['li_urban'] = np.random.choice([True, False], size=len(df), p=[initial_urban, initial_rural])


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        pass

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        pass


class UrbanEvent(RegularEvent, PopulationScopeEventMixin):
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
