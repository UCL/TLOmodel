"""
A skeleton template for disease methods.
"""
import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


class male_circumcision(Module):
    """
    male circumcision, without health system links
    """

    def __init__(self, name=None):
        super().__init__(name)
        self.store = {'Time': [], 'proportion_circumcised': [], 'recently_circumcised': []}

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'p_circumcision': Parameter(Types.REAL, 'Probability that an individual gets circumcised'),
        'initial_circumcision': Parameter(Types.REAL, 'Prevalence of circumcision in the population at baseline'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'is_circumcised': Property(Types.BOOL, 'individual is circumcised'),
        'date_circumcised': Property(Types.DATE, 'Date of circumcision'),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters
        params['p_circumcision'] = 0.05
        params['initial_circumcision'] = 0.1

    def initialise_population(self, population):
        df = population.props  # a shortcut to the dataframe storing data for individuals

        df['is_circumcised'] = False  # default: no individuals circumcised
        df['date_circumcised'] = pd.NaT  # default: not a time

        # randomly selected some individuals as circumcised
        initial_circumcised = self.parameters['initial_circumcision']
        df.loc[df.is_alive, 'is_circumcised'] = np.random.choice([True, False], size=len(df[df.is_alive]),
                                                                 p=[initial_circumcised, 1 - initial_circumcised])

        # set the properties of circumcised individuals
        df.loc[df.is_circumcised, 'date_infected'] = self.sim.date  # TODO: perhaps change to DOB

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        # add the basic event (we will implement below)
        event = CircumcisionEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # add an event to log to screen
        sim.schedule_event(CircumcisionLoggingEvent(self), sim.date + DateOffset(months=6))

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        pass


class CircumcisionEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.p_infection = module.parameters['p_circumcision']

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        df = population.props

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df))

        # probability of circumcision
        circumcision_index = df.index[(random_draw < params['p_circumcision']) & ~df.is_circumcised & df.is_alive]
        df.loc[circumcision_index, 'is_circumcised'] = True
        df.loc[circumcision_index, 'date_circumcised'] = now


class CircumcisionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        circumcised_total = len(df[df.is_alive & df.is_circumcised])
        proportion_circumcised = circumcised_total / len(df[df.is_alive])

        mask = (df['date_circumcised'] > self.sim.date - DateOffset(months=self.repeat))
        circumcised_in_last_timestep = mask.sum()

        self.module.store['Time'].append(self.sim.date)
        self.module.store['proportion_circumcised'].append(proportion_circumcised)
        self.module.store['recently_circumcised'].append(circumcised_in_last_timestep)
