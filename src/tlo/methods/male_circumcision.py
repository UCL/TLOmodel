"""
A skeleton template for disease methods.
"""
import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin


class male_circumcision(Module):
    """
    male circumcision, without health system links
    """

    def __init__(self, name=None):
        super().__init__(name)
        self.store = {'Time': [], 'proportion_circumcised': [], 'circumcised_in_last_timestep': []}

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
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props  # a shortcut to the dataframe storing data for individuals

        df['is_circumcised'] = False  # default: no individuals circumcised
        df['date_circumcised'] = pd.NaT  # default: not a time


        # randomly selected some individuals as infected
        initial_infected = self.parameters['initial_circumcision']
        initial_uninfected = 1 - initial_infected
        df['is_circumcised'] = np.random.choice([True, False], size=len(df), p=[initial_infected, initial_uninfected])

        # set the properties of infected individuals
        df.loc[df.mi_is_infected, 'mi_date_infected'] = self.sim.date   # TODO: perhaps change to DOB


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

        circumcised_total = df.mi_is_circumcised.sum()
        proportion_circumcised = circumcised_total / len(df)

        mask = (df['date_circumcised'] > self.sim.date - DateOffset(months=self.repeat))
        circumcised_in_last_timestep = mask.sum()

        self.module.store.append(proportion_circumcised, circumcised_in_last_timestep)



