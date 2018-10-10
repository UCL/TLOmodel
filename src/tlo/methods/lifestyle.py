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
        'r_urban': Parameter(Types.REAL, 'probability per 3 mths of change from rural to urban'),
        'r_rural': Parameter(Types.REAL, 'probability per 3 mths of change from urban to rural'),
        'initial_p_urban': Parameter(Types.REAL, 'proportion urban at baseline'),
        'initial_p_wealth_1': Parameter(Types.REAL, 'pr wealth level 1'),
        'initial_p_wealth_2': Parameter(Types.REAL, 'pr wealth level 2'),
        'initial_p_wealth_3': Parameter(Types.REAL, 'pr wealth level 3'),
        'initial_p_wealth_4': Parameter(Types.REAL, 'pr wealth level 4'),
        'initial_p_wealth_5': Parameter(Types.REAL, 'pr wealth level 5'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'li_urban': Property(Types.BOOL, 'Currently urban'),
        'li_date_trans_to_urban': Property(Types.DATE, 'date of transition to urban'),
        'li_wealth': Property(Types.CATEGORICAL,categories=['1', '2', '3', '4', '5'])

    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        self.parameters['r_urban'] = 0.05
        self.parameters['r_rural'] = 0.01
        self.parameters['initial_p_urban'] = 0.17
        self.parameters['initial_p_wealth_1_if_urban'] = 0.75
        self.parameters['initial_p_wealth_2_if_urban'] = 0.16
        self.parameters['initial_p_wealth_3_if_urban'] = 0.05
        self.parameters['initial_p_wealth_4_if_urban'] = 0.02
        self.parameters['initial_p_wealth_5_if_urban'] = 0.02
        self.parameters['initial_p_wealth_1_if_rural'] = 0.11
        self.parameters['initial_p_wealth_2_if_rural'] = 0.21
        self.parameters['initial_p_wealth_3_if_rural'] = 0.22
        self.parameters['initial_p_wealth_4_if_rural'] = 0.23
        self.parameters['initial_p_wealth_5_if_rural'] = 0.23

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props  # a shortcut to the dataframe storing data for individiuals
        df['li_urban'] = False  # default: all individuals rural
        df['li_date_trans_to_urban'] = pd.NaT
        df['li_wealth'] = 3  # default: all individuals wealth 3

        # randomly selected some individuals as urban
        initial_urban = self.parameters['initial_p_urban']
        initial_rural = 1 - initial_urban
        df['li_urban'] = np.random.choice([True, False], size=len(df), p=[initial_urban, initial_rural])

        wealth_level_probs_urban = self.parameters['initial_p_wealth_1_if_urban','initial_p_wealth_2_if_urban',
                                    'initial_p_wealth_3_if_urban','initial_p_wealth_4_if_urban',
                                    'initial_p_wealth_5_if_urban']

        wealth_level_probs_rural = self.parameters['initial_p_wealth_1_if_rural','initial_p_wealth_2_if_rural',
                                    'initial_p_wealth_3_if_rural','initial_p_wealth_4_if_rural',
                                    'initial_p_wealth_5_if_rural']

        # assign wealth status

        urban_idx = initial_urban

        for index in urban_idx:
            df['li_wealth'] = np.random.choice(['1', '2', '3', '4', '5'], size=urban_idx.sum(),replace=True, p=wealth_level_probs_urban.values)

        rural_idx = initial_rural

        for index in rural_idx:
            df['li_wealth'] = np.random.choice(['1', '2', '3', '4', '5'], size=rural_idx.sum(),replace=True, p=wealth_level_probs_rural.values)





    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        event = UrbanEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=3))

        event = LifestylesLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=3))

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
        super().__init__(module, frequency=DateOffset(months=3))
        self.r_urban = module.parameters['r_urban']
        self.r_rural = module.parameters['r_rural']

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        df = population.props

        # 1. get (and hold) index of current urban rural status
        currently_rural = df.index[~df.li_urban & df.is_alive]
        currently_urban = df.index[df.li_urban & df.is_alive]

        # 2. handle new transitions
        now_urban = np.random.choice([True, False], size=len(currently_rural),
                                        p=[self.r_urban, 1 - self.r_urban])
        # if any have transitioned to urban
        if now_urban.sum():
            urban_idx = currently_rural[now_urban]

            df.loc[urban_idx, 'li_urban'] = True
            df.loc[urban_idx, 'li_date_trans_to_urban'] = self.sim.date

        # 2. handle new transitions
        now_rural = np.random.choice([True, False], size=len(currently_urban),
                                        p=[self.r_rural, 1 - self.r_rural])
        # if any have transitioned to rural
        if now_rural.sum():
            rural_idx = currently_urban[now_rural]

            df.loc[rural_idx, 'li_urban'] = False

class LifestylesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every 3 month
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        urban_total = df.li_urban.sum()
        proportion_urban = urban_total / len(df)

        mask = (df['li_date_trans_to_urban'] > self.sim.date - DateOffset(months=self.repeat))
        newly_urban_in_last_3mths = mask.sum()

        print('%s lifestyle urban total:%d , proportion_urban: %f , newly urban: %d    ' %
              (self.sim.date,urban_total,proportion_urban, newly_urban_in_last_3mths), flush=True)



