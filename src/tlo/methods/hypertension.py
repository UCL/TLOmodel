"""
This is the method for hypertension
Developed by Mikaela Smit, October 2018

"""

# Questions:
# 1. Should treatment status be in this method or in treatment method?
# 2. Should there be death due to hypertension?

import pandas as pd
import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin


class HT(Module):
    """
    This is the hypertension module

    All disease modules need to be implemented as a class inheriting from Module.
    They need to provide several methods which will be called by the simulation
    framework:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)`

    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # For hypertension we will have two key features:
    # i) an initial population prevalence
    # ii) a probability of new cases of hypertension
    PARAMETERS = {
        'parameter_initial_prevalence': Parameter(Types.REAL, 'Prevalence at the start of the model'),
        'parameter_onset': Parameter(Types.REAL, 'Probability of developing new case of  hypertension'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'ht_current_status': Property(Types.BOOL, 'Current hypertension status'),
        'ht_historic status': Property(Types.CATEGORICAL,
                                       'Historical status: N=never; C=Current, P=Previous',
                                       categories=['N', 'C', 'P']),
        'ht_date_case': Property(Types.DATE, 'Date of latest hypertension'),
        'ht_date_death': Property(Types.DATE, 'Date of hypertension death'),
        # 'ht_treatment status':Property(Types.CATEGORICAL,
        #                                'Historical status: N=never; C=Current, P=Previous',
        #                                categories = ['N', 'C', 'P']),
        #  'ht_date_treatment': Property(Types.DATE, 'Date of latest hypertension treatment'),

    }

    def read_parameters(self, data_folder):

        """Will need to update this to read parameter values from file.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

        params = self.parameters
        params['parameter_initial_prevalence'] = 0.01
        params['parameter_onset'] = 0.01

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props  # a shortcut to the dataframe (df) storing data on individuals

        # Set default values for all variables to be initialised
        df['ht_current_status'] = False  # Default setting: no one has hypertension
        df['ht_historic_status'] = 'N'  # Default setting: no one has hypertension
        df['ht_date_case'] = pd.NaT  # Default setting: no one has a date for hypertension
        df['ht_date_death'] = pd.NaT  # Default setting: no one dies from hypertension

        # Randomly assign hypertension at the start of the model]
        hypertension_yes_atstart = self.parameters['parameter_initial_prevalence']
        hypertension_no_atstart = 1 - hypertension_yes_atstart
        df['ht_current_status'] = np.random.choice([True, False],
                                                   size=len(df),
                                                   p=[hypertension_yes_atstart, hypertension_no_atstart])

        # Count all individuals with hypertension at the start
        hypertension_count = df.ht_current_status.sum()

        # Set date of hypertension amongst those with prevalent cases
        ht_years_ago = np.random.exponential(scale=5, size=hypertension_count)
        infected_td_ago = pd.to_timedelta(ht_years_ago, unit='y')

        # Set date of hypertension death amongst those with prevalent cases
        death_years_ahead = np.random.exponential(scale=2, size=hypertension_count)
        death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')

        # Set the properties of those with prevalent hypertension
        df.loc[df.ht_current_status, 'ht_date_case'] = self.sim.date - infected_td_ago
        df.loc[df.ht_current_status, 'ht_date_death'] = self.sim.date + death_td_ahead

        age = population.age

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        # Add the basic event (implement below)
        event = HTEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # Add an event to log to screen
        sim.schedule_event(HTLoggingEvent(self), sim.date + DateOffset(months=6))

        # Add death event
        df = sim.population.props
        hypertension_individuals = df[df.ht_current_status].index
        for index in hypertension_individuals:
            individual = self.sim.population[index]
            death_event = HTDeathEvent(self, individual)
            self.sim.schedule_event(death_event, individual.ht_date_death)

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        pass


class HTEvent(RegularEvent, PopulationScopeEventMixin):
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
        self.parameter_onset = module.parameters['parameter_onset']

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """

        df = population.props

        # 1. Get (and hold) index of people with and w/o hypertension
        currently_ht_yes = df[df.ht_current_status & df.is_alive].index
        currently_ht_no = df[~df.ht_current_status & df.is_alive].index

        # 2. Handle new cases of hypertension
        now_hypertensive = np.random.choice([True, False], size=len(currently_ht_no,),
                                            p=[self.parameter_onset, 1-self.parameter_onset])

        # if any are hypertensive
        if now_hypertensive.sum():
            ht_idx = currently_ht_no[now_hypertensive]

            df.loc[ht_idx, 'ht_date_case'] = self. sim.date
            df.loc[ht_idx, 'ht_date_death'] = self.sim.date + pd.Timedelta(25, unit='Y')
            df.loc[ht_idx, 'ht_current_status'] = True
            df.loc[ht_idx, 'ht_historic status'] = 'C'

            # Schedule death for those infected
            for index in ht_idx:
                individual = self.sim.population[index]
                death_event = HTDeathEvent(self.module, individual)
                self.sim.schedule_event(death_event, individual.ht_date_death)


class HTDeathEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, individual):
        super().__init__(module, person=individual)

    def apply(self, individual):
        pass


class HTLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # Get some summary statistics
        df = population.props

        ht_total = df.ht_current_status.sum()
        proportion_ht = ht_total / len(df)

        mask = (df['ht_date_case'] > self.sim.date - DateOffset(months=self.repeat))
        infected_in_last_month = mask.sum()

        counts = {'N': 0, 'C': 0, 'P': 0}
        counts.update(df['ht_current_status'].value_counts().to_dict())
        status = 'Status: { N: %(N)d; C: %(C)d; P: %(P)d }' % counts

        print('%s - Hypertension: {TotHT: %d; PropHT: %.3f }' %
              (self.sim.date,
               ht_total,
               proportion_ht,
               ), flush=True)

