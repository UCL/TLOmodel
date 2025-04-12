"""
A simple test module for demonstration purposes.

It polls the population each month, randomly making non-pregnant people pregnant with
a fixed probability, set as a module parameter. Nine months after becoming pregnant
(assuming the mother does not die) a new person is born.

This is not intended to be realistic - there are no sexes, and newborn infants may
become pregnant!
"""

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent


class RandomBirth(Module):
    """Randomly become pregnant with fixed probability.

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
        'pregnancy_probability': Parameter(
            Types.REAL, 'Fixed probability of pregnancy each month (if not already pregnant)'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'is_pregnant': Property(Types.BOOL, 'Whether this individual is currently pregnant'),
        'date_of_birth': Property(
            Types.DATE, 'When the individual was born'),
        'children': Property(Types.LIST, 'The children born to this individual, in birth order'),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        pass

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """
        df = population.props
        # Everyone starts off not pregnant
        # We use 'broadcasting' to set the same value for every individual
        df.is_pregnant = False
        # We randomly sample birth dates for the initial population during the preceding decade
        start_date = self.sim.date
        dates = pd.date_range(start_date - DateOffset(years=10), start_date, freq='M')
        df.date_of_birth = self.rng.choice(dates, size=len(df))
        # No children have yet been born. We iterate over the population to ensure each
        # person gets a distinct list.
        for index, row in df.iterrows():
            df.at[index, 'children'] = []

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.

        Here we add our monthly event to poll the population for deaths.
        """
        pregnancy_poll = RandomPregnancyEvent(
            self, self.parameters["pregnancy_probability"]
        )
        sim.schedule_event(pregnancy_poll, sim.date + DateOffset(months=1))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props

        df.at[child_id, 'date_of_birth'] = self.sim.date
        df.at[child_id, 'is_pregnant'] = False
        df.at[child_id, 'children'] = []
        df.at[mother_id, 'children'].append(child_id)
        df.at[mother_id, 'is_pregnant'] = False


class RandomPregnancyEvent(RegularEvent, PopulationScopeEventMixin):
    """The regular event that actually makes people pregnant.

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module, pregnancy_probability):
        """Create a new random pregnancy event.

        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.

        :param module: the module that created this event
        :param pregnancy_probability: the per-person probability of pregnancy each month
        """
        self.pregnancy_probability = pregnancy_probability
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        """Apply this event to the population.

        For efficiency, we use pandas operations to scan the entire population
        and initiate pregnancies at random.

        :param population: the current population
        """
        df = population.props
        # Find live and non-pregnant individuals
        candidates = df[df.is_alive & ~df.is_pregnant]
        # OR: candidates = population.props.query('is_alive & ~is_pregnant')
        # Throw a die for each
        rng = self.module.rng
        birth_date = self.sim.date + DateOffset(months=9)
        for person_index in candidates.index:
            if rng.rand() < self.pregnancy_probability:
                # Schedule a birth event for this person
                birth = DelayedBirthEvent(self.module, person_index)
                self.sim.schedule_event(birth, birth_date)
                df.loc[person_index, 'is_pregnant'] = True


class DelayedBirthEvent(Event, IndividualScopeEventMixin):
    """A one-off event in which a pregnant mother gives birth.

    For an individual-scoped event we need to specify the person it applies to in
    the constructor.
    """

    def __init__(self, module, mother_id):
        """Create a new birth event.

        We need to pass the person this event happens to to the base class constructor
        using super(). We also pass the module that created this event, so that random
        number generators can be scoped per-module.

        :param module: the module that created this event
        :param mother_id: the person giving birth
        """
        super().__init__(module, person_id=mother_id)

    def apply(self, mother_id):
        """Apply this event to the given person.

        Assuming the person is still alive, we ask the simulation to create a new offspring.

        :param mother_id: the person the event happens to, i.e. the mother giving birth
        """
        df = self.sim.population.props
        if df.at[mother_id, 'is_alive']:
            self.sim.do_birth(mother_id)
