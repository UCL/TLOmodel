"""
A simple test module for demonstration purposes.

It polls the population each month, randomly killing people with a fixed probability.
The probability may be set as a module parameter.

This is thus mainly useful for demonstrating how to write a disease module, without
getting distracted by modelling details.
"""

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


class RandomDeath(Module):
    """Randomly kill individuals with fixed probability.

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
        'death_probability': Parameter(Types.REAL, 'Fixed probability of death each month'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'is_alive': Property(Types.BOOL, 'Whether each individual is currently alive'),
        'date_of_death': Property(
            Types.DATE, 'When the individual died (if they have)'),
    }

    def __init__(self, *args, **kwargs):
        """Constructor: create an instance of this module.

        This method can usually be omitted, but may be a useful place to set up module-level state
        that isn't appropriate to put elsewhere. Constructor arguments can be supplied when the
        module is instantiated.

        Here we do nothing, except that we must call the base class constructor using super().
        Alternatively, we could omit the method entirely here, and just the base class constructor
        would be used.

        :param args: list of positional arguments
        :param kwargs: dict of keyword arguments
        """
        super().__init__(*args, **kwargs)

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

        TODO: We probably need to declare somehow which properties we 'read' here, so the
        simulation knows what order to initialise modules in!

        :param population: the population of individuals
        """
        # Everyone starts off alive
        # We use 'broadcasting' to set the same value for every individual
        population.props.is_alive = True
        # No-one has a death date yet, so we can leave that uninitialised
        # (which means it will be full of 'not a time' values)

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.

        Here we add our monthly event to poll the population for deaths.
        """
        death_poll = RandomDeathEvent(self, self.parameters['death_probability'])
        sim.schedule_event(death_poll, sim.date + DateOffset(months=1))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props
        df.at[child_id, 'is_alive'] = True


class RandomDeathEvent(RegularEvent, PopulationScopeEventMixin):
    """The regular event that actually kills people.

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module, death_probability):
        """Create a new random death event.

        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.

        :param module: the module that created this event
        :param death_probability: the per-person probability of death each month
        """
        self.death_probability = death_probability
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        """Apply this event to the population.

        For efficiency, we use pandas operations to scan the entire population
        and kill individuals at random.

        :param population: the current population
        """
        df = population.props
        # Generate a series of random numbers, one per individual
        probs = self.module.rng.rand(len(df))
        # Figure out which individuals are newly dead
        # ('deaths' here is a pandas.Series of True/False values)
        deaths = df.is_alive & (probs < self.death_probability)
        # Record their date of death
        df.loc[deaths, 'date_of_death'] = self.sim.date
        # Kill them
        df.loc[deaths, 'is_alive'] = False
        # We could do this more verbosely:
        # population.is_alive = population.is_alive & (probs >= self.death_probability)
