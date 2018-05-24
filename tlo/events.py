"""Support for creating different kinds of events."""

from pandas.tseries.offsets import DateOffset


class Event:
    """Base event class, from which all others inherit.

    Concrete subclasses should also inherit from one of the EventMixin classes
    defined below, and implement at least an `apply` method.
    """

    def __init__(self, module):
        """Create a new event.

        Note that just creating an event does not schedule it to happen; that
        must be done by calling Simulation.schedule_event.

        :param module: the module that created this event.
            All subclasses of Event take this as the first argument in their
            constructor, but may also take further keyword arguments.
        """
        self.module = module
        self.sim = module.sim

    def post_apply_hook(self):
        """Do any required processing after apply() completes."""
        pass

    def run(self):
        """Make the event happen."""
        self.apply(self.target)
        self.post_apply_hook()


class RegularEvent(Event):
    """An event that automatically reschedules itself at a fixed frequency."""

    def __init__(self, module, *, frequency):
        """Create a new regular event.

        :param module: the module that created this event
        :param frequency: the interval from one occurrence to the next
            (must be supplied as a keyword argument)
        :type frequency: pandas.tseries.offsets.DateOffset
        """
        super().__init__(module)
        assert isinstance(frequency, DateOffset)
        self.frequency = frequency

    def post_apply_hook(self):
        """Schedule the next occurrence of this event."""
        self.sim.schedule_event(self, self.sim.date + self.frequency)


class PopulationScopeEventMixin:
    """Makes an event operate on the entire population.

    This class is designed to be used via multiple inheritance along with one
    of the main event classes. It indicates that when an event happens, it is
    applied to the entire population, rather than a single individual.
    Contrast IndividualScopeEventMixin.

    Subclasses should implement `apply(self, population)` to contain their
    behaviour.
    """

    def __init__(self, *args, **kwargs):
        """Create a new population-scoped event.

        This calls the base class constructor, passing any arguments through,
        and sets the event target as the whole population.
        """
        super().__init__(*args, **kwargs)
        self.target = self.sim.population

    def apply(self, population):
        """Apply this event to the population.

        Must be implemented by subclasses.

        :param population: the current population
        """
        raise NotImplementedError


class IndividualScopeEventMixin:
    """Makes an event operate on a single individual.

    This class is designed to be used via multiple inheritance along with one
    of the main event classes. It indicates that when an event happens, it is
    applied to a single individual, rather than the entire population.
    Contrast PopulationScopeEventMixin.

    Subclasses should implement `apply(self, person)` to contain their
    behaviour.
    """

    def __init__(self, *args, person, **kwargs):
        """Create a new individual-scoped event.

        This calls the base class constructor, passing any arguments through,
        and sets the event target as the provided person.

        :param person: the Person this event applies to
            (must be supplied as a keyword argument)
        """
        super().__init__(*args, **kwargs)
        self.target = person

    def apply(self, person):
        """Apply this event to the given person.

        Must be implemented by subclasses.

        :param person: the person the event happens to
        """
        raise NotImplementedError
