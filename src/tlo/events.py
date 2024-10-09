"""Support for creating different kinds of events."""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from tlo import DateOffset

if TYPE_CHECKING:
    from tlo import Simulation

import pandas as pd


class Priority(Enum):
    """Enumeration for the Priority, which is used in sorting the events in the simulation queue."""
    START_OF_DAY = 0
    FIRST_HALF_OF_DAY = 25
    LAST_HALF_OF_DAY = 75
    END_OF_DAY = 100

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Event:
    """Base event class, from which all others inherit.

    Concrete subclasses should also inherit from one of the EventMixin classes
    defined below, and implement at least an `apply` method.
    """

    def __init__(self, module, *args, priority=Priority.FIRST_HALF_OF_DAY, **kwargs):
        """Create a new event.

        Note that just creating an event does not schedule it to happen; that
        must be done by calling Simulation.schedule_event.

        :param module: the module that created this event.
            All subclasses of Event take this as the first argument in their
            constructor, but may also take further keyword arguments.
        :param priority: a keyword-argument to set the priority (see Priority enum)
        """
        assert isinstance(priority, Priority), "priority argument should be a value from Priority enum"
        self.module = module
        self.sim = module.sim
        self.priority = priority
        self.target = None
        # This is needed so mixin constructors are called
        super().__init__(*args, **kwargs)

    def post_apply_hook(self):
        """Do any required processing after apply() completes."""

    def apply(self, target):
        """Apply this event to the given target.

        Must be implemented by subclasses.

        :param target: the target of the event
        """
        raise NotImplementedError

    def run(self):
        """Make the event happen."""
        
        print_chains = False
        df_before = []
        
        if self.sim.generate_event_chains:
            # Only print event if it belongs to modules of interest and if it is not in the list of events to ignore
            if (self.module in self.sim.generate_event_chains_modules_of_interest) and not set(self.sim.generate_event_chains_ignore_events).intersection(str(self)):
                print_chains = True
                if self.target != self.sim.population:
                    row = self.sim.population.props.loc[[self.target]]
                    row['person_ID'] = self.target
                    row['event'] = str(self)
                    row['event_date'] = self.sim.date
                    row['when'] = 'Before'
                    self.sim.event_chains = pd.concat([self.sim.event_chains, row], ignore_index=True)
                else:
                    df_before = self.sim.population.props.copy()
        
        self.apply(self.target)
        self.post_apply_hook()
                
        if print_chains:
            if self.target != self.sim.population:
                row = self.sim.population.props.loc[[self.target]]
                row['person_ID'] = self.target
                row['event'] = str(self)
                row['event_date'] = self.sim.date
                row['when'] = 'After'
                self.sim.event_chains = pd.concat([self.sim.event_chains, row], ignore_index=True)
            else:
                df_after = self.sim.population.props.copy()
                change = df_before.compare(df_after)
                if ~change.empty:
                    indices = change.index
                    new_rows_before = df_before.loc[indices]
                    new_rows_before['person_ID'] = new_rows_before.index
                    new_rows_before['event'] = self
                    new_rows_before['event_date'] = self.sim.date
                    new_rows_before['when'] = 'Before'
                    new_rows_after = df_after.loc[indices]
                    new_rows_after['person_ID'] = new_rows_after.index
                    new_rows_after['event'] = self
                    new_rows_after['event_date'] = self.sim.date
                    new_rows_after['when'] = 'After'

                    self.sim.event_chains = pd.concat([self.sim.event_chains,new_rows_before], ignore_index=True)
                    self.sim.event_chains = pd.concat([self.sim.event_chains,new_rows_after], ignore_index=True)


class RegularEvent(Event):
    """An event that automatically reschedules itself at a fixed frequency."""

    def __init__(self, module, *, frequency, end_date=None, **kwargs):
        """Create a new regular event.

        :param module: the module that created this event
        :param frequency: the interval from one occurrence to the next
            (must be supplied as a keyword argument)
        :type frequency: pandas.tseries.offsets.DateOffset
        """
        super().__init__(module, **kwargs)
        assert isinstance(frequency, DateOffset)
        self.frequency = frequency
        self.end_date = end_date

    def apply(self, target):
        """Apply this event to the given target.

        This is a no-op; subclasses should override this method.

        :param target: the target of the event
        """

    def post_apply_hook(self):
        """Schedule the next occurrence of this event."""
        next_apply_date = self.sim.date + self.frequency
        if not self.end_date or next_apply_date <= self.end_date:
            self.sim.schedule_event(self, next_apply_date)


class PopulationScopeEventMixin:
    """Makes an event operate on the entire population.

    This class is designed to be used via multiple inheritance along with one
    of the main event classes. It indicates that when an event happens, it is
    applied to the entire population, rather than a single individual.
    Contrast IndividualScopeEventMixin.

    Subclasses should implement `apply(self, population)` to contain their
    behaviour.
    """

    sim: Simulation

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

    def __init__(self, *args, person_id, **kwargs):
        """Create a new individual-scoped event.

        This calls the base class constructor, passing any arguments through,
        and sets the event target as the provided person.

        :param person_id: the id of the person this event applies to
            (must be supplied as a keyword argument)
        """
        super().__init__(*args, **kwargs)
        self.target = person_id

    def apply(self, person_id):
        """Apply this event to the given person.

        Must be implemented by subclasses.

        :param person_id: the person the event happens to
        """
        raise NotImplementedError
