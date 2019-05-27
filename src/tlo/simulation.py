"""The main simulation controller."""

import heapq
import itertools
import logging
import sys

import numpy as np

from tlo import Population


class Simulation:
    """The main control centre for a simulation.

    This class contains the core simulation logic and event queue, and holds
    references to all the information required to run a complete simulation:
    the population, disease modules, etc.

    Key attributes include:

    `date`
        The current simulation date.

    `modules`
        A list of the disease modules contributing to this simulation.

    `population`
        The Population being simulated.

    `rng`
        The simulation-level random number generator.
        Note that individual modules also have their own random number generator
        with independent state.
    """

    def __init__(self, *, start_date):
        """Create a new simulation.

        :param start_date: the date the simulation begins; must be given as
            a keyword parameter for clarity
        """
        self.date = self.start_date = start_date
        self.modules = {}
        self.rng = np.random.RandomState()
        self.event_queue = EventQueue()

        # TODO: allow override of logging
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s|%(name)s|%(message)s')
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logging.basicConfig(level=logging.DEBUG)

    def register(self, *modules):
        """Register one or more disease modules with the simulation.

        :param modules: the disease module(s) to use as part of this simulation.
            Multiple modules may be given as separate arguments to one call.
        """
        for module in modules:
            assert module.name not in self.modules, (
                'A module named {} has already been registered'.format(module.name))
            self.modules[module.name] = module
            module.sim = self
            module.read_parameters('')  # TODO: Use a proper data_folder

    def seed_rngs(self, seed):
        """Seed all random number generators with the given seed.

        Each module has its own RNG with its own state. This call will seed them all
        with the same value.

        :param seed: the RNG seed to use
        """
        self.rng.seed(seed)
        for module in self.modules.values():
            module.rng.seed(seed)

    def make_initial_population(self, *, n):
        """Create the initial population to simulate.

        :param n: the number of individuals to create; must be given as
            a keyword parameter for clarity
        """
        self.population = Population(self, n)
        for module in self.modules.values():
            module.initialise_population(self.population)

    def simulate(self, *, end_date):
        """Simulation until the given end date

        :param end_date: when to stop simulating. Only events strictly before this
            date will be allowed to occur.
            Must be given as a keyword parameter for clarity.
        """
        self.end_date = end_date # store the end_date so that others can reference it
        for module in self.modules.values():
            module.initialise_simulation(self)
        while self.event_queue:
            event, date = self.event_queue.next_event()
            if date >= end_date:
                self.date = end_date
                break
            self.fire_single_event(event, date)

    def schedule_event(self, event, date):
        """Schedule an event to happen on the given future date.

        :param event: the Event to schedule
        :param date: when the event should happen
        """
        assert date >= self.date, 'Cannot schedule events in the past'

        assert not 'TREATMENT_ID' in dir(event), 'This looks like an HSI event. It should be handed to the healthsystem scheduler'
        assert ( event.__str__().find('HSI_') < 0 ), 'This looks like an HSI event. It should be handed to the healthsystem scheduler'

        self.event_queue.schedule(event, date)

    def fire_single_event(self, event, date):
        """Fires the event once for the given date

        :param event: :py:class:`Event` to fire
        :param date: the date of the event
        """
        self.date = date
        event.run()

    def do_birth(self, mother_id):
        """Create a new child person.

        We create a new person in the population and then call the `on_birth` method in
        all modules to initialise the child's properties.

        :param mother_id: the maternal parent
        :return: the new child
        """
        child_id = self.population.do_birth()
        for module in self.modules.values():
            module.on_birth(mother_id, child_id)
        return child_id


class EventQueue:
    """A simple priority queue for events.

    This doesn't really care what events and dates are, provided dates are comparable
    so we can tell which is least, i.e. earliest.
    """

    def __init__(self):
        """Create an empty event queue."""
        self.counter = itertools.count()
        self.queue = []

    def schedule(self, event, date):
        """Schedule a new event.

        :param event: the event to schedule
        :param date: when it should happen
        """

        entry = (date, next(self.counter), event)
        heapq.heappush(self.queue, entry)

    def next_event(self):
        """Get the earliest event in the queue.

        :returns: an (event, date) pair
        """
        date, count, event = heapq.heappop(self.queue)
        return event, date

    def __len__(self):
        """:return: the length of the queue"""
        return len(self.queue)
