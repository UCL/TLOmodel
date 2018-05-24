"""The main simulation controller."""

import numpy as np


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

    def register(self, *modules):
        """Register one or more disease modules with the simulation.

        :param modules: the disease module(s) to use as part of this simulation.
            Multiple modules may be given as separate arguments to one call.
        """
        for module in modules:
            assert module.name not in self.modules, (
                'A module named {} has already been registered'.format(module.name))
            self.modules[module.name] = module

    def seed_rngs(self, seed):
        """Seed all random number generators with the given seed.

        Each module has its own RNG with its own state. This call will seed them all
        with the same value.

        :param seed: the RNG seed to use
        """
        self.rng.seed(seed)
        for module in self.modules:
            module.rng.seed(seed)

    def make_initial_population(self, *, n):
        """Create the initial population to simulate.

        :param n: the number of individuals to create; must be given as
            a keyword parameter for clarity
        """
        raise NotImplementedError

    def simulate(self, *, end_date):
        """Simulation until the given end date

        :param end_date: when to stop simulating. Only events strictly before this
            date will be allowed to occur.
            Must be given as a keyword parameter for clarity.
        """
        raise NotImplementedError

    def schedule_event(self, event, date):
        """Schedule an event to happen on the given future date.

        :param event: the Event to schedule
        :param date: when the event should happen
        """
        assert date >= self.date, 'Cannot schedule events in the past'
        raise NotImplementedError
