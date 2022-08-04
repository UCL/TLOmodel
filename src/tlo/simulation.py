"""The main simulation controller."""

import datetime
import heapq
import itertools
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Union

import numpy as np

from tlo import Date, Population, logging
from tlo.dependencies import check_dependencies_present, topologically_sort_modules
from tlo.events import Event, IndividualScopeEventMixin
from tlo.progressbar import ProgressBar

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    def __init__(self, *, start_date: Date, seed: int = None, log_config: dict = None,
                 show_progress_bar=False):
        """Create a new simulation.

        :param start_date: the date the simulation begins; must be given as
            a keyword parameter for clarity
        :param seed: the seed for random number generator. class will create one if not supplied
        :param log_config: sets up the logging configuration for this simulation
        :param show_progress_bar: whether to show a progress bar instead of the logger
            output during the simulation
        """
        # simulation
        self.date = self.start_date = start_date
        self.modules = OrderedDict()
        self.event_queue = EventQueue()
        self.end_date = None
        self.output_file = None

        self.show_progress_bar = show_progress_bar

        # logging
        if log_config is None:
            log_config = dict()
        self._custom_log_levels = None
        self._log_filepath = None
        self.configure_logging(**log_config)

        # random number generator
        seed_from = 'auto' if seed is None else 'user'
        self._seed = seed
        self._seed_seq = np.random.SeedSequence(seed)
        logger.info(
            key='info',
            data=f'Simulation RNG {seed_from} entropy = {self._seed_seq.entropy}'
        )
        self.rng = np.random.RandomState(np.random.MT19937(self._seed_seq))

    def configure_logging(self, filename: str = None, directory: Union[Path, str] = "./outputs",
                          custom_levels: Dict[str, int] = None, suppress_stdout: bool = False):
        """Configure logging, can write logging to a logfile in addition the default of stdout.

        Minimum custom levels for each loggers can be specified for filtering out messages

        :param filename: Prefix for logfile name, final logfile will have a datetime appended
        :param directory: Path to output directory, default value is the outputs folder.
        :param custom_levels: dictionary to set logging levels, '*' can be used as a key for all registered modules.
                              This is likely to be used to disable all disease modules, and then enable one of interest
                              e.g. ``{'*': logging.CRITICAL 'tlo.methods.hiv': logging.INFO}``
        :param suppress_stdout: If True, suppresses logging to standard output stream (default is False)

        :return: Path of the log file if a filename has been given.
        """
        # clear logging environment
        # if using progress bar we do not print log messages to stdout to avoid
        # clashes between progress bar and log output
        logging.init_logging(add_stdout_handler=not (self.show_progress_bar or suppress_stdout))
        logging.set_simulation(self)

        if custom_levels:
            # if modules have already been registered
            if self.modules:
                module_paths = (module.__module__ for module in self.modules.values())
                logging.set_logging_levels(custom_levels, module_paths)
            else:
                # save the configuration and apply in the `register` phase
                self._custom_log_levels = custom_levels

        if filename:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')
            log_path = Path(directory) / f"{filename}__{timestamp}.log"
            self.output_file = logging.set_output_file(log_path)
            logger.info(key='info', data=f'Log output: {log_path}')
            self._log_filepath = log_path
            return log_path

    @property
    def log_filepath(self):
        return self._log_filepath

    def _set_module_log_level(self, module_path, level):
        logging.set_logging_levels({module_path: level}, [module_path])

    def register(self, *modules, sort_modules=True, check_all_dependencies=True):
        """Register one or more disease modules with the simulation.

        :param modules: the disease module(s) to use as part of this simulation.
            Multiple modules may be given as separate arguments to one call.
        :param sort_modules: Whether to topologically sort the modules so that any
            initialisation dependencies (specified by the ``INIT_DEPENDENCIES``
            attribute) of a module are initialised before the module itself is. A
            ``ModuleDependencyError`` exception will be raised if there are missing
            initialisation dependencies or circular initialisation dependencies between
            modules that cannot be resolved. If this flag is set to ``True`` there is
            also a requirement that at most one instance of each module is registered
            and ``MultipleModuleInstanceError`` will be raised if this is not the case.
        :param check_all_dependencies: Whether to check if all of each modules declared
            dependencies (that is, the union of the ``INIT_DEPENDENCIES`` and
            ``ADDITIONAL_DEPENDENCIES`` attributes) have been included in the set of
            modules to be registered. A ``ModuleDependencyError`` exception will
            be raised if there are missing dependencies.
        """
        if sort_modules:
            modules = list(topologically_sort_modules(modules))
        if check_all_dependencies:
            check_dependencies_present(modules)
        # Iterate over modules and per-module seed sequences spawned from simulation
        # level seed sequence
        for module, seed_seq in zip(modules, self._seed_seq.spawn(len(modules))):
            assert module.name not in self.modules, (
                'A module named {} has already been registered'.format(module.name))

            # Seed the RNG for the registered module using spawned seed sequence
            logger.info(
                key='info',
                data=(
                    f'{module.name} RNG auto (entropy, spawn key) = '
                    f'({seed_seq.entropy}, {seed_seq.spawn_key[0]})'
                )
            )
            module.rng = np.random.RandomState(np.random.MT19937(seed_seq))

            # if user provided custom log levels
            if self._custom_log_levels is not None:
                # get the log level of this module
                path = module.__module__
                if path in self._custom_log_levels:
                    self._set_module_log_level(path, self._custom_log_levels[path])
                elif '*' in self._custom_log_levels:
                    self._set_module_log_level(path, self._custom_log_levels['*'])

            self.modules[module.name] = module
            module.sim = self
            module.read_parameters('')

    def make_initial_population(self, *, n):
        """Create the initial population to simulate.

        :param n: the number of individuals to create; must be given as
            a keyword parameter for clarity
        """
        start = time.time()

        # Collect information from all modules, that is required the population dataframe
        for module in self.modules.values():
            module.pre_initialise_population()

        # Make the initial population
        self.population = Population(self, n)
        for module in self.modules.values():
            start1 = time.time()
            module.initialise_population(self.population)
            logger.debug(key='debug', data=f'{module.name}.initialise_population() {time.time() - start1} s')

        end = time.time()
        logger.info(key='info', data=f'make_initial_population() {end - start} s')

    def simulate(self, *, end_date):
        """Simulation until the given end date

        :param end_date: when to stop simulating. Only events strictly before this
            date will be allowed to occur.
            Must be given as a keyword parameter for clarity.
        """
        start = time.time()
        self.end_date = end_date  # store the end_date so that others can reference it

        for module in self.modules.values():
            module.initialise_simulation(self)

        if self.show_progress_bar:
            start_date = self.date
            num_simulated_days = (end_date - start_date).days
            progress_bar = ProgressBar(
                num_simulated_days, "Simulation progress", unit="day")
            progress_bar.start()

        while self.event_queue:
            event, date = self.event_queue.next_event()

            if self.show_progress_bar:
                simulation_day = (date - start_date).days
                progress_bar.update(
                    simulation_day,
                    stats_dict={"date": str(date.date())}
                )

            if date >= end_date:
                self.date = end_date
                break
            self.fire_single_event(event, date)

        # The simulation has ended. Call 'on_simulation_end' method at the end of simulation
        if self.show_progress_bar:
            progress_bar.stop()

        # The simulation has ended. Call 'on_simulation_end' method at the end of simulation (if a module has it)
        for module in self.modules.values():
            module.on_simulation_end()

        # complete logging
        if self.output_file:
            self.output_file.flush()
            self.output_file.close()

        logger.info(key='info', data=f'simulate() {time.time() - start} s')

    def schedule_event(self, event, date):
        """Schedule an event to happen on the given future date.

        :param event: the Event to schedule
        :param date: when the event should happen
        """
        assert date >= self.date, 'Cannot schedule events in the past'

        assert 'TREATMENT_ID' not in dir(event), \
            'This looks like an HSI event. It should be handed to the healthsystem scheduler'
        assert (event.__str__().find('HSI_') < 0), \
            'This looks like an HSI event. It should be handed to the healthsystem scheduler'
        assert isinstance(event, Event)

        self.event_queue.schedule(event=event, date=date)

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

    def find_events_for_person(self, person_id: int):
        """Find the events in the queue for a particular person.
        :param person_id: the person_id of interest
        :returns list of tuples (date_of_event, event) for that person_id in the queue.

        NB. This is for debugging and testing only - not for use in real simulations as it is slow
        """
        person_events = list()

        for date, _, _, event in self.event_queue.queue:
            if isinstance(event, IndividualScopeEventMixin):
                if event.target == person_id:
                    person_events.append((date, event))

        return person_events


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
        entry = (date, event.priority, next(self.counter), event)
        heapq.heappush(self.queue, entry)

    def next_event(self):
        """Get the earliest event in the queue.

        :returns: an (event, date) pair
        """
        date, _, _, event = heapq.heappop(self.queue)
        return event, date

    def __len__(self):
        """:return: the length of the queue"""
        return len(self.queue)
