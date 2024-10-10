"""The main simulation controller."""

from __future__ import annotations

import datetime
import heapq
import itertools
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union
from typing import TYPE_CHECKING, Optional
import pandas as pd

import numpy as np

try:
    import dill

    DILL_AVAILABLE = True
except ImportError:
    DILL_AVAILABLE = False

from tlo import Date, Population, logging
from tlo.dependencies import (
    check_dependencies_present,
    initialise_missing_dependencies,
    topologically_sort_modules,
)
from tlo.events import Event, IndividualScopeEventMixin
from tlo.progressbar import ProgressBar

if TYPE_CHECKING:
    from tlo.core import Module
    from tlo.logging.core import LogLevel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimulationPreviouslyInitialisedError(Exception):
    """Exception raised when trying to initialise an already initialised simulation."""


class SimulationNotInitialisedError(Exception):
    """Exception raised when trying to run simulation before initialising."""


class Simulation:
    """The main control centre for a simulation.

    This class contains the core simulation logic and event queue, and holds references
    to all the information required to run a complete simulation: the population,
    disease modules, etc.

    Key attributes include:

    :ivar date: The current simulation date.
    :ivar modules: A dictionary of the disease modules used in this simulation, keyed
       by the module name.
    :ivar population: The population being simulated.
    :ivar rng: The simulation-level random number generator. 
    
    .. note::
       Individual modules also have their own random number generator with independent
       state.
    """

    def __init__(
        self,
        *,
        start_date: Date,
        seed: Optional[int] = None,
        log_config: Optional[dict] = None,
        show_progress_bar: bool = False,
        resourcefilepath: Optional[Path] = None,
    ):
        """Create a new simulation.

        :param start_date: The date the simulation begins; must be given as
            a keyword parameter for clarity.
        :param seed: The seed for random number generator. class will create one if not
            supplied
        :param log_config: Dictionary specifying logging configuration for this
            simulation. Can have entries: `filename` - prefix for log file name, final 
            file name will have a date time appended, if not present default is to not
            output log to a file; `directory` - path to output directory to write log
            file to, default if not specified is to output to the `outputs` folder;
            `custom_levels` - dictionary to set logging levels, '*' can be used as a key
            for all registered modules; `suppress_stdout` -  if `True`, suppresses
            logging to standard output stream (default is `False`).
        :param show_progress_bar: Whether to show a progress bar instead of the logger
            output during the simulation.
        :param resourcefilepath: Path to resource files folder. Assign ``None` if no 
            path is provided.
            
        .. note::
           The `custom_levels` entry in `log_config` argument can be used to disable
           logging on all disease modules by setting a high level to `*`, and then
           enabling logging on one module of interest by setting a low level, for
           example ``{'*': logging.CRITICAL 'tlo.methods.hiv': logging.INFO}``.
        """
        # simulation
        self.date = self.start_date = start_date
        self.modules = OrderedDict()
        self.event_queue = EventQueue()
        self.generate_event_chains = None
        self.generate_event_chains_overwrite_epi = None
        self.generate_event_chains_modules_of_interest = []
        self.generate_event_chains_ignore_events = []
        self.end_date = None
        self.output_file = None
        self.population: Optional[Population] = None
        self.event_chains: Optinoal[Population] = None

        self.show_progress_bar = show_progress_bar
        self.resourcefilepath = resourcefilepath

        # logging
        if log_config is None:
            log_config = {}
        self._custom_log_levels = None
        self._log_filepath = self._configure_logging(**log_config)
        

        # random number generator
        seed_from = "auto" if seed is None else "user"
        self._seed = seed
        self._seed_seq = np.random.SeedSequence(seed)
        logger.info(
            key="info",
            data=f"Simulation RNG {seed_from} entropy = {self._seed_seq.entropy}",
        )
        self.rng = np.random.RandomState(np.random.MT19937(self._seed_seq))

        # Whether simulation has been initialised
        self._initialised = False

    def _configure_logging(
        self,
        filename: Optional[str] = None, 
        directory: Path | str = "./outputs",
        custom_levels: Optional[dict[str, LogLevel]] = None,
        suppress_stdout: bool = False
    ):
        """Configure logging of simulation outputs.
         
        Can write log output to a file in addition the default of `stdout`. Mnimum
        custom levels for each logger can be specified for filtering out messages.

        :param filename: Prefix for log file name, final file name will have a date time
            appended.
        :param directory: Path to output directory, default value is the outputs folder.
        :param custom_levels: Dictionary to set logging levels, '*' can be used as a key
            for all registered modules. This is likely to be used to disable logging on
            all disease modules by setting a high level, and then enable one of interest
            by setting a low level, for example
            ``{'*': logging.CRITICAL 'tlo.methods.hiv': logging.INFO}``.
        :param suppress_stdout: If `True`, suppresses logging to standard output stream
            (default is `False`).

        :return: Path of the log file if a filename has been given.
        """
        # clear logging environment
        # if using progress bar we do not print log messages to stdout to avoid
        # clashes between progress bar and log output
        logging.initialise(
            add_stdout_handler=not (self.show_progress_bar or suppress_stdout),
            simulation_date_getter=lambda: self.date.isoformat(),
        )

        if custom_levels:
            # if modules have already been registered
            if self.modules:
                logging.set_logging_levels(custom_levels)
            else:
                # save the configuration and apply in the `register` phase
                self._custom_log_levels = custom_levels

        if filename and directory:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
            log_path = Path(directory) / f"{filename}__{timestamp}.log"
            self.output_file = logging.set_output_file(log_path)
            logger.info(key='info', data=f'Log output: {log_path}')
            return log_path

        return None

    @property
    def log_filepath(self) -> Path:
        """The path to the log file, if one has been set."""
        return self._log_filepath

    def register(
        self,
        *modules: Module,
        sort_modules: bool = True,
        check_all_dependencies: bool = True,
        auto_register_dependencies: bool = False,
    ) -> None:
        """Register one or more disease modules with the simulation.

        :param modules: The disease module(s) to use as part of this simulation.
            Multiple modules may be given as separate arguments to one call.
        :param sort_modules: Whether to topologically sort the modules so that any
            initialisation dependencies (specified by the ``INIT_DEPENDENCIES``
            attribute) of a module are initialised before the module itself is. A
            :py:exc:`.ModuleDependencyError` exception will be raised if there are
            missing initialisation dependencies or circular initialisation dependencies
            between modules that cannot be resolved. If this flag is set to ``True``
            there is also a requirement that at most one instance of each module is
            registered and :py:exc:`.MultipleModuleInstanceError` will be raised if this
            is not the case.
        :param check_all_dependencies: Whether to check if all of each module's declared
            dependencies (that is, the union of the ``INIT_DEPENDENCIES`` and
            ``ADDITIONAL_DEPENDENCIES`` attributes) have been included in the set of
            modules to be registered. A :py:exc:`.ModuleDependencyError` exception will
            be raised if there are missing dependencies.
        :param auto_register_dependencies: Whether to register missing module dependencies
            or not. If this argument is set to True, all module dependencies will be 
            automatically registered.
        """
        if auto_register_dependencies:
            modules = [
                *modules,
                *initialise_missing_dependencies(modules, resourcefilepath=self.resourcefilepath)
            ]

        if sort_modules:
            modules = list(topologically_sort_modules(modules))
        if check_all_dependencies:
            check_dependencies_present(modules)
        # Iterate over modules and per-module seed sequences spawned from simulation
        # level seed sequence
        for module, seed_seq in zip(modules, self._seed_seq.spawn(len(modules))):
            assert (
                module.name not in self.modules
            ), f"A module named {module.name} has already been registered"

            # Seed the RNG for the registered module using spawned seed sequence
            logger.info(
                key="info",
                data=(
                    f"{module.name} RNG auto (entropy, spawn key) = "
                    f"({seed_seq.entropy}, {seed_seq.spawn_key[0]})"
                ),
            )
            module.rng = np.random.RandomState(np.random.MT19937(seed_seq))

            self.modules[module.name] = module
            module.sim = self
            module.read_parameters("")

        if self._custom_log_levels:
            logging.set_logging_levels(self._custom_log_levels)

    def make_initial_population(self, *, n: int) -> None:
        """Create the initial population to simulate.

        :param n: The number of individuals to create; must be given as
            a keyword parameter for clarity.
        """
        start = time.time()

        # Collect information from all modules, that is required the population dataframe
        for module in self.modules.values():
            module.pre_initialise_population()

        # Make the initial population
        properties = {
            name: prop
            for module in self.modules.values()
            for name, prop in module.PROPERTIES.items()
        }
        self.population = Population(properties, n)
        for module in self.modules.values():
            start1 = time.time()
            module.initialise_population(self.population)
            logger.debug(
                key="debug",
                data=f"{module.name}.initialise_population() {time.time() - start1} s",
            )

        self.event_chains = pd.DataFrame(columns= list(self.population.props.columns)+['person_ID'] + ['event'] + ['event_date'] + ['when'] + ['appt_footprint'])

        end = time.time()
        logger.info(key="info", data=f"make_initial_population() {end - start} s")

    def initialise(self, *, end_date: Date, generate_event_chains) -> None:
        """Initialise all modules in simulation.
        :param end_date: Date to end simulation on - accessible to modules to allow
            initialising data structures which may depend (in size for example) on the
            date range being simulated.
        """
        if self._initialised:
            msg = "initialise method should only be called once"
            raise SimulationPreviouslyInitialisedError(msg)
        self.date = self.start_date
        self.end_date = end_date  # store the end_date so that others can reference it

        self.generate_event_chains = generate_event_chains
        if self.generate_event_chains:
            # Eventually this can be made an option
            self.generate_event_chains_overwrite_epi = True
            # For now keep these fixed, eventually they will be input from user
            self.generate_event_chains_modules_of_interest = [self.modules['RTI']]
            self.generate_event_chains_ignore_events =  ['AgeUpdateEvent','HealthSystemScheduler', 'SimplifiedBirthsPoll','DirectBirth'] #['TbActiveCasePollGenerateData','HivPollingEventForDataGeneration','SimplifiedBirthsPoll', 'AgeUpdateEvent', 'HealthSystemScheduler']
        else:
            # If not using to print chains, cannot ignore epi
            self.generate_event_chains_overwrite_epi = False


        # Reorder columns to place the new columns at the front
        pd.set_option('display.max_columns', None)

        for module in self.modules.values():
            module.initialise_simulation(self)
        self._initialised = True

    def finalise(self, wall_clock_time: Optional[float] = None) -> None:
        """Finalise all modules in simulation and close logging file if open.

        :param wall_clock_time: Optional argument specifying total time taken to
            simulate, to be written out to log before closing.
        """
        for module in self.modules.values():
            module.on_simulation_end()
        if wall_clock_time is not None:
            logger.info(key="info", data=f"simulate() {wall_clock_time} s")
        self.close_output_file()

    def close_output_file(self) -> None:
        """Close logging file if open."""
        if self.output_file:
            # From Python logging.shutdown
            try:
                self.output_file.acquire()
                self.output_file.flush()
                self.output_file.close()
            except (OSError, ValueError):
                pass
            finally:
                self.output_file.release()
                self.output_file = None

    def _initialise_progress_bar(self, end_date: Date) -> ProgressBar:
        num_simulated_days = (end_date - self.date).days
        progress_bar = ProgressBar(
            num_simulated_days, "Simulation progress", unit="day"
        )
        progress_bar.start()
        return progress_bar

    def _update_progress_bar(self, progress_bar: ProgressBar, date: Date) -> None:
        simulation_day = (date - self.start_date).days
        stats_dict = {
            "date": str(date.date()),
            "dataframe size": str(len(self.population.props)),
            "queued events": str(len(self.event_queue)),
        }
        if "HealthSystem" in self.modules:
            stats_dict["queued HSI events"] = str(
                len(self.modules["HealthSystem"].HSI_EVENT_QUEUE)
            )
        progress_bar.update(simulation_day, stats_dict=stats_dict)

    def run_simulation_to(self, *, to_date: Date) -> None:
        """Run simulation up to a specified date.

        Unlike :py:meth:`simulate` this method does not initialise or finalise
        simulation and the date simulated to can be any date before or equal to
        simulation end date.

        :param to_date: Date to simulate up to but not including - must be before or
            equal to simulation end date specified in call to :py:meth:`initialise`.
        """
        f = open('output.txt', mode='a')

        if not self._initialised:
            msg = "Simulation must be initialised before calling run_simulation_to"
            raise SimulationNotInitialisedError(msg)
        if to_date > self.end_date:
            msg = f"to_date {to_date} after simulation end date {self.end_date}"
            raise ValueError(msg)
        if self.show_progress_bar:
            progress_bar = self._initialise_progress_bar(to_date)
        while (
            len(self.event_queue) > 0 and self.event_queue.date_of_next_event < to_date
        ):
            event, date = self.event_queue.pop_next_event_and_date()
            if self.show_progress_bar:
                self._update_progress_bar(progress_bar, date)
            self.fire_single_event(event, date)
        self.date = to_date
        self.event_chains.to_csv('output.csv', index=False)

        if self.show_progress_bar:
            progress_bar.stop()

    def simulate(self, *, end_date: Date, generate_event_chains=False) -> None:
        """Simulate until the given end date

        :param end_date: When to stop simulating. Only events strictly before this
            date will be allowed to occur. Must be given as a keyword parameter for
            clarity.
        """
        start = time.time()
        self.initialise(end_date=end_date, generate_event_chains=generate_event_chains)
        self.run_simulation_to(to_date=end_date)
        self.finalise(time.time() - start)

    def schedule_event(self, event: Event, date: Date) -> None:
        """Schedule an event to happen on the given future date.

        :param event: The event to schedule.
        :param date: wWen the event should happen.
        """
        assert date >= self.date, "Cannot schedule events in the past"

        assert "TREATMENT_ID" not in dir(
            event
        ), "This looks like an HSI event. It should be handed to the healthsystem scheduler"
        assert (
            event.__str__().find("HSI_") < 0
        ), "This looks like an HSI event. It should be handed to the healthsystem scheduler"
        assert isinstance(event, Event)

        self.event_queue.schedule(event=event, date=date)

    def fire_single_event(self, event: Event, date: Date) -> None:
        """Fires the event once for the given date

        :param event: :py:class:`Event` to fire.
        :param date: The date of the event.
        """
        self.date = date
        event.run()
        

    def do_birth(self, mother_id: int) -> int:
        """Create a new child person.

        We create a new person in the population and then call the `on_birth` method in
        all modules to initialise the child's properties.

        :param mother_id: Row index label of the maternal parent.
        :return: Row index label of the new child.
        """
        child_id = self.population.do_birth()
        for module in self.modules.values():
            module.on_birth(mother_id, child_id)
        if self.generate_event_chains:
            row = self.population.props.iloc[[child_id]]
            row['person_ID'] = child_id
            row['event'] = 'Birth'
            row['event_date'] = self.date
            row['when'] = 'After'
            self.event_chains = pd.concat([self.event_chains, row], ignore_index=True)
        return child_id

    def find_events_for_person(self, person_id: int) -> list[tuple[Date, Event]]:
        """Find the events in the queue for a particular person.
    
        :param person_id: The row index of the person of interest.
        :return: List of tuples `(date_of_event, event)` for that `person_id` in the
            queue.

        .. note::
           This is for debugging and testing only. Not for use in real simulations as it
           is slow.
        """
        person_events = []

        for date, _, _, event in self.event_queue.queue:
            if isinstance(event, IndividualScopeEventMixin):
                if event.target == person_id:
                    person_events.append((date, event))

        return person_events

    def save_to_pickle(self, pickle_path: Path) -> None:
        """Save simulation state to a pickle file using :py:mod:`dill`.

        Requires :py:mod:`dill` to be importable.

        :param pickle_path: File path to save simulation state to.
        """
        if not DILL_AVAILABLE:
            raise RuntimeError("Cannot save to pickle as dill is not installed")
        with open(pickle_path, "wb") as pickle_file:
            dill.dump(self, pickle_file)

    @staticmethod
    def load_from_pickle(
        pickle_path: Path, log_config: Optional[dict] = None
    ) -> Simulation:
        """Load simulation state from a pickle file using :py:mod:`dill`.

        Requires :py:mod:`dill` to be importable.

        :param pickle_path: File path to load simulation state from.
        :param log_config: New log configuration to override previous configuration. If
            `None` previous configuration (including output file) will be retained. 

        :returns: Loaded :py:class:`Simulation` object.
        """
        if not DILL_AVAILABLE:
            raise RuntimeError("Cannot load from pickle as dill is not installed")
        with open(pickle_path, "rb") as pickle_file:
            simulation = dill.load(pickle_file)
        if log_config is not None:
            simulation._log_filepath = simulation._configure_logging(**log_config)
        return simulation


class EventQueue:
    """A simple priority queue for events.

    This doesn't really care what events and dates are, provided dates are comparable.
    """

    def __init__(self):
        """Create an empty event queue."""
        self.counter = itertools.count()
        self.queue = []

    def schedule(self, event: Event, date: Date) -> None:
        """Schedule a new event.

        :param event: The event to schedule.
        :param date: When it should happen.
        """
        entry = (date, event.priority, next(self.counter), event)
        heapq.heappush(self.queue, entry)

    def pop_next_event_and_date(self) -> tuple[Event, Date]:
        """Get and remove the earliest event and corresponding date in the queue.

        :returns: An `(event, date)` pair.
        """
        date, _, _, event = heapq.heappop(self.queue)
        return event, date

    @property
    def date_of_next_event(self) -> Date:
        """Get the date of the earliest event in queue without removing from queue.

        :returns: Date of next event in queue.
        """
        date, *_ = self.queue[0]
        return date

    def __len__(self) -> int:
        """:return: The length of the queue."""
        return len(self.queue)
