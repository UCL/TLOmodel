"""The main simulation controller."""

import datetime
import heapq
import itertools
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd
import numpy as np

from tlo import Date, Population, logging
from tlo.dependencies import (
    check_dependencies_present,
    initialise_missing_dependencies,
    topologically_sort_modules,
)
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
                 show_progress_bar=False, resourcefilepath: Optional[Path] = None):
        """Create a new simulation.

        :param start_date: the date the simulation begins; must be given as
            a keyword parameter for clarity
        :param seed: the seed for random number generator. class will create one if not supplied
        :param log_config: sets up the logging configuration for this simulation
        :param show_progress_bar: whether to show a progress bar instead of the logger
            output during the simulation
        :param resourcefilepath: Path to resource files folder. Assign ``None` if no path is provided.
        """
        # simulation
        self.date = self.start_date = start_date
        self.modules = OrderedDict()
        self.event_queue = EventQueue()
        self.generate_data = None
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
        seed_from = 'auto' if seed is None else 'user'
        self._seed = seed
        self._seed_seq = np.random.SeedSequence(seed)
        logger.info(
            key='info',
            data=f'Simulation RNG {seed_from} entropy = {self._seed_seq.entropy}'
        )
        self.rng = np.random.RandomState(np.random.MT19937(self._seed_seq))

    def _configure_logging(self, filename: str = None, directory: Union[Path, str] = "./outputs",
                           custom_levels: Dict[str, int] = None, suppress_stdout: bool = False):
        """Configure logging, can write logging to a logfile in addition the default of stdout.

        Minimum custom levels for each logger can be specified for filtering out messages

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
            timestamp = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')
            log_path = Path(directory) / f"{filename}__{timestamp}.log"
            self.output_file = logging.set_output_file(log_path)
            logger.info(key='info', data=f'Log output: {log_path}')
            return log_path

        return None

    @property
    def log_filepath(self):
        """The path to the log file, if one has been set."""
        return self._log_filepath

    def register(self, *modules, sort_modules=True, check_all_dependencies=True, auto_register_dependencies: bool = False):
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
        :param check_all_dependencies: Whether to check if all of each module's declared
            dependencies (that is, the union of the ``INIT_DEPENDENCIES`` and
            ``ADDITIONAL_DEPENDENCIES`` attributes) have been included in the set of
            modules to be registered. A ``ModuleDependencyError`` exception will
            be raised if there are missing dependencies.
        :param auto_register_dependencies: Whether to register missing module dependencies or not. If this argument is
         set to True, all module dependencies will be automatically registered.
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
            assert module.name not in self.modules, f'A module named {module.name} has already been registered'

            # Seed the RNG for the registered module using spawned seed sequence
            logger.info(
                key='info',
                data=(
                    f'{module.name} RNG auto (entropy, spawn key) = '
                    f'({seed_seq.entropy}, {seed_seq.spawn_key[0]})'
                )
            )
            module.rng = np.random.RandomState(np.random.MT19937(seed_seq))

            self.modules[module.name] = module
            module.sim = self
            module.read_parameters('')

        if self._custom_log_levels:
            logging.set_logging_levels(self._custom_log_levels)

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
        properties = {
            name: prop
            for module in self.modules.values()
            for name, prop in module.PROPERTIES.items()
        }
        self.population = Population(properties, n)
        for module in self.modules.values():
            start1 = time.time()
            module.initialise_population(self.population)
            logger.debug(key='debug', data=f'{module.name}.initialise_population() {time.time() - start1} s')

        self.event_chains = pd.DataFrame(columns= list(self.population.props.columns)+['person_ID'] + ['event'] + ['event_date'] + ['when'])

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
        self.generate_data = True # for now ensure we're always aiming to print data

        f = open('output.txt', mode='a')
        #df_event_chains = pd.DataFrame(columns= list(self.population.props.columns)+['person_ID'] + ['event'] + ['event_date'] + ['when'])

        # Reorder columns to place the new columns at the front
        pd.set_option('display.max_columns', None)
        print(self.event_chains.columns)
        for module in self.modules.values():
            module.initialise_simulation(self)

        progress_bar = None
        if self.show_progress_bar:
            num_simulated_days = (end_date - self.start_date).days
            progress_bar = ProgressBar(
                num_simulated_days, "Simulation progress", unit="day")
            progress_bar.start()

        while self.event_queue:
            event, date = self.event_queue.next_event()

            if self.show_progress_bar:
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

            if date >= end_date:
                self.date = end_date
                self.event_chains.to_csv('output.csv', index=False)
                break

            #if event.target != self.population:
            #    print("Event: ", event)
            go_ahead = False
            df_before = []
            
            # Only print events relevant to modules of interest
            # Do not want to compare before/after in births because it may expand the pop dataframe
            print_output = True
            if print_output:
                if (event.module == self.modules['Tb'] or event.module == self.modules['Hiv']) and 'TbActiveCasePollGenerateData' not in str(event) and 'HivPollingEventForDataGeneration' not in str(event) and "SimplifiedBirthsPoll" not in str(event) and "AgeUpdateEvent" not in str(event) and "HealthSystemScheduler" not in str(event):
                #if 'TbActiveCasePollGenerateData' not in str(event) and 'HivPollingEventForDataGeneration' not in str(event) and "SimplifiedBirthsPoll" not in str(event) and "AgeUpdateEvent" not in str(event):
                    go_ahead = True
                    if event.target != self.population:
                        row = self.population.props.iloc[[event.target]]
                        row['person_ID'] = event.target
                        row['event'] = event
                        row['event_date'] = date
                        row['when'] = 'Before'
                        self.event_chains = pd.concat([self.event_chains, row], ignore_index=True)
                    else:
                        df_before = self.population.props.copy()
                    
            self.fire_single_event(event, date)
            
            if print_output:
                if go_ahead == True:
                    if event.target != self.population:
                        row = self.population.props.iloc[[event.target]]
                        row['person_ID'] = event.target
                        row['event'] = event
                        row['event_date'] = date
                        row['when'] = 'After'
                        self.event_chains = pd.concat([self.event_chains, row], ignore_index=True)
                    else:
                        df_after = self.population.props.copy()
                       # if not df_before.columns.equals(df_after.columns):
                       #     print("Number of columns in pop dataframe", len(self.population.props.columns))
                       #     print("Before", df_before.columns)
                       #     print("After", df_after.columns#)
                      #      exit(-1)
                      #  if not df_before.index.equals(df_after.index):
                       #     print("Number of indices in pop dataframe", len(self.population.props.index))
                      #      print("----> ", event)
                      #      print("Before", df_before.index#)
                      #      print("After", df_after.index)
                      #      exit(-1)
                            
                        change = df_before.compare(df_after)
                        if ~change.empty:
                            indices = change.index
                            new_rows_before = df_before.loc[indices]
                            new_rows_before['person_ID'] = new_rows_before.index
                            new_rows_before['event'] = event
                            new_rows_before['event_date'] = date
                            new_rows_before['when'] = 'Before'
                            new_rows_after = df_after.loc[indices]
                            new_rows_after['person_ID'] = new_rows_after.index
                            new_rows_after['event'] = event
                            new_rows_after['event_date'] = date
                            new_rows_after['when'] = 'After'

                            self.event_chains = pd.concat([self.event_chains,new_rows_before], ignore_index=True)
                            self.event_chains = pd.concat([self.event_chains,new_rows_after], ignore_index=True)

        # The simulation has ended.
        if self.show_progress_bar:
            progress_bar.stop()

        for module in self.modules.values():
            module.on_simulation_end()

        logger.info(key='info', data=f'simulate() {time.time() - start} s')

        # From Python logging.shutdown
        if self.output_file:
            try:
                self.output_file.acquire()
                self.output_file.flush()
                self.output_file.close()
            except (OSError, ValueError):
                pass
            finally:
                self.output_file.release()

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
        person_events = []

        for date, _, _, event in self.event_queue.queue:
            if isinstance(event, IndividualScopeEventMixin):
                if event.target == person_id:
                    person_events.append((date, event))

        return person_events


class EventQueue:
    """A simple priority queue for events.

    This doesn't really care what events and dates are, provided dates are comparable.
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
