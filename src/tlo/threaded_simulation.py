from threading import Thread
from time import sleep
from typing import Callable, List
from queue import Queue
from warnings import warn

from tlo.simulation import _BaseSimulation
from tlo.events import Event, IndividualScopeEventMixin

MAX_THREADS = 4 # make more elegant, probably examine the OS

class ThreadController:
    """
    Thread controllers serve an organisational role, and allow us to
    keep track of threads that we create for debugging purposes.
    They also provide convenient wrapper functions to batch start the
    threads they control all at once, and will manage the teardown of
    their own threads when this is ready.

    Threads spawned by the controller are intended to form a "pool" of
    workers that will routinely check a Queue object for tasks to
    perform, and otherwise will be idle. Worker targets should be
    functions that allow the thread access to the job queue, whilst
    they persist.
    """
    _n_threads: int
    _thread_list: List[Thread]

    _worker_name: str

    @property
    def n_threads(self) -> int:
        """
        Number of threads that this controller is operating.
        """
        return self._n_threads

    def __init__(self, n_threads: int = 1, name: str = "Worker") -> None:
        """
        Create a new thread controller.

        :param n_threads: Number of threads to be spawned, in addition to
        the main thread.
        :param name: Name to assign to worker threads that this controller
        creates, for logging and internal ID purposes.
        """
        # Determine how many threads to use given the machine maximum,
        # and the user's request. Be sure to save one for the main thread!
        self._n_threads = min(n_threads, MAX_THREADS - 1)
        if self._n_threads < n_threads:
            warn(
                f"Requested {n_threads} but this exceeds the maximum possible number of worker threads ({MAX_THREADS - 1}). Restricting to {self._n_threads}."
            )
        assert (
            self._n_threads > 0
        ), f"Instructed to use {self._n_threads} threads, which must be non-negative. Use a serial simulation if you do not want to delegate event execution to threads."

        # Prepare the list of threads, but do not initialise threads yet
        # since they need access to some of the Simulation properties
        self._thread_list = []

        self._worker_name = name

    def create_all(self, target: Callable[[], None]) -> None:
        """
        Creates the threads that will be managed by this controller,
        and sets their targets.

        Targets are not executed until the start_all method is called.

        Targets are functions that take no arguments and return
        no values. Workers will execute these functions - preserving
        context and access of the functions that are passed in.
        Passing in something like foo.bar will provide access to the
        foo object and attempt to run the bar method on said object,
        for example.
        """
        for i in range(self._n_threads):
            self._thread_list.append(
                Thread(target=target, daemon=True, name=f"{self._worker_name}-{i}")
            )

    def start_all(self) -> None:
        """
        Start all threads managed by this controller.
        """
        for thread in self._thread_list:
            thread.start()


class ThreadedSimulation(_BaseSimulation):
    """
    Class for running threaded simulations. Events in the queue that can
    be executed in parallel are delegated to a worker pool, to be executed
    when resources become available.

    Certain events cannot be executed in parallel threads safely (notably
    population-level events, but also events that attempt to advance time).
    When encountering such events, all workers complete the remaining
    "thread-safe" events before the unsafe event is triggered.

    Progress bar for threaded simulations only advances when time advances,
    and statistics do not dynamically update as each event is fired.

    TODO: Prints to actually using the logger
    TODO: Prints to include the worker thread they were spit out from
    """
    # Tracks the job queue that will be dispatched to worker threads
    _worker_queue: Queue
    # Workers must always work on different individuals due to
    # their ability to edit the population DataFrame.
    _worker_patient_targets: set
    # Provides programmatic access to the threads created for the
    # simulation
    thread_controller: ThreadController

    # Safety-catch variables to ensure safe execution of events.
    _individuals_currently_being_examined: set

    def __init__(self, n_threads: int = 1, **kwargs) -> None:
        """
        In addition to the usual simulation instantiation arguments,
        threaded simulations must also be passed the number of
        worker threads to be used.

        :param n_threads: Number of threads to use - in addition to
        the main thread - when running simulation events.
        """
        # Initialise as you would for any other simulation
        super().__init__(**kwargs)

        # Progress bar currently not supported
        self.show_progress_bar = False

        # Setup the thread controller
        self.thread_controller = ThreadController(n_threads=n_threads, name = "EventWorker-")

        # Set the target workflow of all workers
        self.thread_controller.create_all(self._worker_target)

        self._worker_queue = Queue()
        # Initialise the set tracking which individuals the event workers
        # are currently targeting.
        self._worker_patient_targets = set()

    def _worker_target(self) -> None:
        """
        Workflow that threads will execute.

        The workflow assumes that events added to the worker queue
        are always safe to execute in any thread, alongside any
        other events that might currently be in the queue.
        """
        # While thread/worker is alive
        # WOULD LIKE TO NOT HAVE THIS. We could spawn threads only when they're needed
        # and then limit the number we have spawned at once, but creating a thread is also an expensive operation.
        # Plus, the .get() method puts the thread to sleep until it gets something, so this should be fine.
        while True:
            # Check for the next job in the queue
            event_to_run: Event = self._worker_queue.get()
            target = event_to_run.target
            # Wait for other events targeting the same individual to complete
            while target in self._worker_patient_targets:
                # Stall if another thread is currently executing an event
                # which targets the same individual.
                # Add some sleep time here to avoid near-misses.
                sleep(0.01)
            # Flag that this thread is running an event on this patient
            self._worker_patient_targets.add(target)
            event_to_run.run()
            self._worker_patient_targets.remove(target)
            # Report success and await next task
            self._worker_queue.task_done()

    @staticmethod
    def event_must_run_in_main_thread(event: Event) -> bool:
        """
        Return True if the event passed in must be run in the main thread, in serial.

        Population-level events must always run in the main thread with no worker
        events running in parallel, since they need to scan the state of the simulation
        at that moment in time and workers have write access to simulation properties.
        """
        if not isinstance(event, IndividualScopeEventMixin):
            return True
        return False

    def step_through_events(self) -> None:
        # Start the threads
        self.thread_controller.start_all()

        # Whilst the event queue is not empty
        while self.event_queue:
            event, date = self.event_queue.next_event()

            # If the simulation should end, escape
            if date >= self.end_date:
                break
            # If we want to advance time, we need to ensure that
            # the worker queue. Otherwise, a worker might be running an
            # event from the previous date but may still call sim.date
            # to get the "current" time, which would then be out-of-sync.
            elif date != self.date:
                # This event moves time forward, wait until all jobs
                # from the current day have finished before advancing time
                self.wait_for_workers()
                # All jobs from the previous day have ended.
                # Advance time and continue.
                self.date = date
                self.update_progress_bar(self.date)

            # Next, determine if the event to be run can be delegated to the
            # worker pool.
            if self.event_must_run_in_main_thread(event):
                print("MAIN THREAD: Waiting to run population level event...")
                # Event needs all workers to finish, then to run in
                # the main thread (this one)
                self.wait_for_workers()
                print("running", flush=True, end="...")
                event.run()
                print("done")
            else:
                # This job can be delegated to the worker pool, and run safely
                self._worker_queue.put(event)

        # We may have exhausted all the events in the queue, but the workers will
        # still need time to process them all!
        self.wait_for_workers()
        print("MAIN THREAD: Simulation has now ended, worker queue empty.")

    def wait_for_workers(self) -> None:
        """
        Pauses simulation progression until all worker threads
        are ready and waiting to receive a new job.
        """
        self._worker_queue.join()
