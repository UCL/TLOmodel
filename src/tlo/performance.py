"""A module to collect metrics, checkpoints, hashes and a variety of other information about the simulation as it runs,
for debugging and performance monitoring purposes."""
import datetime
import time
from dataclasses import dataclass
from pathlib import Path

from tlo import DateOffset, Module, Simulation, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.util import hash_dataframe

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

@dataclass
class MagpieOptions:
    log_perf: bool = False
    log_perf_freq: int = 1
    log_pop_hash: bool = False
    save_sim: bool = False
    save_sim_freq: int = 6
    save_sim_on_end: bool = False
    wall_time_limit_minutes: float | None = None


class Magpie(Module):
    """
    Register in simulation:
    ```
        Magpie(
            log_perf=True,           # turn on logging of performance statistics...
            log_perf_freq=2,         # ...every n months
            log_pop_hash=True,       # include hash of population dataframe (can be slow)
            save_sim=True,           # save the simulation to a pickle file...
            save_sim_freq=3,         # ...every n months
            save_sim_on_end=True     # save the simulation to a pickle file at the end of the simulation
            wall_time_limit_minutes=60  # stop the simulation if it runs for more than n minutes (checked every month)
        )
    ```

    e.g. to add to an existing scenario:
    ```
       def modules(self):
           return fullmodel() + [Magpie(log_perf=True,
                                log_perf_freq=2, log_pop_hash=True,
                                save_sim=False, save_sim_on_end=True)]
    ```
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.options = MagpieOptions(**kwargs)

    def read_parameters(self, data_folder):
        pass

    def initialise_simulation(self, sim: Simulation):
        if self.options.log_perf:
            event = LogPerfProfile(self, self.options.log_perf_freq, self.options.log_pop_hash)
            sim.schedule_event(event, sim.start_date)

        if self.options.save_sim:
            sim.schedule_event(SaveSimulation(self, self.options.save_sim_freq), sim.start_date)

        if self.options.wall_time_limit_minutes is not None:
            sim.schedule_event(WallTimeLimit(self, self.options.wall_time_limit_minutes), sim.start_date)

    def on_birth(self, mother_id, child_id):
        pass

    def on_simulation_end(self):
        if self.options.save_sim_on_end:
            self.sim.save_to_pickle(Path(make_pickle_filename(self.sim.date)))

class SaveSimulation(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module, frequency_months):
        super().__init__(module, frequency=DateOffset(months=frequency_months))

    def apply(self, population):
        self.sim.save_to_pickle(Path(make_pickle_filename(self.sim.date)))


class LogPerfProfile(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module, frequency_months, do_hash):
        super().__init__(module, frequency=DateOffset(months=frequency_months))
        self.time = time.time()
        self.do_hash = do_hash

    def apply(self, population):
        data = profile_statistics(self.module.sim, self.do_hash)

        now = time.time()
        duration = (now - self.time) / 60  # minutes
        self.time = now
        data["duration_minutes"] = duration
        logger.info(key="stats", data=data)


class WallTimeLimit(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module, wall_time_limit_minutes):
        super().__init__(module, frequency=DateOffset(months=1))
        self.wall_time_limit_minutes = wall_time_limit_minutes
        self.start_time = time.time()

    def apply(self, population):
        now = time.time()
        elapsed_minutes = (now - self.start_time) / 60
        if elapsed_minutes >= self.wall_time_limit_minutes:
            logger.info(
                key="msg",
                data=f"Terminating simulation after {self.wall_time_limit_minutes} minutes of runtime"
            )
            self.sim.terminate()


def make_pickle_filename(sim_date):
    strftime = "%Y%m%dT%H%M%S"
    timestamp = time.strftime(strftime)
    return f"simulation-{sim_date.strftime(strftime)}-{timestamp}.pkl"


def profile_statistics(sim, do_hash):
    df = sim.population.props
    sim_queue_size = len(sim.event_queue)

    # not great, but...hardcoded for now
    if "HealthSystem" in sim.modules:
        hsi_queue_size = len(sim.modules["HealthSystem"].HSI_EVENT_QUEUE)
    else:
        hsi_queue_size = 0

    data = {
        "time": datetime.datetime.now().isoformat(),
        "pop_df_number_alive": df.is_alive.sum(),
        "pop_df_rows": len(df),
        "pop_df_mem_MiB": df.memory_usage(index=True, deep=True).sum() / 2 ** 20,
        "sim_queue_size": sim_queue_size,
        "hsi_queue_size": hsi_queue_size,
        **memory_statistics(),
    }

    if do_hash:
        # warning - allocates lots of memory
        pop_df_hash = hash_dataframe(df)
        data["pop_df_hash"] = pop_df_hash

    return data


def memory_statistics() -> dict[str, float]:
    """
    Extract memory usage statistics in current process using `psutil` if available.
    Statistics are returned as a dictionary. If `psutil` not installed an empty dict is returned.

    Key / value pairs are:
    memory_rss_MiB: float
        Resident set size in mebibytes. The non-swapped physical memory the process has used.
    memory_vms_MiB: float
        Virtual memory size in mebibytes. The total amount of virtual memory used by the process.
    memory_uss_MiB: float
        Unique set size in mebibytes. The memory which is unique to a process and which would be freed if the process
        was terminated right now
    """
    if psutil is None:
        return {}
    process = psutil.Process()
    memory_info = process.memory_full_info()
    return {
        "memory_rss_MiB": memory_info.rss / 2**20,
        "memory_vms_MiB": memory_info.vms / 2**20,
        "memory_uss_MiB": memory_info.uss / 2**20,
    }

