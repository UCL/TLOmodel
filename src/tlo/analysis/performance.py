import datetime
from pathlib import Path

import time

from tlo import Module, Simulation, DateOffset, logging, Date
from tlo.events import RegularEvent, PopulationScopeEventMixin
from tlo.methods.healthsystem import HealthSystem

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

class PerformanceMonitor(Module):

    def read_parameters(self, data_folder: str | Path) -> None:
        pass

    def initialise_simulation(self, sim: Simulation) -> None:
        sim.schedule_event(LogProgress(self), sim.start_date)
        sim.schedule_event(SaveSimulation(self), sim.start_date)

    def on_birth(self, mother_id: int, child_id: int) -> None:
        pass

    def on_simulation_end(self) -> None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.sim.save_to_pickle(Path(f"simulation-{timestamp}.pkl"))


class SaveSimulation(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module, frequency_months=12):
        super().__init__(module, frequency=DateOffset(months=frequency_months))
        self.time = time.time()

    def apply(self, population):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.module.sim.save_to_pickle(Path(f"simulation-{timestamp}.pkl"))


class LogProgress(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module, frequency_months=3):
        super().__init__(module, frequency=DateOffset(months=frequency_months))
        self.time = time.time()

    def apply(self, population):
        df = population.props
        now = time.time()
        duration = (now - self.time) / 60  # minutes
        self.time = now
        sim = self.module.sim
        sim_queue_size = len(sim.event_queue)
        health_system: "HealthSystem" = sim.modules["HealthSystem"]
        hsi_queue_size = len(health_system.HSI_EVENT_QUEUE)

        logger.info(
            key="stats",
            data={
                "time": datetime.datetime.now().isoformat(),
                "duration_minutes": duration,
                "pop_df_number_alive": df.is_alive.sum(),
                "pop_df_rows": len(df),
                "pop_df_mem_MiB": df.memory_usage(index=True, deep=True).sum() / 2**20,
                "sim_queue_size": sim_queue_size,
                "hsi_queue_size": hsi_queue_size,
                **memory_statistics(),
            },
        )

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

