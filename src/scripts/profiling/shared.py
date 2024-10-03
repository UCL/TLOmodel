import datetime
import random
import time

import pandas as pd
import psutil

from tlo import DateOffset, Simulation, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.util import hash_dataframe

logger = logging.getLogger("tlo.profiling")
logger.setLevel(logging.INFO)


def memory_statistics() -> dict[str, float]:
    """
    Extract memory usage statistics in current process using `psutil`.
    Statistics are returned as a dictionary.
    
    Key / value pairs are:
    memory_rss_MiB: float
        Resident set size in mebibytes. The non-swapped physical memory the process has used.
    memory_vms_MiB: float
        Virtual memory size in mebibytes. The total amount of virtual memory used by the process.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "memory_rss_MiB": memory_info.rss / 2**20,
        "memory_vms_MiB": memory_info.vms / 2**20,
    }


class LogProgress(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module, frequency_months=3):
        super().__init__(module, frequency=DateOffset(months=frequency_months))
        self.time = time.time()

    def apply(self, population):
        df = population.props
        now = time.time()
        duration = (now - self.time) / 60  # minutes
        self.time = now
        logger.info(
            key="stats",
            data={
                "time": datetime.datetime.now().isoformat(),
                "duration_minutes": duration,
                "pop_df_number_alive": df.is_alive.sum(),
                "pop_df_rows": len(df),
                "pop_df_mem_MiB": df.memory_usage(index=True, deep=True).sum() / 2**20,
                **memory_statistics(),
            },
        )


def schedule_profile_log(sim: Simulation, frequency_months: int = 3) -> None:
    """Schedules the log progress event, used only for profiling"""
    sim.schedule_event(LogProgress(sim.modules["Demography"], frequency_months), sim.start_date)


def print_checksum(sim: Simulation) -> None:
    """Output checksum of dataframe to screen"""
    logger.info(
        key="msg", data=f"Population checksum: {hash_dataframe(sim.population.props)}"
    )


def save_population(sim: Simulation) -> None:
    df: pd.DataFrame = sim.population.props
    filename = "profiling_population_%010x.pickle" % random.randrange(16**10)
    df.to_pickle(filename)
    logger.info(key="msg", data=f"Pickled population dataframe: {filename}")
