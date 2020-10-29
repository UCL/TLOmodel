import hashlib
import random

import pandas as pd

from tlo import DateOffset, Simulation, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger('tlo.profiling')
logger.setLevel(logging.INFO)


class LogProgress(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        df = population.props
        logger.info(key="stats", data={"alive": df.is_alive.sum(), "total": len(df)})


def schedule_profile_log(sim: Simulation) -> None:
    """Schedules the log progress event, used only for profiling"""
    sim.schedule_event(LogProgress(sim.modules["Demography"]), sim.start_date)


def dataframe_hash(sim: Simulation) -> str:
    """Returns checksum of the simulation

    Only uses at the population dataframe
    TODO: add simulation queue
    """
    return hashlib.sha1(pd.util.hash_pandas_object(sim.population.props).values).hexdigest()


def print_checksum(sim: Simulation) -> None:
    """Output checksum of dataframe to screen"""
    logger.info(key="msg", data=f"Population checksum: {dataframe_hash(sim)}")


def save_population(sim: Simulation) -> None:
    df: pd.DataFrame = sim.population.props
    filename = 'profiling_population_%010x.pickle' % random.randrange(16**10)
    df.to_pickle(filename)
    logger.info(key="msg", data=f"Pickled population dataframe: {filename}")
