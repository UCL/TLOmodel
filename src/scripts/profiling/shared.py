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


def schedule_profile_log(sim: Simulation):
    sim.schedule_event(LogProgress(sim.modules["Demography"]), sim.start_date)
