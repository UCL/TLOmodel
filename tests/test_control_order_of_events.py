import os
from pathlib import Path

import pandas as pd

from tlo import Date, Module, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.events import PopulationScopeEventMixin, Priority, RegularEvent

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
start_date = Date(2010, 1, 1)


def test_priority_enum():
    """Check that the EventPriority Enumeration works can be used and have an identified order (and so can be used to
     determine ordering in the heapq)."""
    assert Priority.START_OF_DAY \
           < Priority.FIRST_HALF_OF_DAY \
           < Priority.LAST_HALF_OF_DAY \
           < Priority.END_OF_DAY


def test_control_of_ordering_in_the_day(seed, tmpdir):
    """Check that the ordering of regular events in a day can be controlled
     * Create regular events to occur at the start, middle and end of the day.
     * Schedule in mixed-up order
     * Run simulation
     * Examine order in which the events actually ran on each day
    """

    class EventForStartOfDay(RegularEvent, PopulationScopeEventMixin):
        def apply(self, population):
            logger = logging.getLogger('tlo.simulation')
            logger.info(key='event', data={'id': self.__class__.__name__})
            assert self.priority == Priority.START_OF_DAY

    class EventForMiddleOfDay(RegularEvent, PopulationScopeEventMixin):
        def apply(self, population):
            logger = logging.getLogger('tlo.simulation')
            logger.info(key='event', data={'id': self.__class__.__name__})
            assert self.priority is Priority.FIRST_HALF_OF_DAY

    class EventForSecondToLastAtEndOfDay(RegularEvent, PopulationScopeEventMixin):
        def apply(self, population):
            logger = logging.getLogger('tlo.simulation')
            logger.info(key='event', data={'id': self.__class__.__name__})
            assert self.priority == Priority.LAST_HALF_OF_DAY

    class EventForEndOfDay(RegularEvent, PopulationScopeEventMixin):
        def apply(self, population):
            logger = logging.getLogger('tlo.simulation')
            logger.info(key='event', data={'id': self.__class__.__name__})
            assert self.priority == Priority.END_OF_DAY

    class DummyModule(Module):
        def on_birth(self, mother, child):
            pass

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            one_day = pd.DateOffset(days=1)
            # No `event_priority` argument provided
            sim.schedule_event(EventForMiddleOfDay(self, frequency=one_day), sim.date)
            sim.schedule_event(EventForSecondToLastAtEndOfDay(self,
                                                              frequency=one_day,
                                                              priority=Priority.LAST_HALF_OF_DAY), sim.date)
            sim.schedule_event(EventForEndOfDay(self, frequency=one_day, priority=Priority.END_OF_DAY), sim.date)
            sim.schedule_event(EventForStartOfDay(self, frequency=one_day, priority=Priority.START_OF_DAY), sim.date)

    log_config = {
        'filename': 'tmpfile',
        'directory': tmpdir,
        'custom_levels': {
            "tlo.simulation": logging.INFO,
        }
    }
    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed, log_config=log_config)
    sim.register(DummyModule())
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.start_date + pd.DateOffset(days=10))

    # Examine order in which the events actually ran on each day
    events = parse_log_file(sim.log_filepath)['tlo.simulation']['event'].reset_index()

    # Check that order is as expected: Start -> Middle --> End
    events['date'] = pd.to_datetime(events['date']).dt.date
    order_on_day_one = tuple(events.loc[events['date'] == Date(2010, 1, 1), 'id'])
    assert order_on_day_one == ("EventForStartOfDay",
                                "EventForMiddleOfDay",
                                "EventForSecondToLastAtEndOfDay",
                                "EventForEndOfDay")

    # Check order is the same every day
    dates = pd.to_datetime(events['date']).dt.date.drop_duplicates()
    for day in dates:
        assert order_on_day_one == tuple(events.loc[events['date'] == day, 'id'])
