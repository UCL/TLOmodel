import os
from pathlib import Path

import pandas as pd

from tlo import Date, Module, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.simulation import EventPriority

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
start_date = Date(2010, 1, 1)


def test_enum_event_priority():
    """Check that the EventPriority Enumeration works can be used and have an identified order (and so can be used to
     determine ordering in the heapq)."""
    assert EventPriority.START_OF_DAY \
           < EventPriority.FIRST_HALF_OF_DAY \
           < EventPriority.LAST_HALF_OF_DAY \
           < EventPriority.END_OF_DAY


def test_control_of_ordering_in_the_day(seed, tmpdir):
    """Check that the ordering of regular events in a day can be controlled
     * Create regular events to occur at the start, middle and end of the day.
     * Schedule in mixed-up order
     * Run simulation
     * Examine order in which the events actually ran on each day
    """

    class Event_For_Start_Of_Day(RegularEvent, PopulationScopeEventMixin):

        def __init__(self, module):
            super().__init__(module, frequency=pd.DateOffset(days=1), event_priority=EventPriority.START_OF_DAY)

        def apply(self, population):
            logger = logging.getLogger('tlo.simulation')
            logger.info(key='event', data={'id': self.__class__.__name__})
            assert self._event_priority == EventPriority.START_OF_DAY

    class Event_For_Middle_Of_Day(RegularEvent, PopulationScopeEventMixin):

        def __init__(self, module):
            super().__init__(module, frequency=pd.DateOffset(days=1))  # order_in_day argument not given

        def apply(self, population):
            logger = logging.getLogger('tlo.simulation')
            logger.info(key='event', data={'id': self.__class__.__name__})
            assert self._event_priority is None

    class Event_For_Second_to_Last_At_End_Of_Day(RegularEvent, PopulationScopeEventMixin):

        def __init__(self, module):
            super().__init__(module, frequency=pd.DateOffset(days=1), event_priority=EventPriority.LAST_HALF_OF_DAY)

        def apply(self, population):
            logger = logging.getLogger('tlo.simulation')
            logger.info(key='event', data={'id': self.__class__.__name__})
            assert self._event_priority == EventPriority.LAST_HALF_OF_DAY

    class Event_For_End_Of_Day(RegularEvent, PopulationScopeEventMixin):

        def __init__(self, module):
            super().__init__(module, frequency=pd.DateOffset(days=1), event_priority=EventPriority.END_OF_DAY)

        def apply(self, population):
            logger = logging.getLogger('tlo.simulation')
            logger.info(key='event', data={'id': self.__class__.__name__})
            assert self._event_priority == EventPriority.END_OF_DAY

    class DummyModule(Module):

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            sim.schedule_event(Event_For_Second_to_Last_At_End_Of_Day(self), sim.date,
                               event_priority=EventPriority.LAST_HALF_OF_DAY)
            sim.schedule_event(Event_For_Middle_Of_Day(self), sim.date)  # No `event_priority` argument provided
            sim.schedule_event(Event_For_End_Of_Day(self), sim.date, event_priority=EventPriority.END_OF_DAY)
            sim.schedule_event(Event_For_Start_Of_Day(self), sim.date, event_priority=EventPriority.START_OF_DAY)

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
    assert order_on_day_one == ("Event_For_Start_Of_Day",
                                "Event_For_Middle_Of_Day",
                                "Event_For_Second_to_Last_At_End_Of_Day",
                                "Event_For_End_Of_Day")

    # Check order is the same every day
    dates = pd.to_datetime(events['date']).dt.date.drop_duplicates()
    for day in dates:
        assert order_on_day_one == tuple(events.loc[events['date'] == day, 'id'])
