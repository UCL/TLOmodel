import os
from pathlib import Path

import pandas as pd

from tlo import Date, Module, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.events import PopulationScopeEventMixin, Priority, RegularEvent
from tlo.methods.fullmodel import fullmodel

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


def test_control_of_ordering_in_full_model(seed, tmpdir):
    """Check that the ordering of regular events in the full_model is as expected, i.e.
    * Second-to-last in the day is: `HealthSeekingBehaviourPoll`
    * Last in the day is: `HealthSystemScheduler`
    """
    log_config = {
        "filename": "log",
        "directory": tmpdir,
    }
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)
    sim.register(*fullmodel(resourcefilepath=resourcefilepath))
    sim.make_initial_population(n=100)

    # Replace the `fire_single_event_function` in simulation to a version that include a logging line.
    original = sim.fire_single_event

    def replacement_fire_single_event(*args):
        original(*args)
        logger = logging.getLogger('tlo.simulation')
        event = args[0]
        logger.info(key='events', data={'event': event.__class__.__name__})

    sim.fire_single_event = replacement_fire_single_event

    # Run simulation
    sim.simulate(end_date=start_date + pd.DateOffset(day=10))

    # Retrieve log of every event that has run
    log = parse_log_file(sim.log_filepath)['tlo.simulation']['events'].set_index('date')['event']

    # Check order is as expected every day
    for _date in pd.unique(log.index.date):
        order_of_events = tuple(log.loc[[_date]])
        assert 'HealthSeekingBehaviourPoll' == order_of_events[-2], "HealthSeekingBehaviourPoll is not the " \
                                                                    "second-to-last event of the day"
        assert 'HealthSystemScheduler' == order_of_events[-1], "HealthSystemScheduler is not the last event of the day"
