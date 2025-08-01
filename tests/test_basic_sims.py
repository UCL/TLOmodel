
import pandas as pd

from tlo import Date, DateOffset, Module, Property, Simulation, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.test import random_birth, random_death


def test_individual_death():
    # Create a new simulation to orchestrate matters
    sim = Simulation(start_date=Date(2010, 1, 1), seed=1)

    # Register just a test module with random death
    # Note: this approach would allow us to give a different name if desired,
    # so the same module could appear twice, e.g. with different parameters.
    # Not sure if that would be helpful, but...!
    rd = random_death.RandomDeath(name='rd')
    sim.register(rd)
    assert sim.modules['rd'] is rd
    sim.modules['rd'].parameters['death_probability'] = 0.1
    # We can also use attribute-style access if the name doesn't clash
    assert rd.parameters['death_probability'] == 0.1
    rd.parameters['death_probability'] = 0.2

    # Seed the random number generators (manually)
    sim.modules['rd'].rng.seed(1)

    # Create a population of 2 individuals
    sim.make_initial_population(n=2)
    df = sim.population.props
    assert len(df) == 2

    # Test individual-based property access
    assert df.at[0, 'is_alive']

    # Test population-based property access
    assert len(df.is_alive) == 2
    assert isinstance(df.is_alive, pd.Series)
    pd.testing.assert_series_equal(
        pd.Series([True, True]),
        df.is_alive,
        check_names=False)

    # Simulate for 4 months
    assert sim.date == Date(2010, 1, 1)
    sim.simulate(end_date=Date(2010, 5, 1))
    assert sim.date == Date(2010, 5, 1)

    # Check death dates match reference data
    pd.testing.assert_series_equal(
        pd.Series([False, False]),
        df.is_alive,
        check_names=False)
    pd.testing.assert_series_equal(
        pd.Series([Date(2010, 3, 1), Date(2010, 4, 1)]),
        df.date_of_death,
        check_names=False)


def test_single_step_death():
    # This demonstrates how to test the implementation of a single event in isolation

    # Set up minimal simulation
    sim = Simulation(start_date=Date(2010, 1, 1), seed=1)
    rd = random_death.RandomDeath(name='rd')
    rd.parameters['death_probability'] = 0.1
    sim.register(rd)
    sim.modules['rd'].rng.seed(1)
    sim.make_initial_population(n=10)

    # Create and fire the event of interest
    event = random_death.RandomDeathEvent(rd, rd.parameters['death_probability'])
    sim.fire_single_event(event, Date(2010, 2, 1))

    # Check it has behaved as expected
    assert sim.date == Date(2010, 2, 1)

    pd.testing.assert_series_equal(
        pd.Series([True, True, False, True, True, False, True, True, True, True]),
        sim.population.props.is_alive,
        check_names=False
    )


def test_make_test_property(seed):
    # This tests the Population.make_test_property method
    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed)
    sim.make_initial_population(n=3)
    # There should be no properties
    pop = sim.population
    assert len(pop.props.columns) == 0
    # Create one
    pop.make_test_property('test', Types.BOOL)
    # Check it's there
    assert len(pop.props.columns) == 1
    pd.testing.assert_series_equal(
        pd.Series([False, False, False]),
        sim.population.props.test,
        check_names=False
    )


def test_birth_and_death():
    # This combines both population-scope and individual-scope events,
    # with more complex logic.
    # Create a new simulation to orchestrate matters
    sim = Simulation(start_date=Date(1950, 1, 1), seed=2)

    # Register modules
    rb = random_birth.RandomBirth(name='rb')
    rb.parameters['pregnancy_probability'] = 0.1
    rd = random_death.RandomDeath(name='rd')
    rd.parameters['death_probability'] = 0.01
    sim.register(rb, rd)
    assert sim.modules['rb'] is rb
    assert sim.modules['rd'] is rd

    # Seed the random number generators
    sim.modules['rd'].rng.seed(2)
    sim.modules['rb'].rng.seed(2)

    # Create a population of 10 individuals
    sim.make_initial_population(n=10)
    assert len(sim.population.props) == 10

    # Simulate for 2 years
    assert sim.date == Date(1950, 1, 1)
    sim.simulate(end_date=Date(1952, 1, 1))
    assert sim.date == Date(1952, 1, 1)

    # Test further population indexing
    pd.testing.assert_series_equal(
        pd.Series([True, False], index=[4, 5]),
        sim.population.props.loc[4:5, 'is_alive'],
        check_names=False)
    pd.testing.assert_frame_equal(
        pd.DataFrame({'is_alive': [True, True], 'is_pregnant': [False, True]}, index=[8, 9]),
        sim.population.props.loc[8:9, ('is_alive', 'is_pregnant')],
        check_names=False)
    pd.testing.assert_frame_equal(
        pd.DataFrame({'is_alive': [False], 'is_pregnant': [False]}, index=[6]),
        sim.population.props.loc[[6], ('is_alive', 'is_pregnant')],
        check_names=False)

    # Check birth stats match reference data
    assert len(sim.population.props) == 20

    # Check no-one gave birth after dying
    df = sim.population.props

    for person in df.itertuples():
        if not person.is_alive:
            for child_index in person.children:
                assert df.loc[child_index, 'date_of_birth'] <= person.date_of_death

    # Check people can't become pregnant while already pregnant
    for person in df.itertuples():
        # Iterate over pairs of adjacent children
        for child1i, child2i in zip(person.children[:-1], person.children[1:]):
            child1, child2 = df.at[child1i, 'date_of_birth'], \
                             df.at[child2i, 'date_of_birth']
            # Children earlier in the list are born earlier
            assert child1 < child2
            # Birth dates need to be at least 9 months apart
            assert child1 + DateOffset(months=9) <= child2


def test_regular_event_with_end(seed):
    # A small module (that does nothing) and event with end date
    class MyModule(Module):
        PROPERTIES = {'last_run': Property(Types.DATE, '')}

        def read_parameters(self, data_folder): pass

        def initialise_population(self, population): pass

        def initialise_simulation(self, sim): pass

        def on_birth(self, mother, child): pass

    class MyEvent(PopulationScopeEventMixin, RegularEvent):
        def __init__(self, module, end_date):
            # This regular event runs every day but ends as specified
            super().__init__(module=module, frequency=DateOffset(days=1), end_date=end_date)

        def apply(self, population):
            population.props.loc[0, 'last_run'] = self.module.sim.date

    class MyOtherEvent(PopulationScopeEventMixin, RegularEvent):
        def __init__(self, module):
            # This regular event runs every day and does not end
            super().__init__(module=module, frequency=DateOffset(days=1))

        def apply(self, population):
            pass

    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed)

    my_module = MyModule()
    sim.register(my_module)
    sim.make_initial_population(n=1)

    my_event = MyEvent(my_module, end_date=Date(2010, 3, 1))
    my_other_event = MyOtherEvent(my_module)

    sim.schedule_event(my_event, sim.date + DateOffset(days=1))
    sim.schedule_event(my_other_event, sim.date + DateOffset(days=1))
    sim.simulate(end_date=Date(2011, 1, 1))

    # The last update to the data frame is the last event run (not the end of the simulation)
    assert sim.population.props.loc[0, 'last_run'] == pd.Timestamp(Date(2010, 3, 1))
    # The last event the simulation ran was my_other_event that doesn't have end date
    assert sim.date == pd.Timestamp(Date(2011, 1, 1))


def test_show_progress_bar(capfd, seed):
    start_date = Date(2010, 1, 1)
    end_date = Date(2010, 2, 1)
    sim = Simulation(start_date=start_date, seed=seed, show_progress_bar=True)
    logger = logging.getLogger('tlo')
    assert len(logger.handlers) == 0
    rd = random_death.RandomDeath(name='rd')
    sim.register(rd)
    sim.modules['rd'].parameters['death_probability'] = 0.1
    sim.make_initial_population(n=1)
    sim.simulate(end_date=end_date)
    captured = capfd.readouterr()
    assert "Simulation progress" in captured.out
