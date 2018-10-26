
import pandas as pd

from tlo import Date, DateOffset, Person, Simulation, Types
from tlo.test import random_birth, random_death


def test_individual_death():
    # Create a new simulation to orchestrate matters
    sim = Simulation(start_date=Date(2010, 1, 1))

    # Register just a test module with random death
    # Note: this approach would allow us to give a different name if desired,
    # so the same module could appear twice, e.g. with different parameters.
    # Not sure if that would be helpful, but...!
    rd = random_death.RandomDeath(name='rd')
    sim.register(rd)
    assert sim.modules['rd'] is rd
    sim.modules['rd'].parameters['death_probability'] = 0.1
    # We can also use attribute-style access if the name doesn't clash
    assert rd.death_probability == 0.1
    rd.death_probability = 0.2

    # Seed the random number generators
    sim.seed_rngs(1)

    # Create a population of 2 individuals
    sim.make_initial_population(n=2)
    assert len(sim.population) == 2

    # Test individual-based property access
    assert isinstance(sim.population[0], Person)
    assert sim.population[0].is_alive
    assert sim.population[0, 'is_alive']
    assert sim.population[0].props['is_alive'][0]  # Treat this as read-only! It may be removed.

    # Test population-based property access
    assert len(sim.population.is_alive) == 2
    assert isinstance(sim.population.is_alive, pd.Series)
    pd.testing.assert_series_equal(
        pd.Series([True, True]),
        sim.population.is_alive,
        check_names=False)
    assert sim.population.is_alive is sim.population['is_alive']
    assert sim.population.is_alive is sim.population.props['is_alive']  # For now...

    # Simulate for 4 months
    assert sim.date == Date(2010, 1, 1)
    sim.simulate(end_date=Date(2010, 5, 1))
    assert sim.date == Date(2010, 5, 1)

    # Check death dates match reference data
    pd.testing.assert_series_equal(
        pd.Series([False, False]),
        sim.population.is_alive,
        check_names=False)
    pd.testing.assert_series_equal(
        pd.Series([Date(2010, 3, 1), Date(2010, 4, 1)]),
        sim.population.date_of_death,
        check_names=False)


def test_single_step_death():
    # This demonstrates how to test the implementation of a single event in isolation

    # Set up minimal simulation
    sim = Simulation(start_date=Date(2010, 1, 1))
    rd = random_death.RandomDeath(name='rd')
    rd.parameters['death_probability'] = 0.1
    sim.register(rd)
    sim.seed_rngs(1)
    sim.make_initial_population(n=10)

    # Create and fire the event of interest
    event = random_death.RandomDeathEvent(rd, rd.death_probability)
    sim.fire_single_event(event, Date(2010, 2, 1))

    # Check it has behaved as expected
    assert sim.date == Date(2010, 2, 1)

    pd.testing.assert_series_equal(
        pd.Series([True, True, False, True, True, False, True, True, True, True]),
        sim.population.is_alive,
        check_names=False
    )


def test_make_test_property():
    # This tests the Population.make_test_property method
    sim = Simulation(start_date=Date(2010, 1, 1))
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
        sim.population.test,
        check_names=False
    )


def test_birth_and_death():
    # This combines both population-scope and individual-scope events,
    # with more complex logic.
    # Create a new simulation to orchestrate matters
    sim = Simulation(start_date=Date(1950, 1, 1))

    # Register modules
    rb = random_birth.RandomBirth(name='rb')
    rb.pregnancy_probability = 0.1
    rd = random_death.RandomDeath(name='rd')
    rd.death_probability = 0.01
    sim.register(rb, rd)
    assert sim.modules['rb'] is rb
    assert sim.modules['rd'] is rd

    # Seed the random number generators
    sim.seed_rngs(2)

    # Create a population of 10 individuals
    sim.make_initial_population(n=10)
    assert len(sim.population) == 10

    # Test iteration and individual-based property access
    for i, person in enumerate(sim.population):
        assert isinstance(person, Person)
        assert person.index == i
        assert person.is_alive
        assert not person.is_pregnant

    # Simulate for 2 years
    assert sim.date == Date(1950, 1, 1)
    sim.simulate(end_date=Date(1952, 1, 1))
    assert sim.date == Date(1952, 1, 1)

    # Test further population indexing
    pd.testing.assert_series_equal(
        pd.Series([True, False], index=[4, 5]),
        sim.population[4:5, 'is_alive'],
        check_names=False)
    pd.testing.assert_frame_equal(
        pd.DataFrame({'is_alive': [True, True], 'is_pregnant': [False, True]}, index=[8, 9]),
        sim.population[8:9, ('is_alive', 'is_pregnant')],
        check_names=False)
    pd.testing.assert_frame_equal(
        pd.DataFrame({'is_alive': [False], 'is_pregnant': [False]}, index=[6]),
        sim.population[[6], ('is_alive', 'is_pregnant')],
        check_names=False)

    # Check birth stats match reference data
    assert len(sim.population) == 20

    # Check no-one gave birth after dying
    for person in sim.population:
        if not person.is_alive:
            for child_index in person.children:
                child = sim.population[child_index]
                assert child.date_of_birth <= person.date_of_death

    # Check people can't become pregnant while already pregnant
    for person in sim.population:
        # Iterate over pairs of adjacent children
        for child1i, child2i in zip(person.children[:-1], person.children[1:]):
            child1, child2 = sim.population[child1i], sim.population[child2i]
            # Children earlier in the list are born earlier
            assert child1.date_of_birth < child2.date_of_birth
            # Birth dates need to be at least 9 months apart
            assert child1.date_of_birth + DateOffset(months=9) <= child2.date_of_birth
