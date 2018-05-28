
import pandas as pd

from tlo import Date, DateOffset, Person, Simulation
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
    assert sim.population[0].props['is_alive'][0]

    # Test population-based property access
    assert len(sim.population.props['is_alive']) == 2
    assert isinstance(sim.population.props['is_alive'], pd.Series)
    pd.testing.assert_series_equal(
        pd.Series([True, True]),
        sim.population.props['is_alive'],
        check_names=False)
    assert sim.population.is_alive is sim.population.props['is_alive']

    # Simulate for 4 months
    assert sim.date == Date(2010, 1, 1)
    sim.simulate(end_date=Date(2010, 5, 1))
    assert sim.date == Date(2010, 5, 1)

    # Check death dates match reference data
    pd.testing.assert_series_equal(
        pd.Series([False, False]),
        sim.population.props['is_alive'],
        check_names=False)
    pd.testing.assert_series_equal(
        pd.Series([Date(2010, 3, 1), Date(2010, 4, 1)]),
        sim.population.props['date_of_death'],
        check_names=False)


def test_birth_and_death():
    # This combines both population-scope and individual-scope events,
    # with more complex logic.
    # Create a new simulation to orchestrate matters
    sim = Simulation(start_date=Date(1950, 1, 1))

    # Register modules
    rb = random_birth.RandomBirth(name='rb')
    rb.pregnancy_probability = 0.5
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

    # Simulate for 5 years
    assert sim.date == Date(1950, 1, 1)
    sim.simulate(end_date=Date(1955, 1, 1))
    assert sim.date == Date(1955, 1, 1)

    # Check birth stats match reference data
    assert len(sim.population) == 35

    # Check no-one gave birth after dying
    for person in sim.population:
        if not person.is_alive:
            for child in person.children:
                assert child.date_of_birth < person.date_of_death

    # Check people can't become pregnant while already pregnant
    for person in sim.population:
        # Iterate over pairs of adjacent children
        for child1, child2 in zip(person.children[:-1], person.children[1:]):
            # Children earlier in the list are born earlier
            assert child1.date_of_birth < child2.date_of_birth
            # Birth dates need to be at least 9 months apart
            assert child1.date_of_birth + DateOffset(months=9) <= child2.date_of_birth
