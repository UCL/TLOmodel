
import pandas as pd

from tlo import Date, Person, Simulation
from tlo.test import random_death


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
        pd.Series([True, True], name='is_alive'),
        sim.population.props['is_alive'])
    assert sim.population.is_alive is sim.population.props['is_alive']

    # Simulate for 10 years
    assert sim.date == Date(2010, 1, 1)
    sim.simulate(end_date=Date(2010, 5, 1))
    assert sim.date == Date(2010, 5, 1)

    # Check death dates match reference data
    pd.testing.assert_series_equal(
        pd.Series([False, False], name='is_alive'),
        sim.population.props['is_alive'])
    pd.testing.assert_series_equal(
        pd.Series([Date(2010, 3, 1), Date(2010, 4, 1)], name='date_of_death'),
        sim.population.props['date_of_death'])
