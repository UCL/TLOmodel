
from datetime import date

import pandas as pd

from tlo import Person, PopulationBuilder, Simulation
from tlo.test import random_death


def test_individual_death():
    # Create a new simulation to orchestrate matters
    sim = Simulation(start_date=date(2010, 1, 1))

    # Register just a test module with random death
    # Note: this approach would allow us to give a different name if desired,
    # so the same module could appear twice, e.g. with different parameters.
    # Not sure if that would be helpful, but...!
    rd = random_death.RandomDeath()
    sim.register(rd)
    assert sim.modules['rd'] is rd
    sim.modules['rd'].parameters['death_probability'] = 0.1
    # We can also use attribute-style access if the name doesn't clash
    assert rd.death_probability == 0.1
    rd.death_probability = 0.2

    # Seed the random number generators
    sim.seed_rngs(0)

    # Create a population of 2 individuals
    builder = PopulationBuilder(sim)
    builder.make_population(n=2)
    assert len(sim.population) == 2

    # Test individual-based property access
    assert isinstance(sim.population[0], Person)
    assert sim.population[0].props['is_alive']

    # Test population-based property access
    assert len(sim.population.props['is_alive']) == 2
    assert isinstance(sim.population.props['is_alive'], pd.Series)
    pd.testing.assert_series_equal(
        pd.Series([True, True]),
        sim.population.props['is_alive'])

    # Simulate for 10 years
    assert sim.date == date(2010, 1, 1)
    sim.simulate(end_date=date(2020, 1, 1))
    assert sim.date == date(2020, 1, 1)

    # Check death dates match reference data
    pd.testing.assert_series_equal(
        pd.Series([False, False]),
        sim.population.props['is_alive'])
    pd.testing.assert_series_equal(
        pd.Series([date(2012, 1, 1), date(2018, 1, 1)]),
        sim.population.props['date_of_death'])
