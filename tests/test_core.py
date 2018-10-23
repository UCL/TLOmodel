import pandas as pd
import pytest

from tlo import Date, Property, Simulation, Types


def test_categorical():
    categories = ['M', 'F']
    p = Property(Types.CATEGORICAL, description='Biological Sex', categories=categories)
    s = p.create_series('Sex', 10)
    assert s.dtype.name == 'category'
    assert list(s.cat.categories) == categories

    # assignment of valid category
    try:
        s[0] = 'M'
    except ValueError:
        pytest.fail('Unexpected ValueError')

    # assignment of invalid category
    with pytest.raises(ValueError):
        s[0] = 'A'


def test_age():
    """Tests the automatic calculation of age columns by Population class (reqs. date_of_birth)"""
    sim = Simulation(start_date=Date(2000, 1, 1))
    sim.make_initial_population(n=10)
    sim.population.make_test_property('date_of_birth', Types.DATE)

    # convert these ages into dates of birth
    ages_in = pd.Series([0, 2, 4, 8, 16, 32, 50, 64, 100, 120])
    date_of_birth = sim.date - pd.to_timedelta(ages_in, unit='Y')
    sim.population.props['date_of_birth'] = date_of_birth

    # use the 'age' property to get ages back out
    ages_out = sim.population.age.years
    assert list(ages_out) == list(ages_in)

    assert sim.population.age.age_range.dtypes == 'category'

    age_ranges = ['0-4', '0-4', '0-4', '5-9', '15-19', '30-34', '50-54', '60-64', '100+', '100+']
    assert age_ranges == list(sim.population.age.age_range)
