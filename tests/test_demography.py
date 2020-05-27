import os
import time
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.methods import demography
from tlo.methods.demography import AgeUpdateEvent

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 500


@pytest.fixture(scope='module')
def simulation():
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)
    core_module = demography.Demography(resourcefilepath=resourcefilepath)
    sim.register(core_module)
    sim.seed_rngs(0)
    return sim


def test_run(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


def test_dypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_mothers_female(simulation):
    # check all mothers are female
    df = simulation.population.props
    mothers = df.loc[df.mother_id >= 0, 'mother_id']
    is_female = mothers.apply(lambda mother_id: df.at[mother_id, 'sex'] == 'F')
    assert is_female.all()


def test_py_calc(simulation):
    # make population of one person:
    simulation.make_initial_population(n=1)

    df = simulation.population.props
    df.sex = 'M'
    simulation.date += pd.DateOffset(days=1)
    age_update = AgeUpdateEvent(simulation.modules['Demography'], simulation.modules['Demography'].AGE_RANGE_LOOKUP)
    now = simulation.date

    # calc py: person is born and died before sim.date
    df.date_of_birth = now - pd.DateOffset(years=10)
    df.date_of_death = now - pd.DateOffset(years=9)
    df.is_alive = False
    age_update.apply(simulation.population)
    assert (0 == simulation.modules['Demography'].calc_py_lived_in_last_year()['M']).all()

    # calc py of person who is not yet born:
    df.date_of_birth = pd.NaT
    df.date_of_death = pd.NaT
    df.is_alive = False
    age_update.apply(simulation.population)
    assert (0 == simulation.modules['Demography'].calc_py_lived_in_last_year()['M']).all()

    # calc person who is alive and aged 20, with birthdays on today's date and lives throughout the period
    df.date_of_birth = now - pd.DateOffset(years=20)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    assert 1.0 == simulation.modules['Demography'].calc_py_lived_in_last_year()['M'][20]

    # calc person who is alive and aged 20, with birthdays on today's date, and dies 3 months ago
    df.date_of_birth = now - pd.DateOffset(years=20)
    df.date_of_death = now - pd.DateOffset(months=3)
    df.is_alive = False
    age_update.apply(simulation.population)
    assert (0.75 - simulation.modules['Demography'].calc_py_lived_in_last_year()['M'][20].sum()) < 0.01
    assert (0.75 - simulation.modules['Demography'].calc_py_lived_in_last_year()['M'][20]) < 0.01

    # calc person who is alive and aged 19, has birthday mid-way through the last year, and lives throughout
    df.date_of_birth = now - pd.DateOffset(years=20) - pd.DateOffset(months=6)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    assert 1.0 == simulation.modules['Demography'].calc_py_lived_in_last_year()['M'].sum()
    assert (0.5 - simulation.modules['Demography'].calc_py_lived_in_last_year()['M'][19]) < 0.01
    assert (0.5 - simulation.modules['Demography'].calc_py_lived_in_last_year()['M'][20]) < 0.01

    # calc person who is alive and aged 19, has birthday mid-way through the last year, and died 3 months ago
    df.date_of_birth = now - pd.DateOffset(years=20) - pd.DateOffset(months=6)
    df.date_of_death = now - pd.DateOffset(months=3)
    df.is_alive = False
    age_update.apply(simulation.population)
    assert (0.75 - simulation.modules['Demography'].calc_py_lived_in_last_year()['M'].sum()) < 0.01
    assert (0.5 - simulation.modules['Demography'].calc_py_lived_in_last_year()['M'][19]) < 0.01
    assert (0.25 - simulation.modules['Demography'].calc_py_lived_in_last_year()['M'][20]) < 0.01

    # 0/1 year-old with first birthday during the last year
    df.date_of_birth = now - pd.DateOffset(months=15)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    assert 1 == simulation.modules['Demography'].calc_py_lived_in_last_year()['M'].sum()
    assert (0.75 - simulation.modules['Demography'].calc_py_lived_in_last_year()['M'][0]) < 0.01
    assert (0.25 - simulation.modules['Demography'].calc_py_lived_in_last_year()['M'][1]) < 0.01

    # 0 year born in the last year
    df.date_of_birth = now - pd.DateOffset(months=9)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    assert (0.75 - simulation.modules['Demography'].calc_py_lived_in_last_year()['M'][0]) < 0.01
    assert (0 == simulation.modules['Demography'].calc_py_lived_in_last_year()['M'][1:]).any()

    # 99 years-old turning 100 in the last year
    df.date_of_birth = now - pd.DateOffset(years=100) - pd.DateOffset(months=6)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    assert (0.5 - simulation.modules['Demography'].calc_py_lived_in_last_year()['M'][99]) < 0.01
    assert (0.5 - simulation.modules['Demography'].calc_py_lived_in_last_year()['M'].sum()) < 0.01


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
