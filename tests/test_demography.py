import os
import time
from pathlib import Path

import numpy as np
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
    sim = Simulation(start_date=start_date, seed=0)
    core_module = demography.Demography(resourcefilepath=resourcefilepath)
    sim.register(core_module)
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
    one_year = np.timedelta64(1, 'Y')
    one_month = np.timedelta64(1, 'M')

    calc_py_lived_in_last_year = simulation.modules['Demography'].calc_py_lived_in_last_year

    # calc py: person is born and died before sim.date
    df.date_of_birth = now - (one_year * 10)
    df.date_of_death = now - (one_year * 9)
    df.is_alive = False
    age_update.apply(simulation.population)
    assert (0 == calc_py_lived_in_last_year(delta=one_year)['M']).all()

    # calc py of person who is not yet born:
    df.date_of_birth = pd.NaT
    df.date_of_death = pd.NaT
    df.is_alive = False
    age_update.apply(simulation.population)
    assert (0 == calc_py_lived_in_last_year(delta=one_year)['M']).all()

    # calc person who is alive and aged 20, with birthdays on today's date and lives throughout the period
    df.date_of_birth = now - (one_year * 20)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    np.testing.assert_almost_equal(calc_py_lived_in_last_year(delta=one_year)['M'][19], 1.0)

    # calc person who is alive and aged 20, with birthdays on today's date, and dies 3 months ago
    df.date_of_birth = now - (one_year * 20)
    df.date_of_death = now - pd.Timedelta(one_year) * 0.25
    # we have to set the age at time of death - usually this would have been set by the AgeUpdateEvent
    df.age_exact_years = (df.date_of_death - df.date_of_birth) / one_year
    df.age_years = df.age_exact_years.astype('int64')
    df.is_alive = False
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    np.testing.assert_almost_equal(0.75, df_py['M'][19])
    assert df_py['M'][20] == 0.0

    # calc person who is alive and aged 19, has birthday mid-way through the last year, and lives throughout
    df.date_of_birth = now - (one_year * 20) - (one_month * 6)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    np.testing.assert_almost_equal(0.5, df_py['M'][19])
    np.testing.assert_almost_equal(0.5, df_py['M'][20])

    # calc person who is alive and aged 19, has birthday mid-way through the last year, and died 3 months ago
    df.date_of_birth = now - (one_year * 20) - (one_month * 6)
    df.date_of_death = now - np.timedelta64(3, 'M')
    # we have to set the age at time of death - usually this would have been set by the AgeUpdateEvent
    df.age_exact_years = (df.date_of_death - df.date_of_birth) / one_year
    df.age_years = df.age_exact_years.astype('int64')
    df.is_alive = False
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    assert 0.75 == df_py['M'].sum()
    assert 0.5 == df_py['M'][19]
    assert 0.25 == df_py['M'][20]

    # 0/1 year-old with first birthday during the last year
    df.date_of_birth = now - (one_month * 15)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    assert 0.75 == df_py['M'][0]
    assert 0.25 == df_py['M'][1]

    # 0 year born in the last year
    df.date_of_birth = now - (one_month * 9)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    assert 0.75 == df_py['M'][0]
    assert (0 == df_py['M'][1:]).all()

    # 99 years-old turning 100 in the last year
    df.date_of_birth = now - (one_year * 100) - (one_month * 6)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    assert 0.5 == df_py['M'][99]
    assert 1 == df_py['M'].sum()


def test_py_calc_w_mask(simulation):
    # make population of two people
    simulation.make_initial_population(n=2)

    df = simulation.population.props
    df.sex = 'M'
    simulation.date += pd.DateOffset(days=1)
    age_update = AgeUpdateEvent(simulation.modules['Demography'], simulation.modules['Demography'].AGE_RANGE_LOOKUP)
    now = simulation.date

    calc_py_lived_in_last_year = simulation.modules['Demography'].calc_py_lived_in_last_year

    # calc two people who are alive and aged 20, with birthdays on today's date and live throughout the period,
    # neither has hypertension

    df.date_of_birth = now - pd.Timedelta(20, 'Y')
    df.date_of_death = pd.NaT
    df.is_alive = True
    df['nc_hypertension'] = False
    mask = (df.is_alive & ~df['nc_hypertension'])
    df = df[mask]
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=pd.Timedelta(1, 'Y'), mask=mask)
    np.testing.assert_almost_equal(2.0, df_py['M'][19])

    # calc two people who are alive and aged 20, with birthdays on today's date and live throughout the period,
    # one has hypertension

    df.date_of_birth = now - pd.Timedelta(20, 'Y')
    df.date_of_death = pd.NaT
    df.is_alive = True
    df['nc_hypertension'].iloc[0] = True
    mask = (df.is_alive & ~df['nc_hypertension'])
    df = df[mask]
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=pd.Timedelta(1, 'Y'), mask=mask)
    np.testing.assert_almost_equal(1.0, df_py['M'][19])


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
