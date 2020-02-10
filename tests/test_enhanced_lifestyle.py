
import os
import time
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import demography, enhanced_lifestyle

start_date = Date(2010, 1, 1)
end_date = Date(2012, 4, 1)
popsize = 10000


@pytest.fixture(scope='module')
def simulation():
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.seed_rngs(0)
    return sim


def __check_properties(df):
    # no one under 15 can be overweight, low exercise, tobacco, excessive alcohol, married
    under15 = df.age_years < 15
    assert not (under15 & df.li_bmi > 0).any()
    assert not (under15 & df.li_low_ex).any()
    assert not (under15 & df.li_tob).any()
    assert not (under15 & df.li_ex_alc).any()
    assert not (under15 & (df.li_mar_stat != 1)).any()
    # education: no one 0-5 should be in education
    assert not ((df.age_years < 5) & (df.li_in_ed | (df.li_ed_lev != 1))).any()

    # education: no one under 13 can be in secondary education
    assert not ((df.age_years < 13) & (df.li_ed_lev == 3)).any()

    # education: no one over age 20 in education
    assert not ((df.age_years > 20) & df.li_in_ed).any()


def test_make_initial_population(simulation):
    simulation.make_initial_population(n=popsize)


def test_initial_population(simulation):
    df = simulation.population.props
    __check_properties(df)


def test_simulate(simulation):
    simulation.simulate(end_date=end_date)


def test_final_population(simulation):
    df = simulation.population.props
    __check_properties(df)


def test_dypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)
    t1 = time.time()
    print('Time taken', t1 - t0)
