import logging
import os
import time

import pytest
from pathlib import Path

from tlo import Date, Simulation

from tlo.methods import demography, enhanced_lifestyle, labour, newborn_outcomes, healthburden, healthsystem, antenatal_care,\
    pregnancy_supervisor, contraception

start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 1)
popsize = 50

outputpath = Path("./outputs")  # folder for convenience of storing outputs


@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.INFO)


@pytest.fixture(scope='module')
def simulation():

    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(labour.Labour(resourcefilepath=resourcefilepath))
    sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
    sim.register(antenatal_care.AntenatalCare(resourcefilepath=resourcefilepath))
    sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))

    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           mode_appt_constraints=0))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))

    sim.seed_rngs(1)
    return sim


def test_run(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


def __check_properties(df):
    # Cannot have a partiy of higher than allowed per age group
    assert not ((df.age_years < 24) & (df.la_parity >4)).any()
    assert not ((df.age_years < 40) & (df.la_parity >5)).any()

    # Confirming PTB and previous CS logic
    assert not ((df.la_parity <= 1) & (df.la_previous_cs > 1)).any()
    assert not (df.la_previous_cs > 2).any()
    assert not ((df.la_previous_cs == 1) & (df.la_previous_ptb == 1)).any()
    assert not ((df.la_previous_cs == 2) & (df.la_previous_ptb >= 1) & (df.la_parity ==2)).any()


#def test_make_initial_population(simulation):
#    simulation.make_initial_population(n=popsize)


#def test_initial_population(simulation):
#    __check_properties(simulation.population.props)


#def test_simulate(simulation):
#    simulation.simulate(end_date=end_date)


#def test_final_population(simulation):
#    __check_properties(simulation.population.props)



def test_dypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    counter = 0
    for orig_type, df_type in zip(orig.dtypes, df.dtypes):
        counter += 1
        assert orig_type == df_type, f"column number {counter}\n - orig: {orig_type},  df: {df_type}"
    assert (df.dtypes == orig.dtypes).all()


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
    test_dypes(simulation)
