import logging
import os
import time

import pytest
from pathlib import Path

from tlo import Date, Simulation
from tlo.methods import demography,enhanced_lifestyle, labour, newborn_outcomes, healthburden, healthsystem, antenatal_care,\
    pregnancy_supervisor, contraception

workbook_name = 'demography.xlsx'

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 1000

outputpath = Path("./outputs")  # folder for convenience of storing outputs


@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.INFO)


@pytest.fixture(scope='module')
def simulation():
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(labour.Labour(resourcefilepath=resourcefilepath))
    sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
    sim.register(antenatal_care.AntenatalCare(resourcefilepath=resourcefilepath))
    sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))

    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           mode_appt_constraints=0))

    sim.seed_rngs(1)
    return sim


def test_run(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


def __check_properties(df):
    pass
    # TODO: TBC


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
    assert (df.dtypes == orig.dtypes).all()


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
