import logging
import os

import pytest
from pathlib import Path

from tlo import Date, Simulation
from tlo.methods import demography, lifestyle, labour,newborn_outcomes, healthburden, healthsystem, antenatal_care,\
    abortion_and_miscarriage, pregnancy_supervisor

workbook_name = 'demography.xlsx'

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 1000


@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.INFO)


@pytest.fixture(scope='module')
def simulation():
    demography_workbook = os.path.join(os.path.dirname(__file__),
                                       'resources',
                                       workbook_name)

    sim = Simulation(start_date=start_date)
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)

    core_module = demography.Demography(resourcefilepath=resourcefilepath)
    service_availability = ['*']
    lab_module = labour.Labour(resourcefilepath=resourcefilepath)
    nb_module = newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath)
    anc_module = antenatal_care.AntenatalCare(resourcefilepath=resourcefilepath)
    am_module = abortion_and_miscarriage.AbortionAndMiscarriage(resourcefilepath=resourcefilepath)
    ps_module = pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath)

    sim.register(core_module)
    sim.register(lab_module)
    sim.register(nb_module)
    sim.register(anc_module)
    sim.register(am_module)
    sim.register(ps_module)

    sim.register(lifestyle.Lifestyle())
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability))
#    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))


    logging.getLogger('tlo.methods.lifestyle').setLevel(logging.CRITICAL)
    logging.getLogger('tlo.methods.lifestyle').setLevel(logging.WARNING)
    sim.seed_rngs(1)
    return sim


def __check_properties(df):
    # Cannot have a partiy of higher than allowed per age group
    assert not (df.age_years < 24 & df.la_parity >4)
    assert not (df.age_years < 40 & df.la_parity >5)

    # Confirming PTB and previous CS logic
    assert not ((df.la_parity <= 1) & (df.la_previous_cs > 1)).any()
    assert not (df.la_previous_cs > 2).any()
    assert not (df.la_previous_cs == 1 & df.la_previous_ptb == 1).any()
    assert not (df.la_previous_cs == 2 & df.la_previous_ptb >= 1 & df.la_parity ==2).any()


def test_make_initial_population(simulation):
    simulation.make_initial_population(n=popsize)


def test_initial_population(simulation):
    __check_properties(simulation.population.props)



def test_simulate(simulation):
    simulation.simulate(end_date=end_date)


def test_final_population(simulation):
    __check_properties(simulation.population.props)


def test_dypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


if __name__ == '__main__':
    simulation = simulation()
    logging.getLogger('tlo.methods.demography').setLevel(logging.WARNING)
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)
