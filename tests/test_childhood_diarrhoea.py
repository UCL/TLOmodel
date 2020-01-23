
#
#  Ines -- whilst we work on this --- I recommend using the file 'diarrhoaea_run_me.py' in th sceipts/childhood_analysis.pu
#
#
#


import logging
import os
from pathlib import Path

import pytest
from tlo import Date, Simulation
from tlo.methods import demography, enhanced_lifestyle, diarrhoea, healthsystem, symptommanager, healthburden

workbook_name = 'demography.xlsx'

start_date = Date(2010, 1, 1)
end_date = Date(2019, 1, 1)
popsize = 10000


@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.INFO)


@pytest.fixture(scope='module')
def simulation():
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
# NB - contraception is not registered here so there would be no births
    # sim.register(hiv.hiv(resourcefilepath=resourcefilepath))
    logging.getLogger('tlo.methods.lifestyle').setLevel(logging.CRITICAL)
#   logging.getLogger('tlo.methods.lifestyle').setLevel(logging.WARNING)
#   sim.seed_rngs(1)
    return sim


def __check_properties(df):

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
