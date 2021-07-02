import os
import time
from pathlib import Path

import pytest

from tlo import Date, Simulation, Types, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthsystem,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager
)
from tlo.methods.hiv import DummyHivModule

start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 1)
popsize = 200

# todo reorgnsie this to remove duplication and to use the logs to check that things are happening (pregnancies, contracpeitvr start, stop etc)

@pytest.fixture(scope='module')
def simulation():
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

    sim = Simulation(start_date=start_date)
    sim.register(
        # - core modules:
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=False),

        # - modules for mechanistic representation of contraception -> pregnancy -> labour -> delivery etc.
        contraception.Contraception(resourcefilepath=resourcefilepath),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
        care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
        labour.Labour(resourcefilepath=resourcefilepath),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
        postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),

        # - Dummy HIV module (as contraception requires the property hv_inf)
        DummyHivModule()
    )
    return sim


def __check_properties(df):
    assert not ((~df.date_of_birth.isna()) & (df.sex == 'M') & (df.co_contraception != 'not_using')).any()
    assert not ((~df.date_of_birth.isna()) & (df.age_years < 15) & (df.co_contraception != 'not_using')).any()
    assert not ((~df.date_of_birth.isna()) & (df.sex == 'M') & df.is_pregnant).any()


def test_make_initial_population(simulation):
    simulation.make_initial_population(n=popsize)


def test_initial_population(simulation):
    __check_properties(simulation.population.props)


def test_run(simulation):
    simulation.simulate(end_date=end_date)

def test_final_population(simulation):
    __check_properties(simulation.population.props)


def test_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_log(tmpdir):
    """test that what comes out in log is as expected"""
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

    log_config = {
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            "*": logging.WARNING,
            'tlo.methods.contraception': logging.INFO
        }
    }

    sim = Simulation(start_date=start_date, log_config=log_config)
    sim.register(
        # - core modules:
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=False),

        # - modules for mechanistic representation of contraception -> pregnancy -> labour -> delivery etc.
        contraception.Contraception(resourcefilepath=resourcefilepath),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
        care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
        labour.Labour(resourcefilepath=resourcefilepath),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
        postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),

        # - Dummy HIV module (as contraception requires the property hv_inf)
        DummyHivModule()
    )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    logs = parse_log_file(sim.log_filepath)['tlo.methods.contraception']
    assert set(logs.keys()) == set([
        'contraception_use_yearly_summary',
        'contraception_costs_yearly_summary',
        'pregnancy',
        'contraception',
    ])

    # todo - check that there are some births, and some switches in contraception
    # todo - organize this file: the first (long) test is superfluous.


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_make_initial_population(simulation)
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
