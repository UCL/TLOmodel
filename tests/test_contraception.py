import os
import time
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)

start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 1)
popsize = 200


@pytest.fixture(scope='module')
def simulation():
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    service_availability = ['*']

    sim = Simulation(start_date=start_date, seed=0)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

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


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_make_initial_population(simulation)
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
