import logging
import os
import time
from pathlib import Path
import datetime
from tlo.analysis.utils import parse_log_file

import pytest

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    contraception,
    healthburden,
    healthsystem,
    enhanced_lifestyle,
    malaria,
    dx_algorithm_child,
    dx_algorithm_adult,
    healthseekingbehaviour,
    symptommanager,
)

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 50

outputpath = './src/scripts/malaria/'
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.INFO)


@pytest.fixture(scope='module')
def simulation():
    # Establish the logger
    logfile = outputpath + 'LogFile' + datestamp + '.log'

    if os.path.exists(logfile):
        os.remove(logfile)
    fh = logging.FileHandler(logfile)
    fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
    fh.setFormatter(fr)
    logging.getLogger().addHandler(fh)

    service_availability = ["*"]
    malaria_strat = 0  # levels: 0 = national; 1 = district

    sim = Simulation(start_date=start_date)

    resourcefilepath = Path("./resources")

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           mode_appt_constraints=0,
                                           ignore_cons_constraints=True,
                                           ignore_priority=True,
                                           capabilities_coefficient=1.0))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthseekingbehaviour.HealthSeekingBehaviour())
    sim.register(dx_algorithm_child.DxAlgorithmChild())
    sim.register(dx_algorithm_adult.DxAlgorithmAdult())
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(malaria.Malaria(resourcefilepath=resourcefilepath,
                                 level=malaria_strat))
    return sim


def test_run(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


def test_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_run_no_hsi():
    # There should be no HSI events run or scheduled

    # Establish the simulation object
    sim = Simulation(start_date=start_date)

    # Establish the logger
    logfile = outputpath + 'LogFile' + datestamp + '.log'

    if os.path.exists(logfile):
        os.remove(logfile)
    fh = logging.FileHandler(logfile)
    fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
    fh.setFormatter(fr)
    logging.getLogger().addHandler(fh)

    service_availability = []
    malaria_strat = 0  # levels: 0 = national; 1 = district
    resourcefilepath = Path("./resources")

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           mode_appt_constraints=0,
                                           ignore_cons_constraints=True,
                                           ignore_priority=True,
                                           capabilities_coefficient=1.0))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthseekingbehaviour.HealthSeekingBehaviour())
    sim.register(dx_algorithm_child.DxAlgorithmChild())
    sim.register(dx_algorithm_adult.DxAlgorithmAdult())
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(malaria.Malaria(resourcefilepath=resourcefilepath,
                                 level=malaria_strat))

    sim.seed_rngs(0)

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    test_dtypes(sim)

    # read the results
    fh.flush()
    output = parse_log_file(logfile)
    fh.close()

    # check no-one has been treated
    assert output['tlo.methods.malaria']['tx_coverage'] == 0


def __check_deaths(simulation):
    # deaths only occur in severe malaria cases
    # check types of columns
    df = simulation.population.props

    assert not (
            (df.ma_date_death) & ((df.ma_specific_symptoms == 'clinical') | (df.ma_specific_symptoms == 'none'))).any()


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
