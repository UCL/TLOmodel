import logging
import os
import time
from pathlib import Path
import datetime
from tlo.analysis.utils import parse_log_file

import pytest

from tlo import Date, Simulation
from tlo.methods import demography, healthsystem, enhanced_lifestyle, healthburden, contraception, malaria

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
    sim = Simulation(start_date=start_date)

    resourcefilepath = Path("./resources")

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           mode_appt_constraints=0,
                                           capabilities_coefficient=1.0))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    # sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(malaria.Malaria(resourcefilepath=resourcefilepath))
    return sim


def test_run(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


def test_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def __check_properties(df):
    # no-one on treatment if not clinical or severe

    # specific symptoms categorial: none, clinical or severe

    # clinical counter >=0, integer

    assert not ((df.sex == 'M') & (df.hv_sexual_risk != 'sex_work')).any()
    assert not ((df.hv_number_tests >= 1) & ~df.hv_ever_tested).any()

    assert not (df.mc_is_circumcised & (df.sex == 'F')).any()


def test_run_no_capability(tmpdir):
    # no HSI events should run

    # Establish the logger
    logfile = outputpath + 'LogFile' + datestamp + 'no_capability' + '.log'

    if os.path.exists(logfile):
        os.remove(logfile)
    fh = logging.FileHandler(logfile)
    fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
    fh.setFormatter(fr)
    logging.getLogger().addHandler(fh)

    service_availability = ["*"]
    sim = Simulation(start_date=start_date)

    resourcefilepath = Path("./resources")

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           mode_appt_constraints=0,
                                           capabilities_coefficient=1.0))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(malaria.Malaria(resourcefilepath=resourcefilepath))

    sim.seed_rngs(0)

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    test_dtypes(sim)

    # read the results
    fh.flush()
    output = parse_log_file(logfile)
    fh.close()

    # Do the checks
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    assert output['tlo.methods.healthsystem']['HSI_Event']['did_run'].all()
    assert (output['tlo.methods.healthsystem']['HSI_Event']['Squeeze_Factor'] == 0.0).all()


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
