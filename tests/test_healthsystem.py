import logging
import os
import tempfile
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import chronicsyndrome, demography, healthburden, healthsystem, lifestyle, mockitis

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 10


# Simply test whether the system runs under multiple configurations of the healthsystem
# The Mockitits and ChronicSyndrome module test all aspects of the healthsystem module.

@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.DEBUG)


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_run_with_healthsystem_no_interventions_defined():
    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(lifestyle.Lifestyle())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_run_with_healthsystem_and_healthburden():
    # Establish the simulation object
    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(lifestyle.Lifestyle())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_run_with_healthsystem_interventions_off():
    # Establish the simulation object
    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)

    # Define the service availability
    service_availability = []

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability))
    sim.register(lifestyle.Lifestyle())
    sim.register(mockitis.Mockitis())
    sim.register(chronicsyndrome.ChronicSyndrome())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    assert sim.modules['HealthSystem'].hsi_event_queue_counter == 0
    check_dtypes(sim)


def test_run_with_healthsystem_interventions_on():
    # Establish the simulation object
    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)

    # Define the service availability
    service_availability = list(['Mockitis*', 'ChronicSyndrome*'])

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability))
    sim.register(lifestyle.Lifestyle())
    sim.register(mockitis.Mockitis())
    sim.register(chronicsyndrome.ChronicSyndrome())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_run_with_healthsystem_interventions_on_but_no_capabilities():
    f = tempfile.NamedTemporaryFile(dir='.')
    fh = logging.FileHandler(f.name)
    fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
    fh.setFormatter(fr)
    logging.getLogger().addHandler(fh)

    # Establish the simulation object
    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)
    # Define the service availability
    service_availability = list(['Mockitis*', 'ChronicSyndrome*'])

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           # this effectively removes capabilities of HS:
                                           capabilities_coefficient=0.0))
    sim.register(lifestyle.Lifestyle())
    sim.register(mockitis.Mockitis())
    sim.register(chronicsyndrome.ChronicSyndrome())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    fh.flush()
    output = parse_log_file(f.name)
    f.close()

    # check that there have been no HSI events (due to there being no capabilities)
    assert 'HSI_Event' not in output['tlo.methods.healthsystem'], 'one'
    assert 'Consumables' not in output['tlo.methods.healthsystem'], 'two'
    assert (output['tlo.methods.healthsystem']['Capacity']['Frac_Time_Used_Overall'] == 0).all(), 'three'




def test_run_with_healthsystem_interventions_on_but_no_capabilities_and_ignore_appt_constraints():
    f = tempfile.NamedTemporaryFile(dir='.')
    fh = logging.FileHandler(f.name)
    fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
    fh.setFormatter(fr)
    logging.getLogger().addHandler(fh)

    # Establish the simulation object
    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)

    # Define the service availability
    service_availability = list(['Mockitis*', 'ChronicSyndrome*'])

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           # this effectively remove capabilities of HS:
                                           capabilities_coefficient=0.0,
                                           # ... but this says ignore any constraints:
                                           ignore_appt_constraints=True))
    sim.register(lifestyle.Lifestyle())
    sim.register(mockitis.Mockitis())
    sim.register(chronicsyndrome.ChronicSyndrome())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    fh.flush()
    output = parse_log_file(f.name)
    f.close()


    # check that there have been some HSI events having occured
    assert len(output['tlo.methods.healthsystem']['HSI_Event'])>0
