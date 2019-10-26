import logging
import os
import tempfile
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import chronicsyndrome, demography, healthsystem, lifestyle, mockitis

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 10


# Simply test whether the system runs under multiple configurations of the healthsystem
# NB. Running the dummy Mockitits and ChronicSyndrome modules test all aspects of the healthsystem module.

@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.DEBUG)


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_run_with_healthsystem_no_disease_modules_defined():
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


def test_run_no_interventions_allowed():
    # There should be no events run or scheduled

    # Get ready for temporary log-file
    f = tempfile.NamedTemporaryFile(dir='.')
    fh = logging.FileHandler(f.name)
    fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
    fh.setFormatter(fr)
    logging.getLogger().addHandler(fh)

    # Establish the simulation object
    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)

    # Define the service availability as null
    service_availability = []

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=0))
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

    # Do the checks
    assert (output['tlo.methods.healthsystem']['Capacity']['Frac_Time_Used_Overall'] == 0.0).all()
    assert len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE) == 0
    check_dtypes(sim)


def test_run_in_mode_0_with_capacity():
    # Events should run and there be no squeeze factors
    # (Mode 0 -> No Constraints)

    # Get ready for temporary log-file
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
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=0))
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

    # Do the checks
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    assert output['tlo.methods.healthsystem']['HSI_Event']['did_run'].all()
    assert (output['tlo.methods.healthsystem']['HSI_Event']['Squeeze_Factor'] == 0.0).all()
    check_dtypes(sim)


def test_run_in_mode_0_no_capacity():
    # Every events should run (no did_not_run)
    # (Mode 0 -> No Constraints)

    # Get ready for temporary log-file
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
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=0))
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

    # Do the checks
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    assert output['tlo.methods.healthsystem']['HSI_Event']['did_run'].all()
    assert (output['tlo.methods.healthsystem']['HSI_Event']['Squeeze_Factor'] == 0.0).all()
    check_dtypes(sim)


def test_run_in_mode_1_with_capacity():
    # All events should run with some zero squeeze factors
    # (Mode 1 -> elastic constraints)

    # Get ready for temporary log-file
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
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=1))
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

    # Do the checks
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    assert output['tlo.methods.healthsystem']['HSI_Event']['did_run'].all()
    assert (output['tlo.methods.healthsystem']['HSI_Event']['Squeeze_Factor'] == 0.0).all()
    check_dtypes(sim)


# TODO; This one!
def test_run_in_mode_1_with_no_capacity():
    # Events should run but with high squeeze factors
    # (Mode 1 -> elastic constraints)

    # Get ready for temporary log-file
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
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=1))
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

    # Do the checks
    check_dtypes(sim)
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    hsi_events = output['tlo.methods.healthsystem']['HSI_Event']
    assert hsi_events['did_run'].all()
    assert (hsi_events.loc[hsi_events['Person_ID'] >= 0, 'Squeeze_Factor'] == 100.0).all()
    assert (hsi_events.loc[hsi_events['Person_ID'] < 0, 'Squeeze_Factor'] == 0.0).all()
    check_dtypes(sim)


def test_run_in_mode_2_with_capacity():
    # All events should run
    # (Mode 2 -> hard constraints)

    # Get ready for temporary log-file
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
                                           capabilities_coefficient=1.0,
                                           mode_appt_constraints=2))
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

    # Do the checks
    assert len(output['tlo.methods.healthsystem']['HSI_Event']) > 0
    assert output['tlo.methods.healthsystem']['HSI_Event']['did_run'].all()
    assert (output['tlo.methods.healthsystem']['HSI_Event']['Squeeze_Factor'] == 0.0).all()
    check_dtypes(sim)


def test_run_in_mode_2_with_no_capacity():
    # No individual level events should run and the log should contain events with a flag showing that all individual
    # events did not run. Population level events should have run.
    # (Mode 2 -> hard constraints)

    # Get ready for temporary log-file
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
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=2))
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

    # Do the checks
    hsi_events = output['tlo.methods.healthsystem']['HSI_Event']
    assert (hsi_events.loc[hsi_events['Person_ID'] >= 0, 'did_run'] == False).all()  # Individual level
    assert (output['tlo.methods.healthsystem']['Capacity']['Frac_Time_Used_Overall'] == 0.0).all()
    assert (hsi_events.loc[hsi_events['Person_ID'] < 0, 'did_run'] == True).all()  # Population level
    check_dtypes(sim)
