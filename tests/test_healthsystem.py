import pytest
import datetime
import logging
import os

import pandas as pd

from tlo import Date, Simulation
from tlo.methods import chronicsyndrome, demography, healthsystem, lifestyle, mockitis, healthburden

resourcefilepath = os.path.join(os.path.dirname(__file__), '../resources')
start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 10


# Simply test whether the system runs under multiple configurations of the healthsystem
# The Mockitits and ChronicSyndrome module test all aspects of the healthsystem module.

@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.INFO)

def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()

def test_run_with_healthsystem_no_interventions_defined():
    sim = Simulation(start_date=start_date)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(lifestyle.Lifestyle())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_run_with_healthsystem_and_healthburden():
    # Establish the simulation object
    sim = Simulation(start_date=start_date)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(lifestyle.Lifestyle())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


def test_run_with_healthsystem_interventions_off():
    # Establish the simulation object
    sim = Simulation(start_date=start_date)

    # Define the service availability
    service_availability = []

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability))
    sim.register(lifestyle.Lifestyle())
    sim.register(mockitis.Mockitis())
    sim.register(chronicsyndrome.ChronicSyndrome())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    assert sim.modules['HealthSystem'].hsi_event_queue_counter==0
    check_dtypes(sim)


def test_run_with_healthsystem_interventions_on():
    # Establish the simulation object
    sim = Simulation(start_date=start_date)

    # Define the service availability
    service_availability = list(['Mockitis*', 'ChronicSyndrome*'])

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability))
    sim.register(lifestyle.Lifestyle())
    sim.register(mockitis.Mockitis())
    sim.register(chronicsyndrome.ChronicSyndrome())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)

