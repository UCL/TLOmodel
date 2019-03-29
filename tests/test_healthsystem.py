import pytest
import datetime
import logging
import os

import pandas as pd

from tlo import Date, Simulation
from tlo.methods import chronicsyndrome, demography, healthsystem, lifestyle, mockitis, qaly

resourcefilepath = os.path.join(os.path.dirname(__file__), 'resources')
# resourcefilepath='/Users/tbh03/PycharmProjects/TLOmodel/tests/resources/'

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 50

# Simply test whether the system runs under multiple configurations
# The Mockitits and ChronicSyndrome module test all aspects of the healthsystem module.

@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.INFO)


def test_RunWithHealthSystem_NoInterventionsDefined():

    sim = Simulation(start_date=start_date)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(lifestyle.Lifestyle())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    assert True # if got here with no errors, it's working



def test_RunWithHealthSystem_WithQALY():
    # Establish the simulation object
    sim = Simulation(start_date=start_date)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(qaly.QALY(resourcefilepath=resourcefilepath))
    sim.register(lifestyle.Lifestyle())


    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    assert True  # if got here with no errors, it's working



def test_RunWithHealthSystem_InterventionsOn():

    # Establish the simulation object
    sim = Simulation(start_date=start_date)

    # Define the service availability
    service_availability = pd.DataFrame(data=[], columns=['Service', 'Available'])
    service_availability.loc[0] = ['Mockitis_Treatment', True]
    service_availability.loc[1] = ['ChronicSyndrome_Treatment', True]
    service_availability['Service'] = service_availability['Service'].astype('object')
    service_availability['Available'] = service_availability['Available'].astype('bool')

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability))
    sim.register(qaly.QALY(resourcefilepath=resourcefilepath))
    sim.register(lifestyle.Lifestyle())
    sim.register(mockitis.Mockitis())
    sim.register(chronicsyndrome.ChronicSyndrome())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)


    assert True  # if got here with no errors, it's working







