import logging
import os

import pandas as pd
import pytest

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


def _check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_healthsystem_no_interventions():

    sim = Simulation(start_date=start_date)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(lifestyle.Lifestyle())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    _check_dtypes(sim)


def test_healthsystem_with_qaly():
    # Establish the simulation object
    sim = Simulation(start_date=start_date)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(qaly.QALY(resourcefilepath=resourcefilepath))
    sim.register(lifestyle.Lifestyle())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    _check_dtypes(sim)


def test_health_system_interventions_on():

    # Establish the simulation object
    sim = Simulation(start_date=start_date)

    # Define the service availability
    service_availability = pd.DataFrame.from_records(
        [
            ('Mockitis_Treatment', True),
            ('ChronicSyndrome_Treatment', True)
        ],
        columns=['Service', 'Available'],
    ).astype({'Service': object, 'Available': bool})

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability))
    sim.register(qaly.QALY(resourcefilepath=resourcefilepath))
    sim.register(lifestyle.Lifestyle())
    sim.register(mockitis.Mockitis())
    sim.register(chronicsyndrome.ChronicSyndrome())

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    _check_dtypes(sim)
