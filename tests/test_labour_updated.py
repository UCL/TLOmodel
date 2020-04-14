import os
import pandas as pd
import datetime
from pathlib import Path

import pytest

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 200


# Establish the simulation object
def run_simulation_with_set_service_coverage_parameter(service_availability, healthsystemdisable):
    """
    This helper function will run a simulation with a given service coverage parameter and return the path of
    the logfile.
    :param service_availability: list indicating which serivces to include (see HealthSystem)
    :param healthsystemdisable: bool to indicate whether or not to disable healthsystem (see HealthSystem)
    :return: logfile name
    """

    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)

    # Register the appropriate modules

    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(labour.Labour(resourcefilepath=resourcefilepath))
    sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
    sim.register(antenatal_care.AntenatalCare(resourcefilepath=resourcefilepath))
    sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))

    sim.register(healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        service_availability=service_availability,
        disable=healthsystemdisable
    ))

    # Establish the logger
    logfile = sim.configure_logging(filename="LogFile")

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return logfile


results_health_system_disabled = (
    parse_log_file(run_simulation_with_set_service_coverage_parameter(
            service_availability=['*'],
            healthsystemdisable=True)))
