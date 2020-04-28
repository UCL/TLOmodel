from pathlib import Path
import os
import pytest
from datetime import datetime, timedelta

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    contraception,
    healthburden,
    healthsystem,
    enhanced_lifestyle,
    epi
)

start_date = Date(2010, 1, 1)
end_date = Date(2025, 1, 1)
popsize = 500

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"
except NameError:
    # running interactively
    resourcefilepath = "resources"


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


# checking no vaccines administered through health system
# only hpv should stay at zero, other vaccines start as individual events (year=2010-2018)
# coverage should gradually decline for all after 2018
# hard constraints (mode=2) and zero capabilities
def test_no_health_system(tmpdir):

    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            service_availability=[],  # no services allowed
            mode_appt_constraints=2,  # hard constraints
            ignore_priority=True,
            capabilities_coefficient=0.0  # no officer time
        )
    )
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(epi.Epi(resourcefilepath=resourcefilepath))

    sim.seed_rngs(0)

    # Run the simulation and flush the logger
    custom_levels = {"*": logging.WARNING, "tlo.methods.epi": logging.INFO}

    # configure_logging automatically appends datetime
    f = sim.configure_logging("test_log", directory=tmpdir, custom_levels=custom_levels)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(f)
    df = sim.population.props

    # check no vaccines being administered through health system
    # only hpv currently, all others start as individual events
    assert (df.ep_hpv == 0).all()

    # check all infants born after Jan 2019 have no bcg / penta etc. through HSIs
    assert not ((df.ep_bcg > 0) & (df.date_of_birth > datetime(2019, 1, 1))).any()
    assert not ((df.ep_dtp > 0) & (df.date_of_birth > datetime(2019, 1, 1))).any()


# check epi module does schedule hsi events
def test_epi_scheduling_hsi_events(tmpdir):

    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            service_availability=["*"],  # all services allowed
            ignore_cons_constraints=True,
            ignore_priority=True,
            capabilities_coefficient=1.0,  # full capacity
            mode_appt_constraints=0,  # no constraints
            disable=False
        )
    )
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(epi.Epi(resourcefilepath=resourcefilepath))

    sim.seed_rngs(0)

    # Run the simulation and flush the logger
    f = sim.configure_logging("test_log", directory=tmpdir)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(f)
    df = sim.population.props

    # check vaccine coverage is above zero for all vaccine types post 2019
    # 2010-2018 vaccines administered through individual events
    ep_out = output["tlo.methods.epi"]["ep_vaccine_coverage"]

    #TODO have to select 2019 onwards
    assert not (ep_out.epBcgCoverage > 0).all()

    # check only 3 doses max of dtp/pneumo
    assert (df.ep_dtp <= 3).all()
    assert (df.ep_pneumo <= 3).all()
