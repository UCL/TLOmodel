from pathlib import Path
import os
import pytest

from tlo import Date, Simulation, logging
from tlo.methods import (
    demography,
    contraception,
    healthburden,
    healthsystem,
    enhanced_lifestyle,
    epi
)

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 500

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"
except NameError:
    # running interactively
    resourcefilepath = "resources"


# @pytest.fixture(autouse=True)
# def disable_logging():
#     logging.disable(logging.DEBUG)


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


# @pytest.fixture(scope='module')
def test_no_health_system(tmpdir):

    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            service_availability=[],
            mode_appt_constraints=2,  # no constraints by officer type/time
            ignore_cons_constraints=True,
            ignore_priority=True,
            capabilities_coefficient=1.0,
            disable=False
        )
    )
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(epi.Epi(resourcefilepath=resourcefilepath))

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    df = sim.population.props

    # check no vaccines being administered through health system
    # only hpv currently, all others start as individual events
    assert (df.ep_hpv == 0).all()

    # check only 3 doses max of dtp
    assert (df.ep_dtp <= 3).all()
    assert (df.ep_pneumo <= 3).all()

