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
popsize = 100

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"
except NameError:
    # running interactively
    resourcefilepath = "resources"


@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.DEBUG)


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


# @pytest.fixture(scope='module')
def test_sims(tmpdir):
    service_availability = []

    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            service_availability=service_availability,
            mode_appt_constraints=2,  # no constraints by officer type/time
            ignore_cons_constraints=True,
            ignore_priority=True,
            capabilities_coefficient=1.0,
            disable=True,
        )
    )  # disables the health system constraints so all HSI events run
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(epi.Epi(resourcefilepath=resourcefilepath))

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # check scheduled malaria deaths occurring only due to severe malaria (not clinical or asym)
    df = sim.population.props

    # check no vaccines being administered
    assert not df.ep_bcg.all()
    assert (df.ep_dtp == 0).all()
    assert (df.ep_opv == 0).all()
    assert (df.ep_hep == 0).all()
    assert (df.ep_hib == 0).all()
    assert (df.ep_rota == 0).all()
