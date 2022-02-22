import os
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import demography, enhanced_lifestyle, epilepsy, healthburden, healthsystem

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 10000


@pytest.fixture
def simulation():
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date, seed=0)

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            mode_appt_constraints=0),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        epilepsy.Epilepsy(resourcefilepath=resourcefilepath)
    )
    return sim


@pytest.mark.slow
def test_dtypes(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()
