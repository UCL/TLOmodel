import os
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epilepsy,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
)

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 10000


@pytest.fixture
def simulation(seed):
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date, seed=seed)

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            mode_appt_constraints=0),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        healthseekingbehaviour. HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
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


def test_epilepsy_treatment(simulation):
    """Run simulation in which HSI occur, due to high risk of onset of epilepsy with frequent seizures."""
    params = simulation.modules['Epilepsy'].parameters
    params['base_3m_prob_epilepsy'] = 0.10
    params['prop_inc_epilepsy_seiz_freq'] = 1.00
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)
