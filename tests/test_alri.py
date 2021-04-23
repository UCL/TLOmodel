"""Test file for the Alri module (alri.py)"""


import os
from pathlib import Path


from tlo import Date, Simulation
from tlo.methods import (
    alri,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)


# Path to the resource files used by the disease and intervention methods
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')


# %% Run the Simulation

def check_dtypes(sim):
    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_basic_run():
    start_date = Date(2010, 1, 1)
    end_date = Date(2013, 1, 1)
    pop_size = 100

    sim = Simulation(start_date=start_date)

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
        dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        alri.Alri(resourcefilepath=resourcefilepath)
    )

    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)


# TODO -- @ines: We need some tests here to make sure everything is working, like in the diarrhoea code.


