"""
Basic tests for the Diarrhoea Module
"""
import os
from pathlib import Path

from tlo import Date, Simulation
from tlo.methods import (
    contraception,
    demography,
    diarrhoea,
    healthsystem,
    enhanced_lifestyle,
    symptommanager,
    healthburden,
    healthseekingbehaviour, dx_algorithm_child)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_basic_run_of_diarhoea_module_no_health_care():
    start_date = Date(2010, 1, 1)
    end_date = Date(2011, 1, 2)
    popsize = 200

    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath))

    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)

    # Todo: Check there is some non-zero level diarrhaea


def test_basic_run_of_diarhoea_module_with_health_care():
    start_date = Date(2010, 1, 1)
    end_date = Date(2011, 1, 2)
    popsize = 200

    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
    sim.register(diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath))
    sim.register(dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath))

    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)

# Todo: Check there is some treatments due to diarrhoea
