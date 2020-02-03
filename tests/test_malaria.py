import logging
import os
from pathlib import Path
import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    contraception,
    healthburden,
    healthsystem,
    enhanced_lifestyle,
    malaria,
    dx_algorithm_child,
    dx_algorithm_adult,
    healthseekingbehaviour,
    symptommanager,
)

start_date = Date(2010, 1, 1)
end_date = Date(2014, 1, 1)
popsize = 100

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'


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
    service_availability = list(['malaria*'])
    malaria_strat = 1  # levels: 0 = national; 1 = district
    malaria_testing = 0.35  # adjust this to match rdt/tx levels

    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability,
                                           mode_appt_constraints=0,
                                           ignore_cons_constraints=True,
                                           ignore_priority=True,
                                           capabilities_coefficient=1.0))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthseekingbehaviour.HealthSeekingBehaviour())
    sim.register(dx_algorithm_child.DxAlgorithmChild())
    sim.register(dx_algorithm_adult.DxAlgorithmAdult())
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(malaria.Malaria(resourcefilepath=resourcefilepath,
                                 level=malaria_strat, testing=malaria_testing, itn=None))

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # check scheduled malaria deaths occurring only due to severe malaria (not clinical or asym)
    df = sim.population.props
    assert not (
        df.ma_date_death & ((df.ma_inf_type == 'clinical') | (df.ma_inf_type == 'none'))).any()

    # check cases /  treatment are occurring
    assert not (df.ma_clinical_counter == 0).all()
    assert not (df.ma_date_tx == pd.NaT).all()

    # check clinical malaria in pregnancy counter not including males
    assert not any((df.sex == 'M') & (df.ma_clinical_preg_counter > 0))
