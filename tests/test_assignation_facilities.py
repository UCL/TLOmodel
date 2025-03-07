import os
from pathlib import Path

import pytest
from tlo.methods.healthsystem import HealthSystem, HealthSystemChangeParameters

from tlo import DAYS_IN_MONTH, DAYS_IN_YEAR, Date, Module, Simulation, logging
from tlo.methods import Metadata, demography
from tlo.methods import (
    Metadata,
    chronicsyndrome,
    demography,
    enhanced_lifestyle,
    epi,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    mockitis,
    simplified_births,
    symptommanager,
    tb,
)

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 500

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_setting_climate_disruptions(seed):
    """Check that the switches for turning on/off climate disruptions to healthcare access work"""
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath)
    )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    assert sim.modules['HealthSystem'].services_affected_precip == 'none'


def test_setting_climate_disruptions(seed):
    """Check that the switches for turning on/off climate disruptions to healthcare access work"""
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  climate_ssp = 'ssp126',
                                  climate_model_ensemble_model='mean',
                                  services_affected_precip = 'ANC')
    )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    assert sim.modules['HealthSystem'].services_affected_precip == 'ANC'

