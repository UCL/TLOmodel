# todo: add a "basic_run" with checks on the logical consistency of all values, as done for the original diabetes/HT
# (inserted here is the basic scaffold)

import os
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.methods import (
    contraception,
    demography,
    depression,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager, ncds,
)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

def routine_checks(sim):
    """
    Insert checks here:
    """

    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    #åassert (df.dtypes == orig.dtypes).all()

    # check that someone has had onset of each condition

    df = sim.population.props
    assert df.nc_diabetes.any()
    assert df.nc_hypertension.any()
    assert df.nc_depression.any()
    assert df.nc_chronic_lower_back_pain.any()
    assert df.nc_chronic_kidney_disease.any()
    assert df.nc_chronic_ischemic_hd.any()
    assert df.nc_cancers.any()

    # check that someone has had onset of each event

    assert df.nc_ever_stroke.any()

    # check that someone dies of each condition

    assert df.cause_of_death.loc[~df.is_alive].str.startswith('nc_diabetes').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('nc_hypertension').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('nc_depression').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('nc_chronic_lower_back_pain').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('nc_chronic_ischemic_hd').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('nc_chronic_kidney_disease').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('nc_cancers').any()

    pass


def test_basic_run():
    # --------------------------------------------------------------------------
    # Create and run a short but big population simulation for use in the tests
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 ncds.Ncds(resourcefilepath=resourcefilepath)
                 )

    sim.make_initial_population(n=5000)
    sim.simulate(end_date=Date(year=2015, month=1, day=1))


    routine_checks(sim)




