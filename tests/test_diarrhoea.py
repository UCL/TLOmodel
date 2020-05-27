"""
Basic tests for the Diarrhoea Module
"""
import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.methods import (
    contraception,
    demography,
    diarrhoea,
    healthsystem,
    enhanced_lifestyle,
    symptommanager,
    healthburden,
    healthseekingbehaviour,
    dx_algorithm_child,
    labour,
    pregnancy_supervisor)

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

def check_configuration_of_properties(sim):
    # check that the properties are ok:

    df = sim.population.props

    # Those that have never had diarrhoea, should have null values:
    assert pd.isnull(df.loc[df['gi_last_diarrhoea_dehydration'] == 'none',[
        'gi_last_diarrhoea_date_of_onset',
        'gi_last_diarrhoea_duration',
        'gi_last_diarrhoea_recovered_date',
        'gi_last_diarrhoea_death_date']
    ]).any().any()

    # Those that have had diarrhoea, should have a pathogen and a number of days duration
    assert (df.loc[
        ~pd.isnull(df['gi_last_diarrhoea_date_of_onset']),
        'gi_last_diarrhoea_pathogen'] != 'none').all()

    assert not pd.isnull(df.loc[
        ~pd.isnull(df['gi_last_diarrhoea_date_of_onset']),
        'gi_last_diarrhoea_duration']).any()


    # Those that have had diarrhoea, should have either a recovery date or a death_date
    has_recovery_date = ~pd.isnull(df.loc[
        ~pd.isnull(df['gi_last_diarrhoea_date_of_onset']),
        'gi_last_diarrhoea_recovered_date'])

    has_death_date = ~pd.isnull(df.loc[
        ~pd.isnull(df['gi_last_diarrhoea_date_of_onset']),
        'gi_last_diarrhoea_death_date'])

    assert (has_recovery_date | has_death_date).all()

    # Those for whom the death date has past should be dead
    assert not df.loc[
        ~pd.isnull(df['gi_last_diarrhoea_death_date']) &
        (df['gi_last_diarrhoea_death_date'] < sim.date),
        'is_alive'].any()


def test_basic_run_of_diarrhoea_module():
    start_date = Date(2010, 1, 1)
    end_date = Date(2010, 12, 31)
    popsize = 1000

    sim = Simulation(start_date=start_date)

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
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )

    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)

    check_configuration_of_properties(sim)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)
    check_configuration_of_properties(sim)

    # Todo: Check there is some non-zero level diarrhaea; that there has been some treatment; etc


# Run with no intervention: check no effect; duration == diff in death or recovery


# Run with intervention: check that cures happen etc




