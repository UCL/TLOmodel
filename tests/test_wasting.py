"""
Basic tests for the Wasting Module
"""
import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.events import IndividualScopeEventMixin
from tlo.methods import (
    demography,
    wasting,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)

from tlo.methods.healthsystem import HSI_Event

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

    # Those that were never wasted, should have normal WHZ score:
    assert (df.loc[~df.un_ever_wasted & ~df.date_of_birth.isna(), 'un_WHZ_category'] == 'WHZ>=-2').all().all()

    # Those that were never wasted and not clinically malnourished,
    # should have not_applicable/null values for all the other properties:
    assert pd.isnull(df.loc[~df.un_ever_wasted & ~df.date_of_birth.isna() & df.un_clinical_acute_malnutrition == 'well',
                            ['un_last_wasting_date_of_onset',
                             'un_sam_death_date',
                             'un_am_recovery_date',
                             'un_am_discharge_date',
                             'un_acute_malnutrition_tx_start_date']
                            ]).all().all()

    # Those that were ever wasted, should have a WHZ socre below <-2
    assert (df.loc[df.un_ever_wasted, 'un_WHZ_category'] != 'WHZ>=-2').all()

    # Those that had wasting and no treatment, should have either a recovery date or a death_date
    # (but not both)
    has_recovery_date = ~pd.isnull(df.loc[df.un_ever_wasted & pd.isnull(df.un_acute_malnutrition_tx_start_date),
                                          'un_am_recovery_date'])
    has_death_date = ~pd.isnull(df.loc[df.un_ever_wasted & pd.isnull(df.un_acute_malnutrition_tx_start_date),
                                       'un_sam_death_date'])

    has_recovery_date_or_death_date = has_recovery_date | has_death_date
    has_both_recovery_date_and_death_date = has_recovery_date & has_death_date
    assert has_recovery_date_or_death_date.all()
    assert not has_both_recovery_date_and_death_date.any()

    # Those for whom the death date has past should be dead
    assert not df.loc[df.un_ever_wasted & (df['un_sam_death_date'] < sim.date), 'is_alive'].any()
    assert not df.loc[(df.un_clinical_acute_malnutrition == 'SAM') & (
        df['un_sam_death_date'] < sim.date), 'is_alive'].any()

    # Check that those in a current episode have symptoms of diarrhoea [caused by the diarrhoea module]
    #  but not others (among those who are alive)
    has_symptoms_of_wasting = set(sim.modules['SymptomManager'].who_has('weight_loss'))
    has_symptoms = set([p for p in has_symptoms_of_wasting if
                        'Wasting' in sim.modules['SymptomManager'].causes_of(p, 'weight_loss')
                        ])

    in_current_episode_before_recovery = \
        df.is_alive & \
        df.un_ever_wasted & \
        (df.un_last_wasting_date_of_onset <= sim.date) & \
        (sim.date <= df.un_am_recovery_date)
    set_of_person_id_in_current_episode_before_recovery = set(
        in_current_episode_before_recovery[in_current_episode_before_recovery].index
    )

    in_current_episode_before_death = \
        df.is_alive & \
        df.un_ever_wasted & \
        (df.un_last_wasting_date_of_onset <= sim.date) & \
        (sim.date <= df.un_sam_death_date)
    set_of_person_id_in_current_episode_before_death = set(
        in_current_episode_before_death[in_current_episode_before_death].index
    )

    in_current_episode_before_cure = \
        df.is_alive & \
        df.un_ever_wasted & \
        (df.un_last_wasting_date_of_onset <= sim.date) & \
        (df.un_acute_malnutrition_tx_start_date <= sim.date) & \
        pd.isnull(df.un_am_recovery_date) & \
        pd.isnull(df.un_sam_death_date)
    set_of_person_id_in_current_episode_before_cure = set(
        in_current_episode_before_cure[in_current_episode_before_cure].index
    )

    assert set() == set_of_person_id_in_current_episode_before_recovery.intersection(
        set_of_person_id_in_current_episode_before_death
    )

    set_of_person_id_in_current_episode = set_of_person_id_in_current_episode_before_recovery.union(
        set_of_person_id_in_current_episode_before_death, set_of_person_id_in_current_episode_before_cure
    )
    assert set_of_person_id_in_current_episode == has_symptoms


def test_basic_run_of_diarrhoea_module_with_default_params():
    # Check that the module run and that properties are maintained correctly, using health system and default parameters
    start_date = Date(2010, 1, 1)
    end_date = Date(2010, 12, 31)
    popsize = 1000

    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 wasting.Wasting(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )

    sim.make_initial_population(n=popsize)
    check_configuration_of_properties(sim)

    sim.simulate(end_date=end_date)

    check_dtypes(sim)
    check_configuration_of_properties(sim)
