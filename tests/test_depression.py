import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.methods import (
    contraception,
    demography,
    depression,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    symptommanager,
    healthseekingbehaviour)

# --------------------------------------------------------------------------
# Create and run a simulation for use in the tests
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

# Establish the simulation object
sim = Simulation(start_date=Date(year=2010, month=1, day=1))

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True))
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(depression.Depression(resourcefilepath=resourcefilepath))

sim.seed_rngs(0)
sim.make_initial_population(n=2000)
sim.simulate(end_date=Date(year=2012, month=1, day=1))
# --------------------------------------------------------------------------

def test_dtypes():
    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()

def test_configuration_of_properties():
    # Check that all value of all properties for depression make sesne
    df = sim.population.props

    def is_subset_of(col_of_ever, col_of_now):
        """
        Confirms logical consistency between an property for ever occurrence of something and its current occurrence
        """
        # If it it occurring now, it must have ever occurred:
        assert (col_of_now[col_of_now] == col_of_ever[col_of_now]).all()

        # If it has never occurred, it cannot be occurring now
        assert (col_of_ever[(~col_of_ever)] == col_of_now[~col_of_ever]).all()

    is_subset_of(df['de_ever_depr'], df['de_depr'])
    assert (pd.isnull(df['de_intrinsic_3mo_risk_of_depr_resolution']) == ~df['de_depr']).all()

    never_had_an_episode = ~df['de_ever_depr']
    assert (pd.isnull(df['de_date_init_most_rec_depr']) == never_had_an_episode).all()
    assert (pd.isnull(df['de_date_depr_resolved']) == never_had_an_episode).all()

    had_an_episode_now_resolved = (~pd.isnull(df['de_date_init_most_rec_depr']) & pd.isnull(df['de_date_depr_resolved']))
    assert (df.loc[had_an_episode_now_resolved, 'de_ever_depr'] == True).all()
    assert (df.loc[had_an_episode_now_resolved, 'de_depr'] == False).all()
    assert (df.loc[~pd.isnull(df['de_date_depr_resolved']), 'de_date_depr_resolved'] <= sim.date).all()

    had_an_episode_still_ongoing = (~pd.isnull(df['de_date_init_most_rec_depr']) & ~pd.isnull(df['de_date_depr_resolved']))
    assert (df.loc[had_an_episode_still_ongoing, 'de_ever_depr'] == True).all()
    assert (df.loc[had_an_episode_still_ongoing, 'de_depr'] == True).all()

    is_subset_of(df['de_ever_depr'], df['de_ever_non_fatal_self_harm_event'])
    is_subset_of(df['de_ever_diagnosed_depression'], df['de_ever_current_talk_ther'])
    is_subset_of(df['de_ever_diagnosed_depression'], df['de_on_antidepr'])


def test_epi_assumptions():
    # Check that all value of all properties for depression make sense
    df = sim.population.props

    # No one aged less than 15 is depressed

    # There have been some deaths due to suicide


def test_hsi_functions():
    pass
    # Check that there have been been some cases of Talking Therapy

    # Check that there have been some uses of anti-depressants


# TODO: TEST IN WHICH THERE IS NO HEALTH SEEKING BHEAVIOUR OR NO RESOURCES OF HSI APPOINTMENTS AND THERE SHOULD NOT BE ANY TREATMENT APART FROM THOSE INITIALLY

# TODO: TEST IN WHICH HEALTH SYSEM HAS NO ANTIDEPRESSANT MEDICATION -- THERE SHOULD BE NO ONE ON TREATMENT



