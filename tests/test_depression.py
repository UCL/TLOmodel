import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.lm import LinearModel, LinearModelType
from tlo.methods import (
    demography,
    depression,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')


@pytest.mark.slow
def test_configuration_of_properties(seed):
    # --------------------------------------------------------------------------
    # Create and run a short but big population simulation for use in the tests
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath))

    sim.make_initial_population(n=2000)
    sim.simulate(end_date=Date(year=2013, month=1, day=1))
    # --------------------------------------------------------------------------

    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()

    # Check that all value of all properties for depression make sesne
    df = sim.population.props

    def check_ever_vs_now_properties(col_of_ever, col_of_now):
        # Confirms logical consistency between an property for ever occurrence of something and its current occurrence
        assert set(col_of_now.loc[col_of_now].index).issubset(col_of_ever.loc[col_of_ever].index)

    check_ever_vs_now_properties(df['de_ever_depr'], df['de_depr'])
    assert (pd.isnull(df['de_intrinsic_3mo_risk_of_depr_resolution']) == ~df['de_depr']).all()

    never_had_an_episode = ~df['de_ever_depr']
    assert (pd.isnull(df.loc[never_had_an_episode, 'de_date_init_most_rec_depr'])).all()
    assert (pd.isnull(df.loc[never_had_an_episode, 'de_date_depr_resolved'])).all()
    assert (pd.isnull(df.loc[never_had_an_episode, 'de_intrinsic_3mo_risk_of_depr_resolution'])).all()
    assert (False == (df.loc[never_had_an_episode, 'de_ever_diagnosed_depression'])).all()
    assert (False == (df.loc[never_had_an_episode, 'de_on_antidepr'])).all()
    assert (False == (df.loc[never_had_an_episode, 'de_ever_talk_ther'])).all()
    assert (False == (df.loc[never_had_an_episode, 'de_ever_non_fatal_self_harm_event'])).all()

    had_an_episode_now_resolved = (
        ~pd.isnull(df['de_date_init_most_rec_depr']) & (~pd.isnull(df['de_date_depr_resolved'])))
    assert (df.loc[had_an_episode_now_resolved, 'de_ever_depr']).all()
    assert not (df.loc[had_an_episode_now_resolved, 'de_depr']).any()
    assert (df.loc[had_an_episode_now_resolved, 'de_date_depr_resolved'] <= sim.date).all()

    had_an_episode_still_ongoing = (
        ~pd.isnull(df['de_date_init_most_rec_depr']) & pd.isnull(df['de_date_depr_resolved']))
    assert (df.loc[had_an_episode_still_ongoing, 'de_ever_depr']).all()
    assert (df.loc[had_an_episode_still_ongoing, 'de_depr']).all()

    # check access to intervention
    # NB. These tests assume that no one who is not depressed would be wrongly diagnosed as depressed.
    # (i.e. the specificity of the DxTest = 1.0)
    check_ever_vs_now_properties(df['de_ever_depr'], df['de_ever_non_fatal_self_harm_event'])
    check_ever_vs_now_properties(df['de_ever_diagnosed_depression'], df['de_ever_talk_ther'])
    check_ever_vs_now_properties(df['de_ever_diagnosed_depression'], df['de_on_antidepr'])

    # Check No one aged less than 15 is depressed
    assert not (df.loc[df['age_years'] < 15, 'de_depr']).any()
    assert not (df.loc[df['age_years'] < 15, 'de_ever_depr']).any()
    assert pd.isnull(df.loc[df['age_years'] < 15, 'de_date_init_most_rec_depr']).all()
    assert pd.isnull(df.loc[df['age_years'] < 15, 'de_date_depr_resolved']).all()
    assert pd.isnull(df.loc[df['age_years'] < 15, 'de_intrinsic_3mo_risk_of_depr_resolution']).all()
    assert not (df.loc[df['age_years'] < 15, 'de_ever_diagnosed_depression']).any()
    assert not (df.loc[df['age_years'] < 15, 'de_on_antidepr']).any()
    assert not (df.loc[df['age_years'] < 15, 'de_ever_talk_ther']).any()
    assert not (df.loc[df['age_years'] < 15, 'de_ever_non_fatal_self_harm_event']).any()
    assert not (df.loc[df['age_years'] < 15, 'de_ever_diagnosed_depression']).any()

    # There is some non-zero prevalence of ever having had depression in the initial population
    assert df.loc[df['date_of_birth'] < sim.start_date, 'de_ever_depr'].sum()


def get_hsi_events_run(sim):
    return {
        event_details.event_name
        for event_details in sim.modules['HealthSystem'].hsi_event_counts.keys()
    }


@pytest.mark.slow
def test_hsi_functions(tmpdir, seed):
    # With health seeking and healthsystem functioning and no constraints --
    #   --- people should have both talking therapies and antidepressants
    # --------------------------------------------------------------------------
    # Create and run a longer simulation on a small population.
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           mode_appt_constraints=0,
                                           cons_availability='all',
                                           hsi_event_count_log_period="simulation"),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath))

    # Make it more likely that individual with depression seeks care
    sim.modules['Depression'].parameters['prob_3m_selfharm_depr'] = 0.25
    sim.modules['Depression'].linearModels['Risk_of_SelfHarm_per3mo'] = LinearModel(
        LinearModelType.MULTIPLICATIVE,
        sim.modules['Depression'].parameters['prob_3m_selfharm_depr']
    )

    sim.make_initial_population(n=2000)

    df = sim.population.props

    # zero-out all instances of current or ever depression, or ever talking therapies
    df['de_depr'] = False
    df['de_ever_depr'] = False
    df['de_date_init_most_rec_depr'] = pd.NaT
    df['de_date_depr_resolved'] = pd.NaT
    df['de_intrinsic_3mo_risk_of_depr_resolution'] = np.NaN
    df['de_ever_diagnosed_depression'] = False
    df['de_on_antidepr'] = False
    df['de_ever_talk_ther'] = False
    df['de_ever_non_fatal_self_harm_event'] = False

    sim.simulate(end_date=Date(year=2012, month=1, day=1))
    # --------------------------------------------------------------------------

    df = sim.population.props

    # Check that there have been been some cases of Talking Therapy and anti-depressants
    assert df['de_ever_talk_ther'].sum()

    hsi_events_run = get_hsi_events_run(sim)
    assert 'HSI_Depression_TalkingTherapy' in hsi_events_run
    assert 'HSI_Depression_Start_Antidepressant' in hsi_events_run
    assert 'HSI_Depression_Refill_Antidepressant' in hsi_events_run


@pytest.mark.slow
def test_hsi_functions_no_medication_available(tmpdir, seed):
    # With health seeking and healthsystem functioning but no medication available ---
    #   --- people should have talking therapy but not antidepressants,
    #       (though appointments to try to start them can occur)

    # --------------------------------------------------------------------------
    # Create and run a longer simulation on a small population
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           mode_appt_constraints=0,
                                           cons_availability='none',
                                           hsi_event_count_log_period="simulation"),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath))

    # Make it more likely that individual with depression seeks care
    sim.modules['Depression'].parameters['prob_3m_selfharm_depr'] = 0.25
    sim.modules['Depression'].linearModels['Risk_of_SelfHarm_per3mo'] = LinearModel(
        LinearModelType.MULTIPLICATIVE,
        sim.modules['Depression'].parameters['prob_3m_selfharm_depr']
    )

    sim.make_initial_population(n=2000)

    df = sim.population.props

    # zero-out all instances of current or ever depression, or ever talking therapies
    df['de_depr'] = False
    df['de_ever_depr'] = False
    df['de_date_init_most_rec_depr'] = pd.NaT
    df['de_date_depr_resolved'] = pd.NaT
    df['de_intrinsic_3mo_risk_of_depr_resolution'] = np.NaN
    df['de_ever_diagnosed_depression'] = False
    df['de_on_antidepr'] = False
    df['de_ever_talk_ther'] = False
    df['de_ever_non_fatal_self_harm_event'] = False

    sim.simulate(end_date=Date(year=2012, month=1, day=1))
    # --------------------------------------------------------------------------

    df = sim.population.props

    # Check that there have been been some cases of Talking Therapy but no-one on anti-depressants
    assert df['de_ever_talk_ther'].sum()
    assert 0 == df['de_on_antidepr'].sum()

    hsi_events_run = get_hsi_events_run(sim)
    assert 'HSI_Depression_TalkingTherapy' in hsi_events_run
    assert 'HSI_Depression_Start_Antidepressant' in hsi_events_run
    assert 'HSI_Depression_Refill_Antidepressant' not in hsi_events_run


@pytest.mark.slow
def test_hsi_functions_no_healthsystem_capability(tmpdir, seed):
    # With health seeking and healthsystem functioning and no medication ---
    #   --- people should have nothing (no talking therapy or antidepressants) and no HSI events run at all

    # --------------------------------------------------------------------------
    log_config = {
        "filename": "log",   # The prefix for the output file. A timestamp will be added to this.
        "directory": tmpdir,  # The default output path is `./output`. Change it here, if necessary
    }

    # Create and run a longer simulation on a small population
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=seed, log_config=log_config)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable_and_reject_all=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath))

    # Make it more likely that individual with depression seeks care
    sim.modules['Depression'].parameters['prob_3m_selfharm_depr'] = 0.25
    sim.modules['Depression'].linearModels['Risk_of_SelfHarm_per3mo'] = LinearModel(
        LinearModelType.MULTIPLICATIVE,
        sim.modules['Depression'].parameters['prob_3m_selfharm_depr']
    )

    sim.make_initial_population(n=2000)

    df = sim.population.props

    # zero-out all instances of current or ever depression, or ever talking therapies
    df['de_depr'] = False
    df['de_ever_depr'] = False
    df['de_date_init_most_rec_depr'] = pd.NaT
    df['de_date_depr_resolved'] = pd.NaT
    df['de_intrinsic_3mo_risk_of_depr_resolution'] = np.NaN
    df['de_ever_diagnosed_depression'] = False
    df['de_on_antidepr'] = False
    df['de_ever_talk_ther'] = False
    df['de_ever_non_fatal_self_harm_event'] = False

    sim.simulate(end_date=Date(year=2012, month=1, day=1))
    # --------------------------------------------------------------------------

    df = sim.population.props

    # Check that there have been been no use of talking therapy or anti-depressants
    assert 0 == df['de_ever_talk_ther'].sum()
    assert 0 == df['de_on_antidepr'].sum()
