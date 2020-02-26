import os
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    depression,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    symptommanager,
    healthseekingbehaviour)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')


def test_configuration_of_properties():
    # --------------------------------------------------------------------------
    # Create and run a short but big population simulation for use in the tests
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
    sim.make_initial_population(n=20000)
    sim.simulate(end_date=Date(year=2015, month=1, day=1))
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
    assert (False == (df.loc[never_had_an_episode, 'de_ever_current_talk_ther'])).all()
    assert (False == (df.loc[never_had_an_episode, 'de_ever_non_fatal_self_harm_event'])).all()

    had_an_episode_now_resolved = (
        ~pd.isnull(df['de_date_init_most_rec_depr']) & (~pd.isnull(df['de_date_depr_resolved'])))
    assert (df.loc[had_an_episode_now_resolved, 'de_ever_depr'] == True).all()
    assert (df.loc[had_an_episode_now_resolved, 'de_depr'] == False).all()
    assert (df.loc[had_an_episode_now_resolved, 'de_date_depr_resolved'] <= sim.date).all()

    had_an_episode_still_ongoing = (
        ~pd.isnull(df['de_date_init_most_rec_depr']) & pd.isnull(df['de_date_depr_resolved']))
    assert (df.loc[had_an_episode_still_ongoing, 'de_ever_depr'] == True).all()
    assert (df.loc[had_an_episode_still_ongoing, 'de_depr'] == True).all()

    # check access to intervention
    # NB. These tests assume that no one who is not depressed would be wrongly diagnosed as depressed.
    # (i.e. the specificity of the DxTest = 1.0)
    check_ever_vs_now_properties(df['de_ever_depr'], df['de_ever_non_fatal_self_harm_event'])
    check_ever_vs_now_properties(df['de_ever_diagnosed_depression'], df['de_ever_current_talk_ther'])
    check_ever_vs_now_properties(df['de_ever_diagnosed_depression'], df['de_on_antidepr'])

    # Check No one aged less than 15 is depressed
    assert (df.loc[df['age_years'] < 15, 'de_depr'] == False).all()
    assert (df.loc[df['age_years'] < 15, 'de_ever_depr'] == False).all()
    assert pd.isnull(df.loc[df['age_years'] < 15, 'de_date_init_most_rec_depr']).all()
    assert pd.isnull(df.loc[df['age_years'] < 15, 'de_date_depr_resolved']).all()
    assert pd.isnull(df.loc[df['age_years'] < 15, 'de_intrinsic_3mo_risk_of_depr_resolution']).all()
    assert (df.loc[df['age_years'] < 15, 'de_ever_diagnosed_depression'] == False).all()
    assert (df.loc[df['age_years'] < 15, 'de_on_antidepr'] == False).all()
    assert (df.loc[df['age_years'] < 15, 'de_ever_current_talk_ther'] == False).all()
    assert (df.loc[df['age_years'] < 15, 'de_ever_non_fatal_self_harm_event'] == False).all()
    assert (df.loc[df['age_years'] < 15, 'de_ever_diagnosed_depression'] == False).all()

    # There is some non-zero prevalence of ever having had depression in the initial population
    assert df.loc[df['date_of_birth'] < sim.start_date, 'de_ever_depr'].sum()


def test_hsi_functions(tmpdir):
    # With health seeking and healthsystem functioning and no constraints --
    #   --- people should have both talking therapies and antidepressants
    # --------------------------------------------------------------------------
    # Create and run a longer simulation on a small population.

    sim = Simulation(start_date=Date(year=2010, month=1, day=1))

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        mode_appt_constraints=0,
        ignore_cons_constraints=True))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(depression.Depression(resourcefilepath=resourcefilepath))
    f = sim.configure_logging("log", directory=tmpdir, custom_levels={"*": logging.INFO})

    sim.seed_rngs(0)
    sim.make_initial_population(n=2000)

    # zero-out all instances of current or ever depression, or ever talking therapies
    sim.population.props['de_depr'] = False
    sim.population.props['de_ever_depr'] = False
    sim.population.props['de_date_init_most_rec_depr'] = pd.NaT
    sim.population.props['de_date_depr_resolved'] = pd.NaT
    sim.population.props['de_intrinsic_3mo_risk_of_depr_resolution'] = np.NaN
    sim.population.props['de_ever_diagnosed_depression'] = False
    sim.population.props['de_on_antidepr'] = False
    sim.population.props['de_ever_current_talk_ther'] = False
    sim.population.props['de_ever_non_fatal_self_harm_event'] = False

    sim.simulate(end_date=Date(year=2012, month=1, day=1))
    # --------------------------------------------------------------------------

    df = sim.population.props

    output = parse_log_file(f)

    # Check that there have been been some cases of Talking Therapy and anti-depressants
    assert df['de_ever_current_talk_ther'].sum()

    hsi = output['tlo.methods.healthsystem']['HSI_Event']
    assert 'Depression_TalkingTherapy' in hsi['TREATMENT_ID'].values
    assert 'Depression_Antidepressant_Start' in hsi['TREATMENT_ID'].values
    assert 'Depression_Antidepressant_Refill' in hsi['TREATMENT_ID'].values


def test_hsi_functions_no_medication_available(tmpdir):
    # With health seeking and healthsystem functioning but no medication available ---
    #   --- people should have talking therapy but not antidepressants,
    #       (though appointments to try to start them can occur)

    # --------------------------------------------------------------------------
    # Create and run a longer simulation on a small population
    sim = Simulation(start_date=Date(year=2010, month=1, day=1))

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        mode_appt_constraints=0))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(depression.Depression(resourcefilepath=resourcefilepath))
    f = sim.configure_logging("log", directory=tmpdir, custom_levels={"*": logging.INFO})

    sim.seed_rngs(0)
    sim.make_initial_population(n=2000)

    # zero-out all instances of current or ever depression, or ever talking therapies
    sim.population.props['de_depr'] = False
    sim.population.props['de_ever_depr'] = False
    sim.population.props['de_date_init_most_rec_depr'] = pd.NaT
    sim.population.props['de_date_depr_resolved'] = pd.NaT
    sim.population.props['de_intrinsic_3mo_risk_of_depr_resolution'] = np.NaN
    sim.population.props['de_ever_diagnosed_depression'] = False
    sim.population.props['de_on_antidepr'] = False
    sim.population.props['de_ever_current_talk_ther'] = False
    sim.population.props['de_ever_non_fatal_self_harm_event'] = False

    # zero-out the availability of the consumable that is required for the treatment of antidepressants
    item_code = sim.modules['Depression'].parameters['anti_depressant_medication_item_code']
    sim.modules['HealthSystem'].prob_unique_item_codes_available.loc[item_code] = 0.0

    sim.simulate(end_date=Date(year=2012, month=1, day=1))
    # --------------------------------------------------------------------------

    df = sim.population.props

    output = parse_log_file(f)

    # Check that there have been been some cases of Talking Therapy and anti-depressants
    assert df['de_ever_current_talk_ther'].sum()
    assert 0 == df['de_on_antidepr'].sum()

    hsi = output['tlo.methods.healthsystem']['HSI_Event']
    assert 'Depression_TalkingTherapy' in hsi['TREATMENT_ID'].values
    assert 'Depression_Antidepressant_Start' in hsi['TREATMENT_ID'].values
    assert 'Depression_Antidepressant_Refill' not in hsi['TREATMENT_ID'].values

    # Check no anti-depressants used
    cons = output['tlo.methods.healthsystem']['Consumables']
    assert (False == cons['Available'].apply(pd.Series)[item_code]).all()


def test_hsi_functions_no_healthsystem_capability(tmpdir):
    # With health seeking and healthsystem functioning and no medication ---
    #   --- people should have nothing (no talking therapy or antidepressants) and no HSI events run at all

    # --------------------------------------------------------------------------
    # Create and run a longer simulation on a small population
    sim = Simulation(start_date=Date(year=2010, month=1, day=1))

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        mode_appt_constraints=2,
        ignore_cons_constraints=True,
        capabilities_coefficient=0.0))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(depression.Depression(resourcefilepath=resourcefilepath))
    f = sim.configure_logging("log", directory=tmpdir, custom_levels={"*": logging.INFO})

    sim.seed_rngs(0)
    sim.make_initial_population(n=2000)

    # zero-out all instances of current or ever depression, or ever talking therapies
    sim.population.props['de_depr'] = False
    sim.population.props['de_ever_depr'] = False
    sim.population.props['de_date_init_most_rec_depr'] = pd.NaT
    sim.population.props['de_date_depr_resolved'] = pd.NaT
    sim.population.props['de_intrinsic_3mo_risk_of_depr_resolution'] = np.NaN
    sim.population.props['de_ever_diagnosed_depression'] = False
    sim.population.props['de_on_antidepr'] = False
    sim.population.props['de_ever_current_talk_ther'] = False
    sim.population.props['de_ever_non_fatal_self_harm_event'] = False

    sim.simulate(end_date=Date(year=2012, month=1, day=1))
    # --------------------------------------------------------------------------

    df = sim.population.props

    output = parse_log_file(f)

    # Check that there have been been no some cases of talking Therapy and anti-depressants
    assert 0 == df['de_ever_current_talk_ther'].sum()
    assert 0 == df['de_on_antidepr'].sum()

    hsi = output['tlo.methods.healthsystem']['HSI_Event']
    assert 0 == hsi['did_run'].sum()

    # Check no antidepresants used
    assert 'Consumables' not in output['tlo.methods.healthsystem']

