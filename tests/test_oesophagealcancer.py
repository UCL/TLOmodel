import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    oesophagealcancer,
    simplified_births,
    symptommanager,
)

# %% Setup:
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

# parameters for whole suite of tests:
start_date = Date(2010, 1, 1)
popsize = 1000


# %% Construction of simulation objects:
def make_simulation_healthsystemdisabled(seed):
    """Make the simulation with:
    * the demography module with the OtherDeathsPoll not running
    """
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath)
                 )
    return sim


def make_simulation_nohsi(seed):
    """Make the simulation with:
    * the healthsystem enable but with no service availabilty (so no HSI run)
    """
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=[]),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath)
                 )
    return sim


# %% Manipulation of parameters:
def zero_out_init_prev(sim):
    # Set initial prevalence to zero:
    sim.modules['OesophagealCancer'].parameters['init_prop_oes_cancer_stage'] = [0.0] * 6
    return sim


def seed_init_prev_in_first_stage_only(sim):
    # Set initial prevalence to zero:
    sim.modules['OesophagealCancer'].parameters['init_prop_oes_cancer_stage'] = [0.0] * 6
    # Put everyone in first stage ('low-grade-dysplasia')
    sim.modules['OesophagealCancer'].parameters['init_prop_oes_cancer_stage'][0] = 1.0
    return sim


def make_high_init_prev(sim):
    # Set initial prevalence to a high value:
    sim.modules['OesophagealCancer'].parameters['init_prop_oes_cancer_stage'] = [0.1] * 6
    return sim


def incr_rate_of_onset_lgd(sim):
    # Rate of cancer onset per 3 months:
    sim.modules['OesophagealCancer'].parameters['r_low_grade_dysplasia_none'] = 0.05
    return sim


def zero_rate_of_onset_lgd(sim):
    # Rate of cancer onset per 3 months:
    sim.modules['OesophagealCancer'].parameters['r_low_grade_dysplasia_none'] = 0.00
    return sim


def incr_rates_of_progression(sim):
    # Rates of cancer progression per 3 months:
    sim.modules['OesophagealCancer'].parameters['r_high_grade_dysplasia_low_grade_dysp'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage1_high_grade_dysp'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage2_stage1'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage3_stage2'] *= 5
    sim.modules['OesophagealCancer'].parameters['r_stage4_stage3'] *= 5
    return sim


def make_treatment_ineffective(sim):
    # Treatment effect of 1.0 will not retard progression
    sim.modules['OesophagealCancer'].parameters['rr_high_grade_dysp_undergone_curative_treatment'] = 1.0
    sim.modules['OesophagealCancer'].parameters['rr_stage1_undergone_curative_treatment'] = 1.0
    sim.modules['OesophagealCancer'].parameters['rr_stage2_undergone_curative_treatment'] = 1.0
    sim.modules['OesophagealCancer'].parameters['rr_stage3_undergone_curative_treatment'] = 1.0
    sim.modules['OesophagealCancer'].parameters['rr_stage4_undergone_curative_treatment'] = 1.0
    return sim


def make_treamtment_perfectly_effective(sim):
    # Treatment effect of 0.0 will stop progression
    sim.modules['OesophagealCancer'].parameters['rr_high_grade_dysp_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage1_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage2_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage3_undergone_curative_treatment'] = 0.0
    sim.modules['OesophagealCancer'].parameters['rr_stage4_undergone_curative_treatment'] = 0.0
    return sim


# %% Checks:
def check_dtypes(sim):
    # check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def check_configuration_of_population(sim):
    # get df for alive persons:
    df = sim.population.props.loc[sim.population.props.is_alive]

    # for convenience, define a bool for any stage of dysplasia of cancer
    df['oc_status_any_dysplasia_or_cancer'] = df.oc_status != 'none'

    # check that no one under twenty has cancer
    assert not df.loc[df.age_years < 20].oc_status_any_dysplasia_or_cancer.any()

    # check that diagnosis and treatment is never applied to someone who has never had cancer:
    assert pd.isnull(df.loc[df.oc_status == 'none', 'oc_date_diagnosis']).all()
    assert pd.isnull(df.loc[df.oc_status == 'none', 'oc_date_treatment']).all()
    assert pd.isnull(df.loc[df.oc_status == 'none', 'oc_date_palliative_care']).all()
    assert (df.loc[df.oc_status == 'none', 'oc_stage_at_which_treatment_applied'] == 'none').all()

    # check that treatment is never done for those with oc_status 'stage4'
    assert 0 == (df.oc_stage_at_which_treatment_applied == 'level4').sum()
    assert 0 == (df.loc[~pd.isnull(df.oc_date_treatment)].oc_stage_at_which_treatment_applied == 'none').sum()

    # check that those with symptom are a subset of those with cancer:
    assert set(sim.modules['SymptomManager'].who_has('dysphagia')).issubset(df.index[df.oc_status != 'none'])

    # check that those diagnosed are a subset of those with the symptom (and that the date makes sense):
    assert set(df.index[~pd.isnull(df.oc_date_diagnosis)]).issubset(df.index[df.oc_status_any_dysplasia_or_cancer])
    assert set(df.index[~pd.isnull(df.oc_date_diagnosis)]).issubset(sim.modules['SymptomManager'].who_has('dysphagia'))
    assert (df.loc[~pd.isnull(df.oc_date_diagnosis)].oc_date_diagnosis <= sim.date).all()

    # check that date diagnosed is consistent with the age of the person (ie. not before they were 20.0
    age_at_dx = (df.loc[~pd.isnull(df.oc_date_diagnosis)].oc_date_diagnosis - df.loc[
        ~pd.isnull(df.oc_date_diagnosis)].date_of_birth)
    assert all([int(x.days / 365.25) >= 20 for x in age_at_dx])

    # check that those treated are a subset of those diagnosed (and that the order of dates makes sense):
    assert set(df.index[~pd.isnull(df.oc_date_treatment)]).issubset(df.index[~pd.isnull(df.oc_date_diagnosis)])
    assert (df.loc[~pd.isnull(df.oc_date_treatment)].oc_date_diagnosis <= df.loc[
        ~pd.isnull(df.oc_date_treatment)].oc_date_treatment).all()

    # check that those on palliative care are a subset of those diagnosed (and that the order of dates makes sense):
    assert set(df.index[~pd.isnull(df.oc_date_palliative_care)]).issubset(df.index[~pd.isnull(df.oc_date_diagnosis)])
    assert (df.loc[~pd.isnull(df.oc_date_palliative_care)].oc_date_diagnosis <= df.loc[
        ~pd.isnull(df.oc_date_palliative_care)].oc_date_diagnosis).all()


# %% Tests:
def test_initial_config_of_pop_high_prevalence(seed):
    """Tests of the the way the population is configured: with high initial prevalence values """
    sim = make_simulation_healthsystemdisabled(seed=seed)
    sim = make_high_init_prev(sim)
    sim.make_initial_population(n=popsize)
    check_dtypes(sim)
    check_configuration_of_population(sim)


def test_initial_config_of_pop_zero_prevalence(seed):
    """Tests of the the way the population is configured: with zero initial prevalence values """
    sim = make_simulation_healthsystemdisabled(seed=seed)
    sim = zero_out_init_prev(sim)
    sim.make_initial_population(n=popsize)
    check_dtypes(sim)
    check_configuration_of_population(sim)


def test_initial_config_of_pop_usual_prevalence(seed):
    """Tests of the the way the population is configured: with usual initial prevalence values"""
    sim = make_simulation_healthsystemdisabled(seed=seed)
    sim.make_initial_population(n=popsize)
    check_dtypes(sim)
    check_configuration_of_population(sim)


@pytest.mark.slow
def test_run_sim_from_high_prevalence(seed):
    """Run the simulation from the usual prevalence values and high rates of incidence and check configuration of
    properties at the end"""
    sim = make_simulation_healthsystemdisabled(seed=seed)
    sim = make_high_init_prev(sim)
    sim = incr_rates_of_progression(sim)
    sim = incr_rate_of_onset_lgd(sim)
    sim.make_initial_population(n=popsize)
    check_dtypes(sim)
    check_configuration_of_population(sim)
    sim.simulate(end_date=Date(2012, 1, 1))
    check_dtypes(sim)
    check_configuration_of_population(sim)


@pytest.mark.slow
def test_check_progression_through_stages_is_happeneing(seed):
    """Put all people into the first stage, let progression happen (with no treatment effect) and check that people end
    up in late stages and some die of this cause.
    Use a functioning healthsystem that allows HSI and check that diagnosis, treatment and palliative care is happening.
    """
    sim = make_simulation_healthsystemdisabled(seed=seed)

    # set initial prevalence to be zero
    sim = zero_out_init_prev(sim)

    # no incidence of new cases
    sim = zero_rate_of_onset_lgd(sim)

    # remove effect of treatment:
    sim = make_treatment_ineffective(sim)

    # increase progression rates:
    sim = incr_rates_of_progression(sim)

    # make initial population
    sim.make_initial_population(n=popsize)

    # force that all persons aged over 20 are in the low_grade dysplasia stage to begin with:
    sim.population.props.loc[
        sim.population.props.is_alive & (sim.population.props.age_years >= 20), "oc_status"] = 'low_grade_dysplasia'
    check_configuration_of_population(sim)

    # Simulate
    sim.simulate(end_date=Date(2015, 1, 1))
    check_dtypes(sim)
    check_configuration_of_population(sim)

    # check that there are now some people in each of the later stages:
    df = sim.population.props
    assert len(df.loc[df.is_alive & (df.oc_status != 'none')]) > 0
    assert not (~df.date_of_birth.isna() & df.oc_status.isna()).any()
    assert (df.loc[df.is_alive].oc_status.value_counts().drop(index='none') > 0).all()

    # check that some people have died of oesophagal cancer
    yll = sim.modules['HealthBurden'].years_life_lost
    assert yll['OesophagealCancer'].sum() > 0

    # check that people are being diagnosed, going onto treatment and palliative care:
    assert (df.oc_date_diagnosis > start_date).any()
    assert (df.oc_date_treatment > start_date).any()
    assert (df.oc_stage_at_which_treatment_applied != 'none').any()
    assert (df.oc_date_palliative_care > start_date).any()


@pytest.mark.slow
def test_that_there_is_no_treatment_without_the_hsi_running(seed):
    """Put all people into the first stage, let progression happen (with no treatment effect) and check that people end
    up in late stages and some die of this cause.
    Use a healthsystem that does not allows HSI and check that diagnosis, treatment and palliative care do not occur.
    """
    sim = make_simulation_nohsi(seed=seed)

    # set initial prevalence to be zero
    sim = zero_out_init_prev(sim)

    # no incidence of new cases
    sim = zero_rate_of_onset_lgd(sim)

    # remove effect of treatment:
    sim = make_treatment_ineffective(sim)

    # increase progression rates:
    sim = incr_rates_of_progression(sim)

    # make initial population
    sim.make_initial_population(n=popsize)

    # force that all persons aged over 20 are in the low_grade dysplasia stage to begin with:
    sim.population.props.loc[
        sim.population.props.is_alive & (sim.population.props.age_years >= 20), "oc_status"] = 'low_grade_dysplasia'
    check_configuration_of_population(sim)

    # Simulate
    sim.simulate(end_date=Date(2015, 1, 1))
    check_dtypes(sim)
    check_configuration_of_population(sim)

    # check that there are now some people in each of the later stages:
    df = sim.population.props
    assert len(df.loc[df.is_alive & (df.oc_status != 'none')]) > 0
    assert not (~df.date_of_birth.isna() & df.oc_status.isna()).any()
    assert (df.loc[df.is_alive].oc_status.value_counts().drop(index='none') > 0).all()

    # check that some people have died of oesophagal cancer
    yll = sim.modules['HealthBurden'].years_life_lost
    assert yll['OesophagealCancer'].sum() > 0

    # w/o healthsystem - check that people are NOT being diagnosed, going onto treatment and palliative care:
    assert not (~df.date_of_birth.isna() & (df.oc_date_diagnosis > start_date)).any()
    assert not (~df.date_of_birth.isna() & (df.oc_date_treatment > start_date)).any()
    assert not (~df.date_of_birth.isna() & (df.oc_stage_at_which_treatment_applied != 'none')).any()
    assert not (~df.date_of_birth.isna() & (df.oc_date_palliative_care > start_date)).any()


@pytest.mark.slow
def test_check_progression_through_stages_is_blocked_by_treatment(seed):
    """Put all people into the first stage but on treatment, let progression happen, and check that people do move into
    a late stage or die"""
    sim = make_simulation_healthsystemdisabled(seed=seed)

    # set initial prevalence to be zero
    sim = zero_out_init_prev(sim)

    # no incidence of new cases
    sim = zero_rate_of_onset_lgd(sim)

    # remove effect of treatment:
    sim = make_treamtment_perfectly_effective(sim)

    # increase progression rates:
    sim = incr_rates_of_progression(sim)

    # make inital popuation
    sim.make_initial_population(n=popsize)

    # force that all persons aged over 20 are in the low_grade dysplasis stage to begin with:
    has_lgd = sim.population.props.is_alive & (sim.population.props.age_years >= 20)
    sim.population.props.loc[has_lgd, "oc_status"] = 'low_grade_dysplasia'

    # force that they are all symptomatic, diagnosed and already on treatment:
    sim.modules['SymptomManager'].change_symptom(
        person_id=has_lgd.index[has_lgd].tolist(),
        symptom_string='dysphagia',
        add_or_remove='+',
        disease_module=sim.modules['OesophagealCancer']
    )
    sim.population.props.loc[
        sim.population.props.is_alive & (sim.population.props.age_years >= 20), "oc_date_diagnosis"] = sim.date
    sim.population.props.loc[
        sim.population.props.is_alive & (sim.population.props.age_years >= 20), "oc_date_treatment"] = sim.date
    sim.population.props.loc[sim.population.props.is_alive & (
            sim.population.props.age_years >= 20), "oc_stage_at_which_treatment_applied"] = 'low_grade_dysplasia'
    check_configuration_of_population(sim)

    # Simulate
    sim.simulate(end_date=Date(2015, 1, 1))
    check_dtypes(sim)
    check_configuration_of_population(sim)

    # check that there are not any people in each of the later stages and everyone is still in 'low_grade_dysplasia':
    df = sim.population.props
    assert len(df.loc[df.is_alive & (df.age_years >= 20), "oc_status"]) > 0
    assert (df.loc[df.is_alive & (df.age_years >= 20), "oc_status"].isin(["none", "low_grade_dysplasia"])).all()
    assert (df.loc[has_lgd.index[has_lgd].tolist(), "oc_status"] == "low_grade_dysplasia").all()

    # check that no people have died of oesophageal cancer
    yll = sim.modules['HealthBurden'].years_life_lost
    assert 'YLL_OesophagealCancer_OesophagealCancer' not in yll.columns
