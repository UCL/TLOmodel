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
    other_adult_cancers,
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
popsize = 2000


# %% Construction of simulation objects:
def make_simulation_healthsystemdisabled(seed):
    """Make the simulation with:
    * the demography module with the OtherDeathsPoll not running
    """
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 other_adult_cancers.OtherAdultCancer(resourcefilepath=resourcefilepath),
                 oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
                 )
    return sim


def make_simulation_nohsi(seed):
    """Make the simulation with:
    * the healthsystem enable but with no service availabilty (so no HSI run)
    """
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=[]),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 other_adult_cancers.OtherAdultCancer(resourcefilepath=resourcefilepath),
                 oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath)
                 )
    return sim


# %% Manipulation of parameters:
def zero_out_init_prev(sim):
    # Set initial prevalence to zero:
    sim.modules['OtherAdultCancer'].parameters['in_prop_other_adult_cancer_stage'] = [0.0, 0.0, 0.0]
    return sim


def seed_init_prev_in_first_stage_only(sim):
    # Set initial prevalence to zero:
    sim.modules['OtherAdultCancer'].parameters['in_prop_other_adult_cancer_stage'] = [1.0, 0.0, 0.0]
    return sim


def make_high_init_prev(sim):
    # Set initial prevalence to a high value:
    sim.modules['OtherAdultCancer'].parameters['in_prop_other_adult_cancer_stage'] = [0.1, 0.1, 0.1]
    return sim


def incr_rate_of_onset_lgd(sim):
    # Rate of cancer onset per 3 months:
    sim.modules['OtherAdultCancer'].parameters['r_site_confined_none'] = 0.05
    return sim


def zero_rate_of_onset_lgd(sim):
    # Rate of cancer onset per 3 months:
    sim.modules['OtherAdultCancer'].parameters['r_site_confined_none'] = 0.00
    return sim


def incr_rates_of_progression(sim):
    # Rates of cancer progression per 3 months:
    sim.modules['OtherAdultCancer'].parameters['r_local_ln_site_confined_other_adult_ca'] *= 2
    sim.modules['OtherAdultCancer'].parameters['r_metastatic_local_ln'] *= 2
    return sim


def make_treatment_ineffective(sim):
    # Treatment effect of 1.0 will not retard progression
    sim.modules['OtherAdultCancer'].parameters['rr_local_ln_undergone_curative_treatment'] = 1.0
    sim.modules['OtherAdultCancer'].parameters['rr_metastatic_undergone_curative_treatment'] = 1.0
    return sim


def make_treatment_perfectly_effective(sim):
    # Treatment effect of 0.0 will stop progression
    sim.modules['OtherAdultCancer'].parameters['rr_local_ln_other_adult_ca_undergone_curative_treatment'] = 0.0
    sim.modules['OtherAdultCancer'].parameters['rr_metastatic_undergone_curative_treatment'] = 0.0
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

    # for convenience, define a bool for any stage of cancer
    df['oac_status_any_stage'] = df.oac_status != 'none'

    # get df for alive persons:
    df = df.loc[df.is_alive]

    # check that no one under twenty has cancer
    assert not df.loc[df.age_years < 15].oac_status_any_stage.any()

    # check that diagnosis and treatment is never given to someone who has never had cancer:
    assert pd.isnull(df.loc[df.oac_status == 'none', 'oac_date_diagnosis']).all()
    assert pd.isnull(df.loc[df.oac_status == 'none', 'oac_date_treatment']).all()
    assert pd.isnull(df.loc[df.oac_status == 'none', 'oac_date_palliative_care']).all()
    assert (df.loc[df.oac_status == 'none', 'oac_stage_at_which_treatment_given'] == 'none').all()

    # check that treatment is never done for those with oac_status 'metastatic'
    assert 0 == (df.oac_stage_at_which_treatment_given == 'metastatic').sum()
    assert 0 == (df.loc[~pd.isnull(df.oac_date_treatment)].oac_stage_at_which_treatment_given == 'none').sum()

    # Create a short hand form of the symptom manager module
    symptom_manager = sim.modules['SymptomManager']
    # check that those with symptom are a subset of those with cancer:
    assert set(symptom_manager.who_has('early_other_adult_ca_symptom')).issubset(df.index[df.oac_status != 'none'])

    # check that those diagnosed are a subset of those with the symptom (and that the date makes sense):
    assert set(df.index[~pd.isnull(df.oac_date_diagnosis)]).issubset(df.index[df.oac_status_any_stage])
    assert set(df.index[~pd.isnull(df.oac_date_diagnosis)]).issubset(
        symptom_manager.who_has('early_other_adult_ca_symptom')
    )
    assert (df.loc[~pd.isnull(df.oac_date_diagnosis)].oac_date_diagnosis <= sim.date).all()

    # check that date diagnosed is consistent with the age of the person (ie. not before they were 15.0
    age_at_dx = (df.loc[~pd.isnull(df.oac_date_diagnosis)].oac_date_diagnosis - df.loc[
        ~pd.isnull(df.oac_date_diagnosis)].date_of_birth)
    assert all([int(x.days / 365.25) >= 15 for x in age_at_dx])

    # check that those treated are a subset of those diagnosed (and that the order of dates makes sense):
    assert set(df.index[~pd.isnull(df.oac_date_treatment)]).issubset(df.index[~pd.isnull(df.oac_date_diagnosis)])
    assert (df.loc[~pd.isnull(df.oac_date_treatment)].oac_date_diagnosis <= df.loc[
        ~pd.isnull(df.oac_date_treatment)].oac_date_treatment).all()

    # check that those on palliative care are a subset of those diagnosed (and that the order of dates makes sense):
    assert set(df.index[~pd.isnull(df.oac_date_palliative_care)]).issubset(df.index[~pd.isnull(df.oac_date_diagnosis)])
    assert (df.loc[~pd.isnull(df.oac_date_palliative_care)].oac_date_diagnosis <= df.loc[
        ~pd.isnull(df.oac_date_palliative_care)].oac_date_diagnosis).all()


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
    df = sim.population.props
    assert (df.loc[df.is_alive].oac_status == 'none').all()


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
def test_check_progression_through_stages_is_happening(seed):
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
    sim.make_initial_population(n=5000)

    # force that all persons aged over 15 are in the site_confined stage to begin with:
    sim.population.props.loc[
        sim.population.props.is_alive & (sim.population.props.age_years >= 15), "oac_status"] = 'site_confined'
    check_configuration_of_population(sim)

    # Simulate
    sim.simulate(end_date=Date(2010, 8, 1))
    check_dtypes(sim)
    check_configuration_of_population(sim)

    # check that there are now some people in each of the later stages:
    df = sim.population.props
    assert len(df.loc[df.is_alive & (df.oac_status != 'none')]) > 0
    assert not pd.isnull(df.oac_status).any()
    assert (df.loc[df.is_alive].oac_status.value_counts().drop(index='none') > 0).all()

    # check that some people have died of other adult cancer
    yll = sim.modules['HealthBurden'].years_life_lost
    assert yll['OtherAdultCancer'].sum() > 0

    # check that people are being diagnosed, going onto treatment and palliative care:
    assert (df.oac_date_diagnosis > start_date).any()
    assert (df.oac_date_treatment > start_date).any()
    assert (df.oac_stage_at_which_treatment_given != 'none').any()
    assert (df.oac_date_palliative_care > start_date).any()


@pytest.mark.slow
def test_that_there_is_no_treatment_without_the_hsi_running(seed):

    # I have checked that there is no treatment without the hsi running but this test is not running and not sure why

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
        sim.population.props.is_alive & (sim.population.props.age_years >= 15), "oac_status"] = 'site_confined'
    check_configuration_of_population(sim)

    # Simulate
    sim.simulate(end_date=Date(2010, 6, 1))
    check_dtypes(sim)
    check_configuration_of_population(sim)

    # check that there are now some people in each of the later stages:
    df = sim.population.props
    assert len(df.loc[df.is_alive & (df.oac_status != 'none')]) > 0
    assert not pd.isnull(df.oac_status).any()
    assert (df.loc[df.is_alive].oac_status.value_counts().drop(index='none') > 0).all()

    # check that some people have died of other adult cancer
    yll = sim.modules['HealthBurden'].years_life_lost
    assert yll['OtherAdultCancer'].sum() > 0

    # w/o healthsystem - check that people are NOT being diagnosed, going onto treatment and palliative care:
    assert not (df.oac_date_diagnosis > start_date).any()
    assert not (df.oac_date_treatment > start_date).any()
    assert not (df.oac_stage_at_which_treatment_given != 'none').any()
    assert not (df.oac_date_palliative_care > start_date).any()


def test_check_progression_through_stages_is_blocked_by_treatment(seed):
    """Put all people into the first stage but on treatment, let progression happen, and check that people do move into
    a late stage or die"""
    sim = make_simulation_healthsystemdisabled(seed=seed)

    # set initial prevalence to be zero
    sim = zero_out_init_prev(sim)

    # no incidence of new cases
    sim = zero_rate_of_onset_lgd(sim)

    # remove effect of treatment:
    sim = make_treatment_perfectly_effective(sim)

    # increase progression rates:
    sim = incr_rates_of_progression(sim)

    # make inital popuation
    sim.make_initial_population(n=popsize)

    # force that all persons aged over 15 are in the site-confined stage to begin with:
    has_lgd = sim.population.props.is_alive & (sim.population.props.age_years >= 15)
    sim.population.props.loc[has_lgd, "oac_status"] = 'site_confined'

    # force that they are all symptomatic, diagnosed and already on treatment:
    sim.modules['SymptomManager'].change_symptom(
        person_id=has_lgd.index[has_lgd].tolist(),
        symptom_string='early_other_adult_ca_symptom',
        add_or_remove='+',
        disease_module=sim.modules['OtherAdultCancer']
    )
    sim.population.props.loc[
        sim.population.props.is_alive & (sim.population.props.age_years >= 15) & (
            sim.population.props.oac_status != 'none'), "oac_date_diagnosis"] = sim.date
    sim.population.props.loc[
        sim.population.props.is_alive & (sim.population.props.age_years >= 15) & (
            sim.population.props.oac_status != 'none'), "oac_date_treatment"] = sim.date
    sim.population.props.loc[sim.population.props.is_alive & (
            sim.population.props.age_years >= 15) & (
            sim.population.props.oac_status != 'none'), "oac_stage_at_which_treatment_given"] = 'site_confined'
    check_configuration_of_population(sim)

    # Simulate
    sim.simulate(end_date=Date(2010, 6, 1))
    check_dtypes(sim)
#   check_configuration_of_population(sim)

    # check that there are not any people in each of the later stages and everyone is still in 'site_confined':
    df = sim.population.props
    assert len(df.loc[df.is_alive & (df.age_years >= 15), "oac_status"]) > 0
    assert (df.loc[df.is_alive & (df.age_years >= 15), "oac_status"].isin(["none", "site_confined"])).all()
    assert (df.loc[has_lgd.index[has_lgd].tolist(), "oac_status"] == "site_confined").all()

    # check that no people have died of other adult cancer
    yll = sim.modules['HealthBurden'].years_life_lost
    assert 'YLL_OtherAdultCancer' not in yll.columns
