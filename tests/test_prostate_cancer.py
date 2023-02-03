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
    prostate_cancer,
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
popsize = 10000


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
                 oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 prostate_cancer.ProstateCancer(resourcefilepath=resourcefilepath))
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
                 oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 prostate_cancer.ProstateCancer(resourcefilepath=resourcefilepath)
                 )
    return sim


# %% Manipulation of parameters:
def zero_out_init_prev(sim):
    # Set initial prevalence to zero:
    sim.modules['ProstateCancer'].parameters['init_prop_prostate_ca_stage'] = [0.0] * 3
    return sim


def seed_init_prev_in_first_stage_only(sim):
    # Set initial prevalence to zero:
    sim.modules['ProstateCancer'].parameters['init_prop_prostate_ca_stage'] = [0.0] * 3
    # Put everyone in first stage ('prostate_confined')
    sim.modules['ProstateCancer'].parameters['init_prop_prostate_ca_stage'][0] = 1.0
    return sim


"""
init_prop_prostate_ca_stage
init_prop_urinary_symptoms_by_stage
init_prop_pelvic_pain_symptoms_by_stage
init_prop_with_urinary_symptoms_diagnosed_prostate_ca_by_stage
init_prop_with_pelvic_pain_symptoms_diagnosed_prostate_ca_by_stage
init_prop_treatment_status_prostate_ca
"""


def make_high_init_prev(sim):
    # Set initial prevalence to a high value:
    sim.modules['ProstateCancer'].parameters['init_prop_prostate_ca_stage'] = [0.1] * 3
    return sim


def incr_rate_of_onset_lgd(sim):
    # Rate of cancer onset per 3 months:
    sim.modules['ProstateCancer'].parameters['r_prostate_confined_prostate_ca_none'] = 0.05
    return sim


def zero_rate_of_onset_lgd(sim):
    # Rate of cancer onset per 3 months:
    sim.modules['ProstateCancer'].parameters['r_prostate_confined_prostate_ca_none'] = 0.00
    return sim


def incr_rates_of_progression(sim):
    # Rates of cancer progression per 3 months:
    sim.modules['ProstateCancer'].parameters['r_local_ln_prostate_ca_prostate_confined'] *= 5
    sim.modules['ProstateCancer'].parameters['r_metastatic_prostate_ca_local_ln'] *= 5
    return sim


def make_treatment_ineffective(sim):
    # Treatment effect of 1.0 will not retard progression
    sim.modules['ProstateCancer'].parameters['rr_local_ln_prostate_ca_undergone_curative_treatment'] = 1.0
    sim.modules['ProstateCancer'].parameters['rr_metastatic_prostate_ca_undergone_curative_treatment'] = 1.0
    return sim


def make_treamtment_perfectly_effective(sim):
    # Treatment effect of 0.0 will stop progression
    sim.modules['ProstateCancer'].parameters['rr_local_ln_prostate_ca_undergone_curative_treatment'] = 0.0
    sim.modules['ProstateCancer'].parameters['rr_metastatic_prostate_ca_undergone_curative_treatment'] = 0.0
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

    # for convenience, define a bool for any stage of prostate cancer
    df.loc[df.pc_status != 'none', 'pc_status_any_stage'] = True
    df.loc[df.pc_status == 'none', 'pc_status_any_stage'] = False

    # check that no one under 35 has cancer
    assert not df.loc[df.age_years < 35].pc_status_any_stage.any()

    # check that diagnosis and treatment is never applied to someone who has never had cancer:
    assert pd.isnull(df.loc[df.pc_status == 'none', 'pc_date_diagnosis']).all()
    assert pd.isnull(df.loc[df.pc_status == 'none', 'pc_date_treatment']).all()
    assert pd.isnull(df.loc[df.pc_status == 'none', 'pc_date_palliative_care']).all()
    assert (df.loc[df.pc_status == 'none', 'pc_stage_at_which_treatment_given'] == 'none').all()

    # check that treatment is never given for those with pc_status 'metastatic'
    assert 0 == (df.pc_stage_at_which_treatment_given == 'metastatic').sum()
    assert 0 == (df.loc[~pd.isnull(df.pc_date_treatment)].pc_stage_at_which_treatment_given == 'none').sum()

    # check that those with symptom are a subset of those with cancer:
    # todo: note this will not always be true - people can have urinary symptoms for other reasons and we will need
    # todo: to add this in
    assert set(sim.modules['SymptomManager'].who_has('urinary')).issubset(df.index[df.pc_status != 'none'])
    assert set(sim.modules['SymptomManager'].who_has('pelvic_pain')).issubset(df.index[df.pc_status != 'none'])

    # check that those diagnosed are a subset of those with the symptom (and that the date makes sense):
    assert set(df.index[~pd.isnull(df.pc_date_diagnosis)]).issubset(df.index[df.pc_status_any_stage])
    # todo: note that urinary symptoms may go after treatment (if we code that as intended)
    assert (df.loc[~pd.isnull(df.pc_date_diagnosis)].pc_date_diagnosis <= sim.date).all()
    assert set(df.index[~pd.isnull(df.pc_date_diagnosis)]).issubset(
        set(sim.modules['SymptomManager'].who_has('urinary')
            ).union(sim.modules['SymptomManager'].who_has('pelvic_pain')
                    )
    )

    # check that date diagnosed is consistent with the age of the person (ie. not before they were 35.0
    age_at_dx = (df.loc[~pd.isnull(df.pc_date_diagnosis)].pc_date_diagnosis - df.loc[
        ~pd.isnull(df.pc_date_diagnosis)].date_of_birth)
    assert all([int(x.days / 365.25) >= 35 for x in age_at_dx])

    # check that those treated are a subset of those diagnosed (and that the order of dates makes sense):
    assert set(df.index[~pd.isnull(df.pc_date_treatment)]).issubset(df.index[~pd.isnull(df.pc_date_diagnosis)])
    assert (df.loc[~pd.isnull(df.pc_date_treatment)].pc_date_diagnosis <= df.loc[
        ~pd.isnull(df.pc_date_treatment)].pc_date_treatment).all()

    # check that those on palliative care are a subset of those diagnosed (and that the order of dates makes sense):
    assert set(df.index[~pd.isnull(df.pc_date_palliative_care)]).issubset(df.index[~pd.isnull(df.pc_date_diagnosis)])
    assert (df.loc[~pd.isnull(df.pc_date_palliative_care)].pc_date_diagnosis <= df.loc[
        ~pd.isnull(df.pc_date_palliative_care)].pc_date_diagnosis).all()


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
    sim.simulate(end_date=Date(2010, 4, 1))
    check_dtypes(sim)
    check_configuration_of_population(sim)


@pytest.mark.slow
def test_check_progression_through_stages_is_happeneing(seed):

    # progression through stages is happening as I have checked - I'm not sure why these tests are failing

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

    # force that all persons aged over 35 are in the prostate confined stage to begin with:
    sim.population.props.loc[
        sim.population.props.is_alive & (sim.population.props.age_years >= 35), "pc_status"] = 'prostate_confined'
    check_configuration_of_population(sim)

    # Simulate
    sim.simulate(end_date=Date(2010, 6, 1))
    check_dtypes(sim)
    check_configuration_of_population(sim)

    # check that there are now some people in each of the later stages:
    df = sim.population.props
    assert len(df.loc[df.is_alive & (df.pc_status != 'none')]) > 0
    assert not pd.isnull(df.pc_status).any()
    assert (df.loc[df.is_alive].pc_status.value_counts().drop(index='none') > 0).all()

    # check that some people have died of prostate cancer
    yll = sim.modules['HealthBurden'].years_life_lost
    assert yll['ProstateCancer'].sum() > 0

    # check that people are being diagnosed, going onto treatment and palliative care:
    assert (df.pc_date_diagnosis > start_date).any()
    assert (df.pc_date_treatment > start_date).any()
    assert (df.pc_stage_at_which_treatment_given != 'none').any()
    assert (df.pc_date_palliative_care > start_date).any()


@pytest.mark.slow
def test_that_there_is_no_treatment_without_the_hsi_running(seed):

    # i've checked that there is no treatment without the hsi running - but I can't see why this test fails

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

    # force that all persons aged over 35 are in the prostate-confined stage to begin with:
    sim.population.props.loc[
        sim.population.props.is_alive & (sim.population.props.age_years >= 35), "pc_status"] = 'prostate_confined'
    check_configuration_of_population(sim)

    # Simulate
    sim.simulate(end_date=Date(2010, 6, 1))
    check_dtypes(sim)
    check_configuration_of_population(sim)

    # check that there are now some people in each of the later stages:
    df = sim.population.props
    assert len(df.loc[df.is_alive & (df.pc_status != 'none')]) > 0
    assert not pd.isnull(df.oc_status).any()
    assert (df.loc[df.is_alive].pc_status.value_counts().drop(index='none') > 0).all()

    # check that some people have died of prostate cancer
    yll = sim.modules['HealthBurden'].years_life_lost
    assert yll['ProstateCancer'].sum() > 0

    # w/o healthsystem - check that people are NOT being diagnosed, going onto treatment and palliative care:
    assert not (df.pc_date_diagnosis > start_date).any()
    assert not (df.pc_date_treatment > start_date).any()
    assert not (df.pc_stage_at_which_treatment_given != 'none').any()
    assert not (df.pc_date_palliative_care > start_date).any()


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

    # force that all persons aged over 35 are in the prostate-confined stage to begin with:
    has_lgd = sim.population.props.is_alive & (sim.population.props.age_years >= 35)
    sim.population.props.loc[has_lgd, "pc_status"] = 'prostate_confined'

    # todo: as below for pelvic_pain symptom
    # force that they are all symptomatic, diagnosed and already on treatment:
    sim.modules['SymptomManager'].change_symptom(
        person_id=has_lgd.index[has_lgd].tolist(),
        symptom_string='urinary',
        add_or_remove='+',
        disease_module=sim.modules['ProstateCancer']
    )
    sim.population.props.loc[
        sim.population.props.is_alive & (sim.population.props.age_years >= 35), "pc_date_diagnosis"] = sim.date
    sim.population.props.loc[
        sim.population.props.is_alive & (sim.population.props.age_years >= 35), "pc_date_treatment"] = sim.date
    sim.population.props.loc[sim.population.props.is_alive & (
            sim.population.props.age_years >= 35), "pc_stage_at_which_treatment_given"] = 'prostate_confined'
    check_configuration_of_population(sim)

    # Simulate
    sim.simulate(end_date=Date(2010, 4, 1))
    check_dtypes(sim)
    check_configuration_of_population(sim)

    # check that there are not any people in each of the later stages and everyone is still in 'prostate_confined':
    df = sim.population.props
    assert len(df.loc[df.is_alive & (df.age_years >= 35), "pc_status"]) > 0
    assert (df.loc[df.is_alive & (df.age_years >= 35), "pc_status"].isin(["none", "prostate_confined"])).all()
    assert (df.loc[has_lgd.index[has_lgd].tolist(), "pc_status"] == "prostate_confined").all()

    # check that no people have died of prostate cancer
    yll = sim.modules['HealthBurden'].years_life_lost
    assert 'YLL_ProstateCancer_ProstateCancer' not in yll.columns
