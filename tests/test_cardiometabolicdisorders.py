import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.methods import (
    cardio_metabolic_disorders,
    demography,
    depression,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)
from tlo.methods.cardio_metabolic_disorders import (
    CardioMetabolicDisordersDeathEvent,
    CardioMetabolicDisordersEvent,
    CardioMetabolicDisordersWeightLossEvent,
    HSI_CardioMetabolicDisorders_InvestigationFollowingSymptoms,
    HSI_CardioMetabolicDisorders_InvestigationNotFollowingSymptoms,
    HSI_CardioMetabolicDisorders_Refill_Medication,
    HSI_CardioMetabolicDisorders_SeeksEmergencyCareAndGetsTreatment,
    HSI_CardioMetabolicDisorders_StartWeightLossAndMedication,
)
from tlo.methods.demography import AgeUpdateEvent, age_at_date
from tlo.methods.healthsystem import HealthSystemScheduler

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

output_files = dict()


def routine_checks(sim):
    """
    Basic checks for the module: types of columns, onset and deaths of each condition and event
    """

    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()

    # check that someone has had onset of each condition

    df = sim.population.props
    assert df.nc_diabetes.any()
    assert df.nc_hypertension.any()
    assert df.nc_chronic_lower_back_pain.any()
    assert df.nc_chronic_kidney_disease.any()
    assert df.nc_chronic_ischemic_hd.any()

    # check that someone has had onset of each event

    assert df.nc_ever_stroke.any()
    assert df.nc_ever_heart_attack.any()

    # check that someone dies of each condition that has a death rate associated with it
    assert (df.cause_of_death.loc[~df.is_alive & ~df.date_of_birth.isna()] == 'diabetes').any()
    assert (df.cause_of_death.loc[~df.is_alive & ~df.date_of_birth.isna()] == 'chronic_ischemic_hd').any()
    assert (df.cause_of_death.loc[~df.is_alive & ~df.date_of_birth.isna()] == 'chronic_kidney_disease').any()
    assert (df.cause_of_death.loc[~df.is_alive & ~df.date_of_birth.isna()] == 'ever_stroke').any()
    assert (df.cause_of_death.loc[~df.is_alive & ~df.date_of_birth.isna()] == 'ever_heart_attack').any()


@pytest.mark.slow
def test_basic_run(seed):
    # --------------------------------------------------------------------------
    # Create and run a short but big population simulation for use in the tests
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath)
                 )

    # Set incidence/death rates of conditions to high values to decrease run time

    p = sim.modules['CardioMetabolicDisorders'].parameters

    p['diabetes_onset']["baseline_annual_probability"] = 0.75
    p['hypertension_onset']["baseline_annual_probability"] = 0.75
    p['chronic_lower_back_pain_onset']["baseline_annual_probability"] = 0.75
    p['chronic_kidney_disease_onset']["baseline_annual_probability"] = 0.75
    p['chronic_ischemic_hd_onset']["baseline_annual_probability"] = 0.75
    p['ever_stroke_onset']["baseline_annual_probability"] = 0.75
    p['ever_heart_attack_onset']["baseline_annual_probability"] = 0.75
    p['diabetes_death']["baseline_annual_probability"] = 0.75
    p['chronic_kidney_disease_death']["baseline_annual_probability"] = 0.75
    p['chronic_ischemic_hd_death']["baseline_annual_probability"] = 0.75
    p['ever_stroke_death']["baseline_annual_probability"] = 0.75
    p['ever_heart_attack_death']["baseline_annual_probability"] = 0.75

    sim.make_initial_population(n=5000)
    sim.simulate(end_date=Date(year=2011, month=1, day=1))

    routine_checks(sim)
    hsi_checks(sim)


@pytest.mark.slow
def test_basic_run_with_high_incidence_hypertension(seed):
    """This sim makes one condition very common and the others non-existent to check basic functions for prevalence and
    death"""

    # Create and run a short but big population simulation for use in the tests
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath)
                 )
    p = sim.modules['CardioMetabolicDisorders'].parameters

    # Set incidence of hypertension very high and incidence of all other conditions to 0, set initial prevalence of
    # other conditions to 0

    p['hypertension_onset']["baseline_annual_probability"] = float('inf')
    p['chronic_ischemic_hd_onset']["baseline_annual_probability"] = float('inf')
    p['diabetes_onset'] *= 0.0
    p['chronic_lower_back_pain_onset'] *= 0.0
    p['chronic_kidney_disease_onset'] *= 0.0
    p['diabetes_initial_prev'] *= 0.0
    p['chronic_lower_back_pain_initial_prev'] *= 0.0
    p['chronic_kidney_disease_initial_prev'] *= 0.0

    # Increase RR of heart disease very high if individual has hypertension
    p['chronic_ischemic_hd_onset']["rr_hypertension"] = float('inf')

    sim.make_initial_population(n=2000)
    sim.simulate(end_date=Date(year=2013, month=1, day=1))

    df = sim.population.props

    # check that no one has any conditions that were set to zero incidence
    assert not df.nc_diabetes.any()
    assert not df.nc_chronic_lower_back_pain.any()
    assert not df.nc_chronic_kidney_disease.any()

    # check that no one has died from conditions that were set to zero incidence
    assert not (df.loc[~df.is_alive & ~df.date_of_birth.isna(), 'cause_of_death'] == 'diabetes').any()
    assert not (df.loc[~df.is_alive & ~df.date_of_birth.isna(), 'cause_of_death'] == 'chronic_kidney_disease').any()

    # restrict population to individuals aged >=20 at beginning of sim
    start_date = pd.Timestamp(year=2010, month=1, day=1)
    df['start_date'] = pd.to_datetime(start_date)
    df['diff_years'] = age_at_date(df.start_date, df.date_of_birth)
    df = df[df['diff_years'] >= 20]
    df = df[df.is_alive]

    # check that everyone has hypertension and CIHD by end
    assert df.loc[df.is_alive & (df.age_years >= 20)].nc_hypertension.all()
    assert df.loc[df.is_alive & (df.age_years >= 20)].nc_chronic_ischemic_hd.all()


def hsi_checks(sim):
    """
    Basic checks for the module: types of columns, onset and deaths of each condition and event
    """

    df = sim.population.props

    # check that all conditions and events have someone diagnosed
    assert df.nc_diabetes_ever_diagnosed.any()
    assert df.nc_hypertension_ever_diagnosed.any()
    assert df.nc_chronic_lower_back_pain_ever_diagnosed.any()
    assert df.nc_chronic_kidney_disease_ever_diagnosed.any()
    assert df.nc_chronic_ischemic_hd_ever_diagnosed.any()
    assert df.nc_ever_stroke_ever_diagnosed.any()
    assert df.nc_ever_heart_attack_ever_diagnosed.any()

    # check that diagnosis and treatment are never applied to those who have never had the condition

    # check that all conditions and events have someone on medication
    assert df.nc_diabetes_on_medication.any()
    assert df.nc_hypertension_on_medication.any()
    assert df.nc_chronic_lower_back_pain_on_medication.any()
    assert df.nc_chronic_kidney_disease_on_medication.any()
    assert df.nc_chronic_ischemic_hd_on_medication.any()
    assert df.nc_ever_stroke_on_medication.any()
    assert df.nc_ever_heart_attack_on_medication.any()

    # check that those who have ever been diagnosed have a date of last test

    assert pd.notnull(df.loc[df.nc_diabetes_ever_diagnosed, 'nc_diabetes_date_last_test']).all()
    assert pd.notnull(df.loc[df.nc_hypertension_ever_diagnosed, 'nc_hypertension_date_last_test']).all()
    assert pd.notnull(
        df.loc[df.nc_chronic_lower_back_pain_ever_diagnosed, 'nc_chronic_lower_back_pain_date_last_test']).all()
    assert pd.notnull(df.loc[df.nc_chronic_kidney_disease_ever_diagnosed,
                             'nc_chronic_kidney_disease_date_last_test']).all()
    assert pd.notnull(df.loc[df.nc_chronic_ischemic_hd_ever_diagnosed, 'nc_chronic_ischemic_hd_date_last_test']).all()

    # check that everyone receiving medication for a condition has been diagnosed

    assert df.loc[df.is_alive & df.nc_diabetes_on_medication].nc_diabetes_ever_diagnosed.all()
    assert df.loc[df.is_alive & df.nc_hypertension_on_medication].nc_hypertension_ever_diagnosed.all()
    assert df.loc[
        df.is_alive & df.nc_chronic_lower_back_pain_on_medication].nc_chronic_lower_back_pain_ever_diagnosed.all()
    assert df.loc[
        df.is_alive & df.nc_chronic_kidney_disease_on_medication].nc_chronic_kidney_disease_ever_diagnosed.all()
    assert df.loc[
        df.is_alive & df.nc_chronic_ischemic_hd_on_medication].nc_chronic_ischemic_hd_ever_diagnosed.all()
    assert df.loc[df.is_alive & df.nc_ever_stroke_on_medication].nc_ever_stroke_ever_diagnosed.all()
    assert df.loc[df.is_alive & df.nc_ever_heart_attack_on_medication].nc_ever_heart_attack_ever_diagnosed.all()


def start_sim_and_clear_event_queues(sim):
    """Simulate for 0 days so as to complete all the initialisation steps, but then clear the event queues"""
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    sim.modules['HealthSystem'].reset_queue()
    return sim


def test_if_health_system_cannot_run(seed):
    # Make the health-system unavailable to run any HSI event and check events to make sure no one initiates or
    # continues treatment

    start_date = Date(2010, 1, 1)
    popsize = 1000
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=2),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath)
                 )

    # Make the population
    sim.make_initial_population(n=popsize)

    # Get the simulation running and clear the event queues:
    sim = start_sim_and_clear_event_queues(sim)

    # person_id=0:  Adult woman with diabetes, diagnosed and ready to start medication
    df = sim.population.props
    df.at[0, "sex"] = "F"
    df.at[0, "age_years"] = 60
    df.at[0, "nc_diabetes"] = True
    df.at[0, "nc_diabetes_on_medication"] = False
    df.at[0, "nc_diabetes_ever_diagnosed"] = True

    # person_id=1:  Adult woman with diabetes, diagnosed and already on medication
    df.at[1, "sex"] = "F"
    df.at[1, "age_years"] = 60
    df.at[1, "nc_diabetes"] = True
    df.at[1, "nc_diabetes_on_medication"] = True
    df.at[1, "nc_diabetes_ever_diagnosed"] = True

    # schedule each person  a treatment
    sim.modules['HealthSystem'].schedule_hsi_event(
        HSI_CardioMetabolicDisorders_StartWeightLossAndMedication(person_id=0, module=sim.modules[
            'CardioMetabolicDisorders'], condition='diabetes'),
        topen=sim.date,
        tclose=sim.date + pd.DateOffset(days=1),
        priority=0
    )
    sim.modules['HealthSystem'].schedule_hsi_event(
        HSI_CardioMetabolicDisorders_Refill_Medication(person_id=1, module=sim.modules['CardioMetabolicDisorders'],
                                                       condition='diabetes'),
        topen=sim.date,
        tclose=sim.date + pd.DateOffset(days=1),
        priority=0
    )

    # Run the HealthSystemScheduler for the days (the HSI should not be run and the never_run function should be called)
    hss = HealthSystemScheduler(module=sim.modules['HealthSystem'])
    for i in range(3):
        sim.date = sim.date + pd.DateOffset(days=i)
        hss.apply(sim.population)

    # check that neither person is on medication
    assert not df.at[0, "nc_diabetes_on_medication"]
    assert not df.at[1, "nc_diabetes_on_medication"]


# helper function to run the sim with the healthcare system disabled
def make_simulation_health_system_disabled(seed):
    """Make the simulation with the healthcare system disabled
    """
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable_and_reject_all=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath)
                 )
    return sim


# helper function to run the sim with the healthcare system disabled
def make_simulation_health_system_functional(seed, cons_availability='all'):
    """Make the simulation with the healthcare system enabled and no cons constraints
    """
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=False,
                                           mode_appt_constraints=0,
                                           cons_availability=cons_availability
                                           ),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath)
                 )
    return sim


@pytest.mark.slow
def test_if_no_health_system_and_zero_death(seed):
    """"
    Make the health-system unavailable to run any HSI event and set death rate to zero to check that no one dies
    """

    # Make a list of all conditions and events to run this test for
    condition_list = ['diabetes', 'chronic_kidney_disease', 'chronic_ischemic_hd']

    for condition in condition_list:
        # Disable the healthcare system
        sim = make_simulation_health_system_disabled(seed)
        # make initial population
        sim.make_initial_population(n=2000)
        # force all individuals to have condition
        sim.population.props.loc[
            sim.population.props.is_alive & (sim.population.props.age_years >= 20), f"nc_{condition}"] = True

        p = sim.modules['CardioMetabolicDisorders'].parameters

        # set annual probability of death from condition to zero
        p[f'{condition}_death']["baseline_annual_probability"] = 0

        # simulate for one year
        sim.simulate(end_date=Date(year=2011, month=1, day=1))

        # check that no one died of condition
        df = sim.population.props

        assert not (df.loc[~df.is_alive & ~df.date_of_birth.isna(), 'cause_of_death'] == f'{condition}').any()


@pytest.mark.slow
def test_if_no_health_system_and_high_risk_of_death(seed):
    """"
    Make the health-system unavailable to run any HSI event and set death rate to 100% to check that everyone dies
    """

    # Make a list of all conditions and events to run this test for
    condition_list = ['diabetes', 'chronic_kidney_disease', 'chronic_ischemic_hd']

    for condition in condition_list:
        # Disable the healthcare system
        sim = make_simulation_health_system_disabled(seed)
        # make initial population
        sim.make_initial_population(n=50)
        # force all individuals to have condition
        sim.population.props.loc[
            sim.population.props.is_alive & (sim.population.props.age_years >= 20), f"nc_{condition}"] = True

        p = sim.modules['CardioMetabolicDisorders'].parameters

        # set annual probability of death from condition to 1
        p[f'{condition}_death']["baseline_annual_probability"] = float('inf')
        p[f'{condition}_death']["rr_20_24"] = 1
        p[f'{condition}_death']["rr_25_29"] = 1
        p[f'{condition}_death']["rr_30_34"] = 1
        p[f'{condition}_death']["rr_35_39"] = 1
        p[f'{condition}_death']["rr_40_44"] = 1
        p[f'{condition}_death']["rr_45_49"] = 1
        p[f'{condition}_death']["rr_50_54"] = 1
        p[f'{condition}_death']["rr_55_59"] = 1
        p[f'{condition}_onset']["baseline_annual_probability"] = 0
        p[f'{condition}_hsi']["pr_treatment_works"] = 0

        # simulate for one year
        sim.simulate(end_date=Date(year=2011, month=1, day=1))

        # check that everyone one died
        df = sim.population.props

        assert not (df.loc[~df.date_of_birth.isna() & df[f'nc_{condition}'] & (df.age_years >= 20), 'is_alive']).any()

    event_list = ['ever_stroke', 'ever_heart_attack']

    for event in event_list:
        # Disable the healthcare system
        sim = make_simulation_health_system_disabled(seed)
        # make initial population
        sim.make_initial_population(n=50)

        p = sim.modules['CardioMetabolicDisorders'].parameters

        # increase annual probability of onset and death (values > 1 to ensure probability is still greater than 1.0
        #  after accounting for protective characteristics).
        p[f'{event}_onset']["baseline_annual_probability"] = 10_000.0
        p[f'{event}_death']["baseline_annual_probability"] = 10_000.0

        # simulate for one year
        sim.simulate(end_date=Date(year=2011, month=1, day=1))

        # check that no one died of condition
        df = sim.population.props

        assert not (
            df.loc[~df.date_of_birth.isna() & pd.isnull(df[f'nc_{event}']) & (df.age_years >= 20), 'is_alive']).any()


@pytest.mark.slow
def test_if_medication_prevents_all_death(seed):
    """"
    Make medication 100% effective to check that no one dies
    """

    # Make a list of all conditions and events to run this test for
    condition_list = ['diabetes', 'chronic_kidney_disease', 'chronic_ischemic_hd']
    for condition in condition_list:
        sim = make_simulation_health_system_functional(seed=seed, cons_availability='all')
        sim.make_initial_population(n=50)

        # force all individuals to have condition and be on medication
        sim.population.props.loc[
            sim.population.props.is_alive & (sim.population.props.age_years >= 20), f"nc_{condition}"] = True
        sim.population.props.loc[
            sim.population.props.is_alive & (sim.population.props.age_years >= 20),
            f"nc_{condition}_on_medication"] = True
        sim.population.props.loc[
            sim.population.props.is_alive & (sim.population.props.age_years >= 20),
            f"nc_{condition}_medication_prevents_death"] = True

        # set probability of treatment working to 1 and increase annual risk of death
        p = sim.modules['CardioMetabolicDisorders'].parameters
        p[f'{condition}_hsi']["pr_treatment_works"] = 1
        p[f'{condition}_death']["baseline_annual_probability"] = float('inf')

        sim.simulate(end_date=Date(year=2011, month=1, day=1))

        # check that no one died of condition while on medication
        df = sim.population.props

        assert not (df.loc[~df.is_alive & ~df.date_of_birth.isna(), 'cause_of_death'] == f'{condition}').any()

    event_list = ['ever_stroke', 'ever_heart_attack']

    for event in event_list:
        # Create the sim with an enabled healthcare system
        sim = make_simulation_health_system_functional(seed=seed, cons_availability='all')
        # make initial population
        sim.make_initial_population(n=50)

        p = sim.modules['CardioMetabolicDisorders'].parameters

        # increase annual probability of onset & probability of death ((values > 1 to ensure probability is still
        #  greater than 1.0 after accounting for protective characteristics).
        p[f'{event}_onset']["baseline_annual_probability"] = 10_000.0
        p[f'{event}_death']["baseline_annual_probability"] = 10_000.0

        # set probability of treatment working to 1
        p[f'{event}_hsi']["pr_treatment_works"] = 1

        # simulate for one year
        sim.simulate(end_date=Date(year=2011, month=1, day=1))

        # check that no one died of event
        df = sim.population.props
        assert not (df.loc[~df.is_alive & ~df.date_of_birth.isna(), 'cause_of_death'] == f'{event}').any()


@pytest.mark.slow
def test_symptoms(seed):
    """"
    Test that if symptoms are onset with 100% probability, all persons with condition have symptoms
    """

    # Create and run a short simulation for use in the tests
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=seed)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath)
                 )
    p = sim.modules['CardioMetabolicDisorders'].parameters

    # Set incidence of hypertension very high and incidence of all other conditions to 0, set initial prevalence of
    # other conditions to 0
    p['diabetes_onset']["baseline_annual_probability"] = 10_000.0
    p['diabetes_symptoms']['diabetes_symptoms'] = 10_000.0
    p['chronic_ischemic_hd_onset']["baseline_annual_probability"] = 10_000.0
    p['chronic_ischemic_hd_symptoms']['chronic_ischemic_hd_symptoms'] = 10_000.0
    p['chronic_lower_back_pain_onset']["baseline_annual_probability"] = 10_000.0
    p['chronic_lower_back_pain_symptoms']['chronic_lower_back_pain_symptoms'] = 10_000.0
    p['chronic_kidney_disease_onset']["baseline_annual_probability"] = 10_000.0
    p['chronic_kidney_disease_symptoms']['chronic_kidney_disease_symptoms'] = 10_000.0

    sim.make_initial_population(n=100)
    sim.simulate(end_date=Date(year=2011, month=1, day=1))

    df = sim.population.props
    df = df[df.is_alive]
    who_has_diabetes = df[df['nc_diabetes']].index.to_list()
    who_has_diabetes_symptoms = sim.modules['SymptomManager'].who_has('diabetes_symptoms')
    who_has_chronic_ischemic_hd = df[df['nc_chronic_ischemic_hd']].index.to_list()
    who_has_chronic_ischemic_hd_symptoms = sim.modules['SymptomManager'].who_has('chronic_ischemic_hd_symptoms')
    who_has_chronic_lower_back_pain = df[df['nc_chronic_lower_back_pain']].index.to_list()
    who_has_chronic_lower_back_pain_symptoms = sim.modules['SymptomManager'].who_has('chronic_lower_back_pain_symptoms')
    who_has_chronic_kidney_disease = df[df['nc_chronic_kidney_disease']].index.to_list()
    who_has_chronic_kidney_disease_symptoms = sim.modules['SymptomManager'].who_has('chronic_kidney_disease_symptoms')

    assert who_has_diabetes == who_has_diabetes_symptoms
    assert who_has_chronic_ischemic_hd == who_has_chronic_ischemic_hd_symptoms
    assert who_has_chronic_lower_back_pain == who_has_chronic_lower_back_pain_symptoms
    assert who_has_chronic_kidney_disease == who_has_chronic_kidney_disease_symptoms


def test_hsi_investigation_not_following_symptoms(seed):
    """Create a person and check if the functions in HSI_CardioMetabolicDisorders_InvestigationNotFollowingSymptoms
    create the correct HSI"""

    # Make a list of all conditions and events to run this test for
    condition_list = ['diabetes', 'hypertension', 'chronic_lower_back_pain', 'chronic_kidney_disease',
                      'chronic_ischemic_hd']
    for condition in condition_list:
        # Create the sim with an enabled healthcare system
        sim = make_simulation_health_system_functional(seed=seed)

        # make initial population
        sim.make_initial_population(n=50)

        # simulate for zero days
        sim = start_sim_and_clear_event_queues(sim)

        df = sim.population.props

        # Get target person and make them have condition but not be diagnosed yet, and high enough BMI for them to
        # receive weight loss treatment
        person_id = 0
        df.at[person_id, f"nc_{condition}"] = True
        df.at[person_id, f"nc_{condition}_ever_diagnosed"] = False
        df.at[person_id, f"nc_{condition}_on_medication"] = False
        df.at[person_id, 'li_bmi'] = 4

        # Run the InvestigationNotFollowingSymptoms event
        t = HSI_CardioMetabolicDisorders_InvestigationNotFollowingSymptoms(module=sim.modules[
            'CardioMetabolicDisorders'], person_id=person_id, condition=f'{condition}')
        t.apply(person_id=person_id, squeeze_factor=0.0)

        # Check that there is StartWeightLossAndMedication event scheduled
        date_event, event = [
            ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
            isinstance(ev[1], cardio_metabolic_disorders.HSI_CardioMetabolicDisorders_StartWeightLossAndMedication)
        ][0]

        # Run the event:
        event.apply(person_id=person_id, squeeze_factor=0.0)

        # Check that the person has now experienced weight loss treatment (except for CKD)
        if condition != 'chronic_kidney_disease':
            assert df.at[person_id, "nc_ever_weight_loss_treatment"]


def test_hsi_investigation_following_symptoms(seed):
    """Create a person and check if the functions in HSI_CardioMetabolicDisorders_InvestigationFollowingSymptoms
    create the correct HSI"""

    # Make a list of all conditions and events to run this test for
    condition_list = ['diabetes', 'chronic_lower_back_pain', 'chronic_kidney_disease', 'chronic_ischemic_hd']
    for condition in condition_list:
        # Create the sim with an enabled healthcare system
        sim = make_simulation_health_system_functional(seed=seed)

        # make initial population
        sim.make_initial_population(n=50)

        # simulate for zero days
        sim = start_sim_and_clear_event_queues(sim)

        df = sim.population.props

        # Get target person and make them have condition, be symptomatic, but not be diagnosed yet, and high enough
        # BMI for them to receive weight loss treatment
        person_id = 0
        df.at[person_id, f"nc_{condition}"] = True
        df.at[person_id, f"nc_{condition}_ever_diagnosed"] = False
        df.at[person_id, f"nc_{condition}_on_medication"] = False
        df.at[person_id, 'li_bmi'] = 4

        sim.modules['SymptomManager'].change_symptom(
            person_id=person_id,
            symptom_string=f'{condition}_symptoms',
            disease_module=sim.modules['CardioMetabolicDisorders'],
            add_or_remove='+'
        )

        # Run the InvestigationNotFollowingSymptoms event
        t = HSI_CardioMetabolicDisorders_InvestigationFollowingSymptoms(module=sim.modules[
            'CardioMetabolicDisorders'], person_id=person_id, condition=f'{condition}')
        t.apply(person_id=person_id, squeeze_factor=0.0)

        # Check that there is StartWeightLossAndMedication event scheduled
        date_event, event = [
            ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
            isinstance(ev[1], cardio_metabolic_disorders.HSI_CardioMetabolicDisorders_StartWeightLossAndMedication)
        ][0]

        # Run the event:
        event.apply(person_id=person_id, squeeze_factor=0.0)

        # Check that the person has now experienced weight loss treatment (except for CKD)
        if condition != 'chronic_kidney_disease':
            assert df.at[person_id, "nc_ever_weight_loss_treatment"]


def test_hsi_weight_loss_and_medication(seed):
    """Create a person and check if the functions in HSI_CardioMetabolicDisorders_StartWeightLossAndMedication
    create the correct HSI"""

    # Make a list of all conditions and events to run this test for
    condition_list = ['diabetes', 'chronic_lower_back_pain', 'chronic_kidney_disease', 'chronic_ischemic_hd']
    for condition in condition_list:
        # Create the sim with an enabled healthcare system
        sim = make_simulation_health_system_functional(seed=seed)

        # make initial population
        sim.make_initial_population(n=50)

        # set probability of weight loss working to 1
        p = sim.modules['CardioMetabolicDisorders'].parameters
        p["pr_bmi_reduction"] = 1

        # simulate for zero days
        sim = start_sim_and_clear_event_queues(sim)

        df = sim.population.props

        # Get target person and make them have condition and diagnosed but not on medication yet
        person_id = 0
        df.at[person_id, f"nc_{condition}"] = True
        df.at[person_id, f"nc_{condition}_ever_diagnosed"] = True
        df.at[person_id, f"nc_{condition}_on_medication"] = False
        df.at[person_id, "li_bmi"] = 4

        # Run the StartWeightLossAndMedication event
        t = HSI_CardioMetabolicDisorders_StartWeightLossAndMedication(module=sim.modules[
            'CardioMetabolicDisorders'], person_id=person_id, condition=f'{condition}')
        t.apply(person_id=person_id, squeeze_factor=0.0)

        # Check that the individual has received weight loss treatment, is now on medication, and that there is a
        # Refill_Medication event scheduled

        assert df.at[person_id, f"nc_{condition}_on_medication"]
        assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[0].hsi_event,
                          HSI_CardioMetabolicDisorders_Refill_Medication)

        if condition != 'chronic_kidney_disease':  # those with CKD are not recommended to lose weight
            # Check that the individual has received weight loss treatment
            assert df.at[person_id, 'nc_ever_weight_loss_treatment']
            # Check that the individual has a CardioMetabolicDisordersWeightLossEvent scheduled
            events_for_this_person = sim.find_events_for_person(person_id)
            assert 1 == len(events_for_this_person)
            next_event_date, next_event_obj = events_for_this_person[0]
            assert isinstance(next_event_obj, cardio_metabolic_disorders.CardioMetabolicDisordersWeightLossEvent)
            assert next_event_date >= sim.date

            # Run the WeightLossEvent
            t = CardioMetabolicDisordersWeightLossEvent(module=sim.modules['CardioMetabolicDisorders'],
                                                        person_id=person_id)
            t.apply(person_id=person_id)

            # Check that individual's BMI has reduced by 1 and they are flagged as having experienced weight loss
            assert df.at[person_id, "li_bmi"] == 3
            assert df.at[person_id, "nc_weight_loss_worked"]


def test_hsi_emergency_events(seed):
    """Create a person and check if the functions in HSI_CardioMetabolicDisorders_SeeksEmergencyCareAndGetsTreatment
    result in the correct order of events and create the correct HSI"""

    # Make a list of all events to run this test for
    event_list = ['ever_stroke', 'ever_heart_attack']
    for event in event_list:
        # Create the sim with an enabled healthcare system
        sim = make_simulation_health_system_functional(seed=seed)

        # make initial population
        sim.make_initial_population(n=50)

        # change treatment parameter to always work
        p = sim.modules['CardioMetabolicDisorders'].parameters
        p[f'{event}_hsi']["pr_treatment_works"] = 1

        # simulate for zero days
        sim = start_sim_and_clear_event_queues(sim)

        df = sim.population.props

        # Get target person and make them have condition and diagnosed but not on medication yet
        person_id = 0
        df.at[person_id, f"nc_{event}"] = True
        df.at[person_id, f"nc_{event}_ever_diagnosed"] = True
        df.at[person_id, f"nc_{event}_on_medication"] = False
        df.at[person_id, f'nc_{event}_scheduled_date_death'] = sim.date + pd.DateOffset(days=7)

        # Make them have symptoms of event
        sim.modules['SymptomManager'].change_symptom(
            person_id=person_id,
            symptom_string=f'{event}_damage',
            disease_module=sim.modules['CardioMetabolicDisorders'],
            add_or_remove='+'
        )

        # Run the SeeksEmergencyCareAndGetsTreatment event
        t = HSI_CardioMetabolicDisorders_SeeksEmergencyCareAndGetsTreatment(module=sim.modules[
            'CardioMetabolicDisorders'], person_id=person_id, ev=f'{event}')
        t.apply(person_id=person_id, squeeze_factor=0.0)

        # Check that the individual is now diagnosed, on medication, there is a StartMedication event scheduled, and
        # symptoms have been removed
        assert df.at[person_id, f'nc_{event}_date_diagnosis'] == sim.date
        assert df.at[person_id, f'nc_{event}_on_medication']
        assert pd.isnull(df.at[person_id, f'nc_{event}_scheduled_date_death'])
        assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[0].hsi_event,
                          HSI_CardioMetabolicDisorders_StartWeightLossAndMedication)
        assert f"{event}_damage" not in sim.modules['SymptomManager'].has_what(person_id)


def test_no_availability_of_consumables_for_conditions(seed):
    """Check if consumables aren't available that everyone drops off of treatment"""

    # Make a list of all conditions and events to run this test for
    condition_list = ['diabetes', 'chronic_lower_back_pain', 'chronic_kidney_disease', 'chronic_ischemic_hd']
    for condition in condition_list:
        # Create the sim with an enabled healthcare system but no consumables
        sim = make_simulation_health_system_functional(seed=seed, cons_availability='none')

        # make initial population
        sim.make_initial_population(n=50)

        # simulate for zero days
        sim = start_sim_and_clear_event_queues(sim)

        df = sim.population.props

        # Get target person and make them have condition, diagnosed and on medication
        person_id = 0
        df.at[person_id, f"nc_{condition}"] = True
        df.at[person_id, f"nc_{condition}_ever_diagnosed"] = True
        df.at[person_id, f"nc_{condition}_on_medication"] = True

        # Run the Refill_Medication event
        t = HSI_CardioMetabolicDisorders_Refill_Medication(module=sim.modules['CardioMetabolicDisorders'],
                                                           person_id=person_id, condition=f'{condition}')
        t.apply(person_id=person_id, squeeze_factor=0.0)

        # Check that the individual has dropped off of medication due to lack of consumables
        assert not df.at[person_id, f"nc_{condition}_on_medication"]


def test_no_availability_of_consumables_for_events(seed):
    """Check if consumables aren't available that HSI events are commissioned but individual dies of event anyway"""

    # Make a list of all events to run this test for
    event_list = ['ever_stroke', 'ever_heart_attack']
    for event in event_list:
        # Create the sim with an enabled healthcare system but no consumables
        sim = make_simulation_health_system_functional(seed=seed, cons_availability='none')

        # Make probability of death 100%
        p = sim.modules['CardioMetabolicDisorders'].parameters
        p[f'{event}_death']["baseline_annual_probability"] = 10_000.0
        # (Use very high value to ensure that risk will be >1 for all individuals (this is the intercept term to a
        # linear model).

        # make initial population
        sim.make_initial_population(n=100)

        # simulate for zero days
        sim = start_sim_and_clear_event_queues(sim)

        df = sim.population.props

        # Get target person and make them have condition, diagnosed and on medication
        person_id = 0
        df.at[person_id, f"nc_{event}"] = False
        df.at[person_id, f"nc_{event}_ever_diagnosed"] = False
        df.at[person_id, f"nc_{event}_on_medication"] = False

        # Set age and BMI of target to ensure high probability of death
        df.at[person_id, "date_of_birth"] = sim.date - pd.DateOffset(years=70)
        df.at[person_id, "age_years"] = 70
        df.at[person_id, "li_bmi"] = 4

        # Run the CardioMetabolicDisordersEvent event
        t = CardioMetabolicDisordersEvent(module=sim.modules['CardioMetabolicDisorders'],
                                          person_id=person_id, event=event)
        t.apply(person_id=person_id)

        events_for_this_person = sim.find_events_for_person(person_id)
        assert 1 == len(events_for_this_person)
        next_event_date, next_event_obj = events_for_this_person[0]
        assert isinstance(next_event_obj, cardio_metabolic_disorders.CardioMetabolicDisordersDeathEvent)
        assert next_event_date >= sim.date

        # Run the SeeksEmergencyCareAndGetsTreatment event on this person
        t = HSI_CardioMetabolicDisorders_SeeksEmergencyCareAndGetsTreatment(module=sim.modules[
            'CardioMetabolicDisorders'], person_id=person_id, ev=f'{event}')
        t.apply(person_id=person_id, squeeze_factor=0.0)

        # Check that the individual is not on medication due to lack of consumables, and that there is a scheduled date
        # of death still and DeathEvent  is still in queue
        assert not df.at[person_id, f'nc_{event}_on_medication']
        assert df.at[person_id, f'nc_{event}_scheduled_date_death'] == next_event_date

        # change date of death to today's date to run DeathEvent
        df.at[person_id, f'nc_{event}_scheduled_date_death'] = sim.date

        # Run the DeathEvent on this person
        t = CardioMetabolicDisordersDeathEvent(module=sim.modules['CardioMetabolicDisorders'], person_id=person_id,
                                               originating_cause=f'{event}')
        t.apply(person_id=person_id)

        assert not df.at[person_id, 'is_alive']
        assert df.at[person_id, 'cause_of_death'] == f'{event}'


def test_logging_works_for_person_older_than_100(seed):
    """Check that no error is caused when someone older than 100 years is onset with a prevalent condition. (This has
    previously caused an error.) """

    sim = make_simulation_health_system_functional(seed=seed)
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.start_date)

    cdm = sim.modules['CardioMetabolicDisorders']
    age_update_event = AgeUpdateEvent(
        module=sim.modules['Demography'],
        age_range_lookup=sim.modules['Demography'].AGE_RANGE_LOOKUP
    )

    event = cdm.events[0]
    person_id = 0

    for age in (80.0, 90.0, 100.0, 110.0, 120.0, 121.0):

        # Make one person that age
        df = sim.population.props
        df.at[person_id, 'date_of_birth'] = sim.date - pd.DateOffset(days=age * 365)
        age_update_event.run()

        # Call the tracker
        cdm.trackers['prevalent_event'].add(event, {df.at[person_id, 'age_range']: 1})
