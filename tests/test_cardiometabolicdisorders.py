import os
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.methods import (
    cardio_metabolic_disorders,
    demography,
    depression,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)
from tlo.methods.cardio_metabolic_disorders import (
    HSI_CardioMetabolicDisorders_InvestigationNotFollowingSymptoms,
    HSI_CardioMetabolicDisorders_InvestigationFollowingSymptoms,
    HSI_CardioMetabolicDisorders_StartWeightLossAndMedication,
    HSI_CardioMetabolicDisorders_Refill_Medication,
    HSI_CardioMetabolicDisorders_SeeksEmergencyCareAndGetsTreatment
)
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


def test_basic_run():
    # --------------------------------------------------------------------------
    # Create and run a short but big population simulation for use in the tests
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
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
    p['ever_stroke_onset']["baseline_annual_probability"] = 0.75
    p['ever_heart_attack_onset']["baseline_annual_probability"] = 0.75
    p['diabetes_death']["baseline_annual_probability"] = 0.75
    p['chronic_kidney_disease_death']["baseline_annual_probability"] = 0.75
    p['ever_stroke_death']["baseline_annual_probability"] = 0.75
    p['ever_heart_attack_death']["baseline_annual_probability"] = 0.75

    sim.make_initial_population(n=5000)
    sim.simulate(end_date=Date(year=2011, month=1, day=1))

    routine_checks(sim)
    hsi_checks(sim)


def test_basic_run_with_high_incidence_hypertension():
    """This sim makes one condition very common and the others non-existent to check basic functions for prevalence and
    death"""

    # Create and run a short but big population simulation for use in the tests
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
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

    p['hypertension_onset']["baseline_annual_probability"] = 10000
    p['chronic_ischemic_hd_onset']["baseline_annual_probability"] = 10
    p['diabetes_onset'] = p['diabetes_onset'].mask(p['diabetes_onset'] > 0, 0)
    p['chronic_lower_back_pain_onset'] = p['chronic_lower_back_pain_onset'].mask(p['chronic_lower_back_pain_onset'] > 0,
                                                                                 0)
    p['chronic_kidney_disease_onset'] = p['chronic_kidney_disease_onset'].mask(p['chronic_kidney_disease_onset'] > 0, 0)
    p['diabetes_initial_prev'] = p['diabetes_initial_prev'].mask(p['diabetes_initial_prev'] > 0, 0)
    p['chronic_lower_back_pain_initial_prev'] = p['chronic_lower_back_pain_initial_prev']. \
        mask(p['chronic_lower_back_pain_initial_prev'] > 0, 0)
    p['chronic_kidney_disease_initial_prev'] = p['chronic_kidney_disease_initial_prev']. \
        mask(p['chronic_kidney_disease_initial_prev'] > 0, 0)

    # Increase RR of heart disease very high if individual has hypertension
    p['chronic_ischemic_hd_onset']["rr_hypertension"] = 1000

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
    df['diff_years'] = df.start_date - df.date_of_birth
    df['diff_years'] = df.diff_years / np.timedelta64(1, 'Y')
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

    assert ~pd.isnull(df.loc[df.nc_diabetes_ever_diagnosed, 'nc_diabetes_date_last_test']).all()
    assert ~pd.isnull(df.loc[df.nc_hypertension_ever_diagnosed, 'nc_hypertension_date_last_test']).all()
    assert ~pd.isnull(
        df.loc[df.nc_chronic_lower_back_pain_ever_diagnosed, 'nc_chronic_lower_back_pain_date_last_test']).all()
    assert ~pd.isnull(df.loc[df.nc_chronic_kidney_disease_ever_diagnosed,
                             'nc_chronic_kidney_disease_date_last_test']).all()
    assert ~pd.isnull(df.loc[df.nc_chronic_ischemic_hd_ever_diagnosed, 'nc_chronic_ischemic_hd_date_last_test']).all()

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
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()
    return sim


def test_if_health_system_cannot_run():
    # Make the health-system unavailable to run any HSI event and check events to make sure no one initiates or
    # continues treatment

    start_date = Date(2010, 1, 1)
    popsize = 1000
    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           capabilities_coefficient=0.0,
                                           mode_appt_constraints=2),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath)
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
def make_simulation_health_system_disabled():
    """Make the simulation with:
    * the demography module with the OtherDeathsPoll not running
    """
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
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
def make_simulation_health_system_functional():
    """Make the simulation with:
    * the demography module with the OtherDeathsPoll not running
    """
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=False,
                                           ignore_cons_constraints=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath)
                 )
    return sim


def test_if_no_health_system_and_zero_death():
    """"
    Make the health-system unavailable to run any HSI event and set death rate to zero to check that no one dies
    """

    # Make a list of all conditions and events to run this test for
    condition_list = ['diabetes', 'chronic_kidney_disease', 'chronic_ischemic_hd']

    for condition in condition_list:
        # Disable the healthcare system
        sim = make_simulation_health_system_disabled()
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


def test_if_no_health_system_and_hundred_death():
    """"
    Make the health-system unavailable to run any HSI event and set death rate to 100% to check that everyone dies
    """

    # Make a list of all conditions and events to run this test for
    condition_list = ['diabetes', 'chronic_kidney_disease', 'chronic_ischemic_hd']

    for condition in condition_list:
        # Disable the healthcare system
        sim = make_simulation_health_system_disabled()
        # make initial population
        sim.make_initial_population(n=50)
        # force all individuals to have condition
        sim.population.props.loc[
            sim.population.props.is_alive & (sim.population.props.age_years >= 20), f"nc_{condition}"] = True

        p = sim.modules['CardioMetabolicDisorders'].parameters

        # set annual probability of death from condition to 1
        p[f'{condition}_death']["baseline_annual_probability"] = 10000
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
        sim = make_simulation_health_system_disabled()
        # make initial population
        sim.make_initial_population(n=50)

        p = sim.modules['CardioMetabolicDisorders'].parameters

        # increase annual probability of onset and death
        p[f'{event}_onset']["baseline_annual_probability"] = 10000
        p[f'{event}_death']["baseline_annual_probability"] = 10000

        # simulate for one year
        sim.simulate(end_date=Date(year=2011, month=1, day=1))

        # check that no one died of condition
        df = sim.population.props

        assert not (df.loc[~df.date_of_birth.isna() & df[f'nc_{event}'] & (df.age_years >= 20), 'is_alive']).any()


def test_if_medication_prevents_all_death():
    """"
    Make medication 100% effective to check that no one dies
    """

    # Make a list of all conditions and events to run this test for
    condition_list = ['diabetes', 'chronic_kidney_disease', 'chronic_ischemic_hd']
    for condition in condition_list:
        sim = make_simulation_health_system_functional()
        sim.make_initial_population(n=50)

        # force all individuals to have condition and be on medication
        sim.population.props.loc[
            sim.population.props.is_alive & (sim.population.props.age_years >= 20), f"nc_{condition}"] = True
        sim.population.props.loc[
            sim.population.props.is_alive & (sim.population.props.age_years >= 20),
            f"nc_{condition}_on_medication"] = True

        # set probability of treatment working to 1 and increase annual risk of death
        p = sim.modules['CardioMetabolicDisorders'].parameters
        p[f'{condition}_hsi']["pr_treatment_works"] = 1
        p[f'{condition}_death']["baseline_annual_probability"] = 10000

        sim.simulate(end_date=Date(year=2011, month=1, day=1))

        # check that no one died of condition while on medication
        df = sim.population.props

        assert not (df.loc[~df.is_alive & ~df.date_of_birth.isna(), 'cause_of_death'] == f'{condition}').any()

    event_list = ['ever_stroke', 'ever_heart_attack']

    for event in event_list:
        # Create the sim with an enabled healthcare system
        sim = make_simulation_health_system_functional()
        # make initial population
        sim.make_initial_population(n=50)

        p = sim.modules['CardioMetabolicDisorders'].parameters

        # increase annual probability of onset & probability of death
        p[f'{event}_onset']["baseline_annual_probability"] = 10000
        p[f'{event}_death']["baseline_annual_probability"] = 10000

        # set probability of treatment working to 1
        p[f'{event}_hsi']["pr_treatment_works"] = 1

        # simulate for one year
        sim.simulate(end_date=Date(year=2011, month=1, day=1))

        # check that no one died of event
        df = sim.population.props
        assert not (df.loc[~df.is_alive & ~df.date_of_birth.isna(), 'cause_of_death'] == f'{event}').any()


def test_hsi_investigation_not_following_symptoms():
    """Create a person and check if the functions in HSI_CardioMetabolicDisorders_InvestigationNotFollowingSymptoms
    create the correct HSI"""

    # Make a list of all conditions and events to run this test for
    condition_list = ['diabetes', 'hypertension', 'chronic_lower_back_pain', 'chronic_kidney_disease',
                      'chronic_ischemic_hd']
    for condition in condition_list:
        # Create the sim with an enabled healthcare system
        sim = make_simulation_health_system_functional()

        # make initial population
        sim.make_initial_population(n=50)

        # simulate for zero days
        sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
        sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
        sim.event_queue.queue.clear()

        df = sim.population.props

        # Get target person and make them have condition but not be diagnosed yet
        person_id = 0
        df.at[person_id, f"nc_{condition}"] = True
        df.at[person_id, f"nc_{condition}_ever_diagnosed"] = False
        df.at[person_id, f"nc_{condition}_on_medication"] = False

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


def test_hsi_investigation_following_symptoms():
    """Create a person and check if the functions in HSI_CardioMetabolicDisorders_InvestigationFollowingSymptoms
    create the correct HSI"""

    # Make a list of all conditions and events to run this test for
    condition_list = ['diabetes', 'chronic_lower_back_pain', 'chronic_kidney_disease', 'chronic_ischemic_hd']
    for condition in condition_list:
        # Create the sim with an enabled healthcare system
        sim = make_simulation_health_system_functional()

        # make initial population
        sim.make_initial_population(n=50)

        # simulate for zero days
        sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
        sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
        sim.event_queue.queue.clear()

        df = sim.population.props

        # Get target person and make them have condition, be symptomatic, but not be diagnosed yet
        person_id = 0
        df.at[person_id, f"nc_{condition}"] = True
        df.at[person_id, f"nc_{condition}_ever_diagnosed"] = False
        df.at[person_id, f"nc_{condition}_on_medication"] = False

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


def test_hsi_weight_loss_and_medication():
    """Create a person and check if the functions in HSI_CardioMetabolicDisorders_StartWeightLossAndMedication
    create the correct HSI"""

    # Make a list of all conditions and events to run this test for
    condition_list = ['diabetes', 'chronic_lower_back_pain', 'chronic_kidney_disease', 'chronic_ischemic_hd']
    for condition in condition_list:
        # Create the sim with an enabled healthcare system
        sim = make_simulation_health_system_functional()

        # make initial population
        sim.make_initial_population(n=50)

        # simulate for zero days
        sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
        sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
        sim.event_queue.queue.clear()

        df = sim.population.props

        # Get target person and make them have condition and diagnosed but not on medication yet
        person_id = 0
        df.at[person_id, f"nc_{condition}"] = True
        df.at[person_id, f"nc_{condition}_ever_diagnosed"] = True
        df.at[person_id, f"nc_{condition}_on_medication"] = False

        # Run the StartWeightLossAndMedication event
        t = HSI_CardioMetabolicDisorders_StartWeightLossAndMedication(module=sim.modules[
            'CardioMetabolicDisorders'], person_id=person_id, condition=f'{condition}')
        t.apply(person_id=person_id, squeeze_factor=0.0)

        # Check that the individual is now on medication and that there is Refill_Medication event scheduled
        assert df.at[person_id, f"nc_{condition}_on_medication"]
        assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[0][4],
                          HSI_CardioMetabolicDisorders_Refill_Medication)


def test_hsi_emergency_events():
    """Create a person and check if the functions in HSI_CardioMetabolicDisorders_SeeksEmergencyCareAndGetsTreatment
    create the correct HSI"""

    # Make a list of all conditions and events to run this test for
    event_list = ['ever_stroke', 'ever_heart_attack']
    for event in event_list:
        # Create the sim with an enabled healthcare system
        sim = make_simulation_health_system_functional()

        # make initial population
        sim.make_initial_population(n=50)

        # change treatment parameter to always work
        p = sim.modules['CardioMetabolicDisorders'].parameters
        p[f'{event}_hsi']["pr_treatment_works"] = 1

        # simulate for zero days
        sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
        sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
        sim.event_queue.queue.clear()

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
        assert isinstance(sim.modules['HealthSystem'].HSI_EVENT_QUEUE[0][4],
                          HSI_CardioMetabolicDisorders_StartWeightLossAndMedication)
        assert f"{event}_damage" not in sim.modules['SymptomManager'].has_what(person_id)
