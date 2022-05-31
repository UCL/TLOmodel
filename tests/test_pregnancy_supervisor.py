import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.lm import LinearModel, LinearModelType
from tlo.methods import (
    alri,
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    diarrhoea,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_helper_functions,
    pregnancy_supervisor,
    stunting,
    symptommanager,
    tb,
    wasting,
)

# The resource files
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

start_date = Date(2010, 1, 1)


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def set_all_women_as_pregnant_and_reset_baseline_parity(sim):
    """Force all women of reproductive age to be pregnant at the start of the simulation and overrides parity set at
     initialisation of simulation """
    df = sim.population.props

    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    df.loc[women_repro.index, 'is_pregnant'] = True
    df.loc[women_repro.index, 'date_of_last_pregnancy'] = sim.start_date
    for person in women_repro.index:
        sim.modules['Labour'].set_date_of_labour(person)

    all_women = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14)]
    df.loc[all_women.index, 'la_parity'] = 0


def turn_off_antenatal_pregnancy_loss(sim):
    """Set all parameters which output probability of pregnancy loss to 0"""
    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_ectopic_pregnancy'] = 0.0
    params['prob_induced_abortion_per_month'] = 0.0
    params['prob_still_birth_per_month'] = 0.0

    sim.modules['PregnancySupervisor'].ps_linear_models['spontaneous_abortion'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0.0)


def register_modules(sim):
    """Defines sim variable and registers all modules that can be called when running the full suite of pregnancy
    modules"""

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           cons_availability='all'),  # went set disable=true, cant check HSI queue
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),

                 hiv.DummyHivModule(),
                 )


@pytest.mark.slow
def test_run_core_modules_normal_allocation_of_pregnancy(seed, tmpdir):
    """Runs the simulation using only core modules without manipulation of pregnancy rates or parameters and checks
    dtypes at the end"""

    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})

    register_modules(sim)
    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(2015, 2, 1))   # run to ensure ParameterUpdateEvent is called and works
    check_dtypes(sim)

    # check that no errors have been logged during the simulation run
    output = parse_log_file(sim.log_filepath)
    assert 'error' not in output['tlo.methods.pregnancy_supervisor']
    assert 'error' not in output['tlo.methods.care_of_women_during_pregnancy']


@pytest.mark.slow
def test_run_core_modules_high_volumes_of_pregnancy(seed, tmpdir):
    """Runs the simulation with the core modules and all women of reproductive age being pregnant at the start of the
    simulation"""
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})

    register_modules(sim)
    sim.make_initial_population(n=5000)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=Date(2011, 1, 1))
    check_dtypes(sim)

    # check that no errors have been logged during the simulation run
    output = parse_log_file(sim.log_filepath)
    assert 'error' not in output['tlo.methods.pregnancy_supervisor']
    assert 'error' not in output['tlo.methods.care_of_women_during_pregnancy']


@pytest.mark.slow
def test_run_core_modules_high_volumes_of_pregnancy_hsis_cant_run(seed, tmpdir):
    """Runs the simulation with the core modules and all women of reproductive age being pregnant at the start of the
    simulation. In addition scheduled HSI events will not run- testing the did_not_run functions of the relevant HSIs"""
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           mode_appt_constraints=2,
                                           cons_availability='all'),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),

                 hiv.DummyHivModule(),
                 )

    sim.make_initial_population(n=5000)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=Date(2011, 1, 1))
    check_dtypes(sim)

    # check that no errors have been logged during the simulation run
    output = parse_log_file(sim.log_filepath)
    for module in ['pregnancy_supervisor', 'care_of_women_during_pregnancy', 'labour', 'postnatal_supervisor',
                   'newborn_outcomes']:
        assert 'error' not in output[f'tlo.methods.{module}']


@pytest.mark.slow
def test_run_with_all_referenced_modules_registered(seed, tmpdir):
    """
    Runs the simulation for one year where all the referenced modules are registered to ensure
    """
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           cons_availability='all'),  # went set disable=true, cant check HSI queue
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),

                 # Register all the modules that are reference in the maternal perinatal health suite (including their
                 # dependencies)
                 alri.Alri(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath),
                 malaria.Malaria(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 stunting.Stunting(resourcefilepath=resourcefilepath),
                 wasting.Wasting(resourcefilepath=resourcefilepath),
                 diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
                 epi.Epi(resourcefilepath=resourcefilepath),
                 tb.Tb(resourcefilepath=resourcefilepath),
                 )

    sim.make_initial_population(n=5000)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)  # keep high volume of pregnancy to increase risk of error
    sim.simulate(end_date=Date(2011, 1, 1))
    check_dtypes(sim)

    # check that no errors have been logged during the simulation run
    output = parse_log_file(sim.log_filepath)
    assert 'error' not in output['tlo.methods.pregnancy_supervisor']
    assert 'error' not in output['tlo.methods.care_of_women_during_pregnancy']


def test_store_dalys_in_mni_function_and_daly_calculations(seed):
    """This test checks how we calculate, store and report back individuals disability weight for the previous month
    in the model."""

    # Set up sim and run for 0 days
    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)

    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    # Store functions from pregnancy supervisor as variables
    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
    store_dalys_in_mni = pregnancy_helper_functions.store_dalys_in_mni
    params = sim.modules['PregnancySupervisor'].parameters

    # Select pregnant woman from dataframe and Generate the MNI dictionary
    pregnant_women = df.loc[df.is_alive & df.is_pregnant]
    for woman in pregnant_women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

    mother_id = pregnant_women.index[0]

    # First we test the logic for 'acute' complications
    # Call store_dalys_in_mni function which is called when any woman experiences one of the complications stored in
    # the mni
    store_dalys_in_mni(mother_id, mni, 'ectopic_onset', sim.date)
    store_dalys_in_mni(mother_id, mni, 'ectopic_rupture_onset', sim.date)

    # Check that the correct date of onset has been captured in the mni dictionary
    assert mni[mother_id]['ectopic_onset'] == sim.date
    assert mni[mother_id]['ectopic_rupture_onset'] == sim.date

    # Call the report_daly_function which contains the functionality to calculate the daly weight this woman should have
    # following her complications

    dalys_from_pregnancy = sim.modules['PregnancySupervisor'].report_daly_values()
    assert dalys_from_pregnancy.loc[mother_id] > 0

    # We assume any of the 'acute' complications cause women to accrue 7 days of disability weight
    ectopic_preg_weight = params['ps_daly_weights']['ectopic']
    ectopic_rupture_weight = params['ps_daly_weights']['ectopic_rupture']

    # Therefore for this month this woman should have accrued the sum of these two weight values
    # (rounded to allow comparison)
    total_weight = round((ectopic_preg_weight + ectopic_rupture_weight), 4)

    # Now we use the output returned from the report_daly_values function and check that the function has correctly
    # calculated the weight for this month and store that in the output against this mothers index
    rounded_dalys = round(dalys_from_pregnancy.loc[mother_id], 4)
    assert total_weight == rounded_dalys

    # Chronic complications
    # Now we check the logic for complications that acrue daly weights every day
    # Store onset of severe anaemia and check the correct date has been stored
    store_dalys_in_mni(mother_id, mni, 'severe_anaemia_onset', sim.date)
    assert mni[mother_id]['severe_anaemia_onset'] == sim.date

    # Move the date forward 1 month and run the report_daly_values
    sim.date = sim.date + pd.DateOffset(months=1)
    dalys_from_pregnancy = sim.modules['PregnancySupervisor'].report_daly_values()

    # Check a daly weight has been stored for this woman
    assert dalys_from_pregnancy.loc[mother_id] > 0

    # This woman has had this complication for the entire month (01/01/2010 - 01/02/2010) and it has not resolved,
    # therefore we expect her to have accrued 1 months weight
    sev_anemia_weight = round((params['ps_daly_weights']['severe_anaemia'] / 365.25) * (365.25 / 12), 2)
    reported_weight = round(dalys_from_pregnancy.loc[mother_id], 2)
    assert sev_anemia_weight == reported_weight

    # Move the date forward 2 weeks and set the date of resolution for the complication
    sim.date = sim.date + pd.DateOffset(weeks=2)
    store_dalys_in_mni(mother_id, mni, 'severe_anaemia_resolution', sim.date)

    # Move the date forward another 2 weeks and call report_daly_values. This woman will only have experienced the
    # complication for half the month as it was resolved 2 weeks prior to report_daly_values being called
    sim.date = sim.date + pd.DateOffset(weeks=2)
    dalys_from_pregnancy = sim.modules['PregnancySupervisor'].report_daly_values()

    # Ensure some weight was captured
    assert dalys_from_pregnancy.loc[mother_id] > 0
    # Ensure the complication variables are reset within the report_daly_values
    assert pd.isnull(mni[mother_id]['severe_anaemia_onset'])
    assert pd.isnull(mni[mother_id]['severe_anaemia_resolution'])

    # We know she has experience 15 days of complication this month, check the function returns the correct daly weight
    sev_anemia_weight = round((params['ps_daly_weights']['severe_anaemia'] / 365.25) * 15, 3)
    reported_weight = round(dalys_from_pregnancy.loc[mother_id], 3)

    assert sev_anemia_weight == reported_weight


def test_calculation_of_gestational_age(seed):
    """This is a simple test to check that when called, the pregnancy supervisor event updates the age of all women's
    gestational age correctly"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)

    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    turn_off_antenatal_pregnancy_loss(sim)

    # Run the sim for 0 days
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()
    df = sim.population.props

    # Set each pregnant womans conception date as some random date in the past
    pregnant_women = df.loc[df.is_alive & df.is_pregnant]
    for person in pregnant_women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], person)
        random_days = sim.modules['PregnancySupervisor'].rng.randint(1, 274)
        df.at[person, 'date_of_last_pregnancy'] = sim.date - pd.DateOffset(days=random_days)

    # Run the event
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    pregnancy_sup.apply(sim.population)

    # Check some gestational age has been recorded for each woman
    assert (df.loc[df.is_alive & df.is_pregnant, 'ps_gestational_age_in_weeks'] != 0).all().all()

    # Now check that, for each woman, gestational age is correctly calculated as 2 weeks greater than total number of
    # weeks pregnant
    for person in df.loc[df.is_alive & df.is_pregnant].index:
        foetal_age_weeks = np.ceil((sim.date - df.at[person, 'date_of_last_pregnancy']) / np.timedelta64(1, 'W'))
        assert df.at[person, 'ps_gestational_age_in_weeks'] == (foetal_age_weeks + 2)


def test_application_of_risk_of_twin_pregnancy(seed):
    """Runs the simulation with the core modules, all reproductive age women as pregnant and forces all pregnancies to
    be twins. Other functionality related to or dependent upon twin birth is tested in respective module test files"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    sim.make_initial_population(n=100)

    # Force all reproductive age women to be pregnant
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # set risk of twin birth to one
    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_multiples'] = 1.0
    params['prob_ectopic_pregnancy'] = 0.0

    df = sim.population.props
    women = df.loc[df.is_alive & (df.sex == 'F') & df.is_pregnant]
    df.loc[women.index, 'date_of_last_pregnancy'] = sim.date - pd.DateOffset(weeks=1)
    for woman in women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

    # Run the pregnancy supervisor event
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    pregnancy_sup.apply(sim.population)

    # check all pregnancies are now twins as expected
    assert df.loc[women.index, 'ps_multiple_pregnancy'].all().all()


def test_spontaneous_abortion_ends_pregnancies_as_expected(seed):
    """Test to check that risk of spontaneous abortion is applied as expected within the population and leads to the
    end of pregnancy"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    starting_population = 100

    sim.make_initial_population(n=starting_population)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # Define women of interest in the population
    df = sim.population.props
    women = df.loc[df.is_alive & (df.sex == 'F') & df.is_pregnant]
    df.loc[women.index, 'date_of_last_pregnancy'] = sim.date - pd.DateOffset(weeks=2)
    for woman in women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

    # Set risk of miscarriage to 1 (block other pregnancy loss)
    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_ectopic_pregnancy'] = 0.0
    params['prob_induced_abortion_per_month'] = 0.0

    sim.modules['PregnancySupervisor'].ps_linear_models['spontaneous_abortion'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1.0)

    # Run the event
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    pregnancy_sup.apply(sim.population)

    # Check all important variables are updated
    assert not df.loc[women.index, 'is_pregnant'].any().any()
    assert (df.loc[women.index, 'ps_prev_spont_abortion']).all().all()
    assert (df.loc[women.index, 'ps_gestational_age_in_weeks'] == 0).all().all()
    assert pd.isnull(df.loc[women.index, 'la_due_date_current_pregnancy']).all().all()

    for person in women.index:
        assert sim.modules['PregnancySupervisor'].mother_and_newborn_info[person]['delete_mni']


def test_induced_abortion_ends_pregnancies_as_expected(seed):
    """Test to check that risk of induced abortion is applied as expected within the population and leads to the
    end of pregnancy"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    starting_population = 100

    sim.make_initial_population(n=starting_population)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_ectopic_pregnancy'] = 0.0
    sim.modules['PregnancySupervisor'].ps_linear_models['spontaneous_abortion'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0.0)

    # set risk of abortion to 1
    params['prob_induced_abortion_per_month'] = 1.0

    # Define women of interest in the population
    df = sim.population.props
    women = df.loc[df.is_alive & (df.sex == 'F') & df.is_pregnant]
    df.loc[women.index, 'co_unintended_preg'] = True
    df.loc[women.index, 'date_of_last_pregnancy'] = sim.date - pd.DateOffset(weeks=6)
    for woman in women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

    # Run the event
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    pregnancy_sup.apply(sim.population)

    # check variables
    assert ~df.loc[women.index, 'is_pregnant'].all().all()
    assert (df.loc[women.index, 'ps_gestational_age_in_weeks'] == 0).all().all()
    assert pd.isnull(df.loc[women.index, 'la_due_date_current_pregnancy']).all().all()
    for person in women.index:
        assert sim.modules['PregnancySupervisor'].mother_and_newborn_info[person]['delete_mni']


def test_abortion_complications(seed):
    """Test that complications associate with abortion are correctly applied via the pregnancy supervisor event. Also
     test women seek care and/or experience risk of death as expected """

    def check_abortion_logic(abortion_type):
        sim = Simulation(start_date=start_date, seed=seed)
        register_modules(sim)

        starting_population = 100
        sim.make_initial_population(n=starting_population)
        set_all_women_as_pregnant_and_reset_baseline_parity(sim)
        sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

        params = sim.modules['PregnancySupervisor'].current_parameters

        # Set the relvant risk of pregnancy loss to 1
        if abortion_type == 'spontaneous':
            sim.modules['PregnancySupervisor'].ps_linear_models['spontaneous_abortion'] = \
                LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    1)
            weeks = 2
            params['prob_spontaneous_abortion_death'] = 1.0

        else:
            sim.modules['PregnancySupervisor'].ps_linear_models['spontaneous_abortion'] = \
                LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    0.0)
            params['prob_induced_abortion_per_month'] = 1.0
            params['prob_induced_abortion_death'] = 1.0
            weeks = 6

        # Set risk of complications and care seeking to 1 - set treatment to 100% effective
        params['prob_complicated_sa'] = 1.0
        params['prob_complicated_ia'] = 1.0
        params['prob_seek_care_pregnancy_loss'] = 1.0
        params['prob_haemorrhage_post_abortion'] = 1.0
        params['prob_sepsis_post_abortion'] = 1.0
        params['prob_injury_post_abortion'] = 1.0
        params['prob_ectopic_pregnancy'] = 0.0
        params['treatment_effect_post_abortion_care'] = 0.0

        lab_params = sim.modules['Labour'].current_parameters
        lab_params['mean_hcw_competence_hc'] = [1, 1]
        lab_params['mean_hcw_competence_hp'] = [1, 1]
        lab_params['prob_hcw_avail_retained_prod'] = 1

        df = sim.population.props
        pregnant_women = df.loc[df.is_alive & df.is_pregnant]
        for woman in pregnant_women.index:
            pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

        # Select one pregnant woman and run the pregnancy supervisor event (populate key variables)
        mother_id = pregnant_women.index[0]

        pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])

        # Move the sim date forward and run event again
        sim.date = sim.date + pd.DateOffset(weeks=weeks)
        pregnancy_sup.apply(sim.population)

        # check abortion complications correctly stored in bitset properties
        assert sim.modules['PregnancySupervisor'].abortion_complications.has_all(mother_id, 'haemorrhage')
        assert sim.modules['PregnancySupervisor'].abortion_complications.has_all(mother_id, 'sepsis')
        if abortion_type == 'induced':
            assert sim.modules['PregnancySupervisor'].abortion_complications.has_all(mother_id, 'injury')

        # Check that date of onset stored in mni dict to allow for daly calculations
        assert (sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['abortion_onset'] == sim.date)
        assert (sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['abortion_haem_onset'] ==
                sim.date)
        assert (sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['abortion_sep_onset'] == sim.date)

        # Check the sim event queue for the death event
        events = sim.find_events_for_person(person_id=mother_id)
        events = [e.__class__ for d, e in events]
        assert pregnancy_supervisor.EarlyPregnancyLossDeathEvent in events

        # And then check the HSI queue for the care seeking event, care is sought via generic appts
        health_system = sim.modules['HealthSystem']
        hsi_events = health_system.find_events_for_person(person_id=mother_id)
        hsi_events = [e.__class__ for d, e in hsi_events]
        from tlo.methods.hsi_generic_first_appts import (
            HSI_GenericEmergencyFirstApptAtFacilityLevel1,
        )
        assert HSI_GenericEmergencyFirstApptAtFacilityLevel1 in hsi_events

        emergency_appt = HSI_GenericEmergencyFirstApptAtFacilityLevel1(person_id=mother_id,
                                                                       module=sim.modules['PregnancySupervisor'])
        emergency_appt.apply(person_id=mother_id, squeeze_factor=0.0)
        hsi_events = health_system.find_events_for_person(person_id=mother_id)
        hsi_events = [e.__class__ for d, e in hsi_events]
        assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement in hsi_events

        # set treatment effect to 0 to ensure treatment is effective
        pac = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement(
            person_id=mother_id, module=sim.modules['CareOfWomenDuringPregnancy'])
        pac.apply(person_id=mother_id, squeeze_factor=0.0)

        assert sim.population.props.at[mother_id, 'ac_received_post_abortion_care']

        # Define and run event, check woman has correctly died
        death_event = pregnancy_supervisor.EarlyPregnancyLossDeathEvent(module=sim.modules['PregnancySupervisor'],
                                                                        individual_id=mother_id,
                                                                        cause=f'{abortion_type}_abortion')

        death_event.apply(mother_id)

        assert sim.population.props.at[mother_id, 'is_alive']
        assert sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['delete_mni']

    # Run these check for both types of abortion
    check_abortion_logic('spontaneous')
    check_abortion_logic('induced')


def test_still_births_ends_pregnancies_as_expected(seed):
    """Runs the simulation with the core modules and all women of reproductive age as pregnant. Sets antenatal still
    birth risk to 1 and runs checks """
    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    starting_population = 100

    sim.make_initial_population(n=starting_population)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # Only allow pregnancy loss from stillbirth
    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_ectopic_pregnancy'] = 0.0
    params['prob_induced_abortion_per_month'] = 0.0

    sim.modules['PregnancySupervisor'].ps_linear_models['spontaneous_abortion'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0.0)
    sim.modules['PregnancySupervisor'].ps_linear_models['early_onset_labour'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0.0)

    params['prob_still_birth_per_month'] = 1.0

    # select the relevant women
    df = sim.population.props

    women = df.loc[df.is_alive & (df.sex == 'F') & df.is_pregnant]
    df.loc[women.index, 'date_of_last_pregnancy'] = sim.date - pd.DateOffset(weeks=25)
    for woman in women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

    # Run population level event
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    pregnancy_sup.apply(sim.population)

    # Check key properties
    assert ~df.loc[women.index, 'is_pregnant'].all().all()
    assert df.loc[women.index, 'ps_prev_stillbirth'].all().all()
    assert (df.loc[women.index, 'ps_gestational_age_in_weeks'] == 0).all().all()
    assert pd.isnull(df.loc[women.index, 'la_due_date_current_pregnancy']).all().all()

    for person in women.index:
        assert sim.modules['PregnancySupervisor'].mother_and_newborn_info[person]['delete_mni']


def test_run_all_births_end_ectopic_no_care_seeking(seed):
    """Test to check that risk of ectopic pregnancy, progression, careseeking and treatment occur as expected"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    starting_population = 100
    sim.make_initial_population(n=starting_population)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)

    # run sim for 0 days
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # We force all pregnancies to be ectopic, never trigger care seeking, always lead to rupture and death
    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_ectopic_pregnancy'] = 1.0
    params['prob_seek_care_pregnancy_loss'] = 0.0
    params['prob_ectopic_pregnancy_death'] = 1.0
    params['treatment_effect_ectopic_pregnancy_treatment'] = 1.0

    # generate pregnancies and MNI dictionaries
    df = sim.population.props
    pregnant_women = df.loc[df.is_alive & df.is_pregnant]
    for woman in pregnant_women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

    # Select one pregnant woman and run the pregnancy supervisor event (populate key variables)
    mother_id = pregnant_women.index[0]
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])

    sim.date = sim.date + pd.DateOffset(weeks=1)
    pregnancy_sup.apply(sim.population)

    # Check she has experience ectopic pregnancy as expected
    assert (df.at[mother_id, 'ps_ectopic_pregnancy'] == 'not_ruptured')

    # and the correct event is scheduled
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]
    assert pregnancy_supervisor.EctopicPregnancyEvent in events

    # run the ectopic even and check she is no longer pregnant and will experince rupture as she hasnt sought care
    sim.date = sim.date + pd.DateOffset(weeks=7)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 8

    ectopic_event = pregnancy_supervisor.EctopicPregnancyEvent(individual_id=mother_id,
                                                               module=sim.modules['PregnancySupervisor'])
    ectopic_event.apply(mother_id)

    assert not df.at[mother_id, 'is_pregnant']
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]
    assert pregnancy_supervisor.EctopicPregnancyRuptureEvent in events

    # Run the rupture event, check rupture has occurred and that death has been scheduled
    rupture_event = pregnancy_supervisor.EctopicPregnancyRuptureEvent(individual_id=mother_id,
                                                                      module=sim.modules['PregnancySupervisor'])
    rupture_event.apply(mother_id)

    assert (df.at[mother_id, 'ps_ectopic_pregnancy'] == 'ruptured')
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]
    assert pregnancy_supervisor.EarlyPregnancyLossDeathEvent in events

    # run the death event checking death has occurred as expected
    death_event = pregnancy_supervisor.EarlyPregnancyLossDeathEvent(individual_id=mother_id,
                                                                    module=sim.modules['PregnancySupervisor'],
                                                                    cause='ectopic_pregnancy')
    death_event.apply(mother_id)
    assert not df.at[mother_id, 'is_alive']


def test_preterm_labour_logic(seed):
    """Test to check that risk of preterm labour is applied as expected and triggers early labour through correct event
     scheduling """

    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    params = sim.modules['PregnancySupervisor'].current_parameters

    # We force risk of preterm birth to be 1, meaning all women will go into labour at month 5
    sim.modules['PregnancySupervisor'].ps_linear_models['early_onset_labour'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1.0)

    # And stop attendance to ANC
    sim.modules['PregnancySupervisor'].ps_linear_models['early_initiation_anc4'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0.0)
    params['prob_late_initiation_anc4'] = 1.0
    params['prob_anc1_months_5_to_9'] = [0, 0, 0, 0, 0, 1]

    # Stop any pregnancies ending in pregnancy loss
    turn_off_antenatal_pregnancy_loss(sim)

    # Clear events, including original scheduling of labour
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    df = sim.population.props
    pregnant_women = df.loc[df.is_pregnant & df.is_alive]
    for woman in pregnant_women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

    # Move the sim date forward
    sim.date = sim.date + pd.DateOffset(weeks=20)

    # Run the event again to apply risk of preterm birth
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    pregnancy_sup.apply(sim.population)

    # check due date for all women has been reset to occur in less than one months time
    latest_labour_can_happen = sim.date + pd.DateOffset(days=((27 - 22) * 7))
    assert (df.loc[pregnant_women.index, 'la_due_date_current_pregnancy'] <= latest_labour_can_happen).all().all()

    # Check that each woman has had a new labour onset event scheduled (original event has been cleared so if logic
    # hasnt held no labour onset event would have been set)
    mother_id = pregnant_women.index[0]
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]
    assert labour.LabourOnsetEvent in events

    # move time forward to due date
    sim.date = df.at[mother_id, 'la_due_date_current_pregnancy']

    # Call checker function from labour which returns BOOl if labour can proceed and check labour can proceed for
    # all women
    can_labour_now_go_ahead = sim.modules['Labour'].check_labour_can_proceed(mother_id)
    assert can_labour_now_go_ahead

    # Next we run the labour onset event and check that the key events are scheduled (birth and death events)
    labour_event = labour.LabourOnsetEvent(individual_id=mother_id, module=sim.modules['Labour'])
    labour_event.apply(mother_id)

    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]
    assert labour.LabourDeathAndStillBirthEvent in events
    assert labour.BirthAndPostnatalOutcomesEvent in events


def test_check_first_anc_visit_scheduling(seed):
    """Test to ensure first ANC visit is scheduled for women as expected """
    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    turn_off_antenatal_pregnancy_loss(sim)

    # Set parameters so that women will attend ANC in 1 months time
    params = sim.modules['PregnancySupervisor'].current_parameters
    sim.modules['PregnancySupervisor'].ps_linear_models['early_initiation_anc4'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)
    params['prob_anc1_months_2_to_4'] = [1, 0, 0]

    df = sim.population.props
    women = df.loc[df.is_alive & (df.sex == 'F') & df.is_pregnant]
    df.loc[women.index, 'date_of_last_pregnancy'] = sim.date - pd.DateOffset(weeks=6)
    for woman in women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

    # Define and run the pregnancy supervisor event
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    pregnancy_sup.apply(sim.population)

    # Check the ps_date_of_anc1 property as its used to schedule ANC. Make sure ANC will occur in one month and before
    # two months
    earliest_anc_can_happen = sim.date
    latest_anc_can_happen = sim.date + pd.DateOffset(days=6)

    assert (df.loc[women.index, 'ps_date_of_anc1'] >= earliest_anc_can_happen).all().all()
    assert (df.loc[women.index, 'ps_date_of_anc1'] <= latest_anc_can_happen).all().all()

    # Finally check that the HSI event has been correctly scheduled
    mother_id = women.index[0]
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=mother_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact in hsi_events


def test_pregnancy_supervisor_anaemia(seed):
    """Tests the application of risk of maternal anaemia within the pregnancy supervisor event"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    turn_off_antenatal_pregnancy_loss(sim)

    # Set the risk of anaemia to 1
    params = sim.modules['PregnancySupervisor'].current_parameters
    params['baseline_prob_anaemia_per_month'] = 1.0
    params['prob_mild_mod_sev_anaemia'] = [0, 0, 1]

    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    df = sim.population.props
    pregnant_women = df.loc[df.is_alive & df.is_pregnant]
    for woman in pregnant_women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])

    # Move the date forward, this will mean when the event is run women will be 4 weeks pregnant, this is the
    # time point at which risk of anaemia is first applied
    sim.date = sim.date + pd.DateOffset(weeks=2)
    pregnancy_sup.apply(sim.population)

    assert (df.loc[pregnant_women.index, 'ps_anaemia_in_pregnancy'] == 'severe').all().all()
    for person in pregnant_women.index:
        assert (sim.modules['PregnancySupervisor'].mother_and_newborn_info[person]['severe_anaemia_onset'] == sim.date)
        assert pd.isnull(sim.modules['PregnancySupervisor'].mother_and_newborn_info[person]['severe_anaemia_'
                                                                                            'resolution'])

    # Reset anaemia status
    df.loc[pregnant_women.index, 'ps_anaemia_in_pregnancy'] = 'none'
    for person in pregnant_women.index:
        sim.modules['PregnancySupervisor'].mother_and_newborn_info[person]['severe_anaemia_onset'] = pd.NaT

    # Reset anaemia status in the women of interest and set that they are receiving iron and folic acid treatment, which
    # should reduce the risk of iron or folate deficiency (which increase risk of anaemia)
    df.loc[pregnant_women.index, 'ps_anaemia_in_pregnancy'] = 'none'
    df.loc[pregnant_women.index, 'ac_receiving_iron_folic_acid'] = True

    # Set treatment effect to 100% (i.e 0)
    params['treatment_effect_iron_folic_acid_anaemia'] = 0.0

    sim.date = sim.date + pd.DateOffset(weeks=5)
    pregnancy_sup.apply(sim.population)

    assert (df.loc[pregnant_women.index, 'ps_anaemia_in_pregnancy'] == 'none').all().all()
    for person in pregnant_women.index:
        assert pd.isnull(sim.modules['PregnancySupervisor'].mother_and_newborn_info[person]['severe_anaemia_onset'])


def test_pregnancy_supervisor_placental_conditions_and_antepartum_haemorrhage(seed):
    """Tests the application of risk of placenta praevia, abruption and antenatal haemorrhage within the pregnancy
    supervisor event"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    turn_off_antenatal_pregnancy_loss(sim)

    # Set the probability of the placental conditions which lead to haemorrhage as 1 and that the woman who experiences
    # haemorrhage will choose to seek care
    params = sim.modules['PregnancySupervisor'].current_parameters

    params['prob_placenta_praevia'] = 1.0
    params['prob_placental_abruption_per_month'] = 1.0
    params['prob_seek_care_pregnancy_complication'] = 1.0

    # Similarly set the probability that these conditions will trigger haemorrhage to 1
    params['prob_aph_placenta_praevia'] = 1.0
    params['prob_aph_placental_abruption'] = 1.0
    # Force all haemorrhage as severe
    params['prob_mod_sev_aph'] = [0, 1]

    # Run the PregnancySupervisorEvent to generate the mni dictionary
    sim.date = sim.date + pd.DateOffset(weeks=1)
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    pregnancy_sup.apply(sim.population)

    # All women have risk of placenta praevia applied early in pregnancy, assert this has happened as it should
    df = sim.population.props
    pregnant_women = df.loc[df.is_alive & df.is_pregnant]
    assert (df.loc[pregnant_women.index, 'ps_placenta_praevia']).all().all()

    # Set the date forward to when risk of placental abruption and  antepartum haemorrhage is applied to women in the
    # PregnancySupervisorEvent and run the event
    sim.date = sim.date + pd.DateOffset(weeks=19)
    pregnancy_sup.apply(sim.population)

    # Check the dataframe has been updated correctly for these women and that the date of onset is stored in the MNI
    assert (df.loc[pregnant_women.index, 'ps_placental_abruption']).all().all()
    assert (df.loc[pregnant_women.index, 'ps_antepartum_haemorrhage'] == 'severe').all().all()
    for person in pregnant_women.index:
        assert (sim.modules['PregnancySupervisor'].mother_and_newborn_info[person]['severe_aph_onset'] == sim.date)

    # Select one of the women from the series and check that the correct HSI has been scheduled for her after she has
    # chosen to seek care
    mother_id = pregnant_women.index[0]
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=mother_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment in hsi_events

    # Now clear the event queue and reset haemorrhage variables
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    df.loc[pregnant_women.index, 'ps_antepartum_haemorrhage'] = 'none'
    df.loc[pregnant_women.index, 'ps_emergency_event'] = False

    # Set risk of care seeking to 0 and risk of death to 1 so all women who experience haemorrhage should die
    params['prob_seek_care_pregnancy_complication'] = 0.0
    params['prob_antepartum_haemorrhage_death'] = 1.0

    # Move date forward again to the next time point in pregnancy risk is applied and run the event
    sim.date = sim.date + pd.DateOffset(weeks=5)
    pregnancy_sup.apply(sim.population)

    # Check that a woman from the series has correctly died
    mother_id = pregnant_women.index[0]
    assert not sim.population.props.at[mother_id, 'is_alive']
    assert mother_id not in list(sim.modules['PregnancySupervisor'].mother_and_newborn_info)


def test_pregnancy_supervisor_pre_eclampsia_and_progression(seed):
    """Tests the application of risk of pre-eclampsia within the pregnancy supervisor event"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    turn_off_antenatal_pregnancy_loss(sim)

    # Set the monthly risk of pre-eclampsia to 1, ensuring all pregnant women develop the condition the first month
    # risk is applied (22 week)
    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_pre_eclampsia_per_month'] = 1.0
    params['treatment_effect_calcium_pre_eclamp'] = 1.0

    # pre-eclampsia/spe/ec
    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
    pregnant_women = df.loc[df.is_alive & df.is_pregnant]
    for woman in pregnant_women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

    # Set sim date to when risk of pre-eclampsia is applied
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    sim.date = sim.date + pd.DateOffset(weeks=20)
    pregnancy_sup.apply(sim.population)

    # Check all women of interest have developed the correct condition after the events run
    assert (df.loc[pregnant_women.index, 'ps_htn_disorders'] == 'mild_pre_eclamp').all().all()
    assert (df.loc[pregnant_women.index, 'ps_prev_pre_eclamp']).all().all()

    # Now we modify the probability that women with mild pre-eclampsia will progress to severe pre-eclampsia when the
    # pregnancy supervisor event is ran at the next key point in their gestation (week 27)
    params['probs_for_mpe_matrix'] = [0, 0, 0, 1, 0]
    params['prob_seek_care_pregnancy_complication'] = 1.0

    sim.date = sim.date + pd.DateOffset(weeks=5)
    pregnancy_sup.apply(sim.population)

    # Check women have correctly progressed to a more severe disease state
    assert (df.loc[pregnant_women.index, 'ps_htn_disorders'] == 'severe_pre_eclamp').all().all()
    for person in pregnant_women.index:
        assert mni[person]['new_onset_spe']

    # And that to correct HSI has been scheduled, as we set prob of care seeking to 1
    mother_id = pregnant_women.index[0]
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=mother_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment in hsi_events

    # Now clear the event queue
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    # Now we set monthly risk of death from severe pre-eclampsia to 1 and remove care seeking
    params['prob_severe_pre_eclampsia_death'] = 1.0
    params['prob_seek_care_pregnancy_complication'] = 0.0

    # prevent womans disease state from changing during the next event run (mechanism of death for eclampsia is
    # different)
    params['probs_for_spe_matrix'] = [0, 0, 0, 1, 0]
    for person in pregnant_women.index:
        mni[person]['new_onset_spe'] = True
        df.at[person, 'ps_emergency_event'] = True

    sim.date = sim.date + pd.DateOffset(weeks=4)
    pregnancy_sup.apply(sim.population)

    # Check the death has occurred
    assert not df.loc[pregnant_women.index, 'is_alive'].any().any()


def test_pregnancy_supervisor_gestational_hypertension_and_progression(seed):
    """Tests the application of risk of gestational_hypertension within the pregnancy supervisor event"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    turn_off_antenatal_pregnancy_loss(sim)

    # set risk of gestational hypertension to 1
    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_pre_eclampsia_per_month'] = 0.0
    params['prob_gest_htn_per_month'] = 1.0
    params['treatment_effect_gest_htn_calcium'] = 1.0

    df = sim.population.props
    pregnant_women = df.loc[df.is_alive & df.is_pregnant]
    for woman in pregnant_women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

    # run the event
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    sim.date = sim.date + pd.DateOffset(weeks=20)
    pregnancy_sup.apply(sim.population)

    # Check all women of interest have developed the correct condition after the events run
    assert (df.loc[pregnant_women.index, 'ps_htn_disorders'] == 'gest_htn').all().all()

    # TODO: test progression (need to sort progression matrix as a parameter)
    # TODO: test that anti htn reduces risk of progression from mild to moderate


def test_pregnancy_supervisor_gdm(seed):
    """Tests the application of risk of gestational diabetes within the pregnancy supervisor event"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    turn_off_antenatal_pregnancy_loss(sim)

    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_gest_diab_per_month'] = 1.0

    df = sim.population.props
    pregnant_women = df.loc[df.is_alive & df.is_pregnant]
    for woman in pregnant_women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], woman)

    # Run pregnancy supervisor event
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    sim.date = sim.date + pd.DateOffset(weeks=20)
    pregnancy_sup.apply(sim.population)

    # check GDM status correctly applied
    assert (df.loc[pregnant_women.index, 'ps_gest_diab'] == 'uncontrolled').all().all()
    assert (df.loc[pregnant_women.index, 'ps_prev_gest_diab']).all().all()

    mother_id = pregnant_women.index[0]

    # Run birth and check GDM has resolved
    child_id = sim.population.do_birth()
    sim.modules['Labour'].on_birth(mother_id, child_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    assert (sim.population.props.at[mother_id, 'ps_gest_diab'] == 'none')


def test_pregnancy_supervisor_chorio_and_prom(seed):
    """Tests the application of risk of chorioamnionitis and PROM within the pregnancy supervisor event"""

    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    turn_off_antenatal_pregnancy_loss(sim)

    # Set risk of PROM to 1
    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_prom_per_month'] = 0.0
    params['prob_chorioamnionitis'] = 1.0
    params['prob_seek_care_pregnancy_complication'] = 1.0

    df = sim.population.props
    pregnant_women = df.loc[df.is_alive & df.is_pregnant]
    for woman in pregnant_women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

    # Run the event
    sim.date = sim.date + pd.DateOffset(weeks=20)
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    pregnancy_sup.apply(sim.population)

    # Check that despite risk of chorio being 1, not one should develop is because PROM cannot occur
    assert not df.loc[pregnant_women.index, 'ps_premature_rupture_of_membranes'].any().any()

    sim.date = sim.date + pd.DateOffset(weeks=5)
    pregnancy_sup.apply(sim.population)

    assert not df.loc[pregnant_women.index, 'ps_premature_rupture_of_membranes'].any().any()
    assert not df.loc[pregnant_women.index, 'ps_chorioamnionitis'].any().any()

    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_prom_per_month'] = 1.0
    params['prob_chorioamnionitis'] = 1.0

    # Clear the event queue
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    # Roll back gestational age, set risk of stillbirth to 1
    df.loc[pregnant_women.index, 'ps_gestational_age_in_weeks'] = \
        (df.loc[pregnant_women.index, 'ps_gestational_age_in_weeks'] - 1)

    df.loc[pregnant_women.index, 'ps_premature_rupture_of_membranes'] = True
    pregnancy_sup.apply(sim.population)

    # Check women have correctly developed chorio
    assert (df.loc[pregnant_women.index, 'ps_chorioamnionitis']).all().all()

    # Check care seeking has occured as expected
    mother_id = pregnant_women.index[0]
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=mother_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment in hsi_events

    # Now clear the event queue
    df.loc[pregnant_women.index, 'ps_gestational_age_in_weeks'] = \
        (df.loc[pregnant_women.index, 'ps_gestational_age_in_weeks'] - 1)

    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    df.loc[pregnant_women.index, 'ps_chorioamnionitis'] = False

    # Finally, block care seeking and set risk of death to high
    params['prob_seek_care_pregnancy_complication'] = 0.0
    params['prob_antenatal_sepsis_death'] = 1.0

    # prevent preterm birth which can effect care seeking by updating la_due_date_current_pregnancy
    params['baseline_prob_early_labour_onset'] = [0.0, 0.0, 0.0, 0.0]

    pregnancy_sup.apply(sim.population)

    # Check women from the series has correctly died
    assert (df.loc[pregnant_women.index, 'ps_chorioamnionitis']).all().all()
    assert not (df.loc[pregnant_women.index, 'is_alive']).any().any()
    for person in pregnant_women.index:
        assert person not in list(sim.modules['PregnancySupervisor'].mother_and_newborn_info)


def test_induction_of_labour_logic(seed):
    """Tests the that woman who are post-term are seeking care for induction of labour"""

    sim = Simulation(start_date=start_date, seed=seed)
    register_modules(sim)
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant_and_reset_baseline_parity(sim)
    turn_off_antenatal_pregnancy_loss(sim)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_seek_care_induction'] = 1.0

    df = sim.population.props
    pregnant_women = df.loc[df.is_alive & df.is_pregnant]
    for woman in pregnant_women.index:
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], woman)

    # Run the event, asume all women are now 41 weeks pregnant
    pregnancy_sup = pregnancy_supervisor.PregnancySupervisorEvent(module=sim.modules['PregnancySupervisor'])
    sim.date = sim.date + pd.DateOffset(weeks=39)
    pregnancy_sup.apply(sim.population)

    # Check care seeking for induction has occured
    mother_id = pregnant_women.index[0]
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=mother_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_PresentsForInductionOfLabour in hsi_events

    # Clear the event queue
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    # Roll back gestational age, set risk of stillbirth to 1
    df.loc[pregnant_women.index, 'ps_gestational_age_in_weeks'] = \
        (df.loc[pregnant_women.index, 'ps_gestational_age_in_weeks'] - 1)

    # block care seeking for induction
    params['prob_seek_care_induction'] = 0.0

    # module code divides monthly risk by weeks (assuming 4.5 weeks in a month) so we set the intercept
    # 4.5 times greater than 1 to assure sb will happen
    params['prob_still_birth_per_month'] = 4.5

    # Check that instead of seeking care for induction women have experience post term stillbirth
    pregnancy_sup.apply(sim.population)
    assert not (df.loc[pregnant_women.index, 'is_pregnant']).any().any()
    assert (df.loc[pregnant_women.index, 'ps_prev_stillbirth']).all().all()
    for person in pregnant_women.index:
        assert sim.modules['PregnancySupervisor'].mother_and_newborn_info[person]['delete_mni']
