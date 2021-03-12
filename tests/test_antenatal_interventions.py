import pytest
import os
import pandas as pd
from pathlib import Path
from tlo import Date, Simulation, logging
from tlo.lm import LinearModel, LinearModelType
from tlo.methods.hsi_generic_first_appts import (
    HSI_GenericEmergencyFirstApptAtFacilityLevel1)

from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    depression,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)

seed = 560

log_config = {
    "filename": "pregnancy_supervisor_test",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # warning  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.contraception": logging.DEBUG,
        "tlo.methods.labour": logging.DEBUG,
        "tlo.methods.healthsystem": logging.FATAL,
        "tlo.methods.hiv": logging.FATAL,
        "tlo.methods.newborn_outcomes": logging.DEBUG,
        "tlo.methods.antenatal_care": logging.DEBUG,
        "tlo.methods.pregnancy_supervisor": logging.DEBUG,
        "tlo.methods.postnatal_supervisor": logging.DEBUG,
    }
}

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


def register_all_modules():
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*']),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 malaria.Malaria(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath))

    return sim


def test_daisy_chain_scheduling_from_anc_1():
    """This test checks the inbuilt daisy-chain scheduling within antenatal care contacts. The test checks that the
    correct HSI is scheduled by each proceeding HSI and on the correct date and gestational age. It ensures the correct
     maternal variables are updated in relation to scheduling
    """
    # Register the key modules and run the simulation for one day
    sim = register_all_modules()
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # Select a woman from the dataframe of reproductive age
    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]

    # Set key pregnancy variables and schedule labour
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'date_of_last_pregnancy'] = start_date
    df.at[mother_id, 'ps_will_attend_four_or_more_anc'] = True
    df.at[mother_id, 'ps_date_of_anc1'] = start_date + pd.DateOffset(weeks=8)
    sim.modules['Labour'].set_date_of_labour(mother_id)

    # ensure care seeking will continue for all ANC visits
    params = sim.modules['CareOfWomenDuringPregnancy'].parameters
    params['ac_linear_equations']['anc_continues'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)

    # Define the first ANC HSI and set the date to when this mother has been scheduled to attend this visit (via
    # pregnancy supervisor event please see test_check_first_anc_visit_scheduling for proof of concept)
    updated_mother_id = int(mother_id)  # todo: had to force as int as dx_manager doesnt recognise int64

    # Register the anc hsi
    first_anc = antenatal_care.HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id, facility_level_of_this_hsi=2)
    second_anc = antenatal_care.HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id, facility_level_of_this_hsi=2)
    third_anc = antenatal_care.HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id, facility_level_of_this_hsi=2)
    fourth_anc = antenatal_care.HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id, facility_level_of_this_hsi=2)
    fifth_anc = antenatal_care.HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id, facility_level_of_this_hsi=2)
    sixth_anc = antenatal_care.HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id, facility_level_of_this_hsi=2)
    seventh_anc = antenatal_care.HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id, facility_level_of_this_hsi=2)
    eight_anc = antenatal_care.HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id, facility_level_of_this_hsi=2)

    sim.date = sim.date + pd.DateOffset(weeks=8)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 10

    recommended_ga_at_next_visit = \
        sim.modules['CareOfWomenDuringPregnancy'].determine_gestational_age_for_next_contact(mother_id)
    assert (recommended_ga_at_next_visit == 20)

    # Run the event and check that the visit has been recorded in the dataframe along with the date of the next visit
    first_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 1)
    assert (df.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=10)))

    # Check that the next event in the schedule
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(updated_mother_id) if
        isinstance(ev[1], antenatal_care.HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact)
    ][0]
    assert date_event == (sim.date + pd.DateOffset(weeks=10))

    # Move date of sim to the date next visit should occur and update gestational age so event will run
    sim.date = date_event
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 10
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    # Run the same checks
    second_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 2)
    assert (df.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=6)))

    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(updated_mother_id) if
        isinstance(ev[1], antenatal_care.HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact)
    ][0]
    assert date_event == (sim.date + pd.DateOffset(weeks=6))

    # Now repeat this process for the rest of the series
    sim.date = date_event
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 6
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    third_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 3)
    assert (df.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=4)))

    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(updated_mother_id) if
        isinstance(ev[1], antenatal_care.HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact)
    ][0]
    assert date_event == (sim.date + pd.DateOffset(weeks=4))

    sim.date = date_event
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 4
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    fourth_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 4)
    assert (df.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=4)))

    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(updated_mother_id) if
        isinstance(ev[1], antenatal_care.HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact)
    ][0]
    assert date_event == (sim.date + pd.DateOffset(weeks=4))

    sim.date = date_event
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 4
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    fifth_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 5)
    assert (df.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=2)))

    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(updated_mother_id) if
        isinstance(ev[1], antenatal_care.HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact)
    ][0]
    assert date_event == (sim.date + pd.DateOffset(weeks=2))

    sim.date = date_event
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 2
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    sixth_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 6)
    assert (df.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=2)))

    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(updated_mother_id) if
        isinstance(ev[1], antenatal_care.HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact)
    ][0]
    assert date_event == (sim.date + pd.DateOffset(weeks=2))

    sim.date = date_event
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 2
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    seventh_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 7)
    assert (df.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=2)))

    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(updated_mother_id) if
        isinstance(ev[1], antenatal_care.HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact)
    ][0]
    assert date_event == (sim.date + pd.DateOffset(weeks=2))

    sim.date = date_event
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 2
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    eight_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 8)


def test_anc_contacts_that_should_not_run_wont_run():
    # Register the key modules and run the simulation for one day
    sim = register_all_modules()
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]

    # Set key pregnancy variables
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'date_of_last_pregnancy'] = start_date
    df.at[mother_id, 'ps_date_of_anc1'] = start_date + pd.DateOffset(weeks=8)
    sim.modules['Labour'].set_date_of_labour(mother_id)

    updated_mother_id = int(mother_id)

    # define HSIs
    first_anc = antenatal_care.HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id, facility_level_of_this_hsi=2)
    second_anc = antenatal_care.HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id, facility_level_of_this_hsi=2)

    sim.date = start_date + pd.DateOffset(weeks=8)

    # Set mother to being currently in labour
    df.at[mother_id, 'la_currently_in_labour'] = True

    # Check HSI has not ran and another ANC is not scheduled (as she will no longer be pregnant)
    first_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 0)
    assert pd.isnull(df.at[mother_id, 'ac_date_next_contact'])

    # Reset labour variable and set gestational age to very young- ANC likely scheduled from last pregnancy and woman
    # is now pregnant again. Therefore event shouldnt run
    df.at[mother_id, 'la_currently_in_labour'] = False
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 4

    first_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 0)
    assert pd.isnull(df.at[mother_id, 'ac_date_next_contact'])

    # Reset the gestational age and set the squeeze factor of the HSI as very high. Woman will leave and HSI should not
    # run
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 10

    first_anc.apply(person_id=updated_mother_id, squeeze_factor=1)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 0)
    assert pd.isnull(df.at[mother_id, 'ac_date_next_contact'])

    # TODO: check she returns tomorrow?

    # Finally set woman as inpatient when she is due for her first ANC appointment
    df.at[mother_id, 'hs_is_inpatient'] = True

    # Check that ANC hasnt ran BUT woman has correctly been scheduled to return for ANC 1 at the next gestational age
    # in the schedule
    first_anc.apply(person_id=updated_mother_id, squeeze_factor=0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 0)
    assert pd.isnull(df.at[mother_id, 'ac_date_next_contact'])

    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(updated_mother_id) if
        isinstance(ev[1], antenatal_care.HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact)
    ][1]
    assert date_event == (sim.date + pd.DateOffset(weeks=10))
    assert (df.at[mother_id, 'ps_date_of_anc1'] == (sim.date + pd.DateOffset(weeks=10)))

    # todo: more test?
    # todo: check anc 2 doesnt run and the next event is scheduled


def test_care_seeking_for_next_contact():
    sim = register_all_modules()
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # Select a woman from the dataframe of reproductive age
    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]
    updated_mother_id = int(mother_id)

    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 10

    # This woman has been determined to attend at least four visits
    df.at[mother_id, 'ps_will_attend_four_or_more_anc'] = True

    # call the function called by all ANC HSIs to scheduled the next visit
    sim.modules['CareOfWomenDuringPregnancy'].antenatal_care_scheduler(individual_id=updated_mother_id,
                                                                       visit_to_be_scheduled=2,
                                                                       recommended_gestation_next_anc=20,
                                                                       facility_level=2)

    # As this woman is set to attend at least 4 ANC contacts, the next visit should be correctly sheduled at the correct
    # gestational age
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(updated_mother_id) if
        isinstance(ev[1], antenatal_care.HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact)
    ][0]

    assert date_event == (sim.date + pd.DateOffset(weeks=10))
    assert (df.at[mother_id, 'ac_date_next_contact'] == date_event)

    # Clear the event queue and reset 'ps_will_attend_four_or_more_anc'
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    df.at[mother_id, 'ac_date_next_contact'] = pd.NaT
    df.at[mother_id, 'ps_will_attend_four_or_more_anc'] = False

    # For women who are not predetermined to attend at least 4 anc contact we use a probability to determine if they
    # will return for the next visit. Here we set that to 0 and check that no event has been sheduled
    params = sim.modules['CareOfWomenDuringPregnancy'].parameters
    params['ac_linear_equations']['anc_continues'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            0)

    sim.modules['CareOfWomenDuringPregnancy'].antenatal_care_scheduler(individual_id=updated_mother_id,
                                                                       visit_to_be_scheduled=3,
                                                                       recommended_gestation_next_anc=26,
                                                                       facility_level=2)

    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=updated_mother_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    assert antenatal_care.HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact not in hsi_events


def test_anc_one_interventions_delivered_as_expected():
    pass

# todo: another test that just calls the other interventions not called in ANC 1?
# TODO: test inpatient stuff...

def test_bp_monitoring():
    pass # causing admissions it shouldnt
