import os
from pathlib import Path

import pandas as pd

from tlo.lm import LinearModel, LinearModelType, Predictor

import pytest
from tlo import Date, Simulation, logging
from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
    symptommanager, postnatal_supervisor
)

seed = 567


# The resource files
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def run_sim_for_0_days_get_mother_id(sim):
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]
    return mother_id


def find_and_return_hsi_events_list(sim, individual_id):
    """Returns HSI event list for an individual"""
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=individual_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    return hsi_events


def set_pregnancy_characteristics(sim, mother_id):
    df = sim.population.props
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'la_due_date_current_pregnancy'] = sim.date
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date - pd.DateOffset(weeks=38)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 40
    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)


def check_event_queue_for_event_and_return_scheduled_event_date(sim, queue_of_interest, individual_id,
                                                                event_of_interest):
    """Checks the hsi OR event queue for an event and returns scheduled date"""
    if queue_of_interest == 'event':
        date_event, event = [
            ev for ev in sim.find_events_for_person(person_id=individual_id) if
            isinstance(ev[1], event_of_interest)
        ][0]
        return date_event
    else:
        date_event, event = [
            ev for ev in sim.modules['HealthSystem'].find_events_for_person(individual_id) if
            isinstance(ev[1], event_of_interest)
        ][0]
        return date_event


def register_modules(ignore_cons_constraints):
    """Register all modules that are required for labour to run"""

    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           ignore_cons_constraints=ignore_cons_constraints),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

    return sim


def test_run_no_constraints():
    """This test runs a simulation with a functioning health system with full service availability and no set
    constraints"""

    sim = register_modules(ignore_cons_constraints=False)

    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(2015, 1, 1))

    check_dtypes(sim)

# TODO: issue with forcing LMs that output odds to occur (i.e. equal 1)


def test_event_scheduling_for_labour_onset_and_home_birth_no_care_seeking():
    """Test that the right events are scheduled during the labour module (and in the right order) for women who delivery
     at home. Spacing between events (in terms of days since labour onset) is enforced via assert functions within the
    labour module"""
    sim = register_modules(ignore_cons_constraints=False)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
    params = sim.modules['Labour'].parameters

    # Set pregnancy characteristics that will allow the labour events to run
    set_pregnancy_characteristics(sim, mother_id)

    # force this woman to decide to deliver at home
    # params['test_care_seeking_probs'] = [1, 0, 0]

    # define and run labour onset event
    labour_onset = labour.LabourOnsetEvent(individual_id=mother_id, module=sim.modules['Labour'])
    labour_onset.apply(mother_id)
    assert (mni[mother_id]['labour_state'] == 'term_labour')

    # Check that the correct events are scheduled for this woman whose labour has started
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]

    assert labour.BirthEvent in events
    assert labour.LabourDeathAndStillBirthEvent in events
    # todo: struggling to force care seeking in this event (way the regression is coded)
    # assert labour.LabourAtHomeEvent in events

    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert labour.HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour not in hsi_events

    # run birth event as this event manages scheduling of postpartum events (home birth and death events have their own
    # tests)
    mni[mother_id]['delivery_setting'] = 'home_birth'
    sim.date = sim.date + pd.DateOffset(days=5)
    sim.event_queue.queue.clear()

    birth_event = labour.BirthEvent(mother_id=mother_id, module=sim.modules['Labour'])
    birth_event.apply(mother_id)

    # Ensure that the postpartum home birth event is scheduled correctly
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]
    assert labour.PostpartumLabourAtHomeEvent in events

    # Set care seeking odds to 0 (as changes event sequence if women seek care- tested later)
    params['prob_careseeking_for_complication'] = 0

    # Define and run postpartum event
    pn_event = labour.PostpartumLabourAtHomeEvent(individual_id=mother_id, module=sim.modules['Labour'])
    pn_event.apply(mother_id)

    # And finally check the first event of the postnatal module is correctly scheduled
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]
    assert postnatal_supervisor.PostnatalWeekOneEvent in events


def test_event_scheduling_for_care_seeking_during_home_birth():
    """Test that women who develop complications during a home birth are scheduled to receive care correctly """
    sim = register_modules(ignore_cons_constraints=False)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    set_pregnancy_characteristics(sim, mother_id)

    df = sim.population.props

    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
    params = sim.modules['Labour'].parameters

    labour_onset = labour.LabourOnsetEvent(individual_id=mother_id, module=sim.modules['Labour'])
    labour_onset.apply(mother_id)

    # Clear the event queues
    sim.event_queue.queue.clear()
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    # Force the woman to experience some complications and force that she will seek care
    mni[mother_id]['delivery_setting'] = 'home_birth'

    params['prob_cephalopelvic_dis'] = 1
    params['prob_obstruction_cpd'] = 1
    params['prob_careseeking_for_complication'] = 1

    # Run the intrapartum home birth event
    home_birth = labour.LabourAtHomeEvent(individual_id=mother_id, module=sim.modules['Labour'])
    home_birth.apply(mother_id)

    # check the correct variables are stored in the mni
    assert mni[mother_id]['sought_care_for_complication']
    assert (mni[mother_id]['sought_care_labour_phase'] == 'intrapartum')

    # Check that the woman will correctly seek care through HSI_GenericEmergencyFirstApptAtFacilityLevel1
    from tlo.methods.hsi_generic_first_appts import ( HSI_GenericEmergencyFirstApptAtFacilityLevel1)
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert HSI_GenericEmergencyFirstApptAtFacilityLevel1 in hsi_events

    # Now run the event
    emergency_appt = HSI_GenericEmergencyFirstApptAtFacilityLevel1(person_id=mother_id,
                                                                   module=sim.modules['Labour'])
    emergency_appt.apply(person_id=mother_id, squeeze_factor=0.0)

    # Check she has been correctly identified as being in labour and is sent to the labour ward
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert labour.HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour in hsi_events

    # Clear the queues
    sim.event_queue.queue.clear()
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    # updat key variables
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 0
    df.at[mother_id, 'la_is_postpartum'] = True

    mni[mother_id]['sought_care_for_complication'] = False
    mni[mother_id]['sought_care_labour_phase'] = 'none'
    sim.date = sim.date + pd.DateOffset(days=5)

    # force complication
    params['prob_uterine_atony'] = 1
    params['prob_pph_uterine_atony'] = 1

    # run postpartum home event
    home_birth_pp = labour.PostpartumLabourAtHomeEvent(individual_id=mother_id, module=sim.modules['Labour'])
    home_birth_pp.apply(mother_id)

    assert mni[mother_id]['sought_care_for_complication']
    assert (mni[mother_id]['sought_care_labour_phase'] == 'postpartum')

    # Check the correct scheduling sequence occurs for women seeking care after birth via
    # HSI_GenericEmergencyFirstApptAtFacilityLevel1
    from tlo.methods.hsi_generic_first_appts import (HSI_GenericEmergencyFirstApptAtFacilityLevel1)
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert HSI_GenericEmergencyFirstApptAtFacilityLevel1 in hsi_events

    emergency_appt = HSI_GenericEmergencyFirstApptAtFacilityLevel1(person_id=mother_id,module=sim.modules['Labour'])
    emergency_appt.apply(person_id=mother_id, squeeze_factor=0.0)

    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert labour.HSI_Labour_ReceivesSkilledBirthAttendanceFollowingLabour in hsi_events


def test_event_scheduling_for_labour_onset_and_facility_delivery():
    pass


def test_event_scheduling_for_admissions_from_antenatal_inpatient_ward():
    pass


def test_application_of_risk_of_complications_in_intrapartum_and_postpartum_phases():
    """Test that functions which apply risk of complications correctly set complication properties when risk is set
    to 1"""
    sim = register_modules(ignore_cons_constraints=False)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    set_pregnancy_characteristics(sim, mother_id)

    df = sim.population.props
    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)

    # Store additional data in mni used within the function calls
    additional_keys = {'delivery_setting': 'hospital',
                       'received_blood_transfusion': False,
                       'mode_of_delivery': 'vaginal',
                       'clean_birth_practices': False,
                       'abx_for_prom_given': False,
                       'amtsl_given': False}

    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id].update(additional_keys)

    # Force all complications and preceding complications to occur
    params = sim.modules['Labour'].parameters
    params['prob_cephalopelvic_dis'] = 1
    params['prob_malposition'] = 1
    params['prob_malpresentation'] = 1
    params['prob_obstruction_cpd'] = 1
    params['prob_obstruction_malpos'] = 1
    params['prob_obstruction_malpres'] = 1
    params['prob_placental_abruption_during_labour'] = 1
    params['prob_aph_placental_abruption_labour'] = 1
    params['prob_chorioamnionitis_ip'] = 1
    params['prob_other_maternal_infection_ip'] = 1
    params['prob_sepsis_chorioamnionitis'] = 1
    params['prob_sepsis_chorioamnionitis'] = 1

    # todo: uterine rupture is odds, cant be forced to be one

    # Call the function responsible to applying risk of complications
    for complication in ['cephalopelvic_dis', 'malpresentation', 'malposition', 'obstructed_labour',
                         'placental_abruption', 'antepartum_haem', 'chorioamnionitis',
                         'other_maternal_infection', 'sepsis', 'uterine_rupture']:
        sim.modules['Labour'].set_intrapartum_complications(mother_id, complication=complication)

    # Check that this mother is experiencing all complications as she should
    assert sim.modules['Labour'].cause_of_obstructed_labour.has_all(mother_id, 'cephalopelvic_dis')
    assert sim.modules['Labour'].cause_of_obstructed_labour.has_all(mother_id, 'malpresentation')
    # todo: this should be either or
    assert sim.modules['Labour'].cause_of_obstructed_labour.has_all(mother_id, 'malposition')
    assert df.at[mother_id, 'la_obstructed_labour']
    assert df.at[mother_id, 'la_placental_abruption']
    assert (df.at[mother_id, 'la_antepartum_haem'] != 'none')
    assert sim.modules['Labour'].intrapartum_infections.has_all(mother_id, 'chorioamnionitis')
    assert sim.modules['Labour'].intrapartum_infections.has_all(mother_id, 'other_maternal_infection')
    assert df.at[mother_id, 'la_sepsis']

    # todo: uterine rupture

    # Now repeat this process checking the application of risk of complications in the immediate postpartum period
    params['prob_endometritis_pp'] = 1
    params['prob_urinary_tract_inf_pp'] = 1
    params['prob_skin_soft_tissue_inf_pp'] = 1
    params['prob_other_maternal_infection_pp'] = 1
    params['prob_sepsis_endometritis'] = 1
    params['prob_sepsis_urinary_tract_inf'] = 1
    params['prob_sepsis_skin_soft_tissue_inf'] = 1
    params['prob_sepsis_other_maternal_infection_pp'] = 1
    params['prob_uterine_atony'] = 1
    params['prob_lacerations'] = 1
    params['prob_retained_placenta'] = 1
    params['prob_other_pph_cause'] = 1
    params['prob_pph_uterine_atony'] = 1
    params['prob_pph_lacerations'] = 1
    params['prob_pph_retained_placenta'] = 1
    params['prob_pph_other_causes'] = 1

    for complication in ['endometritis', 'urinary_tract_inf', 'skin_soft_tissue_inf', 'other_maternal_infection',
                         'sepsis', 'uterine_atony', 'lacerations', 'retained_placenta', 'other_pph_cause',
                         'postpartum_haem']:
        sim.modules['Labour'].set_postpartum_complications(mother_id, complication=complication)

    assert sim.modules['Labour'].postpartum_infections.has_all(mother_id, 'endometritis')
    assert sim.modules['Labour'].postpartum_infections.has_all(mother_id, 'urinary_tract_inf')
    assert sim.modules['Labour'].postpartum_infections.has_all(mother_id, 'skin_soft_tissue_inf')
    assert sim.modules['Labour'].postpartum_infections.has_all(mother_id, 'other_maternal_infection')
    assert df.at[mother_id, 'la_sepsis_pp']
    assert sim.modules['Labour'].cause_of_primary_pph.has_all(mother_id, 'uterine_atony')
    assert sim.modules['Labour'].cause_of_primary_pph.has_all(mother_id, 'lacerations')
    assert sim.modules['Labour'].cause_of_primary_pph.has_all(mother_id, 'retained_placenta')
    assert sim.modules['Labour'].cause_of_primary_pph.has_all(mother_id, 'other_pph_cause')
    assert df.at[mother_id, 'la_postpartum_haem']


def test_run_health_system_high_squeeze():
    """This test runs a simulation in which the contents of scheduled HSIs will not be performed because the squeeze
    factor is too high. Therefore it tests the logic in the did_not_run functions of the Labour HSIs to ensure women
    who want to deliver in a facility, but cant, due to lacking capacity, have the correct events scheduled to continue
    their labour"""
    pass


@pytest.mark.group2
def test_run_health_system_events_wont_run():
    """This test runs a simulation in which no scheduled HSIs will run.. Therefore it tests the logic in the
    not_available functions of the Labour HSIs to ensure women who want to deliver in a facility, but cant, due to the
    service being unavailble, have the correct events scheduled to continue their labour"""
    pass

def test_custom_linear_models():
    pass
    """sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*']),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.start_date + pd.DateOffset(days=0))

    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]

    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'la_due_date_current_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 37
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date - pd.DateOffset(months=9)

    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)

    labour_onset = labour.LabourOnsetEvent(module=sim.modules['Labour'], individual_id=mother_id)
    labour_onset.apply(mother_id)

    params = sim.modules['Labour'].parameters
    params['la_labour_equations']['predict_chorioamnionitis_ip'].predict(
        df.loc[[mother_id]])[mother_id] """


# todo: test event scheduling in all different methiods
