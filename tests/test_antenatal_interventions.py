import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
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
    symptommanager,
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


def register_all_modules(seed):
    """Register all modules that are required for ANC to run"""

    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           cons_availability='all'),  # went set disable=true, cant check HSI queue,
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),

                 hiv.DummyHivModule())

    return sim


def find_and_return_events_list(sim, individual_id):
    """Returns event list for an individual"""
    events = sim.find_events_for_person(person_id=individual_id)
    events = [e.__class__ for d, e in events]
    return events


def find_and_return_hsi_events_list(sim, individual_id):
    """Returns HSI event list for an individual"""
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=individual_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    return hsi_events


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


def run_sim_for_0_days_get_mother_id(sim):
    """Run a simulation for 0 days and select a mother_id from dataframe"""
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)

    return mother_id


def test_perfect_run_of_anc_contacts_no_constraints(seed):
    """This test calls all 8 of the ANC contacts for a relevant woman and tests that sequential daisy-chain event
    scheduling is happening correctly and that (when no quality or consumable constraints are applied) women receive all
    the correct screening and medication-based interventions during ANC (at the correct gestational age).
    """

    # Register the modules called within ANC and run the simulation for one day
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           cons_availability='all'),  # went set disable=true, cant check HSI queue,
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 malaria.Malaria(resourcefilepath=resourcefilepath),
                 epi.Epi(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath))

    sim.make_initial_population(n=100)

    params = sim.modules['CareOfWomenDuringPregnancy'].current_parameters
    params_dep = sim.modules['Depression'].parameters

    # Set sensitivity/specificity of dx_tests to one
    params_dep['sensitivity_of_assessment_of_depression'] = 1.0
    params['sensitivity_bp_monitoring'] = 1.0
    params['specificity_bp_monitoring'] = 1.0
    params['sensitivity_urine_protein_1_plus'] = 1.0
    params['specificity_urine_protein_1_plus'] = 1.0
    params['sensitivity_poc_hb_test'] = 1.0
    params['specificity_poc_hb_test'] = 1.0
    params['sensitivity_fbc_hb_test'] = 1.0
    params['specificity_fbc_hb_test'] = 1.0
    params['sensitivity_blood_test_glucose'] = 1.0
    params['specificity_blood_test_glucose'] = 1.0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # Select a woman from the dataframe of reproductive age
    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]
    updated_mother_id = int(mother_id)

    # Set key pregnancy variables
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'date_of_last_pregnancy'] = start_date
    df.at[mother_id, 'ps_anc4'] = True
    df.at[mother_id, 'ps_date_of_anc1'] = start_date + pd.DateOffset(weeks=8)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 10
    df.at[mother_id, 'li_bmi'] = 1
    df.at[mother_id, 'la_parity'] = 0
    df.at[mother_id, 'ps_prev_gest_diab'] = True

    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)

    # Set some complications that should be be detected in ANC leading to further action
    df.at[mother_id, 'ps_htn_disorders'] = 'mild_pre_eclamp'
    df.at[mother_id, 'de_depr'] = True

    # ensure care seeking will continue for all ANC visits
    params = sim.modules['CareOfWomenDuringPregnancy'].current_parameters
    params['prob_seek_anc2'] = 1.0
    params['prob_seek_anc3'] = 1.0
    params['prob_seek_anc5'] = 1.0
    params['prob_seek_anc6'] = 1.0
    params['prob_seek_anc7'] = 1.0
    params['prob_seek_anc8'] = 1.0

    # Set parameters used to determine if HCW will deliver intervention (if consumables available) to 1
    params['prob_intervention_delivered_urine_ds'] = 1.0
    params['prob_intervention_delivered_bp'] = 1.0
    params['prob_intervention_delivered_depression_screen'] = 1.0
    params['prob_intervention_delivered_ifa'] = 1.0
    params['prob_intervention_delivered_gdm_test'] = 1.0
    params['prob_adherent_ifa'] = 1.0
    params['prob_intervention_delivered_bep'] = 1.0
    params['prob_intervention_delivered_llitn'] = 1.0
    params['prob_intervention_delivered_iptp'] = 1.0
    params['prob_intervention_delivered_hiv_test'] = 1.0
    params['prob_intervention_delivered_poct'] = 1.0
    params['prob_intervention_delivered_tt'] = 1.0
    params['sensitivity_blood_test_glucose'] = 1.0
    params['specificity_blood_test_glucose'] = 1.0

    # set coverage of tetatnus for her district to 1
    sim.modules['Epi'].parameters['district_vaccine_coverage']['TT2+'] = 1.0

    # Register the anc HSIs
    first_anc = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    second_anc = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    third_anc = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    fourth_anc = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    fifth_anc = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    sixth_anc = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    seventh_anc = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    eight_anc = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)

    # ========================================== ANC 1 TESTING =======================================================
    # Here we test that during the first ANC contact, women receive the correct interventions
    sim.date = sim.date + pd.DateOffset(weeks=8)

    # Check that this HSI will schedule the next HSI at the right gestation in the contact scheduled for this woman
    recommended_ga_at_next_visit = \
        sim.modules['CareOfWomenDuringPregnancy'].determine_gestational_age_for_next_contact(mother_id)
    assert (recommended_ga_at_next_visit == 20)

    # Run the event and check that the visit has been recorded in the dataframe along with the date of the next visit
    first_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 1)

    # Next, ensure that this woman has received key interventions that should have been started at ANC1: iron & folic
    # acid supplementation, balanced energy and protein supplementation
    assert (df.at[mother_id, 'ac_receiving_iron_folic_acid'])
    assert (df.at[mother_id, 'ac_receiving_bep_supplements'])

    # We would expect som additional HSI events to have been scheduled for a woman on her first ANC and for this woman
    # specifically due to her complications (pre-eclampsia and depression)
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)

    # Should should have undergone depression screening, and then (via the depression module) been diagnosed and
    # started on treatment
    assert (df.at[mother_id, 'de_ever_diagnosed_depression'])
    assert depression.HSI_Depression_TalkingTherapy in hsi_events
    assert depression.HSI_Depression_Start_Antidepressant in hsi_events

    # Additionally she should be scheduled to receive tetanus vaccination
    assert epi.HSI_TdVaccine in hsi_events

    # Additionally she would be scheduled to undergo HIV testing as part of her ANC appointment
    assert hiv.HSI_Hiv_TestAndRefer in hsi_events

    # And finally, as she has pre-eclampsia, this should have been detected via screening and she should have been
    # admitted for treatment (and that DALY onset is stored)
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare in hsi_events
    assert (sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['hypertension_onset'] == sim.date)

    # We then check that the next ANC event in the schedule has been scheduled (and at the correct time)
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact in hsi_events
    assert (sim.population.props.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=10)))

    # =============================================== ANC 2 TESTING ==================================================
    # Move date of sim to the date next visit should occur and update gestational age so event will run
    sim.date = sim.date + pd.DateOffset(weeks=10)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 10

    # Clear the event queue
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    # Assume she is now on treatment for her hypertension (via inpatient ward)
    df.at[mother_id, 'ac_gest_htn_on_treatment'] = True

    # Run the event and check the visit number is stored
    second_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 2)

    # Now check she has received interventions specific to visit 2, calcium supplementation, the first dose of IPTP and
    # second dose of tetanus
    assert (df.at[mother_id, 'ac_receiving_calcium_supplements'])

    # Here, we check that a woman with a pre-exsisting condition (pre-eclampsia) who is on treatment is not readmitted
    # if her condition has remained static (not progressed)
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare not in hsi_events

    # Check scheduling for second tetanus vaccine
    assert epi.HSI_TdVaccine in hsi_events

    # Check scheduling of the next ANC contact
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact in hsi_events
    assert (sim.population.props.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=6)))

    # =============================================== ANC 3 TESTING ==================================================
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    sim.date = sim.date + pd.DateOffset(weeks=6)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 6

    # we set this woman to have developed gestational diabetes (and to have a risk factor for GDM, which should trigger
    # screening)
    df.at[mother_id, 'ps_gest_diab'] = 'uncontrolled'
    df.at[mother_id, 'ps_prev_gest_diab'] = True
    df.at[mother_id, 'li_bmi'] = 4

    third_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 3)

    # Check that this woman has undergone screening for diabetes, and will be admitted for treatment
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare in hsi_events
    assert (sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['gest_diab_onset'] == sim.date)

    # Check scheduling of the next ANC contact
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact in hsi_events
    assert (sim.population.props.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=4)))

    # =============================================== ANC 4 TESTING ==================================================
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    sim.date = sim.date + pd.DateOffset(weeks=4)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 4

    # Set that her diabetes is now controlled by treatment
    df.at[mother_id, 'ps_gest_diab'] = 'controlled'
    df.at[mother_id, 'ac_gest_diab_on_treatment'] = 'insulin'

    # Run the HSI
    fourth_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 4)

    hsi_events = find_and_return_hsi_events_list(sim, mother_id)

    # Check scheduling of the next ANC contact
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact in hsi_events
    assert (sim.population.props.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=4)))

    # =============================================== ANC 5 TESTING ==================================================
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    sim.date = sim.date + pd.DateOffset(weeks=4)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 4

    # The woman has experienced progression of her disease between appointments
    df.at[mother_id, 'ps_htn_disorders'] = 'severe_pre_eclamp'
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['new_onset_spe'] = True

    fifth_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 5)

    # Check her severe pre-eclampsia has correctly been identified and she is scheduled to be admitted as an inpatient
    # (despite already being on antihypertensives)
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare in hsi_events

    # Check scheduling of the next ANC contact
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact in hsi_events
    assert (df.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=2)))

    # =============================================== ANC 6 TESTING ==================================================
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    sim.date = sim.date + pd.DateOffset(weeks=2)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 2

    # Set that she has developed severe anaemia between appointments
    df.at[mother_id, 'ps_htn_disorders'] = 'mild_pre_eclamp'
    df.at[mother_id, 'ps_anaemia_in_pregnancy'] = 'severe'

    sixth_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 6)

    # Check her severe anaemia has correctly been identified via point of care screening (occurs at visit 2 and visit 6)
    # and she is scheduled to be admitted as an inpatient for further treatment
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare in hsi_events

    # Check scheduling of the next ANC contact
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact in hsi_events
    assert (df.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=2)))

    # =============================================== ANC 7 TESTING ==================================================
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    sim.date = sim.date + pd.DateOffset(weeks=2)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 2
    df.at[mother_id, 'ps_anaemia_in_pregnancy'] = 'none'

    seventh_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 7)

    # Check scheduling of the next ANC contact
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)

    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact in hsi_events
    assert (df.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=2)))

    # =============================================== ANC 8 TESTING ==================================================
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    sim.date = sim.date + pd.DateOffset(weeks=2)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 2

    # Run the event and check its counted- no further interventions delivered or events scheduled
    eight_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 8)

    # TODO: test that TB screening is happening correctly (not coded in TB yet)
    # TODO: test that other blood tests are occuring (hep b and syphilis- currently not linked to anything)
    # todo: test with probabilities low/0? same with dx test?


def test_anc_contacts_that_should_not_run_wont_run(seed):
    """This test checks the inbuilt functions within ANC1 and ANC subsequent that should block, and in some cases
     reschedule, ANC visits for women when the HSI runs and is no longer appropriate for them """

    # Register the key modules and run the simulation for one day
    sim = register_all_modules(seed)
    mother_id = run_sim_for_0_days_get_mother_id(sim)

    params = sim.modules['CareOfWomenDuringPregnancy'].current_parameters
    params['prob_seek_anc2'] = 1.0
    params['prob_seek_anc3'] = 1.0
    params['prob_seek_anc5'] = 1.0
    params['prob_seek_anc6'] = 1.0
    params['prob_seek_anc7'] = 1.0
    params['prob_seek_anc8'] = 1.0

    # Set key pregnancy variables
    df = sim.population.props
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'date_of_last_pregnancy'] = start_date
    df.at[mother_id, 'ps_date_of_anc1'] = start_date + pd.DateOffset(weeks=8)
    sim.modules['Labour'].set_date_of_labour(mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)

    updated_mother_id = int(mother_id)

    # define HSIs
    first_anc = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    second_anc = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)

    sim.date = start_date + pd.DateOffset(weeks=8)

    # ----------------------------------------- ANC 1 ----------------------------------------------------------------
    # Set mother to being currently in labour
    df.at[mother_id, 'la_currently_in_labour'] = True

    # Check HSI has not ran and another ANC is not scheduled (as she will no longer be pregnant)
    first_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 0)
    assert pd.isnull(df.at[mother_id, 'ac_date_next_contact'])

    # Reset labour variable and set gestational age to very young- ANC likely scheduled from last pregnancy and woman
    # is now pregnant again. Therefore event shouldn't run
    df.at[mother_id, 'la_currently_in_labour'] = False
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 4

    first_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 0)
    assert pd.isnull(df.at[mother_id, 'ac_date_next_contact'])

    # Reset the gestational age and set the squeeze factor of the HSI as very high. Woman will leave and HSI should not
    # run
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 10

    first_anc.apply(person_id=updated_mother_id, squeeze_factor=1001)  # todo: replace
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 0)
    assert pd.isnull(df.at[mother_id, 'ac_date_next_contact'])

    # check that she will return for this event
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact in hsi_events

    # Finally set woman as inpatient when she is due for her first ANC appointment
    df.at[mother_id, 'hs_is_inpatient'] = True
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    # Check that ANC hasnt ran BUT woman has correctly been scheduled to return for ANC 1 at the next gestational age
    # in the schedule
    first_anc.apply(person_id=updated_mother_id, squeeze_factor=0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 0)
    assert pd.isnull(df.at[mother_id, 'ac_date_next_contact'])

    date_event = check_event_queue_for_event_and_return_scheduled_event_date(
        sim, queue_of_interest='hsi', individual_id=updated_mother_id,
        event_of_interest=care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact)

    assert date_event == (sim.date + pd.DateOffset(weeks=10))
    assert (df.at[mother_id, 'ps_date_of_anc1'] == (sim.date + pd.DateOffset(weeks=10)))

    # ----------------------------------- SUBSEQUENT ANC CONTACTS ----------------------------------------------------
    # Next we check that subsequent ANC contacts wont run when they shouldn't

    # Clear the event queue
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    # Set this woman to be an inpatient
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 20
    df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] = 1
    df.at[mother_id, 'hs_is_inpatient'] = True

    # Run the event and check its hasnt ran as expected
    second_anc.apply(person_id=updated_mother_id, squeeze_factor=0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 1)

    # check the event has been rescheduled for the next gestational age in the schedule
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=updated_mother_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact in hsi_events
    assert not pd.isnull(df.at[mother_id, 'ac_date_next_contact'])

    # todo: expand this test


def test_daisy_chain_care_seeking_logic_to_ensure_certain_number_of_contact(seed):
    """This test checks the logic around care seeking for the next ANC visit in the schedule. We test that women who are
     predicited at least 4 visits are automatically scheduled the next visit in the schedule (if that visit number is
    below 4). We also test that women who are not predicted 4 or more visits will seek care based on the value of a
    care seeking parameter"""
    sim = register_all_modules(seed=seed)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    updated_mother_id = int(mother_id)

    df = sim.population.props
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 10

    # This woman has been determined to attend at least four visits
    df.at[mother_id, 'ps_anc4'] = True

    # call the function called by all ANC HSIs to scheduled the next visit
    sim.modules['CareOfWomenDuringPregnancy'].antenatal_care_scheduler(individual_id=updated_mother_id,
                                                                       visit_to_be_scheduled=2,
                                                                       recommended_gestation_next_anc=20)

    # As this woman is set to attend at least 4 ANC contacts, the next visit should be correctly sheduled at the correct
    # gestational age
    date_event = check_event_queue_for_event_and_return_scheduled_event_date(
        sim, queue_of_interest='hsi', individual_id=updated_mother_id,
        event_of_interest=care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact)

    assert date_event == (sim.date + pd.DateOffset(weeks=10))
    assert (df.at[mother_id, 'ac_date_next_contact'] == date_event)

    # Clear the event queue and reset 'ps_anc4'
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    df.at[mother_id, 'ac_date_next_contact'] = pd.NaT
    df.at[mother_id, 'ps_anc4'] = False

    # For women who are not predetermined to attend at least 4 anc contact we use a probability to determine if they
    # will return for the next visit. Here we set that to 0 and check that no event has been sheduled
    params = sim.modules['CareOfWomenDuringPregnancy'].current_parameters
    params['prob_seek_anc3'] = 0

    sim.modules['CareOfWomenDuringPregnancy'].antenatal_care_scheduler(individual_id=updated_mother_id,
                                                                       visit_to_be_scheduled=3,
                                                                       recommended_gestation_next_anc=26)

    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=updated_mother_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact not in hsi_events


def test_initiation_of_treatment_for_maternal_anaemia_during_antenatal_inpatient_care(seed):
    """Test the treatment delivered to women who are admitted to the antenatal ward with anaemia of differing
    severities"""
    sim = register_all_modules(seed)
    sim.make_initial_population(n=100)

    # Set DxTest parameters to 1 to ensure anaemia is detected correctly
    params = sim.modules['CareOfWomenDuringPregnancy'].current_parameters
    params['sensitivity_fbc_hb_test'] = 1.0
    params['specificity_fbc_hb_test'] = 1.0

    # Set treatment parameters to 1
    params['treatment_effect_blood_transfusion_anaemia'] = 1.0
    params['prob_intervention_delivered_ifa'] = 1.0
    params['prob_adherent_ifa'] = 1.0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # Select a woman from the dataframe of reproductive age
    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]
    updated_mother_id = int(mother_id)

    # set key pregnancy characteristics
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 22
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id] = {'severe_anaemia_resolution': pd.NaT,
                                                                             'delay_one_two': False,
                                                                             'delay_three': False}

    # Set anaemia status
    df.at[mother_id, 'ps_anaemia_in_pregnancy'] = 'severe'

    # define inpatient treatment and run the event
    inpatient_hsi = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    inpatient_hsi.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check that the woman has correctly been treated via blood transfusion and is no longer anaemic (and the correct
    # date of resolution for her condition has been stored)
    assert (df.at[mother_id, 'ps_anaemia_in_pregnancy'] == 'none')
    assert (sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id][
                'severe_anaemia_resolution'] == sim.date)

    # Check that she has now been started on regular iron and folic acid to reduce risk of future anaemia episodes
    assert df.at[mother_id, 'ac_receiving_iron_folic_acid']

    # And finally check she has been scheduled to return for follow up testing in 1 months time
    date_event = check_event_queue_for_event_and_return_scheduled_event_date(
        sim, queue_of_interest='hsi', individual_id=updated_mother_id,
        event_of_interest=care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfAnaemia)  # noqa: E501

    assert date_event == (sim.date + pd.DateOffset(days=28))

    # Clear the event queue
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    # The womans anaemia has returned between inpatient care and outpatient care, she has ANC visit in 2 weeks time
    df.at[mother_id, 'ps_anaemia_in_pregnancy'] = 'moderate'
    df.at[mother_id, 'ac_date_next_contact'] = sim.date + pd.DateOffset(weeks=2)

    # Run the outpatient appointment
    outpatient_check = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfAnaemia(  # noqa: E501
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    outpatient_check.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check that she has been correctly identified as anaemic again and will be readmitted for treatment
    date_event = check_event_queue_for_event_and_return_scheduled_event_date(
        sim, queue_of_interest='hsi', individual_id=updated_mother_id,
        event_of_interest=care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare)
    assert date_event == sim.date

    # Because she is due to have ANC in 2 weeks we check here that has been correctly identified and that she has not
    # been scheduled for outpatient care (as screening for anaemia will happen as part of routine treatment)
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfAnaemia not in hsi_events  # noqa: E501


def test_initiation_of_treatment_for_hypertensive_disorder_during_antenatal_inpatient_care(seed):
    """Test that the correct treatment is delivered to women with differing severities of hypertensive disorder when
    admitted to inpatient ward"""
    sim = register_all_modules(seed)
    mother_id = run_sim_for_0_days_get_mother_id(sim)

    updated_mother_id = int(mother_id)

    # set key parameters
    params = sim.modules['Labour'].current_parameters
    params['mean_hcw_competence_hc'] = 1
    params['mean_hcw_competence_hp'] = 1
    params['prob_hcw_avail_anticonvulsant'] = 1

    # set key pregnancy characteristics
    df = sim.population.props
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 22

    # Set hypertension status
    df.at[mother_id, 'ps_htn_disorders'] = 'mild_pre_eclamp'

    # define inpatient treatment and run the event
    inpatient_hsi = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    inpatient_hsi.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check that she has been started on treatment and that no futher HSIs have been scheduled (i.e. her treatment is
    # complete)
    assert df.at[mother_id, 'ac_gest_htn_on_treatment']
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert len(hsi_events) == 0

    # Set hypertension status to severe gestational hypertension and run the event
    df.at[mother_id, 'ps_htn_disorders'] = 'severe_gest_htn'
    inpatient_hsi.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check treatment has been given and the womans hypertension has been set to mild
    assert (df.at[mother_id, 'ps_htn_disorders'] == 'gest_htn')

    # Set hypertension status to severe pre-eclampsia and run the event
    df.at[mother_id, 'ps_htn_disorders'] = 'severe_pre_eclamp'
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 32
    inpatient_hsi.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check the correct treatment has been given
    assert (df.at[mother_id, 'ac_mag_sulph_treatment'])
    assert (df.at[mother_id, 'ac_iv_anti_htn_treatment'])

    # Check that this woman has been correctly marked as admission for assisted delivery of some form
    assert df.at[mother_id, 'ac_admitted_for_immediate_delivery'] in ('induction_now', 'avd_now', 'caesarean_now')

    # Ensure the correct labour event is scheduled
    events = find_and_return_events_list(sim, mother_id)
    assert labour.LabourOnsetEvent in events


def test_initiation_of_treatment_for_gestational_diabetes_during_antenatal_inpatient_care(seed):
    """Test that the correct treatment is delivered to women with gestational diabetes when
    admitted to inpatient ward"""
    sim = register_all_modules(seed)
    mother_id = run_sim_for_0_days_get_mother_id(sim)

    updated_mother_id = int(mother_id)

    # set key pregnancy characteristics
    df = sim.population.props
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 22
    df.at[mother_id, 'ps_gest_diab'] = 'uncontrolled'

    # define inpatient treatment and run the event
    inpatient_hsi = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    inpatient_hsi.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check that she is now being treated with diet/exercise and her diabetes is assumed to be controlled
    assert (df.at[mother_id, 'ps_gest_diab'] == 'controlled')
    assert (df.at[mother_id, 'ac_gest_diab_on_treatment'] == 'diet_exercise')

    # look at both the events and hsi events queues
    events = find_and_return_events_list(sim, mother_id)
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)

    # Check bother the glycaemic control event and outpatient follow up events are scheduled
    assert pregnancy_supervisor.GestationalDiabetesGlycaemicControlEvent in events
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfGestationalDiabetes in hsi_events  # noqa: E501

    # Clear the event queue
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    # Set the probability that diet and exercise alone will controll this womans diabetes to 0
    params = sim.modules['PregnancySupervisor'].current_parameters
    params['prob_glycaemic_control_diet_exercise'] = 0.0

    # Run the event
    glyc_event = pregnancy_supervisor.GestationalDiabetesGlycaemicControlEvent(
        individual_id=mother_id, module=sim.modules['PregnancySupervisor'])
    glyc_event.apply(mother_id)

    # Check her diabetes is stored as uncontrolled
    assert (df.at[mother_id, 'ps_gest_diab'] == 'uncontrolled')

    # Define the outpatient HSI and run
    outpatient_hsi = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfGestationalDiabetes(  # noqa: E501
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    outpatient_hsi.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check she has been correctly started on the next treatment and her diabetes is assumed to be controlled
    assert (df.at[mother_id, 'ps_gest_diab'] == 'controlled')
    assert (df.at[mother_id, 'ac_gest_diab_on_treatment'] == 'orals')

    # Check the events again
    events = find_and_return_events_list(sim, mother_id)
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)

    assert pregnancy_supervisor.GestationalDiabetesGlycaemicControlEvent in events
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfGestationalDiabetes in hsi_events  # noqa: E501

    # Set the probability that oral medication alone will control this womans diabetes to 0
    params['prob_glycaemic_control_orals'] = 0.0

    # run the event and check
    glyc_event.apply(mother_id)
    assert (df.at[mother_id, 'ps_gest_diab'] == 'uncontrolled')

    # run the next follow up HSI and perform the checks
    outpatient_hsi.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ps_gest_diab'] == 'controlled')
    assert (df.at[mother_id, 'ac_gest_diab_on_treatment'] == 'insulin')
    events = find_and_return_events_list(sim, mother_id)
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert pregnancy_supervisor.GestationalDiabetesGlycaemicControlEvent in events
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfGestationalDiabetes in hsi_events  # noqa: E501


def test_initiation_of_treatment_for_prom_with_or_without_chorioamnionitis_during_antenatal_inpatient_care(seed):
    """Test that the correct treatment is delivered to women with PROM +/- chorioamnionitis when admitted to inpatient
    ward"""
    sim = register_all_modules(seed)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    updated_mother_id = int(mother_id)

    # set key parameters
    params = sim.modules['Labour'].current_parameters
    params['mean_hcw_competence_hc'] = 1
    params['mean_hcw_competence_hp'] = 1
    params['prob_hcw_avail_iv_abx'] = 1

    # set key pregnancy characteristics
    df = sim.population.props
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 22

    # set complication
    df.at[mother_id, 'ps_premature_rupture_of_membranes'] = True
    df.at[mother_id, 'ps_chorioamnionitis'] = False

    # define inpatient treatment and run the event
    inpatient_hsi = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    inpatient_hsi.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check that antibiotics have been given
    assert df.at[mother_id, 'ac_received_abx_for_prom']

    # Check that the mother is correctly scheduled for induction when she is the correct gestational age
    assert (df.at[mother_id, 'ac_admitted_for_immediate_delivery'] == 'induction_future')

    date_event = check_event_queue_for_event_and_return_scheduled_event_date(
        sim, queue_of_interest='event', individual_id=updated_mother_id, event_of_interest=labour.LabourOnsetEvent)

    days_until_safe_for_cs = int((37 * 7) - (df.at[mother_id, 'ps_gestational_age_in_weeks'] * 7))
    assert date_event == (sim.date + pd.DateOffset(days=days_until_safe_for_cs))

    # Clear the event queues
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    # Set the woman to have an infection and run the event
    df.at[mother_id, 'ps_chorioamnionitis'] = True
    inpatient_hsi.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check she has received treatment and has correctly been scheduled for immediate delivery
    assert (df.at[mother_id, 'ac_admitted_for_immediate_delivery'] == 'induction_now')

    date_event = check_event_queue_for_event_and_return_scheduled_event_date(
        sim, queue_of_interest='event', individual_id=updated_mother_id, event_of_interest=labour.LabourOnsetEvent)
    assert date_event == sim.date


def test_initiation_of_treatment_for_antepartum_haemorrhage_during_antenatal_inpatient_care(seed):
    """Test that the correct treatment is delivered to women with antenatal haemorrhage when admitted to inpatient
    ward"""
    sim = register_all_modules(seed)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    updated_mother_id = int(mother_id)

    # set key pregnancy characteristics
    df = sim.population.props
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 22

    # set complication properties
    df.at[mother_id, 'ps_antepartum_haemorrhage'] = 'severe'
    df.at[mother_id, 'ps_placental_abruption'] = True
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)

    # define inpatient HSI and run in
    inpatient_hsi = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    inpatient_hsi.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check that any woman with placental abruption is correctly scheduled for immediate caesarean regardless of
    # gestation
    assert (df.at[mother_id, 'ac_admitted_for_immediate_delivery'] == 'caesarean_now')

    # Check the labour even is correctly scheduled for today
    date_event = check_event_queue_for_event_and_return_scheduled_event_date(
        sim, queue_of_interest='event', individual_id=updated_mother_id, event_of_interest=labour.LabourOnsetEvent)
    assert date_event == sim.date

    # Clear the event queues
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    # Reset abruption variable
    df.at[mother_id, 'ps_placental_abruption'] = False

    # Reduce severity of bleeding and set praevia variable, run the HSI
    df.at[mother_id, 'ps_placenta_praevia'] = True
    inpatient_hsi.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check woman is correctly scheduled for immediate caesarean due to severity of her bleeding
    assert (df.at[mother_id, 'ac_admitted_for_immediate_delivery'] == 'caesarean_now')
    date_event = check_event_queue_for_event_and_return_scheduled_event_date(
        sim, queue_of_interest='event', individual_id=updated_mother_id, event_of_interest=labour.LabourOnsetEvent)
    assert date_event == sim.date

    # Clear the event queues
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    # Set bleeding severity to mild/moderate and change gestation
    df.at[mother_id, 'ps_antepartum_haemorrhage'] = 'mild_moderate'
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 30

    # Run the event
    inpatient_hsi.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check she is correctly scheduled the labour onset event when her gestation is 37 weeks
    assert (df.at[mother_id, 'ac_admitted_for_immediate_delivery'] == 'caesarean_future')
    date_event = check_event_queue_for_event_and_return_scheduled_event_date(
        sim, queue_of_interest='event', individual_id=updated_mother_id, event_of_interest=labour.LabourOnsetEvent)
    days_until_safe_for_cs = int((37 * 7) - (df.at[mother_id, 'ps_gestational_age_in_weeks'] * 7))
    assert date_event == (sim.date + pd.DateOffset(days=days_until_safe_for_cs))

    # Clear the event queue
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.event_queue.queue.clear()

    # Set a later gestation and run the event
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 38
    inpatient_hsi.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check she is correctly scheduled for caesarean delivery as her gestation is greater
    assert (df.at[mother_id, 'ac_admitted_for_immediate_delivery'] == 'caesarean_now')
    date_event = check_event_queue_for_event_and_return_scheduled_event_date(
        sim, queue_of_interest='event', individual_id=updated_mother_id, event_of_interest=labour.LabourOnsetEvent)
    assert date_event == sim.date


def test_scheduling_and_treatment_effect_of_post_abortion_care(seed):
    """Test women who present to the health system with complications of abortion are sent to the correct HSI, receive
    treatment and this reduces their risk of death from abortion complications """
    sim = register_all_modules(seed)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    updated_mother_id = int(mother_id)

    # set key pregnancy characteristics
    df = sim.population.props
    df.at[mother_id, 'is_pregnant'] = False

    # set key parameters
    params = sim.modules['Labour'].current_parameters
    params['mean_hcw_competence_hc'] = 1
    params['mean_hcw_competence_hp'] = 1
    params['prob_hcw_avail_retained_prod'] = 1

    # set complications
    sim.modules['PregnancySupervisor'].abortion_complications.set([mother_id], 'haemorrhage')
    sim.modules['PregnancySupervisor'].abortion_complications.set([mother_id], 'sepsis')
    sim.modules['PregnancySupervisor'].abortion_complications.set([mother_id], 'injury')

    # Import and run HSI_GenericEmergencyFirstApptAtFacilityLevel1, where women with abortion complications are
    # scheduled to present via Pregnancy Supervisor
    from tlo.methods.hsi_generic_first_appts import HSI_GenericEmergencyFirstApptAtFacilityLevel1

    # Run the event
    emergency_appt = HSI_GenericEmergencyFirstApptAtFacilityLevel1(person_id=updated_mother_id,
                                                                   module=sim.modules['PregnancySupervisor'])
    emergency_appt.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check that this event correctly identified abortion complications and scheduled the correct HSI
    hsi_list = find_and_return_hsi_events_list(sim, mother_id)
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement in hsi_list

    # Define and run the Post Abortion Care HSI
    pac = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    pac.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check that this mother receives treatments
    assert df.at[mother_id, 'ac_received_post_abortion_care']

    # Define the death event
    death_event = pregnancy_supervisor.EarlyPregnancyLossDeathEvent(module=sim.modules['PregnancySupervisor'],
                                                                    individual_id=mother_id,
                                                                    cause='spontaneous_abortion')

    # Replicate the death equation from the event so risk of death == 1 but treatment is completely effective
    sim.modules['PregnancySupervisor'].ps_linear_models['spontaneous_abortion_death'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1,
            Predictor('ac_received_post_abortion_care').when(True, 0))

    # Run the death event
    death_event.apply(mother_id)

    # Check that the woman survived thanks to treatment
    assert df.at[mother_id, 'is_alive']

    # Check that the mni will be deleted on the next daly poll and that treatment properties have been reset
    assert (sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['delete_mni'])
    assert not df.at[mother_id, 'ac_received_post_abortion_care']


def test_scheduling_and_treatment_effect_of_ectopic_pregnancy_case_management(seed):
    """Test women who present to the health system with complications of ectopic pregnancy are sent to the correct HSI,
    receive treatment and this reduces their risk of death from abortion complications """
    sim = register_all_modules(seed)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    updated_mother_id = int(mother_id)

    # set key pregnancy characteristics
    df = sim.population.props
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'ps_ectopic_pregnancy'] = 'not_ruptured'
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 8

    # set prob care seeking to 1
    sim.modules['PregnancySupervisor'].current_parameters['prob_care_seeking_ectopic_pre_rupture'] = 1

    # define and run ectopic event
    ectopic_event = pregnancy_supervisor.EctopicPregnancyEvent(individual_id=mother_id,
                                                               module=sim.modules['PregnancySupervisor'])
    ectopic_event.apply(mother_id)

    # Check the woman has correctly sought care via HSI_GenericEmergencyFirstApptAtFacilityLevel1
    from tlo.methods.hsi_generic_first_appts import HSI_GenericEmergencyFirstApptAtFacilityLevel1
    hsi_list = find_and_return_hsi_events_list(sim, mother_id)
    assert HSI_GenericEmergencyFirstApptAtFacilityLevel1 in hsi_list

    # Run the event
    emergency_appt = HSI_GenericEmergencyFirstApptAtFacilityLevel1(person_id=updated_mother_id,
                                                                   module=sim.modules['PregnancySupervisor'])
    emergency_appt.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check she has correctly been scheduled treatment for ectopic pregnancy
    hsi_list = find_and_return_hsi_events_list(sim, mother_id)
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy in hsi_list

    # Define and run treatment events
    ectopic_treatment = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    ectopic_treatment.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check that this event has correctly blocked the ectopic rupture event, and therefore stopped risk of death
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]
    assert pregnancy_supervisor.EctopicPregnancyRuptureEvent not in events

    # Check treatment for ruptured ectopic pregnancy...
    df.at[mother_id, 'ps_ectopic_pregnancy'] = 'ruptured'

    # Check sheduling through generic HSI event (as above)
    from tlo.methods.hsi_generic_first_appts import HSI_GenericEmergencyFirstApptAtFacilityLevel1
    emergency_appt = HSI_GenericEmergencyFirstApptAtFacilityLevel1(person_id=updated_mother_id,
                                                                   module=sim.modules['PregnancySupervisor'])
    emergency_appt.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check that this event correctly identified ectopic complications and scheduled the correct HSI
    hsi_list = find_and_return_hsi_events_list(sim, mother_id)
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy in hsi_list

    # Run ectopic treatment event
    ectopic_treatment = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)
    ectopic_treatment.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # Check treatment is delivered
    assert df.at[mother_id, 'ac_ectopic_pregnancy_treated']
    assert (sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['delete_mni'])

    # And That treatment averts risk of death in death event
    sim.modules['PregnancySupervisor'].ps_linear_models['ectopic_pregnancy_death'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1,
            Predictor('ac_ectopic_pregnancy_treated').when(True, 0))

    # Run the death event
    death_event = pregnancy_supervisor.EarlyPregnancyLossDeathEvent(module=sim.modules['PregnancySupervisor'],
                                                                    individual_id=mother_id,
                                                                    cause='ectopic_pregnancy')
    death_event.apply(mother_id)
    assert (df.at[mother_id, 'ps_ectopic_pregnancy'] == 'none')

    # Check that the woman survived thanks to treatment
    assert df.at[mother_id, 'is_alive']


def test_focused_anc_scheduling(seed):
    """
    Tests scheduling of Focused ANC HSI is occurring as expected.
    """
    def check_hsi_schedule_and_visit_number(vn, mother_id):
        health_system = sim.modules['HealthSystem']
        hsi_events = health_system.find_events_for_person(person_id=mother_id)

        hsi_events_class_list = [e.__class__ for d, e in hsi_events]
        assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FocusedANCVisit in hsi_events_class_list

        hsi_events_other = [e for d, e in hsi_events]
        for e in hsi_events_other:
            if e.__class__ == care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FocusedANCVisit:
                assert e.visit_number == vn

    sim = register_all_modules(seed)
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # Select a woman from the dataframe of reproductive age
    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]
    updated_mother_id = int(mother_id)

    # Set key pregnancy variables
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'date_of_last_pregnancy'] = start_date
    df.at[mother_id, 'ps_anc4'] = True
    df.at[mother_id, 'ps_date_of_anc1'] = start_date + pd.DateOffset(weeks=8)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 10
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)

    # ensure care seeking will continue for all ANC visits
    params = sim.modules['CareOfWomenDuringPregnancy'].current_parameters
    params['prob_seek_anc2'] = 1.0
    params['prob_seek_anc3'] = 1.0

    params_ps = sim.modules['PregnancySupervisor'].current_parameters
    params_ps['anc_service_structure'] = 4

    # Register the anc HSI and apply
    focused_anc = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FocusedANCVisit(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id, visit_number=1)
    focused_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    # check the anc counter property has been updated and MNI keys have been added as expected
    assert df.at[updated_mother_id, 'ac_total_anc_visits_current_pregnancy'] == 1
    assert 'anc_ints' in sim.modules['PregnancySupervisor'].mother_and_newborn_info[updated_mother_id].keys()

    # Ensure that HSI_CareOfWomenDuringPregnancy_FocusedANCVisit has been rescheduled and that the visit number has
    # correctly increased to
    check_hsi_schedule_and_visit_number(2, mother_id)

    # clear event queue and increase gestational age
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    df.at[updated_mother_id, 'ps_gestational_age_in_weeks'] = 22

    # Repeat checks for remaining instances of the HSI in the ANC schedule
    focused_anc = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FocusedANCVisit(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id, visit_number=2)
    focused_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    assert df.at[updated_mother_id, 'ac_total_anc_visits_current_pregnancy'] == 2
    check_hsi_schedule_and_visit_number(3, mother_id)

    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    df.at[updated_mother_id, 'ps_gestational_age_in_weeks'] = 30

    focused_anc = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FocusedANCVisit(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id, visit_number=3)
    focused_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    assert df.at[updated_mother_id, 'ac_total_anc_visits_current_pregnancy'] == 3
    check_hsi_schedule_and_visit_number(4, mother_id)

    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    df.at[updated_mother_id, 'ps_gestational_age_in_weeks'] = 36

    # At the fourth visit ensure that no futher routine ANC is scheduled
    focused_anc = care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FocusedANCVisit(
        module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id, visit_number=4)
    focused_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)

    assert df.at[updated_mother_id, 'ac_total_anc_visits_current_pregnancy'] == 4
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=updated_mother_id)
    hsi_events_class_list = [e.__class__ for d, e in hsi_events]
    assert care_of_women_during_pregnancy.HSI_CareOfWomenDuringPregnancy_FocusedANCVisit not in hsi_events_class_list
