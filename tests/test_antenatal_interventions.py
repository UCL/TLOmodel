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

def register_all_modules_no_consumable_constraints():
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           ignore_cons_constraints=True), #check thats right
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



# todo: another test that just calls the other interventions not called in ANC 1?
# TODO: test inpatient stuff...
# todo: test when probabilities are block
# todo: test when consumables are blocked
# todo: test when women of different gestations arrive at ANC


def find_hsi_events_list(sim, individual_id):
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=individual_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    return hsi_events

def test_perfect_run_of_anc_contacts_no_constraints():
    """This test calls all 8 of the ANC contacts for a relevant woman and tests that sequential daisy-chain event
    scheduling is happening correctly and that (when no quality or consumable constraints are applied) women receive all
    the correct screening and medication-based interventions during ANC (at the correct gestational age).
    """

    # todo: neaten up with functions

    # Register the key modules and run the simulation for one day
    sim = register_all_modules_no_consumable_constraints()
    sim.make_initial_population(n=100)

    params = sim.modules['CareOfWomenDuringPregnancy'].parameters
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
    updated_mother_id = int(mother_id)  # todo: had to force as int as dx_manager doesnt recognise int64

    # Set key pregnancy variables
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'date_of_last_pregnancy'] = start_date
    df.at[mother_id, 'ps_will_attend_four_or_more_anc'] = True
    df.at[mother_id, 'ps_date_of_anc1'] = start_date + pd.DateOffset(weeks=8)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 10
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id] = {'hypertension_onset': pd.NaT,
                                                                             'gest_diab_onset': pd.NaT}

    # Set some complications that should be be detected in ANC leading to futher action
    df.at[mother_id, 'ps_htn_disorders'] = 'mild_pre_eclamp'
    df.at[mother_id, 'de_depr'] = True

    # ensure care seeking will continue for all ANC visits
    params = sim.modules['CareOfWomenDuringPregnancy'].parameters
    params['ac_linear_equations']['anc_continues'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)

    # Set parameters used to determine if HCW will deliver intervention (if consumables available) to 1
    params['prob_intervention_delivered_urine_ds'] = 1
    params['prob_intervention_delivered_bp'] = 1
    params['prob_intervention_delivered_depression_screen'] = 1
    params['prob_intervention_delivered_ifa'] = 1
    params['prob_intervention_delivered_bep'] = 1
    params['prob_intervention_delivered_llitn'] = 1
    params['prob_intervention_delivered_iptp'] = 1
    params['prob_intervention_delivered_hiv_test'] = 1
    params['prob_intervention_delivered_poct'] = 1
    params['prob_intervention_delivered_tt'] = 1

    # Register the anc HSIs
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
    # acid supplementation, balanced energy and protein supplementation, insecticide treated bed net and a first
    # tetanus vaccine
    assert (df.at[mother_id, 'ac_receiving_iron_folic_acid'])
    assert (df.at[mother_id, 'ac_receiving_bep_supplements'])
    assert (df.at[mother_id, 'ac_itn_provided'])
    assert (df.at[mother_id, 'ac_ttd_received'] == 1)

    # We would expect som additional HSI events to have been scheduled for a woman on her first ANC and for this woman
    # spefically due to her complications (pre-eclampsia and depression)
    hsi_events = find_hsi_events_list(sim, mother_id)

    # Should should have undergone depression screening, and then (via the depression module) been diagnosed and
    # started on treatment
    assert (df.at[mother_id, 'de_ever_diagnosed_depression'])
    assert depression.HSI_Depression_TalkingTherapy in hsi_events
    assert depression.HSI_Depression_Start_Antidepressant in hsi_events

    # Additionally she would be scheduled to undergo HIV testing as part of her ANC appointment
    assert hiv.HSI_Hiv_TestAndRefer in hsi_events

    # And finally, as she has pre-eclampsia, this should have been detected via screening and she should have been
    # admitted for treatment (and that DALY onset is stored)
    assert antenatal_care.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare in hsi_events
    # TODO: ask TH r.e. dx_test error
    # assert (sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['hypertension_onset'] == sim.date)

    # We then check that the next ANC event in the schedule has been scheduled (and at the correct time)
    assert antenatal_care.HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact in hsi_events
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
    assert (df.at[mother_id, 'ac_doses_of_iptp_received'] == 1)
    assert (df.at[mother_id, 'ac_ttd_received'] == 2)

    # Here, we check that a woman with a pre-exsisting condition (pre-eclampsia) who is on treatment is not readmitted
    # if her condition has remained static (not progressed)
    hsi_events = find_hsi_events_list(sim, mother_id)
    assert antenatal_care.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare not in hsi_events

    # Check scheduling of the next ANC contact
    assert antenatal_care.HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact in hsi_events
    assert (sim.population.props.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=6)))

    # =============================================== ANC 3 TESTING ==================================================
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    sim.date = sim.date + pd.DateOffset(weeks=6)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 6

    # we set this woman to have developed gestational diabetes (and to have a risk factor for GDM, which should trigger
    # screening)
    df.at[mother_id, 'ps_gest_diab'] = 'uncontrolled'
    df.at[mother_id, 'ps_prev_gest_diab'] = True

    third_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 3)

    # Check the second dose of IPTp is given
    assert (df.at[mother_id, 'ac_doses_of_iptp_received'] == 2)

    # Check that this woman has undergone screening for diabetes, and will be admitted for treatment
    hsi_events = find_hsi_events_list(sim, mother_id)
    assert antenatal_care.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare in hsi_events
    # TODO: ask TH r.e. dx_test error
    # assert (sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['gest_diab_onset'] == sim.date)

    # Check scheduling of the next ANC contact
    assert antenatal_care.HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact in hsi_events
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

    # Check the third dose of IPTp is given
    assert (df.at[mother_id, 'ac_doses_of_iptp_received'] == 3)

    hsi_events = find_hsi_events_list(sim, mother_id)

    # Check scheduling of the next ANC contact
    assert antenatal_care.HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact in hsi_events
    assert (sim.population.props.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=4)))

    # =============================================== ANC 5 TESTING ==================================================
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    sim.date = sim.date + pd.DateOffset(weeks=4)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 4

    # The woman has experienced progression of her disease between appointments
    df.at[mother_id, 'ps_htn_disorders'] = 'severe_pre_eclamp'

    fifth_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 5)

    # Check the fourth dose of IPTp is given
    assert (df.at[mother_id, 'ac_doses_of_iptp_received'] == 4)

    # Check her severe pre-eclampsia has correctly been identified and she is scheduled to be admitted as an inpatient
    # (despite already being on antihypertensives)
    hsi_events = find_hsi_events_list(sim, mother_id)
    assert antenatal_care.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare in hsi_events

    # TODO: crashing herre

    # Check scheduling of the next ANC contact
    assert antenatal_care.HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact in hsi_events
    assert (df.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=2)))

    # =============================================== ANC 6 TESTING ==================================================
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    sim.date = sim.date + pd.DateOffset(weeks=2)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 2

    # Set that she has developed severe anaemia between appointments
    df.at[mother_id, 'ps_anaemia_in_pregnancy'] = 'severe'

    sixth_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 6)

    # Check her severe anaemia has correctly been identified via point of care screening (occurs at visit 2 and visit 6)
    # and she is scheduled to be admitted as an inpatient for further treatment
    hsi_events = find_hsi_events_list(sim, mother_id)
    assert antenatal_care.HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare in hsi_events

    # Check scheduling of the next ANC contact
    assert antenatal_care.HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact in hsi_events
    assert (df.at[mother_id, 'ac_date_next_contact'] == (sim.date + pd.DateOffset(weeks=2)))

    # =============================================== ANC 7 TESTING ==================================================
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    sim.date = sim.date + pd.DateOffset(weeks=2)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] += 2

    seventh_anc.apply(person_id=updated_mother_id, squeeze_factor=0.0)
    assert (df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] == 7)

    # Check the fifth and final dose of IPTp is given
    assert (df.at[mother_id, 'ac_doses_of_iptp_received'] == 5)

    # Check scheduling of the next ANC contact
    assert antenatal_care.HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact in hsi_events
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
    # TODO: test albendazole

test_perfect_run_of_anc_contacts_no_constraints()


# todo hep b - not stored as property
# todo syphilis - not stored as property
def test_anc_contacts_that_should_not_run_wont_run():
    """This test checks the inbuilt functions within ANC1 and ANC subsequent that should block, and in some cases
     reschedule, ANC visits for women when the HSI runs and is no longer appropriate for them """

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

    # todo: check anc 2 doesnt run and the next event is scheduled


def test_care_seeking_for_next_contact():
    """This test checks the logic around care seeking for the next ANC visit in the schedule. We test that women who are
     predicited at least 4 visits are automatically scheduled the next visit in the schedule (if that visit number is
    below 4). We also test that women who are not predicted 4 or more visits will seek care based on the value of a
    care seeking parameter"""
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


def test_dx_tests():
    sim = register_all_modules_no_consumable_constraints()
    sim.make_initial_population(n=100)

    params = sim.modules['CareOfWomenDuringPregnancy'].parameters
    params['sensitivity_bp_monitoring'] = 1.0
    params['specificity_bp_monitoring'] = 1.0
    params['sensitivity_urine_protein_1_plus'] = 1.0
    params['specificity_urine_protein_1_plus'] = 1.0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]
    updated_mother_id = int(mother_id)

    df.at[mother_id, 'ps_htn_disorders'] = 'mild_pre_eclamp'

    from tlo.methods.healthsystem import HSI_Event
    from tlo.events import IndividualScopeEventMixin

    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = sim.modules['HealthSystem'].get_blank_appt_footprint()
            self.ACCEPTED_FACILITY_LEVEL = 0
            self.ALERT_OTHER_DISEASES = []

        def apply(self, person_id, squeeze_factor):
            pass

    hsi_event = HSI_Dummy(module=sim.modules['CareOfWomenDuringPregnancy'], person_id=updated_mother_id)

    result_bp = sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='blood_pressure_measurement',
                                                                 hsi_event=hsi_event)
    result_protein = sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='urine_dipstick_protein',
                                                                        hsi_event=hsi_event)
    # both should be false
    assert ~result_protein
    assert ~result_bp

def test_antenatal_inpatient_care_anaemia():
    pass

def test_antenatal_inpatient_care_hypertension():
    pass

def test_antenatal_inpatient_care_prom_chorio():
    pass

def test_post_abortion_care():
    pass

def test_ectopic_case_management():
    pass
