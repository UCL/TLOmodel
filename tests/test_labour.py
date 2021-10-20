import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.methods import (
    care_of_women_during_pregnancy,
    cardio_metabolic_disorders,
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

seed = 1896

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
    """Make the initial population of a simulation and return an id number for a woman used within tests"""
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
    """Set key pregnancy characteristics in order for events to run correctly"""
    df = sim.population.props
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'la_due_date_current_pregnancy'] = sim.date
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date - pd.DateOffset(weeks=38)
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 40
    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)


def register_modules(ignore_cons_constraints):
    """Register all modules that are required for labour to run"""

    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           ignore_cons_constraints=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 malaria.Malaria(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 )

    return sim


def test_run_no_constraints():
    """This test runs a simulation with a functioning health system with full service availability and no set
    constraints"""

    sim = register_modules(ignore_cons_constraints=False)

    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(2015, 1, 1))

    check_dtypes(sim)


def test_event_scheduling_for_labour_onset_and_home_birth_no_care_seeking():
    sim = register_modules(ignore_cons_constraints=False)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
    params = sim.modules['Labour'].current_parameters

    # Set pregnancy characteristics that will allow the labour events to run
    set_pregnancy_characteristics(sim, mother_id)
    mni[mother_id]['test_run'] = True

    # force this woman to decide to deliver at home
    params['test_care_seeking_probs'] = [1, 0, 0]
    params['odds_will_attend_pnc'] = 0

    # define and run labour onset event
    labour_onset = labour.LabourOnsetEvent(individual_id=mother_id, module=sim.modules['Labour'])
    labour_onset.apply(mother_id)
    assert (mni[mother_id]['labour_state'] == 'term_labour')

    # Check that the correct events are scheduled for this woman whose labour has started
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]

    assert labour.BirthAndPostnatalOutcomesEvent in events
    assert labour.LabourDeathAndStillBirthEvent in events
    assert labour.LabourAtHomeEvent in events

    # Check she will not attend for facility delivery
    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert labour.HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour not in hsi_events

    # run birth event as this event manages scheduling of postpartum events (home birth and death events have their own
    # tests)
    sim.date = sim.date + pd.DateOffset(days=5)
    sim.event_queue.queue.clear()

    # Set care seeking odds to 0 (as changes event sequence if women seek care- tested later)
    params['prob_careseeking_for_complication'] = 0

    # Stop possibility of care seeking if she has developed a complication
    mni[mother_id]['squeeze_to_high_for_hsi_pp'] = True

    birth_event = labour.BirthAndPostnatalOutcomesEvent(mother_id=mother_id, module=sim.modules['Labour'])
    birth_event.apply(mother_id)

    # And finally check the first event of the postnatal module is correctly scheduled and no PNC is scheduled
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]
    assert postnatal_supervisor.PostnatalWeekOneEvent in events

    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert labour.HSI_Labour_ReceivesPostnatalCheck not in hsi_events


def test_event_scheduling_for_care_seeking_during_home_birth():
    sim = register_modules(ignore_cons_constraints=False)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    set_pregnancy_characteristics(sim, mother_id)

    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
    params = sim.modules['Labour'].current_parameters

    # Define and run the labour event
    labour_onset = labour.LabourOnsetEvent(individual_id=mother_id, module=sim.modules['Labour'])
    labour_onset.apply(mother_id)

    # Clear the event queues
    sim.event_queue.queue.clear()
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    # Force the woman to experience some complications and force that she will seek care
    mni[mother_id]['delivery_setting'] = 'home_birth'

    params['prob_obstruction_cpd'] = 1
    params['prob_careseeking_for_complication'] = 1

    # Run the intrapartum home birth event
    home_birth = labour.LabourAtHomeEvent(individual_id=mother_id, module=sim.modules['Labour'])
    home_birth.apply(mother_id)

    # check the correct variables are stored in the mni
    assert sim.population.props.at[mother_id, 'la_obstructed_labour']
    assert mni[mother_id]['sought_care_for_complication']
    assert (mni[mother_id]['sought_care_labour_phase'] == 'intrapartum')

    # Check that the woman will correctly seek care through HSI_GenericEmergencyFirstApptAtFacilityLevel1
    from tlo.methods.hsi_generic_first_appts import HSI_GenericEmergencyFirstApptAtFacilityLevel1
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

    # update key variables
    mni[mother_id]['sought_care_for_complication'] = False
    mni[mother_id]['sought_care_labour_phase'] = 'none'
    sim.date = sim.date + pd.DateOffset(days=5)

    # force a complication
    params['prob_pph_uterine_atony'] = 1

    # run postpartum home event
    home_birth_pp = labour.BirthAndPostnatalOutcomesEvent(mother_id=mother_id, module=sim.modules['Labour'])
    home_birth_pp.apply(mother_id)

    hsi_events = find_and_return_hsi_events_list(sim, mother_id)
    assert labour.HSI_Labour_ReceivesPostnatalCheck in hsi_events


def test_event_scheduling_for_labour_onset_and_facility_delivery():
    sim = register_modules(ignore_cons_constraints=False)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
    params = sim.modules['Labour'].current_parameters

    # Set pregnancy characteristics that will allow the labour events to run
    set_pregnancy_characteristics(sim, mother_id)

    # force this woman to decide to deliver at a health centre
    mni[mother_id]['test_run'] = True
    params['test_care_seeking_probs'] = [0, 1, 0]

    # run the labour onset event, check the correct mni variables are updated
    labour_onset = labour.LabourOnsetEvent(individual_id=mother_id, module=sim.modules['Labour'])
    labour_onset.apply(mother_id)
    assert (mni[mother_id]['labour_state'] == 'term_labour')
    assert (mni[mother_id]['delivery_setting'] == 'health_centre')

    # check birth and death events sheduled
    events = sim.find_events_for_person(person_id=mother_id)
    events = [e.__class__ for d, e in events]
    assert labour.BirthAndPostnatalOutcomesEvent in events
    assert labour.LabourDeathAndStillBirthEvent in events

    # now check the woman has correctly been scheduled the labour HSI
    date_event, event = [
            ev for ev in sim.modules['HealthSystem'].find_events_for_person(mother_id) if
            isinstance(ev[1], labour.HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour)
        ][0]
    assert date_event == sim.date


def test_event_scheduling_for_admissions_from_antenatal_inpatient_ward_for_caesarean_section():
    sim = register_modules(ignore_cons_constraints=False)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    # Set pregnancy characteristics that will allow the labour events to run
    set_pregnancy_characteristics(sim, mother_id)

    # Set the property that shows shes been admitted from AN care
    df = sim.population.props
    df.at[mother_id, 'ac_admitted_for_immediate_delivery'] = 'caesarean_now'

    # Run the labour onset, check she will correctly deliver at a hospital level facility
    labour_onset = labour.LabourOnsetEvent(individual_id=mother_id, module=sim.modules['Labour'])
    labour_onset.apply(mother_id)
    assert (mni[mother_id]['labour_state'] == 'term_labour')
    assert mni[mother_id]['delivery_setting'] == 'hospital'

    # Check the first HSI is scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(mother_id) if
        isinstance(ev[1], labour.HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour)
    ][0]
    assert date_event == sim.date

    # Run the first HSI
    updated_id = int(mother_id)
    labour_hsi = labour.HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour(
        person_id=updated_id, module=sim.modules['Labour'], facility_level_of_this_hsi=2)
    labour_hsi.apply(person_id=updated_id, squeeze_factor=0.0)

    # Check she has correctly been sent to the next HSI where the caesarean occurs
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(mother_id) if
        isinstance(ev[1], labour.HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare)
    ][0]
    assert date_event == sim.date

    # Run the caesarean HSI
    cs_hsi = labour.HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare(
        person_id=updated_id, module=sim.modules['Labour'], facility_level_of_this_hsi=2, timing='intrapartum')
    cs_hsi.apply(person_id=updated_id, squeeze_factor=0.0)

    # Check key variables are updated
    assert (mni[mother_id]['mode_of_delivery'] == 'caesarean_section')
    assert mni[mother_id]['amtsl_given']
    assert df.at[mother_id, 'la_previous_cs_delivery']

    # Move date forward and run the birth event
    sim.date = sim.date + pd.DateOffset(days=5)
    birth_event = labour.BirthAndPostnatalOutcomesEvent(mother_id=mother_id, module=sim.modules['Labour'])
    birth_event.apply(mother_id)


def test_application_of_risk_of_complications_in_intrapartum_and_postpartum_phases():
    sim = register_modules(ignore_cons_constraints=False)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    set_pregnancy_characteristics(sim, mother_id)
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    df = sim.population.props
    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)
    sim.modules['Labour'].set_labour_mni_variables(mother_id)

    # Force all complications and preceding complications to occur
    params = sim.modules['Labour'].current_parameters
    params['prob_obstruction_cpd'] = 1
    params['prob_obstruction_malpos_malpres'] = 1
    params['prob_placental_abruption_during_labour'] = 1
    params['prob_aph_placental_abruption_labour'] = 1
    params['prob_sepsis_chorioamnionitis'] = 1
    params['prob_uterine_rupture'] = 1

    # Call the function responsible to applying risk of complications
    for complication in ['obstruction_cpd', 'obstruction_malpos_malpres',
                         'placental_abruption', 'antepartum_haem', 'sepsis_chorioamnionitis', 'uterine_rupture']:
        sim.modules['Labour'].set_intrapartum_complications(mother_id, complication=complication)

    # Check that this mother is experiencing all complications as she should
    assert sim.population.props.at[mother_id, 'la_obstructed_labour']
    assert mni[mother_id]['cpd']
    assert sim.population.props.at[mother_id, 'la_placental_abruption']
    assert (sim.population.props.at[mother_id, 'la_antepartum_haem'] != 'none')
    assert sim.population.props.at[mother_id, 'la_sepsis']
    assert mni[mother_id]['chorio_in_preg']
    assert sim.population.props.at[mother_id, 'la_uterine_rupture']

    # Now repeat this process checking the application of risk of complications in the immediate postpartum period
    params['prob_sepsis_endometritis'] = 1
    params['prob_sepsis_urinary_tract'] = 1
    params['prob_sepsis_skin_soft_tissue'] = 1
    params['prob_sepsis_other_maternal_infection_pp'] = 1
    params['prob_pph_uterine_atony'] = 1
    params['prob_pph_retained_placenta'] = 1
    params['prob_pph_other_causes'] = 1

    for complication in ['sepsis_endometritis', 'sepsis_urinary_tract', 'sepsis_skin_soft_tissue',
                         'pph_uterine_atony', 'pph_retained_placenta', 'pph_other']:
        sim.modules['Labour'].set_postpartum_complications(mother_id, complication=complication)

    assert sim.population.props.at[mother_id, 'la_sepsis_pp']
    assert mni[mother_id]['endo_pp']
    assert sim.population.props.at[mother_id, 'la_postpartum_haem']
    assert mni[mother_id]['uterine_atony']
    assert mni[mother_id]['retained_placenta']

# todo: test hypertension logic in labour


def test_logic_within_death_and_still_birth_events():
    sim = register_modules(ignore_cons_constraints=False)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    df = sim.population.props

    # set the intercept values for the death LMs to 1 to force death to occur
    params = sim.modules['Labour'].current_parameters
    params['cfr_sepsis'] = 1
    params['cfr_eclampsia'] = 1
    params['cfr_severe_pre_eclamp'] = 1
    params['cfr_aph'] = 1
    params['cfr_uterine_rupture'] = 1
    params['cfr_severe_pre_eclamp'] = 1
    params['cfr_pp_pph'] = 1

    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)
    sim.modules['Labour'].set_labour_mni_variables(mother_id)

    df.at[mother_id, 'la_sepsis'] = True
    df.at[mother_id, 'la_antepartum_haem'] = 'severe'
    df.at[mother_id, 'la_uterine_rupture'] = True
    df.at[mother_id, 'ps_htn_disorders'] = 'eclampsia'

    causes = sim.modules['Labour'].get_potential_causes_of_death(mother_id, 'intrapartum')

    assert 'intrapartum_sepsis' in causes
    assert 'antepartum_haemorrhage' in causes
    assert 'eclampsia' in causes
    assert 'uterine_rupture' in causes

    result_from_death_calc = sim.modules['Labour'].apply_risk_of_death(mother_id, 'intrapartum', causes)

    assert result_from_death_calc

    df.at[mother_id, 'la_sepsis_pp'] = True
    df.at[mother_id, 'la_postpartum_haem'] = True
    df.at[mother_id, 'pn_htn_disorders'] = 'severe_pre_eclamp'

    causes_pp = sim.modules['Labour'].get_potential_causes_of_death(mother_id, 'postpartum')

    assert 'postpartum_sepsis' in causes_pp
    assert 'postpartum_haemorrhage' in causes_pp
    assert 'severe_pre_eclampsia' in causes_pp

    result_from_death_calc_pp = sim.modules['Labour'].apply_risk_of_death(mother_id, 'postpartum', causes)

    assert result_from_death_calc_pp

    # Now we test the event as a whole...
    # set variables that allow the event to run
    set_pregnancy_characteristics(sim, mother_id)
    sim.modules['Labour'].women_in_labour.append(mother_id)
    df.at[mother_id, 'la_currently_in_labour'] = True

    sim.date = sim.date + pd.DateOffset(days=4)

    # define and run death event
    death_event = labour.LabourDeathAndStillBirthEvent(individual_id=mother_id, module=sim.modules['Labour'])
    death_event.apply(mother_id)

    # Check the mother has died
    assert not df.at[mother_id, 'is_alive']

    # clear the event queue and reset is_alive
    df.at[mother_id, 'is_alive'] = True
    sim.event_queue.queue.clear()
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)
    sim.modules['Labour'].set_labour_mni_variables(mother_id)

    # set risk of death to 0 but set risk of stillbirth to 1
    params['cfr_sepsis'] = 0
    params['cfr_eclampsia'] = 0
    params['cfr_severe_pre_eclamp'] = 0
    params['cfr_aph'] = 0
    params['cfr_uterine_rupture'] = 0
    params['cfr_severe_pre_eclamp'] = 0
    params['cfr_pp_pph'] = 0
    params['prob_ip_still_birth'] = 1

    # run the event again and check the correct variables have been updated to signify stillbirth
    death_event.apply(mother_id)
    assert df.at[mother_id, 'la_intrapartum_still_birth']
    assert df.at[mother_id, 'ps_prev_stillbirth']
    assert not df.at[mother_id, 'is_pregnant']

    # todo: seperate test for the BirthPostnatal event?


def test_bemonc_treatments_are_delivered_correctly_with_no_cons_or_quality_constraints_via_functions():

    sim = register_modules(ignore_cons_constraints=True)
    sim.make_initial_population(n=100)

    # Set sensitivity parameters for dxtests to 1 to ensure all complications are detected and treated
    params = sim.modules['Labour'].current_parameters
    params['sensitivity_of_assessment_of_obstructed_labour_hc'] = 1.0
    params['sensitivity_of_assessment_of_obstructed_labour_hp'] = 1.0
    params['sensitivity_of_assessment_of_sepsis_hc'] = 1.0
    params['sensitivity_of_assessment_of_sepsis_hp'] = 1.0
    params['sensitivity_of_assessment_of_severe_pe_hc'] = 1.0
    params['sensitivity_of_assessment_of_severe_pe_hp'] = 1.0
    params['sensitivity_of_assessment_of_hypertension_hc'] = 1.0
    params['sensitivity_of_assessment_of_hypertension_hp'] = 1.0
    params['sensitivity_of_assessment_of_antepartum_haem_hc'] = 1.0
    params['sensitivity_of_assessment_of_antepartum_haem_hp'] = 1.0
    params['sensitivity_of_assessment_of_uterine_rupture_hc'] = 1.0
    params['sensitivity_of_assessment_of_uterine_rupture_hp'] = 1.0
    params['sensitivity_of_assessment_of_ec_hc'] = 1.0
    params['sensitivity_of_assessment_of_ec_hp'] = 1.0
    params['sensitivity_of_assessment_of_pph_hc'] = 1.0
    params['sensitivity_of_assessment_of_pph_hp'] = 1.0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = int(women_repro.index[0])
    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)
    sim.modules['Labour'].set_labour_mni_variables(mother_id)

    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
    df = sim.population.props

    # create a dummy hsi event that the treatment functions will call
    from tlo.events import IndividualScopeEventMixin
    from tlo.methods.healthsystem import HSI_Event

    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)

            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = sim.modules['HealthSystem'].get_blank_appt_footprint()
            self.ACCEPTED_FACILITY_LEVEL = 0
            self.ALERT_OTHER_DISEASES = []

        def apply(self, person_id, squeeze_factor):
            pass

    hsi_event = HSI_Dummy(module=sim.modules['Labour'], person_id=mother_id)

    # Set woman to have severe pre-eclampsia, run function and check that treatment (mgso4) is delivered
    df.at[mother_id, 'ps_htn_disorders'] = 'severe_pre_eclamp'
    sim.modules['Labour'].assessment_and_treatment_of_severe_pre_eclampsia_mgso4(
        hsi_event=hsi_event, facility_type='hc', labour_stage='ip')
    assert df.at[mother_id, 'la_severe_pre_eclampsia_treatment']

    # Now check she also would receive antihypertensives
    sim.modules['Labour'].assessment_and_treatment_of_hypertension(
        hsi_event=hsi_event, facility_type='hc', labour_stage='ip')
    assert df.at[mother_id, 'la_maternal_hypertension_treatment']

    # Set disease status to eclampsia and run the appropriate intervention function, check treatment delivered
    df.at[mother_id, 'ps_htn_disorders'] = 'eclampsia'
    sim.modules['Labour'].assessment_and_treatment_of_eclampsia(
        hsi_event=hsi_event, facility_type='hc', labour_stage='ip')
    assert df.at[mother_id, 'la_eclampsia_treatment']

    # Set woman to be in obstructed labour due to CPD
    df.at[mother_id, 'la_obstructed_labour'] = True
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['cpd'] = True

    # Run the event and check she has correctly been referred for caesarean
    sim.modules['Labour'].assessment_and_treatment_of_obstructed_labour_via_avd(
        hsi_event=hsi_event, facility_type='hc')
    assert mni[mother_id]['referred_for_cs']

    # Remove CPD as a cause and set probability of AVD being successful to 1, call the function and check she has
    # undergone instrumental delivery
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['cpd'] = False
    params['prob_successful_assisted_vaginal_delivery'] = 1
    sim.modules['Labour'].assessment_and_treatment_of_obstructed_labour_via_avd(
        hsi_event=hsi_event, facility_type='hc')
    assert (mni[mother_id]['mode_of_delivery'] == 'instrumental')

    # Next set the women to have sepsis and check she is treated
    df.at[mother_id, 'la_sepsis'] = True
    sim.modules['Labour'].assessment_and_treatment_of_maternal_sepsis(
        hsi_event=hsi_event, facility_type='hc', labour_stage='ip')
    assert df.at[mother_id, 'la_sepsis_treatment']

    # Next set the woman to having a severe antepartum haemorrhage, check she will correctly be referred for blood and
    # a caesarean section
    df.at[mother_id, 'la_antepartum_haem'] = 'severe'
    sim.modules['Labour'].assessment_and_plan_for_antepartum_haemorrhage(
        hsi_event=hsi_event, facility_type='hc')
    assert mni[mother_id]['referred_for_cs']
    assert mni[mother_id]['referred_for_blood']

    # Reset those properties
    mni[mother_id]['referred_for_cs'] = False
    mni[mother_id]['referred_for_blood'] = False

    # Now check a woman with uterine rupture is correctly referred for surgery, caesarean and blood
    df.at[mother_id, 'la_uterine_rupture'] = True
    sim.modules['Labour'].assessment_for_referral_uterine_rupture(
        hsi_event=hsi_event, facility_type='hc')
    assert mni[mother_id]['referred_for_cs']
    assert mni[mother_id]['referred_for_blood']
    assert mni[mother_id]['referred_for_surgery']

    # reset those properties
    mni[mother_id]['referred_for_cs'] = False
    mni[mother_id]['referred_for_blood'] = False
    mni[mother_id]['referred_for_surgery'] = False

    # Finally check treatment for postpartum haem. Set probablity that uterotonics will stop bleeding to 1
    df.at[mother_id, 'la_postpartum_haem'] = True
    mni[mother_id]['uterine_atony'] = True
    params['prob_haemostatis_uterotonics'] = 1

    # Run the event and check that the woman is referred for blood but not surgery. And that treatment is stored in
    # bitset property
    sim.modules['Labour'].assessment_and_treatment_of_pph_uterine_atony(
        hsi_event=hsi_event, facility_type='hc')
    assert sim.modules['Labour'].pph_treatment.has_all(mother_id, 'uterotonics')
    assert mni[mother_id]['referred_for_blood']
    assert not mni[mother_id]['referred_for_surgery']

    # Reset those properties and set the probability of successful medical management to 0
    sim.modules['Labour'].pph_treatment.unset(mother_id, 'uterotonics')
    mni[mother_id]['referred_for_blood'] = False
    params['prob_haemostatis_uterotonics'] = 0

    # Call treatment function and check she has correctly been referred for surgical management and blood
    sim.modules['Labour'].assessment_and_treatment_of_pph_uterine_atony(
        hsi_event=hsi_event, facility_type='hc')
    assert mni[mother_id]['referred_for_blood']
    assert mni[mother_id]['referred_for_surgery']

    # Rest those variables
    mni[mother_id]['referred_for_blood'] = False
    mni[mother_id]['referred_for_surgery'] = False

    # set retained placenta variable
    mni[mother_id]['retained_placenta'] = True

    # Now assume the bleed is due to retained placenta, set probablity of bedside removal to 1 and call function
    params['prob_successful_manual_removal_placenta'] = 1
    sim.modules['Labour'].assessment_and_treatment_of_pph_retained_placenta(
        hsi_event=hsi_event, facility_type='hc')
    assert mni[mother_id]['referred_for_blood']
    assert not mni[mother_id]['referred_for_surgery']

    mni[mother_id]['referred_for_blood'] = False
    params['prob_successful_manual_removal_placenta'] = 0

    # Now check that surgery is correctly scheduled as manual removal has failed
    sim.modules['Labour'].assessment_and_treatment_of_pph_retained_placenta(
        hsi_event=hsi_event, facility_type='hc')
    assert mni[mother_id]['referred_for_blood']
    assert mni[mother_id]['referred_for_surgery']


def test_pph_treatment():
    sim = register_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    sim.modules['Labour'].current_parameters['prob_haemostatis_uterotonics'] = 1
    sim.modules['Labour'].current_parameters['prob_successful_manual_removal_placenta'] = 1
    sim.modules['Labour'].current_parameters['cfr_pp_pph'] = 1
    sim.modules['Labour'].current_parameters['pph_treatment_effect_uterotonics_md'] = 0
    sim.modules['Labour'].current_parameters['pph_treatment_effect_mrp_md'] = 0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    df.loc[women_repro.index, 'is_pregnant'] = True
    df.loc[women_repro.index, 'la_date_most_recent_delivery'] = sim.date
    df.loc[women_repro.index, 'la_is_postpartum'] = True
    df.loc[women_repro.index, 'la_postpartum_haem'] = True

    for person in women_repro.index:
        sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(person)
        sim.modules['Labour'].set_labour_mni_variables(person)
        sim.modules['Labour'].women_in_labour.append(person)

    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    women_with_ua = int(women_repro.index[0])
    mni[women_with_ua]['uterine_atony'] = True

    pnc = labour.HSI_Labour_ReceivesPostnatalCheck(
        person_id=women_with_ua, module=sim.modules['Labour'], facility_level_of_this_hsi=1)
    pnc.apply(person_id=women_with_ua, squeeze_factor=0.0)

    assert sim.population.props.at[women_with_ua, 'is_alive']

    women_with_rp = int(women_repro.index[1])
    mni[women_with_rp]['retained_placenta'] = True

    pnc = labour.HSI_Labour_ReceivesPostnatalCheck(
        person_id=women_with_rp, module=sim.modules['Labour'], facility_level_of_this_hsi=1)
    pnc.apply(person_id=women_with_rp, squeeze_factor=0.0)

    assert sim.population.props.at[women_with_rp, 'is_alive']


test_pph_treatment()

def test_cemonc_event_and_treatments_are_delivered_correct_with_no_cons_or_quality_constraints():
    sim = register_modules(ignore_cons_constraints=True)
    mother_id = run_sim_for_0_days_get_mother_id(sim)
    updated_id = int(mother_id)
    set_pregnancy_characteristics(sim, mother_id)

    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
    df = sim.population.props
    params = sim.modules['Labour'].current_parameters

    # Run labour onset event to update additional variables
    labour_onset = labour.LabourOnsetEvent(individual_id=mother_id, module=sim.modules['Labour'])
    labour_onset.apply(mother_id)

    # Define the events (timing variable define when sheduling as this event can be called during or after labour)
    ip_cemonc_event = labour.HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare(
        person_id=updated_id, module=sim.modules['Labour'], facility_level_of_this_hsi=2, timing='intrapartum')
    pp_cemonc_event = labour.HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare(
        person_id=updated_id, module=sim.modules['Labour'], facility_level_of_this_hsi=2, timing='postpartum')

    # Test uterine rupture surgery
    # Set variables showing woman has been referred to surgery due to uterine rupture
    mni[mother_id]['referred_for_surgery'] = True
    df.at[mother_id, 'la_uterine_rupture'] = True

    # Force success rate of surgery to 1
    params['success_rate_uterine_repair'] = 1

    # Run the surgery and check the treatment has been delivered
    ip_cemonc_event.apply(person_id=updated_id, squeeze_factor=0.0)
    assert df.at[mother_id, 'la_uterine_rupture_treatment']
    assert not df.at[mother_id, 'la_has_had_hysterectomy']

    # Now set the success rate of repair to 0 and run the event again, check this time the woman has undergone
    # hysterectomy
    params['success_rate_uterine_repair'] = 0
    ip_cemonc_event.apply(person_id=updated_id, squeeze_factor=0.0)
    assert df.at[mother_id, 'la_has_had_hysterectomy']

    # Test blood transfusion
    # Use the mni property to signify this woman needs a blood transfusion and run the event
    df.at[mother_id, 'la_uterine_rupture'] = False
    df.at[mother_id, 'la_has_had_hysterectomy'] = False

    mni[mother_id]['referred_for_blood'] = True
    ip_cemonc_event.apply(person_id=updated_id, squeeze_factor=0.0)

    # Check the transfusion is delivered
    assert mni[mother_id]['received_blood_transfusion']

    # rest the property
    mni[mother_id]['referred_for_blood'] = False

    # Test PPH surgery - uterine atony
    # Set PPH to true and the underlying cause to being uterine atony
    df.at[mother_id, 'la_postpartum_haem'] = True
    df.at[mother_id, 'la_is_postpartum'] = True
    mni[mother_id]['referred_for_surgery'] = True
    mni[mother_id]['uterine_atony'] = True

    df.at[mother_id, 'la_date_most_recent_delivery'] = sim.date + pd.DateOffset(days=4)

    # force surgery to be successful
    params['success_rate_pph_surgery'] = 1

    # run event, check the women underwent successful surgery and didnt have a hysterectomy
    pp_cemonc_event.apply(person_id=updated_id, squeeze_factor=0.0)
    # cant check this is set because it gets reset in the death function called at the end of this event
    # assert sim.modules['Labour'].pph_treatment.has_all(mother_id, 'surgery')
    assert not sim.modules['Labour'].pph_treatment.has_all(mother_id, 'hysterectomy')
    assert not df.at[mother_id, 'la_has_had_hysterectomy']

    # clear queues and add woman back onto labour list
    sim.event_queue.queue.clear()
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()
    sim.modules['Labour'].women_in_labour.append(mother_id)

    # reset comp variables
    df.at[mother_id, 'la_currently_in_labour'] = True
    df.at[mother_id, 'la_postpartum_haem'] = True

    # now set surgery success to 0 and check that hysterectomy occurred
    params['success_rate_pph_surgery'] = 0
    sim.modules['Labour'].pph_treatment.unset(mother_id, 'surgery')
    sim.modules['Labour'].pph_treatment.unset(mother_id, 'uterotonics')
    pp_cemonc_event.apply(person_id=updated_id, squeeze_factor=0.0)

    assert df.at[mother_id, 'la_has_had_hysterectomy']


def test_to_check_similarly_named_and_functioning_dx_tests_work_as_expected():
    sim = register_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    params = sim.modules['Labour'].current_parameters
    params['sensitivity_of_assessment_of_obstructed_labour_hc'] = 1.0
    params['sensitivity_of_assessment_of_obstructed_labour_hp'] = 1.0
    params['sensitivity_of_assessment_of_hypertension_hc'] = 1.0
    params['sensitivity_of_assessment_of_hypertension_hp'] = 1.0
    params['sensitivity_of_assessment_of_sepsis_hc'] = 1.0
    params['sensitivity_of_assessment_of_sepsis_hp'] = 1.0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    updated_id = int(women_repro.index[0])

    # Test a catagorical test
    # Confirm correct sensitivity and specificiy
    sim.modules['HealthSystem'].dx_manager.print_info_about_dx_test('assess_hypertension_hc_ip')
    sim.modules['HealthSystem'].dx_manager.print_info_about_dx_test('assess_hypertension_hp_ip')

    # Find all categories for blood pressure in the df
    all_categories = list(df['ps_htn_disorders'].cat.categories)

    # Find the target categories for blood pressure in the df:
    target_categories = ['gest_htn', 'mild_pre_eclamp', 'severe_gest_htn', 'severe_pre_eclamp', 'eclampsia']

    # check target categories has been specified correctly:
    assert set(target_categories) == set(
        sim.modules['HealthSystem'].dx_manager.dx_tests['assess_hypertension_hc_ip'][0].target_categories
    )
    assert set(target_categories).issubset(target_categories)

    assert set(target_categories) == set(
        sim.modules['HealthSystem'].dx_manager.dx_tests['assess_hypertension_hp_ip'][0].target_categories
    )
    assert set(target_categories).issubset(target_categories)

    # Make Dummy Event
    from tlo.events import IndividualScopeEventMixin
    from tlo.methods.healthsystem import HSI_Event

    class HSI_Dummy(HSI_Event, IndividualScopeEventMixin):
        def __init__(self, module, person_id):
            super().__init__(module, person_id=person_id)
            self.TREATMENT_ID = 'Dummy'
            self.EXPECTED_APPT_FOOTPRINT = sim.modules['HealthSystem'].get_blank_appt_footprint()
            self.ACCEPTED_FACILITY_LEVEL = 0
            self.ALERT_OTHER_DISEASES = []

        def apply(self, person_id, squeeze_factor):
            pass

    hsi_event = HSI_Dummy(module=sim.modules['Labour'], person_id=updated_id)

    # for each of the categories, test that DX test give the expected answer:
    for true_value in all_categories:
        sim.population.props.at[updated_id, 'ps_htn_disorders'] = true_value
        # Run DxTest for 'blood_pressure_measurement'
        test_result = sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run='assess_hypertension_hc_ip',
            hsi_event=hsi_event
        )
        # check that result of dx test is as expected:
        assert test_result is (true_value in target_categories)

    for true_value in all_categories:
        sim.population.props.at[updated_id, 'ps_htn_disorders'] = true_value
        # Run DxTest for 'blood_pressure_measurement'
        test_result = sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run='assess_hypertension_hp_ip',
            hsi_event=hsi_event
        )
        # check that result of dx test is as expected:
        assert test_result is (true_value in target_categories)

    df.at[updated_id, 'la_obstructed_labour'] = True
    assert (sim.modules['HealthSystem'].dx_manager.dx_tests['assess_obstructed_labour_hc'][0].property == 'la_'
                                                                                                          'obstructed'
                                                                                                          '_labour')
    assert (sim.modules['HealthSystem'].dx_manager.dx_tests['assess_obstructed_labour_hp'][0].property == 'la_'
                                                                                                          'obstructed_'
                                                                                                          'labour')

    test_result = sim.modules['HealthSystem'].dx_manager.run_dx_test(
        dx_tests_to_run='assess_obstructed_labour_hc',
        hsi_event=hsi_event
    )
    assert test_result

    test_result = sim.modules['HealthSystem'].dx_manager.run_dx_test(
        dx_tests_to_run='assess_obstructed_labour_hp',
        hsi_event=hsi_event
    )
    assert test_result

    # additional examples

    sim.modules['HealthSystem'].dx_manager.print_info_about_dx_test('assess_hypertension_hc_ip')
    sim.modules['HealthSystem'].dx_manager.print_info_about_dx_test('assess_hypertension_hp_ip')

    df.at[updated_id, 'la_sepsis'] = True

    assert (sim.modules['HealthSystem'].dx_manager.dx_tests['assess_sepsis_hc_ip'][0].property == 'la_sepsis')
    assert (sim.modules['HealthSystem'].dx_manager.dx_tests['assess_sepsis_hp_ip'][0].property == 'la_sepsis')

    test_result = sim.modules['HealthSystem'].dx_manager.run_dx_test(
        dx_tests_to_run='assess_sepsis_hc_ip',
        hsi_event=hsi_event
    )
    assert test_result

    test_result = sim.modules['HealthSystem'].dx_manager.run_dx_test(
        dx_tests_to_run='assess_sepsis_hp_ip',
        hsi_event=hsi_event
    )
    assert test_result

# todo: test squeeze factor logic
# todo: further CEmONC testing
# todo: test consumables constraints block interventions


