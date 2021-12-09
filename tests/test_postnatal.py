import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
    joes_fake_props_module
)
from tlo.methods.hiv import DummyHivModule

seed = 123

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


def generate_postnatal_women(sim):
    """Return postnatal women"""
    df = sim.population.props

    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    df.loc[women_repro.index, 'la_is_postpartum'] = True
    df.loc[women_repro.index, 'la_date_most_recent_delivery'] = sim.date - pd.DateOffset(weeks=2)
    for person in women_repro.index:
        sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(person)
        sim.modules['Labour'].set_labour_mni_variables(person)

    return women_repro


def get_mother_id_from_dataframe(sim):
    """Return individual id from dataframe for postnatal testing"""
    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]
    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)
    sim.modules['Labour'].set_labour_mni_variables(mother_id)

    return mother_id


def register_core_modules(ignore_cons_constraints):
    _cons_availability = 'all' if ignore_cons_constraints else 'none'
    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           cons_availability=_cons_availability),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),

                 # - Dummy HIV module (as contraception requires the property hv_inf)
                 DummyHivModule()
                 )

    return sim


def test_run_and_check_dtypes():
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(2015, 1, 1))
    check_dtypes(sim)


def test_antenatal_disease_is_correctly_carried_over_to_postnatal_period_on_birth():
    """Test that complications which may continue from the antenatal period to the postnatal period transition as
    expected"""
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    mother_id = get_mother_id_from_dataframe(sim)

    # Set properties from the antenatal period that we would expect to be carried into the postnatal period, if they
    # dont resolve before or on birth
    df = sim.population.props
    df.at[mother_id, 'ac_gest_htn_on_treatment'] = True
    sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.set(mother_id, 'iron')
    sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.set(mother_id, 'folate')
    sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.set(mother_id, 'b12')
    df.at[mother_id, 'ps_anaemia_in_pregnancy'] = 'moderate'
    df.at[mother_id, 'ps_htn_disorders'] = 'mild_pre_eclamp'

    # Run the birth events
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    # check the properties are carried across into new postnatal properties
    assert sim.population.props.at[mother_id, 'la_gest_htn_on_treatment']
    assert sim.population.props.at[mother_id, 'pn_anaemia_following_pregnancy'] == 'moderate'
    assert sim.population.props.at[mother_id, 'pn_htn_disorders'] == 'mild_pre_eclamp'
    assert sim.modules['PostnatalSupervisor'].deficiencies_following_pregnancy.has_all(mother_id, 'iron')
    assert sim.modules['PostnatalSupervisor'].deficiencies_following_pregnancy.has_all(mother_id, 'folate')
    assert sim.modules['PostnatalSupervisor'].deficiencies_following_pregnancy.has_all(mother_id, 'b12')

    # todo: properties are now reset at week 6 to allow for use within linear models postnatally


def test_application_of_maternal_complications_and_care_seeking_postnatal_week_one_event():
    """Test that risk of complications is correctly applied in the first week postnatal and that women seek care as
    expected"""
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # set risk of maternal complications (occuring in week one) to one to insure risk applied as expected
    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_secondary_pph'] = 1
    params['prob_late_sepsis_endometritis'] = 1
    params['prob_late_sepsis_urinary_tract'] = 1
    params['prob_late_sepsis_skin_soft_tissue'] = 1
    params['prob_iron_def_per_week_pn'] = 1
    params['prob_folate_def_per_week_pn'] = 1
    params['prob_b12_def_per_week_pn'] = 1
    params['baseline_prob_anaemia_per_week'] = 1
    params['prob_type_of_anaemia_pn'] = [1, 0, 0]
    params['weekly_prob_gest_htn_pn'] = 1
    params['prob_early_onset_neonatal_sepsis_week_1'] = 1
    params['prob_care_seeking_postnatal_emergency'] = 1

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    # Get id number of mother
    mother_id = get_mother_id_from_dataframe(sim)

    # Run birth event to generate child
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    # check key postnatal variables correctly set
    assert sim.population.props.at[mother_id, 'pn_id_most_recent_child'] == child_id
    assert sim.population.props.at[mother_id, 'la_date_most_recent_delivery'] == sim.date
    assert sim.population.props.at[mother_id, 'la_is_postpartum']

    # define and run the event
    postnatal_week_one = postnatal_supervisor.PostnatalWeekOneEvent(
        individual_id=mother_id, module=sim.modules['PostnatalSupervisor'])
    postnatal_week_one.apply(mother_id)

    # check that complications have been correctly set (storing dalys where appropriate)
    assert sim.population.props.at[mother_id, 'pn_sepsis_late_postpartum']
    assert (mni[mother_id]['sepsis_onset'] == sim.date)
    assert mni[mother_id]['endo_pp']

    assert sim.population.props.at[mother_id, 'pn_postpartum_haem_secondary']
    assert (mni[mother_id]['secondary_pph_onset'] == sim.date)

    assert sim.modules['PostnatalSupervisor'].deficiencies_following_pregnancy.has_all(mother_id, 'iron')
    assert sim.modules['PostnatalSupervisor'].deficiencies_following_pregnancy.has_all(mother_id, 'folate')
    assert sim.modules['PostnatalSupervisor'].deficiencies_following_pregnancy.has_all(mother_id, 'b12')
    assert (sim.population.props.at[mother_id, 'pn_anaemia_following_pregnancy'] == 'mild')

    assert sim.population.props.at[mother_id, 'pn_htn_disorders'] == 'gest_htn'

    # Check HSI has been correctly scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(mother_id) if
        isinstance(ev[1], labour.HSI_Labour_ReceivesPostnatalCheck)
    ][0]
    assert date_event == sim.date

# todo: htn progression/resolution


def test_application_of_neonatal_complications_and_care_seeking_postnatal_week_one_event():
    """Test that risk of complications is correctly applied in the first week postnatal and that women seek care as
    expected"""
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # set risk of newborn complications (occuring in week one) to one to insure risk applied as expected
    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_early_onset_neonatal_sepsis_week_1'] = 1
    params['prob_care_seeking_postnatal_emergency_neonate'] = 1

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # Get id number of mother
    mother_id = get_mother_id_from_dataframe(sim)

    # Set mother to be pregnant with twins to test twin logic of this event
    sim.population.props.at[mother_id, 'ps_multiple_pregnancy'] = True

    # Run birth event to generate child
    child_id_one = sim.do_birth(mother_id)
    child_id_two = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id_one)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id_two)
    sim.modules['NewbornOutcomes'].link_twins(child_id_one, child_id_two, mother_id)

    sim.modules['NewbornOutcomes'].newborn_care_info[child_id_one]['will_receive_pnc'] = 'late'
    sim.modules['NewbornOutcomes'].newborn_care_info[child_id_two]['will_receive_pnc'] = 'late'

    # define and run the event using the mother_id as this is how its coded
    postnatal_week_one = postnatal_supervisor.PostnatalWeekOneEvent(
        individual_id=mother_id, module=sim.modules['PostnatalSupervisor'])
    postnatal_week_one.apply(mother_id)

    assert sim.modules['NewbornOutcomes'].newborn_care_info[child_id_one]['passed_through_week_one']
    assert sim.modules['NewbornOutcomes'].newborn_care_info[child_id_two]['passed_through_week_one']

    assert sim.population.props.at[child_id_one, 'pn_sepsis_early_neonatal']
    assert sim.population.props.at[child_id_two, 'pn_sepsis_early_neonatal']

    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(child_id_one) if
        isinstance(ev[1], newborn_outcomes.HSI_NewbornOutcomes_ReceivesPostnatalCheck)
    ][0]
    assert date_event == sim.date

    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(child_id_two) if
        isinstance(ev[1], newborn_outcomes.HSI_NewbornOutcomes_ReceivesPostnatalCheck)
    ][0]
    assert date_event == sim.date

# Todo: test to check things ARNENT sheduled


def test_all_appropriate_pregnancy_variables_are_reset_at_then_end_of_postnatal():
    pass


def test_application_of_risk_of_death_to_mothers_postnatal_week_one_event():
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # set risk of complications at 1 so woman is at risk of death
    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_secondary_pph'] = 1
    params['prob_late_sepsis_endometritis'] = 1
    params['prob_late_sepsis_urinary_tract_inf'] = 1

    # Prevent care seeking and set risk of death due to comps as 1
    params['prob_care_seeking_postnatal_emergency'] = 0
    params['cfr_secondary_postpartum_haemorrhage'] = 1
    params['cfr_postpartum_sepsis'] = 1

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # Get id number of mother
    mother_id = get_mother_id_from_dataframe(sim)
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    # Run the event, as care seeking is blocked risk of death should be applied in the event and then carried out
    postnatal_week_one = postnatal_supervisor.PostnatalWeekOneEvent(
        individual_id=mother_id, module=sim.modules['PostnatalSupervisor'])
    postnatal_week_one.apply(mother_id)

    assert not sim.population.props.at[mother_id, 'is_alive']


def test_application_of_risk_of_death_to_neonates_postnatal_week_one_event():
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # set risk of complications at 1 so woman is at risk of death
    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_early_onset_neonatal_sepsis_week_1'] = 1

    # Prevent care seeking and set risk of death due to comps as 1
    params['prob_care_seeking_postnatal_emergency_neonate'] = 0
    params['cfr_early_onset_neonatal_sepsis'] = 1

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # Get id number of newborns
    mother_id = get_mother_id_from_dataframe(sim)

    # Set mother to be pregnant with twins to test twin logic of this event
    sim.population.props.at[mother_id, 'ps_multiple_pregnancy'] = True

    child_id_one = sim.do_birth(mother_id)
    child_id_two = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id_one)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id_two)
    sim.modules['NewbornOutcomes'].link_twins(child_id_one, child_id_two, mother_id)

    # Run the event, as care seeking is blocked risk of death should be applied in the event and then carried out
    postnatal_week_one = postnatal_supervisor.PostnatalWeekOneEvent(
        individual_id=mother_id, module=sim.modules['PostnatalSupervisor'])
    postnatal_week_one.apply(mother_id)

    assert sim.population.props.at[child_id_one, 'pn_sepsis_early_neonatal']
    assert sim.population.props.at[child_id_two, 'pn_sepsis_early_neonatal']
    assert not sim.population.props.at[child_id_one, 'is_alive']
    assert not sim.population.props.at[child_id_two, 'is_alive']


def test_application_of_risk_of_infection_and_sepsis_postnatal_supervisor_event():
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # set risk of infection and sepsis to 1
    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_late_sepsis_endometritis'] = 1
    params['prob_late_sepsis_urinary_tract_inf'] = 1
    params['prob_late_sepsis_skin_soft_tissue_inf'] = 1

    # and risk of care seeking to 1
    params['prob_care_seeking_postnatal_emergency'] = 1

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # Run the postnatal supervisor event
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
    postnatal_women = generate_postnatal_women(sim)

    post_natal_sup = postnatal_supervisor.PostnatalSupervisorEvent(module=sim.modules['PostnatalSupervisor'])
    post_natal_sup.apply(sim.population)

    # Select a mother to check properties against
    mother_id = postnatal_women.index[0]

    # Check that she has developed sepsis, DALYS are stored
    assert sim.population.props.at[mother_id, 'pn_sepsis_late_postpartum']
    assert (mni[mother_id]['sepsis_onset'] == sim.date)
    assert mni[mother_id]['endo_pp']
    assert sim.population.props.at[mother_id, 'is_alive']

    # finally check she will now seek care
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(mother_id) if
        isinstance(ev[1], labour.HSI_Labour_ReceivesPostnatalCheck)
    ][0]
    assert date_event == sim.date

    # clear event queues
    sim.event_queue.queue.clear()
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    # reset variables
    sim.population.props.at[mother_id, 'pn_sepsis_late_postpartum'] = False

    # set risk of care seeking to 0, risk of death to 1
    params['prob_care_seeking_postnatal_emergency'] = 0
    params['cfr_postpartum_sepsis'] = 1

    # call the event again
    post_natal_sup.apply(sim.population)

    # check women are scheduled for death not careseeking
    assert not sim.population.props.at[mother_id, 'is_alive']


def test_application_of_risk_of_spph_postnatal_supervisor_event():
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_secondary_pph'] = 1
    params['prob_care_seeking_postnatal_emergency'] = 1
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    postnatal_women = generate_postnatal_women(sim)

    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    # Run the event
    post_natal_sup = postnatal_supervisor.PostnatalSupervisorEvent(module=sim.modules['PostnatalSupervisor'])
    post_natal_sup.apply(sim.population)

    # check that properties are set correctly
    mother_id = postnatal_women.index[0]
    assert sim.population.props.at[mother_id, 'pn_postpartum_haem_secondary']
    assert (mni[mother_id]['secondary_pph_onset'] == sim.date)
    assert sim.population.props.at[mother_id, 'is_alive']

    # and care has been sought
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(mother_id) if
        isinstance(ev[1], labour.HSI_Labour_ReceivesPostnatalCheck)
    ][0]
    assert date_event == sim.date

    # clear event queue
    sim.event_queue.queue.clear()
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    # reset variables
    sim.population.props.at[mother_id, 'pn_postpartum_haem_secondary'] = False

    # set risk of care seeking to 0, risk of death to 1
    params['prob_care_seeking_postnatal_emergency'] = 0
    params['cfr_secondary_pph'] = 1

    # call the event again
    post_natal_sup.apply(sim.population)

    # check women are scheduled for death not care seeking
    assert not sim.population.props.at[mother_id, 'is_alive']


def test_application_of_risk_of_anaemia_postnatal_supervisor_event():
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_iron_def_per_week_pn'] = 1
    params['prob_folate_def_per_week_pn'] = 1
    params['prob_b12_def_per_week_pn'] = 1
    params['baseline_prob_anaemia_per_week'] = 1
    params['prob_type_of_anaemia_pn'] = [1, 0, 0]
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    postnatal_women = generate_postnatal_women(sim)

    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    # run the event
    post_natal_sup = postnatal_supervisor.PostnatalSupervisorEvent(module=sim.modules['PostnatalSupervisor'])
    post_natal_sup.apply(sim.population)

    # check the properties are set correctly
    mother_id = postnatal_women.index[0]
    assert sim.modules['PostnatalSupervisor'].deficiencies_following_pregnancy.has_all(mother_id, 'iron')
    assert sim.modules['PostnatalSupervisor'].deficiencies_following_pregnancy.has_all(mother_id, 'folate')
    assert sim.modules['PostnatalSupervisor'].deficiencies_following_pregnancy.has_all(mother_id, 'b12')

    assert sim.population.props.at[mother_id, 'pn_anaemia_following_pregnancy'] == 'mild'
    assert (mni[mother_id]['mild_anaemia_pp_onset'] == sim.date)


def test_application_of_risk_of_hypertensive_disorders_postnatal_supervisor_event():
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # Set parameters to force resolution and onset of disease
    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_htn_resolves'] = 1
    params['weekly_prob_pre_eclampsia_pn'] = 1

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
    df = sim.population.props
    postnatal_women = generate_postnatal_women(sim)

    # Select two women, one who should experience resoultion of her disease during the event
    mother_id_resolve = postnatal_women.index[0]
    df.at[mother_id_resolve, 'pn_htn_disorders'] = 'mild_pre_eclamp'
    mni[mother_id_resolve]['hypertension_onset'] = sim.date - pd.DateOffset(weeks=5)

    # one who should develop disease for the first time
    mother_id_onset = postnatal_women.index[1]

    # define and run the event
    post_natal_sup = postnatal_supervisor.PostnatalSupervisorEvent(module=sim.modules['PostnatalSupervisor'])
    post_natal_sup.apply(sim.population)

    # check this mother has developed disease as expected
    assert sim.population.props.at[mother_id_onset, 'pn_htn_disorders'] == 'mild_pre_eclamp'

    # check this mother has resolved as expected
    assert sim.population.props.at[mother_id_resolve, 'pn_htn_disorders'] == 'resolved'
    assert (mni[mother_id_resolve]['hypertension_resolution'] == sim.date)

    # move date forward 1 week
    sim.date = sim.date + pd.DateOffset(weeks=1)

    # now force woman with new disease to progress to severe disease (block resolution and force care seeking)
    params['probs_for_mpe_matrix_pn'] = [0, 0, 0, 0, 1]
    params['prob_care_seeking_postnatal_emergency'] = 1
    params['prob_htn_resolves'] = 0

    # run the event and check disease has progressed
    post_natal_sup.apply(sim.population)
    assert sim.population.props.at[mother_id_onset, 'pn_htn_disorders'] == 'eclampsia'

    # Check shes correctly sort care
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(mother_id_onset) if
        isinstance(ev[1], labour.HSI_Labour_ReceivesPostnatalCheck)
    ][0]
    assert date_event == sim.date

    # now reset her disease to a more mild form, block care seeking and run the event again
    sim.population.props.at[mother_id_onset, 'pn_htn_disorders'] = 'mild_pre_eclamp'
    params['prob_care_seeking_postnatal_emergency'] = 0
    params['cfr_eclampsia_pn'] = 1

    # todo this doesnt work and i dont know why!!!!!!!!!!!!!!!!!!!
    post_natal_sup.apply(sim.population)

    # check this time that she will die as she didnt seek care
    assert not sim.population.props.at[mother_id_onset, 'is_alive']

# todo: effect of orals on reduced risk of progression
# todo: death from severe hypertension


def test_application_of_risk_of_late_onset_neonatal_sepsis():
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # Set parameters to force onset of disease
    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_late_onset_neonatal_sepsis'] = 1
    params['prob_care_seeking_postnatal_emergency_neonate'] = 1
    params['treatment_effect_early_init_bf'] = 1

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    mother_id = get_mother_id_from_dataframe(sim)
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    sim.event_queue.queue.clear()
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    # set child and mother to be in week 2 postnatal
    sim.population.props.at[mother_id, 'la_date_most_recent_delivery'] = sim.date - pd.DateOffset(days=10)
    sim.population.props.at[child_id, 'age_days'] = 10
    sim.population.props.at[child_id, 'date_of_birth'] = sim.start_date + pd.DateOffset(days=10)

    # define and run the event
    post_natal_sup = postnatal_supervisor.PostnatalSupervisorEvent(module=sim.modules['PostnatalSupervisor'])
    post_natal_sup.apply(sim.population)

    # check he's developed late sepsis
    assert sim.population.props.at[child_id, 'pn_sepsis_late_neonatal']

    # and care has been sought
    date_event, event = [
            ev for ev in sim.modules['HealthSystem'].find_events_for_person(child_id) if
            isinstance(ev[1], newborn_outcomes.HSI_NewbornOutcomes_ReceivesPostnatalCheck)
        ][0]
    assert date_event == sim.date

    # reset the property and set care seeking to 0, risk of death to 1
    sim.population.props.at[child_id, 'pn_sepsis_late_neonatal'] = False
    params['prob_care_seeking_postnatal_emergency_neonate'] = 0
    params['cfr_late_neonatal_sepsis'] = 1

    # run event again but check the child has died
    post_natal_sup.apply(sim.population)
    assert not sim.population.props.at[child_id, 'is_alive']
