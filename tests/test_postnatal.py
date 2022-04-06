import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_helper_functions,
    pregnancy_supervisor,
    symptommanager,
)

start_date = Date(2010, 1, 1)

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
        df.at[person, 'is_pregnant'] = True
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], person)
        pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], person)
        df.at[person, 'is_pregnant'] = False

    return women_repro


def get_mother_id_from_dataframe(sim):
    """Return individual id from dataframe for postnatal testing"""
    df = sim.population.props
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]
    df.at[mother_id, 'is_pregnant'] = True
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date

    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)
    df.at[mother_id, 'is_pregnant'] = False

    return mother_id


def register_core_modules(sim):
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           cons_availability='all'),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),

                 hiv.DummyHivModule(),
                 )

    return sim


@pytest.mark.slow
def test_run_and_check_dtypes(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})
    register_core_modules(sim)
    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(2015, 1, 1))
    check_dtypes(sim)

    # check that no errors have been logged during the simulation run
    output = parse_log_file(sim.log_filepath)
    assert 'error' not in output['tlo.methods.postnatal_supervisor']


def test_antenatal_disease_is_correctly_carried_over_to_postnatal_period_on_birth(seed):
    """Test that complications which may continue from the antenatal period to the postnatal period transition as
    expected"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_core_modules(sim)
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    mother_id = get_mother_id_from_dataframe(sim)

    # Set properties from the antenatal period that we would expect to be carried into the postnatal period, if they
    # dont resolve before or on birth
    df = sim.population.props
    df.at[mother_id, 'ac_gest_htn_on_treatment'] = True

    df.at[mother_id, 'ps_anaemia_in_pregnancy'] = 'moderate'
    df.at[mother_id, 'ps_htn_disorders'] = 'mild_pre_eclamp'
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['delivery_setting'] = 'health_centre'

    # Run the birth events
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    # check the properties are carried across into new postnatal properties
    assert sim.population.props.at[mother_id, 'la_gest_htn_on_treatment']
    assert sim.population.props.at[mother_id, 'pn_anaemia_following_pregnancy'] == 'moderate'
    assert sim.population.props.at[mother_id, 'pn_htn_disorders'] == 'mild_pre_eclamp'


def test_application_of_maternal_complications_and_care_seeking_postnatal_week_one_event(seed):
    """Test that risk of complications is correctly applied in the first week postnatal and that women seek care as
    expected"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_core_modules(sim)
    sim.make_initial_population(n=100)

    # set risk of maternal complications (occuring in week one) to one to insure risk applied as expected
    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_secondary_pph'] = 1.0
    params['prob_late_sepsis_endometritis'] = 1.0
    params['prob_late_sepsis_urinary_tract'] = 1.0
    params['prob_late_sepsis_skin_soft_tissue'] = 1.0
    params['baseline_prob_anaemia_per_week'] = 1.0
    params['prob_type_of_anaemia_pn'] = [1, 0, 0]
    params['weekly_prob_gest_htn_pn'] = 1.0
    params['prob_care_seeking_postnatal_emergency'] = 1.0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
    df = sim.population.props

    # Get id number of mother
    mother_id = get_mother_id_from_dataframe(sim)
    df.at[mother_id, 'la_is_postpartum'] = True
    df.at[mother_id, 'la_date_most_recent_delivery'] = sim.date - pd.DateOffset(days=2)
    sim.modules['Labour'].women_in_labour.append(mother_id)

    # define and run the event
    postnatal_week_one = postnatal_supervisor.PostnatalWeekOneMaternalEvent(
        individual_id=mother_id, module=sim.modules['PostnatalSupervisor'])
    postnatal_week_one.apply(mother_id)

    assert mni[mother_id]['passed_through_week_one']

    # check that complications have been correctly set (storing dalys where appropriate)
    assert sim.population.props.at[mother_id, 'pn_sepsis_late_postpartum']
    assert (mni[mother_id]['sepsis_onset'] == sim.date)
    assert mni[mother_id]['endo_pp']

    assert sim.population.props.at[mother_id, 'pn_postpartum_haem_secondary']
    assert (mni[mother_id]['secondary_pph_onset'] == sim.date)

    assert (sim.population.props.at[mother_id, 'pn_anaemia_following_pregnancy'] == 'mild')

    assert sim.population.props.at[mother_id, 'pn_htn_disorders'] == 'gest_htn'

    # Check HSI has been correctly scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(mother_id) if
        isinstance(ev[1], labour.HSI_Labour_ReceivesPostnatalCheck)
    ][0]
    assert date_event == sim.date

# todo: htn progression/resolution


def test_application_of_neonatal_complications_and_care_seeking_postnatal_week_one_event(seed):
    """Test that risk of complications is correctly applied in the first week postnatal and that women seek care as
    expected"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_core_modules(sim)
    sim.make_initial_population(n=100)

    # set risk of newborn complications (occuring in week one) to one to insure risk applied as expected
    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_early_onset_neonatal_sepsis_week_1'] = 1.0
    params['prob_care_seeking_postnatal_emergency_neonate'] = 1.0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # Get id number of mother
    mother_id = get_mother_id_from_dataframe(sim)

    # Set mother to be pregnant with twins to test twin logic of this event
    sim.population.props.at[mother_id, 'ps_multiple_pregnancy'] = True
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['delivery_setting'] = 'health_centre'

    # Run birth event to generate child
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)
    sim.modules['NewbornOutcomes'].newborn_care_info[child_id]['will_receive_pnc'] = 'late'

    # define and run the event using the mother_id as this is how its coded
    postnatal_week_one = postnatal_supervisor.PostnatalWeekOneNeonatalEvent(
        individual_id=child_id, module=sim.modules['PostnatalSupervisor'])
    postnatal_week_one.apply(child_id)

    assert sim.modules['NewbornOutcomes'].newborn_care_info[child_id]['passed_through_week_one']
    assert sim.population.props.at[child_id, 'pn_sepsis_early_neonatal']

    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(child_id) if
        isinstance(ev[1], newborn_outcomes.HSI_NewbornOutcomes_ReceivesPostnatalCheck)
    ][0]
    assert date_event == sim.date


# Todo: test to check things ARNENT sheduled

def test_all_appropriate_pregnancy_variables_are_reset_at_then_end_of_postnatal():
    pass


def test_application_of_risk_of_death_to_mothers_postnatal_week_one_event(seed):
    """Test that risk of death is applied to women in the first week postnatal in the context of complications, as
    expected"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_core_modules(sim)
    sim.make_initial_population(n=100)

    # set risk of complications at 1 so woman is at risk of death
    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_secondary_pph'] = 1.0
    params['prob_late_sepsis_endometritis'] = 1.0
    params['prob_late_sepsis_urinary_tract_inf'] = 1.0

    # Prevent care seeking and set risk of death due to comps as 1
    params['prob_care_seeking_postnatal_emergency'] = 0.0
    params['cfr_secondary_postpartum_haemorrhage'] = 1.0
    params['cfr_postpartum_sepsis'] = 1.0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # Get id number of mother
    mother_id = get_mother_id_from_dataframe(sim)
    df.at[mother_id, 'la_is_postpartum'] = True
    df.at[mother_id, 'la_date_most_recent_delivery'] = sim.date - pd.DateOffset(days=2)
    sim.modules['Labour'].women_in_labour.append(mother_id)

    # Run the event, as care seeking is blocked risk of death should be applied in the event and then carried out
    postnatal_week_one = postnatal_supervisor.PostnatalWeekOneMaternalEvent(
        individual_id=mother_id, module=sim.modules['PostnatalSupervisor'])
    postnatal_week_one.apply(mother_id)

    assert not sim.population.props.at[mother_id, 'is_alive']


def test_application_of_risk_of_death_to_neonates_postnatal_week_one_event(seed):
    """Test that risk of death is applied to neonates in the first week postnatal in the context of complications, as
    expected"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_core_modules(sim)
    sim.make_initial_population(n=100)

    # set risk of complications at 1 so woman is at risk of death
    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_early_onset_neonatal_sepsis_week_1'] = 1.0

    # Prevent care seeking and set risk of death due to comps as 1
    params['prob_care_seeking_postnatal_emergency_neonate'] = 0.0
    params['cfr_early_onset_neonatal_sepsis'] = 1.0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # Get id number of newborns
    mother_id = get_mother_id_from_dataframe(sim)
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['delivery_setting'] = 'health_centre'

    child_id_one = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id_one)

    # Run the event, as care seeking is blocked risk of death should be applied in the event and then carried out
    postnatal_week_one = postnatal_supervisor.PostnatalWeekOneNeonatalEvent(
        individual_id=mother_id, module=sim.modules['PostnatalSupervisor'])
    postnatal_week_one.apply(child_id_one)

    assert sim.population.props.at[child_id_one, 'pn_sepsis_early_neonatal']
    assert not sim.population.props.at[child_id_one, 'is_alive']


def test_application_of_risk_of_infection_and_sepsis_postnatal_supervisor_event(seed):
    """Test that risk of maternal infection is applied within the population level postnatal event as expected,
     including care seeking and application of risk of death"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_core_modules(sim)
    sim.make_initial_population(n=100)

    # set risk of infection and sepsis to 1
    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_late_sepsis_endometritis'] = 1.0
    params['prob_late_sepsis_urinary_tract_inf'] = 1.0
    params['prob_late_sepsis_skin_soft_tissue_inf'] = 1.0

    # and risk of care seeking to 1
    params['prob_care_seeking_postnatal_emergency'] = 1.0

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
    params['prob_care_seeking_postnatal_emergency'] = 0.0
    params['cfr_postpartum_sepsis'] = 1.0

    # call the event again
    post_natal_sup.apply(sim.population)

    # check women are scheduled for death not careseeking
    assert not sim.population.props.at[mother_id, 'is_alive']


def test_application_of_risk_of_spph_postnatal_supervisor_event(seed):
    """Test that risk of maternal haemorrhage is applied within the population level postnatal event as expected,
    including care seeking and application of risk of death"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_core_modules(sim)
    sim.make_initial_population(n=100)

    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_secondary_pph'] = 1.0
    params['prob_care_seeking_postnatal_emergency'] = 1.0
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
    params['prob_care_seeking_postnatal_emergency'] = 0.0
    params['cfr_secondary_postpartum_haemorrhage'] = 1.0

    # call the event again
    post_natal_sup.apply(sim.population)

    # check women are scheduled for death not care seeking
    assert not sim.population.props.at[mother_id, 'is_alive']


def test_application_of_risk_of_anaemia_postnatal_supervisor_event(seed):
    """Test that risk of maternal anaemia is applied within the population level postnatal event as expected"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_core_modules(sim)
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['baseline_prob_anaemia_per_week'] = 1.0
    params['prob_type_of_anaemia_pn'] = [1, 0, 0]

    postnatal_women = generate_postnatal_women(sim)

    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    # run the event
    post_natal_sup = postnatal_supervisor.PostnatalSupervisorEvent(module=sim.modules['PostnatalSupervisor'])
    post_natal_sup.apply(sim.population)

    # check the properties are set correctly
    mother_id = postnatal_women.index[0]
    assert sim.population.props.at[mother_id, 'pn_anaemia_following_pregnancy'] == 'mild'
    assert (mni[mother_id]['mild_anaemia_pp_onset'] == sim.date)


def test_application_of_risk_of_hypertensive_disorders_postnatal_supervisor_event(seed):
    """Test that risk of maternal hypertensive disorders is applied within the population level postnatal event as
     expected, including care seeking and application of risk of death"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_core_modules(sim)
    sim.make_initial_population(n=100)

    # Set parameters to force resolution and onset of disease
    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_htn_resolves'] = 1.0
    params['weekly_prob_pre_eclampsia_pn'] = 1.0

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
    params['prob_care_seeking_postnatal_emergency'] = 1.0
    params['prob_htn_resolves'] = 0.0

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
    params['prob_care_seeking_postnatal_emergency'] = 0.0
    params['cfr_eclampsia'] = 1.0

    # todo this doesnt work and i dont know why!!!!!!!!!!!!!!!!!!!
    post_natal_sup.apply(sim.population)

    # check this time that she will die as she didnt seek care
    assert not sim.population.props.at[mother_id_onset, 'is_alive']

# todo: effect of orals on reduced risk of progression
# todo: death from severe hypertension


def test_application_of_risk_of_late_onset_neonatal_sepsis(seed):
    """Test that risk of neonatal sepsis is applied within the population level postnatal event as
    expected, including care seeking and application of risk of death"""
    sim = Simulation(start_date=start_date, seed=seed)
    register_core_modules(sim)
    sim.make_initial_population(n=100)

    # Set parameters to force onset of disease
    params = sim.modules['PostnatalSupervisor'].current_parameters
    params['prob_late_onset_neonatal_sepsis'] = 1.0
    params['prob_care_seeking_postnatal_emergency_neonate'] = 1.0
    params['treatment_effect_early_init_bf'] = 1.0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    mother_id = get_mother_id_from_dataframe(sim)
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['delivery_setting'] = 'health_centre'

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
    params['prob_care_seeking_postnatal_emergency_neonate'] = 0.0
    params['cfr_late_neonatal_sepsis'] = 1.0

    # run event again but check the child has died
    post_natal_sup.apply(sim.population)
    assert not sim.population.props.at[child_id, 'is_alive']
