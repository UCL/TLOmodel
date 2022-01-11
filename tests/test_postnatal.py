import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)
from tlo.methods.hiv import DummyHivModule

seed = 6987

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

    return women_repro


def get_mother_id_from_dataframe(sim):
    """Return individual id from dataframe for postnatal testing"""
    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    mother_id = women_repro.index[0]

    mni[mother_id] = {
        'twin_count': 0, 'single_twin_still_birth': False, 'labour_state': 'term_labour',
        'stillbirth_in_labour': False, 'abx_for_prom_given': False, 'corticosteroids_given': False,
        'delivery_setting': 'health_centre', 'clean_birth_practices': False, 'sought_care_for_twin_one': False,
        'sepsis_onset': pd.NaT, 'secondary_pph_onset': pd.NaT
    }
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


@pytest.mark.slow
def test_run_and_check_dtypes():
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(2015, 1, 1))
    check_dtypes(sim)


def test_antenatal_disease_is_correctly_carried_over_to_postnatal_period_on_birth():
    """Test that complications which may continue from the antenatal period to the postnatal period transition as e
    xpected"""
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # set key parameters
    params = sim.modules['PostnatalSupervisor'].parameters
    params['prob_htn_persists'] = 1

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
    assert sim.population.props.at[mother_id, 'pn_gest_htn_on_treatment']
    assert sim.population.props.at[mother_id, 'pn_anaemia_following_pregnancy'] == 'moderate'
    assert sim.population.props.at[mother_id, 'pn_htn_disorders'] == 'mild_pre_eclamp'
    assert sim.modules['PostnatalSupervisor'].deficiencies_following_pregnancy.has_all(mother_id, 'iron')
    assert sim.modules['PostnatalSupervisor'].deficiencies_following_pregnancy.has_all(mother_id, 'folate')
    assert sim.modules['PostnatalSupervisor'].deficiencies_following_pregnancy.has_all(mother_id, 'b12')

    # check that the antenatal properties are correctly reset
    assert not sim.population.props.at[mother_id, 'ac_gest_htn_on_treatment']
    assert sim.population.props.at[mother_id, 'ps_anaemia_in_pregnancy'] == 'none'
    assert sim.population.props.at[mother_id, 'ps_htn_disorders'] == 'none'
    assert not sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.has_all(mother_id, 'iron')
    assert not sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.has_all(mother_id, 'folate')
    assert not sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.has_all(mother_id, 'b12')


def test_application_of_complications_and_care_seeking_postnatal_week_one_event():
    """Test that risk of complications is correctly applied in the first week postnatal and that women seek care as
    expected"""
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # set risk of maternal and newborn complications (occuring in week one) to one to insure risk applied as expected
    params = sim.modules['PostnatalSupervisor'].parameters
    params['prob_secondary_pph'] = 1
    params['prob_endometritis_pn'] = 1
    params['prob_urinary_tract_inf_pn'] = 1
    params['prob_skin_soft_tissue_inf_pn'] = 1
    params['prob_other_inf_pn'] = 1
    params['prob_late_sepsis_endometritis'] = 1
    params['prob_late_sepsis_urinary_tract_inf'] = 1
    params['prob_late_sepsis_skin_soft_tissue_inf'] = 1
    params['prob_late_sepsis_other_maternal_infection_pp'] = 1
    params['prob_iron_def_per_week_pn'] = 1
    params['prob_folate_def_per_week_pn'] = 1
    params['prob_b12_def_per_week_pn'] = 1
    params['baseline_prob_anaemia_per_week'] = 1
    params['prob_type_of_anaemia_pn'] = [1, 0, 0]
    params['weekly_prob_gest_htn_pn'] = 1
    params['prob_early_onset_neonatal_sepsis_week_1'] = 1
    params['prob_pnc1_at_day_7'] = 1

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
    assert sim.modules['PostnatalSupervisor'].postpartum_infections_late.has_all(mother_id, 'endometritis')
    assert sim.modules['PostnatalSupervisor'].postpartum_infections_late.has_all(mother_id, 'urinary_tract_inf')
    assert sim.modules['PostnatalSupervisor'].postpartum_infections_late.has_all(mother_id, 'skin_soft_tissue_inf')
    assert sim.modules['PostnatalSupervisor'].postpartum_infections_late.has_all(mother_id, 'other_maternal_infection')

    assert sim.population.props.at[mother_id, 'pn_sepsis_late_postpartum']
    assert (mni[mother_id]['sepsis_onset'] == sim.date)

    assert sim.population.props.at[mother_id, 'pn_postpartum_haem_secondary']
    assert (mni[mother_id]['secondary_pph_onset'] == sim.date)

    assert sim.modules['PostnatalSupervisor'].deficiencies_following_pregnancy.has_all(mother_id, 'iron')
    assert sim.modules['PostnatalSupervisor'].deficiencies_following_pregnancy.has_all(mother_id, 'folate')
    assert sim.modules['PostnatalSupervisor'].deficiencies_following_pregnancy.has_all(mother_id, 'b12')
    assert (sim.population.props.at[mother_id, 'pn_anaemia_following_pregnancy'] == 'mild')

    assert sim.population.props.at[mother_id, 'pn_htn_disorders'] == 'gest_htn'

    assert sim.population.props.at[child_id, 'pn_sepsis_early_neonatal']

    # Check HSI has been correctly scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(mother_id) if
        isinstance(ev[1], postnatal_supervisor.HSI_PostnatalSupervisor_PostnatalCareContactOne)
    ][0]
    assert date_event == sim.date

# todo: twins logic
# todo: htn progression/resolution


def test_application_of_risk_of_death_postnatal_week_one_event():
    """Test that risk of death is applied to women and children who do not seek care for treatment of complications in
    the first week postnatal """
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # set risk of complications at 1 so woman and child are at risk of death
    params = sim.modules['PostnatalSupervisor'].parameters
    params['prob_secondary_pph'] = 1
    params['prob_endometritis_pn'] = 1
    params['prob_urinary_tract_inf_pn'] = 1
    params['prob_skin_soft_tissue_inf_pn'] = 1
    params['prob_other_inf_pn'] = 1
    params['prob_late_sepsis_endometritis'] = 1
    params['prob_late_sepsis_urinary_tract_inf'] = 1
    params['prob_late_sepsis_skin_soft_tissue_inf'] = 1
    params['prob_late_sepsis_other_maternal_infection_pp'] = 1
    params['prob_iron_def_per_week_pn'] = 1
    params['prob_folate_def_per_week_pn'] = 1
    params['prob_b12_def_per_week_pn'] = 1
    params['baseline_prob_anaemia_per_week'] = 1
    params['prob_type_of_anaemia_pn'] = [1, 0, 0]
    params['weekly_prob_gest_htn_pn'] = 1
    params['prob_early_onset_neonatal_sepsis_week_1'] = 1

    # Prevent care seeking and set risk of death due to comps as 1
    params['prob_pnc1_at_day_7'] = 0
    params['cfr_secondary_pph'] = 1
    params['cfr_postnatal_sepsis'] = 1
    params['cfr_early_onset_neonatal_sepsis'] = 1

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # Get id number of mother
    mother_id = get_mother_id_from_dataframe(sim)
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    postnatal_week_one = postnatal_supervisor.PostnatalWeekOneEvent(
        individual_id=mother_id, module=sim.modules['PostnatalSupervisor'])
    postnatal_week_one.apply(mother_id)

    # Check that both mother and newborn have had risk of death immediately applied after failing to seek treatment,
    # and will now die
    assert not sim.population.props.at[mother_id, 'is_alive']
    assert not sim.population.props.at[child_id, 'is_alive']


def test_application_of_risk_of_infection_and_sepsis_postnatal_supervisor_event():
    """Test that risk of infection and sepsis is applied to women via the postnatal supervisor event. Check that women
     seek care and experience risk of death as expected """
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # set risk of infection and sepsis to 1
    params = sim.modules['PostnatalSupervisor'].parameters
    params['prob_secondary_pph'] = 1
    params['prob_endometritis_pn'] = 1
    params['prob_urinary_tract_inf_pn'] = 1
    params['prob_skin_soft_tissue_inf_pn'] = 1
    params['prob_other_inf_pn'] = 1
    params['prob_late_sepsis_endometritis'] = 1
    params['prob_late_sepsis_urinary_tract_inf'] = 1
    params['prob_late_sepsis_skin_soft_tissue_inf'] = 1
    params['prob_late_sepsis_other_maternal_infection_pp'] = 1

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

    # Check that she has developed infections
    assert sim.modules['PostnatalSupervisor'].postpartum_infections_late.has_all(mother_id, 'endometritis')
    assert sim.modules['PostnatalSupervisor'].postpartum_infections_late.has_all(mother_id, 'urinary_tract_inf')
    assert sim.modules['PostnatalSupervisor'].postpartum_infections_late.has_all(mother_id, 'skin_soft_tissue_inf')
    assert sim.modules['PostnatalSupervisor'].postpartum_infections_late.has_all(mother_id, 'other_maternal_infection')

    # and has developed sepsis, DALYS are stored
    assert sim.population.props.at[mother_id, 'pn_sepsis_late_postpartum']
    assert (mni[mother_id]['sepsis_onset'] == sim.date)

    # finally check she will now seek care
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(mother_id) if
        isinstance(ev[1], postnatal_supervisor.HSI_PostnatalSupervisor_PostnatalWardInpatientCare)
    ][0]
    assert date_event == sim.date

    # clear event queues
    sim.event_queue.queue.clear()
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE.clear()

    # reset variables
    sim.population.props.at[mother_id, 'pn_sepsis_late_postpartum'] = False
    sim.modules['PostnatalSupervisor'].postpartum_infections_late.unset(mother_id, 'endometritis')
    sim.modules['PostnatalSupervisor'].postpartum_infections_late.unset(mother_id, 'urinary_tract_inf')
    sim.modules['PostnatalSupervisor'].postpartum_infections_late.unset(mother_id, 'skin_soft_tissue_inf')
    sim.modules['PostnatalSupervisor'].postpartum_infections_late.unset(mother_id, 'other_maternal_infection')

    # set risk of care seeking to 0, risk of death to 1
    params['prob_care_seeking_postnatal_emergency'] = 0
    params['cfr_postnatal_sepsis'] = 1

    # call the event again
    post_natal_sup.apply(sim.population)

    # check women are scheduled for death not careseeking
    assert not sim.population.props.at[mother_id, 'is_alive']


def test_application_of_risk_of_spph_postnatal_supervisor_event():
    """Test that risk of secondary postpartum haemorrhage is applied to women via the postnatal supervisor event. Check
    that women seek care and experience risk of death as expected """
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)
    params = sim.modules['PostnatalSupervisor'].parameters
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

    # and care has been sought
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(mother_id) if
        isinstance(ev[1], postnatal_supervisor.HSI_PostnatalSupervisor_PostnatalWardInpatientCare)
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
    """Test that risk of anaemia is applied to women via the postnatal supervisor event. Check
    that women seek care and experience risk of death as expected """
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)
    params = sim.modules['PostnatalSupervisor'].parameters
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
    """Test that risk of hypertensive disorders is applied to women via the postnatal supervisor event. Check
    that women seek care and experience risk of death as expected """
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # Set parameters to force resolution and onset of disease
    params = sim.modules['PostnatalSupervisor'].parameters
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
        isinstance(ev[1], postnatal_supervisor.HSI_PostnatalSupervisor_PostnatalWardInpatientCare)
    ][0]
    assert date_event == sim.date

    # now reset her disease to a more mild form, block care seeking and run the event again
    sim.population.props.at[mother_id_onset, 'pn_htn_disorders'] = 'mild_pre_eclamp'
    params['prob_care_seeking_postnatal_emergency'] = 0
    params['cfr_eclampsia_pn'] = 1

    post_natal_sup.apply(sim.population)

    # check this time that she will die as she didnt seek care
    assert not df.at[mother_id_onset, 'is_alive']

# todo: effect of orals on reduced risk of progression
# todo: death from severe hypertension


def test_application_of_risk_of_late_onset_neonatal_sepsis():
    """Test that risk of late onset neonatal sepsis is applied to neonates via the postnatal supervisor event. Check
    that they seek care and experience risk of death as expected """
    sim = register_core_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # Set parameters to force onset of disease
    params = sim.modules['PostnatalSupervisor'].parameters
    params['prob_late_onset_neonatal_sepsis'] = 1
    params['prob_care_seeking_postnatal_emergency_neonate'] = 1
    params['treatment_effect_early_init_bf'] = 1

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    mother_id = get_mother_id_from_dataframe(sim)
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    # set child and mother to be in week 2 postnatal
    sim.population.props.at[mother_id, 'la_date_most_recent_delivery'] = sim.date - pd.DateOffset(days=10)
    sim.population.props.at[child_id, 'age_days'] = 10

    # define and run the event
    post_natal_sup = postnatal_supervisor.PostnatalSupervisorEvent(module=sim.modules['PostnatalSupervisor'])
    post_natal_sup.apply(sim.population)

    # check he's developed late sepsis
    assert sim.population.props.at[child_id, 'pn_sepsis_late_neonatal']

    # and care has been sought
    date_event, event = [
            ev for ev in sim.modules['HealthSystem'].find_events_for_person(child_id) if
            isinstance(ev[1], postnatal_supervisor.HSI_PostnatalSupervisor_NeonatalWardInpatientCare)
        ][0]
    assert date_event == sim.date

    # reset the property and set care seeking to 0, risk of death to 1
    sim.population.props.at[child_id, 'pn_sepsis_late_neonatal'] = False
    params['prob_care_seeking_postnatal_emergency_neonate'] = 0
    params['cfr_late_neonatal_sepsis'] = 1

    # run event again but check the child has died
    post_natal_sup.apply(sim.population)
    assert not sim.population.props.at[child_id, 'is_alive']


def test_postnatal_care():
    """Test that routine postnatal care behaves as expected. Test that women and neonates are correctly screened for
    key complications and admitted for futher interventions  """
    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           cons_availability='all'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

    sim.make_initial_population(n=100)

    # Set the parameters which control if interventions are delivered to 1
    params = sim.modules['PostnatalSupervisor'].parameters
    params['prob_intervention_delivered_depression_screen_pnc'] = 1
    params['prob_intervention_delivered_urine_ds_pnc'] = 1
    params['prob_intervention_delivered_bp_pnc'] = 1
    params['prob_intervention_delivered_sep_assessment_pnc'] = 1
    params['prob_intervention_delivered_pph_assessment_pnc'] = 1
    params['prob_intervention_poct_pnc'] = 1
    params['prob_intervention_neonatal_sepsis_pnc'] = 1
    params['prob_intervention_delivered_hiv_test_pnc'] = 1
    params['prob_attend_pnc2'] = 1
    params['sensitivity_bp_monitoring_pn'] = 1.0
    params['specificity_bp_monitoring_pn'] = 1.0
    params['sensitivity_urine_protein_1_plus_pn'] = 1.0
    params['specificity_urine_protein_1_plus_pn'] = 1.0
    params['sensitivity_poc_hb_test_pn'] = 1.0
    params['specificity_poc_hb_test_pn'] = 1.0
    params['sensitivity_maternal_sepsis_assessment'] = 1.0
    params['sensitivity_pph_assessment'] = 1.0
    params['sensitivity_lons_assessment'] = 1.0
    params['sensitivity_eons_assessment'] = 1.0

    dep_params = sim.modules['Depression'].parameters
    dep_params['sensitivity_of_assessment_of_depression'] = 1.0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    mother_id = int(get_mother_id_from_dataframe(sim))
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    # set mother and newborn to experience set of complications
    sim.population.props.at[mother_id, 'pn_htn_disorders'] = 'mild_pre_eclamp'
    sim.population.props.at[mother_id, 'hv_diagnosed'] = True
    sim.population.props.at[mother_id, 'de_depr'] = True
    sim.population.props.at[child_id, 'pn_sepsis_early_neonatal'] = True

    # Define and run the event
    postnatal_care_one = postnatal_supervisor.HSI_PostnatalSupervisor_PostnatalCareContactOne(
        person_id=mother_id, module=sim.modules['PostnatalSupervisor'])
    postnatal_care_one.apply(person_id=mother_id, squeeze_factor=0.0)

    # Check the event has ran and been stored
    assert sim.population.props.at[mother_id, 'pn_pnc_visits_maternal'] == 1
    assert sim.population.props.at[child_id, 'pn_pnc_visits_neonatal'] == 1

    # check her depression has been detected as part of screening
    assert (sim.population.props.at[mother_id, 'de_ever_diagnosed_depression'])

    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=mother_id)
    hsi_events = [e.__class__ for d, e in hsi_events]

    # Check she has correctly been identified as needing admission due to her hypertensive disorder
    assert postnatal_supervisor.HSI_PostnatalSupervisor_PostnatalWardInpatientCare in hsi_events

    # And that she will be scheduled to return for her next PNC visits
    assert postnatal_supervisor.HSI_PostnatalSupervisor_PostnatalCareContactTwo in hsi_events

    # Then check she had her depression correctly identified and started on treatment
    assert depression.HSI_Depression_TalkingTherapy in hsi_events
    assert depression.HSI_Depression_Start_Antidepressant in hsi_events

    # check the child was correctly identified as having sepsis and will be admitted
    hsi_events_newborn = health_system.find_events_for_person(person_id=child_id)
    hsi_events_newborn = [e.__class__ for d, e in hsi_events_newborn]
    assert postnatal_supervisor.HSI_PostnatalSupervisor_NeonatalWardInpatientCare in hsi_events_newborn

    # clear the event queue and reset the properties
    sim.population.props.at[mother_id, 'de_depr'] = False
    sim.population.props.at[child_id, 'pn_sepsis_early_neonatal'] = False

    # set both mother and child to have developed sepsis between visits
    sim.population.props.at[child_id, 'pn_sepsis_late_neonatal'] = True
    sim.population.props.at[mother_id, 'pn_sepsis_late_postpartum'] = True

    # run the second postnatal event
    postnatal_care_two = postnatal_supervisor.HSI_PostnatalSupervisor_PostnatalCareContactTwo(
        person_id=mother_id, module=sim.modules['PostnatalSupervisor'])
    postnatal_care_two.apply(person_id=mother_id, squeeze_factor=0.0)

    # check stored in property
    assert sim.population.props.at[mother_id, 'pn_pnc_visits_maternal'] == 2
    assert sim.population.props.at[child_id, 'pn_pnc_visits_neonatal'] == 2

    # Check the mother and newborn have been correctly referred for treatment as in
    hsi_events = health_system.find_events_for_person(person_id=mother_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    assert postnatal_supervisor.HSI_PostnatalSupervisor_PostnatalWardInpatientCare in hsi_events

    hsi_events_newborn = health_system.find_events_for_person(person_id=child_id)
    hsi_events_newborn = [e.__class__ for d, e in hsi_events_newborn]
    assert postnatal_supervisor.HSI_PostnatalSupervisor_NeonatalWardInpatientCare in hsi_events_newborn

    # Check the child will get HIV testing at final PNC visit as mother is HIV positive
    assert hiv.HSI_Hiv_TestAndRefer in hsi_events_newborn

# todo: postnatal inpatient care
