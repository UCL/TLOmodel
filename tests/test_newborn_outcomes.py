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

seed = 8974


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


def find_and_return_hsi_events_list(sim, individual_id):
    """Returns HSI event list for an individual"""
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=individual_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    return hsi_events


def check_newborn_death_function_acts_as_expected(sim, individual_id, cause):
    """Calls the apply_risk_of_death_from_complication function and checks that this function sets the correct variables
     in the DF and NCI to signify a child has died due to the complication passed to the function"""
    nci = sim.modules['NewbornOutcomes'].newborn_care_info

    sim.modules['NewbornOutcomes'].apply_risk_of_death_from_complication(individual_id, complication=cause)
    assert sim.population.props.at[individual_id, 'nb_death_after_birth']
    assert (sim.population.props.at[individual_id, 'nb_death_after_birth_date'] == sim.date)
    assert cause in nci[individual_id]['cause_of_death_after_birth']

    # Reset the properties as this function is looped over for all possible complications
    sim.population.props.at[individual_id, 'nb_death_after_birth'] = False
    sim.population.props.at[individual_id, 'nb_death_after_birth_date'] = pd.NaT


def register_modules(ignore_cons_constraints):
    """Register all modules that are required for newborn outcomes to run"""
    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*'],
                                           ignore_cons_constraints=ignore_cons_constraints),
                 joes_fake_props_module.JoesFakePropsModule(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

    return sim


def test_run_and_check_dtypes():
    """Run the sim for five years and check dtypes at the end """
    sim = register_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(2015, 1, 1))

    check_dtypes(sim)


def test_to_check_babies_delivered_in_facility_receive_post_birth_care():
    """Test that babies that are born within a health facility are correctly scheduled to receive post delivery care
    following birth """
    sim = register_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    # Define key variables of the mother
    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 38

    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)
    sim.modules['Labour'].set_labour_mni_variables(mother_id)

    # Set the variable that the mother has delivered at a facility
    mni[mother_id]['delivery_setting'] = 'hospital'

    # Run the birth event
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    # check scheduling
    hsi_events = find_and_return_hsi_events_list(sim, child_id)
    assert newborn_outcomes.HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendantAtBirth in hsi_events


def test_to_check_babies_delivered_at_home_dont_receive_post_birth_care():
    sim = register_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 38

    # Populate the minimum set of keys within the mni dict so the on_birth function will run
    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)
    sim.modules['Labour'].set_labour_mni_variables(mother_id)

    # Set the variable that the mother has delivered at home
    mni[mother_id]['delivery_setting'] = 'home_birth'
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    # check the baby is term as expected
    assert not sim.population.props.at[child_id, 'nb_early_preterm']
    assert not sim.population.props.at[child_id, 'nb_late_preterm']

    # Check they are not scheduled to receive post birth care
    hsi_events = find_and_return_hsi_events_list(sim, child_id)
    assert newborn_outcomes.HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendantAtBirth not in hsi_events


def test_care_seeking_for_babies_delivered_at_home_who_develop_complications():
    sim = register_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # set risk of comps to 1 and force care seeking
    params = sim.modules['NewbornOutcomes'].current_parameters
    params['prob_early_onset_neonatal_sepsis_day_0'] = 1
    params['prob_early_breastfeeding_hb'] = 0
    params['prob_pnc_check_newborn'] = 1

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 38

    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)
    sim.modules['Labour'].set_labour_mni_variables(mother_id)

    # set delivery setting for the mother as home_birth
    mni[mother_id]['delivery_setting'] = 'home_birth'
    child_id = sim.do_birth(mother_id)

    # Run the on birth function and check the baby has developed the complication and care will be sought
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)
    assert sim.population.props.at[child_id, 'nb_early_onset_neonatal_sepsis']

    # Check the event is scheduled
    hsi_events = find_and_return_hsi_events_list(sim, child_id)
    assert newborn_outcomes.HSI_NewbornOutcomes_ReceivesPostnatalCheck in hsi_events



def test_twin_and_single_twin_still_birth_logic_for_twins():

    sim = register_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]

    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 38

    # Show mother is pregnant with twins and has lost one twin during labour
    df.at[mother_id, 'ps_multiple_pregnancy'] = True
    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)
    sim.modules['Labour'].set_labour_mni_variables(mother_id)
    mni[mother_id]['single_twin_still_birth'] = True

    # Define the children and run the newborn outcomes on_birth for each twin
    child_id_one = sim.do_birth(mother_id)
    child_id_two = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id_one)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id_two)
    sim.modules['NewbornOutcomes'].link_twins(child_id_one, child_id_two, mother_id)

    # check linking
    assert sim.population.props.at[child_id_one, 'nb_is_twin']
    assert sim.population.props.at[child_id_two, 'nb_is_twin']

    assert (sim.population.props.at[child_id_one, 'nb_twin_sibling_id'] == child_id_two)
    assert (sim.population.props.at[child_id_two, 'nb_twin_sibling_id'] == child_id_one)

    # Check that the event correctly logged twins in the mni
    assert (mni[mother_id]['twin_count'] == 2)

    # And using that logging registered that one twin had died intrapartum and scheduled the death event accordingly
    assert not sim.population.props.at[child_id_two, 'is_alive']


def test_care_seeking_for_twins_delivered_at_home_who_develop_complications():
    sim = register_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # set risk of complications to 1 so that both twins develop sepsis immediately following birth and set probability
    # of care seeking to 1
    params = sim.modules['NewbornOutcomes'].current_parameters
    params['prob_early_onset_neonatal_sepsis_day_0'] = 1
    params['prob_early_breastfeeding_hb'] = 0
    params['prob_pnc_check_newborn'] = 1

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    # Set key variables for the mother
    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 38
    df.at[mother_id, 'ps_multiple_pregnancy'] = True
    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)
    sim.modules['Labour'].set_labour_mni_variables(mother_id)

    mni[mother_id]['delivery_setting'] = 'home_birth'

    # Define the children and run the newborn outcomes on_birth for each twin
    child_id_one = sim.do_birth(mother_id)
    child_id_two = sim.do_birth(mother_id)

    # Run the on_birth function twice as would occur for twins
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id_one)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id_two)

    # check both twins have developed complications as expected
    assert sim.population.props.at[child_id_one, 'nb_early_onset_neonatal_sepsis']
    assert sim.population.props.at[child_id_two, 'nb_early_onset_neonatal_sepsis']

    # and that care will be sought
    assert sim.modules['NewbornOutcomes'].newborn_care_info[child_id_one]['will_receive_pnc'] == 'early'
    assert sim.modules['NewbornOutcomes'].newborn_care_info[child_id_two]['will_receive_pnc'] == 'early'

    # Check the event is scheduled for both twins
    hsi_events_child_one = find_and_return_hsi_events_list(sim, child_id_one)
    hsi_events_child_two = find_and_return_hsi_events_list(sim, child_id_two)

    assert newborn_outcomes.HSI_NewbornOutcomes_ReceivesPostnatalCheck in hsi_events_child_one
    assert newborn_outcomes.HSI_NewbornOutcomes_ReceivesPostnatalCheck in hsi_events_child_two


def test_on_birth_applies_risk_of_complications_and_death_in_term_newborns_delivered_at_home_correctly():
    sim = register_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # set risk of comps to 1 and force care seeking
    params = sim.modules['NewbornOutcomes'].current_parameters
    params['prob_early_onset_neonatal_sepsis_day_0'] = 1
    params['treatment_effect_early_init_bf'] = 1
    params['prob_failure_to_transition'] = 1
    params['prob_encephalopathy'] = 1
    params['prob_enceph_severity'] = [0, 0, 1]
    params['prob_congenital_heart_anomaly'] = 1
    params['prob_limb_musc_skeletal_anomaly'] = 1
    params['prob_urogenital_anomaly'] = 1
    params['prob_digestive_anomaly'] = 1
    params['prob_other_anomaly'] = 1

    params['prob_pnc_check_newborn'] = 0

    params['cfr_preterm_birth'] = 1
    params['cfr_failed_to_transition'] = 1
    params['cfr_enceph'] = 1
    params['cfr_neonatal_sepsis'] = 1

    # Also set the risk of preterm comps to 1 and check these are not applied to this newborn as they are born term
    params['prob_retinopathy_preterm'] = 1
    params['prob_respiratory_distress_preterm'] = 1

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # Set key maternal variables
    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 38
    sim.modules['PregnancySupervisor'].generate_mother_and_newborn_dictionary_for_individual(mother_id)
    sim.modules['Labour'].set_labour_mni_variables(mother_id)
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['delivery_setting'] = 'home_birth'

    # run on_birth
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    # Check that the newborn has developed all the complications as expected
    assert sim.population.props.at[child_id, 'nb_early_onset_neonatal_sepsis']
    assert sim.population.props.at[child_id, 'nb_not_breathing_at_birth']
    assert (sim.population.props.at[child_id, 'nb_encephalopathy'] == 'severe_enceph')

    assert sim.modules['NewbornOutcomes'].congeintal_anomalies.has_all(mother_id, 'heart')
    assert sim.modules['NewbornOutcomes'].congeintal_anomalies.has_all(mother_id, 'limb_musc_skeletal')
    assert sim.modules['NewbornOutcomes'].congeintal_anomalies.has_all(mother_id, 'urogenital')
    assert sim.modules['NewbornOutcomes'].congeintal_anomalies.has_all(mother_id, 'digestive')
    assert sim.modules['NewbornOutcomes'].congeintal_anomalies.has_all(mother_id, 'other')

    # Ensure no complications specific to preterm newborns have occured to this term baby (despite risk being set to 1)
    assert not sim.population.props.at[child_id, 'nb_preterm_respiratory_distress']
    assert (sim.population.props.at[child_id, 'nb_retinopathy_prem'] == 'none')

test_on_birth_applies_risk_of_complications_and_death_in_term_newborns_delivered_at_home_correctly()

"""
def test_on_birth_applies_risk_of_complications_and_death_in_preterm_newborns_delivered_at_home_correctly():

    sim = register_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    # set risk of comps to 1 and force care seeking
    params = sim.modules['NewbornOutcomes'].parameters
    params['prob_early_onset_neonatal_sepsis_day_0'] = 1
    params['prob_early_breastfeeding_hb'] = 0
    params['prob_failure_to_transition'] = 1
    params['prob_retinopathy_preterm'] = 1
    params['prob_respiratory_distress_preterm'] = 1
    params['prob_retinopathy_severity'] = [0, 0, 0, 1]
    params['prob_congenital_ba'] = 1
    params['prev_types_of_ca'] = [1, 0, 0, 0, 0, 0, 0]

    # Also set the risk of term comps to 1 and check these are not applied to this newborn as they are born preterm
    params['prob_encephalopathy'] = 1
    params['prob_enceph_severity'] = [0, 0, 1]

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 32
    set_min_mni_keys_for_on_birth_to_run(sim, mother_id)

    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['delivery_setting'] = 'home_birth'
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['labour_state'] = 'early_preterm_labour'

    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    assert sim.population.props.at[child_id, 'nb_early_onset_neonatal_sepsis']
    assert (sim.population.props.at[child_id, 'nb_congenital_anomaly'] == 'musculoskeletal')
    assert sim.population.props.at[child_id, 'nb_preterm_respiratory_distress']
    assert sim.population.props.at[child_id, 'nb_not_breathing_at_birth']
    assert (sim.population.props.at[child_id, 'nb_retinopathy_prem'] == 'blindness')

    assert (sim.population.props.at[child_id, 'nb_encephalopathy'] == 'none')


def test_newborn_hsi_applies_risk_of_complications_and_delivers_treatment_to_facility_births():

    sim = register_modules(ignore_cons_constraints=True)
    sim.make_initial_population(n=100)

    # set risk of comps very high and force care seeking
    params = sim.modules['NewbornOutcomes'].parameters
    params['prob_early_onset_neonatal_sepsis_day_0'] = 1
    params['treatment_effect_clean_birth'] = 1
    params['treatment_effect_cord_care'] = 1
    params['treatment_effect_early_init_bf'] = 1
    params['treatment_effect_abx_prom'] = 1
    params['prob_early_breastfeeding_hf'] = 0
    params['prob_failure_to_transition'] = 1
    params['prob_congenital_ba'] = 1
    params['prev_types_of_ca'] = [1, 0, 0, 0, 0, 0, 0]
    params['prob_encephalopathy'] = 1
    params['prob_enceph_severity'] = [0, 0, 1]

    # set probabilities that effect delivery of treatment to 1
    params['sensitivity_of_assessment_of_neonatal_sepsis_hc'] = 1.0
    params['sensitivity_of_assessment_of_ftt_hc'] = 1.0
    params['sensitivity_of_assessment_of_lbw_hc'] = 1.0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # set maternal variables
    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    set_min_mni_keys_for_on_birth_to_run(sim, mother_id)
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['delivery_setting'] = 'health_centre'

    # do the birth
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    # Ensure risk of complications is not applied during the on birth function (as expected)
    assert not sim.population.props.at[child_id, 'nb_early_onset_neonatal_sepsis']
    assert not sim.population.props.at[child_id, 'nb_not_breathing_at_birth']
    assert (sim.population.props.at[child_id, 'nb_encephalopathy'] == 'none')

    # set this baby to be low birthweight to check KMC
    sim.population.props.at[child_id, 'nb_low_birth_weight_status'] = 'low_birth_weight'

    # Run the newborn care event
    newborn_care = newborn_outcomes.HSI_NewbornOutcomes_CareOfTheNewbornBySkilledAttendantAtBirth(
        person_id=child_id, module=sim.modules['NewbornOutcomes'], facility_level_of_this_hsi=2)
    newborn_care.apply(person_id=child_id, squeeze_factor=0.0)

    # check that the risk of complications has correctly been applied
    assert sim.population.props.at[child_id, 'nb_early_onset_neonatal_sepsis']
    assert sim.population.props.at[child_id, 'nb_not_breathing_at_birth']
    assert (sim.population.props.at[child_id, 'nb_encephalopathy'] == 'severe_enceph')

    # And that the complications have been successfully identified and treatment delivered
    assert sim.population.props.at[child_id, 'nb_received_cord_care']
    assert sim.population.props.at[child_id, 'nb_kangaroo_mother_care']
    assert sim.population.props.at[child_id, 'nb_received_neonatal_resus']
    assert sim.population.props.at[child_id, 'nb_inj_abx_neonatal_sepsis']


def test_function_which_applies_risk_of_death_following_birth():

    sim = register_modules(ignore_cons_constraints=False)
    sim.make_initial_population(n=100)

    params = sim.modules['NewbornOutcomes'].parameters
    params['cfr_preterm_birth'] = 1
    params['cfr_failed_to_transition'] = 1
    params['cfr_mild_enceph'] = 1
    params['cfr_moderate_enceph'] = 1
    params['cfr_severe_enceph'] = 1
    params['cfr_congenital_anomaly'] = 1
    params['cfr_rds_preterm'] = 1
    params['cfr_neonatal_sepsis'] = 1

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    set_min_mni_keys_for_on_birth_to_run(sim, mother_id)
    child_id = sim.do_birth(mother_id)

    # do some coarse checks on the function that lives within the death event which applies risk of death
    for cause in ['neonatal_sepsis', 'mild_enceph', 'moderate_enceph', 'severe_enceph', 'respiratory_distress',
                  'preterm_birth_other', 'not_breathing_at_birth', 'congenital_anomaly']:
        check_newborn_death_function_acts_as_expected(sim, individual_id=child_id, cause=cause)

    # set complications that should cause death
    sim.population.props.at[child_id, 'nb_early_onset_neonatal_sepsis'] = True

    sim.modules['NewbornOutcomes'].set_death_status(child_id)

    # Check the event has correctly scheduled the instantaneous death event
    events = sim.find_events_for_person(person_id=child_id)
    events = [e.__class__ for d, e in events]
    assert demography.InstantaneousDeath in events

# todo: test breastfeeding logic
# todo: test daly output
# todo: hsi did_not_run behaves as expected
# todo:treatment blocks death (?)"""
