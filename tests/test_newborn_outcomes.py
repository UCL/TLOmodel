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


def find_and_return_hsi_events_list(sim, individual_id):
    """Returns HSI event list for an individual"""
    health_system = sim.modules['HealthSystem']
    hsi_events = health_system.find_events_for_person(person_id=individual_id)
    hsi_events = [e.__class__ for d, e in hsi_events]
    return hsi_events


def register_modules(sim):
    """Register all modules that are required for newborn outcomes to run"""

    sim.register(demography.Demography(),
                 contraception.Contraception(),
                 enhanced_lifestyle.Lifestyle(),
                 healthburden.HealthBurden(),
                 healthsystem.HealthSystem(service_availability=['*'], cons_availability='all'),
                 newborn_outcomes.NewbornOutcomes(),
                 pregnancy_supervisor.PregnancySupervisor(),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(),
                 symptommanager.SymptomManager(),
                 labour.Labour(),
                 postnatal_supervisor.PostnatalSupervisor(),
                 healthseekingbehaviour.HealthSeekingBehaviour(),
                 hiv.DummyHivModule(),
                 )

    return sim


@pytest.mark.slow
def test_run_and_check_dtypes(tmpdir, seed):
    """Run the sim for five years and check dtypes at the end """
    sim = Simulation(start_date=start_date, seed=seed,
                     log_config={"filename": "log", "directory": tmpdir}, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(2015, 1, 1))
    check_dtypes(sim)

    # check that no errors have been logged during the simulation run
    output = parse_log_file(sim.log_filepath)
    assert 'error' not in output['tlo.methods.newborn_outcomes']


def test_care_seeking_for_babies_delivered_at_home_who_develop_complications(seed):
    """Test that babies that are born at home and develop complications will have care sought for them as expected
    """
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=100)

    # set risk of comps to 1 and force care seeking
    params = sim.modules['NewbornOutcomes'].parameters
    params['prob_early_onset_neonatal_sepsis_day_0'] = 1.0
    params['prob_early_breastfeeding_hb'] = 0.0
    params['prob_care_seeking_for_complication'] = 1.0
    params['prob_timings_pnc_newborns'] = [[1.0, 0.0], [1.0, 0.0]]

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 38
    df.at[mother_id, 'is_pregnant'] = True

    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)

    # set delivery setting for the mother as home_birth
    mni[mother_id]['delivery_setting'] = 'home_birth'
    child_id = sim.do_birth(mother_id)

    # Run the on birth function and check the baby has developed the complication and care will be sought
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)
    df = sim.population.props
    assert df.at[child_id, 'nb_early_onset_neonatal_sepsis']

    # Check the event is scheduled
    hsi_events = find_and_return_hsi_events_list(sim, child_id)
    assert newborn_outcomes.HSI_NewbornOutcomes_ReceivesPostnatalCheck in hsi_events


def test_twin_and_single_twin_still_birth_logic_for_twins(seed):
    """Test that for women who experience a single twin stillbirth only produce one newborn child as expected"""
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]

    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 38
    df.at[mother_id, 'is_pregnant'] = True
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)

    # Show mother is pregnant with twins and has lost one twin during labour
    df.at[mother_id, 'ps_multiple_pregnancy'] = True
    mni[mother_id]['single_twin_still_birth'] = True
    mni[mother_id]['delivery_setting'] = 'health_centre'
    df.at[mother_id, 'la_due_date_current_pregnancy'] = sim.date - pd.DateOffset(days=5)
    df.at[mother_id, 'la_currently_in_labour'] = True
    sim.modules['Labour'].women_in_labour.append(mother_id)

    # Run the birth event
    birth_event = labour.BirthAndPostnatalOutcomesEvent(mother_id=mother_id, module=sim.modules['Labour'])
    birth_event.apply(mother_id)

    # Check that only one child was born, that child is of a twin pair, but has no matched sibling
    df = sim.population.props
    child = df.loc[(df.mother_id == mother_id) & df.is_alive]
    assert len(child) == 1
    for person in child.index:
        assert sim.population.props.at[person, 'nb_is_twin']
        assert sim.population.props.at[person, 'nb_twin_sibling_id'] == -1

    # Check only one twin was born again
    assert (mni[mother_id]['twin_count'] == 1)


def test_care_seeking_for_twins_delivered_at_home_who_develop_complications(seed):
    """Test that for twin births, if both develop a complication and care is sought for one twin, care will be received
    by both"""
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=100)

    # set risk of complications to 1 so that both twins develop sepsis immediately following birth and set probability
    # of care seeking to 1
    params = sim.modules['NewbornOutcomes'].parameters
    params['prob_early_onset_neonatal_sepsis_day_0'] = 1.0
    params['prob_early_breastfeeding_hb'] = 0.0
    params['prob_pnc_check_newborn'] = 1.0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    # Set key variables for the mother
    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 38
    df.at[mother_id, 'ps_multiple_pregnancy'] = True
    df.at[mother_id, 'is_pregnant'] = True

    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)

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


def test_on_birth_applies_risk_of_complications_and_death_in_term_newborns_delivered_at_home_correctly(seed):
    """Test that for neonates born at home that develop complications, care seeking and risk of death is applied as
    expected"""
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=100)

    # set risk of comps to 1 and force care seeking
    params = sim.modules['NewbornOutcomes'].parameters
    params['prob_early_onset_neonatal_sepsis_day_0'] = 1.0
    params['treatment_effect_early_init_bf'] = 1.0
    params['prob_failure_to_transition'] = 1.0
    params['prob_encephalopathy'] = 1.0
    params['prob_enceph_severity'] = [[0, 0, 1],  [0, 0, 1]]
    params['prob_congenital_heart_anomaly'] = 1.0
    params['prob_limb_musc_skeletal_anomaly'] = 1.0
    params['prob_urogenital_anomaly'] = 1.0
    params['prob_digestive_anomaly'] = 1.0
    params['prob_other_anomaly'] = 1.0

    params['prob_pnc_check_newborn'] = 0

    params['cfr_preterm_birth'] = 1.0
    params['cfr_failed_to_transition'] = 1.0
    params['cfr_enceph'] = 1.0
    params['cfr_neonatal_sepsis'] = 1.0

    # Also set the risk of preterm comps to 1 and check these are not applied to this newborn as they are born term
    params['prob_retinopathy_preterm'] = 1.0
    params['prob_respiratory_distress_preterm'] = 1.0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # Set key maternal variables
    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 38
    df.at[mother_id, 'is_pregnant'] = True

    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['delivery_setting'] = 'home_birth'

    # run on_birth
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    # Check that the newborn has developed all the complications as expected
    assert sim.population.props.at[child_id, 'nb_early_onset_neonatal_sepsis']
    assert sim.population.props.at[child_id, 'nb_not_breathing_at_birth']
    assert (sim.population.props.at[child_id, 'nb_encephalopathy'] == 'severe_enceph')

    assert sim.modules['NewbornOutcomes'].congeintal_anomalies.has_all(child_id, 'heart')
    assert sim.modules['NewbornOutcomes'].congeintal_anomalies.has_all(child_id, 'limb_musc_skeletal')
    assert sim.modules['NewbornOutcomes'].congeintal_anomalies.has_all(child_id, 'urogenital')
    assert sim.modules['NewbornOutcomes'].congeintal_anomalies.has_all(child_id, 'digestive')
    assert sim.modules['NewbornOutcomes'].congeintal_anomalies.has_all(child_id, 'other')

    # Ensure no complications specific to preterm newborns have occured to this term baby (despite risk being set to 1)
    assert not sim.population.props.at[child_id, 'nb_preterm_respiratory_distress']
    assert (sim.population.props.at[child_id, 'nb_retinopathy_prem'] == 'none')


def test_on_birth_applies_risk_of_complications_and_death_in_preterm_newborns_delivered_at_home_correctly(seed):
    """Test that for preterm neonates (who are at risk of a different complication set) that born at home and develop
     complications, care seeking and risk of death is applied as expected"""
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=100)

    # set risk of comps to 1 and force care seeking
    params = sim.modules['NewbornOutcomes'].parameters
    params['prob_retinopathy_preterm_early'] = 1.0
    params['prob_respiratory_distress_preterm'] = 1.0

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # update mothers variables
    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 32
    df.at[mother_id, 'is_pregnant'] = True

    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)

    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['delivery_setting'] = 'home_birth'
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['labour_state'] = 'early_preterm_labour'

    # Run the birth event
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    # check complications are applied
    assert sim.population.props.at[child_id, 'nb_preterm_respiratory_distress']
    assert sim.population.props.at[child_id, 'nb_not_breathing_at_birth']

    # check retinopathy risk applied during diasbility function
    sim.modules['NewbornOutcomes'].current_parameters['prob_retinopathy_severity_no_treatment'] = [0, 0, 0, 0, 1]
    sim.modules['NewbornOutcomes'].set_disability_status(child_id)
    assert (sim.population.props.at[child_id, 'nb_retinopathy_prem'] == 'blindness')


def test_sba_hsi_deliveries_resuscitation_treatment_as_expected(seed):
    """ Test that resuscitation treatment is delivered as expected to newborns in respiratory distress who deliver in
    facilities """
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    # set risk of comps very high and force care seeking
    params = sim.modules['NewbornOutcomes'].current_parameters
    params['prob_encephalopathy'] = 1.0
    params['prob_enceph_severity'] = [0, 0, 1]
    params['treatment_effect_resuscitation'] = 0.0
    params['cfr_enceph'] = 1.0
    params['prob_pnc_check_newborn'] = 0.0

    la_params = sim.modules['Labour'].current_parameters
    la_params['prob_hcw_avail_neo_resus'] = 1.0
    la_params['mean_hcw_competence_hc'] = 1.0
    la_params['mean_hcw_competence_hp'] = 1.0


    df = sim.population.props
    mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info

    # set maternal variables
    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'is_pregnant'] = True

    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)

    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['delivery_setting'] = 'health_centre'

    labour_hsi = labour.HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour(
        person_id=mother_id, module=sim.modules['Labour'], facility_level_of_this_hsi=2)
    labour_hsi.apply(person_id=mother_id, squeeze_factor=0.0)

    assert mni[mother_id]['neo_will_receive_resus_if_needed']

    # do the birth
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    assert (sim.population.props.at[child_id, 'nb_encephalopathy'] == 'severe_enceph')
    assert sim.population.props.at[child_id, 'nb_received_neonatal_resus']
    assert sim.population.props.at[child_id, 'is_alive']


def test_newborn_postnatal_check_hsi_delivers_treatment_as_expected(seed):
    """ Test that interventions delivered as part of PNC are delivered as expected to newborns with complications"""
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=100)

    # set risk of comps very high and force care seeking
    params = sim.modules['NewbornOutcomes'].parameters
    la_params = sim.modules['Labour'].parameters

    # set probabilities that effect delivery of treatment to 1
    params['prob_kmc_available'] = [1.0, 1.0]
    la_params['prob_hcw_avail_iv_abx'] = 1.0
    la_params['mean_hcw_competence_hc'] = [[1.0, 1.0], [1.0, 1.0]]
    la_params['mean_hcw_competence_hp'] = [[1.0, 1.0], [1.0, 1.0]]

    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # set maternal variables
    mother_id = df.loc[df.is_alive & (df.sex == "F") & (df.age_years > 14) & (df.age_years < 50)].index[0]
    df.at[mother_id, 'date_of_last_pregnancy'] = sim.date
    df.at[mother_id, 'ps_gestational_age_in_weeks'] = 32
    df.at[mother_id, 'is_pregnant'] = True

    pregnancy_helper_functions.update_mni_dictionary(sim.modules['PregnancySupervisor'], mother_id)
    pregnancy_helper_functions.update_mni_dictionary(sim.modules['Labour'], mother_id)
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['delivery_setting'] = 'health_centre'
    sim.modules['PregnancySupervisor'].mother_and_newborn_info[mother_id]['clean_birth_practices'] = True

    # do the birth
    child_id = sim.do_birth(mother_id)
    sim.modules['NewbornOutcomes'].on_birth(mother_id, child_id)

    sim.population.props.at[child_id, 'nb_early_onset_neonatal_sepsis'] = True
    sim.population.props.at[child_id, 'nb_low_birth_weight_status'] = 'low_birth_weight'
    sim.modules['NewbornOutcomes'].newborn_care_info[child_id]['delivery_setting'] = 'hospital'

    # Run the newborn care event
    newborn_care = newborn_outcomes.HSI_NewbornOutcomes_ReceivesPostnatalCheck(
        person_id=child_id, module=sim.modules['NewbornOutcomes'])
    newborn_care.apply(person_id=child_id, squeeze_factor=0.0)

    assert (sim.population.props.at[child_id, 'nb_pnc_check'] == 1)

    assert sim.population.props.at[child_id, 'nb_supp_care_neonatal_sepsis']
    assert sim.modules['NewbornOutcomes'].newborn_care_info[child_id]['tetra_eye_d']
    assert sim.modules['NewbornOutcomes'].newborn_care_info[child_id]['vit_k']
    assert sim.population.props.at[child_id, 'nb_kangaroo_mother_care']
