import os

import pandas as pd

from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.methods import service_integration
from tlo.methods.fullmodel import fullmodel
from tlo.analysis.utils import parse_log_file

# The resource files
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

start_date = Date(2010, 1, 1)


def register_modules(sim):
    """Defines sim variable and registers all modules that can be called when running the full suite of pregnancy
    modules"""

    sim.register(*fullmodel(),
                  service_integration.ServiceIntegration())

def check_cons_processed_params_have_been_overridden(initial_p, updated_p, data):
    mod_cons = ['pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization',
                         'other_modern']  # update with your actual column names

    if data == 'series':
        # Restrict to the relevant subset
        s1_sub = initial_p.loc[mod_cons]
        s2_sub = updated_p.loc[mod_cons]

        # Mask: where s1 > 0
        mask = s1_sub > 0

        # Assert: in these positions, s2 > s1
        condition_ok = (s2_sub[mask] > s1_sub[mask]).all()
        assert condition_ok, "In some positions where series1 > 0, series2 is not greater"

    else:
        for key in initial_p:
            df1 = initial_p[key]
            df2 = updated_p[key]
            # Ensure required columns are present
            for col in mod_cons:
                assert col in df1.columns and col in df2.columns, f"Column '{col}' missing in DataFrame '{key}'"

            # Extract relevant columns
            df1_sub = df1[mod_cons]
            df2_sub = df2[mod_cons]

            # Create mask where df1 > 0
            mask = df1_sub > 0

            # Check df2 > df1 where mask is True
            diff_check = df2_sub > df1_sub

            # Assert condition holds for all cells where df1 > 0
            condition_ok = diff_check[mask].all().all()
            assert condition_ok, f"df2 is not greater than df1 in some cells of '{key}' where df1 > 0"
            # Ensure shapes match
            assert df1.shape == df2.shape, f"Shape mismatch in DataFrame '{key}'"


def test_parameter_update_event_runs_and_cancels_as_expected(tmpdir, seed):
    """Test that when no scenarios are stored as parameters of the service integration module the event runs and then is
    cancelled"""
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels":{
                "*": logging.DEBUG},"directory": tmpdir}, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=50)

    # Set parameter update event to run before end of sim
    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2010
    sim.simulate(end_date=Date(2010, 1, 2))

    # Because switches are unchanged check logging occurred as expected
    output= parse_log_file(sim.log_filepath)
    assert 'event_runs' in output['tlo.methods.service_integration']
    assert 'event_cancelled' in output['tlo.methods.service_integration']

def test_correct_treatment_ids_are_provided_to_hs_to_override_consumables(tmpdir, seed):
    """Test that TREATMENT_IDs are correctly passed to the health system AND consumables class meaning that
    consumable availability for these HSIs is overridden"""
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels":{
                "*": logging.DEBUG},"directory": tmpdir}, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=50)
    sim.simulate(end_date=Date(2010, 1, 2))

    # Define the update event
    serv_int_event = service_integration.ServiceIntegrationParameterUpdateEvent(module=sim.modules['ServiceIntegration'])

    # for each scenario in which cons availability will be overridden, check correct list of TREATMENT_IDs is passed
    # to the health system
    for scenario, treatment_ids in zip(['htn_max',
                                        'dm_max',
                                        'hiv_max',
                                        'tb_max',
                                        'fp_scr_max',
                                        'mal_max',
                                        'anc_max',
                                        'pnc_max',
                                        'fp_pn_max',
                                        'epi',
                                        'chronic_care_max',
                                        'all_screening_max',
                                        'all_mch_max',
                                        'all_int_max'],

                                       [['CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension',
                                         'CardioMetabolicDisorders_Investigation_hypertension',
                                         'CardioMetabolicDisorders_Prevention_WeightLoss_hypertension'],

                                       ['CardioMetabolicDisorders_Investigation_diabetes',
                                         'CardioMetabolicDisorders_Prevention_WeightLoss_diabetes'],

                                       ['Hiv_Test', 'Hiv_Treatment'],

                                       ['Tb_Test_Screening',
                                         'Tb_Test_Clinical',
                                         'Tb_Test_Culture',
                                         'Tb_Test_Xray',
                                         'Tb_Treatment'],

                                        ['Contraception_Routine'],

                                        ['Undernutrition_Feeding'],

                                        ['AntenatalCare_Outpatient',
                                         'AntenatalCare_FollowUp'],

                                        ['PostnatalCare_Neonatal',
                                         'PostnatalCare_Maternal'],

                                       ['Contraception_Routine_Postnatal'],

                                        ['Epi_Childhood_Bcg',
                                         'Epi_Childhood_Opv',
                                         'Epi_Childhood_DtpHibHep',
                                         'Epi_Childhood_Rota',
                                         'Epi_Childhood_Pneumo',
                                         'Epi_Childhood_MeaslesRubella',
                                         'Epi_Adolescent_Hpv',
                                         'Epi_Pregnancy_Td'
                                         ],

                                        ['CardioMetabolicDisorders_Investigation_diabetes',
                                         'CardioMetabolicDisorders_Investigation_hypertension',
                                         'CardioMetabolicDisorders_Prevention_WeightLoss_diabetes',
                                         'CardioMetabolicDisorders_Prevention_WeightLoss_hypertension',
                                         'Hiv_Test',
                                         'Hiv_Treatment',
                                         'Tb_Test_Screening',
                                         'Tb_Test_Clinical',
                                         'Tb_Test_Culture',
                                         'Tb_Test_Xray',
                                         'Tb_Treatment',
                                         'Depression_TalkingTherapy',
                                         'Depression_Treatment',
                                         'Epilepsy_Treatment_Start',
                                         'Epilepsy_Treatment_Followup'],

                                       ['CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension',
                                        'CardioMetabolicDisorders_Investigation_hypertension',
                                        'CardioMetabolicDisorders_Prevention_WeightLoss_hypertension',
                                        'CardioMetabolicDisorders_Investigation_diabetes',
                                        'CardioMetabolicDisorders_Prevention_WeightLoss_diabetes',
                                        'Contraception_Routine',
                                        'Undernutrition_Feeding',
                                        'Hiv_Test',
                                        'Hiv_Treatment',
                                        'Tb_Test_Screening',
                                        'Tb_Test_Clinical',
                                        'Tb_Test_Culture',
                                        'Tb_Test_Xray',
                                        'Tb_Treatment'],

                                       ['Undernutrition_Feeding',
                                        'AntenatalCare_Outpatient',
                                        'AntenatalCare_FollowUp',
                                        'PostnatalCare_Neonatal',
                                        'PostnatalCare_Maternal',
                                        'Contraception_Routine_Postnatal',
                                        'Epi_Childhood_Bcg',
                                        'Epi_Childhood_Opv',
                                        'Epi_Childhood_DtpHibHep',
                                        'Epi_Childhood_Rota',
                                        'Epi_Childhood_Pneumo',
                                        'Epi_Childhood_MeaslesRubella',
                                        'Epi_Adolescent_Hpv',
                                        'Epi_Pregnancy_Td'
                                        ],

                                        ['CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension',
                                         'CardioMetabolicDisorders_Investigation_hypertension',
                                         'CardioMetabolicDisorders_Prevention_WeightLoss_hypertension',
                                         'CardioMetabolicDisorders_Investigation_diabetes',
                                         'CardioMetabolicDisorders_Prevention_WeightLoss_diabetes',
                                         'Contraception_Routine',
                                         'Undernutrition_Feeding',
                                         'Hiv_Test',
                                         'Hiv_Treatment',
                                         'Tb_Test_Screening',
                                         'Tb_Test_Clinical',
                                         'Tb_Test_Culture',
                                         'Tb_Test_Xray',
                                         'Tb_Treatment',
                                         'AntenatalCare_Outpatient',
                                         'AntenatalCare_FollowUp',
                                         'PostnatalCare_Neonatal',
                                         'PostnatalCare_Maternal',
                                         'Contraception_Routine_Postnatal',
                                         'Epi_Childhood_Bcg',
                                         'Epi_Childhood_Opv',
                                         'Epi_Childhood_DtpHibHep',
                                         'Epi_Childhood_Rota',
                                         'Epi_Childhood_Pneumo',
                                         'Epi_Childhood_MeaslesRubella',
                                         'Epi_Adolescent_Hpv',
                                         'Epi_Pregnancy_Td',
                                         'Depression_TalkingTherapy',
                                         'Depression_Treatment',
                                         'Epilepsy_Treatment_Start',
                                         'Epilepsy_Treatment_Followup']]):

        sim.modules['ServiceIntegration'].parameters['serv_integration'] = scenario
        serv_int_event.apply(sim.population.props)

        assert sim.modules['HealthSystem'].parameters['cons_override_treatment_ids'] == treatment_ids
        assert sim.modules['HealthSystem'].consumables._treatment_ids_overridden == treatment_ids

        sim.modules['HealthSystem'].set_availability_for_treatment_ids(
            treatment_ids=[],
            availability=1.0)


def test_parameter_update_event_runs_as_expected_when_updates_required_screening_parameters(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels": {
        "*": logging.DEBUG}, "directory": tmpdir}, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=50)

    # Set parameter update event to run before end of sim

    sim.modules['ServiceIntegration'].parameters['serv_integration'] = 'all_screening'
    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2010

    cons_params_init = sim.modules['Contraception'].processed_params['p_start_per_month']

    sim.simulate(end_date=Date(2010, 1, 2))

    output = parse_log_file(sim.log_filepath)
    assert 'event_runs' in output['tlo.methods.service_integration']
    assert 'event_cancelled' not in output['tlo.methods.service_integration']

    assert sim.modules['CardioMetabolicDisorders'].parameters['hypertension_hsi']['pr_assessed_other_symptoms'] == 1.0
    assert sim.modules['CardioMetabolicDisorders'].parameters['diabetes_hsi']['pr_assessed_other_symptoms'] ==  1.0
    assert sim.modules['Stunting'].parameters['prob_stunting_diagnosed_at_generic_appt'] == 1.0

    htn_test_lm =  sim.modules['CardioMetabolicDisorders'].lms_testing['hypertension']
    assert htn_test_lm.intercept == 1.0
    assert not htn_test_lm.predictors

    cons_params_init_update = sim.modules['Contraception'].processed_params['p_start_per_month']
    check_cons_processed_params_have_been_overridden(cons_params_init, cons_params_init_update, 'dict')

    assert (sim.modules['Hiv'].parameters["hiv_testing_rates"]["annual_testing_rate_adults"] == 0.4).all()
    assert (sim.modules['Tb'].parameters["rate_testing_active_tb"]["treatment_coverage"] == 90).all()


def test_parameter_update_event_runs_as_expected_when_updates_required_mch(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels": {
        "*": logging.DEBUG}, "directory": tmpdir}, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=50)

    # Set parameter update event to run before end of sim
    sim.modules['ServiceIntegration'].parameters['serv_integration'] = 'all_mch'
    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2010

    cons_p_params_b1 = sim.modules['Contraception'].processed_params['p_start_after_birth_below30']
    cons_p_params_b2 = sim.modules['Contraception'].processed_params['p_start_after_birth_30plus']


    sim.simulate(end_date=Date(2010, 1, 2))

    output = parse_log_file(sim.log_filepath)
    assert 'event_runs' in output['tlo.methods.service_integration']
    assert 'event_cancelled' not in output['tlo.methods.service_integration']

    assert sim.modules['PregnancySupervisor'].current_parameters['alternative_anc_coverage']
    assert sim.modules['PregnancySupervisor'].current_parameters['anc_availability_odds'] == 9.0
    assert sim.modules['PregnancySupervisor'].current_parameters['ps_analysis_in_progress']
    assert (sim.modules['PregnancySupervisor'].current_parameters['prob_anc1_months_2_to_4'] ==
            [1.0, 0, 0])
    assert (sim.modules['PregnancySupervisor'].current_parameters['prob_late_initiation_anc4'] ==
            0)

    assert sim.modules['Labour'].current_parameters['alternative_pnc_coverage']
    assert sim.modules['Labour'].current_parameters['pnc_availability_odds'] == 15.0
    assert sim.modules['Labour'].current_parameters['la_analysis_in_progress']
    cov_prob =  sim.modules['Labour'].current_parameters['pnc_availability_odds'] / (sim.modules['Labour'].current_parameters['pnc_availability_odds'] + 1)

    assert sim.modules['Labour'].current_parameters['prob_timings_pnc'] == [1.0, 0]
    assert sim.modules['NewbornOutcomes'].current_parameters['prob_pnc_check_newborn'] == cov_prob
    assert sim.modules['NewbornOutcomes'].current_parameters['prob_timings_pnc_newborns'] == [1.0, 0]

    assert sim.modules['Stunting'].parameters['prob_stunting_diagnosed_at_generic_appt'] == 1.0

    cons_params_b1_update = sim.modules['Contraception'].processed_params['p_start_after_birth_below30']
    cons_p_params_b2_update = sim.modules['Contraception'].processed_params['p_start_after_birth_30plus']
    check_cons_processed_params_have_been_overridden(cons_p_params_b1, cons_params_b1_update, 'series')
    check_cons_processed_params_have_been_overridden(cons_p_params_b2, cons_p_params_b2_update, 'series')

def test_parameter_update_event_runs_as_expected_when_updates_required_chronic(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels": {
        "*": logging.DEBUG}, "directory": tmpdir}, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=50)

    # Set parameter update event to run before end of sim

    # sim.modules['ServiceIntegration'].parameters['serv_int_chronic'] = True
    sim.modules['ServiceIntegration'].parameters['serv_integration'] = 'chronic_care'
    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2010
    sim.simulate(end_date=Date(2010, 1, 2))

    output = parse_log_file(sim.log_filepath)
    assert 'event_runs' in output['tlo.methods.service_integration']
    assert 'event_cancelled' not in output['tlo.methods.service_integration']

    assert sim.modules['Hiv'].parameters['virally_suppressed_on_art'] == 1.0
    assert sim.modules['Tb'].parameters['tb_prob_tx_success_ds'] == 0.9
    assert sim.modules['Tb'].parameters['tb_prob_tx_success_mdr'] == 0.9
    assert sim.modules['Epilepsy'].parameters['prob_start_anti_epilep_when_seizures_detected_in_generic_first_appt'] == 1.0
    assert sim.modules['Depression'].parameters['pr_assessed_for_depression_in_generic_appt_level1'] == 1.0

def test_cons_params_all_updated_with_all_integration_scenario(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels": {
        "*": logging.DEBUG}, "directory": tmpdir}, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=50)

    cons_params_init = sim.modules['Contraception'].processed_params['p_start_per_month']
    cons_p_params_b1 = sim.modules['Contraception'].processed_params['p_start_after_birth_below30']
    cons_p_params_b2 = sim.modules['Contraception'].processed_params['p_start_after_birth_30plus']

    sim.modules['ServiceIntegration'].parameters['serv_integration'] = 'all_int'
    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2010
    sim.simulate(end_date=Date(2010, 1, 2))

    output = parse_log_file(sim.log_filepath)
    assert 'event_runs' in output['tlo.methods.service_integration']
    assert 'event_cancelled' not in output['tlo.methods.service_integration']

    cons_params_init_update = sim.modules['Contraception'].processed_params['p_start_per_month']
    cons_params_b1_update = sim.modules['Contraception'].processed_params['p_start_after_birth_below30']
    cons_p_params_b2_update = sim.modules['Contraception'].processed_params['p_start_after_birth_30plus']

    check_cons_processed_params_have_been_overridden(cons_params_init, cons_params_init_update, 'dict')
    check_cons_processed_params_have_been_overridden(cons_p_params_b1, cons_params_b1_update,'series')
    check_cons_processed_params_have_been_overridden(cons_p_params_b2, cons_p_params_b2_update, 'series')


def test_long_run_screening_integration(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels": {
        "*": logging.DEBUG}, "directory": tmpdir}, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=1000)

    # Set parameter update event to run before end of sim
    sim.modules['ServiceIntegration'].parameters['serv_integration'] = 'all_screening'
    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2010
    sim.simulate(end_date=Date(2015, 1, 1))

    output = parse_log_file(sim.log_filepath)
    assert 'event_runs' in output['tlo.methods.service_integration']
    assert 'event_cancelled' not in output['tlo.methods.service_integration']


def test_long_run_mch_integration(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels": {
        "*": logging.DEBUG}, "directory": tmpdir}, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=1000)

    sim.modules['ServiceIntegration'].parameters['serv_integration'] = 'all_mch'
    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2010
    sim.simulate(end_date=Date(2015, 1, 1))

    output = parse_log_file(sim.log_filepath)
    assert 'event_runs' in output['tlo.methods.service_integration']
    assert 'event_cancelled' not in output['tlo.methods.service_integration']


def test_long_run_chronic_integration(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels": {
        "*": logging.DEBUG}, "directory": tmpdir}, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=1000)

    sim.modules['ServiceIntegration'].parameters['serv_integration'] = 'chronic_care'
    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2010
    sim.simulate(end_date=Date(2015, 1, 1))

    output = parse_log_file(sim.log_filepath)
    assert 'event_runs' in output['tlo.methods.service_integration']
    assert 'event_cancelled' not in output['tlo.methods.service_integration']


def test_long_run_no_integration(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels": {
        "*": logging.DEBUG}, "directory": tmpdir}, resourcefilepath=resourcefilepath)
    register_modules(sim)
    sim.make_initial_population(n=1000)

    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2010
    sim.simulate(end_date=Date(2015, 1, 1))

    output = parse_log_file(sim.log_filepath)
    assert 'event_cancelled' in output['tlo.methods.service_integration']
