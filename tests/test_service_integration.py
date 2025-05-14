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

    sim.register(*fullmodel(resourcefilepath=resourcefilepath),
                  service_integration.ServiceIntegration(resourcefilepath=resourcefilepath))

def test_parameter_update_event_runs_and_cancels_as_expected(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels":{
                "*": logging.DEBUG},"directory": tmpdir})
    register_modules(sim)
    sim.make_initial_population(n=50)

    # Set parameter update event to run before end of sim
    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2011
    sim.simulate(end_date=Date(2011, 1, 2))

    # Because switches are unchanged check logging occurred as expected
    output= parse_log_file(sim.log_filepath)
    assert 'event_runs' in output['tlo.methods.service_integration']
    assert 'event_cancelled' in output['tlo.methods.service_integration']


def test_parameter_update_event_runs_as_expected_when_updates_required_screening_parameters(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels": {
        "*": logging.DEBUG}, "directory": tmpdir})
    register_modules(sim)
    sim.make_initial_population(n=500)

    # Set parameter update event to run before end of sim

    sim.modules['ServiceIntegration'].parameters['serv_int_screening'] = ['htn', 'dm', 'hiv' ,'tb', 'fp', 'mal']
    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2011
    sim.simulate(end_date=Date(2011, 1, 2))

    output = parse_log_file(sim.log_filepath)
    assert 'event_runs' in output['tlo.methods.service_integration']
    assert 'event_cancelled' not in output['tlo.methods.service_integration']

    assert sim.modules['CardioMetabolicDisorders'].parameters['hypertension_hsi']['pr_assessed_other_symptoms'] == 1.0
    assert sim.modules['CardioMetabolicDisorders'].parameters['diabetes_hsi']['pr_assessed_other_symptoms'] ==  1.0
    assert sim.modules['Stunting'].parameters['prob_stunting_diagnosed_at_generic_appt'] == 1.0

    htn_test_lm =  sim.modules['CardioMetabolicDisorders'].lms_testing['hypertension']
    assert htn_test_lm.intercept == 1.0
    assert not htn_test_lm.predictors

    # TODO: add tests for HIV/Tb/Contraception screening interventions


def test_parameter_update_event_runs_as_expected_when_updates_required_mch(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels": {
        "*": logging.DEBUG}, "directory": tmpdir})
    register_modules(sim)
    sim.make_initial_population(n=500)

    # Set parameter update event to run before end of sim

    sim.modules['ServiceIntegration'].parameters['serv_int_mch'] = ['pnc', 'fp', 'mal', 'epi']
    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2011
    sim.simulate(end_date=Date(2011, 1, 2))

    output = parse_log_file(sim.log_filepath)
    assert 'event_runs' in output['tlo.methods.service_integration']
    assert 'event_cancelled' not in output['tlo.methods.service_integration']


    assert sim.modules['Labour'].current_parameters['alternative_pnc_coverage']
    assert sim.modules['Labour'].current_parameters['pnc_availability_odds'] == 15.0
    assert sim.modules['Stunting'].parameters['prob_stunting_diagnosed_at_generic_appt'] == 1.0

    # TODO: add tests for EPI/Contraception interventions

def test_parameter_update_event_runs_as_expected_when_updates_required_chronic(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels": {
        "*": logging.DEBUG}, "directory": tmpdir})
    register_modules(sim)
    sim.make_initial_population(n=500)

    # Set parameter update event to run before end of sim

    sim.modules['ServiceIntegration'].parameters['serv_int_chronic'] = True
    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2011
    sim.simulate(end_date=Date(2011, 1, 2))

    output = parse_log_file(sim.log_filepath)
    assert 'event_runs' in output['tlo.methods.service_integration']
    assert 'event_cancelled' not in output['tlo.methods.service_integration']

    assert sim.modules['Hiv'].parameters['virally_suppressed_on_art'] == 1.0
    assert sim.modules['Tb'].parameters['tb_prob_tx_success_ds'] == 0.9
    assert sim.modules['Tb'].parameters['tb_prob_tx_success_mdr'] == 0.9
    assert sim.modules['Tb'].parameters['tb_prob_tx_success_0_4'] == 0.9
    assert sim.modules['Tb'].parameters['tb_prob_tx_success_5_14'] == 0.9
    assert sim.modules['Epilepsy'].parameters['prob_start_anti_epilep_when_seizures_detected_in_generic_first_appt'] == 1.0
    assert sim.modules['Depression'].parameters['pr_assessed_for_depression_in_generic_appt_level1'] == 1.0


def test_long_run_screening_integration(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels": {
        "*": logging.DEBUG}, "directory": tmpdir})
    register_modules(sim)
    sim.make_initial_population(n=1000)

    # Set parameter update event to run before end of sim

    sim.modules['ServiceIntegration'].parameters['serv_int_screening'] = ['htn', 'dm', 'hiv' ,'tb', 'fp']
    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2010
    sim.simulate(end_date=Date(2015, 1, 1))

    output = parse_log_file(sim.log_filepath)
    assert 'event_runs' in output['tlo.methods.service_integration']
    assert 'event_cancelled' not in output['tlo.methods.service_integration']


def test_long_run_mch_integration(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels": {
        "*": logging.DEBUG}, "directory": tmpdir})
    register_modules(sim)
    sim.make_initial_population(n=1000)

    # Set parameter update event to run before end of sim

    sim.modules['ServiceIntegration'].parameters['serv_int_mch'] = ['pnc', 'fp', 'mal', 'epi']
    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2010
    sim.simulate(end_date=Date(2015, 1, 1))

    output = parse_log_file(sim.log_filepath)
    assert 'event_runs' in output['tlo.methods.service_integration']
    assert 'event_cancelled' not in output['tlo.methods.service_integration']


def test_long_run_chronic_integration(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels": {
        "*": logging.DEBUG}, "directory": tmpdir})
    register_modules(sim)
    sim.make_initial_population(n=1000)

    # Set parameter update event to run before end of sim

    sim.modules['ServiceIntegration'].parameters['serv_int_chronic'] = True
    sim.modules['ServiceIntegration'].parameters['integration_year'] = 2010
    sim.simulate(end_date=Date(2015, 1, 1))

    output = parse_log_file(sim.log_filepath)
    assert 'event_runs' in output['tlo.methods.service_integration']
    assert 'event_cancelled' not in output['tlo.methods.service_integration']
