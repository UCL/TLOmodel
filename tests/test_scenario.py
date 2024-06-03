import json
import os
from pathlib import Path

import pytest

from tlo import Date
from tlo.scenario import BaseScenario, SampleRunner, ScenarioLoader


@pytest.fixture
def scenario_path():
    return Path(f'{os.path.dirname(__file__)}/resources/scenario.py')


@pytest.fixture
def pop_size():
    return 100


@pytest.fixture
def loaded_scenario(scenario_path):
    return ScenarioLoader(scenario_path).get_scenario()


@pytest.fixture
def arguments(pop_size):
    return ['--pop-size', str(pop_size)]


@pytest.fixture
def loaded_scenario_with_parsed_arguments(loaded_scenario, arguments):
    loaded_scenario.parse_arguments(arguments)
    return loaded_scenario


def test_load(loaded_scenario, scenario_path):
    """Check we can load the scenario class from a file"""
    assert isinstance(loaded_scenario, BaseScenario)
    assert loaded_scenario.scenario_path == scenario_path
    assert hasattr(loaded_scenario, "pop_size")  # Default value set in initialiser


def test_parse_arguments(loaded_scenario_with_parsed_arguments, pop_size):
    """Check we can parse arguments related to the scenario. pop-size is used by our scenario,
    suspend-date is used in base class"""
    assert loaded_scenario_with_parsed_arguments.pop_size == pop_size
    assert not hasattr(loaded_scenario_with_parsed_arguments, 'resume_simulation')


def test_config(tmp_path, loaded_scenario_with_parsed_arguments, arguments):
    """Create the run configuration and check we've got the right values in there."""
    config = loaded_scenario_with_parsed_arguments.save_draws(return_config=True)
    assert config['scenario_seed'] == loaded_scenario_with_parsed_arguments.seed
    assert config['arguments'] == arguments
    assert len(config['draws']) == loaded_scenario_with_parsed_arguments.number_of_draws


def test_runner(tmp_path, loaded_scenario_with_parsed_arguments, pop_size, suspend_date):
    """Check we can load the scenario from a configuration file."""
    config = loaded_scenario_with_parsed_arguments.save_draws(return_config=True)
    config_path = tmp_path / 'scenario.json'
    with open(config_path, 'w') as f:
        f.write(json.dumps(config, indent=2))
    runner = SampleRunner(config_path)
    scenario = runner.scenario
    assert isinstance(scenario, BaseScenario)
    assert scenario.__class__.__name__ == 'TestScenario'
    assert scenario.pop_size == pop_size
    assert runner.number_of_draws == loaded_scenario_with_parsed_arguments.number_of_draws
