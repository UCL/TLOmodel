import json
import os
from pathlib import Path

from tlo import Date
from tlo.scenario import ScenarioLoader, BaseScenario, SampleRunner


class TestScenarioStore:
    scenario = None
    config_path = None


def test_load():
    """Check we can load the scenario class from a file"""
    path = Path(f'{os.path.dirname(__file__)}/resources/scenario.py')
    scenario = ScenarioLoader(path).get_scenario()
    assert isinstance(scenario, BaseScenario)
    assert scenario.scenario_path == path
    TestScenarioStore.scenario = scenario


def test_arguments():
    """Check we can parse arguments related to the scenario. pop-size is used by our scenario,
    suspend-date is used in base class"""
    scenario = TestScenarioStore.scenario
    assert scenario.pop_size == 2000  # this is the default pop size set in the scenario class
    scenario.parse_arguments(['--pop-size', '100', '--suspend-date', '2012-03-04'])
    assert scenario.pop_size == 100  # this is the value we passed in
    assert scenario.suspend_date == Date(year=2012, month=3, day=4)
    assert not hasattr(scenario, 'resume_simulation')


def test_config(tmp_path):
    """Create the run configuration and check we've got the right values in there."""
    scenario = TestScenarioStore.scenario
    config = scenario.save_draws(return_config=True)
    assert config['scenario_seed'] == 655123742
    assert config['arguments'] == ['--pop-size', '100', '--suspend-date', '2012-03-04']
    assert len(config['draws']) == 5

    config_path = tmp_path / 'scenario.json'
    TestScenarioStore.config_path = config_path
    with open(config_path, 'w') as f:
        f.write(json.dumps(config, indent=2))


def test_runner():
    """Check we can load the scenario from the configuration file."""
    runner = SampleRunner(TestScenarioStore.config_path)
    scenario = runner.scenario
    assert isinstance(scenario, BaseScenario)
    assert scenario.__class__.__name__ == 'TestScenario'
    assert scenario.pop_size == 100
    assert scenario.suspend_date == Date(year=2012, month=3, day=4)
    assert runner.number_of_draws == 5
