from pathlib import Path

import click

from tlo import logging
from tlo.scenario import BaseScenario, ScenarioLoader, SampleRunner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.group()
def cli():
    pass


@cli.command()
def scenario_create():
    click.echo("create scenario")


@cli.command()
@click.argument('file', type=click.Path(exists=True))
def scenario_run(file):
    """Run locally the scenario defined in FILE

    FILE is path to file containing a scenario class
    """
    scenario_path = Path(file)
    scenario_class: BaseScenario = ScenarioLoader(scenario_path.parent / scenario_path.name).get_scenario()
    logger.info(key="message", data=f"Loaded {scenario_class.__class__.__name__} from {scenario_path}")
    draws_json_path = scenario_class.save_draws()
    logger.info(key="message", data=f"Saved draws configuration to {draws_json_path}")
    runner = SampleRunner(draws_json_path)
    runner.run()


@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--branch', default='master', help='checkout the specified branch for running the scenario')
def scenario_submit(path, branch):
    click.echo(f"generate_scenario {path} on {branch}")

