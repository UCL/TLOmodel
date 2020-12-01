import textwrap
from pathlib import Path

import click
from git import Repo

from tlo import logging
from tlo.scenario import BaseScenario, SampleRunner, ScenarioLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("scenario_file", type=click.Path(exists=True))
@click.option("--draw-only", is_flag=True, help="Only generate draws; do not run the simulation")
def scenario_run(scenario_file, draw_only):
    """Run locally the scenario defined in SCENARIO_FILE

    SCENARIO_FILE is path to file containing a scenario class
    """
    scenario = load_scenario(scenario_file)
    run_json = scenario.save_draws()
    if not draw_only:
        runner = SampleRunner(run_json)
        runner.run()


@cli.command()
@click.argument("scenario_file", type=click.Path(exists=True))
def batch_submit(scenario_file):
    """Submit a scenario to run on Azure Batch.

    SCENARIO_FILE is path to file containing scenario class.

    Your working branch must have all changes committed and pushed to the remote repository. This is to ensure that the
    copy of the code used by Azure Batch is identical to your own.
    """
    scenario_file = Path(scenario_file).as_posix()

    current_branch = is_file_clean(scenario_file)
    if current_branch is False:
        return

    scenario = load_scenario(scenario_file)
    run_json = scenario.save_draws()
    user_id = 'xyz'
    job_id = 'abc'
    # create a job id: <scenario_file_name>-timestamp
    # create job
    # create storage location for this user: <user_id>
    # create storage location for this job: <user_id>/<jobid>
    # upload json file (run_json is path - always rename) to <user_id>/<jobid>/run.json
    azure_run_json = f"/azure/storage/path/{user_id}/{job_id}/{run_json}"  # TODO: on shared storage
    azure_container_wd = "/azure/batch/wd"  # TODO: mounted inside container
    # build list of tasks
    for draw_number in range(0, scenario.number_of_draws):
        for sample_number in range(0, scenario.samples_per_draw):
            # make task to run the following:
            script = f"""
            git fetch --all
            git checkout {current_branch}
            git pull
            tlo batch-run {azure_run_json} {azure_container_wd} {draw_number} {sample_number}
            """
            script = textwrap.dedent(script)
            # add task to job
    # submit job
    # echo to screen: this is your job_id, you can see your files by xyz


@cli.command(hidden=True)
@click.argument("path_to_json", type=click.Path(exists=True))
@click.argument("work_directory", type=click.Path(exists=True))
@click.argument("draw", type=int)
@click.argument("sample", type=int)
def batch_run(path_to_json, work_directory, draw, sample):
    runner = SampleRunner(path_to_json)
    output_directory = Path(work_directory) / f"{draw}/{sample}"
    output_directory.mkdir(parents=True, exist_ok=True)
    runner.run_sample_by_number(output_directory, draw, sample)


def load_scenario(scenario_file):
    scenario_path = Path(scenario_file)
    scenario_class: BaseScenario = ScenarioLoader(scenario_path.parent / scenario_path.name).get_scenario()
    logger.info(key="message", data=f"Loaded {scenario_class.__class__.__name__} from {scenario_path}")
    return scenario_class


def is_file_clean(scenario_file):
    """Checks whether the scenario file and current branch is clean and unchanged.

    :returns: current branch name if all okay, False otherwise
    """
    repo = Repo('.')  # assumes you're running tlo command from TLOmodel root directory

    if scenario_file in repo.untracked_files:
        click.echo(f"ERROR: Untracked file {scenario_file}. Add file to repository, commit and push.")
        return False

    if repo.is_dirty(path=scenario_file):
        click.echo(f"ERROR: Uncommitted changes in file {scenario_file}. Rollback or commit+push changes.")
        return False

    current_branch = repo.head.reference
    commits_ahead = list(repo.iter_commits(f'origin/{current_branch}..{current_branch}'))
    commits_behind = list(repo.iter_commits(f'{current_branch}..origin/{current_branch}'))
    if not len(commits_behind) == len(commits_ahead) == 0:
        click.echo(f"ERROR: Branch '{current_branch}' isn't in-sync with remote: "
                   f"{len(commits_ahead)} ahead; {len(commits_behind)} behind. Push and/or pull changes.")
        return False

    return current_branch
