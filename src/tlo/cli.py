import textwrap
from pathlib import Path
import datetime
import json
import math
import os

import click
from git import Repo
from azure import batch
import azure.batch.batch_auth as batch_auth
import azure.batch.models as batchmodels
from azure.core.exceptions import (
    ResourceExistsError,
    ResourceNotFoundError
)
from azure.storage.fileshare import (
    ShareClient,
    ShareDirectoryClient,
    ShareFileClient
)

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
@click.argument("config_file", type=click.Path(exists=True))  # TODO: remove argument
def batch_submit(scenario_file, config_file):
    """Submit a scenario to run on Azure Batch.

    SCENARIO_FILE is path to file containing scenario class.

    Your working branch must have all changes committed and pushed to the remote repository. This is to ensure that the
    copy of the code used by Azure Batch is identical to your own.
    """
    scenario_file = Path(scenario_file).as_posix()

    current_branch = is_file_clean(scenario_file)
    # current_branch = "mg/scenarios-batch"
    if current_branch is False:
        return

    scenario = load_scenario(scenario_file)
    repo = Repo('.')
    commit = next(repo.iter_commits(max_count=1, paths=scenario_file))
    run_json = scenario.save_draws(commit=commit.hexsha)

    with open(Path(config_file).as_posix()) as json_file:
        config = json.load(json_file)

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")

    # Name of the user to be used in the storage directory
    user_name = config["AZURE_STORAGE_ACCOUNT_NAME"]

    # ID of the Batch job.
    job_id = Path(scenario_file).stem + "-" + timestamp
    # job_id = config["JOB_ID"] + "-" + timestamp

    # Path in Azure storage where to store the files for this job
    azure_directory = f"{user_name}/{job_id}"

    # Create a Batch service client.
    credentials = batch_auth.SharedKeyCredentials(config["AZURE_BATCH_ACCOUNT_NAME"],
                                                  config["AZURE_BATCH_ACCOUNT_KEY"],
                                                  )

    batch_client = batch.BatchServiceClient(
        credentials,
        batch_url=config["AZURE_BATCH_ACCOUNT_URL"],
    )

    create_file_share(config["AZURE_STORAGE_CONNECTION_STRING"],
                      config["SHARE_NAME"],
                      )

    # Recursively create all nested directories,
    for idx in range(len(os.path.split(azure_directory))):
        create_directory(config["AZURE_STORAGE_CONNECTION_STRING"],
                         config["SHARE_NAME"],
                         '/'.join(os.path.split(azure_directory)[:idx+1]),
                         )

    upload_local_file(config["AZURE_STORAGE_CONNECTION_STRING"],
                      run_json,
                      config["SHARE_NAME"],
                      azure_directory + "/" + os.path.basename(run_json),
                      )

    # Configuration of the pool: type of machines and number of nodes.
    vm_size = config["POOL_VM_SIZE"]
    # TODO: cap the number of nodes in the pool?  Take the number of nodes in
    # input from the user, but always at least 2?
    pool_node_count = max(2, math.ceil(scenario.number_of_draws * scenario.runs_per_draw))

    # User identity in the Batch tasks
    auto_user = batchmodels.AutoUserSpecification(
        elevation_level=batchmodels.ElevationLevel.admin,
        scope=batchmodels.AutoUserScope.task,
    )

    user_identity = batchmodels.UserIdentity(
        auto_user=auto_user,
    )

    # URL of the Azure File share
    azure_file_url = "https://{}.file.core.windows.net/{}".format(
        config["AZURE_STORAGE_ACCOUNT_NAME"],
        config["SHARE_NAME"],
    )

    # Specify a container registry
    container_registry = batchmodels.ContainerRegistry(
        registry_server=config["AZURE_REGISTRY_SERVER"],
        user_name=config["AZURE_REGISTRY_ACCOUNT_NAME"],
        password=config["AZURE_REGISTRY_ACCOUNT_KEY"],
    )

    # Name of the image in the registry
    image_name = config["AZURE_REGISTRY_SERVER"] + "/" + config["CONTAINER_IMAGE_NAME"]

    # Create container configuration, prefetching Docker images from the container registry
    container_conf = batchmodels.ContainerConfiguration(
        container_image_names=[image_name],
        container_registries=[container_registry],
    )

    # Options for running the Docker container
    container_run_options = "--rm --workdir /TLOmodel"

    # Directory where the file share will be mounted, relative to
    # ${AZ_BATCH_NODE_MOUNTS_DIR}.
    file_share_mount_point = "mnt"

    azure_file_share_configuration = batchmodels.AzureFileShareConfiguration(
        account_name=config["AZURE_STORAGE_ACCOUNT_NAME"],
        azure_file_url=azure_file_url,
        account_key=config["AZURE_STORAGE_ACCOUNT_KEY"],
        relative_mount_path=file_share_mount_point,
        mount_options="-o rw",
    )

    mount_configuration = batchmodels.MountConfiguration(
        azure_file_share_configuration=azure_file_share_configuration,
    )

    azure_directory = "${{AZ_BATCH_NODE_MOUNTS_DIR}}/" + \
        f"{file_share_mount_point}/{azure_directory}"
    azure_run_json = f"{azure_directory}/{os.path.basename(run_json)}"
    working_dir = "${{AZ_BATCH_TASK_WORKING_DIR}}"
    command = f"""
    git fetch --all
    git checkout -b {current_branch} origin/{current_branch}
    git pull
    tlo batch-run {azure_run_json} {working_dir} {{}} {{}}
    cp -r {working_dir}/* {azure_directory}/.
    """
    command = f"/bin/bash -c '{command}'"

    try:
        # Create the job that will run the tasks.
        create_job(batch_client, vm_size, pool_node_count, job_id,
                   container_conf, [mount_configuration])

        # Add the tasks to the job.
        add_tasks(batch_client, user_identity, job_id, image_name,
                  container_run_options, scenario, command)

    except batchmodels.BatchErrorException as err:
        print_batch_exception(err)
        raise


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


def print_batch_exception(batch_exception):
    """
    Prints the contents of the specified Batch exception.

    :param batch_exception:
    """
    print("-------------------------------------------")
    print("Exception encountered:")
    if batch_exception.error and \
            batch_exception.error.message and \
            batch_exception.error.message.value:
        print(batch_exception.error.message.value)
        if batch_exception.error.values:
            print()
            for mesg in batch_exception.error.values:
                print(f"{mesg.key}:\t{mesg.value}")
    print("-------------------------------------------")


def create_file_share(connection_string, share_name):
    """
    Uses a ShareClient object to create a share if it does not exist.
    """
    try:
        # Create a ShareClient from a connection string
        share_client = ShareClient.from_connection_string(
            connection_string, share_name)

        print("Creating share:", share_name)
        share_client.create_share()

    except ResourceExistsError as ex:
        print("-------------------------------------------")
        print("ResourceExistsError:", ex.message)
        print("-------------------------------------------")


def create_directory(connection_string, share_name, dir_name):
    """
    Creates a directory in the root of the specified file share by using a
    ShareDirectoryClient object.
    """
    try:
        # Create a ShareDirectoryClient from a connection string
        dir_client = ShareDirectoryClient.from_connection_string(
            connection_string, share_name, dir_name)

        print("Creating directory:", share_name + "/" + dir_name)
        dir_client.create_directory()

    except ResourceExistsError as ex:
        print("ResourceExistsError:", ex.message)


def upload_local_file(connection_string, local_file_path, share_name, dest_file_path):
    """
    Uploads the contents of the specified file into the specified directory in
    the specified Azure file share.
    """
    try:
        source_file = open(local_file_path, "rb")
        data = source_file.read()

        # Create a ShareFileClient from a connection string
        file_client = ShareFileClient.from_connection_string(
            connection_string, share_name, dest_file_path)

        print("Uploading to:", share_name + "/" + dest_file_path)
        file_client.upload_file(data)

    except ResourceExistsError as ex:
        print("ResourceExistsError:", ex.message)

    except ResourceNotFoundError as ex:
        print("ResourceNotFoundError:", ex.message)


def create_job(batch_service_client, vm_size, pool_node_count, job_id,
               container_conf, mount_configuration):
    """
    Creates a job with the specified ID, associated with the specified pool.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str vm_size: Type of virtual machine to use as pool.
    :param int pool_node_count: Number of nodes in the pool.
    :param str job_id: The ID for the job.
    :param container_conf: Configuration of a container.
    :type container_conf: `azure.batch.models.ContainerConfiguration`
    :param mount_configuration: Configuration of the images to mount on the nodes.
    :type mount_configuration: `list[azure.batch.models.MountConfiguration]`
    """
    print(f"Creating job [{job_id}]...")

    image_reference = batchmodels.ImageReference(
        publisher="microsoft-azure-batch",
        offer="ubuntu-server-container",
        sku="16-04-lts",
        version="latest",
    )

    virtual_machine_configuration = batchmodels.VirtualMachineConfiguration(
        image_reference=image_reference,
        container_configuration=container_conf,
        node_agent_sku_id="batch.node.ubuntu 16.04",
    )

    pool = batchmodels.PoolSpecification(
        virtual_machine_configuration=virtual_machine_configuration,
        vm_size=vm_size,
        target_dedicated_nodes=pool_node_count,
        mount_configuration=mount_configuration,
        task_slots_per_node=1
    )

    auto_pool_specification = batchmodels.AutoPoolSpecification(
        pool_lifetime_option="job",
        pool=pool,
    )

    pool_info = batchmodels.PoolInformation(
        auto_pool_specification=auto_pool_specification,
    )

    job = batchmodels.JobAddParameter(
        id=job_id,
        pool_info=pool_info,
        on_all_tasks_complete="terminateJob",
    )

    batch_service_client.job.add(job)


def add_tasks(batch_service_client, user_identity, job_id,
              image_name, container_run_options, scenario, command):
    """
    Adds the simulation tasks in the collection to the specified job.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param user_identity: User account to use in the jobs.
    :type user_identity: `azure.batch.models.UserIdentity`
    :param str job_id: The ID of the job to which to add the tasks.
    :param str image_name: Name of the Docker image to mount for the task.
    :param str container_run_options: Options to pass to Docker to run the image.
    :param str command: Command to run during the taks inside the Docker image.
    """

    print("Adding {} task(s) to job [{}]...".format(
        scenario.number_of_draws * scenario.runs_per_draw,
        job_id))

    tasks = list()

    task_container_settings = batchmodels.TaskContainerSettings(
        image_name=image_name,
        container_run_options=container_run_options,
    )

    for draw_number in range(0, scenario.number_of_draws):
        for run_number in range(0, scenario.runs_per_draw):
            task = batchmodels.TaskAddParameter(
                id=f"draw_{draw_number}-run_{run_number}",
                command_line=command.format(draw_number, run_number),
                container_settings=task_container_settings,
                user_identity=user_identity,
            )
            tasks.append(task)

    batch_service_client.task.add_collection(job_id, tasks)
