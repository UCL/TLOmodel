"""The TLOmodel command-line interface"""
import configparser
import datetime
import json
import os
import pickle
import tempfile
from collections import defaultdict
from pathlib import Path
from shutil import copytree
from typing import Dict

import click
import dateutil.parser
import pandas as pd
from azure import batch
from azure.batch import models as batch_models
from azure.batch.models import BatchErrorException
from azure.common.credentials import ServicePrincipalCredentials
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.fileshare import ShareClient, ShareDirectoryClient, ShareFileClient
from git import Repo

from tlo.analysis.utils import parse_log_file
from tlo.scenario import SampleRunner, ScenarioLoader

JOB_LABEL_PADDING = len("State transition time")


@click.group()
@click.option("--config-file", type=click.Path(exists=False), default="tlo.conf", hidden=True)
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.pass_context
def cli(ctx, config_file, verbose):
    """tlo - the TLOmodel command line utility.

    * run scenarios locally
    * submit scenarios to batch system
    * query batch system about job and tasks
    * download output results for completed job
    * combine runs from multiple batch jobs with same draws
    """
    ctx.ensure_object(dict)
    ctx.obj["config_file"] = config_file
    ctx.obj["verbose"] = verbose


@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("scenario_file", type=click.Path(exists=True))
@click.option("--draw-only", is_flag=True, help="Only generate draws; do not run the simulation")
@click.option("--draw", "-d", nargs=2, type=int)
@click.option("--output-dir", type=str)
@click.argument('scenario_args', nargs=-1, type=click.UNPROCESSED)
def scenario_run(scenario_file, draw_only, draw: tuple, output_dir=None, scenario_args=None):
    """Run the specified scenario locally.

    SCENARIO_FILE is path to file containing a scenario class
    """
    scenario = load_scenario(scenario_file)

    # if we have other scenario arguments, parse them
    if scenario_args is not None:
        scenario.parse_arguments(scenario_args)

    config = scenario.save_draws(return_config=True)
    json_string = json.dumps(config, indent=2)

    if draw_only:
        # pretty-print json
        print(json_string)
        return

    # Write the JSON config to a temporary file (to prevent clobbering by multiple processes running same scenario)
    # We set delete=False on the file, otherwise Windows does not allow the opening of the file multiple times
    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w')
    tmp.write(json_string)
    tmp.flush()

    # SampleRunner loads the scenario information from the JSON config file
    filename = tmp.name
    runner = SampleRunner(filename)

    # Runner has been created - clean up (we have to do this ourselves because we set delete=False)
    tmp.close()
    os.unlink(filename)

    if draw:
        runner.run_sample_by_number(output_directory=output_dir, draw_number=draw[0], sample_number=draw[1])
    else:
        runner.run()

@cli.command()
@click.argument("LOG_DIRECTORY", type=click.Path(exists=True))
def parse_log(log_directory):
    assert os.path.isdir(log_directory), f"{log_directory} must be a directory"

    path = Path(log_directory)

    log_files = list(path.glob("*.log"))
    assert len(log_files) == 1, f"directory {log_directory} must contain exactly one file with extension .log"

    outputs = parse_log_file(log_files[0])
    for key, output in outputs.items():
        if key.startswith("tlo."):
            with open(path / f"{key}.pickle", "wb") as f:
                pickle.dump(output, f)

@cli.command()
@click.argument("scenario_file", type=click.Path(exists=True))
@click.option("--asserts-on", type=bool, default=False, is_flag=True, help="Enable assertions in simulation run.")
@click.option("--more-memory", type=bool, default=False, is_flag=True,
              help="Request machine wth more memory (for larger population sizes).")
@click.option("--image-tag", type=str, help="Tag of the Docker image to use.")
@click.option("--keep-pool-alive", type=bool, default=False, is_flag=True, hidden=True)
@click.pass_context
def batch_submit(ctx, scenario_file, asserts_on, more_memory, keep_pool_alive, image_tag=None):
    """Submit a scenario to the batch system.

    SCENARIO_FILE is path to file containing scenario class.

    Your working branch must have all changes committed and pushed to the remote repository.
    This is to ensure that the copy of the code used by Azure Batch is identical to your own.
    """
    print(">Setting up scenario\r", end="")
    scenario_file = Path(scenario_file).as_posix()

    current_branch = is_file_clean(scenario_file)
    if current_branch is False:
        return

    scenario = load_scenario(scenario_file)

    # get the commit we're going to submit to run on batch, and save the run config for that commit
    # it's the most recent commit on current branch
    repo = Repo(".")
    commit = next(repo.iter_commits(max_count=1))
    run_json = scenario.save_draws(commit=commit.hexsha)

    print(">Setting up batch\r", end="")

    config = load_config(ctx.obj['config_file'])

    # ID of the Batch job.
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    job_id = scenario.get_log_config()["filename"] + "-" + timestamp

    # Path in Azure storage where to store the files for this job
    azure_directory = f"{config['DEFAULT']['USERNAME']}/{job_id}"

    batch_client = get_batch_client(
        config["BATCH"]["CLIENT_ID"], config["BATCH"]["SECRET"], config["AZURE"]["TENANT_ID"], config["BATCH"]["URL"]
    )

    create_file_share(
        config["STORAGE"]["CONNECTION_STRING"],
        config["STORAGE"]["FILESHARE"]
    )

    # Recursively create all nested directories,
    for idx in range(len(os.path.split(azure_directory))):
        create_directory(config["STORAGE"]["CONNECTION_STRING"],
                         config["STORAGE"]["FILESHARE"],
                         "/".join(os.path.split(azure_directory)[:idx+1]),
                         )

    upload_local_file(config["STORAGE"]["CONNECTION_STRING"],
                      run_json,
                      config["STORAGE"]["FILESHARE"],
                      azure_directory + "/" + os.path.basename(run_json),
                      )

    # Configuration of the pool: type of machines and number of nodes.
    if more_memory:
        vm_size = config["BATCH"]["POOL_VM_SIZE_MORE_MEMORY"]
    else:
        vm_size = config["BATCH"]["POOL_VM_SIZE"]

    pool_node_count = scenario.number_of_draws * scenario.runs_per_draw

    # User identity in the Batch tasks
    auto_user = batch_models.AutoUserSpecification(
        elevation_level=batch_models.ElevationLevel.admin,
        scope=batch_models.AutoUserScope.task,
    )

    user_identity = batch_models.UserIdentity(
        auto_user=auto_user,
    )

    # URL of the Azure File share
    azure_file_url = "https://{}.file.core.windows.net/{}".format(
        config["STORAGE"]["NAME"],
        config["STORAGE"]["FILESHARE"],
    )

    # Specify a container registry
    container_registry = batch_models.ContainerRegistry(
        registry_server=config["REGISTRY"]["SERVER"],
        user_name=config["REGISTRY"]["NAME"],
        password=config["REGISTRY"]["KEY"],
    )

    # url of the docker image to run the tasks
    image_name = f"{config['REGISTRY']['SERVER']}/{config['REGISTRY']['IMAGE']}"

    # use the supplied image tag if provided, otherwise use the default
    if image_tag is None:
        image_name = f"{image_name}:{config['REGISTRY']['DEFAULT_TAG']}"
    else:
        image_name = f"{image_name}:{image_tag}"

    # Create container configuration, prefetching Docker images from the container registry
    container_conf = batch_models.ContainerConfiguration(
        type="dockerCompatible",
        container_image_names=[image_name],
        container_registries=[container_registry],
    )

    # Options for running the Docker container
    container_run_options = "--rm --workdir /TLOmodel"

    # Directory where the file share will be mounted, relative to
    # ${AZ_BATCH_NODE_MOUNTS_DIR}.
    file_share_mount_point = "mnt"

    azure_file_share_configuration = batch_models.AzureFileShareConfiguration(
        account_name=config["STORAGE"]["NAME"],
        azure_file_url=azure_file_url,
        account_key=config["STORAGE"]["KEY"],
        relative_mount_path=file_share_mount_point,
        mount_options="-o rw",
    )

    mount_configuration = batch_models.MountConfiguration(
        azure_file_share_configuration=azure_file_share_configuration,
    )

    # we turn off assertions by default
    py_opt = "PYTHONOPTIMIZE=1"
    if asserts_on:
        py_opt = ""

    azure_directory = "${{AZ_BATCH_NODE_MOUNTS_DIR}}/" + \
        f"{file_share_mount_point}/{azure_directory}"
    azure_run_json = f"{azure_directory}/{os.path.basename(run_json)}"
    working_dir = "${{AZ_BATCH_TASK_WORKING_DIR}}"
    task_dir = "${{AZ_BATCH_TASK_DIR}}"
    gzip_pattern_match = "{{txt,log}}"
    command = f"""
    git fetch origin {commit.hexsha}
    git checkout {commit.hexsha}
    pip install -r requirements/base.txt
    env | grep "^AZ_" | while read line; do echo "$line"; done
    {py_opt} tlo --config-file tlo.example.conf batch-run {azure_run_json} {working_dir} {{draw_number}} {{run_number}}
    tlo --config-file tlo.example.conf parse-log {working_dir}/{{draw_number}}/{{run_number}}
    cp {task_dir}/std*.txt {working_dir}/{{draw_number}}/{{run_number}}/.
    gzip {working_dir}/{{draw_number}}/{{run_number}}/*.{gzip_pattern_match}
    cp -r {working_dir}/* {azure_directory}/.
    """
    command = f"/bin/bash -c '{command}'"

    try:
        # Create the job that will run the tasks.
        create_job(
            batch_client,
            vm_size,
            pool_node_count,
            job_id,
            container_conf,
            [mount_configuration],
            keep_pool_alive,
            config["BATCH"]["SUBNET_ID"],
        )

        # Add the tasks to the job.
        add_tasks(batch_client, user_identity, job_id, image_name,
                  container_run_options, scenario, command)

    except batch_models.BatchErrorException as err:
        print_batch_exception(err)
        raise

    print(f"Job ID: {job_id}")


@cli.command(hidden=True)
@click.argument("path_to_json", type=click.Path(exists=True))
@click.argument("work_directory", type=click.Path(exists=True))
@click.argument("draw", type=int)
@click.argument("sample", type=int)
def batch_run(path_to_json, work_directory, draw, sample):
    """Runs the specified draw and sample for the Scenario"""
    runner = SampleRunner(path_to_json)
    output_directory = Path(work_directory) / f"{draw}/{sample}"
    output_directory.mkdir(parents=True, exist_ok=True)
    runner.run_sample_by_number(output_directory, draw, sample)


@cli.command()
@click.argument("job_id", type=str)
@click.pass_context
def batch_terminate(ctx, job_id):
    """Terminate running job having the specified JOB_ID.
    Note, you can only terminate your own jobs (user configured in tlo.conf)"""

    # we check that directory has been created for this job in the user's folder
    # (set up when the job is submitted)
    config = load_config(ctx.obj["config_file"])
    username = config["DEFAULT"]["USERNAME"]
    directory = f"{username}/{job_id}"
    share_client = ShareClient.from_connection_string(config['STORAGE']['CONNECTION_STRING'],
                                                      config['STORAGE']['FILESHARE'])

    try:
        directories = share_client.list_directories_and_files(directory)
        assert len(list(directories)) > 0
    except ResourceNotFoundError:
        print("ERROR: directory ", directory, "not found.")
        return

    batch_client = get_batch_client(
        config["BATCH"]["CLIENT_ID"], config["BATCH"]["SECRET"], config["AZURE"]["TENANT_ID"], config["BATCH"]["URL"]
    )

    # check the job is running
    try:
        job = batch_client.job.get(job_id=job_id)
    except BatchErrorException:
        print("ERROR: job ", job_id, "not found.")
        return

    if job.state == "completed":
        print("ERROR: Job already finished.")
        return

    # job is running & not finished - terminate the job
    batch_client.job.terminate(job_id=job_id)
    print(f"Job {job_id} terminated.")
    print("To download output run:")
    print(f"\ttlo batch-download {job_id}")


@cli.command()
@click.argument("job_id", type=str)
@click.option("show_tasks", "--tasks", is_flag=True, default=False, help="Display task information")
@click.option("--raw", default=False, help="Display raw output (only when retrieving using --id)",
              is_flag=True, hidden=True)
@click.pass_context
def batch_job(ctx, job_id, raw, show_tasks):
    """Display information about a specific job."""
    print(">Querying batch system\r", end="")
    config = load_config(ctx.obj['config_file'])
    batch_client = get_batch_client(
        config["BATCH"]["CLIENT_ID"], config["BATCH"]["SECRET"], config["AZURE"]["TENANT_ID"], config["BATCH"]["URL"]
    )

    tasks = None

    try:
        job = batch_client.job.get(job_id=job_id)
        if show_tasks:
            tasks = batch_client.task.list(job_id)
    except BatchErrorException as e:
        print(e.message.value)
        return

    job = job.as_dict()

    if raw:
        print(json.dumps(job, sort_keys=True, indent=2))
        print(json.dumps(tasks, sort_keys=True, indent=2))
        return

    print_basic_job_details(job)

    if tasks is not None:
        print()
        print("Tasks\n-----")
        total = 0
        state_counts = defaultdict(int)
        for task in tasks:
            task = task.as_dict()
            dt = dateutil.parser.isoparse(task['state_transition_time'])
            dt = dt.strftime("%d %b %Y %H:%M")
            total += 1
            state_counts[task['state']] += 1
            running_time = ""
            if task["state"] == "completed":
                if "execution_info" in task:
                    start_time = dateutil.parser.isoparse(task["execution_info"]["start_time"])
                    end_time = dateutil.parser.isoparse(task["execution_info"]["end_time"])
                    running_time = str(end_time - start_time).split(".")[0]
            print(f"{task['id']}\t\t{task['state']}\t\t{dt}\t\t{running_time}")
        print()
        status_line = []
        for k, v in state_counts.items():
            status_line.append(f"{k}: {v}")
        status_line.append(f"total: {total}")
        print("; ".join(status_line))

    if job["state"] == "completed":
        print("\nTo download output run:\n")
        print(f"\ttlo batch-download {job_id}")


@cli.command()
@click.option("--find", "-f", type=str, default=None, help="Show jobs where identifier contains supplied string")
@click.option("--completed", "status", flag_value="completed", default=False, help="Only display completed jobs")
@click.option("--active", "status", flag_value="active", default=False, help="Only display active jobs")
@click.option("-n", default=5, type=int, help="Maximum number of jobs to list (default is 5)")
@click.option("--username", type=str, hidden=True)
@click.pass_context
def batch_list(ctx, status, n, find, username):
    """List and find running and completed jobs.
    By default, the 5 most recent jobs are displayed for the current user.
    """
    print("Querying Batch...")
    config = load_config(ctx.obj["config_file"])

    if username is None:
        username = config["DEFAULT"]["USERNAME"]

    batch_client = get_batch_client(
        config["BATCH"]["CLIENT_ID"], config["BATCH"]["SECRET"], config["AZURE"]["TENANT_ID"], config["BATCH"]["URL"]
    )

    # create client to connect to file share
    share_client = ShareClient.from_connection_string(config['STORAGE']['CONNECTION_STRING'],
                                                      config['STORAGE']['FILESHARE'])

    # get list of all directories in user_directory
    directories = list(share_client.list_directories_and_files(f"{username}/"))

    if len(directories) == 0:
        print("No jobs found.")
        return

    # convert directories to set
    directories = set([directory["name"] for directory in directories])

    # get all jobs in batch system
    jobs_list = list(batch_client.job.list(
        job_list_options=batch_models.JobListOptions(
            expand='stats'
        )
    ))

    # create a dataframe of the jobs, using the job.as_dict() record
    # filter the list of jobs by those ids in the directories
    jobs_list = [job.as_dict() for job in jobs_list if job.id in directories]
    jobs: pd.DataFrame = pd.DataFrame(jobs_list)
    jobs = jobs[["id", "creation_time", "state"]]

    # get subset where id contains the find string
    if find is not None:
        jobs = jobs[jobs["id"].str.contains(find)]

    # filter by status
    if status is not None:
        jobs = jobs[jobs["state"] == status]

    if len(jobs) == 0:
        print("No jobs found.")
        return

    # sort by creation time
    jobs = jobs.sort_values("creation_time", ascending=False)

    # split datetime into date and time
    jobs["creation_time"] = pd.to_datetime(jobs.creation_time)
    jobs["creation_date"] = jobs.creation_time.dt.date
    jobs["creation_time"] = jobs.creation_time.dt.floor("S").dt.time

    # reorder columns
    jobs = jobs[["id", "creation_date", "creation_time", "state"]]

    # get the first n rows
    jobs = jobs.head(n)

    # print the dataframe
    print(jobs.to_string(index=False))


def print_basic_job_details(job: dict):
    """Display basic job information"""
    job_labels = {
        "id": "ID",
        "creation_time": "Creation time",
        "state": "State",
        "state_transition_time": "State transition time"
    }
    for _k, _v in job_labels.items():
        if _v.endswith("time"):
            _dt = dateutil.parser.isoparse(job[_k])
            _dt = _dt.strftime("%d %b %Y %H:%M")
            print(f"{_v.ljust(JOB_LABEL_PADDING)}: {_dt}")
        else:
            print(f"{_v.ljust(JOB_LABEL_PADDING)}: {job[_k]}")


@cli.command()
@click.argument("job_ids", type=str, nargs=-1)
@click.option("--username", type=str, hidden=True)
@click.option("--verbose", default=False, is_flag=True, hidden=True)
@click.pass_context
def batch_download(ctx, job_ids, username, verbose):
    """Download output files for a job."""
    config = load_config(ctx.obj["config_file"])

    directory_count = 0

    def walk_fileshare(dir_name):
        """Recursively visit directories, create local directories and download files"""
        nonlocal directory_count
        try:
            directories = list(share_client.list_directories_and_files(dir_name))
        except ResourceNotFoundError as e:
            print("ERROR:", dir_name, "not found.")
            print()
            print(e.message)
            return
        create_dir = Path(".", "outputs", dir_name)
        os.makedirs(create_dir, exist_ok=True)
        if verbose:
            print("Creating directory", str(create_dir))
            print("Downloading", dir_name)

        for item in directories:
            if item["is_directory"]:
                walk_fileshare(f"{dir_name}/{item['name']}")
                print(f"\r{directory_count} directories downloaded", end="")
                directory_count += 1
            else:
                filepath = f"{dir_name}/{item['name']}"
                file_client = share_client.get_file_client(filepath)
                dest_file_name = Path(".", "outputs", dir_name, item["name"])
                if verbose:
                    print("File:", filepath, "\n\t->", dest_file_name)
                with open(dest_file_name, "wb") as data:
                    # Download the file from Azure into a stream
                    stream = file_client.download_file()
                    # Write the stream to the local file
                    data.write(stream.readall())

    if username is None:
        username = config["DEFAULT"]["USERNAME"]

    share_client = ShareClient.from_connection_string(config['STORAGE']['CONNECTION_STRING'],
                                                      config['STORAGE']['FILESHARE'])

    for job_id in job_ids:
        # if the job directory exist, print error and continue to next job_id
        top_level = f"{username}/{job_id}"
        destination = Path(".", "outputs", top_level)
        if os.path.exists(destination):
            print("ERROR: Local directory already exists. Please move or delete.")
            print("Directory:", destination)
            continue

        print(f"Downloading {top_level}")
        walk_fileshare(top_level)
        print("\rDownload complete.              ")


def load_config(config_file):
    """Load configuration for accessing Batch services"""
    config = configparser.ConfigParser()
    config.read(config_file)
    server_config = load_server_config(config["AZURE"]["KV_URI"], config["AZURE"]["TENANT_ID"])
    merged_config = {**config, **server_config}
    return merged_config


def load_server_config(kv_uri, tenant_id) -> Dict[str, Dict]:
    """Retrieve the server configuration for running Batch using the user's Azure credentials

    Allows user to login using credentials from Azure CLI or interactive browser.

    On Windows, login might fail because pywin32 is not installed correctly. Resolve by
    running (as Administrator) `python Scripts\\pywin32_postinstall.py -install`
    For more information, see https://github.com/mhammond/pywin32/issues/1431
    """
    credential = DefaultAzureCredential(
        interactive_browser_tenant_id=tenant_id,
        exclude_cli_credential=False,
        exclude_interactive_browser_credential=False,
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True,
        exclude_visual_studio_code_credential=True,
        exclude_shared_token_cache_credential=True
    )

    client = SecretClient(vault_url=kv_uri, credential=credential)
    storage_config = json.loads(client.get_secret("storageaccount").value)
    batch_config = json.loads(client.get_secret("batchaccount").value)
    registry_config = json.loads(client.get_secret("registryserver").value)

    return {"STORAGE": storage_config, "BATCH": batch_config, "REGISTRY": registry_config}


def get_batch_client(client_id, secret, tenant_id, url):
    """Create a Batch service client"""
    resource = "https://batch.core.windows.net/"

    credentials = ServicePrincipalCredentials(client_id=client_id, secret=secret, tenant=tenant_id, resource=resource)

    batch_client = batch.BatchServiceClient(credentials, batch_url=url)
    return batch_client


def load_scenario(scenario_file):
    """Load the Scenario class from the specified file"""
    scenario_path = Path(scenario_file)
    scenario_class = ScenarioLoader(scenario_path.parent / scenario_path.name).get_scenario()
    print(f"Found class {scenario_class.__class__.__name__} in {scenario_path}")
    return scenario_class


def is_file_clean(scenario_file):
    """Checks whether the scenario file and current branch is clean and unchanged.

    :returns: current branch name if all okay, False otherwise
    """
    repo = Repo(".")  # assumes you're running tlo command from TLOmodel root directory

    if scenario_file in repo.untracked_files:
        click.echo(
            f"ERROR: Untracked file {scenario_file}. Add file to repository, commit and push."
        )
        return False

    if repo.is_dirty(path=scenario_file):
        click.echo(
            f"ERROR: Uncommitted changes in file {scenario_file}. Rollback or commit+push changes."
        )
        return False

    current_branch = repo.head.reference
    commits_ahead = list(repo.iter_commits(f"origin/{current_branch}..{current_branch}"))
    commits_behind = list(repo.iter_commits(f"{current_branch}..origin/{current_branch}"))
    if not len(commits_behind) == len(commits_ahead) == 0:
        click.echo(f"ERROR: Branch '{current_branch}' isn't in-sync with remote: "
                   f"{len(commits_ahead)} ahead; {len(commits_behind)} behind. "
                   "Push and/or pull changes.")
        return False

    return current_branch


def print_batch_exception(batch_exception):
    """Prints the contents of the specified Batch exception.

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
    """Uses a ShareClient object to create a share if it does not exist."""
    try:
        # Create a ShareClient from a connection string
        share_client = ShareClient.from_connection_string(
            connection_string, share_name)

        print("Creating share:", share_name)
        share_client.create_share()

    except ResourceExistsError as ex:
        print("ResourceExistsError:", ex.message.splitlines()[0])


def create_directory(connection_string, share_name, dir_name):
    """Creates a directory in the root of the specified file share by using a
    ShareDirectoryClient object.
    """
    try:
        # Create a ShareDirectoryClient from a connection string
        dir_client = ShareDirectoryClient.from_connection_string(
            connection_string, share_name, dir_name)

        print("Creating directory:", share_name + "/" + dir_name)
        dir_client.create_directory()

    except ResourceExistsError as ex:
        print("ResourceExistsError:", ex.message.splitlines()[0])


def upload_local_file(connection_string, local_file_path, share_name, dest_file_path):
    """Uploads the contents of the specified file into the specified directory in
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


def create_job(
    batch_service_client,
    vm_size,
    pool_node_count,
    job_id,
    container_conf,
    mount_configuration,
    keep_pool_alive,
    subnet_id,
):
    """Creates a job with the specified ID, associated with the specified pool.

    :param subnet_id:
    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str vm_size: Type of virtual machine to use as pool.
    :param int pool_node_count: Number of nodes in the pool.
    :param str job_id: The ID for the job.
    :param container_conf: Configuration of a container.
    :type container_conf: `azure.batch.models.ContainerConfiguration`
    :param mount_configuration: Configuration of the images to mount on the nodes.
    :type mount_configuration: `list[azure.batch.models.MountConfiguration]`
    :param bool keep_pool_alive: auto pool lifetime configuration - use to debug
    """
    print("Creating job.")

    # From https://docs.microsoft.com/en-us/azure/batch/batch-docker-container-workloads#linux-support
    # We require Ubuntu image with container support
    # Get the latest SKU by inspecting output of `az batch pool supported-images list` for publisher+offer
    # Update node_agent_sku_id (below), if necessary
    image_reference = batch_models.ImageReference(
        publisher="microsoft-dsvm",
        offer="ubuntu-hpc",
        sku="2204",
        version="latest",
    )

    virtual_machine_configuration = batch_models.VirtualMachineConfiguration(
        image_reference=image_reference,
        container_configuration=container_conf,
        node_agent_sku_id="batch.node.ubuntu 22.04",
    )

    auto_scale_formula = f"""
    startingNumberOfVMs = {pool_node_count};
    maxNumberofVMs = {pool_node_count};
    pTaskSamplePerc = $PendingTasks.GetSamplePercent(120 * TimeInterval_Second);
    pTaskSample = pTaskSamplePerc < 70 ? startingNumberOfVMs : avg($PendingTasks.GetSample(120 * TimeInterval_Second));
    $TargetDedicatedNodes=min(maxNumberofVMs, pTaskSample);
    $NodeDeallocationOption = taskcompletion;
    """

    network_configuration = batch_models.NetworkConfiguration(
        subnet_id=subnet_id,
        public_ip_address_configuration=batch_models.PublicIPAddressConfiguration(provision="noPublicIPAddresses"),
    )

    pool = batch_models.PoolSpecification(
        virtual_machine_configuration=virtual_machine_configuration,
        vm_size=vm_size,
        mount_configuration=mount_configuration,
        task_slots_per_node=1,
        enable_auto_scale=True,
        auto_scale_formula=auto_scale_formula,
        network_configuration=network_configuration,
        target_node_communication_mode="simplified",
    )

    auto_pool_specification = batch_models.AutoPoolSpecification(
        pool_lifetime_option="job",
        pool=pool,
        keep_alive=keep_pool_alive,
    )

    pool_info = batch_models.PoolInformation(
        auto_pool_specification=auto_pool_specification,
    )

    job = batch_models.JobAddParameter(
        id=job_id,
        pool_info=pool_info,
        on_all_tasks_complete="terminateJob",
    )

    batch_service_client.job.add(job)


def add_tasks(batch_service_client, user_identity, job_id,
              image_name, container_run_options, scenario, command):
    """Adds the simulation tasks in the collection to the specified job.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param user_identity: User account to use in the jobs.
    :type user_identity: `azure.batch.models.UserIdentity`
    :param str job_id: The ID of the job to which to add the tasks.
    :param str image_name: Name of the Docker image to mount for the task.
    :param str container_run_options: Options to pass to Docker to run the image.
    :param scenario.Scenario scenario: instance of Scenario we're running
    :param str command: Command to run during the taks inside the Docker image.
    """

    print("Adding {} task(s) to job.".format(scenario.number_of_draws * scenario.runs_per_draw))

    tasks = list()

    task_container_settings = batch_models.TaskContainerSettings(
        image_name=image_name,
        container_run_options=container_run_options,
    )

    for draw_number in range(0, scenario.number_of_draws):
        for run_number in range(0, scenario.runs_per_draw):
            cmd = command.format(draw_number=draw_number, run_number=run_number)
            task = batch_models.TaskAddParameter(
                id=f"draw_{draw_number}-run_{run_number}",
                command_line=cmd,
                container_settings=task_container_settings,
                user_identity=user_identity,
            )
            tasks.append(task)

    batch_service_client.task.add_collection(job_id, tasks)


@cli.command()
@click.argument(
    "output_results_directory",
    type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path),
)
@click.argument(
    "additional_result_directories",
    nargs=-1,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def combine_runs(output_results_directory: Path, additional_result_directories: tuple[Path]) -> None:
    """Combine runs from multiple batch jobs locally.

    Merges runs from each draw in one or more additional results directories in
    to corresponding draws in output results directory.

    All results directories must contain same draw numbers and the draw numbers
    must be consecutive integers starting from 0. All run numbers in the output
    result directory draw directories must be consecutive integers starting
    from 0.
    """
    if len(additional_result_directories) == 0:
        msg = "One or more additional results directories to merge must be specified"
        raise click.UsageError(msg)
    results_directories = (output_results_directory,) + additional_result_directories
    draws_per_directory = [
        sorted(
            int(draw_directory.name)
            for draw_directory in results_directory.iterdir()
            if draw_directory.is_dir()
        )
        for results_directory in results_directories
    ]
    for draws in draws_per_directory:
        if not draws == list(range(len(draws_per_directory[0]))):
            msg = (
                "All results directories must contain same draws, "
                "consecutively numbered from 0."
            )
            raise click.UsageError(msg)
    draws = draws_per_directory[0]
    runs_per_draw = [
        sorted(
            int(run_directory.name)
            for run_directory in (output_results_directory / str(draw)).iterdir()
            if run_directory.is_dir()
        )
        for draw in draws
    ]
    for runs in runs_per_draw:
        if not runs == list(range(len(runs))):
            msg = "All runs in output directory must be consecutively numbered from 0."
            raise click.UsageError(msg)
    for results_directory in additional_result_directories:
        for draw in draws:
            run_counter = len(runs_per_draw[draw])
            for source_path in sorted((results_directory / str(draw)).iterdir()):
                if not source_path.is_dir():
                    continue
                destination_path = output_results_directory / str(draw) / str(run_counter)
                run_counter = run_counter + 1
                copytree(source_path, destination_path)


if __name__ == "__main__":
    cli(obj={})
