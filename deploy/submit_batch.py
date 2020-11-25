#!/bin/env python

import datetime

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

import config


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
                print("{}:\t{}".format(mesg.key, mesg.value))
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
               container_conf, azure_storage_account_name,
               azure_storage_account_key, azure_file_url,
               file_share_mount_point):
    """
    Creates a job with the specified ID, associated with the specified pool.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str vm_size: Type of virtual machine to use as pool.
    :param int pool_node_count: Number of nodes in the pool.
    :param str job_id: The ID for the job.
    :type batch_service_client: `azure.batch.models.ContainerRegistry`
    :param container_conf: Configuration of a container.
    :type batch_service_client: `azure.batch.models.ContainerConfiguration`
    :param str azure_storage_account_name: Azure Storage account name.
    :param str azure_storage_account_key: Azure Storage account key.
    :param str azure_file_url: The ID for the job.
    :param str file_share_mount_point: Mount point of the file share.
    """
    print("Creating job [{}]...".format(job_id))

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

    azure_file_share_configuration = batchmodels.AzureFileShareConfiguration(
        account_name=azure_storage_account_name,
        azure_file_url=azure_file_url,
        account_key=azure_storage_account_key,
        relative_mount_path=file_share_mount_point,
        mount_options="-o rw",
    )

    mount_configuration = batchmodels.MountConfiguration(
        azure_file_share_configuration=azure_file_share_configuration,
    )

    pool = batchmodels.PoolSpecification(
        virtual_machine_configuration=virtual_machine_configuration,
        vm_size=vm_size,
        target_dedicated_nodes=pool_node_count,
        mount_configuration=[mount_configuration],
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
              image_name, container_run_options, command):
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

    print("Adding {} tasks to job [{}]...".format(1, job_id))

    tasks = list()

    task_container_settings = batchmodels.TaskContainerSettings(
        image_name=image_name,
        container_run_options=container_run_options,
    )

    task = batchmodels.TaskAddParameter(
        id="Task{}".format(1),
        command_line=command,
        container_settings=task_container_settings,
        user_identity=user_identity,
    )

    tasks.append(task)

    batch_service_client.task.add_collection(job_id, tasks)


def submit_job():
    """
    Submit job to the cluster in the cloud.
    """
    start_time = datetime.datetime.now().replace(microsecond=0)
    print("Sample start: {}".format(start_time))
    print()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # Create a Batch service client.
    credentials = batch_auth.SharedKeyCredentials(config.AZURE_BATCH_ACCOUNT_NAME,
                                                  config.AZURE_BATCH_ACCOUNT_KEY,
                                                  )

    batch_client = batch.BatchServiceClient(
        credentials,
        batch_url=config.AZURE_BATCH_ACCOUNT_URL,
    )

    create_file_share(config.AZURE_STORAGE_CONNECTION_STRING,
                      config.SHARE_NAME,
                      )

    create_directory(config.AZURE_STORAGE_CONNECTION_STRING,
                     config.SHARE_NAME,
                     timestamp,
                     )

    upload_local_file(config.AZURE_STORAGE_CONNECTION_STRING,
                      "requirements.txt",
                      config.SHARE_NAME,
                      timestamp + "/requirements.txt",
                      )

    # Configuration of the pool: type of machines and number of nodes.
    vm_size = config.POOL_VM_SIZE
    pool_node_count = config.POOL_NODE_COUNT

    # ID of the Batch job.
    job_id = config.JOB_ID + "-" + timestamp

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
        config.AZURE_STORAGE_ACCOUNT_NAME,
        config.SHARE_NAME,
    )

    # Specify a container registry
    container_registry = batchmodels.ContainerRegistry(
        registry_server=config.AZURE_REGISTRY_SERVER,
        user_name=config.AZURE_REGISTRY_ACCOUNT_NAME,
        password=config.AZURE_REGISTRY_ACCOUNT_KEY,
    )

    # Name of the image in the registry
    image_name = config.AZURE_REGISTRY_SERVER + "/" + config.CONTAINER_IMAGE_NAME

    # Create container configuration, prefetching Docker images from the container registry
    container_conf = batchmodels.ContainerConfiguration(
        container_image_names=[image_name],
        container_registries=[container_registry],
    )

    # Options for running the Docker container
    container_run_options = "--rm --workdir /TLOmodel"

    # Directory where the file share will be mounted, relative to
    # ${AZ_BATCH_NODE_MOUNTS_DIR}.
    file_share_mount_point = config.FILE_SHARE_MOUNT_POINT

    # Command to run in the main job
    command = '/bin/sh -c "python src/scripts/profiling/batch_test.py 2;' + \
        'cp outputs/*.log ${{AZ_BATCH_NODE_MOUNTS_DIR}}/{}/{}/."'.format(file_share_mount_point,
                                                                         timestamp)
    # command = '/bin/sh -c "echo hello world > ${{AZ_BATCH_NODE_MOUNTS_DIR}}/outputs/{}/foo.txt"'.format(timestamp)

    try:
        # Create the job that will run the tasks.
        create_job(batch_client, vm_size, pool_node_count, job_id, container_conf,
                   config.AZURE_STORAGE_ACCOUNT_NAME, config.AZURE_STORAGE_ACCOUNT_KEY,
                   azure_file_url, file_share_mount_point)

        # Add the tasks to the job.
        add_tasks(batch_client, user_identity, job_id, image_name, container_run_options, command)

    except batchmodels.BatchErrorException as err:
        print_batch_exception(err)
        raise


if __name__ == "__main__":

    submit_job()
