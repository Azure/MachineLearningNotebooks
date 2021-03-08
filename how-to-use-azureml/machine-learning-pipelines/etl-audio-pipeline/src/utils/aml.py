"""
Utilities for interacting with Azure Machine Learning (AML)
"""
import logging
import os
import re
import sys
from typing import Optional

from azureml.core import Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.datastore import Datastore
from azureml.exceptions import ComputeTargetException
from msrest.exceptions import HttpOperationError


def get_or_create_compute(
    workspace: Workspace,
    compute_name: str,
    vm_size: str,
    vm_priority: str = "lowpriority",
    min_nodes: Optional[int] = 0,
    max_nodes: Optional[int] = 16,
    scale_down: Optional[int] = 120,
    vnet_name: Optional[str] = None,
    vnet_subnet_name: Optional[str] = None,
    vnet_resourcegroup_name: Optional[str] = None,
):
    """
    Returns an existing compute or creates a new one.

    Parameters
    ----------
    workspace : Workspace
        AzureML workspace
    compute_name : str
        Name of the compute
    vm_size : str
        VM size
    vm_priority : {"lowpriority", "dedicated"}
        Low priority or dedicated cluster
    min_nodes : int, optional
        Minimum number of nodes
    max_nodes : int, optional
        Maximum number of nodes in the cluster
    scale_down : int, optional
        Number of seconds to wait before scaling down the cluster
    vnet_name : str, optional
        A name of the virtual network for compute cluster
    vnet_subnet_name : str, optional
        A name of the subnet
    vnet_resourcegroup_name : str, optional
        A name of the resource group

    Returns
    -------
    compute_target : ComputeTarget
        A reference to the AML compute

    Raises
    ------
    ComputeTargetException
        If the compute was unable to be accessed (if it exists) or unable to be provisioned (if it did not exist)
    """
    # Currently a bug for numpy docstring for parameters with a set of options
    # https://github.com/PyCQA/pylint/issues/4035
    # pylint: disable=missing-param-doc
    log = get_logger(__name__)

    try:
        if compute_name in workspace.compute_targets:
            compute_target = workspace.compute_targets[compute_name]
            if compute_target and isinstance(compute_target, AmlCompute):
                log.info(f"Found existing compute target {compute_name} so using it.")
        else:
            log.info(
                f"Did not find existing compute target {compute_name}. Creating {compute_name}"
            )
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=vm_size,
                vm_priority=vm_priority,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                idle_seconds_before_scaledown=scale_down,
                vnet_name=vnet_name,
                subnet_name=vnet_subnet_name,
                vnet_resourcegroup_name=vnet_resourcegroup_name,
            )

            compute_target = ComputeTarget.create(
                workspace, compute_name, compute_config
            )
            compute_target.wait_for_completion(show_output=True)
            log.info(f"Created compute target {compute_name}")
        return compute_target
    except ComputeTargetException as err:
        log.error(f"An error occurred trying to provision {compute_name}: {str(err)}")
        sys.exit(-1)


def get_or_register_blob_datastore(
    workspace: Workspace,
    datastore_name: str,
    storage_name: str,
    storage_key: str,
    container_name: str,
):
    """
    Returns a reference to an AML datastore.

    If the AML Datastore does not already exist, it will create one instead

    Parameters
    ----------
    workspace : AML
        Existing AzureML Workspace object
    datastore_name : str
        Name of AML Datastore
    storage_name : str
        Blob storage account name
    storage_key : str
        Blob storage account key
    container_name : str
        Blob container name

    Returns
    -------
    blob_datastore : Datastore
        Reference to the found or newly created AML Datastore
    """
    log = get_logger(__name__)

    try:
        blob_datastore = Datastore.get(workspace, datastore_name)
        log.info(f"Found Blob Datastore with name: {datastore_name}")
    except HttpOperationError:
        blob_datastore = Datastore.register_azure_blob_container(
            workspace=workspace,
            datastore_name=datastore_name,
            account_name=storage_name,
            container_name=container_name,
            account_key=storage_key,
        )
        log.info(f"Registered blob datastore with name: {datastore_name}")
    return blob_datastore


def get_logger(name: str):
    """Logging with the user of Azure Machine Learning

    Due to the fact that ParallelRunStep utilizes a different logger than it only obtainable by calling EntryScript,
    having a wrapper get_logger function allows logs to be utilized on and off Azure Machine Learning

    Parameters
    ----------
    name : str
        Name of logger to retrieve

    Returns
    -------
    log : logging.Logger
        Log with the handlers set to output to either Azure Machine Learning or to local stdout
    """
    if "AZUREML_RUN_ID" in os.environ:
        # The package azureml_user is not a public package but exists when a container is built on top of
        # Azure Machine Learning. Thus, we import only when we are within that environment
        # pylint: disable=import-error,import-outside-toplevel
        from azureml_user.parallel_run import EntryScript

        entry_script = EntryScript()
        log = entry_script.logger
    else:
        log = logging.getLogger(name)

    return log


def remove_mini_batch_directory_from_path(filepath: str):
    """Remove the mini batch directory used during ParallelRunStep from a Dataset.File.from_files call
    from the filepaths.

    Parameters
    ----------
    filepath : str
        Filepath to remove the mini batch directory from

    Returns
    -------
    filepath : str
        Filepath with mini batch directory removed
    """
    filepath = re.sub(
        r".*azureml/[0-9a-f]{8}\b-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-\b[0-9a-f]{12}/wd/tmp[a-z0-9]*/",
        "",
        filepath,
    )

    return filepath
