# %% [markdown]
# # Extract, Transform, Load Audio Azure Machine Learning Pipeline
#
# The following jupytext notebook / script publishes and runs an Azure Machine Learning Pipeline to apply preprocessing functions to audio data contained in an Azure Blob Storage container.
#
# The following steps executed are
# - Extraction of audio from video files
# - Dynamic Range Compression to amplify main signal
# - Denoising to reduce background noise
#
# Note:
#
# This notebook is paired with the script `create_etl_pipeline.py` using [jupytext](https://github.com/mwouts/jupytext). Any updates to either this notebook or the script will result in changes for both files. This notebook is intended to serve as a previewable walkthrough of the script via Jupyter preview.

# %%
# pylint: skip-file
import os

from dotenv import load_dotenv
from IPython import get_ipython

if get_ipython() is not None:
    os.chdir("..")

load_dotenv()

# %%
import logging
import sys

from azureml.core import Dataset, Environment, Experiment, Workspace
from azureml.data.data_reference import DataReference
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.pipeline.core import Pipeline, PipelineParameter
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep
from src.utils.aml import (
    get_logger,
    get_or_create_compute,
    get_or_register_blob_datastore,
)

# %%
logging.basicConfig(level=logging.INFO)
log = get_logger(__name__)
stdout_handler = logging.StreamHandler(sys.stdout)
log.addHandler(stdout_handler)


# %% [markdown]
# ## Setting Environment Variables
#
# Azure Machine Learning Compute Variables
# - `AML_COMPUTE_NAME` a compute name for the compute instance
# - `AML_COMPUTE_VM_PRIORITY` should be set to a VM size listed under "Size" per the names in the following doc https://docs.microsoft.com/en-us/azure/cloud-services/cloud-services-sizes-specs
# - `AML_COMPUTE_MIN_NODES` is the minimum number of nodes that will be allocated even when no pipelines are running
# - `AML_COMPUTE_MAX_NODES` is the maximum number of nodes to allocate even if there are more compute nodes requested than pipelines queued
# - `AML_COMPUTE_SCALE_DOWN` the amount of time in seconds for a compute node to stay idle before deallocating
#
# Azure Blob Storage Variables
# - `AML_BLOB_DATASTORE_NAME` the name to register on Azure Machine Learning for the associated Azure Blob Storage instance
# - `AML_BLOB_ACCOUNT_NAME` the account name for the Azure Blob Storage instance
# - `AML_BLOB_ACCOUNT_KEY` the account key for the Azure Blob Storage instance
# - `AML_BLOB_CONTAINER_NAME` the name of the container to read and write source video / audio files to processed audio files

# %%
# Azure Machine Learning (AML) Compute Variables
AML_COMPUTE_NAME = os.getenv("AML_COMPUTE_NAME")
AML_COMPUTE_VM_SIZE = os.getenv("AML_COMPUTE_VM_SIZE")
AML_COMPUTE_VM_PRIORITY = os.getenv("AML_COMPUTE_VM_PRIORITY")
AML_COMPUTE_MIN_NODES = int(os.getenv("AML_COMPUTE_MIN_NODES"))
AML_COMPUTE_MAX_NODES = int(os.getenv("AML_COMPUTE_MAX_NODES"))
AML_COMPUTE_SCALE_DOWN = int(os.getenv("AML_COMPUTE_SCALE_DOWN"))

# Azure Blob Storage account to register to the AML Workspace
AML_BLOB_DATASTORE_NAME = os.getenv("AML_BLOB_DATASTORE_NAME")
AML_BLOB_ACCOUNT_NAME = os.getenv("AML_BLOB_ACCOUNT_NAME")
AML_BLOB_ACCOUNT_KEY = os.getenv("AML_BLOB_ACCOUNT_KEY")
AML_BLOB_CONTAINER_NAME = os.getenv("AML_BLOB_CONTAINER_NAME")

# %% [markdown]
# ## AML Workspace Config
#
# Reference the following AML doc https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace to obtain the `config.json` from your AML workspace

# %%
# Restore AML workspace from config.json file (can be downloaded through the portal)
ws = Workspace.from_config()

# %%
compute_target = get_or_create_compute(
    workspace=ws,
    compute_name=AML_COMPUTE_NAME,
    vm_size=AML_COMPUTE_VM_SIZE,
    vm_priority=AML_COMPUTE_VM_PRIORITY,
    min_nodes=AML_COMPUTE_MIN_NODES,
    max_nodes=AML_COMPUTE_MAX_NODES,
    scale_down=AML_COMPUTE_SCALE_DOWN,
)

# %% [markdown]
# ## Registering Azure Blob Storage
#
# Azure Machine Learning has a notion of [Datasets](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets) that are associated with a storage instance. The files will be read from a `DatasetConsumptionConfig` object generated during the `inputs` section of `ParallelRunStep`

# %%
root_datastore = get_or_register_blob_datastore(
    workspace=ws,
    datastore_name=AML_BLOB_DATASTORE_NAME,
    storage_name=AML_BLOB_ACCOUNT_NAME,
    storage_key=AML_BLOB_ACCOUNT_KEY,
    container_name=AML_BLOB_CONTAINER_NAME,
)

root_dir = DataReference(
    datastore=root_datastore, data_reference_name="source_files", mode="mount"
)

input_files = Dataset.File.from_files((root_datastore, "source"))

# %% [markdown]
# ## Pipeline Parameters
#
# These parameters allow you to configure the pipeline run. Once the pipeline is published, these parameters can also be modified via the Azure Machine Learning Portal

# %%
output_dir = PipelineParameter(name="output_dir", default_value="outputs")
overwrite = PipelineParameter(name="overwrite", default_value=True)

# Sample Rate of 0 indicates that we will use the file's default
sample_rate = PipelineParameter(name="sample_rate", default_value=0)

# Refer to documentation in steps/etl.py for the available options
transform_order = PipelineParameter(
    name="transform_order", default_value="compress, denoise"
)

# %% [markdown]
# ## Configuring the Environment
#
# The environment is built off of the requirements specified in `requirements.in`. Note that the `lock` file equivalent is `requirements.txt` so dependencies may be incrementally updated if the docker image is re-built.

# %%
env = Environment.from_pip_requirements("etl_audio", "requirements.in")
env.docker.enabled = True
env.docker.base_image = None
env.docker.base_dockerfile = "./Dockerfile"

etl_config = ParallelRunConfig(
    entry_script="mlops/steps/etl.py",
    mini_batch_size="1",
    error_threshold=0,
    output_action="summary_only",
    compute_target=compute_target,
    environment=env,
    node_count=AML_COMPUTE_MAX_NODES,
    run_invocation_timeout=600,
)

# %% [markdown]
# ## ParallelRunStep
#
# The `ParallelRunStep` receives as input a list of file paths to the source audio / video files on blob storage. After they are processed through the ETL step, they are written to the same container under `output_dir` with the same folder structure that the input files had.
#
# Note that the arguments receive a `PipelineParameter` as their input. The script `etl.py` uses `argparse` to receive these arguments and at runtime the `PipelineParameter` is converted to the standard type it represents such as `str`, `int`, `bool`, etc.

# %%
etl_step = ParallelRunStep(
    name="etl",
    parallel_run_config=etl_config,
    inputs=[input_files.as_named_input("input_files").as_mount()],
    side_inputs=[root_dir],
    arguments=[
        "--base-dir",
        root_dir,
        "--output-dir",
        output_dir,
        "--overwrite",
        overwrite,
        "--sample-rate",
        sample_rate,
        "--transform-order",
        transform_order,
    ],
)

# %%
steps = [etl_step]

etl_pipeline = Pipeline(workspace=ws, steps=steps)
etl_pipeline.validate()

etl_pipeline.publish(
    name="etl_pipeline",
    description="Extract, Transform, Load Pipeline for Audio Data",
)

exp = Experiment(ws, "etl_pipeline").submit(etl_pipeline)
exp.wait_for_completion()
