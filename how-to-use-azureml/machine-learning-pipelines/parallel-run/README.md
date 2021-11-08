# Azure Machine Learning Batch Inference

Azure Machine Learning Batch Inference targets large inference jobs that are not time-sensitive. Batch Inference provides cost-effective inference compute scaling, with unparalleled throughput for asynchronous applications. It is optimized for high-throughput, fire-and-forget inference over large collections of data.

# Getting Started with Batch Inference

Batch inference offers a platform in which to do large inference or generic parallel map-style operations. Below introduces the major steps to use this new functionality. For a quick try, please follow the prerequisites and simply run the sample notebooks provided in this directory.

## Prerequisites

### Python package installation
If you're unfamiliar with creating a new Python environment, you may follow this example for [creating a conda environment](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#local). Batch Inference package can be installed through the following pip command.
```
pip install azureml-pipeline-steps
```

### Creation of Azure Machine Learning Workspace
If you do not already have a Azure ML Workspace, please run the [configuration Notebook](https://aka.ms/pl-config).

## Configure a Batch Inference job

To run a Batch Inference job, you will need to gather some configuration data.

1. **ParallelRunConfig**
    - **entry_script**: the local file path to the scoring script. If source_directory is specified, use relative path, otherwise use any path accessible on machine.
    - **error_threshold**: the number of record failures for TabularDataset and file failures for FileDataset that should be ignored during processing. If the aggregated error count (across all workers) goes above this value, then the job will be aborted. Set to -1 to ignore all failures during processing.
    - **output_action**: one of the following values
        - **"append_row"**: all values output by run() method invocations will be aggregated into one unique file named parallel_run_step.txt that is created in the output location.
        - **"summary_only"** – scoring script will handle the output by itself.  The script still needs to return one output row per successfully-processed input item. This is used for error threshold calculation (the actual value of the output row is ignored).
    - **source_directory**: supporting files for scoring (optional)
    - **compute_target**: only **AmlCompute** is supported currently
    - **node_count**: number of compute nodes to use.
    - **process_count_per_node**: number of processes per node (optional, default value is 1).
    - **mini_batch_size**: the approximate amount of input data passed to each run() invocation.  For FileDataset input, this is number of files user script can process in one run() call. For TabularDataset input it is approximate size of data user script can process in one run() call. E.g. 1024, 1024KB, 10MB, 1GB (optional, default value 10 files for FileDataset and 1MB for TabularDataset.)
    - **partition_keys**: the keys used to partition the input data into mini-batches passed to each run() invocation. This parameter is mutually exclusive with `mini_batch_size`, and it requires the input datasets to have `partition_keys` attribute, the value of which is a superset of the value of this parameter. Each run() call would process a part of data that has identical value on the `partition_keys` specified. You can follow the examples in [file-dataset-partition-per-folder.ipynb](./file-dataset-partition-per-folder.ipynb) and [tabular-dataset-partition-per-column.ipynb](./tabular-dataset-partition-per-column.ipynb) to see how to create such datasets.
    - **logging_level**: log verbosity. Values in increasing verbosity are: 'WARNING', 'INFO', 'DEBUG' (optional, default value is 'INFO').
    - **run_invocation_timeout**: run method invocation timeout period in seconds (optional, default value is 60).
    - **environment**: The environment definition. This field configures the Python environment. It can be configured to use an existing Python environment or to set up a temp environment for the experiment. The definition is also responsible for setting the required application dependencies.
    - **description**: name given to batch service.

2. **Scoring (entry) script**: entry point for execution, scoring script should contain two functions:
    - **init()**: this function should be used for any costly or common preparation for subsequent inferences, e.g., deserializing and loading the model into a global object.
    - **run(mini_batch)**: The method to be parallelized. Each invocation will have one minibatch.
        - **mini_batch**: Batch inference will invoke run method and pass either a list or Pandas DataFrame as an argument to the method. Each entry in min_batch will be - a filepath if input is a FileDataset, a Pandas DataFrame if input is a TabularDataset.
        - **return value**: run() method should return a Pandas DataFrame or an array. For append_row output_action, these returned elements are appended into the common output file. For summary_only, the contents of the elements are ignored. For all output actions, each returned output element indicates one successful inference of input element in the input mini-batch.

3. **Base image** (optional)
    - if GPU is required, use DEFAULT_GPU_IMAGE as base image in environment. [Example GPU environment](./file-dataset-image-inference-mnist.ipynb#specify-the-environment-to-run-the-script)

Example image pull:
```python
from azureml.core.runconfig import ContainerRegistry

# use an image available in public Container Registry without authentication
public_base_image = "mcr.microsoft.com/azureml/o16n-sample-user-base/ubuntu-miniconda"

# or use an image available in a private Container Registry
base_image = "myregistry.azurecr.io/mycustomimage:1.0"
base_image_registry = ContainerRegistry()
base_image_registry.address = "myregistry.azurecr.io"
base_image_registry.username = "username"
base_image_registry.password = "password"
```


## Create a batch inference job

**ParallelRunStep** is a newly added step in the azureml.pipeline.steps package. You will use it to add a step to create a batch inference job with your Azure machine learning pipeline. (Use batch inference without an Azure machine learning pipeline is not supported yet). ParallelRunStep has all the following parameters:
  - **name**: this name will be used to register batch inference service, has the following naming restrictions: (unique, 3-32 chars and regex ^\[a-z\]([-a-z0-9]*[a-z0-9])?$)
  - **parallel_run_config**: ParallelRunConfig as defined above.
  - **inputs**: one or more Dataset objects.
  - **output**: this should be a PipelineData object encapsulating an Azure BLOB container path.
  - **arguments**: list of custom arguments passed to scoring script (optional)
  - **allow_reuse**: optional, default value is True. If the inputs remain the same as a previous run, it will make the previous run results immediately available (skips re-computing the step).

## Passing arguments from pipeline submission to script

Many tasks require arguments to be passed from job submission to the distributed runs. Below is an example to pass such information.
```
# from script which creates pipeline job
parallelrun_step = ParallelRunStep(
  ...
  arguments=["--model_name", "mosaic"]     # name of the model we want to use, in case we have more than one option
)
```
```
# from driver.py/score.py/task.py
import argparse

parser.add_argument('--model_name', dest="model_name")

args, unknown_args = parser.parse_known_args()

# to access values
args.model_name # "mosaic"
```

## Submit a batch inference job

You can submit a batch inference job by pipeline_run, or through REST calls with a published pipeline. To control node count using REST API/experiment, please use aml_node_count(special) pipeline parameter. A typical use case follows:

```python
pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])
pipeline_run = Experiment(ws, 'name_of_pipeline_run').submit(pipeline)
```

## Monitor your batch inference job

A batch inference job can take a long time to finish. You can monitor your job's progress from Azure portal, using Azure ML widgets, view console output through SDK, or check out overview.txt in log/azureml directory.

```python
# view with widgets (will display GUI inside a browser)
from azureml.widgets import RunDetails
RunDetails(pipeline_run).show()

# simple console output
pipeline_run.wait_for_completion(show_output=True)
```

# Sample notebooks

-  [file-dataset-image-inference-mnist.ipynb](./file-dataset-image-inference-mnist.ipynb) demonstrates how to run batch inference on an MNIST dataset using FileDataset.
-  [tabular-dataset-inference-iris.ipynb](./tabular-dataset-inference-iris.ipynb) demonstrates how to run batch inference on an IRIS dataset using TabularDataset.
-  [pipeline-style-transfer.ipynb](../pipeline-style-transfer/pipeline-style-transfer-parallel-run.ipynb) demonstrates using ParallelRunStep in multi-step pipeline and using output from one step as input to ParallelRunStep.
-  [file-dataset-partition-per-folder.ipynb](./file-dataset-partition-per-folder.ipynb) demonstrates how to run batch inference on file data by treating files inside each leaf folder as a mini-batch.
-  [tabular-dataset-partition-per-column.ipynb](./tabular-dataset-partition-per-column.ipynb) demonstrates how to run batch inference on tabular data by treating rows with identical value on specified columns as a mini-batch.

# Troubleshooting guide

- [Troubleshooting the ParallelRunStep](https://aka.ms/prstsg) includes answers to frequently asked questions. You can find more references there.

![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/how-to-use-azureml/machine-learning-pipelines/parallel-run/README.png)
