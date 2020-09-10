# Test-Driven Inference Script Development

This guide introduces how to write inference script in Test-driven development (TDD) approach. You can test your inference script work locally with one or several mini batches before submitting a job to process your full input with ParallelRunStep.

## General guide for writing an inference script
1. Find a sample close to your scenario.
2. Copy the sample, and revise the user logic. Keep the skeleton as is.
3. Revise and run test cases. A set of well defined test cases will speed up you development.

## Inference script samples
| Scenario | Note  | Test Case |
|---|---|---|
| [Simple File DataSet](simple_file_dataset.py) | run() accepts a list of file names and returns a list. | `pytest -s test_simple_file_dataset.py`
| [Simple Tabular DataSet](simple_tabular_dataset.py) | run() accepts and returns a pandas DataFrame. | `pytest -s test_simple_tabular_dataset.py`
| [Init once with singleton](init_once_singleton.py)| Init once inside init() and use in run(). | `pytest -s test_init_once_singleton.py`
| [Init once with global variable](init_once_global_variable.py)| Init once inside init() and use in run(). | `pytest -s test_init_once_global_variable.py`
| [Use logger](using_logger.py)| Show how to use logger. | `pytest -s test_using_logger.py`


## About the helper class `EntryScript`
ParallelRunStep provide a help class `azureml_user.parallel_run.EntryScript`, which has dependency of AmlCompute.
To allow running your code locally without the `azureml_user.parallel_run` package, you can substitute EntryScript with DummyEntryScript for local testing as below:
```python
import os

AML_COMPUTE = "AZUREML_RUN_ID" in os.environ  # Inside AmlCompute.
if AML_COMPUTE:
    from azureml_user.parallel_run import EntryScript
else:  # Fallback to the dummy helper for local testing.
    from .dummy_entry_script import DummyEntryScript as EntryScript

```

# Sample notebooks

-  [file-dataset-image-inference-mnist.ipynb](./file-dataset-image-inference-mnist.ipynb) demonstrates how to run batch inference on an MNIST dataset using FileDataset.
-  [tabular-dataset-inference-iris.ipynb](./tabular-dataset-inference-iris.ipynb) demonstrates how to run batch inference on an IRIS dataset using TabularDataset.
-  [pipeline-style-transfer.ipynb](../pipeline-style-transfer/pipeline-style-transfer-parallel-run.ipynb) demonstrates using ParallelRunStep in multi-step pipeline and using output from one step as input to ParallelRunStep.

![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/how-to-use-azureml/machine-learning-pipelines/parallel-run/README.png)
