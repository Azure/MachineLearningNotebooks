This is a guide of how to write and test an entry script.

## General Guide for Writing an Entry Script
1. Find a sample close to your scenario.
2. Copy the sample, and revise the user logic. Keep the skeleton as is.
3. Revise and run test cases. A set of well defined test cases will speed up you development.

## Samples
| Scenario | Note  | Test Case |
|---|---|---|
| [Simple File DataSet](simple_file_dataset.py) | run() accepts a list of file names and returns a list. | `pytest -s test_simple_file_dataset.py`
| [Simple Tabular DataSet](simple_tabular_dataset.py) | run() accepts and returns a pandas DataFrame. | `pytest -s test_simple_tabular_dataset.py`
| [Init once with singleton](init_once_singleton.py)| Init once inside init() and use in run(). | `pytest -s test_init_once_singleton.py`
| [Init once with global variable](init_once_global_variable.py)| Init once inside init() and use in run(). | `pytest -s test_init_once_global_variable.py`
| [Use logger](using_logger.py)| Show how to use logger. | `pytest -s test_using_logger.py`


## About EntryScript
ParallelRunStep provide a help class `azureml_user.parallel_run.EntryScript`, which has dependency of AmlCompute.
To allow running your code locally, you can substitute EntryScript with DummyEntryScript for local testing as below:
```python
import os

AML_COMPUTE = "AZUREML_RUN_ID" in os.environ  # Inside AmlCompute.
if AML_COMPUTE:
    from azureml_user.parallel_run import EntryScript
else:  # Fallback to the dummy helper for local testing.
    from .dummy_entry_script import DummyEntryScript as EntryScript

```