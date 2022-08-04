from typing import Any, Dict, Optional

import os

import azureml.train.automl.runtime._hts.hts_runtime_utilities as hru

from azureml._restclient.jasmine_client import JasmineClient
from azureml.contrib.automl.pipeline.steps import utilities
from azureml.core import RunConfiguration
from azureml.core.compute import ComputeTarget
from azureml.core.experiment import Experiment
from azureml.data import LinkTabularOutputDatasetConfig, TabularDataset
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep, PythonScriptStep
from azureml.train.automl.constants import Scenarios
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig


PROJECT_FOLDER = "assets"
SETTINGS_FILE = "automl_settings.json"


def get_backtest_pipeline(
    experiment: Experiment,
    dataset: TabularDataset,
    process_per_node: int,
    node_count: int,
    compute_target: ComputeTarget,
    automl_settings: Dict[str, Any],
    step_size: int,
    step_number: int,
    model_name: Optional[str] = None,
    model_uid: Optional[str] = None,
) -> Pipeline:
    """
    :param experiment: The experiment used to run the pipeline.
    :param dataset: Tabular data set to be used for model training.
    :param process_per_node: The number of processes per node. Generally it should be the number of cores
                             on the node divided by two.
    :param node_count: The number of nodes to be used.
    :param compute_target: The compute target to be used to run the pipeline.
    :param model_name: The name of a model to be back tested.
    :param automl_settings: The dictionary with automl settings.
    :param step_size: The number of periods to step back in backtesting.
    :param step_number: The number of backtesting iterations.
    :param model_uid: The uid to mark models from this run of the experiment.
    :return: The pipeline to be used for model retraining.
             **Note:** The output will be uploaded in the pipeline output
             called 'score'.
    """
    jasmine_client = JasmineClient(
        service_context=experiment.workspace.service_context,
        experiment_name=experiment.name,
        experiment_id=experiment.id,
    )
    env = jasmine_client.get_curated_environment(
        scenario=Scenarios.AUTOML,
        enable_dnn=False,
        enable_gpu=False,
        compute=compute_target,
        compute_sku=experiment.workspace.compute_targets.get(
            compute_target.name
        ).vm_size,
    )
    data_results = PipelineData(
        name="results", datastore=None, pipeline_output_name="results"
    )
    ############################################################
    # Split the data set using python script.
    ############################################################
    run_config = RunConfiguration()
    run_config.docker.use_docker = True
    run_config.environment = env

    utilities.set_environment_variables_for_run(run_config)

    split_data = PipelineData(name="split_data_output", datastore=None).as_dataset()
    split_step = PythonScriptStep(
        name="split_data_for_backtest",
        script_name="data_split.py",
        inputs=[dataset.as_named_input("training_data")],
        outputs=[split_data],
        source_directory=PROJECT_FOLDER,
        arguments=[
            "--step-size",
            step_size,
            "--step-number",
            step_number,
            "--time-column-name",
            automl_settings.get("time_column_name"),
            "--time-series-id-column-names",
            automl_settings.get("grain_column_names"),
            "--output-dir",
            split_data,
        ],
        runconfig=run_config,
        compute_target=compute_target,
        allow_reuse=False,
    )
    ############################################################
    # We will do the backtest the parallel run step.
    ############################################################
    settings_path = os.path.join(PROJECT_FOLDER, SETTINGS_FILE)
    hru.dump_object_to_json(automl_settings, settings_path)
    mini_batch_size = PipelineParameter(name="batch_size_param", default_value=str(1))
    back_test_config = ParallelRunConfig(
        source_directory=PROJECT_FOLDER,
        entry_script="retrain_models.py",
        mini_batch_size=mini_batch_size,
        error_threshold=-1,
        output_action="append_row",
        append_row_file_name="outputs.txt",
        compute_target=compute_target,
        environment=env,
        process_count_per_node=process_per_node,
        run_invocation_timeout=3600,
        node_count=node_count,
    )
    utilities.set_environment_variables_for_run(back_test_config)
    forecasts = PipelineData(name="forecasts", datastore=None)
    if model_name:
        parallel_step_name = "{}-backtest".format(model_name.replace("_", "-"))
    else:
        parallel_step_name = "AutoML-backtest"

    prs_args = [
        "--target_column_name",
        automl_settings.get("label_column_name"),
        "--output-dir",
        forecasts,
    ]
    if model_name is not None:
        prs_args.append("--model-name")
        prs_args.append(model_name)
    if model_uid is not None:
        prs_args.append("--model-uid")
        prs_args.append(model_uid)
    backtest_prs = ParallelRunStep(
        name=parallel_step_name,
        parallel_run_config=back_test_config,
        arguments=prs_args,
        inputs=[split_data],
        output=forecasts,
        allow_reuse=False,
    )
    ############################################################
    # Then we collect the output and return it as scores output.
    ############################################################
    collection_step = PythonScriptStep(
        name="score",
        script_name="score.py",
        inputs=[forecasts.as_mount()],
        outputs=[data_results],
        source_directory=PROJECT_FOLDER,
        arguments=["--forecasts", forecasts, "--output-dir", data_results],
        runconfig=run_config,
        compute_target=compute_target,
        allow_reuse=False,
    )
    # Build and return the pipeline.
    return Pipeline(
        workspace=experiment.workspace,
        steps=[split_step, backtest_prs, collection_step],
    )
