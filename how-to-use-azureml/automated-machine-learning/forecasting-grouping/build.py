from typing import List, Dict
import copy
import json
import pandas as pd
import re

from azureml.core import RunConfiguration
from azureml.core.compute import ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.dataset import Dataset
from azureml.data import TabularDataset
from azureml.pipeline.core import PipelineData, PipelineParameter, TrainingOutput, StepSequence
from azureml.pipeline.steps import PythonScriptStep
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.runtime import AutoMLStep


def _get_groups(data: Dataset, group_column_names: List[str]) -> pd.DataFrame:
    return data._dataflow.distinct(columns=group_column_names)\
        .keep_columns(columns=group_column_names).to_pandas_dataframe()[group_column_names]


def _get_configs(automlconfig: AutoMLConfig,
                 data: Dataset,
                 target_column: str,
                 compute_target: ComputeTarget,
                 group_column_names: List[str]) -> Dict[str, AutoMLConfig]:
    # remove invalid characters regex
    valid_chars = re.compile('[^a-zA-Z0-9-]')
    groups = _get_groups(data, group_column_names)
    if groups.shape[0] > 40:
        raise RuntimeError("AutoML only supports 40 or less groups. Please modify your "
                           "group_column_names to ensure no more than 40 groups are present.")
    configs = {}
    for i, group in groups.iterrows():
        single = data._dataflow
        group_name = "#####".join(str(x) for x in group.values)
        group_name = valid_chars.sub('', group_name)
        for key in group.index:
            single = single.filter(data._dataflow[key] == group[key])
        t_dataset = TabularDataset._create(single)
        group_conf = copy.deepcopy(automlconfig)
        group_conf.user_settings['training_data'] = t_dataset
        group_conf.user_settings['label_column_name'] = target_column
        group_conf.user_settings['compute_target'] = compute_target
        configs[group_name] = group_conf
    return configs


def build_pipeline_steps(automlconfig: AutoMLConfig,
                         data: Dataset,
                         target_column: str,
                         compute_target: ComputeTarget,
                         group_column_names: list,
                         time_column_name: str,
                         deploy: bool,
                         service_name: str = 'grouping-demo') -> StepSequence:
    steps = []

    metrics_output_name = 'metrics_{}'
    best_model_output_name = 'best_model_{}'
    count = 0
    model_names = []

    # get all automl configs by group
    configs = _get_configs(automlconfig, data, target_column, compute_target, group_column_names)

    # build a runconfig for register model
    register_config = RunConfiguration()
    cd = CondaDependencies()
    cd.add_pip_package('azureml-pipeline')
    register_config.environment.python.conda_dependencies = cd

    # create each automl step end-to-end (train, register)
    for group_name, conf in configs.items():
        # create automl metrics output
        metrics_data = PipelineData(
            name='metrics_data_{}'.format(group_name),
            pipeline_output_name=metrics_output_name.format(group_name),
            training_output=TrainingOutput(type='Metrics'))
        # create automl model output
        model_data = PipelineData(
            name='model_data_{}'.format(group_name),
            pipeline_output_name=best_model_output_name.format(group_name),
            training_output=TrainingOutput(type='Model', metric=conf.user_settings['primary_metric']))

        automl_step = AutoMLStep(
            name='automl_{}'.format(group_name),
            automl_config=conf,
            outputs=[metrics_data, model_data],
            allow_reuse=True)
        steps.append(automl_step)

        # pass the group name as a parameter to the register step ->
        # this will become the name of the model for this group.
        group_name_param = PipelineParameter("group_name_{}".format(count), default_value=group_name)
        count += 1

        reg_model_step = PythonScriptStep(
            'register.py',
            name='register_{}'.format(group_name),
            arguments=["--model_name", group_name_param, "--model_path", model_data],
            inputs=[model_data],
            compute_target=compute_target,
            runconfig=register_config,
            source_directory="register",
            allow_reuse=True
        )
        steps.append(reg_model_step)
        model_names.append(group_name)

    final_steps = steps
    if deploy:
        # modify the conda dependencies to ensure we pick up correct
        # versions of azureml-defaults and azureml-train-automl
        cd = CondaDependencies.create(pip_packages=['azureml-defaults', 'azureml-train-automl'])
        automl_deps = CondaDependencies(conda_dependencies_file_path='deploy/myenv.yml')
        cd._merge_dependencies(automl_deps)
        cd.save('deploy/myenv.yml')

        # add deployment step
        pp_group_column_names = PipelineParameter(
            "group_column_names",
            default_value="#####".join(list(group_column_names)))

        pp_model_names = PipelineParameter(
            "model_names",
            default_value=json.dumps(model_names))

        pp_service_name = PipelineParameter(
            "service_name",
            default_value=service_name)

        deployment_step = PythonScriptStep(
            'deploy.py',
            name='service_deploy',
            arguments=["--group_column_names", pp_group_column_names,
                       "--model_names", pp_model_names,
                       "--service_name", pp_service_name,
                       "--time_column_name", time_column_name],
            compute_target=compute_target,
            runconfig=RunConfiguration(),
            source_directory="deploy"
        )
        final_steps = StepSequence(steps=[steps, deployment_step])

    return final_steps
