# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""The batch script needed for back testing of models using PRS."""
import argparse
import json
import logging
import os
import pickle
import re

import pandas as pd

from azureml.core.experiment import Experiment
from azureml.core.model import Model
from azureml.core.run import Run
from azureml.automl.core.shared import constants
from azureml.automl.runtime.shared.score import scoring
from azureml.train.automl import AutoMLConfig

RE_INVALID_SYMBOLS = re.compile(r"[:\s]")

model_name = None
target_column_name = None
current_step_run = None
output_dir = None

logger = logging.getLogger(__name__)


def _get_automl_settings():
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "automl_settings.json"
        )
    ) as json_file:
        return json.load(json_file)


def init():
    global model_name
    global target_column_name
    global output_dir
    global automl_settings
    global model_uid
    global forecast_quantiles

    logger.info("Initialization of the run.")
    parser = argparse.ArgumentParser("Parsing input arguments.")
    parser.add_argument("--output-dir", dest="out", required=True)
    parser.add_argument("--model-name", dest="model", default=None)
    parser.add_argument("--model-uid", dest="model_uid", default=None)
    parser.add_argument(
        "--forecast_quantiles",
        nargs="*",
        type=float,
        help="forecast quantiles list",
        default=None,
    )

    parsed_args, _ = parser.parse_known_args()
    model_name = parsed_args.model
    automl_settings = _get_automl_settings()
    target_column_name = automl_settings.get("label_column_name")
    output_dir = parsed_args.out
    model_uid = parsed_args.model_uid
    forecast_quantiles = parsed_args.forecast_quantiles
    os.makedirs(output_dir, exist_ok=True)
    os.environ["AUTOML_IGNORE_PACKAGE_VERSION_INCOMPATIBILITIES".lower()] = "True"


def get_run():
    global current_step_run
    if current_step_run is None:
        current_step_run = Run.get_context()
    return current_step_run


def run_backtest(data_input_name: str, file_name: str, experiment: Experiment):
    """Re-train the model and return metrics."""
    data_input = pd.read_csv(
        data_input_name,
        parse_dates=[automl_settings[constants.TimeSeries.TIME_COLUMN_NAME]],
    )
    print(data_input.head())
    if not automl_settings.get(constants.TimeSeries.GRAIN_COLUMN_NAMES):
        # There is no grains.
        data_input.sort_values(
            [automl_settings[constants.TimeSeries.TIME_COLUMN_NAME]], inplace=True
        )
        X_train = data_input.iloc[: -automl_settings["max_horizon"]]
        y_train = X_train.pop(target_column_name).values
        X_test = data_input.iloc[-automl_settings["max_horizon"] :]
        y_test = X_test.pop(target_column_name).values
    else:
        # The data contain grains.
        dfs_train = []
        dfs_test = []
        for _, one_series in data_input.groupby(
            automl_settings.get(constants.TimeSeries.GRAIN_COLUMN_NAMES)
        ):
            one_series.sort_values(
                [automl_settings[constants.TimeSeries.TIME_COLUMN_NAME]], inplace=True
            )
            dfs_train.append(one_series.iloc[: -automl_settings["max_horizon"]])
            dfs_test.append(one_series.iloc[-automl_settings["max_horizon"] :])
        X_train = pd.concat(dfs_train, sort=False, ignore_index=True)
        y_train = X_train.pop(target_column_name).values
        X_test = pd.concat(dfs_test, sort=False, ignore_index=True)
        y_test = X_test.pop(target_column_name).values

    last_training_date = str(
        X_train[automl_settings[constants.TimeSeries.TIME_COLUMN_NAME]].max()
    )

    if file_name:
        # If file name is provided, we will load model and retrain it on backtest data.
        with open(file_name, "rb") as fp:
            fitted_model = pickle.load(fp)
        fitted_model.fit(X_train, y_train)
    else:
        # We will run the experiment and select the best model.
        X_train[target_column_name] = y_train
        automl_config = AutoMLConfig(training_data=X_train, **automl_settings)
        automl_run = current_step_run.submit_child(automl_config, show_output=True)
        best_run, fitted_model = automl_run.get_output()
        # As we have generated models, we need to register them for the future use.
        description = "Backtest model example"
        tags = {"last_training_date": last_training_date, "experiment": experiment.name}
        if model_uid:
            tags["model_uid"] = model_uid
        automl_run.register_model(
            model_name=best_run.properties["model_name"],
            description=description,
            tags=tags,
        )
        print(f"The model {best_run.properties['model_name']} was registered.")

    # By default we will have forecast quantiles of 0.5, which is our target
    if forecast_quantiles:
        if 0.5 not in forecast_quantiles:
            forecast_quantiles.append(0.5)
        fitted_model.quantiles = forecast_quantiles

    x_pred = fitted_model.forecast_quantiles(X_test)
    x_pred["actual_level"] = y_test
    x_pred["backtest_iteration"] = f"iteration_{last_training_date}"
    x_pred.rename({0.5: "predicted_level"}, axis=1, inplace=True)
    date_safe = RE_INVALID_SYMBOLS.sub("_", last_training_date)

    x_pred.to_csv(os.path.join(output_dir, f"iteration_{date_safe}.csv"), index=False)
    return x_pred


def run(input_files):
    """Run the script"""
    logger.info("Running mini batch.")
    ws = get_run().experiment.workspace
    file_name = None
    if model_name:
        models = Model.list(ws, name=model_name)
        cloud_model = None
        if models:
            for one_mod in models:
                if cloud_model is None or one_mod.version > cloud_model.version:
                    logger.info(
                        "Using existing model from the workspace. Model version: {}".format(
                            one_mod.version
                        )
                    )
                    cloud_model = one_mod
        file_name = cloud_model.download(exist_ok=True)

    forecasts = []
    logger.info("Running backtest.")
    for input_file in input_files:
        forecasts.append(run_backtest(input_file, file_name, get_run().experiment))
    return pd.concat(forecasts)
