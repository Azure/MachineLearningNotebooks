from typing import Any, Dict, Optional, List

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from azureml.automl.core.shared import constants
from azureml.automl.core.shared.types import GrainType
from azureml.automl.runtime.shared.score import scoring

GRAIN = "time_series_id"
BACKTEST_ITER = "backtest_iteration"
ACTUALS = "actual_level"
PREDICTIONS = "predicted_level"
ALL_GRAINS = "all_sets"

FORECASTS_FILE = "forecast.csv"
SCORES_FILE = "scores.csv"
PLOTS_FILE = "plots_fcst_vs_actual.pdf"
RE_INVALID_SYMBOLS = re.compile("[: ]")


def _compute_metrics(df: pd.DataFrame, metrics: List[str]):
    """
    Compute metrics for one data frame.

    :param df: The data frame which contains actual_level and predicted_level columns.
    :return: The data frame with two columns - metric_name and metric.
    """
    scores = scoring.score_regression(
        y_test=df[ACTUALS], y_pred=df[PREDICTIONS], metrics=metrics
    )
    metrics_df = pd.DataFrame(list(scores.items()), columns=["metric_name", "metric"])
    metrics_df.sort_values(["metric_name"], inplace=True)
    metrics_df.reset_index(drop=True, inplace=True)
    return metrics_df


def _format_grain_name(grain: GrainType) -> str:
    """
    Convert grain name to string.

    :param grain: the grain name.
    :return: the string representation of the given grain.
    """
    if not isinstance(grain, tuple) and not isinstance(grain, list):
        return str(grain)
    grain = list(map(str, grain))
    return "|".join(grain)


def compute_all_metrics(
    fcst_df: pd.DataFrame,
    ts_id_colnames: List[str],
    metric_names: Optional[List[set]] = None,
):
    """
    Calculate metrics per grain.

    :param fcst_df: forecast data frame. Must contain 2 columns: 'actual_level' and 'predicted_level'
    :param metric_names: (optional) the list of metric names to return
    :param ts_id_colnames: (optional) list of grain column names
    :return: dictionary of summary table for all tests and final decision on stationary vs nonstaionary
    """
    if not metric_names:
        metric_names = list(constants.Metric.SCALAR_REGRESSION_SET)

    if ts_id_colnames is None:
        ts_id_colnames = []

    metrics_list = []
    if ts_id_colnames:
        for grain, df in fcst_df.groupby(ts_id_colnames):
            one_grain_metrics_df = _compute_metrics(df, metric_names)
            one_grain_metrics_df[GRAIN] = _format_grain_name(grain)
            metrics_list.append(one_grain_metrics_df)

    # overall metrics
    one_grain_metrics_df = _compute_metrics(fcst_df, metric_names)
    one_grain_metrics_df[GRAIN] = ALL_GRAINS
    metrics_list.append(one_grain_metrics_df)

    # collect into a data frame
    return pd.concat(metrics_list)


def _draw_one_plot(
    df: pd.DataFrame,
    time_column_name: str,
    grain_column_names: List[str],
    pdf: PdfPages,
) -> None:
    """
    Draw the single plot.

    :param df: The data frame with the data to build plot.
    :param time_column_name: The name of a time column.
    :param grain_column_names: The name of grain columns.
    :param pdf: The pdf backend used to render the plot.
    """
    fig, _ = plt.subplots(figsize=(20, 10))
    df = df.set_index(time_column_name)
    plt.plot(df[[ACTUALS, PREDICTIONS]])
    plt.xticks(rotation=45)
    iteration = df[BACKTEST_ITER].iloc[0]
    if grain_column_names:
        grain_name = [df[grain].iloc[0] for grain in grain_column_names]
        plt.title(f"Time series ID: {_format_grain_name(grain_name)} {iteration}")
    plt.legend(["actual", "forecast"])
    plt.close(fig)
    pdf.savefig(fig)


def calculate_scores_and_build_plots(
    input_dir: str, output_dir: str, automl_settings: Dict[str, Any]
):
    os.makedirs(output_dir, exist_ok=True)
    grains = automl_settings.get(
        constants.TimeSeries.TIME_SERIES_ID_COLUMN_NAMES,
        automl_settings.get(constants.TimeSeries.GRAIN_COLUMN_NAMES, None),
    )
    time_column_name = automl_settings.get(constants.TimeSeries.TIME_COLUMN_NAME)
    if grains is None:
        grains = []
    if isinstance(grains, str):
        grains = [grains]
    while BACKTEST_ITER in grains:
        grains.remove(BACKTEST_ITER)

    dfs = []
    for fle in os.listdir(input_dir):
        file_path = os.path.join(input_dir, fle)
        if os.path.isfile(file_path) and file_path.endswith(".csv"):
            df_iter = pd.read_csv(file_path, parse_dates=[time_column_name])
            for _, iteration in df_iter.groupby(BACKTEST_ITER):
                dfs.append(iteration)
    forecast_df = pd.concat(dfs, sort=False, ignore_index=True)
    # To make sure plots are in order, sort the predictions by grain and iteration.
    ts_index = grains + [BACKTEST_ITER]
    forecast_df.sort_values(by=ts_index, inplace=True)
    pdf = PdfPages(os.path.join(output_dir, PLOTS_FILE))
    for _, one_forecast in forecast_df.groupby(ts_index):
        _draw_one_plot(one_forecast, time_column_name, grains, pdf)
    pdf.close()
    forecast_df.to_csv(os.path.join(output_dir, FORECASTS_FILE), index=False)
    # Remove np.NaN and np.inf from the prediction and actuals data.
    forecast_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    forecast_df.dropna(subset=[ACTUALS, PREDICTIONS], inplace=True)
    metrics = compute_all_metrics(forecast_df, grains + [BACKTEST_ITER])
    metrics.to_csv(os.path.join(output_dir, SCORES_FILE), index=False)


if __name__ == "__main__":
    args = {"forecasts": "--forecasts", "scores_out": "--output-dir"}
    parser = argparse.ArgumentParser("Parsing input arguments.")
    for argname, arg in args.items():
        parser.add_argument(arg, dest=argname, required=True)
    parsed_args, _ = parser.parse_known_args()
    input_dir = parsed_args.forecasts
    output_dir = parsed_args.scores_out
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "automl_settings.json"
        )
    ) as json_file:
        automl_settings = json.load(json_file)
    calculate_scores_and_build_plots(input_dir, output_dir, automl_settings)
