import argparse
import os

import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

from azureml.automl.runtime.shared.score import scoring, constants
from azureml.core import Run

try:
    import torch

    _torch_present = True
except ImportError:
    _torch_present = False


def map_location_cuda(storage, loc):
    return storage.cuda()


def APE(actual, pred):
    """
    Calculate absolute percentage error.
    Returns a vector of APE values with same length as actual/pred.
    """
    return 100 * np.abs((actual - pred) / actual)


def MAPE(actual, pred):
    """
    Calculate mean absolute percentage error.
    Remove NA and values where actual is close to zero
    """
    not_na = ~(np.isnan(actual) | np.isnan(pred))
    not_zero = ~np.isclose(actual, 0.0)
    actual_safe = actual[not_na & not_zero]
    pred_safe = pred[not_na & not_zero]
    return np.mean(APE(actual_safe, pred_safe))


parser = argparse.ArgumentParser()
parser.add_argument(
    "--max_horizon",
    type=int,
    dest="max_horizon",
    default=10,
    help="Max Horizon for forecasting",
)
parser.add_argument(
    "--target_column_name",
    type=str,
    dest="target_column_name",
    help="Target Column Name",
)
parser.add_argument(
    "--time_column_name", type=str, dest="time_column_name", help="Time Column Name"
)
parser.add_argument(
    "--frequency", type=str, dest="freq", help="Frequency of prediction"
)
parser.add_argument(
    "--model_path",
    type=str,
    dest="model_path",
    default="model.pkl",
    help="Filename of model to be loaded",
)

args = parser.parse_args()
max_horizon = args.max_horizon
target_column_name = args.target_column_name
time_column_name = args.time_column_name
freq = args.freq
model_path = args.model_path

print("args passed are: ")
print(max_horizon)
print(target_column_name)
print(time_column_name)
print(freq)
print(model_path)

run = Run.get_context()
# get input dataset by name
test_dataset = run.input_datasets["test_data"]

grain_column_names = []

df = test_dataset.to_pandas_dataframe()

print("Read df")
print(df)

X_test_df = df
y_test = df.pop(target_column_name).to_numpy()

_, ext = os.path.splitext(model_path)
if ext == ".pt":
    # Load the fc-tcn torch model.
    assert _torch_present
    if torch.cuda.is_available():
        map_location = map_location_cuda
    else:
        map_location = "cpu"
    with open(model_path, "rb") as fh:
        fitted_model = torch.load(fh, map_location=map_location)
else:
    # Load the sklearn pipeline.
    fitted_model = joblib.load(model_path)

X_rf = fitted_model.rolling_forecast(X_test_df, y_test, step=1)
assign_dict = {
    fitted_model.forecast_origin_column_name: "forecast_origin",
    fitted_model.forecast_column_name: "predicted",
    fitted_model.actual_column_name: target_column_name,
}
X_rf.rename(columns=assign_dict, inplace=True)

print(X_rf.head())

# Use the AutoML scoring module
regression_metrics = list(constants.REGRESSION_SCALAR_SET)
y_test = np.array(X_rf[target_column_name])
y_pred = np.array(X_rf["predicted"])
scores = scoring.score_regression(y_test, y_pred, regression_metrics)

print("scores:")
print(scores)

for key, value in scores.items():
    run.log(key, value)

print("Simple forecasting model")
rmse = np.sqrt(mean_squared_error(X_rf[target_column_name], X_rf["predicted"]))
print("[Test Data] \nRoot Mean squared error: %.2f" % rmse)
mae = mean_absolute_error(X_rf[target_column_name], X_rf["predicted"])
print("mean_absolute_error score: %.2f" % mae)
print("MAPE: %.2f" % MAPE(X_rf[target_column_name], X_rf["predicted"]))

run.log("rmse", rmse)
run.log("mae", mae)
