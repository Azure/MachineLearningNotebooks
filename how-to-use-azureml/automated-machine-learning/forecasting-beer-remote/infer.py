import argparse
import os

import numpy as np
import pandas as pd

from pandas.tseries.frequencies import to_offset
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

from azureml.automl.runtime.shared.score import scoring, constants
from azureml.core import Run

try:
    import torch

    _torch_present = True
except ImportError:
    _torch_present = False


def align_outputs(y_predicted, X_trans, X_test, y_test,
                  predicted_column_name='predicted',
                  horizon_colname='horizon_origin'):
    """
    Demonstrates how to get the output aligned to the inputs
    using pandas indexes. Helps understand what happened if
    the output's shape differs from the input shape, or if
    the data got re-sorted by time and grain during forecasting.

    Typical causes of misalignment are:
    * we predicted some periods that were missing in actuals -> drop from eval
    * model was asked to predict past max_horizon -> increase max horizon
    * data at start of X_test was needed for lags -> provide previous periods
    """
    if (horizon_colname in X_trans):
        df_fcst = pd.DataFrame({predicted_column_name: y_predicted,
                                horizon_colname: X_trans[horizon_colname]})
    else:
        df_fcst = pd.DataFrame({predicted_column_name: y_predicted})

    # y and X outputs are aligned by forecast() function contract
    df_fcst.index = X_trans.index

    # align original X_test to y_test
    X_test_full = X_test.copy()
    X_test_full[target_column_name] = y_test

    # X_test_full's index does not include origin, so reset for merge
    df_fcst.reset_index(inplace=True)
    X_test_full = X_test_full.reset_index().drop(columns='index')
    together = df_fcst.merge(X_test_full, how='right')

    # drop rows where prediction or actuals are nan
    # happens because of missing actuals
    # or at edges of time due to lags/rolling windows
    clean = together[together[[target_column_name,
                               predicted_column_name]].notnull().all(axis=1)]
    return (clean)


def do_rolling_forecast_with_lookback(fitted_model, X_test, y_test,
                                      max_horizon, X_lookback, y_lookback,
                                      freq='D'):
    """
    Produce forecasts on a rolling origin over the given test set.

    Each iteration makes a forecast for the next 'max_horizon' periods
    with respect to the current origin, then advances the origin by the
    horizon time duration. The prediction context for each forecast is set so
    that the forecaster uses the actual target values prior to the current
    origin time for constructing lag features.

    This function returns a concatenated DataFrame of rolling forecasts.
     """
    print("Using lookback of size: ", y_lookback.size)
    df_list = []
    origin_time = X_test[time_column_name].min()
    X = X_lookback.append(X_test)
    y = np.concatenate((y_lookback, y_test), axis=0)
    while origin_time <= X_test[time_column_name].max():
        # Set the horizon time - end date of the forecast
        horizon_time = origin_time + max_horizon * to_offset(freq)

        # Extract test data from an expanding window up-to the horizon
        expand_wind = (X[time_column_name] < horizon_time)
        X_test_expand = X[expand_wind]
        y_query_expand = np.zeros(len(X_test_expand)).astype(np.float)
        y_query_expand.fill(np.NaN)

        if origin_time != X[time_column_name].min():
            # Set the context by including actuals up-to the origin time
            test_context_expand_wind = (X[time_column_name] < origin_time)
            context_expand_wind = (X_test_expand[time_column_name] < origin_time)
            y_query_expand[context_expand_wind] = y[test_context_expand_wind]

        # Print some debug info
        print("Horizon_time:", horizon_time,
              " origin_time: ", origin_time,
              " max_horizon: ", max_horizon,
              " freq: ", freq)
        print("expand_wind: ", expand_wind)
        print("y_query_expand")
        print(y_query_expand)
        print("X_test")
        print(X)
        print("X_test_expand")
        print(X_test_expand)
        print("Type of X_test_expand: ", type(X_test_expand))
        print("Type of y_query_expand: ", type(y_query_expand))

        print("y_query_expand")
        print(y_query_expand)

        # Make a forecast out to the maximum horizon
        # y_fcst, X_trans = y_query_expand, X_test_expand
        y_fcst, X_trans = fitted_model.forecast(X_test_expand, y_query_expand)

        print("y_fcst")
        print(y_fcst)

        # Align forecast with test set for dates within
        # the current rolling window
        trans_tindex = X_trans.index.get_level_values(time_column_name)
        trans_roll_wind = (trans_tindex >= origin_time) & (trans_tindex < horizon_time)
        test_roll_wind = expand_wind & (X[time_column_name] >= origin_time)
        df_list.append(align_outputs(
            y_fcst[trans_roll_wind], X_trans[trans_roll_wind],
            X[test_roll_wind], y[test_roll_wind]))

        # Advance the origin time
        origin_time = horizon_time

    return pd.concat(df_list, ignore_index=True)


def do_rolling_forecast(fitted_model, X_test, y_test, max_horizon, freq='D'):
    """
    Produce forecasts on a rolling origin over the given test set.

    Each iteration makes a forecast for the next 'max_horizon' periods
    with respect to the current origin, then advances the origin by the
    horizon time duration. The prediction context for each forecast is set so
    that the forecaster uses the actual target values prior to the current
    origin time for constructing lag features.

    This function returns a concatenated DataFrame of rolling forecasts.
     """
    df_list = []
    origin_time = X_test[time_column_name].min()
    while origin_time <= X_test[time_column_name].max():
        # Set the horizon time - end date of the forecast
        horizon_time = origin_time + max_horizon * to_offset(freq)

        # Extract test data from an expanding window up-to the horizon
        expand_wind = (X_test[time_column_name] < horizon_time)
        X_test_expand = X_test[expand_wind]
        y_query_expand = np.zeros(len(X_test_expand)).astype(np.float)
        y_query_expand.fill(np.NaN)

        if origin_time != X_test[time_column_name].min():
            # Set the context by including actuals up-to the origin time
            test_context_expand_wind = (X_test[time_column_name] < origin_time)
            context_expand_wind = (X_test_expand[time_column_name] < origin_time)
            y_query_expand[context_expand_wind] = y_test[
                test_context_expand_wind]

        # Print some debug info
        print("Horizon_time:", horizon_time,
              " origin_time: ", origin_time,
              " max_horizon: ", max_horizon,
              " freq: ", freq)
        print("expand_wind: ", expand_wind)
        print("y_query_expand")
        print(y_query_expand)
        print("X_test")
        print(X_test)
        print("X_test_expand")
        print(X_test_expand)
        print("Type of X_test_expand: ", type(X_test_expand))
        print("Type of y_query_expand: ", type(y_query_expand))
        print("y_query_expand")
        print(y_query_expand)

        # Make a forecast out to the maximum horizon
        y_fcst, X_trans = fitted_model.forecast(X_test_expand, y_query_expand)

        print("y_fcst")
        print(y_fcst)

        # Align forecast with test set for dates within the
        # current rolling window
        trans_tindex = X_trans.index.get_level_values(time_column_name)
        trans_roll_wind = (trans_tindex >= origin_time) & (trans_tindex < horizon_time)
        test_roll_wind = expand_wind & (X_test[time_column_name] >= origin_time)
        df_list.append(align_outputs(y_fcst[trans_roll_wind],
                                     X_trans[trans_roll_wind],
                                     X_test[test_roll_wind],
                                     y_test[test_roll_wind]))

        # Advance the origin time
        origin_time = horizon_time

    return pd.concat(df_list, ignore_index=True)


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


def map_location_cuda(storage, loc):
    return storage.cuda()


parser = argparse.ArgumentParser()
parser.add_argument(
    '--max_horizon', type=int, dest='max_horizon',
    default=10, help='Max Horizon for forecasting')
parser.add_argument(
    '--target_column_name', type=str, dest='target_column_name',
    help='Target Column Name')
parser.add_argument(
    '--time_column_name', type=str, dest='time_column_name',
    help='Time Column Name')
parser.add_argument(
    '--frequency', type=str, dest='freq',
    help='Frequency of prediction')
parser.add_argument(
    '--model_path', type=str, dest='model_path',
    default='model.pkl', help='Filename of model to be loaded')

args = parser.parse_args()
max_horizon = args.max_horizon
target_column_name = args.target_column_name
time_column_name = args.time_column_name
freq = args.freq
model_path = args.model_path

print('args passed are: ')
print(max_horizon)
print(target_column_name)
print(time_column_name)
print(freq)
print(model_path)

run = Run.get_context()
# get input dataset by name
test_dataset = run.input_datasets['test_data']
lookback_dataset = run.input_datasets['lookback_data']

grain_column_names = []

df = test_dataset.to_pandas_dataframe()

print('Read df')
print(df)

X_test_df = test_dataset.drop_columns(columns=[target_column_name])
y_test_df = test_dataset.with_timestamp_columns(
    None).keep_columns(columns=[target_column_name])

X_lookback_df = lookback_dataset.drop_columns(columns=[target_column_name])
y_lookback_df = lookback_dataset.with_timestamp_columns(
    None).keep_columns(columns=[target_column_name])

_, ext = os.path.splitext(model_path)
if ext == '.pt':
    # Load the fc-tcn torch model.
    assert _torch_present
    if torch.cuda.is_available():
        map_location = map_location_cuda
    else:
        map_location = 'cpu'
    with open(model_path, 'rb') as fh:
        fitted_model = torch.load(fh, map_location=map_location)
else:
    # Load the sklearn pipeline.
    fitted_model = joblib.load(model_path)

if hasattr(fitted_model, 'get_lookback'):
    lookback = fitted_model.get_lookback()
    df_all = do_rolling_forecast_with_lookback(
        fitted_model,
        X_test_df.to_pandas_dataframe(),
        y_test_df.to_pandas_dataframe().values.T[0],
        max_horizon,
        X_lookback_df.to_pandas_dataframe()[-lookback:],
        y_lookback_df.to_pandas_dataframe().values.T[0][-lookback:],
        freq)
else:
    df_all = do_rolling_forecast(
        fitted_model,
        X_test_df.to_pandas_dataframe(),
        y_test_df.to_pandas_dataframe().values.T[0],
        max_horizon,
        freq)

print(df_all)

print("target values:::")
print(df_all[target_column_name])
print("predicted values:::")
print(df_all['predicted'])

# Use the AutoML scoring module
regression_metrics = list(constants.REGRESSION_SCALAR_SET)
y_test = np.array(df_all[target_column_name])
y_pred = np.array(df_all['predicted'])
scores = scoring.score_regression(y_test, y_pred, regression_metrics)

print("scores:")
print(scores)

for key, value in scores.items():
    run.log(key, value)

print("Simple forecasting model")
rmse = np.sqrt(mean_squared_error(
    df_all[target_column_name], df_all['predicted']))
print("[Test Data] \nRoot Mean squared error: %.2f" % rmse)
mae = mean_absolute_error(df_all[target_column_name], df_all['predicted'])
print('mean_absolute_error score: %.2f' % mae)
print('MAPE: %.2f' % MAPE(df_all[target_column_name], df_all['predicted']))

run.log('rmse', rmse)
run.log('mae', mae)
