import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset


def align_outputs(y_predicted, X_trans, X_test, y_test, target_column_name,
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
    return(clean)


def do_rolling_forecast(fitted_model, X_test, y_test, target_column_name, time_column_name, max_horizon, freq='D'):
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
            context_expand_wind = (
                X_test_expand[time_column_name] < origin_time)
            y_query_expand[context_expand_wind] = y_test[
                test_context_expand_wind]

        # Make a forecast out to the maximum horizon
        y_fcst, X_trans = fitted_model.forecast(X_test_expand, y_query_expand)

        # Align forecast with test set for dates within the
        # current rolling window
        trans_tindex = X_trans.index.get_level_values(time_column_name)
        trans_roll_wind = (trans_tindex >= origin_time) & (
            trans_tindex < horizon_time)
        test_roll_wind = expand_wind & (
            X_test[time_column_name] >= origin_time)
        df_list.append(align_outputs(y_fcst[trans_roll_wind],
                                     X_trans[trans_roll_wind],
                                     X_test[test_roll_wind],
                                     y_test[test_roll_wind],
                                     target_column_name))

        # Advance the origin time
        origin_time = horizon_time

    return pd.concat(df_list, ignore_index=True)
