"""
This is the script that is executed on the compute instance. It relies
on the model.pkl file which is uploaded along with this script to the
compute instance.
"""

import argparse
import pandas as pd
import numpy as np
from azureml.core import Dataset, Run
from azureml.automl.core.shared.constants import TimeSeriesInternal
from sklearn.externals import joblib
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


parser = argparse.ArgumentParser()
parser.add_argument(
    '--target_column_name', type=str, dest='target_column_name',
    help='Target Column Name')
parser.add_argument(
    '--test_dataset', type=str, dest='test_dataset',
    help='Test Dataset')

args = parser.parse_args()
target_column_name = args.target_column_name
test_dataset_id = args.test_dataset

run = Run.get_context()
ws = run.experiment.workspace

# get the input dataset by id
test_dataset = Dataset.get_by_id(ws, id=test_dataset_id)

X_test = test_dataset.to_pandas_dataframe().reset_index(drop=True)
y_test = X_test.pop(target_column_name).values

# generate forecast
fitted_model = joblib.load('model.pkl')
y_predictions, X_trans = fitted_model.forecast(X_test)

# align output
df_all = align_outputs(y_predictions, X_trans, X_test, y_test, target_column_name)

file_name = 'outputs/predictions.csv'
export_csv = df_all.to_csv(file_name, header=True, index=False)  # added Index

# Upload the predictions into artifacts
run.upload_file(name=file_name, path_or_stream=file_name)
