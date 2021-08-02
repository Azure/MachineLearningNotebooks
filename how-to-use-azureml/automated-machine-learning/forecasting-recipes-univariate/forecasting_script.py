"""
This is the script that is executed on the compute instance. It relies
on the model.pkl file which is uploaded along with this script to the
compute instance.
"""

import argparse
from azureml.core import Dataset, Run
from azureml.automl.core.shared.constants import TimeSeriesInternal
from sklearn.externals import joblib

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

X_test_df = test_dataset.drop_columns(columns=[target_column_name]).to_pandas_dataframe().reset_index(drop=True)
y_test_df = test_dataset.with_timestamp_columns(None).keep_columns(columns=[target_column_name]).to_pandas_dataframe()

# generate forecast
fitted_model = joblib.load('model.pkl')
y_pred, X_trans = fitted_model.forecast(X_test_df)

# rename target column
X_trans.reset_index(drop=False, inplace=True)
X_trans.rename(columns={TimeSeriesInternal.DUMMY_TARGET_COLUMN: 'predicted'}, inplace=True)
X_trans['actual'] = y_test_df[target_column_name].values

file_name = 'outputs/predictions.csv'
export_csv = X_trans.to_csv(file_name, header=True, index=False)  # added Index

# Upload the predictions into artifacts
run.upload_file(name=file_name, path_or_stream=file_name)
