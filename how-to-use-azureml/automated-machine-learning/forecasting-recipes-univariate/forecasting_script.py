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
    "--target_column_name",
    type=str,
    dest="target_column_name",
    help="Target Column Name",
)
parser.add_argument(
    "--test_dataset", type=str, dest="test_dataset", help="Test Dataset"
)

args = parser.parse_args()
target_column_name = args.target_column_name
test_dataset_id = args.test_dataset

run = Run.get_context()
ws = run.experiment.workspace

# get the input dataset by id
test_dataset = Dataset.get_by_id(ws, id=test_dataset_id)

X_test = (
    test_dataset.drop_columns(columns=[target_column_name])
    .to_pandas_dataframe()
    .reset_index(drop=True)
)
y_test_df = (
    test_dataset.with_timestamp_columns(None)
    .keep_columns(columns=[target_column_name])
    .to_pandas_dataframe()
)

# generate forecast
fitted_model = joblib.load("model.pkl")
# We have default quantiles values set as below(95th percentile)
quantiles = [0.025, 0.5, 0.975]
predicted_column_name = "predicted"
PI = "prediction_interval"
fitted_model.quantiles = quantiles
pred_quantiles = fitted_model.forecast_quantiles(X_test)
pred_quantiles[PI] = pred_quantiles[[min(quantiles), max(quantiles)]].apply(
    lambda x: "[{}, {}]".format(x[0], x[1]), axis=1
)
X_test[target_column_name] = y_test_df[target_column_name]
X_test[PI] = pred_quantiles[PI]
X_test[predicted_column_name] = pred_quantiles[0.5]
# drop rows where prediction or actuals are nan
# happens because of missing actuals
# or at edges of time due to lags/rolling windows
clean = X_test[
    X_test[[target_column_name, predicted_column_name]].notnull().all(axis=1)
]
clean.rename(columns={target_column_name: "actual"}, inplace=True)

file_name = "outputs/predictions.csv"
export_csv = clean.to_csv(file_name, header=True, index=False)  # added Index

# Upload the predictions into artifacts
run.upload_file(name=file_name, path_or_stream=file_name)
