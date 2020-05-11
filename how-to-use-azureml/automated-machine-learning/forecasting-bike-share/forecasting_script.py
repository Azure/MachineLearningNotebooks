import argparse
import azureml.train.automl
from azureml.automl.runtime.shared import forecasting_models
from azureml.core import Run
from sklearn.externals import joblib
import forecasting_helper


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

args = parser.parse_args()
max_horizon = args.max_horizon
target_column_name = args.target_column_name
time_column_name = args.time_column_name
freq = args.freq

run = Run.get_context()
# get input dataset by name
test_dataset = run.input_datasets['test_data']

grain_column_names = []

df = test_dataset.to_pandas_dataframe().reset_index(drop=True)

X_test_df = test_dataset.drop_columns(columns=[target_column_name]).to_pandas_dataframe().reset_index(drop=True)
y_test_df = test_dataset.with_timestamp_columns(None).keep_columns(columns=[target_column_name]).to_pandas_dataframe()

fitted_model = joblib.load('model.pkl')

df_all = forecasting_helper.do_rolling_forecast(
    fitted_model,
    X_test_df,
    y_test_df.values.T[0],
    target_column_name,
    time_column_name,
    max_horizon,
    freq)

file_name = 'outputs/predictions.csv'
export_csv = df_all.to_csv(file_name, header=True)

# Upload the predictions into artifacts
run.upload_file(name=file_name, path_or_stream=file_name)
