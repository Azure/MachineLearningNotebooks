import argparse
from azureml.core import Dataset, Run
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

X_test_df = (
    test_dataset.drop_columns(columns=[target_column_name])
    .to_pandas_dataframe()
    .reset_index(drop=True)
)
y_test_df = (
    test_dataset.with_timestamp_columns(None)
    .keep_columns(columns=[target_column_name])
    .to_pandas_dataframe()
)

fitted_model = joblib.load("model.pkl")

y_pred, X_trans = fitted_model.rolling_evaluation(X_test_df, y_test_df.values)

# Add predictions, actuals, and horizon relative to rolling origin to the test feature data
assign_dict = {
    "horizon_origin": X_trans["horizon_origin"].values,
    "predicted": y_pred,
    target_column_name: y_test_df[target_column_name].values,
}
df_all = X_test_df.assign(**assign_dict)

file_name = "outputs/predictions.csv"
export_csv = df_all.to_csv(file_name, header=True)

# Upload the predictions into artifacts
run.upload_file(name=file_name, path_or_stream=file_name)
