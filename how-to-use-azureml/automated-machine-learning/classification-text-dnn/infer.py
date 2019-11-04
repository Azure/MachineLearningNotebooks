import numpy as np
import argparse
from azureml.core import Run
from sklearn.externals import joblib
from azureml.automl.core._vendor.automl.client.core.common import metrics
from automl.client.core.common import constants
from azureml.core.model import Model


parser = argparse.ArgumentParser()
parser.add_argument(
    '--target_column_name', type=str, dest='target_column_name',
    help='Target Column Name')
parser.add_argument(
    '--model_name', type=str, dest='model_name',
    help='Name of registered model')

args = parser.parse_args()
target_column_name = args.target_column_name
model_name = args.model_name

print('args passed are: ')
print('Target column name: ', target_column_name)
print('Name of registered model: ', model_name)

model_path = Model.get_model_path(model_name)
# deserialize the model file back into a sklearn model
model = joblib.load(model_path)

run = Run.get_context()
# get input dataset by name
test_dataset = run.input_datasets['test_data']

X_test_df = test_dataset.drop_columns(columns=[target_column_name]) \
                        .to_pandas_dataframe()
y_test_df = test_dataset.with_timestamp_columns(None) \
                        .keep_columns(columns=[target_column_name]) \
                        .to_pandas_dataframe()

predicted = model.predict_proba(X_test_df)

# use automl metrics module
scores = metrics.compute_metrics_classification(
    np.array(predicted),
    np.array(y_test_df),
    class_labels=model.classes_,
    metrics=list(constants.Metric.SCALAR_CLASSIFICATION_SET)
)

print("scores:")
print(scores)

for key, value in scores.items():
    run.log(key, value)
