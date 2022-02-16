import argparse

import pandas as pd
import numpy as np

from sklearn.externals import joblib

from azureml.automl.runtime.shared.score import scoring, constants
from azureml.core import Run
from azureml.core.model import Model


parser = argparse.ArgumentParser()
parser.add_argument(
    "--target_column_name",
    type=str,
    dest="target_column_name",
    help="Target Column Name",
)
parser.add_argument(
    "--model_name", type=str, dest="model_name", help="Name of registered model"
)

args = parser.parse_args()
target_column_name = args.target_column_name
model_name = args.model_name

print("args passed are: ")
print("Target column name: ", target_column_name)
print("Name of registered model: ", model_name)

model_path = Model.get_model_path(model_name)
# deserialize the model file back into a sklearn model
model = joblib.load(model_path)

run = Run.get_context()
# get input dataset by name
test_dataset = run.input_datasets["test_data"]

X_test_df = test_dataset.drop_columns(
    columns=[target_column_name]
).to_pandas_dataframe()
y_test_df = (
    test_dataset.with_timestamp_columns(None)
    .keep_columns(columns=[target_column_name])
    .to_pandas_dataframe()
)

predicted = model.predict_proba(X_test_df)

if isinstance(predicted, pd.DataFrame):
    predicted = predicted.values

# Use the AutoML scoring module
train_labels = model.classes_
class_labels = np.unique(
    np.concatenate((y_test_df.values, np.reshape(train_labels, (-1, 1))))
)
classification_metrics = list(constants.CLASSIFICATION_SCALAR_SET)
scores = scoring.score_classification(
    y_test_df.values, predicted, classification_metrics, class_labels, train_labels
)

print("scores:")
print(scores)

for key, value in scores.items():
    run.log(key, value)
