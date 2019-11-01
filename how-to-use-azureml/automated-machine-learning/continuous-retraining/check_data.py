import argparse
import os
import azureml.core
from datetime import datetime
import pandas as pd
import pytz
from azureml.core import Dataset, Model
from azureml.core.run import Run, _OfflineRun
from azureml.core import Workspace

run = Run.get_context()
ws = None
if type(run) == _OfflineRun:
    ws = Workspace.from_config()
else:
    ws = run.experiment.workspace


def write_output(df, path):
    os.makedirs(path, exist_ok=True)
    print("%s created" % path)
    df.to_csv(path + "/part-00000", index=False)


print("Check for new data and prepare the data")

parser = argparse.ArgumentParser("split")
parser.add_argument("--target_column", type=str, help="input split features")
parser.add_argument("--ds_name", help="input dataset name")
parser.add_argument("--model_name", help="name of the deployed model")
parser.add_argument("--output_x", type=str,
                    help="output features")
parser.add_argument("--output_y", type=str,
                    help="output labels")


args = parser.parse_args()

print("Argument 1(ds_name): %s" % args.ds_name)
print("Argument 2(target_column): %s" % args.target_column)
print("Argument 3(model_name): %s" % args.model_name)
print("Argument 4(output_x): %s" % args.output_x)
print("Argument 5(output_y): %s" % args.output_y)

# Get the latest registered model
try:
    model = Model(ws, args.model_name)
    last_train_time = model.created_time
    print("Model was last trained on {0}.".format(last_train_time))
except Exception as e:
    print("Could not get last model train time.")
    last_train_time = datetime.min.replace(tzinfo=pytz.UTC)

train_ds = Dataset.get_by_name(ws, args.ds_name)
dataset_changed_time = train_ds.data_changed_time

if dataset_changed_time > last_train_time:
    # New data is available since the model was last trained
    print("Dataset was last updated on {0}. Retraining...".format(dataset_changed_time))
    train_ds = train_ds.drop_columns(["partition_date"])
    X_train = train_ds.drop_columns(
        columns=[args.target_column]).to_pandas_dataframe()
    y_train = train_ds.keep_columns(
        columns=[args.target_column]).to_pandas_dataframe()

    non_null = y_train[args.target_column].notnull()
    y = y_train[non_null]
    X = X_train[non_null]

    if not (args.output_x is None and args.output_y is None):
        write_output(X, args.output_x)
        write_output(y, args.output_y)
else:
    print("Cancelling run since there is no new data.")
    run.parent.cancel()
