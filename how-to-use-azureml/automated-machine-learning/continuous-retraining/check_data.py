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

print("Check for new data.")

parser = argparse.ArgumentParser("split")
parser.add_argument("--ds_name", help="input dataset name")
parser.add_argument("--model_name", help="name of the deployed model")

args = parser.parse_args()

print("Argument 1(ds_name): %s" % args.ds_name)
print("Argument 2(model_name): %s" % args.model_name)

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

if not dataset_changed_time > last_train_time:
    print("Cancelling run since there is no new data.")
    run.parent.cancel()
else:
    # New data is available since the model was last trained
    print("Dataset was last updated on {0}. Retraining...".format(dataset_changed_time))
