import azureml.dataprep as dprep
import azureml.core
import pandas as pd
import logging
import os
import datetime
import shutil

from azureml.core import Workspace, Datastore, Dataset, Experiment, Run
from sklearn.model_selection import train_test_split
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from sklearn.tree import DecisionTreeClassifier

run = Run.get_context()
workspace = run.experiment.workspace

dataset_name = 'training_data'

dataset = Dataset.get(workspace=workspace, name=dataset_name)
dflow = dataset.get_definition()
dflow_val, dflow_train = dflow.random_split(percentage=0.3)

y_df = dflow_train.keep_columns(['HasDetections']).to_pandas_dataframe()
x_df = dflow_train.drop_columns(['HasDetections']).to_pandas_dataframe()
y_val = dflow_val.keep_columns(['HasDetections']).to_pandas_dataframe()
x_val = dflow_val.drop_columns(['HasDetections']).to_pandas_dataframe()

data = {"train": {"X": x_df, "y": y_df},

        "validation": {"X": x_val, "y": y_val}}

clf = DecisionTreeClassifier().fit(data["train"]["X"], data["train"]["y"])

print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(x_df, y_df)))
print('Accuracy of Decision Tree classifier on validation set: {:.2f}'.format(clf.score(x_val, y_val)))
