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

dataset_name = 'clean_Titanic_tutorial'

dataset = Dataset.get(workspace=workspace, name=dataset_name)
df = dataset.to_pandas_dataframe()

x_col = ['Pclass', 'Sex', 'SibSp', 'Parch']
y_col = ['Survived']
x_df = df.loc[:, x_col]
y_df = df.loc[:, y_col]

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=223)

data = {"train": {"X": x_train, "y": y_train},

        "test": {"X": x_test, "y": y_test}}

clf = DecisionTreeClassifier().fit(data["train"]["X"], data["train"]["y"])

print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(x_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(x_test, y_test)))
