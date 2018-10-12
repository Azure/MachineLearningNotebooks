# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import argparse

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
from sklearn.externals import joblib

import numpy as np

os.makedirs('./outputs', exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str,
                    dest='data_folder', help='data folder')
args = parser.parse_args()

print('Data folder is at:', args.data_folder)
print('List all files: ', os.listdir(args.data_folder))

X = np.load(os.path.join(args.data_folder, 'features.npy'))
y = np.load(os.path.join(args.data_folder, 'labels.npy'))

run = Run.get_context()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
data = {"train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}}

# list of numbers from 0.0 to 1.0 with a 0.05 interval
alphas = np.arange(0.0, 1.0, 0.05)

for alpha in alphas:
    # Use Ridge algorithm to create a regression model
    reg = Ridge(alpha=alpha)
    reg.fit(data["train"]["X"], data["train"]["y"])

    preds = reg.predict(data["test"]["X"])
    mse = mean_squared_error(preds, data["test"]["y"])
    run.log('alpha', alpha)
    run.log('mse', mse)

    model_file_name = 'ridge_{0:.2f}.pkl'.format(alpha)
    with open(model_file_name, "wb") as file:
        joblib.dump(value=reg, filename='outputs/' + model_file_name)

    print('alpha is {0:.2f}, and mse is {1:0.2f}'.format(alpha, mse))
