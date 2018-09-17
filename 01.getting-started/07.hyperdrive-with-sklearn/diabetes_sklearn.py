from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import os
import argparse

# Import Run from azureml.core,
from azureml.core.run import Run

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, dest='alpha',
                    default=0.5, help='regularization strength')
args = parser.parse_args()

# Get handle of current run for logging and history purposes
run = Run.get_submitted_run()

X, y = load_diabetes(return_X_y=True)

columns = ['age', 'gender', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
data = {"train": {"x": x_train, "y": y_train},
        "test": {"x": x_test, "y": y_test}}

alpha = args.alpha
print('alpha value is:', alpha)

reg = Ridge(alpha=alpha)
reg.fit(data["train"]["x"], data["train"]["y"])

print('Ridget model fitted.')

preds = reg.predict(data["test"]["x"])
mse = mean_squared_error(preds, data["test"]["y"])

# Log metrics
run.log("alpha", alpha)
run.log("mse", mse)

os.makedirs('./outputs', exist_ok=True)
model_file_name = "model.pkl"

# Save model as part of the run history
with open(model_file_name, "wb") as file:
    joblib.dump(reg, 'outputs/' + model_file_name)

print('Mean Squared Error is:', mse)
