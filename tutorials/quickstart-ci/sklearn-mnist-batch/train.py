import argparse
import os
import numpy as np
import glob

from sklearn.linear_model import LogisticRegression
import joblib

from azureml.core import Run
from utils import load_data

# let user feed in 2 parameters, the dataset to mount or download,
# and the regularization rate of the logistic regression model
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-folder", type=str, dest="data_folder", help="data folder mounting point"
)
parser.add_argument(
    "--regularization", type=float, dest="reg", default=0.01, help="regularization rate"
)
args = parser.parse_args()

data_folder = args.data_folder
print("Data folder:", data_folder)

# load train and test set into numpy arrays
# note we scale the pixel intensity values to 0-1 (by dividing it with 255.0) so the model can converge faster.
X_train = (
    load_data(
        glob.glob(
            os.path.join(data_folder, "**/train-images-idx3-ubyte.gz"), recursive=True
        )[0],
        False,
    ) /
    255.0
)
X_test = (
    load_data(
        glob.glob(
            os.path.join(data_folder, "**/t10k-images-idx3-ubyte.gz"), recursive=True
        )[0],
        False,
    ) /
    255.0
)
y_train = load_data(
    glob.glob(
        os.path.join(data_folder, "**/train-labels-idx1-ubyte.gz"), recursive=True
    )[0],
    True,
).reshape(-1)
y_test = load_data(
    glob.glob(
        os.path.join(data_folder, "**/t10k-labels-idx1-ubyte.gz"), recursive=True
    )[0],
    True,
).reshape(-1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep="\n")

# get hold of the current run
run = Run.get_context()

print("Train a logistic regression model with regularization rate of", args.reg)
clf = LogisticRegression(
    C=1.0 / args.reg, solver="liblinear", multi_class="auto", random_state=42
)
clf.fit(X_train, y_train)

print("Predict the test set")
y_hat = clf.predict(X_test)

# calculate accuracy on the prediction
acc = np.average(y_hat == y_test)
print("Accuracy is", acc)

run.log("regularization rate", np.float(args.reg))
run.log("accuracy", np.float(acc))

os.makedirs("outputs", exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=clf, filename="outputs/sklearn_mnist_model.pkl")
