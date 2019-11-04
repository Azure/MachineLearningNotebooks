import os
import pandas as pd


def get_data():
    print("In get_data")
    print(os.environ['AZUREML_DATAREFERENCE_output_x'])
    X_train = pd.read_csv(
        os.environ['AZUREML_DATAREFERENCE_output_x'] + "/part-00000")
    y_train = pd.read_csv(
        os.environ['AZUREML_DATAREFERENCE_output_y'] + "/part-00000")

    print(X_train.head(3))

    return {"X": X_train.values, "y": y_train.values.flatten()}
