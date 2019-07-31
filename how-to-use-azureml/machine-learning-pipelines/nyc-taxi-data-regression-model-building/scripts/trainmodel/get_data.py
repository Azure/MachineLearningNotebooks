
import os

import pandas as pd


def get_data():
    print("In get_data")
    print(os.environ['AZUREML_DATAREFERENCE_output_split_train_x'])
    file_name="file"
    X_train = pd.read_csv(os.path.join(os.environ['AZUREML_DATAREFERENCE_output_split_train_x'],file_name), header=0)
    y_train = pd.read_csv(os.path.join(os.environ['AZUREML_DATAREFERENCE_output_split_train_y'],file_name), header=0)

    return {"X": X_train.values, "y": y_train.values.flatten()}
