
import os
import pandas as pd

def get_data():
    print("In get_data")
    dir = os.environ['AZUREML_DATAREFERENCE_output_split']
    print(dir)
    X_train = pd.read_csv(os.path.join(dir,'x_train.csv'), header=0)
    y_train = pd.read_csv(os.path.join(dir,'y_train.csv'), header=0)

    return {"X": X_train.values, "y": y_train.values.flatten()}
