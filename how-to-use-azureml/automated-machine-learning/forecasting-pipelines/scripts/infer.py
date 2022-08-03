import argparse
from datetime import datetime
import os
import uuid
import numpy as np
import pandas as pd

from pandas.tseries.frequencies import to_offset
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.automl.runtime.shared.score import scoring, constants as metrics_constants
import azureml.automl.core.shared.constants as constants
from azureml.core import Run, Dataset, Model

try:
    import torch

    _torch_present = True
except ImportError:
    _torch_present = False


def infer_forecasting_dataset_tcn(
    X_test, y_test, model, output_path, output_dataset_name="results"
):

    y_pred, df_all = model.forecast(X_test, y_test)

    run = Run.get_context()

    registered_train = TabularDatasetFactory.register_pandas_dataframe(
        df_all,
        target=(
            run.experiment.workspace.get_default_datastore(),
            datetime.now().strftime("%Y-%m-%d-") + str(uuid.uuid4())[:6],
        ),
        name=output_dataset_name,
    )
    df_all.to_csv(os.path.join(output_path, output_dataset_name + ".csv"), index=False)


def map_location_cuda(storage, loc):
    return storage.cuda()


def get_model(model_path, model_file_name):
    # _, ext = os.path.splitext(model_path)
    model_full_path = os.path.join(model_path, model_file_name)
    print(model_full_path)
    if model_file_name.endswith("pt"):
        # Load the fc-tcn torch model.
        assert _torch_present, "Loading DNN models needs torch to be presented."
        if torch.cuda.is_available():
            map_location = map_location_cuda
        else:
            map_location = "cpu"
        with open(model_full_path, "rb") as fh:
            fitted_model = torch.load(fh, map_location=map_location)
    else:
        # Load the sklearn pipeline.
        fitted_model = joblib.load(model_full_path)
    return fitted_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, dest="model_name", help="Model to be loaded"
    )

    parser.add_argument(
        "--ouput_dataset_name",
        type=str,
        dest="ouput_dataset_name",
        default="results",
        help="Dataset name of the final output",
    )
    parser.add_argument(
        "--target_column_name",
        type=str,
        dest="target_column_name",
        help="The target column name.",
    )
    parser.add_argument(
        "--test_dataset_name",
        type=str,
        dest="test_dataset_name",
        default="results",
        help="Dataset name of the final output",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        dest="output_path",
        default="results",
        help="The output path",
    )
    args = parser.parse_args()
    return args


def get_data(run, fitted_model, target_column_name, test_dataset_name):

    # get input dataset by name
    test_dataset = Dataset.get_by_name(run.experiment.workspace, test_dataset_name)
    test_df = test_dataset.to_pandas_dataframe()
    if target_column_name in test_df:
        y_test = test_df.pop(target_column_name).values
    else:
        y_test = np.full(test_df.shape[0], np.nan)

    return test_df, y_test


def get_model_filename(run, model_name, model_path):
    model = Model(run.experiment.workspace, model_name)
    if "model_file_name" in model.tags:
        return model.tags["model_file_name"]
    is_pkl = True
    if model.tags.get("algorithm") == "TCNForecaster" or os.path.exists(
        os.path.join(model_path, "model.pt")
    ):
        is_pkl = False
    return "model.pkl" if is_pkl else "model.pt"


if __name__ == "__main__":
    run = Run.get_context()

    args = get_args()
    model_name = args.model_name
    ouput_dataset_name = args.ouput_dataset_name
    test_dataset_name = args.test_dataset_name
    target_column_name = args.target_column_name
    print("args passed are: ")

    print(model_name)
    print(test_dataset_name)
    print(ouput_dataset_name)
    print(target_column_name)

    model_path = Model.get_model_path(model_name)
    model_file_name = get_model_filename(run, model_name, model_path)
    print(model_file_name)
    fitted_model = get_model(model_path, model_file_name)

    X_test_df, y_test = get_data(
        run, fitted_model, target_column_name, test_dataset_name
    )

    infer_forecasting_dataset_tcn(
        X_test_df, y_test, fitted_model, args.output_path, ouput_dataset_name
    )
