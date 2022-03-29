import argparse
import os

import pandas as pd

import azureml.train.automl.runtime._hts.hts_runtime_utilities as hru

from azureml.core import Run
from azureml.core.dataset import Dataset

# Parse the arguments.
args = {
    "step_size": "--step-size",
    "step_number": "--step-number",
    "time_column_name": "--time-column-name",
    "time_series_id_column_names": "--time-series-id-column-names",
    "out_dir": "--output-dir",
}
parser = argparse.ArgumentParser("Parsing input arguments.")
for argname, arg in args.items():
    parser.add_argument(arg, dest=argname, required=True)
parsed_args, _ = parser.parse_known_args()
step_number = int(parsed_args.step_number)
step_size = int(parsed_args.step_size)
# Create the working dirrectory to store the temporary csv files.
working_dir = parsed_args.out_dir
os.makedirs(working_dir, exist_ok=True)
# Set input and output
script_run = Run.get_context()
input_dataset = script_run.input_datasets["training_data"]
X_train = input_dataset.to_pandas_dataframe()
# Split the data.
for i in range(step_number):
    file_name = os.path.join(working_dir, "backtest_{}.csv".format(i))
    if parsed_args.time_series_id_column_names:
        dfs = []
        for _, one_series in X_train.groupby([parsed_args.time_series_id_column_names]):
            one_series = one_series.sort_values(
                by=[parsed_args.time_column_name], inplace=False
            )
            dfs.append(one_series.iloc[: len(one_series) - step_size * i])
        pd.concat(dfs, sort=False, ignore_index=True).to_csv(file_name, index=False)
    else:
        X_train.sort_values(by=[parsed_args.time_column_name], inplace=True)
        X_train.iloc[: len(X_train) - step_size * i].to_csv(file_name, index=False)
