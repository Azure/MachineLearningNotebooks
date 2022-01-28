# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import argparse
import os
from azureml.core import Run

print("Cleans the input data")

# Get the input green_taxi_data. To learn more about how to access dataset in your script, please
# see https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-with-datasets.
run = Run.get_context()
raw_data = run.input_datasets["raw_data"]

parser = argparse.ArgumentParser("cleanse")
parser.add_argument("--output_cleanse", type=str, help="cleaned taxi data directory")
parser.add_argument("--useful_columns", type=str, help="useful columns to keep")
parser.add_argument("--columns", type=str, help="rename column pattern")

args = parser.parse_args()

print("Argument 1(columns to keep): %s" % str(args.useful_columns.strip("[]").split(r'\;')))
print("Argument 2(columns renaming mapping): %s" % str(args.columns.strip("{}").split(r'\;')))
print("Argument 3(output cleansed taxi data path): %s" % args.output_cleanse)

# These functions ensure that null data is removed from the dataset,
# which will help increase machine learning model accuracy.

useful_columns = eval(args.useful_columns.replace(';', ','))
columns = eval(args.columns.replace(';', ','))

new_df = (raw_data.to_pandas_dataframe()
          .dropna(how='all')
          .rename(columns=columns))[useful_columns]

new_df.reset_index(inplace=True, drop=True)

if not (args.output_cleanse is None):
    os.makedirs(args.output_cleanse, exist_ok=True)
    print("%s created" % args.output_cleanse)
    path = args.output_cleanse + "/processed.parquet"
    write_df = new_df.to_parquet(path)
