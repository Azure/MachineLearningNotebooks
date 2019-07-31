# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import argparse
import os
import pandas as pd
import azureml.dataprep as dprep


def get_dict(dict_str):
    pairs = dict_str.strip("{}").split("\;")
    new_dict = {}
    for pair in pairs:
        key, value = pair.strip('\\').split(":")
        new_dict[key.strip().strip("'")] = value.strip().strip("'")

    return new_dict


print("Cleans the input data")

parser = argparse.ArgumentParser("cleanse")
parser.add_argument("--input_cleanse", type=str, help="raw taxi data")
parser.add_argument("--output_cleanse", type=str, help="cleaned taxi data directory")
parser.add_argument("--useful_columns", type=str, help="useful columns to keep")
parser.add_argument("--columns", type=str, help="rename column pattern")

args = parser.parse_args()

print("Argument 1(input taxi data path): %s" % args.input_cleanse)
print("Argument 2(columns to keep): %s" % str(args.useful_columns.strip("[]").split("\;")))
print("Argument 3(columns renaming mapping): %s" % str(args.columns.strip("{}").split("\;")))
print("Argument 4(output cleansed taxi data path): %s" % args.output_cleanse)

raw_df = dprep.read_csv(path=args.input_cleanse, header=dprep.PromoteHeadersMode.GROUPED)

# These functions ensure that null data is removed from the data set,
# which will help increase machine learning model accuracy.
# Visit https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-data-prep
# for more details

useful_columns = [s.strip().strip("'") for s in args.useful_columns.strip("[]").split("\;")]
columns = get_dict(args.columns)

all_columns = dprep.ColumnSelector(term=".*", use_regex=True)
drop_if_all_null = [all_columns, dprep.ColumnRelationship(dprep.ColumnRelationship.ALL)]

new_df = (raw_df
          .replace_na(columns=all_columns)
          .drop_nulls(*drop_if_all_null)
          .rename_columns(column_pairs=columns)
          .keep_columns(columns=useful_columns))

if not (args.output_cleanse is None):
    os.makedirs(args.output_cleanse, exist_ok=True)
    print("%s created" % args.output_cleanse)
    write_df = new_df.write_to_csv(directory_path=dprep.LocalFileOutput(args.output_cleanse))
    write_df.run_local()
