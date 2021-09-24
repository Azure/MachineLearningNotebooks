import argparse
import os
import pandas as pd
from azureml.core import Run

print("Replace undefined values to relavant values and rename columns to meaningful names")

run = Run.get_context()

# To learn more about how to access dataset in your script, please
# see https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-with-datasets.
filtered_data = run.input_datasets['filtered_data']
combined_converted_df = filtered_data.to_pandas_dataframe()

parser = argparse.ArgumentParser("normalize")
parser.add_argument("--output_normalize", type=str, help="replaced undefined values and renamed columns")

args = parser.parse_args()

print("Argument (output normalized taxi data path): %s" % args.output_normalize)

# These functions replace undefined values and rename to use meaningful names.
replaced_stfor_vals_df = (combined_converted_df.replace({"store_forward": "0"}, {"store_forward": "N"})
                          .fillna({"store_forward": "N"}))

replaced_distance_vals_df = (replaced_stfor_vals_df.replace({"distance": ".00"}, {"distance": 0})
                             .fillna({"distance": 0}))

normalized_df = replaced_distance_vals_df.astype({"distance": 'float64'})


def time_to_us(time_str):
    hh, mm , ss = map(int, time_str.split(':'))
    return (ss + 60 * (mm + 60 * hh)) * (10**6)


temp = pd.DatetimeIndex(normalized_df["pickup_datetime"])
normalized_df["pickup_date"] = pd.to_datetime(temp.date)
normalized_df["pickup_time"] = temp.time
normalized_df["pickup_time"] = normalized_df["pickup_time"].apply(lambda x: time_to_us(str(x)))

temp = pd.DatetimeIndex(normalized_df["dropoff_datetime"])
normalized_df["dropoff_date"] = pd.to_datetime(temp.date)
normalized_df["dropoff_time"] = temp.time
normalized_df["dropoff_time"] = normalized_df["dropoff_time"].apply(lambda x: time_to_us(str(x)))

del normalized_df["pickup_datetime"]
del normalized_df["dropoff_datetime"]

normalized_df.reset_index(inplace=True, drop=True)

if not (args.output_normalize is None):
    os.makedirs(args.output_normalize, exist_ok=True)
    print("%s created" % args.output_normalize)
    path = args.output_normalize + "/processed.parquet"
    write_df = normalized_df.to_parquet(path)
