import argparse
import os
from azureml.core import Run

print("Transforms the renamed taxi data to the required format")

run = Run.get_context()

# To learn more about how to access dataset in your script, please
# see https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-with-datasets.
normalized_data = run.input_datasets['normalized_data']
normalized_df = normalized_data.to_pandas_dataframe()

parser = argparse.ArgumentParser("transform")
parser.add_argument("--output_transform", type=str, help="transformed taxi data")

args = parser.parse_args()

print("Argument 2(output final transformed taxi data): %s" % args.output_transform)

# These functions transform the renamed data to be used finally for training.

# Split the pickup and dropoff date further into the day of the week, day of the month, and month values.
# The function takes an array parameter of example objects that define the input data,
# and the preferred output. The function automatically determines your preferred transformation.
# For the pickup and dropoff time columns, split the time into the hour, minute, and second by using
# use the drop_columns() function to delete the original fields as the newly generated features are preferred.
# Rename the rest of the fields to use meaningful descriptions.

normalized_df = normalized_df.astype({"pickup_date": 'datetime64[ns]', "dropoff_date": 'datetime64[ns]',
                                      "pickup_time": 'datetime64[us]', "dropoff_time": 'datetime64[us]',
                                      "distance": 'float64', "cost": 'float64'})

normalized_df["pickup_weekday"] = normalized_df["pickup_date"].dt.dayofweek
normalized_df["pickup_month"] = normalized_df["pickup_date"].dt.month
normalized_df["pickup_monthday"] = normalized_df["pickup_date"].dt.day

normalized_df["dropoff_weekday"] = normalized_df["dropoff_date"].dt.dayofweek
normalized_df["dropoff_month"] = normalized_df["dropoff_date"].dt.month
normalized_df["dropoff_monthday"] = normalized_df["dropoff_date"].dt.day

normalized_df["pickup_hour"] = normalized_df["pickup_time"].dt.hour
normalized_df["pickup_minute"] = normalized_df["pickup_time"].dt.minute
normalized_df["pickup_second"] = normalized_df["pickup_time"].dt.second

normalized_df["dropoff_hour"] = normalized_df["dropoff_time"].dt.hour
normalized_df["dropoff_minute"] = normalized_df["dropoff_time"].dt.minute
normalized_df["dropoff_second"] = normalized_df["dropoff_time"].dt.second

# Drop the pickup_date, dropoff_date, pickup_time, dropoff_time columns because they're
# no longer needed (granular time features like hour,
# minute and second are more useful for model training).
del normalized_df["pickup_date"]
del normalized_df["dropoff_date"]
del normalized_df["pickup_time"]
del normalized_df["dropoff_time"]

# Before you package the dataset, run two final filters on the dataset.
# To eliminate incorrectly captured data points,
# filter the dataset on records where both the cost and distance variable values are greater than zero.
# This step will significantly improve machine learning model accuracy,
# because data points with a zero cost or distance represent major outliers that throw off prediction accuracy.

final_df = normalized_df[(normalized_df.distance > 0) & (normalized_df.cost > 0)]
final_df.reset_index(inplace=True, drop=True)

# Writing the final dataframe to use for training in the following steps
if not (args.output_transform is None):
    os.makedirs(args.output_transform, exist_ok=True)
    print("%s created" % args.output_transform)
    path = args.output_transform + "/processed.parquet"
    write_df = final_df.to_parquet(path)
