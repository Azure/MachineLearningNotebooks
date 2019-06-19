import argparse
import os
import azureml.dataprep as dprep

print("Transforms the renamed taxi data to the required format")

parser = argparse.ArgumentParser("transform")
parser.add_argument("--input_transform", type=str, help="renamed taxi data")
parser.add_argument("--output_transform", type=str, help="transformed taxi data")

args = parser.parse_args()

print("Argument 1(input taxi data path): %s" % args.input_transform)
print("Argument 2(output final transformed taxi data): %s" % args.output_transform)

renamed_df = dprep.read_csv(args.input_transform + '/part-*')

# These functions transform the renamed data to be used finally for training.
# Visit https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-data-prep for more details

# Split the pickup and dropoff date further into the day of the week, day of the month, and month values.
# To get the day of the week value, use the derive_column_by_example() function.
# The function takes an array parameter of example objects that define the input data,
# and the preferred output. The function automatically determines your preferred transformation.
# For the pickup and dropoff time columns, split the time into the hour, minute, and second by using
# the split_column_by_example() function with no example parameter. After you generate the new features,
# use the drop_columns() function to delete the original fields as the newly generated features are preferred.
# Rename the rest of the fields to use meaningful descriptions.

transformed_features_df = (renamed_df
                           .derive_column_by_example(
                               source_columns="pickup_date",
                               new_column_name="pickup_weekday",
                               example_data=[("2009-01-04", "Sunday"), ("2013-08-22", "Thursday")])
                           .derive_column_by_example(
                               source_columns="dropoff_date",
                               new_column_name="dropoff_weekday",
                               example_data=[("2013-08-22", "Thursday"), ("2013-11-03", "Sunday")])

                           .split_column_by_example(source_column="pickup_time")
                           .split_column_by_example(source_column="dropoff_time")

                           .split_column_by_example(source_column="pickup_time_1")
                           .split_column_by_example(source_column="dropoff_time_1")
                           .drop_columns(columns=[
                               "pickup_date", "pickup_time", "dropoff_date", "dropoff_time",
                               "pickup_date_1", "dropoff_date_1", "pickup_time_1", "dropoff_time_1"])

                           .rename_columns(column_pairs={
                               "pickup_date_2": "pickup_month",
                               "pickup_date_3": "pickup_monthday",
                               "pickup_time_1_1": "pickup_hour",
                               "pickup_time_1_2": "pickup_minute",
                               "pickup_time_2": "pickup_second",
                               "dropoff_date_2": "dropoff_month",
                               "dropoff_date_3": "dropoff_monthday",
                               "dropoff_time_1_1": "dropoff_hour",
                               "dropoff_time_1_2": "dropoff_minute",
                               "dropoff_time_2": "dropoff_second"}))

# Drop the pickup_datetime and dropoff_datetime columns because they're
# no longer needed (granular time features like hour,
# minute and second are more useful for model training).
processed_df = transformed_features_df.drop_columns(columns=["pickup_datetime", "dropoff_datetime"])

# Use the type inference functionality to automatically check the data type of each field,
# and display the inference results.
type_infer = processed_df.builders.set_column_types()
type_infer.learn()

# The inference results look correct based on the data. Now apply the type conversions to the dataflow.
type_converted_df = type_infer.to_dataflow()

# Before you package the dataflow, run two final filters on the data set.
# To eliminate incorrectly captured data points,
# filter the dataflow on records where both the cost and distance variable values are greater than zero.
# This step will significantly improve machine learning model accuracy,
# because data points with a zero cost or distance represent major outliers that throw off prediction accuracy.

final_df = type_converted_df.filter(dprep.col("distance") > 0)
final_df = final_df.filter(dprep.col("cost") > 0)

# Writing the final dataframe to use for training in the following steps
if not (args.output_transform is None):
    os.makedirs(args.output_transform, exist_ok=True)
    print("%s created" % args.output_transform)
    write_df = final_df.write_to_csv(directory_path=dprep.LocalFileOutput(args.output_transform))
    write_df.run_local()
