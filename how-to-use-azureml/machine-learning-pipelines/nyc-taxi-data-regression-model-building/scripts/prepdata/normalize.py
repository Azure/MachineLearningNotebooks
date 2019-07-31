import argparse
import os
import azureml.dataprep as dprep

print("Replace undefined values to relavant values and rename columns to meaningful names")

parser = argparse.ArgumentParser("normalize")
parser.add_argument("--input_normalize", type=str, help="combined and converted taxi data")
parser.add_argument("--output_normalize", type=str, help="replaced undefined values and renamed columns")

args = parser.parse_args()

print("Argument 1(input taxi data path): %s" % args.input_normalize)
print("Argument 2(output normalized taxi data path): %s" % args.output_normalize)

combined_converted_df = dprep.read_csv(args.input_normalize + '/part-*')

# These functions replace undefined values and rename to use meaningful names.
# Visit https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-data-prep for more details

replaced_stfor_vals_df = combined_converted_df.replace(columns="store_forward",
                                                       find="0",
                                                       replace_with="N").fill_nulls("store_forward", "N")

replaced_distance_vals_df = replaced_stfor_vals_df.replace(columns="distance",
                                                           find=".00",
                                                           replace_with=0).fill_nulls("distance", 0)

replaced_distance_vals_df = replaced_distance_vals_df.to_number(["distance"])

time_split_df = (replaced_distance_vals_df
                 .split_column_by_example(source_column="pickup_datetime")
                 .split_column_by_example(source_column="dropoff_datetime"))

# Split the pickup and dropoff datetime values into the respective date and time columns
renamed_col_df = (time_split_df
                  .rename_columns(column_pairs={
                      "pickup_datetime_1": "pickup_date",
                      "pickup_datetime_2": "pickup_time",
                      "dropoff_datetime_1": "dropoff_date",
                      "dropoff_datetime_2": "dropoff_time"}))

if not (args.output_normalize is None):
    os.makedirs(args.output_normalize, exist_ok=True)
    print("%s created" % args.output_normalize)
    write_df = renamed_col_df.write_to_csv(directory_path=dprep.LocalFileOutput(args.output_normalize))
    write_df.run_local()
