import argparse
import os
import azureml.dataprep as dprep

print("Filters out coordinates for locations that are outside the city border.",
      "Chain the column filter commands within the filter() function",
      "and define the minimum and maximum bounds for each field.")

parser = argparse.ArgumentParser("filter")
parser.add_argument("--input_filter", type=str, help="merged taxi data directory")
parser.add_argument("--output_filter", type=str, help="filter out out of city locations")

args = parser.parse_args()

print("Argument 1(input taxi data path): %s" % args.input_filter)
print("Argument 2(output filtered taxi data path): %s" % args.output_filter)

combined_df = dprep.read_csv(args.input_filter + '/part-*')

# These functions filter out coordinates for locations that are outside the city border.
# Visit https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-data-prep for more details

# Create a condensed view of the dataflow to just show the lat/long fields,
# which makes it easier to evaluate missing or out-of-scope coordinates
decimal_type = dprep.TypeConverter(data_type=dprep.FieldType.DECIMAL)
combined_df = combined_df.set_column_types(type_conversions={
    "pickup_longitude": decimal_type,
    "pickup_latitude": decimal_type,
    "dropoff_longitude": decimal_type,
    "dropoff_latitude": decimal_type
})

# Filter out coordinates for locations that are outside the city border.
# Chain the column filter commands within the filter() function
# and define the minimum and maximum bounds for each field
latlong_filtered_df = (combined_df
                       .drop_nulls(columns=["pickup_longitude",
                                            "pickup_latitude",
                                            "dropoff_longitude",
                                            "dropoff_latitude"],
                                   column_relationship=dprep.ColumnRelationship(dprep.ColumnRelationship.ANY))
                       .filter(dprep.f_and(dprep.col("pickup_longitude") <= -73.72,
                                           dprep.col("pickup_longitude") >= -74.09,
                                           dprep.col("pickup_latitude") <= 40.88,
                                           dprep.col("pickup_latitude") >= 40.53,
                                           dprep.col("dropoff_longitude") <= -73.72,
                                           dprep.col("dropoff_longitude") >= -74.09,
                                           dprep.col("dropoff_latitude") <= 40.88,
                                           dprep.col("dropoff_latitude") >= 40.53)))

if not (args.output_filter is None):
    os.makedirs(args.output_filter, exist_ok=True)
    print("%s created" % args.output_filter)
    write_df = latlong_filtered_df.write_to_csv(directory_path=dprep.LocalFileOutput(args.output_filter))
    write_df.run_local()
