import argparse
import os
from azureml.core import Run

print("Merge Green and Yellow taxi data")

run = Run.get_context()

# To learn more about how to access dataset in your script, please
# see https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-with-datasets.
cleansed_green_data = run.input_datasets["cleansed_green_data"]
cleansed_yellow_data = run.input_datasets["cleansed_yellow_data"]
green_df = cleansed_green_data.to_pandas_dataframe()
yellow_df = cleansed_yellow_data.to_pandas_dataframe()

parser = argparse.ArgumentParser("merge")
parser.add_argument("--output_merge", type=str, help="green and yellow taxi data merged")

args = parser.parse_args()
print("Argument (output merge taxi data path): %s" % args.output_merge)

# Appending yellow data to green data
combined_df = green_df.append(yellow_df, ignore_index=True)
combined_df.reset_index(inplace=True, drop=True)

if not (args.output_merge is None):
    os.makedirs(args.output_merge, exist_ok=True)
    print("%s created" % args.output_merge)
    path = args.output_merge + "/processed.parquet"
    write_df = combined_df.to_parquet(path)
