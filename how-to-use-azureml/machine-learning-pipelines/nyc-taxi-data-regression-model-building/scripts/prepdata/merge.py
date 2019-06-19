
import argparse
import os
import azureml.dataprep as dprep

print("Merge Green and Yellow taxi data")

parser = argparse.ArgumentParser("merge")
parser.add_argument("--input_green_merge", type=str, help="cleaned green taxi data directory")
parser.add_argument("--input_yellow_merge", type=str, help="cleaned yellow taxi data directory")
parser.add_argument("--output_merge", type=str, help="green and yellow taxi data merged")

args = parser.parse_args()

print("Argument 1(input green taxi data path): %s" % args.input_green_merge)
print("Argument 2(input yellow taxi data path): %s" % args.input_yellow_merge)
print("Argument 3(output merge taxi data path): %s" % args.output_merge)

green_df = dprep.read_csv(args.input_green_merge + '/part-*')
yellow_df = dprep.read_csv(args.input_yellow_merge + '/part-*')

# Appending yellow data to green data
combined_df = green_df.append_rows([yellow_df])

if not (args.output_merge is None):
    os.makedirs(args.output_merge, exist_ok=True)
    print("%s created" % args.output_merge)
    write_df = combined_df.write_to_csv(directory_path=dprep.LocalFileOutput(args.output_merge))
    write_df.run_local()
