import argparse
import os
import azureml.dataprep as dprep
import azureml.core

print("Extracts important features from prepared data")

parser = argparse.ArgumentParser("featurization")
parser.add_argument("--input_featurization", type=str, help="input featurization")
parser.add_argument("--useful_columns", type=str, help="columns to use")
parser.add_argument("--output_featurization", type=str, help="output featurization")

args = parser.parse_args()

print("Argument 1(input training data path): %s" % args.input_featurization)
print("Argument 2(column features to use): %s" % str(args.useful_columns.strip("[]").split("\;")))
print("Argument 3:(output featurized training data path) %s" % args.output_featurization)

dflow_prepared = dprep.read_csv(args.input_featurization + '/part-*')

# These functions extracts useful features for training
# Visit https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-auto-train-models for more detail

useful_columns = [s.strip().strip("'") for s in args.useful_columns.strip("[]").split("\;")]
dflow = dflow_prepared.keep_columns(useful_columns)

if not (args.output_featurization is None):
    os.makedirs(args.output_featurization, exist_ok=True)
    print("%s created" % args.output_featurization)
    write_df = dflow.write_to_csv(directory_path=dprep.LocalFileOutput(args.output_featurization))
    write_df.run_local()
