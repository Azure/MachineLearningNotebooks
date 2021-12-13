import argparse
import os
# import azureml.core
from azureml.core import Run
from sklearn.model_selection import train_test_split


def write_output(df, path):
    os.makedirs(path, exist_ok=True)
    print("%s created" % path)
    df.to_parquet(path + "/processed.parquet")


print("Split the data into train and test")
run = Run.get_context()
transformed_data = run.input_datasets['transformed_data']
transformed_df = transformed_data.to_pandas_dataframe()

parser = argparse.ArgumentParser("split")
parser.add_argument("--output_split_train", type=str, help="output split train data")
parser.add_argument("--output_split_test", type=str, help="output split test data")

args = parser.parse_args()

print("Argument 1(output training data split path): %s" % args.output_split_train)
print("Argument 2(output test data split path): %s" % args.output_split_test)

# These functions splits the input features and labels into test and train data
# Visit https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-auto-train-models for more detail

output_split_train, output_split_test = train_test_split(transformed_df, test_size=0.2, random_state=223)
output_split_train.reset_index(inplace=True, drop=True)
output_split_test.reset_index(inplace=True, drop=True)

if not (args.output_split_train
        is None and args.output_split_test is None):
    write_output(output_split_train, args.output_split_train)
    write_output(output_split_test, args.output_split_test)
