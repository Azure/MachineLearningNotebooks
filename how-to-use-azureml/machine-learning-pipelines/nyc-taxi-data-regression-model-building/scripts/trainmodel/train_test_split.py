import argparse
import os
import azureml.dataprep as dprep
import azureml.core
from sklearn.model_selection import train_test_split


def write_output(df, path):
    os.makedirs(path, exist_ok=True)
    print("%s created" % path)
    df.to_csv(path + "/part-00000", index=False)


print("Split the data into train and test")

parser = argparse.ArgumentParser("split")
parser.add_argument("--input_split_features", type=str, help="input split features")
parser.add_argument("--input_split_labels", type=str, help="input split labels")
parser.add_argument("--output_split_train_x", type=str, help="output split train features")
parser.add_argument("--output_split_train_y", type=str, help="output split train labels")
parser.add_argument("--output_split_test_x", type=str, help="output split test features")
parser.add_argument("--output_split_test_y", type=str, help="output split test labels")

args = parser.parse_args()

print("Argument 1(input taxi data features path): %s" % args.input_split_features)
print("Argument 2(input taxi data labels path): %s" % args.input_split_labels)
print("Argument 3(output training features split path): %s" % args.output_split_train_x)
print("Argument 4(output training labels split path): %s" % args.output_split_train_y)
print("Argument 5(output test features split path): %s" % args.output_split_test_x)
print("Argument 6(output test labels split path): %s" % args.output_split_test_y)

x_df = dprep.read_csv(path=args.input_split_features, header=dprep.PromoteHeadersMode.GROUPED).to_pandas_dataframe()
y_df = dprep.read_csv(path=args.input_split_labels, header=dprep.PromoteHeadersMode.GROUPED).to_pandas_dataframe()

# These functions splits the input features and labels into test and train data
# Visit https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-auto-train-models for more detail

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=223)

if not (args.output_split_train_x is None and
        args.output_split_test_x is None and
        args.output_split_train_y is None and
        args.output_split_test_y is None):
    write_output(x_train, args.output_split_train_x)
    write_output(y_train, args.output_split_train_y)
    write_output(x_test, args.output_split_test_x)
    write_output(y_test, args.output_split_test_y)
