import argparse
import os
import azureml.dataprep as dprep
import azureml.core
from sklearn.model_selection import train_test_split


def df2csv(df, path, filename, **kwargs):
    """
    *Write dataframe to disk

    *Args:
        *df (Dataframe): dataframe
        *dir (): The path to write the file to
        *filename (): The filename
        *kwargs (): Optional arguments

    *Returns:
        *None
    """
    print("%s created" % path)
    file_path = os.path.join(path, filename)
    df.to_csv(file_path, index=False, **kwargs)


print("Split the data into train and test")

parser = argparse.ArgumentParser("split")
parser.add_argument("--input_split_features", type=str,
                    help="input split features")
parser.add_argument("--input_split_labels", type=str,
                    help="input split labels")
parser.add_argument("--output_split", type=str,
                    help="output split directory")

args = parser.parse_args()

print("Argument 1(input taxi data features path): %s" %
      args.input_split_features)
print("Argument 2(input taxi data labels path): %s" % args.input_split_labels)
print("Argument 3(output training split path): %s" % args.output_split)

x_df = dprep.read_csv(path=args.input_split_features,
                      header=dprep.PromoteHeadersMode.GROUPED).to_pandas_dataframe()
y_df = dprep.read_csv(path=args.input_split_labels,
                      header=dprep.PromoteHeadersMode.GROUPED).to_pandas_dataframe()

# These functions splits the input features and labels into test and train data
# Visit https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-auto-train-models for more detail

x_train, x_test, y_train, y_test = train_test_split(
    x_df, y_df, test_size=0.2, random_state=223)

csv_files = {
    'x_train.csv': x_train,
    'x_test.csv': x_test,
    'y_train.csv': y_train,
    'y_test.csv': y_test
}

os.makedirs(args.output_split, exist_ok=True)
for (key, value) in csv_files.items():
    df2csv(value, args.output_split, key)