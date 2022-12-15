from pathlib import Path
from azureml.core import Run
import argparse


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    run_context = Run.get_context()
    dataset = run_context.input_datasets["train_10_models"]
    df = dataset.to_pandas_dataframe()

    # Drop the column "Revenue" from the dataset to avoid information leak as
    # "Quantity" = "Revenue" / "Price". Please modify the logic based on your data.
    drop_column_name = "Revenue"
    if drop_column_name in df.columns:
        df.drop(drop_column_name, axis=1, inplace=True)

    # Apply any data pre-processing techniques here

    df.to_parquet(output / "data_prepared_result.parquet", compression=None)


def my_parse_args():
    parser = argparse.ArgumentParser("Test")

    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = my_parse_args()
    main(args)
