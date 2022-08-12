from pathlib import Path
from azureml.core import Run

import argparse
import os


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    run_context = Run.get_context()
    input_path = run_context.input_datasets["train_10_models"]

    for file_name in os.listdir(input_path):
        input_file = os.path.join(input_path, file_name)
        with open(input_file, "r") as f:
            content = f.read()

            # Apply any data pre-processing techniques here

            output_file = os.path.join(output, file_name)
            with open(output_file, "w") as f:
                f.write(content)


def my_parse_args():
    parser = argparse.ArgumentParser("Test")

    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = my_parse_args()
    main(args)
