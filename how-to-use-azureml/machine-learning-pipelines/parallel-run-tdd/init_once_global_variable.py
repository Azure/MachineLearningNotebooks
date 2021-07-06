# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""This is a sample script to show how to use init()."""
import argparse
import json

# Skeleton begin


def init():
    """Call once on the beginning of an agent, i.e., a worker process.

    This is before calling run() and is for init code in a process.
    For example, parse arguments here and then use them in run().

    This is optional.
    """
    setup()


def run(mini_batch):
    """Call for each mini batch."""
    result = process_mini_batch(mini_batch)

    # ParallelRunStep uses the length of result to tell how many items succeeded.
    # For summary_only, the value of each element is not used.
    # For append_row, the value will be appended to the output file.
    return result


def shutdown():
    """Call once on the end of an agent, i.e., a worker process.

    This is optional.
    """


# Skeleton end


# User logic start

ARGS = None


def setup():
    """Parse args which will be used in run() in this process."""
    global ARGS  # pylint: disable=global-statement
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_name", dest="model_name", required=True)
    parser.add_argument("--json_setting", dest="json_setting", required=True)
    ARGS, _ = parser.parse_known_args()
    json_obj = json.loads(ARGS.json_setting)
    assert json_obj["name"] == "John"
    assert json_obj["age"] == 21


def change_file(file_name):
    """Simulate changing a file.

    Actually this does nothing.
    """
    global ARGS  # pylint: disable=global-statement
    print(f"Apply {ARGS.model_name} on {file_name} with additional setting {ARGS.json_setting}.")


def process_mini_batch(mini_batch):
    """Process a mini batch."""
    result = []

    for file_name in mini_batch:
        change_file(file_name)
        result.append(f"{file_name} changed")

    return result


# User logic end
