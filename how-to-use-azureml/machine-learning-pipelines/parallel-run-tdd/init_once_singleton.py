# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""This is a sample script to show how to use init() to init once and pass value to run() with singleton.

Using the global statement is usually considered as a bad practice in python.
"""
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


class SingletonMeta(type):
    """This is a singleton metaclass."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Lookup and create a single instance for the class if not exists, and then return it."""
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Args(metaclass=SingletonMeta):
    """A singleton class for arguments."""

    def __init__(self):
        """Init arguments."""
        self.model_name = None
        self.json_setting = None


def setup():
    """Parse args and share to run() for this process."""
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_name", dest="model_name", required=True)
    parser.add_argument("--json_setting", dest="json_setting", required=True)
    _args, _ = parser.parse_known_args()
    args = Args()
    args.model_name = _args.model_name
    args.json_setting = _args.json_setting


def change_file(file_name):
    """Simulate changing a file.

    Actually this does nothing.
    """
    args = Args()
    assert args.model_name == "test_model_name"

    json_obj = json.loads(args.json_setting)
    assert json_obj["name"] == "John"
    assert json_obj["age"] == 21

    print(f"Apply {args.model_name} on {file_name} with additional setting {args.json_setting}.")


def process_mini_batch(mini_batch):
    """Process a mini batch."""
    result = []

    for file_name in mini_batch:
        change_file(file_name)
        result.append(f"{file_name} changed")

    return result


# User logic end
