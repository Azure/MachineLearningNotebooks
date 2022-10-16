# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""This is a sample script to show how to write an entry script."""

# Skeleton begin


def init():
    """Call once on the beginning of an agent, i.e., a worker process.

    This is before calling run() and is for init code in a process.
    For example, parse arguments here and then use them in run().

    This is optional.
    """


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


def change_file(file_name):
    """Simulate changing a file.

    Actually this does nothing.
    """
    print(f"changed file {file_name}")


def process_mini_batch(mini_batch):
    """Process a mini batch."""
    result = []

    for file_name in mini_batch:
        change_file(file_name)
        result.append(f"{file_name} changed")

    return result


# User logic end
