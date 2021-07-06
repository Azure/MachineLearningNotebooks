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


def process_mini_batch(mini_batch):
    """Process a mini batch."""
    print(mini_batch)
    result = mini_batch.drop(["status"], axis=1)
    print(result)
    return result


# User logic end
