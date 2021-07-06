# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""This is a sample script to show how to write an entry script."""
import os

AML_COMPUTE = "AZUREML_RUN_ID" in os.environ  # Inside AmlCompute.
if AML_COMPUTE:
    from azureml_user.parallel_run import EntryScript
else:  # Fallback to the dummy helper for local testing.
    from dummy_entry_script import DummyEntryScript as EntryScript

# Skeleton begin


def init():
    """Call once on the beginning of an agent, i.e., a worker process.

    This is before calling run() and is for init code in a process.
    For example, parse arguments here and then use them in run().

    This is optional.
    """
    logger = EntryScript().logger
    logger.info("init() started.")


def run(mini_batch):
    """Call for each mini batch."""
    logger = EntryScript().logger
    logger.info(f"run() is called with mini_batch: {mini_batch}.")

    result = process_mini_batch(mini_batch)

    # ParallelRunStep uses the length of result to tell how many items succeeded.
    # For summary_only, the value of each element is not used.
    # For append_row, the value will be appended to the output file.
    return result


def shutdown():
    """Call once on the end of an agent, i.e., a worker process.

    This is optional.
    """
    logger = EntryScript().logger
    logger.info("shutdown is called.")


# Skeleton end


# User logic start


def change_file(file_name):
    """Simulate changing a file.

    Actually this does nothing.
    """
    logger = EntryScript().logger
    logger.info(f"change_file() {file_name}.")

    print(f"changed file {file_name}")


def process_mini_batch(mini_batch):
    """Process a mini batch."""
    result = []

    for file_name in mini_batch:
        change_file(file_name)
        result.append(f"{file_name} changed")

    return result


# User logic end
