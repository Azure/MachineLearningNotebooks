# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Utilities for azureml-contrib-fairness notebooks."""

from sklearn.datasets import fetch_openml
import time


def fetch_openml_with_retries(data_id, max_retries=4, retry_delay=60):
    """Fetch a given dataset from OpenML with retries as specified."""
    for i in range(max_retries):
        try:
            print("Download attempt {0} of {1}".format(i + 1, max_retries))
            data = fetch_openml(data_id=data_id, as_frame=True)
            break
        except Exception as e:
            print("Download attempt failed with exception:")
            print(e)
            if i + 1 != max_retries:
                print("Will retry after {0} seconds".format(retry_delay))
                time.sleep(retry_delay)
                retry_delay = retry_delay * 2
    else:
        raise RuntimeError("Unable to download dataset from OpenML")

    return data
