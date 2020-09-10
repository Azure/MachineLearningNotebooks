# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""This module test simple_file_dataset.py."""

import pandas
import simple_tabular_dataset as t


def get_mini_batch():
    """Return a mini batch as testing data."""
    ping_info = {
        "servername": ["svr_et_1", "svr_et_2", "svr_wt_1", "svr_wt_2", "svr_nr_1", "svr_nr_2", "svr_st_1", "svr_st_2"],
        "lastping": [
            "12.20.15.122",
            "12.20.11.395",
            "12.20.12.836",
            "12.20.16.769",
            "12.20.17.193",
            "12.20.18.416",
            "11.59.55.913",
            "12.20.14.811",
        ],
        "roundtriptime": [300, 400, 0, 200, 100, 500, 350, 0],
        "status": ["PASS", "PASS", "FAIL", "PASS", "PASS", "PASS", "PASS", "FAIL"],
    }

    return pandas.DataFrame(data=ping_info)


def test_tabular_dataset():
    """Test your code locally before submitting a job."""
    t.init()
    mini_batch = get_mini_batch()  # a pandas.DataFrame
    assert len(mini_batch.columns) == 4

    result = t.run(mini_batch)
    assert len(result.columns) == 3, "One column dropped."
