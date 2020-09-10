# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""This module test simple_file_dataset.py."""


import simple_file_dataset as t


def test_file_dataset():
    """Test your code locally before submitting a job."""
    t.init()
    mini_batch = ["file1", "file2"]  # file names
    assert t.run(mini_batch) == ["file1 changed", "file2 changed"]
