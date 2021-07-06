# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""This module test simple_file_dataset.py."""
import json
import sys
from unittest.mock import patch

import init_once_global_variable as t


def test_file_dataset():
    """Test your code locally before submitting a job."""
    test_argv = sys.argv.copy() + [
        "--model_name",
        "test_model_name",
        "--json_setting",
        json.dumps({"name": "John", "age": 21}),
    ]
    with patch.object(
        sys, "argv", test_argv,
    ):
        t.init()
        mini_batch = ["file1", "file2"]  # file names
        assert t.run(mini_batch) == ["file1 changed", "file2 changed"]
