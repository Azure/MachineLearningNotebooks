# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Utilities for azureml-contrib-fairness notebooks."""

import arff
from collections import OrderedDict
from contextlib import closing
import gzip
import pandas as pd
from sklearn.utils import Bunch
from time import sleep


def _is_gzip_encoded(_fsrc):
    return _fsrc.info().get('Content-Encoding', '') == 'gzip'


_categorical_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country'
]


def fetch_census_dataset():
    """Fetch the Adult Census Dataset

    This uses a particular URL for the Adult Census dataset. The code
    is a simplified version of fetch_openml() in sklearn.

    The data are copied from:
    https://openml.org/data/v1/download/1595261.gz
    (as of 2021-03-31)
    """

    dataset_path = "1595261.gz"

    try:
        file_stream = gzip.GzipFile(filename=dataset_path, mode='rb')

        with closing(file_stream):
            def _stream_generator(response):
                for line in response:
                    yield line.decode('utf-8')

            stream = _stream_generator(file_stream)
            data = arff.load(stream)
    except Exception as exc:
        raise Exception("Could not load dataset from {} with exception {}".format(dataset_path, exc))

    attributes = OrderedDict(data['attributes'])
    arff_columns = list(attributes)

    raw_df = pd.DataFrame(data=data['data'], columns=arff_columns)

    target_column_name = 'class'
    target = raw_df.pop(target_column_name)
    for col_name in _categorical_columns:
        dtype = pd.api.types.CategoricalDtype(attributes[col_name])
        raw_df[col_name] = raw_df[col_name].astype(dtype, copy=False)

    result = Bunch()
    result.data = raw_df
    result.target = target

    return result
