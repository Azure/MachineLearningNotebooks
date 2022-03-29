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
    """Fetch the Adult Census Dataset.

    This uses a particular URL for the Adult Census dataset. The code
    is a simplified version of fetch_openml() in sklearn.

    The data are copied from:
    https://openml.org/data/v1/download/1595261.gz
    (as of 2021-03-31)
    """
    try:
        from urllib import urlretrieve
    except ImportError:
        from urllib.request import urlretrieve

    filename = "1595261.gz"
    data_url = "https://rainotebookscdn.blob.core.windows.net/datasets/"

    remaining_attempts = 5
    sleep_duration = 10
    while remaining_attempts > 0:
        try:
            urlretrieve(data_url + filename, filename)

            http_stream = gzip.GzipFile(filename=filename, mode='rb')

            with closing(http_stream):
                def _stream_generator(response):
                    for line in response:
                        yield line.decode('utf-8')

                stream = _stream_generator(http_stream)
                data = arff.load(stream)
        except Exception as exc:  # noqa: B902
            remaining_attempts -= 1
            print("Error downloading dataset from {} ({} attempt(s) remaining)"
                  .format(data_url, remaining_attempts))
            print(exc)
            sleep(sleep_duration)
            sleep_duration *= 2
            continue
        else:
            # dataset successfully downloaded
            break
    else:
        raise Exception("Could not retrieve dataset from {}.".format(data_url))

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
