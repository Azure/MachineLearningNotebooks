# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import gzip
import numpy as np
import os
import struct

from azureml.core import Dataset
from azureml.opendatasets import MNIST
from chainer.datasets import tuple_dataset


# load compressed MNIST gz files and return numpy arrays
def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        struct.unpack('I', gz.read(4))
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res


def download_mnist():
    data_folder = os.path.join(os.getcwd(), 'data/mnist')
    os.makedirs(data_folder, exist_ok=True)

    mnist_file_dataset = MNIST.get_file_dataset()
    mnist_file_dataset.download(data_folder, overwrite=True)

    X_train = load_data(glob.glob(os.path.join(data_folder, "**/train-images-idx3-ubyte.gz"),
                        recursive=True)[0], False) / 255.0
    X_test = load_data(glob.glob(os.path.join(data_folder, "**/t10k-images-idx3-ubyte.gz"),
                       recursive=True)[0], False) / 255.0
    y_train = load_data(glob.glob(os.path.join(data_folder, "**/train-labels-idx1-ubyte.gz"),
                        recursive=True)[0], True).reshape(-1)
    y_test = load_data(glob.glob(os.path.join(data_folder, "**/t10k-labels-idx1-ubyte.gz"),
                       recursive=True)[0], True).reshape(-1)

    train = tuple_dataset.TupleDataset(X_train.astype(np.float32), y_train.astype(np.int32))
    test = tuple_dataset.TupleDataset(X_test.astype(np.float32), y_test.astype(np.int32))

    return train, test
