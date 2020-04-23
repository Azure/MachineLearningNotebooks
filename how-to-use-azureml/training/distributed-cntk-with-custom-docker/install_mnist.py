# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.
# Script from:
# https://github.com/Microsoft/CNTK/blob/master/Examples/Image/DataSets/MNIST/install_mnist.py

from __future__ import print_function
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
import gzip
import os
import struct
import numpy as np


def loadData(src, cimg):
    print('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    print('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x3080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))[0]
            if n != cimg:
                raise Exception('Invalid file: expected {0} entries.'.format(cimg))
            crow = struct.unpack('>I', gz.read(4))[0]
            ccol = struct.unpack('>I', gz.read(4))[0]
            if crow != 28 or ccol != 28:
                raise Exception('Invalid file: expected 28 rows/cols per image.')
            # Read data.
            res = np.fromstring(gz.read(cimg * crow * ccol), dtype=np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, crow * ccol))


def loadLabels(src, cimg):
    print('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    print('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x1080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))
            if n[0] != cimg:
                raise Exception('Invalid file: expected {0} rows.'.format(cimg))
            # Read labels.
            res = np.fromstring(gz.read(cimg), dtype=np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, 1))


def load(dataSrc, labelsSrc, cimg):
    data = loadData(dataSrc, cimg)
    labels = loadLabels(labelsSrc, cimg)
    return np.hstack((data, labels))


def savetxt(filename, ndarray):
    with open(filename, 'w') as f:
        labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
        for row in ndarray:
            row_str = row.astype(str)
            label_str = labels[row[-1]]
            feature_str = ' '.join(row_str[:-1])
            f.write('|labels {} |features {}\n'.format(label_str, feature_str))


def main(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    train = load('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 60000)
    print('Writing train text file...')
    train_txt = os.path.join(data_dir, 'Train-28x28_cntk_text.txt')
    savetxt(train_txt, train)
    print('Done.')
    test = load('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 10000)
    print('Writing test text file...')
    test_txt = os.path.join(data_dir, 'Test-28x28_cntk_text.txt')
    savetxt(test_txt, test)
    print('Done.')


if __name__ == "__main__":
    main('mnist')
