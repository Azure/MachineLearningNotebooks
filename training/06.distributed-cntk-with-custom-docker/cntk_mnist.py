# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Script adapted from:
# 1. https://github.com/Microsoft/CNTK/blob/v2.0/Tutorials/CNTK_103A_MNIST_DataLoader.ipynb
# 2. https://github.com/Microsoft/CNTK/blob/v2.0/Tutorials/CNTK_103C_MNIST_MultiLayerPerceptron.ipynb
# ===================================================================================================
"""Train a CNTK multi-layer perceptron on the MNIST dataset."""

from __future__ import print_function
import gzip
import numpy as np
import os
import shutil
import struct
import sys
import time

import cntk as C
from azureml.core.run import Run
import argparse

run = Run.get_submitted_run()

parser = argparse.ArgumentParser()

parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_hidden_layers', type=int, default=2, help='number of hidden layers')
parser.add_argument('--minibatch_size', type=int, default=64, help='minibatchsize')

args = parser.parse_args()

# Functions to load MNIST images and unpack into train and test set.
# - loadData reads image data and formats into a 28x28 long array
# - loadLabels reads the corresponding labels data, 1 for each image
# - load packs the downloaded image and labels data into a combined format to be read later by
#   CNTK text reader


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


def try_download(dataSrc, labelsSrc, cimg):
    data = loadData(dataSrc, cimg)
    labels = loadLabels(labelsSrc, cimg)
    return np.hstack((data, labels))

# Save the data files into a format compatible with CNTK text reader


def savetxt(filename, ndarray):
    dir = os.path.dirname(filename)

    if not os.path.exists(dir):
        os.makedirs(dir)

    if not os.path.isfile(filename):
        print("Saving", filename)
        with open(filename, 'w') as f:
            labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
            for row in ndarray:
                row_str = row.astype(str)
                label_str = labels[row[-1]]
                feature_str = ' '.join(row_str[:-1])
                f.write('|labels {} |features {}\n'.format(label_str, feature_str))
    else:
        print("File already exists", filename)

# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file


def create_reader(path, is_training, input_dim, num_label_classes):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        labels=C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
        features=C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
    )), randomize=is_training, max_sweeps=C.io.INFINITELY_REPEAT if is_training else 1)

# Defines a utility that prints the training progress


def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb % frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error * 100))

    return mb, training_loss, eval_error

# Create the network architecture


def create_model(features):
    with C.layers.default_options(init=C.layers.glorot_uniform(), activation=C.ops.relu):
        h = features
        for _ in range(num_hidden_layers):
            h = C.layers.Dense(hidden_layers_dim)(h)
        r = C.layers.Dense(num_output_classes, activation=None)(h)
        return r


if __name__ == '__main__':
    run = Run.get_submitted_run()

    try:
        from urllib.request import urlretrieve
    except ImportError:
        from urllib import urlretrieve

    # Select the right target device when this script is being used:
    if 'TEST_DEVICE' in os.environ:
        if os.environ['TEST_DEVICE'] == 'cpu':
            C.device.try_set_default_device(C.device.cpu())
        else:
            C.device.try_set_default_device(C.device.gpu(0))

    # URLs for the train image and labels data
    url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    num_train_samples = 60000

    print("Downloading train data")
    train = try_download(url_train_image, url_train_labels, num_train_samples)

    url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    num_test_samples = 10000

    print("Downloading test data")
    test = try_download(url_test_image, url_test_labels, num_test_samples)

    # Save the train and test files (prefer our default path for the data
    rank = os.environ.get("OMPI_COMM_WORLD_RANK")
    data_dir = os.path.join("outputs", "MNIST")
    sentinel_path = os.path.join(data_dir, "complete.txt")
    if rank == '0':
        print('Writing train text file...')
        savetxt(os.path.join(data_dir, "Train-28x28_cntk_text.txt"), train)

        print('Writing test text file...')
        savetxt(os.path.join(data_dir, "Test-28x28_cntk_text.txt"), test)
        with open(sentinel_path, 'w+') as f:
            f.write("download complete")

        print('Done with downloading data.')
    else:
        while not os.path.exists(sentinel_path):
            time.sleep(0.01)

    # Ensure we always get the same amount of randomness
    np.random.seed(0)

    # Define the data dimensions
    input_dim = 784
    num_output_classes = 10

    # Ensure the training and test data is generated and available for this tutorial.
    # We search in two locations in the toolkit for the cached MNIST data set.
    data_found = False
    for data_dir in [os.path.join("..", "Examples", "Image", "DataSets", "MNIST"),
                     os.path.join("data_" + str(rank), "MNIST"),
                     os.path.join("outputs", "MNIST")]:
        train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
        test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            data_found = True
            break
    if not data_found:
        raise ValueError("Please generate the data by completing CNTK 103 Part A")
    print("Data directory is {0}".format(data_dir))

    num_hidden_layers = args.num_hidden_layers
    hidden_layers_dim = 400

    input = C.input_variable(input_dim)
    label = C.input_variable(num_output_classes)

    z = create_model(input)
    # Scale the input to 0-1 range by dividing each pixel by 255.
    z = create_model(input / 255.0)

    loss = C.cross_entropy_with_softmax(z, label)
    label_error = C.classification_error(z, label)

    # Instantiate the trainer object to drive the model training
    learning_rate = args.learning_rate
    lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
    learner = C.sgd(z.parameters, lr_schedule)
    trainer = C.Trainer(z, (loss, label_error), [learner])

    # Initialize the parameters for the trainer
    minibatch_size = args.minibatch_size
    num_samples_per_sweep = 60000
    num_sweeps_to_train_with = 10
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

    # Create the reader to training data set
    reader_train = create_reader(train_file, True, input_dim, num_output_classes)

    # Map the data streams to the input and labels.
    input_map = {
        label: reader_train.streams.labels,
        input: reader_train.streams.features
    }

    # Run the trainer on and perform model training
    training_progress_output_freq = 500

    errors = []
    losses = []
    for i in range(0, int(num_minibatches_to_train)):
        # Read a mini batch from the training data file
        data = reader_train.next_minibatch(minibatch_size, input_map=input_map)

        trainer.train_minibatch(data)
        batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)
        if (error != 'NA') and (loss != 'NA'):
            errors.append(float(error))
            losses.append(float(loss))

    # log the losses
    if rank == '0':
        run.log_list("Loss", losses)
        run.log_list("Error", errors)

    # Read the training data
    reader_test = create_reader(test_file, False, input_dim, num_output_classes)

    test_input_map = {
        label: reader_test.streams.labels,
        input: reader_test.streams.features,
    }

    # Test data for trained model
    test_minibatch_size = 512
    num_samples = 10000
    num_minibatches_to_test = num_samples // test_minibatch_size
    test_result = 0.0

    for i in range(num_minibatches_to_test):
        # We are loading test data in batches specified by test_minibatch_size
        # Each data point in the minibatch is a MNIST digit image of 784 dimensions
        # with one pixel per dimension that we will encode / decode with the
        # trained model.
        data = reader_test.next_minibatch(test_minibatch_size,
                                          input_map=test_input_map)

        eval_error = trainer.test_minibatch(data)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    print("Average test error: {0:.2f}%".format((test_result * 100) / num_minibatches_to_test))

    out = C.softmax(z)

    # Read the data for evaluation
    reader_eval = create_reader(test_file, False, input_dim, num_output_classes)

    eval_minibatch_size = 25
    eval_input_map = {input: reader_eval.streams.features}

    data = reader_test.next_minibatch(eval_minibatch_size, input_map=test_input_map)

    img_label = data[label].asarray()
    img_data = data[input].asarray()
    predicted_label_prob = [out.eval(img_data[i]) for i in range(len(img_data))]

    # Find the index with the maximum value for both predicted as well as the ground truth
    pred = [np.argmax(predicted_label_prob[i]) for i in range(len(predicted_label_prob))]
    gtlabel = [np.argmax(img_label[i]) for i in range(len(img_label))]

    print("Label    :", gtlabel[:25])
    print("Predicted:", pred)

    # save model to outputs folder
    z.save('outputs/cntk.model')
