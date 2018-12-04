# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.
# Adapted from:
# https://github.com/Microsoft/CNTK/blob/master/Examples/Image/Classification/ConvNet/Python/ConvNet_MNIST.py
# ====================================================================
"""Train a CNN model on the MNIST dataset via distributed training."""

from __future__ import print_function
import numpy as np
import os
import cntk as C
import argparse
from cntk.train.training_session import CheckpointConfig, TestConfig


def create_reader(path, is_training, input_dim, label_dim, total_number_of_samples):
    """Define the reader for both training and evaluation action."""
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        features=C.io.StreamDef(field='features', shape=input_dim),
        labels=C.io.StreamDef(field='labels', shape=label_dim)
    )), randomize=is_training, max_samples=total_number_of_samples)


def convnet_mnist(max_epochs, output_dir, data_dir, debug_output=False, epoch_size=60000, minibatch_size=64):
    """Creates and trains a feedforward classification model for MNIST images."""
    image_height = 28
    image_width = 28
    num_channels = 1
    input_dim = image_height * image_width * num_channels
    num_output_classes = 10

    # Input variables denoting the features and label data
    input_var = C.ops.input_variable((num_channels, image_height, image_width), np.float32)
    label_var = C.ops.input_variable(num_output_classes, np.float32)

    # Instantiate the feedforward classification model
    scaled_input = C.ops.element_times(C.ops.constant(0.00390625), input_var)

    with C.layers.default_options(activation=C.ops.relu, pad=False):
        conv1 = C.layers.Convolution2D((5, 5), 32, pad=True)(scaled_input)
        pool1 = C.layers.MaxPooling((3, 3), (2, 2))(conv1)
        conv2 = C.layers.Convolution2D((3, 3), 48)(pool1)
        pool2 = C.layers.MaxPooling((3, 3), (2, 2))(conv2)
        conv3 = C.layers.Convolution2D((3, 3), 64)(pool2)
        f4 = C.layers.Dense(96)(conv3)
        drop4 = C.layers.Dropout(0.5)(f4)
        z = C.layers.Dense(num_output_classes, activation=None)(drop4)

    ce = C.losses.cross_entropy_with_softmax(z, label_var)
    pe = C.metrics.classification_error(z, label_var)

    # Load train data
    reader_train = create_reader(os.path.join(data_dir, 'Train-28x28_cntk_text.txt'), True,
                                 input_dim, num_output_classes, max_epochs * epoch_size)
    # Load test data
    reader_test = create_reader(os.path.join(data_dir, 'Test-28x28_cntk_text.txt'), False,
                                input_dim, num_output_classes, C.io.FULL_DATA_SWEEP)

    # Set learning parameters
    lr_per_sample = [0.001] * 10 + [0.0005] * 10 + [0.0001]
    lr_schedule = C.learning_parameter_schedule_per_sample(lr_per_sample, epoch_size=epoch_size)
    mms = [0] * 5 + [0.9990239141819757]
    mm_schedule = C.learners.momentum_schedule_per_sample(mms, epoch_size=epoch_size)

    # Instantiate the trainer object to drive the model training
    local_learner = C.learners.momentum_sgd(z.parameters, lr_schedule, mm_schedule)
    progress_printer = C.logging.ProgressPrinter(
        tag='Training',
        rank=C.train.distributed.Communicator.rank(),
        num_epochs=max_epochs,
    )

    learner = C.train.distributed.data_parallel_distributed_learner(local_learner)
    trainer = C.Trainer(z, (ce, pe), learner, progress_printer)

    # define mapping from reader streams to network inputs
    input_map_train = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    input_map_test = {
        input_var: reader_test.streams.features,
        label_var: reader_test.streams.labels
    }

    C.logging.log_number_of_parameters(z)
    print()

    C.train.training_session(
        trainer=trainer,
        mb_source=reader_train,
        model_inputs_to_streams=input_map_train,
        mb_size=minibatch_size,
        progress_frequency=epoch_size,
        checkpoint_config=CheckpointConfig(frequency=epoch_size,
                                           filename=os.path.join(output_dir, "ConvNet_MNIST")),
        test_config=TestConfig(reader_test, minibatch_size=minibatch_size,
                               model_inputs_to_streams=input_map_test)
    ).train()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', help='Total number of epochs to train', type=int, default='40')
    parser.add_argument('--output_dir', help='Output directory', required=False, default='outputs')
    parser.add_argument('--data_dir', help='Directory with training data')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    convnet_mnist(args.num_epochs, args.output_dir, args.data_dir)

    # Must call MPI finalize when process exit without exceptions
    C.train.distributed.Communicator.finalize()
