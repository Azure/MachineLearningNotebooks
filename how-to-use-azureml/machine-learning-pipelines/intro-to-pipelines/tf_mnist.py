# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import argparse
import os
import re
import tensorflow as tf
import time
import glob

from azureml.core import Run
from utils import load_data
from tensorflow.keras import Model, layers


# Create TF Model.
class NeuralNet(Model):
    # Set layers.
    def __init__(self):
        super(NeuralNet, self).__init__()
        # First hidden layer.
        self.h1 = layers.Dense(n_h1, activation=tf.nn.relu)
        # Second hidden layer.
        self.h2 = layers.Dense(n_h2, activation=tf.nn.relu)
        self.out = layers.Dense(n_outputs)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)
        if not is_training:
            # Apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


def cross_entropy_loss(y, logits):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    # Average loss across the batch.
    return tf.reduce_mean(loss)


# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        logits = neural_net(x, is_training=True)
        # Compute loss.
        loss = cross_entropy_loss(y, logits)

    # Variables to update, i.e. trainable variables.
    trainable_variables = neural_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))


print("TensorFlow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', default='data', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=128, help='mini batch size for training')
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=128,
                    help='# of neurons in the first layer')
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=128,
                    help='# of neurons in the second layer')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01, help='learning rate')
parser.add_argument('--resume-from', type=str, default=None,
                    help='location of the model or checkpoint files from where to resume the training')
args = parser.parse_args()

previous_model_location = args.resume_from
# You can also use environment variable to get the model/checkpoint files location
# previous_model_location = os.path.expandvars(os.getenv("AZUREML_DATAREFERENCE_MODEL_LOCATION", None))

data_folder = args.data_folder
print('Data folder:', data_folder)

# load train and test set into numpy arrays
# note we scale the pixel intensity values to 0-1 (by dividing it with 255.0) so the model can converge faster.
X_train = load_data(os.path.join(data_folder, 'train-images.gz'), False) / 255.0
X_test = load_data(os.path.join(data_folder, 'test-images.gz'), False) / 255.0
y_train = load_data(os.path.join(data_folder, 'train-labels.gz'), True).reshape(-1)
y_test = load_data(os.path.join(data_folder, 'test-labels.gz'), True).reshape(-1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

training_set_size = X_train.shape[0]

n_inputs = 28 * 28
n_h1 = args.n_hidden_1
n_h2 = args.n_hidden_2
n_outputs = 10
learning_rate = args.learning_rate
n_epochs = 20
batch_size = args.batch_size

# Build neural network model.
neural_net = NeuralNet()

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

# start an Azure ML run
run = Run.get_context()

if previous_model_location:
    # Restore variables from latest checkpoint.
    checkpoint = tf.train.Checkpoint(model=neural_net, optimizer=optimizer)
    checkpoint_file_path = tf.train.latest_checkpoint(previous_model_location)
    checkpoint.restore(checkpoint_file_path)
    checkpoint_filename = os.path.basename(checkpoint_file_path)
    num_found = re.search(r'\d+', checkpoint_filename)
    if num_found:
        start_epoch = int(num_found.group(0))
        print("Resuming from epoch {}".format(str(start_epoch)))

start_time = time.perf_counter()
for epoch in range(0, n_epochs):

    # randomly shuffle training set
    indices = np.random.permutation(training_set_size)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # batch index
    b_start = 0
    b_end = b_start + batch_size
    for _ in range(training_set_size // batch_size):
        # get a batch
        X_batch, y_batch = X_train[b_start: b_end], y_train[b_start: b_end]

        # update batch index for the next batch
        b_start = b_start + batch_size
        b_end = min(b_start + batch_size, training_set_size)

        # train
        run_optimization(X_batch, y_batch)

    # evaluate training set
    pred = neural_net(X_batch, is_training=False)
    acc_train = accuracy(pred, y_batch)

    # evaluate validation set
    pred = neural_net(X_test, is_training=False)
    acc_val = accuracy(pred, y_test)

    # log accuracies
    run.log('training_acc', np.float(acc_train))
    run.log('validation_acc', np.float(acc_val))
    print(epoch, '-- Training accuracy:', acc_train, '\b Validation accuracy:', acc_val)

    # Save checkpoints in the "./outputs" folder so that they are automatically uploaded into run history.
    checkpoint_dir = './outputs/'
    checkpoint = tf.train.Checkpoint(model=neural_net, optimizer=optimizer)

    if epoch % 2 == 0:
        checkpoint.save(checkpoint_dir)

run.log('final_acc', np.float(acc_val))
os.makedirs('./outputs/model', exist_ok=True)

# files saved in the "./outputs" folder are automatically uploaded into run history
# this is workaround for https://github.com/tensorflow/tensorflow/issues/33913 and will be fixed once we move to >tf2.1
# neural_net._set_inputs(X_train)
tf.saved_model.save(neural_net, './outputs/model/')

stop_time = time.perf_counter()
training_time = (stop_time - start_time) * 1000
print("Total time in milliseconds for training: {}".format(str(training_time)))
