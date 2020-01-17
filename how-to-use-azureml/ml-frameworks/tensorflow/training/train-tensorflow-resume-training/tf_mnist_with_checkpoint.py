# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import argparse
import os
import re
import tensorflow as tf
import glob

from azureml.core import Run
from utils import load_data

print("TensorFlow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')

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

X_train = load_data(glob.glob(os.path.join(data_folder, '**/train-images-idx3-ubyte.gz'),
                              recursive=True)[0], False) / 255.0
X_test = load_data(glob.glob(os.path.join(data_folder, '**/t10k-images-idx3-ubyte.gz'),
                             recursive=True)[0], False) / 255.0
y_train = load_data(glob.glob(os.path.join(data_folder, '**/train-labels-idx1-ubyte.gz'),
                              recursive=True)[0], True).reshape(-1)
y_test = load_data(glob.glob(os.path.join(data_folder, '**/t10k-labels-idx1-ubyte.gz'),
                             recursive=True)[0], True).reshape(-1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

training_set_size = X_train.shape[0]

n_inputs = 28 * 28
n_h1 = 100
n_h2 = 100
n_outputs = 10
learning_rate = 0.01
n_epochs = 20
batch_size = 50

with tf.name_scope('network'):
    # construct the DNN
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')
    h1 = tf.layers.dense(X, n_h1, activation=tf.nn.relu, name='h1')
    h2 = tf.layers.dense(h1, n_h2, activation=tf.nn.relu, name='h2')
    output = tf.layers.dense(h2, n_outputs, name='output')

with tf.name_scope('train'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output)
    loss = tf.reduce_mean(cross_entropy, name='loss')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(output, y, 1)
    acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# start an Azure ML run
run = Run.get_context()

with tf.Session() as sess:
    start_epoch = 0
    if previous_model_location:
        checkpoint_file_path = tf.train.latest_checkpoint(previous_model_location)
        saver.restore(sess, checkpoint_file_path)
        checkpoint_filename = os.path.basename(checkpoint_file_path)
        num_found = re.search(r'\d+', checkpoint_filename)
        if num_found:
            start_epoch = int(num_found.group(0))
            print("Resuming from epoch {}".format(str(start_epoch)))
    else:
        init.run()

    for epoch in range(start_epoch, n_epochs):

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
            sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
        # evaluate training set
        acc_train = acc_op.eval(feed_dict={X: X_batch, y: y_batch})
        # evaluate validation set
        acc_val = acc_op.eval(feed_dict={X: X_test, y: y_test})

        # log accuracies
        run.log('training_acc', np.float(acc_train))
        run.log('validation_acc', np.float(acc_val))
        print(epoch, '-- Training accuracy:', acc_train, '\b Validation accuracy:', acc_val)
        y_hat = np.argmax(output.eval(feed_dict={X: X_test}), axis=1)

        if epoch % 5 == 0:
            saver.save(sess, './outputs/', global_step=epoch)

        # saving only half of the model and resuming again from same epoch
        if not previous_model_location and epoch == 10:
            break

    run.log('final_acc', np.float(acc_val))
