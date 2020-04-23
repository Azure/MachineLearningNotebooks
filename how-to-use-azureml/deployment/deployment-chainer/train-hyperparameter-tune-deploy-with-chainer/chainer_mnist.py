
import argparse
import os

import numpy as np

import chainer
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu

from azureml.core.run import Run
run = Run.get_context()


class MyNetwork(Chain):

    def __init__(self, n_mid_units=100, n_out=10):
        super(MyNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epochs', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--output_dir', '-o', default='./outputs',
                        help='Directory to output the result')
    parser.add_argument('--gpu_id', '-g', default=0,
                        help='ID of the GPU to be used. Set to -1 if you use CPU')
    args = parser.parse_args()

    # Download the MNIST data if you haven't downloaded it yet
    train, test = datasets.mnist.get_mnist(withlabel=True, ndim=1)

    gpu_id = args.gpu_id
    batchsize = args.batchsize
    epochs = args.epochs
    run.log('Batch size', np.int(batchsize))
    run.log('Epochs', np.int(epochs))

    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize,
                                         repeat=False, shuffle=False)

    model = MyNetwork()

    if gpu_id >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(0).use()
        model.to_gpu()  # Copy the model to the GPU

    # Choose an optimizer algorithm
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)

    # Give the optimizer a reference to the model so that it
    # can locate the model's parameters.
    optimizer.setup(model)

    while train_iter.epoch < epochs:
        # ---------- One iteration of the training loop ----------
        train_batch = train_iter.next()
        image_train, target_train = concat_examples(train_batch, gpu_id)

        # Calculate the prediction of the network
        prediction_train = model(image_train)

        # Calculate the loss with softmax_cross_entropy
        loss = F.softmax_cross_entropy(prediction_train, target_train)

        # Calculate the gradients in the network
        model.cleargrads()
        loss.backward()

        # Update all the trainable parameters
        optimizer.update()
        # --------------------- until here ---------------------

        # Check the validation accuracy of prediction after every epoch
        if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch

            # Display the training loss
            print('epoch:{:02d} train_loss:{:.04f} '.format(
                train_iter.epoch, float(to_cpu(loss.array))), end='')

            test_losses = []
            test_accuracies = []
            while True:
                test_batch = test_iter.next()
                image_test, target_test = concat_examples(test_batch, gpu_id)

                # Forward the test data
                prediction_test = model(image_test)

                # Calculate the loss
                loss_test = F.softmax_cross_entropy(prediction_test, target_test)
                test_losses.append(to_cpu(loss_test.array))

                # Calculate the accuracy
                accuracy = F.accuracy(prediction_test, target_test)
                accuracy.to_cpu()
                test_accuracies.append(accuracy.array)

                if test_iter.is_new_epoch:
                    test_iter.epoch = 0
                    test_iter.current_position = 0
                    test_iter.is_new_epoch = False
                    test_iter._pushed_position = None
                    break

            val_accuracy = np.mean(test_accuracies)
            print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
                np.mean(test_losses), val_accuracy))

            run.log("Accuracy", np.float(val_accuracy))

    serializers.save_npz(os.path.join(args.output_dir, 'model.npz'), model)


if __name__ == '__main__':
    main()
