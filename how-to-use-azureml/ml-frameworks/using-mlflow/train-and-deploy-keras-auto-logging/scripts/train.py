# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import mlflow
import mlflow.keras
import numpy as np
import warnings

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

print("Keras version:", keras.__version__)

# Enable auto-logging to MLflow to capture Keras metrics.
mlflow.autolog()

# Model / data parameters
n_inputs = 28 * 28
n_h1 = 300
n_h2 = 100
n_outputs = 10
learning_rate = 0.001

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Flatten image to be (n, 28 * 28)
x_train = x_train.reshape(len(x_train), -1)
x_test = x_test.reshape(len(x_test), -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, n_outputs)
y_test = keras.utils.to_categorical(y_test, n_outputs)


def driver():
    warnings.filterwarnings("ignore")

    with mlflow.start_run() as run:

        # Build a simple MLP model
        model = Sequential()
        # first hidden layer
        model.add(Dense(n_h1, activation='relu', input_shape=(n_inputs,)))
        # second hidden layer
        model.add(Dense(n_h2, activation='relu'))
        # output layer
        model.add(Dense(n_outputs, activation='softmax'))
        model.summary()

        batch_size = 128
        epochs = 5

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=learning_rate),
                      metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    return run


if __name__ == "__main__":
    driver()
