import numpy as np
import os
import json

from chainer import serializers, using_config, Variable, datasets
import chainer.functions as F
import chainer.links as L
from chainer import Chain

from azureml.core.model import Model


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


def init():
    global model

    model_root = Model.get_model_path('chainer-dnn-mnist')

    # Load our saved artifacts
    model = MyNetwork()
    serializers.load_npz(model_root, model)


def run(input_data):
    i = np.array(json.loads(input_data)['data'])

    _, test = datasets.get_mnist()
    x = Variable(np.asarray([test[i][0]]))
    y = model(x)

    return np.ndarray.tolist(y.data.argmax(axis=1))
