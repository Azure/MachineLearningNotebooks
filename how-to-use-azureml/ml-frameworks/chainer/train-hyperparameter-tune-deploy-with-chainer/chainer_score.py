import numpy as np
import os
import json

from utils import download_mnist

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

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_root = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.npz')

    # Load our saved artifacts
    model = MyNetwork()
    serializers.load_npz(model_root, model)


def run(input_data):
    i = np.array(json.loads(input_data)['data'])

    _, test = download_mnist()
    x = Variable(np.asarray([test[i][0]]))
    y = model(x)

    return np.ndarray.tolist(y.data.argmax(axis=1))
