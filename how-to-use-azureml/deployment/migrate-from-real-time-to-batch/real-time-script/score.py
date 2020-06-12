import json
import numpy as np
import os
import tensorflow as tf

from azureml.core.model import Model


def init():
    global X, output, sess
    tf.reset_default_graph()
    model_root = os.getenv("AZUREML_MODEL_DIR")
    # the name of the folder in which to look for tensorflow model files
    tf_model_folder = "model"
    saver = tf.train.import_meta_graph(os.path.join(model_root, tf_model_folder, "mnist-tf.model.meta"))
    X = tf.get_default_graph().get_tensor_by_name("network/X:0")
    output = tf.get_default_graph().get_tensor_by_name("network/output/MatMul:0")

    sess = tf.Session()
    saver.restore(sess, os.path.join(model_root, tf_model_folder, "mnist-tf.model"))


def run(raw_data):
    data = np.array(json.loads(raw_data)["data"])
    # make prediction
    out = output.eval(session=sess, feed_dict={X: data})
    y_hat = np.argmax(out, axis=1)
    return y_hat.tolist()
