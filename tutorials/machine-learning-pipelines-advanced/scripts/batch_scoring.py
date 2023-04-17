# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import argparse
import datetime
import time
import tensorflow.compat.v1 as tf
from math import ceil
import numpy as np
import sys
import shutil
import subprocess
import tf_slim

from azureml.core import Run
from azureml.core.model import Model
from azureml.core.dataset import Dataset
from tf_slim import nets

slim = tf_slim

image_size = 299
num_channel = 3


def get_class_label_dict(labels_dir):
    label = []
    labels_path = os.path.join(labels_dir, 'labels.txt')
    proto_as_ascii_lines = tf.gfile.GFile(labels_path).readlines()
    for temp in proto_as_ascii_lines:
        label.append(temp.rstrip())
    return label


def init():
    global g_tf_sess, probabilities, label_dict, input_images
    subprocess.run(["git", "clone", "https://github.com/tensorflow/models/"])
    sys.path.append("./models/research/slim")

    parser = argparse.ArgumentParser(description="Start a tensorflow model serving")
    parser.add_argument('--model_name', dest="model_name", required=True)
    parser.add_argument('--labels_dir', dest="labels_dir", required=True)
    args, _ = parser.parse_known_args()
    label_dict = get_class_label_dict(args.labels_dir)
    classes_num = len(label_dict)
    tf.disable_v2_behavior()
    with slim.arg_scope(nets.inception.inception_v3_arg_scope()):
        input_images = tf.placeholder(tf.float32, [1, image_size, image_size, num_channel])
        logits, _ = nets.inception.inception_v3(input_images,
                                                num_classes=classes_num,
                                                is_training=False)
        probabilities = tf.argmax(logits, 1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    g_tf_sess = tf.Session(config=config)
    g_tf_sess.run(tf.global_variables_initializer())
    g_tf_sess.run(tf.local_variables_initializer())

    model_path = Model.get_model_path(args.model_name)
    saver = tf.train.Saver()
    saver.restore(g_tf_sess, model_path)


def file_to_tensor(file_path):
    image_string = tf.read_file(file_path)
    image = tf.image.decode_image(image_string, channels=3)

    image.set_shape([None, None, None])
    image = tf.image.resize_images(image, [image_size, image_size])
    image = tf.divide(tf.subtract(image, [0]), [255])
    image.set_shape([image_size, image_size, num_channel])
    return image


def run(mini_batch):
    result_list = []
    for file_path in mini_batch:
        test_image = file_to_tensor(file_path)
        out = g_tf_sess.run(test_image)
        result = g_tf_sess.run(probabilities, feed_dict={input_images: [out]})
        result_list.append(os.path.basename(file_path) + ": " + label_dict[result[0]])
    return result_list
