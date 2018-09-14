from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import json
import base64
from io import BytesIO
from PIL import Image

##############################################
# helper functions
##############################################


def build_model(x, y_, keep_prob):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv


def base64ToImg(base64ImgString):
    if base64ImgString.startswith('b\''):
        base64ImgString = base64ImgString[2:-1]
    base64Img = base64ImgString.encode('utf-8')
    decoded_img = base64.b64decode(base64Img)
    img_buffer = BytesIO(decoded_img)
    img = Image.open(img_buffer)
    return img

##############################################
# API init() and run() methods
##############################################


def init():
    global x, keep_prob, y_conv, sess
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        keep_prob = tf.placeholder(tf.float32)
        y_conv = build_model(x, y_, keep_prob)

        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

    model_dir = os.path.join('sample_projects', 'outputs')
    saved_model_path = os.path.join(model_dir, 'model.ckpt')

    sess = tf.Session(graph=g)
    sess.run(init_op)
    saver.restore(sess, saved_model_path)


def run(input_data):
    img = base64ToImg(json.loads(input_data)['data'])
    img_data = np.array(img, dtype=np.float32).flatten()
    img_data.resize((1, 784))

    y_pred = sess.run(y_conv, feed_dict={x: img_data, keep_prob: 1.0})
    predicted_label = np.argmax(y_pred[0])

    outJsonString = json.dumps({"label": str(predicted_label)})
    return str(outJsonString)
