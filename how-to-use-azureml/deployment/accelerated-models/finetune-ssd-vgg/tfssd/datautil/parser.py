import tensorflow as tf
import numpy as np
import os

from datautil.ssd_vgg_preprocessing import preprocess_for_train, preprocess_for_eval
from model import ssd_common
from tfutil import tf_utils

features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
    'image/height': tf.FixedLenFeature([1], tf.int64),
    'image/width': tf.FixedLenFeature([1], tf.int64),
    'image/channels': tf.FixedLenFeature([1], tf.int64),
    'image/shape': tf.FixedLenFeature([3], tf.int64),
    'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
    'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
}


def get_parser_func(anchors, num_classes, is_training, var_scope):
    '''
    Dataset parser function for training and evaluation

    Arguments:
        preprocess_fn - function that does preprocesing
    '''
    
    preprocess_fn = preprocess_for_train if is_training else preprocess_for_eval

    def parse_tfrec_data(example_proto):
        with tf.variable_scope(var_scope):
            parsed_features = tf.parse_single_example(example_proto, features)
            
            image_string = parsed_features['image/encoded']
            image_decoded = tf.image.decode_jpeg(image_string)

            labels = tf.sparse.to_dense(parsed_features['image/object/bbox/label'])
            
            xmin = tf.sparse.to_dense(parsed_features['image/object/bbox/xmin'])
            xmax = tf.sparse.to_dense(parsed_features['image/object/bbox/xmax'])
            ymin = tf.sparse.to_dense(parsed_features['image/object/bbox/ymin'])
            ymax = tf.sparse.to_dense(parsed_features['image/object/bbox/ymax'])
            bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
            
            if is_training:
                image, labels, bboxes = preprocess_fn(image_decoded, labels, bboxes)
            else:
                image, labels, bboxes, _ = preprocess_fn(image_decoded, labels, bboxes)

        # ground truth encoding
        # each of the returns is a litst of tensors
        if is_training:
            classes, localisations, scores = \
                ssd_common.tf_ssd_bboxes_encode(labels, bboxes, anchors, num_classes)
            return tf_utils.reshape_list([image, classes, localisations, scores])
        else:
            return tf_utils.reshape_list([image, labels, bboxes])

    return parse_tfrec_data