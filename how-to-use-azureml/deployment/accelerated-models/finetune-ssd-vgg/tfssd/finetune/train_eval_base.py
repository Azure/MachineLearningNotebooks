import tensorflow as tf
import numpy as np

import os, sys, time, glob, re

from anchors import generate_anchors
from model import ssd_common, ssd_vgg_300
from datautil.parser import get_parser_func
from datautil.ssd_vgg_preprocessing import preprocess_for_eval, preprocess_for_train
from tfutil import endpoints, tf_utils
import tfextended as tfe
from azureml.accel.models import SsdVgg

slim = tf.contrib.slim

class TrainerBase:
    '''
    Run fine-tuning
    Have training and validation recordset files
    '''

    def __init__(self, ckpt_dir, recordset_files, 
        steps_to_save = 1000, num_steps = 1000, num_classes = 21, print_steps = 10, batch_size=2, is_training=True):

        '''
        ckpt_dir - directory of checkpoint metagraph
        recordset_files - list of files represetnting the recordset for training
        validation_recordset_files - list of files representing validation recordset
        '''
        
        self.is_training = is_training
        
        # This will pull the model with its weights
        # And seed the checkpoint
        self.ssd_net_graph = SsdVgg(ckpt_dir)
        self.ckpt_dir = self.ssd_net_graph.model_path
        self.ckpt_file = tf.train.latest_checkpoint(self.ssd_net_graph.model_path)

        try:
            self.latest_ckpt_step = int(re.findall("-[0-9]+$", self.ckpt_file)[0][1:])
        except:
            self.latest_ckpt_step = 0

        self.recordset = recordset_files
        self.ckpt_prefix = os.path.split(self.ssd_net_graph.model_ref + "_bw")[1]

        self.pb_graph_path = os.path.join(self.ckpt_dir, self.ckpt_prefix + ".graph.pb")
        #if self.is_training:
        self.graph_file = os.path.join(self.ckpt_dir, self.ckpt_prefix + ".meta")
        #else:
        #    self.graph_file = self.ckpt_file + ".meta"

        # anchors
        self.anchors = generate_anchors.ssd_anchors_all_layers()

        # shuffle
        self.n_shuffle = 1000
        self.num_steps = num_steps

        # num of classes
        # REVIEW: this has to be 21!
        self.num_classes = 21

        # initialize data pipeline
        self.batch_size = batch_size
        self.iterator = None
        self.prep_dataset_and_iterator()

        self.steps_to_save = steps_to_save

        self.print_steps = print_steps
        # for losses etc
        self.ssd_net = ssd_vgg_300.SSDNet()

        # input placeholder
        self.input_tensor_name = self.ssd_net_graph.input_tensor_list[0]

        
    def __enter__(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

        self.sess = tf.Session(config=config)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()
        tf.reset_default_graph()

    def prep_dataset_and_iterator(self):
        '''
        Create datasets for training or validation
        '''

        var_scope = "training" if self.is_training else "eval"

        parse_func = get_parser_func(self.anchors, self.num_classes, self.is_training, var_scope)

        with tf.variable_scope(var_scope):
            # data pipeline
            dataset = tf.data.TFRecordDataset(self.recordset)
            if self.is_training:
                dataset = dataset.shuffle(self.n_shuffle)
            dataset = dataset.map(parse_func)
            dataset = dataset.repeat()
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(1)

            self.iterator = dataset.make_initializable_iterator()

    def get_output_tensors(self, image):

        is_training = tf.constant(self.is_training, dtype=tf.bool, shape=())
        input_map = {self.input_tensor_name: image, "is_training": is_training}

        saver = tf.train.import_meta_graph(self.graph_file, input_map=input_map)
        graph = tf.get_default_graph()

        logits = [graph.get_tensor_by_name(tensor_name) for tensor_name in endpoints.logit_names]
        localizations = [graph.get_tensor_by_name(tensor_name) for tensor_name in endpoints.localizations_names]
        predictions = [graph.get_tensor_by_name(tensor_name) for tensor_name in endpoints.predictions_names]

        return predictions, localizations, logits, saver

