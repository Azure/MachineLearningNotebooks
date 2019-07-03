import tensorflow as tf
import numpy as np

import os, sys, time, re

from anchors import generate_anchors
from model import ssd_common, ssd_vgg_300
from datautil.parser import get_parser_func
from datautil.ssd_vgg_preprocessing import preprocess_for_eval, preprocess_for_train
from tfutil import endpoints, tf_utils
import tfextended as tfe
from finetune.train_eval_base import TrainerBase

class TrainVggSsd(TrainerBase):
    '''
    Run fine-tuning
    Have training and validation recordset files
    '''

    def __init__(self, ckpt_dir, train_recordset_files, 
        steps_to_save = 1000, num_steps = 1000, num_classes = 21, 
        print_steps = 10, batch_size = 2,
        learning_rate = 1e-4, learning_rate_decay_steps=None, learning_rate_decay_value = None,
        adam_beta1 = 0.9, adam_beta2 = 0.999, adam_epsilon = 1e-8):

        '''
        ckpt_dir - directory of checkpoint metagraph
        train_recordset_files - list of files represetnting the recordset for training
        validation_recordset_files - list of files representing validation recordset
        '''

        super().__init__(ckpt_dir, train_recordset_files, steps_to_save, num_steps, num_classes, print_steps, batch_size)

        # optimizer parameters
        self.learning_rate = learning_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.learning_rate_decay_value = learning_rate_decay_value

        if self.learning_rate <= 0 \
            or (self.learning_rate_decay_value is not None and self.learning_rate_decay_value <= 0) \
            or (self.learning_rate_decay_steps is not None and self.learning_rate_decay_steps <= 0) \
            or (self.learning_rate_decay_steps is None and self.learning_rate_decay_value is not None) \
            or (self.learning_rate_decay_steps is not None and self.learning_rate_decay_value is None):
                raise ValueError("learning rate, learning rate steps, learning rate decay must be positive, \
                    learning decay steps and value must be both present or both absent")

        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def get_optimizer(self, learning_rate):
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=self.adam_beta1,
            beta2=self.adam_beta2,
            epsilon=self.adam_epsilon)
        return optimizer

    def get_learning_rate(self, global_step):
        '''
        Configure learning rate based on decay specifications
        '''
        if self.learning_rate_decay_steps is None:
            return tf.constant(self.learning_rate, name = 'fixed_learning_rate')
        else:
            return tf.train.exponential_decay(self.learning_rate, global_step, \
                self.learning_rate_decay_steps, self.learning_rate_decay_value, \
                staircase=True, name="exponential_decay_learning_rate")

    def train(self):

        tf.logging.set_verbosity(tf.logging.INFO)

        # shorthand
        sess = self.sess

        batch_data = self.iterator.get_next()

        # image, classes, scores, ground_truths are neatly packed into a flat list
        # this is how we will slice it to extract the data we need:
        # we will convert the flat list into a list of lists, where each sub-list
        # is as long as each slice dimension
        slice_shape = [1] + [len(self.anchors)] * 3

        b_image, b_classes, b_localizations, b_scores = tf_utils.reshape_list(batch_data, slice_shape)
        # network endpoints
        _, localizations, logits, bw_saver = self.get_output_tensors(b_image)

        variables_to_train = tf.trainable_variables()
        sess.run(tf.initialize_variables(variables_to_train))

        # add losses
        total_loss = self.ssd_net.losses(logits, localizations, b_classes, b_localizations, b_scores)
        tf.summary.scalar("total_loss", total_loss)
        
        global_step = tf.train.get_or_create_global_step()
        learning_rate = self.get_learning_rate(global_step)

        # configure learning rate now that we have the global step
        # add optimizer
        optimizer = self.get_optimizer(learning_rate)

        tf.summary.scalar("learning_rate", learning_rate)

        grads_and_vars = optimizer.compute_gradients(total_loss, var_list=variables_to_train)
        grad_updates = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # initialize all the variables we should initialize
        # weights will be restored right after
        sess.run(tf.global_variables_initializer())

        # after the first restore, we want global step in our checkpoint
        saver = tf.train.Saver(variables_to_train + [global_step])
        if self.latest_ckpt_step == 0:
            bw_saver.restore(sess, self.ckpt_file)
        else:
            saver.restore(sess, self.ckpt_file)
            self.ckpt_file = os.path.join(self.ckpt_dir, self.ckpt_prefix)

        # summaries
        train_summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.ckpt_dir + '/train', tf.get_default_graph())

        tf.logging.info(f"Starting training for {self.num_steps} steps")

        sess.run(self.iterator.initializer)

        # training loop
        start = time.time()

        for _ in range(self.num_steps):

            loss, _, cur_step, summary = sess.run([total_loss, grad_updates, global_step, train_summary_op])
            cur_step += 1

            if cur_step % self.print_steps == 0:

                print(f"{cur_step}: loss: {loss:.3f}, avg per step: {(time.time() - start) / self.print_steps:.3f} sec", end='\r', flush=True)
                train_writer.add_summary(summary, cur_step + 1)
                start = time.time()

            if cur_step % self.steps_to_save == 0:
                saver.save(sess, self.ckpt_file, global_step=global_step)
        print("\n")