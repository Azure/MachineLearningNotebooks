import tensorflow as tf
import numpy as np

import os, sys, time

from anchors import generate_anchors
from model import ssd_common, ssd_vgg_300
from datautil.parser import get_parser_func
from datautil.ssd_vgg_preprocessing import preprocess_for_eval, preprocess_for_train
from tfutil import endpoints, tf_utils
import tfextended as tfe
from finetune.train_eval_base import TrainerBase

class EvalVggSsd(TrainerBase):
    '''
    Run fine-tuning
    Have training and validation recordset files
    '''

    def __init__(self, ckpt_dir, validation_recordset_files, steps_to_save = 1000, num_steps = 1000, num_classes = 21, print_steps = 10):

        '''
        ckpt_dir - directory of checkpoint metagraph
        train_recordset_files - list of files represetnting the recordset for training
        validation_recordset_files - list of files representing validation recordset
        '''
        super().__init__(ckpt_dir, validation_recordset_files, steps_to_save, num_steps, num_classes, print_steps, 1, is_training=False)
        self.eval_classes = num_classes

    def get_eval_ops(self, b_labels, b_bboxes, predictions, localizations):
        '''
        Create evaluation operation
        '''
        b_difficults = tf.zeros(tf.shape(b_labels), dtype=tf.int64)

        # Performing post-processing on CPU: loop-intensive, usually more efficient.
        with tf.device('/device:CPU:0'):
            # Detected objects from SSD output.
            detected_localizations = self.ssd_net.bboxes_decode(localizations, self.anchors)

            rscores, rbboxes = \
                self.ssd_net.detected_bboxes(predictions, detected_localizations,
                                        select_threshold=0.01,
                                        nms_threshold=0.45,
                                        clipping_bbox=None,
                                        top_k=400,
                                        keep_top_k=20)

            # Compute TP and FP statistics.
            num_gbboxes, tp, fp, rscores = \
                tfe.bboxes_matching_batch(rscores.keys(), rscores, rbboxes,
                                          b_labels, b_bboxes, b_difficults,
                                          matching_threshold=0.5)

            # =================================================================== #
            # Evaluation metrics.
            # =================================================================== #
            dict_metrics = {}
            metrics_scope = 'ssd_metrics_scope'

            # First add all losses.
            for loss in tf.get_collection(tf.GraphKeys.LOSSES):
                dict_metrics[loss.op.name] = tf.metrics.mean(loss, name=metrics_scope)
            # Extra losses as well.
            for loss in tf.get_collection('EXTRA_LOSSES'):
                dict_metrics[loss.op.name] = tf.metrics.mean(loss, name=metrics_scope)

            # Add metrics to summaries and Print on screen.
            for name, metric in dict_metrics.items():
                # summary_name = 'eval/%s' % name
                summary_name = name
                tf.summary.scalar(summary_name, metric[0])

            # FP and TP metrics.
            tp_fp_metric = tfe.streaming_tp_fp_arrays(num_gbboxes, tp, fp, rscores, name=metrics_scope)

            for c in tp_fp_metric[0].keys():
                dict_metrics['tp_fp_%s' % c] = (tp_fp_metric[0][c],
                                                tp_fp_metric[1][c])

            # Add to summaries precision/recall values.
            aps_voc12 = {}
            # TODO: We cut it short by the actual number of classes we have
            for c in list(tp_fp_metric[0].keys())[:self.eval_classes - 1]:
                # Precison and recall values.
                prec, rec = tfe.precision_recall(*tp_fp_metric[0][c])

                # Average precision VOC12.
                v = tfe.average_precision_voc12(prec, rec)
                summary_name = 'AP_VOC12/%s' % c
                tf.summary.scalar(summary_name, v)

                aps_voc12[c] = v

            # Mean average precision VOC12.
            summary_name = 'AP_VOC12/mAP'
            mAP = tf.add_n(list(aps_voc12.values())) / len(aps_voc12)
            tf.summary.scalar(summary_name, mAP)

        names_to_values, names_to_updates = tf.contrib.metrics.aggregate_metric_map(dict_metrics)

        # Split into values and updates ops.
        return (names_to_values, names_to_updates, mAP)

    def eval(self):

        tf.logging.set_verbosity(tf.logging.INFO)

        # shorthand
        sess = self.sess

        sess.run(self.iterator.initializer)
        batch_data = self.iterator.get_next()

        # image, classes, scores, ground_truths are neatly packed into a flat list
        # this is how we will slice it to extract the data we need:
        # we will convert the flat list into a list of lists, where each sub-list
        # is as long as each slice dimension
        slice_shape = [1] * 3

        b_image, b_labels, b_bboxes = tf_utils.reshape_list(batch_data, slice_shape)
        
        # network endpoints
        predictions, localizations, _, _ = self.get_output_tensors(b_image)

        # branch to create evaluation operation
        _, names_to_updates, mAP = \
            self.get_eval_ops(b_labels, b_bboxes, predictions, localizations)

        eval_update_ops = tf_utils.reshape_list(list(names_to_updates.values()))

        # summaries
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()

        eval_writer = tf.summary.FileWriter(self.ckpt_dir + '/eval')

        # initialize globals
        sess.run(tf.global_variables_initializer())

        saver.restore(self.sess, self.ckpt_file)
        sess.run(tf.local_variables_initializer())

        tf.logging.info(f"Starting evaluation for {self.num_steps} steps")
        cur_step = self.latest_ckpt_step

        for step in range(self.num_steps):
            print(f"Evaluation step: {step + 1}", end='\r', flush=True)
            _, summary = sess.run([eval_update_ops, summary_op])

            if (step + 1) % self.print_steps == 0 or step == self.num_steps:
                eval_writer.add_summary(summary, cur_step + step + 1)

        summary_final, mAP_val = sess.run([summary_op, mAP])

        print(f"\nmAP: {mAP_val:.4f}")

        if (step + 1) % self.print_steps != 0:
            eval_writer.add_summary(summary_final, self.num_steps + cur_step)