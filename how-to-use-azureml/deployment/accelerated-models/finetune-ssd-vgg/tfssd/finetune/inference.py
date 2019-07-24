import tensorflow as tf

import os, time

from anchors import generate_anchors
from model import np_methods
from tfutil import endpoints, tf_utils
from datautil.ssd_vgg_preprocessing import preprocess_for_eval
import tfextended as tfe
from azureml.accel.models import SsdVgg

class InferVggSsd:
    '''
    Run fine-tuning
    Have training and validation recordset files
    '''

    def __init__(self, ckpt_dir, ckpt_file=None, gpu=True):

        '''
        ckpt_dir - directory of checkpoint metagraph
        '''

        if gpu:
            gpu_options = tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        else:
            config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})

        self.sess = tf.Session(config=config)

        ssd_net_graph = SsdVgg(ckpt_dir)
        self.ckpt_dir = ssd_net_graph.model_path
        
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

        # Evaluation pre-processing: resize to SSD net shape.
        image_pre, _, _, self.bbox_img = preprocess_for_eval(
            self.img_input, None, None, generate_anchors.img_shape, "NHWC")
        self.image_4d = tf.expand_dims(image_pre, 0)

        # import the graph
        ssd_net_graph.import_graph_def(self.image_4d, is_training=False)

        graph = tf.get_default_graph()
        self.localizations = [graph.get_tensor_by_name(tensor_name) for tensor_name in endpoints.localizations_names]
        self.predictions = [graph.get_tensor_by_name(tensor_name) for tensor_name in endpoints.predictions_names]

        # Restore SSD model.
        self.sess.run(tf.global_variables_initializer())

        if ckpt_file is None:
            ssd_net_graph.restore_weights(self.sess)
        else:
            saver = tf.train.Saver()
            saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_file))

        # SSD default anchor boxes.
        self.ssd_anchors = generate_anchors.ssd_anchors_all_layers()


    def close(self):
        self.sess.close()
        tf.reset_default_graph()

    def process_image(self, img, select_threshold=0.4, nms_threshold=.45, net_shape=(300, 300)):
        # Run SSD network.
        rpredictions, rlocalisations, rbbox_img = \
            self.sess.run([self.predictions, self.localizations, self.bbox_img],
                                                                feed_dict={self.img_input: img})
        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, self.ssd_anchors,
                select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
        
        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        return rclasses, rscores, rbboxes

    def infer(self, img, visualize):
        rclasses, rscores, rbboxes =  self.process_image(img)

        if visualize:
            from tfutil import visualization
            visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

        return rclasses, rscores, rbboxes

    def infer_file(self, im_file, visualize=False):
        import cv2

        img = cv2.imread(im_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.infer(img, visualize)