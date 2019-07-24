import tensorflow as tf

import os, time

from azureml.accel.models import SsdVgg
import azureml.accel.models.utils as utils

class SaverVggSsd:
    '''
    Run fine-tuning
    Have training and validation recordset files
    '''

    def __init__(self, ckpt_dir):

        '''
        ckpt_dir - directory of checkpoint metagraph
        '''

        config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})

        self.sess = tf.Session(config=config)

        ssd_net_graph = SsdVgg(ckpt_dir, is_frozen=True)
        self.ckpt_dir = ssd_net_graph.model_path

        self.in_images = tf.placeholder(tf.string)
        self.image_tensors = utils.preprocess_array(self.in_images, output_width=300, output_height=300, 
            preserve_aspect_ratio=False)

        self.output_tensors = ssd_net_graph.import_graph_def(self.image_tensors, is_training=False)

        self.output_names = ssd_net_graph.output_tensor_list
        self.input_name_str = self.in_images.name

        # Restore SSD model.
        ssd_net_graph.restore_weights(self.sess)


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.sess.close()
        tf.reset_default_graph()

    def save_for_deployment(self, saved_path):

        output_map = {'out_{}'.format(i): output for i, output in enumerate(self.output_tensors)}

        tf.saved_model.simple_save(self.sess, 
            saved_path, 
            inputs={"images": self.in_images},
            outputs=output_map)