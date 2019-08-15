import json
import logging
import os
import sys
from collections import deque

import cv2
import numpy as np
import requests
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)


def filter_detections_by_score(detections, score_cutoff):
    boxes, classes, scores = detections
    vindex = np.nonzero(scores > score_cutoff)
    vboxes = boxes[vindex]
    vscores = scores[vindex]
    vclasses = classes[vindex]
    new_detections = [vboxes, vclasses, vscores]
    return new_detections, vindex


def separate_detections_by_class(detections, num_class):
    vboxes, vclasses, vscores = detections
    # put data in list of dictionaries for filtering later
    boxlist = []
    for i in range(1, num_class+1):
        vin = np.flatnonzero(vclasses == i)
        if vin.size > 0:
            for j in range(vin.size):
                onebox = {'box': vboxes[vin[j]],
                          'score': vscores[vin[j]],
                          'class': vclasses[vin[j]],
                          'fromNN': 0}
                boxlist.append(onebox)
    return boxlist


def get_box_for_class(boxlist, class_idx):
    # get boxes for current class
    box_of_class = list(
        filter(lambda onebox: onebox['class'] == class_idx, boxlist))
    return box_of_class


def get_best_box(box_of_class):
    # Get the box with the best score
    if box_of_class:
        best_box_of_class = max(box_of_class, key=lambda x: x['score'])
    else:
        best_box_of_class = None
    return best_box_of_class


def img_post_process(detections, num_class):
    box_list = separate_detections_by_class(detections, num_class)
    # Find the best box for each class
    best_box_list = []
    for i in range(1, num_class+1):
        box_of_class = get_box_for_class(box_list, i)
        best_box_of_class = get_best_box(box_of_class)
        if best_box_of_class:
            best_box_list.append(best_box_of_class)
    return best_box_list


def bestbox_to_detections(best_box_list):
    boxes = []
    classes = []
    scores = []
    for bbox in best_box_list:
        boxes.append(bbox['box'])
        classes.append(bbox['class'])
        scores.append(bbox['score'])
    detections = [boxes, classes, scores]
    return detections


def append_data(data_out, detections):
    boxes, classes, scores = detections
    # convert to JSON serializable format
    boxes   = np.array(boxes, dtype = float).tolist()
    classes = np.array(classes, dtype = int).tolist()
    scores  = np.array(scores, dtype = float).tolist()
    
    data_out["bbox"].append(boxes)
    data_out["class"].append(classes)
    data_out["score"].append(scores)

    
def detect(args, comm):
    model_dir = args.model_dir
    content_dir = args.content_dir
    output_dir = args.output_dir
    num_class = args.num_class

    rank = comm.Get_rank()
    size = comm.Get_size()

    data_out = {
        "bbox": [],
        "class": [],
        "score": []
    }

    TENSORS, SESS = init_model(model_dir)

    filenames = os.listdir(content_dir)
    filenames = sorted(filenames)
    partition_size = len(filenames) // size
    partitioned_filenames = filenames[rank * partition_size: (rank + 1) * partition_size]
    print("RANK {} - is processing {} images out of the total {}".format(
        rank, len(partitioned_filenames),len(filenames)))

    for filename in partitioned_filenames:
        img_full_path = os.path.join(content_dir, filename)
        img_bgr = cv2.imread(img_full_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img_rgb, (480, 270))
        img = np.expand_dims(img, 0)
        score_cutoff = 0.9
        detections = eval_img(img, TENSORS, SESS)
        detections, _ = filter_detections_by_score(detections, score_cutoff)
        best_box_list = img_post_process(detections, num_class)
        detections = bestbox_to_detections(best_box_list)
        append_data(data_out, detections)

    output_full_path = os.path.join(output_dir, 'data.json')
    with open(output_full_path, 'w', encoding='utf-8') as f:
        json.dump(data_out, f, ensure_ascii=False, indent=4)


def get_key_tensors(detection_graph):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    tensors = [image_tensor, detection_boxes, detection_scores, detection_classes]
    return tensors


def init_model(model_dir):
    path_to_ckpt = os.path.join(model_dir, 'frozen_inference_graph.pb')
    detection_graph, sess = read_graph_from_ckpt(path_to_ckpt)
    tensors = get_key_tensors(detection_graph)
    return tensors, sess


def read_graph_from_ckpt(path_to_ckpt):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
    return detection_graph, sess


def eval_img(img, tensors, sess):
    image_tensor, detection_boxes, detection_scores, detection_classes = tensors
    (boxes, scores, classes) = sess.run(
        [detection_boxes, detection_scores, detection_classes],
        feed_dict={image_tensor: img})
    detections = [boxes, classes, scores]
    return detections
