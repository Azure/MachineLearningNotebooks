import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

LABELS_FILENAME = 'labels.txt'

import shutil
from os import path

def check_labelmatch(images, annotations):
    data_dir_images = os.path.split(images[0])[0]
    data_dir_annot =  os.path.split(annotations[0])[0]

    im_files = {os.path.splitext(os.path.split(f)[1])[0] for f in images}
    annot_files = {os.path.splitext(os.path.split(f)[1])[0] for f in annotations}

    extra_ims = im_files.difference(annot_files)
    extra_annots = annot_files.difference(im_files)
    mismatch = len(extra_ims) > 0 or len(extra_annots) > 0

    if mismatch:
        print(f"The following files will be removed from the training process:")

    if len(extra_ims) > 0:
        print(f"images without annotations: {extra_ims}")

    if len(extra_annots) > 0:
        print(f"annotations without images: {extra_annots}")

    if not mismatch:    
        print(str(len(images)) + ' images found and ' + str(len(annotations)) + ' matching annotations found.'  )
        return (images, annotations)
    
    im_files = im_files.difference(extra_ims)
    annot_files = annot_files.difference(extra_annots)

    im_files = [os.path.join(data_dir_images, f+".jpg") for f in im_files]
    annot_files = [os.path.join(data_dir_annot, f+".xml") for f in annot_files]

    return(im_files, annot_files)

def create_dir(path):
    try:
        path_annotations = path + '/Annotations'
        path_images = path + '/JPEGImages'
        path_tfrec = path + '/TFreccords'
        
        os.makedirs(path_annotations)
        os.makedirs(path_images)
        os.makedirs(path_tfrec)
        
    except OSError:
        print("Creation of folders in directory %s failed.  Folders may already exist." % path)
    else:
        print("Successfully created Annotations, JPEGImages, and TFreccords folders at %s" % path)

    print('Please copy your annotation and image files to the Annotations and JPEGImages folders before moving to the next step')

        
def move_images(data_dir, train_images, train_annotations,
               test_images, test_annotations):

    source = data_dir + '/'       
    
    for image in train_images:
        image = data_dir + '/' + image
        dst = source + 'train/' + '/JPEGImages'
        
        if path.exists(image):
            shutil.copy(image, dst)            
            
    for image in test_images:
        image = data_dir + '/' + image
        dst = source + 'test/' + '/JPEGImages'
        
        if path.exists(image):
            shutil.copy(image, dst)            
            
    for annot in train_annotations:
        annot = data_dir + '/' + annot
        dst = source + 'train/' + '/Annotations'
        
        if path.exists(annot):
            shutil.copy(annot, dst)            
            
    for annot in test_annotations:
        annot = data_dir + '/' + annot
        dst = source + 'test/' + '/Annotations'
        
        if path.exists(annot):
            shutil.copy(annot, dst)   
    
    print('Images and annotations have been copied to directories: ' + source + 'train' + ' and ' + source + 'test')
 
def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))