# Copyright 2015 The TensorFlow Authors and Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Custom image operations.
Most of the following methods extend TensorFlow image library, and part of
the code is shameless copy-paste of the former!
"""
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables


# =========================================================================== #
# Modification of TensorFlow image routines.
# =========================================================================== #
def _assert(cond, ex_type, msg):
    """A polymorphic assert, works with tensors and boolean expressions.
    If `cond` is not a tensor, behave like an ordinary assert statement, except
    that a empty list is returned. If `cond` is a tensor, return a list
    containing a single TensorFlow assert op.
    Args:
      cond: Something evaluates to a boolean value. May be a tensor.
      ex_type: The exception class to use.
      msg: The error message.
    Returns:
      A list, containing at most one assert op.
    """
    if _is_tensor(cond):
        return [control_flow_ops.Assert(cond, [msg])]
    else:
        if not cond:
            raise ex_type(msg)
        else:
            return []


def _is_tensor(x):
    """Returns `True` if `x` is a symbolic tensor-like object.
    Args:
      x: A python object to check.
    Returns:
      `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.
    """
    return isinstance(x, (ops.Tensor, variables.Variable))


def _ImageDimensions(image):
    """Returns the dimensions of an image tensor.
    Args:
      image: A 3-D Tensor of shape `[height, width, channels]`.
    Returns:
      A list of `[height, width, channels]` corresponding to the dimensions of the
        input image.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(3).as_list()
        dynamic_shape = array_ops.unstack(array_ops.shape(image), 3)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def _Check3DImage(image, require_static=True):
    """Assert that we are working with properly shaped image.
    Args:
      image: 3-D Tensor of shape [height, width, channels]
        require_static: If `True`, requires that all dimensions of `image` are
        known and non-zero.
    Raises:
      ValueError: if `image.shape` is not a 3-vector.
    Returns:
      An empty list, if `image` has fully defined dimensions. Otherwise, a list
        containing an assert op is returned.
    """
    try:
        image_shape = image.get_shape().with_rank(3)
    except ValueError:
        raise ValueError("'image' must be three-dimensional.")
    if require_static and not image_shape.is_fully_defined():
        raise ValueError("'image' must be fully defined.")
    if any(x == 0 for x in image_shape):
        raise ValueError("all dims of 'image.shape' must be > 0: %s" %
                         image_shape)
    if not image_shape.is_fully_defined():
        return [check_ops.assert_positive(array_ops.shape(image),
                                          ["all dims of 'image.shape' "
                                           "must be > 0."])]
    else:
        return []


def fix_image_flip_shape(image, result):
    """Set the shape to 3 dimensional if we don't know anything else.
    Args:
      image: original image size
      result: flipped or transformed image
    Returns:
      An image whose shape is at least None,None,None.
    """
    image_shape = image.get_shape()
    if image_shape == tensor_shape.unknown_shape():
        result.set_shape([None, None, None])
    else:
        result.set_shape(image_shape)
    return result


# =========================================================================== #
# Image + BBoxes methods: cropping, resizing, flipping, ...
# =========================================================================== #
def bboxes_crop_or_pad(bboxes,
                       height, width,
                       offset_y, offset_x,
                       target_height, target_width):
    """Adapt bounding boxes to crop or pad operations.
    Coordinates are always supposed to be relative to the image.

    Arguments:
      bboxes: Tensor Nx4 with bboxes coordinates [y_min, x_min, y_max, x_max];
      height, width: Original image dimension;
      offset_y, offset_x: Offset to apply,
        negative if cropping, positive if padding;
      target_height, target_width: Target dimension after cropping / padding.
    """
    with tf.name_scope('bboxes_crop_or_pad'):
        # Rescale bounding boxes in pixels.
        scale = tf.cast(tf.stack([height, width, height, width]), bboxes.dtype)
        bboxes = bboxes * scale
        # Add offset.
        offset = tf.cast(tf.stack([offset_y, offset_x, offset_y, offset_x]), bboxes.dtype)
        bboxes = bboxes + offset
        # Rescale to target dimension.
        scale = tf.cast(tf.stack([target_height, target_width,
                                  target_height, target_width]), bboxes.dtype)
        bboxes = bboxes / scale
        return bboxes


def resize_image_bboxes_with_crop_or_pad(image, bboxes,
                                         target_height, target_width):
    """Crops and/or pads an image to a target width and height.
    Resizes an image to a target width and height by either centrally
    cropping the image or padding it evenly with zeros.

    If `width` or `height` is greater than the specified `target_width` or
    `target_height` respectively, this op centrally crops along that dimension.
    If `width` or `height` is smaller than the specified `target_width` or
    `target_height` respectively, this op centrally pads with 0 along that
    dimension.
    Args:
      image: 3-D tensor of shape `[height, width, channels]`
      target_height: Target height.
      target_width: Target width.
    Raises:
      ValueError: if `target_height` or `target_width` are zero or negative.
    Returns:
      Cropped and/or padded image of shape
        `[target_height, target_width, channels]`
    """
    with tf.name_scope('resize_with_crop_or_pad'):
        image = ops.convert_to_tensor(image, name='image')

        assert_ops = []
        assert_ops += _Check3DImage(image, require_static=False)
        assert_ops += _assert(target_width > 0, ValueError,
                              'target_width must be > 0.')
        assert_ops += _assert(target_height > 0, ValueError,
                              'target_height must be > 0.')

        image = control_flow_ops.with_dependencies(assert_ops, image)
        # `crop_to_bounding_box` and `pad_to_bounding_box` have their own checks.
        # Make sure our checks come first, so that error messages are clearer.
        if _is_tensor(target_height):
            target_height = control_flow_ops.with_dependencies(
                assert_ops, target_height)
        if _is_tensor(target_width):
            target_width = control_flow_ops.with_dependencies(assert_ops, target_width)

        def max_(x, y):
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.maximum(x, y)
            else:
                return max(x, y)

        def min_(x, y):
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.minimum(x, y)
            else:
                return min(x, y)

        def equal_(x, y):
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.equal(x, y)
            else:
                return x == y

        height, width, _ = _ImageDimensions(image)
        width_diff = target_width - width
        offset_crop_width = max_(-width_diff // 2, 0)
        offset_pad_width = max_(width_diff // 2, 0)

        height_diff = target_height - height
        offset_crop_height = max_(-height_diff // 2, 0)
        offset_pad_height = max_(height_diff // 2, 0)

        # Maybe crop if needed.
        height_crop = min_(target_height, height)
        width_crop = min_(target_width, width)
        cropped = tf.image.crop_to_bounding_box(image, offset_crop_height, offset_crop_width,
                                                height_crop, width_crop)
        bboxes = bboxes_crop_or_pad(bboxes,
                                    height, width,
                                    -offset_crop_height, -offset_crop_width,
                                    height_crop, width_crop)
        # Maybe pad if needed.
        resized = tf.image.pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                               target_height, target_width)
        bboxes = bboxes_crop_or_pad(bboxes,
                                    height_crop, width_crop,
                                    offset_pad_height, offset_pad_width,
                                    target_height, target_width)

        # In theory all the checks below are redundant.
        if resized.get_shape().ndims is None:
            raise ValueError('resized contains no shape.')

        resized_height, resized_width, _ = _ImageDimensions(resized)

        assert_ops = []
        assert_ops += _assert(equal_(resized_height, target_height), ValueError,
                              'resized height is not correct.')
        assert_ops += _assert(equal_(resized_width, target_width), ValueError,
                              'resized width is not correct.')

        resized = control_flow_ops.with_dependencies(assert_ops, resized)
        return resized, bboxes


def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    """Resize an image and bounding boxes.
    """
    # Resize image.
    with tf.name_scope('resize_image'):
        height, width, channels = _ImageDimensions(image)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size,
                                       method, align_corners)
        image = tf.reshape(image, tf.stack([size[0], size[1], channels]))
        return image


def random_flip_left_right(image, bboxes, seed=None):
    """Random flip left-right of an image and its bounding boxes.
    """
    def flip_bboxes(bboxes):
        """Flip bounding boxes coordinates.
        """
        bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                           bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
        return bboxes

    # Random flip. Tensorflow implementation.
    with tf.name_scope('random_flip_left_right'):
        image = ops.convert_to_tensor(image, name='image')
        _Check3DImage(image, require_static=False)
        uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
        mirror_cond = math_ops.less(uniform_random, .5)
        # Flip image.
        result = control_flow_ops.cond(mirror_cond,
                                       lambda: array_ops.reverse_v2(image, [1]),
                                       lambda: image)
        # Flip bboxes.
        bboxes = control_flow_ops.cond(mirror_cond,
                                       lambda: flip_bboxes(bboxes),
                                       lambda: bboxes)
        return fix_image_flip_shape(image, result), bboxes

