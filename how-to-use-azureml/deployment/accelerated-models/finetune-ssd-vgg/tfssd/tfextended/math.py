# Copyright 2017 Paul Balanca. All Rights Reserved.
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
"""TF Extended: additional math functions.
"""
import tensorflow as tf

from tensorflow.python.framework import ops

def safe_divide(numerator, denominator, name):
    """Divides two values, returning 0 if the denominator is <= 0.
    Args:
      numerator: A real `Tensor`.
      denominator: A real `Tensor`, with dtype matching `numerator`.
      name: Name for the returned op.
    Returns:
      0 if `denominator` <= 0, else `numerator` / `denominator`
    """
    return tf.where(
        tf.greater(denominator, 0),
        tf.divide(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)


def cummax(x, reverse=False, name=None):
    """Compute the cumulative maximum of the tensor `x` along `axis`. This
    operation is similar to the more classic `cumsum`. Only support 1D Tensor
    for now.

    Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
       `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
       `complex128`, `qint8`, `quint8`, `qint32`, `half`.
       axis: A `Tensor` of type `int32` (default: 0).
       reverse: A `bool` (default: False).
       name: A name for the operation (optional).
    Returns:
    A `Tensor`. Has the same type as `x`.
    """
    with ops.name_scope(name, "Cummax", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        # Not very optimal: should directly integrate reverse into tf.scan.
        if reverse:
            x = tf.reverse(x, axis=[0])
        # 'Accumlating' maximum: ensure it is always increasing.
        cmax = tf.scan(tf.maximum, x,
                       initializer=None, parallel_iterations=1,
                       back_prop=False, swap_memory=False)
        if reverse:
            cmax = tf.reverse(cmax, axis=[0])
        return cmax
