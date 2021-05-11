"""Adapted from
https://github.com/NifTK/NiftyNet/blob/
935bf4334cd00fa9f9d50f6a95ddcbfdde4031e0/niftynet/layer/linear_resize.py
"""

from __future__ import absolute_import, print_function

import tensorflow as tf
import tensorflow

from tensorflow.keras.layers import Layer


class TrilinearResizeLayer(Layer):
    """
    Resize 3D volumes using ``tf.image.resize_images``
    (without trainable parameters)
    """

    def __init__(self, size_3d, name='trilinear_resize'):
        """

        :param size_3d: 3-element integers set the output 3D spatial shape
        :param name: layer name string
        """
        self.size_3d = size_3d
        super(TrilinearResizeLayer, self).__init__(name=name)

    def call(self, input_tensor):
        """
        Computes trilinear interpolation using TF ``resize_images`` function.

        :param input_tensor: 3D volume, shape
            ``batch, X, Y, Z, Channels``
        :return: interpolated volume
        """

        b_size, x_size, y_size, z_size, c_size = \
            input_tensor.get_shape().as_list()
        x_size_new, y_size_new, z_size_new = self.size_3d

        # resize y-z
        squeeze_b_x = tf.reshape(
            input_tensor, [-1, y_size, z_size, c_size])
        resize_b_x = tf.image.resize(
            squeeze_b_x, [y_size_new, z_size_new])
        resume_b_x = tf.reshape(
            resize_b_x, [-1, x_size, y_size_new, z_size_new, c_size])

        # resize x-y
        #   first reorient
        reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
        #   squeeze and 2d resize
        squeeze_b_z = tf.reshape(
            reoriented, [-1, y_size_new, x_size, c_size])
        resize_b_z = tf.image.resize(
            squeeze_b_z, [y_size_new, x_size_new])
        resume_b_z = tf.reshape(
            resize_b_z, [-1, z_size_new, y_size_new, x_size_new, c_size])

        return tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
