"""
Implementation of FiLM layers with layout inspired by:
https://github.com/jacenkow/inside/blob/master/inside/layers.py

Minor modification: we use `(1 + gamma)*x + beta` instead of `gamma*x + beta`
as gamma is initialized near zero in common DL set ups and modify it for 3D.

# TODO: Make generic for 2D/3D
"""


import tensorflow as tf
import tensorflow.keras.layers as KL


def _film_reshape(gamma, beta, x):
    """Reshape gamma and beta for FiLM."""

    gamma = tf.tile(
        tf.reshape(gamma, (tf.shape(gamma)[0], 1, 1, 1, tf.shape(gamma)[-1])),
        (1, tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], 1))
    beta = tf.tile(
        tf.reshape(beta, (tf.shape(beta)[0], 1, 1, 1, tf.shape(beta)[-1])),
        (1, tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], 1))

    return gamma, beta


class FiLM(KL.Layer):
    """General Conditioning Layer (FiLM) by (Perez et al., 2017)."""
    def __init__(self, init='default', wt_decay=1e-5):
        """
        Args:
            init: str
                How to initialize dense layers.
            wt_decay: float
                L2 penalty on FiLM projection.
        """
        if init == 'orthogonal':
            self.init = 'orthogonal'
        elif init == 'default' or init is None:
            self.init = None
        else:
            raise ValueError

        self.wt_decay = wt_decay

        super().__init__()

    def build(self, input_shape):
        self.channels = input_shape[0][-1]  # input_shape: [x, z].

        self.fc = KL.Dense(
            int(2 * self.channels),
            kernel_initializer=self.init,
            kernel_regularizer=tf.keras.regularizers.l2(
                l=self.wt_decay,
            ),
            bias_regularizer=tf.keras.regularizers.l2(
                l=self.wt_decay,
            ),
        )

    def call(self, inputs):
        x, z = inputs
        gamma, beta = self.hypernetwork(z)
        gamma, beta = _film_reshape(gamma, beta, x)

        return (1. + gamma) * x + beta

    def hypernetwork(self, inputs):
        x = self.fc(inputs)

        return x[..., :self.channels], x[..., self.channels:]
