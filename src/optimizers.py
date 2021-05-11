"""
Initialize GAN optimizers.
"""

import tensorflow as tf


def get_optimizers(
    lr_g, beta1_g, beta2_g, lr_d, beta1_d, beta2_d,
):
    """
    Return optimizer objects for generator and discriminator.
    Note the tf.Variable usage, this is to make Adam restore correctly when
    using tf checkpoints.

    Args:
        lr_g: Adam step size (generator)
        beta1_g: Adam beta1 (generator)
        beta2_g: Adam beta2 (generator)
        lr_d: Adam step size (discriminator)
        beta1_d: Adam beta1 (discriminator)
        beta2_d: Adam beta2 (discriminator)
    """

    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.Variable(lr_g),
        beta_1=tf.Variable(beta1_g),
        beta_2=tf.Variable(beta2_g),
        epsilon=tf.Variable(1e-7),
    )
    generator_optimizer.iterations
    generator_optimizer.decay = tf.Variable(0.0)

    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.Variable(lr_d),
        beta_1=tf.Variable(beta1_d),
        beta_2=tf.Variable(beta2_d),
        epsilon=tf.Variable(1e-7),
    )
    discriminator_optimizer.iterations
    discriminator_optimizer.decay = tf.Variable(0.0)

    return generator_optimizer, discriminator_optimizer
