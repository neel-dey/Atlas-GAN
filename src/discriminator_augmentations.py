"""
Functions for Differentiable Discriminator Augmentation.

Augmentation functions taken and modified for 3D neuroimages from
github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_tf.py

"""

import tensorflow as tf


@tf.function
def disc_augment(image_batch, intensity_mods=False):
    """
    Return augmented training arrays. Args:
        image_batch: tf tensor of batch to augment
        # flip_choice: tf tensor containing (int) choice of flip augmentation.
        intensity_mods: bool indicating whether to use intensity augmentation.
    """

    # 50% of the time flip along axis 1
    if tf.random.uniform((1,)) > 0.5:
        image_batch = tf.reverse(image_batch, axis=[1])
    # 50% of the time flip along axis 2
    if tf.random.uniform((1,)) > 0.5:
        image_batch = tf.reverse(image_batch, axis=[2])
    # 50% of the time flip along axis 3
    if tf.random.uniform((1,)) > 0.5:
        image_batch = tf.reverse(image_batch, axis=[3])

    # # 50% of the time do random left-right reflections:
    # if tf.random.uniform((1,)) > 0.5:
    #     image_batch = tf.reverse(image_batch, axis=[3])
    #
    # # Other flips
    # # TODO: figure out a cleaner way of doing this
    # if flip_choice == 0:
    #     pass
    # elif flip_choice == 1:
    #     image_batch = tf.reverse(image_batch, axis=[1])
    # elif flip_choice == 2:
    #     image_batch = tf.reverse(image_batch, axis=[2])
    # elif flip_choice == 3:
    #     image_batch = tf.reverse(tf.reverse(image_batch, axis=[2]), axis=[1])

    # Random Intensity:
    if intensity_mods:
        image_batch = rand_brightness(image_batch)
        image_batch = rand_saturation(image_batch)
        image_batch = rand_contrast(image_batch)

    # Random Translation:
    image_batch = rand_translation(image_batch)

    return image_batch


@tf.function
def rand_brightness(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1, 1]) - 0.5
    x = x + magnitude
    return x


@tf.function
def rand_saturation(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1, 1]) * 2
    x_mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


@tf.function
def rand_contrast(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1, 1]) + 0.5
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3, 4], keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


@tf.function
def rand_translation(x, ratio=0.05):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:4]
    shift = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)

    translation_x = tf.random.uniform(
        [batch_size, 1], -shift[0], shift[0] + 1, dtype=tf.int32,
    )
    translation_y = tf.random.uniform(
        [batch_size, 1], -shift[1], shift[1] + 1, dtype=tf.int32,
    )
    translation_z = tf.random.uniform(
        [batch_size, 1], -shift[2], shift[2] + 1, dtype=tf.int32,
    )

    grid_x = tf.clip_by_value(
        (tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) +
         translation_x + 1),
        0,
        image_size[0] + 1,
    )
    grid_y = tf.clip_by_value(
        (tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) +
         translation_y + 1),
        0,
        image_size[1] + 1,
    )
    grid_z = tf.clip_by_value(
        (tf.expand_dims(tf.range(image_size[2], dtype=tf.int32), 0) +
         translation_z + 1),
        0,
        image_size[2] + 1,
    )

    x = tf.gather_nd(
        tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]]),
        tf.expand_dims(grid_x, -1),
        batch_dims=1,
    )
    x = tf.transpose(
        tf.gather_nd(
            tf.pad(
                tf.transpose(x, [0, 2, 1, 3, 4]),
                [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]],
            ),
            tf.expand_dims(grid_y, -1),
            batch_dims=1,
        ),
        [0, 2, 1, 3, 4],
    )
    x = tf.transpose(
        tf.gather_nd(
            tf.pad(
                tf.transpose(x, [0, 3, 2, 1, 4]),
                [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]],
            ),
            tf.expand_dims(grid_z, -1),
            batch_dims=1,
        ),
        [0, 3, 2, 1, 4],
    )

    return x
