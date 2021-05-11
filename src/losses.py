import tensorflow as tf
import sys

sys.path.append('./ext/voxelmorph/')
sys.path.append('./ext/neurite-master/')
sys.path.append('./ext/pynd-lib/')
sys.path.append('./ext/pytools-lib/')

from voxelmorph.tf.losses import Grad, NCC, NonSquareNCC

loss_object = tf.keras.losses.MeanSquaredError()  # used for GAN + def. reg.
loss_object_NCC = NCC(win=[9]*3)  # used for registration
loss_object_NonSquareNCC = NonSquareNCC(win=[9]*3)  # not used in paper


# ----------------------------------------------------------------------------
# Generator losses

@tf.function
def total_variation3D(ypred):
    """
    Not used in paper.
    Calculates anisotropic total variation for a 3D image ypred.
    """

    pixel_dif1 = ypred[:, 1:, :, :, :] - ypred[:, :-1, :, :, :]
    pixel_dif2 = ypred[:, :, 1:, :, :] - ypred[:, :, :-1, :, :]
    pixel_dif3 = ypred[:, :, :, 1:, :] - ypred[:, :, :, :-1, :]

    tot_var = (
            tf.reduce_mean(tf.math.abs(pixel_dif1)) +
            tf.reduce_mean(tf.math.abs(pixel_dif2)) +
            tf.reduce_mean(tf.math.abs(pixel_dif3))
    )
    return tf.reduce_mean(tot_var)


@tf.function
def generator_loss(
    disc_opinion_fake_local,
    disp_ms,
    disp,
    moved_atlases,
    fixed_images,
    epoch,
    sharp_atlases,
    loss_wts,
    start_step=0,
    reg_loss_type='NCC',
):
    """Loss function for Generator:
    Args:
        disc_opinion_fake_local: tf float
            Local feedback from discriminator.
        disp_ms: tf float
            Moving average of displacement fields.
        disp: tf float
            Displacement fields.
        moved_atlases: tf float
            Moved template images.
        fixed_images: tf float
            Target images.
        epoch: int
            Training step.
        sharp_atlases: tf float
            Generated Template image.
        loss_wts: list
            List of regularization weights for gan loss, deformation, and TV.
        start_step: int
            Training step to start training adversarial component.
    """

    lambda_gan, lambda_reg, lambda_tv = loss_wts

    # If training registration only, without GAN loss.
    # Need to do this, otherwise graph detaches:
    if epoch >= start_step:
        gan_loss = loss_object(
            tf.ones_like(disc_opinion_fake_local), disc_opinion_fake_local,
        )
        if lambda_tv > 0.0:  # never happens as TV loss not used in paper
            tv_loss = total_variation3D(sharp_atlases)
        else:
            tv_loss = 0.0
    else:
        gan_loss = 0.0
        tv_loss = 0.0

    # Similarity terms:
    if reg_loss_type == 'NCC':
        similarity_loss = tf.reduce_mean(
            loss_object_NCC.loss(moved_atlases, fixed_images),
        )
    elif reg_loss_type == 'NonSquareNCC':  # Not used in paper.
        similarity_loss = tf.reduce_mean(
            loss_object_NonSquareNCC.loss(moved_atlases, fixed_images),
        )

    # smoothness terms:
    smoothness_loss = tf.reduce_mean(
        Grad('l2').loss(tf.zeros_like(disp), disp),
    )

    # magnitude terms:
    magnitude_loss = loss_object(tf.zeros_like(disp), disp)
    moving_magnitude_loss = loss_object(tf.zeros_like(disp_ms), disp_ms)

    # Choose between registration only or reg+gan training:
    if epoch < start_step:
        total_gen_loss = (
            (lambda_reg * smoothness_loss) +
            (0.01 * lambda_reg * magnitude_loss) +
            (lambda_reg * moving_magnitude_loss) +
            1*similarity_loss
        )
    else:
        total_gen_loss = (
            lambda_gan*gan_loss +
            (lambda_reg * smoothness_loss) +
            (0.01 * lambda_reg * magnitude_loss) +
            (lambda_reg * moving_magnitude_loss) +
            1*similarity_loss +
            lambda_tv*tv_loss
        )

    return (
        total_gen_loss, gan_loss, smoothness_loss, magnitude_loss,
        similarity_loss, moving_magnitude_loss, tv_loss,
    )

# ----------------------------------------------------------------------------
# Discriminator losses


@tf.function
def discriminator_loss(
    disc_opinion_real_local,
    disc_opinion_fake_local,
):
    """Loss function for Generator:
    Args:
        disc_opinion_fake_local: tf float
            Local feedback from discriminator on moved templates.
        disc_opinion_real_local: tf float
            Local feedback from discriminator on real fixed images.
    """

    gan_fake_loss = loss_object(
            tf.zeros_like(disc_opinion_fake_local),
            disc_opinion_fake_local,
    )

    gan_real_loss = loss_object(
            tf.ones_like(disc_opinion_real_local),
            disc_opinion_real_local,
    )

    total_loss = 0.5*(gan_fake_loss + gan_real_loss)

    return total_loss
