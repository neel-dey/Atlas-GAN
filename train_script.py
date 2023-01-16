"""Script to train Atlas-HQ.

CLI args:
    epochs: int
        Number of epochs to train for.
    batch_size: int
        Batch size for training. GPU memory typically only allows small batches
    dataset: str
        Dataset of interest. Currently one of {'dHCP', 'pHD'}
    name: str
        Name of experiment. Will be prepended to saved folders.
    d_train_steps: int
        Number of discriminator updates in each GAN cycle.
    g_train_steps: int
        Number of generator updates in each GAN cycle.
    lr_g: float
        Learning rate for generator.
    lr_d: float
        Learning rate for discriminator.
    beta1_g: float
        Adam beta1 parameter for the generator.
    beta2_g: float
        Adam beta2 parameter for the generator.
    beta1_d: float
        Adam beta1 parameter for the generator.
    beta2_d: float
        Adam beta2 parameter for the discriminator.
    unconditional: bool
        Whether to train conditional/unconditional templates.
    nonorm_reg: bool
        Whether to use instance normalization in registration branch.
        Nor used in the paper.
    oversample: bool
        Whether to oversample rare ages during training.
    d_snout: bool
        Whether to apply Spectral Norm to the last layer of the Discriminator.
    clip: bool
        Whether to clip the template background during training.
    reg_loss: str
        Type of registration loss. One of {'NCC', 'NonSquareNCC'}.
        NonSquareNCC not used in paper.
    losswt_reg: float
        Multiplier for deformation regularizers.
    losswt_gan: float
        GAN loss weight in generator loss.
    losswt_tv: float
        Weight of TV penalty on generated templates.
        Not used in paper.
    losswt_gp: float
        Gradient penalty for discriminator loss.
    gen_config: str
        Template generator architecture. One of {'ours', 'voxelmorph'}.
    steps_per_epoch: int
        Number of steps per epoch.
    rng_seed: int
        Seed for random number generators.
    start_step: int
        Step to activate GAN training (as opposed to just registration).
        Not used in paper. GAN training is active from the first iteration.
    resume_ckpt: int
        If >0 then resume training from given ckpt index
    g_ch: int
        Channel width multiplier for generator.
    d_ch: int
        Channel width multiplier for discriminator.
    init: str
        Weight initialization. One of {'default', 'orthogonal'}.
    lazy_reg: int
        Calculate/apply gradient penalty only once every lazy_reg iterations.
        Not used in the paper.
"""

import numpy as np
import tensorflow as tf
import os
import datetime
import time
import glob
import random
import argparse

from numpy.random import seed
from tensorflow.compat.v1 import set_random_seed

from src.networks import Generator, Discriminator
from src.losses import generator_loss, discriminator_loss
from src.data_generators import D_data_generator, G_data_generator
from src.discriminator_augmentations import disc_augment
from src.optimizers import get_optimizers

# ----------------------------------------------------------------------------
# Set up CLI arguments:
# TODO: replace with a config json. CLI is unmanageably large now.
# TODO: add option for type of discriminator augmentation.

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--dataset', type=str, default='dHCP')
parser.add_argument('--name', type=str, default='experiment_name')
parser.add_argument('--d_train_steps', type=int, default=1)
parser.add_argument('--g_train_steps', type=int, default=1)
parser.add_argument('--lr_g', type=float, default=1e-4)
parser.add_argument('--lr_d', type=float, default=3e-4)
parser.add_argument('--beta1_g', type=float, default=0.0)
parser.add_argument('--beta2_g', type=float, default=0.9)
parser.add_argument('--beta1_d', type=float, default=0.0)
parser.add_argument('--beta2_d', type=float, default=0.9)
parser.add_argument(
    '--unconditional', dest='conditional', default=True, action='store_false',
)
parser.add_argument(
    '--nonorm_reg', dest='norm_reg', default=True, action='store_false',
)
parser.add_argument(
    '--oversample', dest='oversample', default=True, action='store_false',
)
parser.add_argument(
    '--d_snout', dest='d_snout', default=False, action='store_true',
)
parser.add_argument(
    '--clip', dest='clip_bckgnd', default=False, action='store_true',
)
parser.add_argument('--reg_loss', type=str, default='NCC')
parser.add_argument('--losswt_reg', type=float, default=1.0)
parser.add_argument('--losswt_gan', type=float, default=0.1)
parser.add_argument('--losswt_tv', type=float, default=0.00)
parser.add_argument('--losswt_gp', type=float, default=1e-3)
parser.add_argument('--gen_config', type=str, default='ours')
parser.add_argument('--steps_per_epoch', type=int, default=1000)
parser.add_argument('--rng_seed', type=int, default=33)
parser.add_argument('--start_step', type=int, default=0)
parser.add_argument('--resume_ckpt', type=int, default=0)
parser.add_argument('--g_ch', type=int, default=32)
parser.add_argument('--d_ch', type=int, default=64)
parser.add_argument('--init', type=str, default='default')
parser.add_argument('--lazy_reg', type=int, default=1)

args = parser.parse_args()

# Get CLI information:
epochs = args.epochs
batch_size = args.batch_size
dataset = args.dataset
exp_name = args.name
lr_g = args.lr_g
lr_d = args.lr_d
beta1_g = args.beta1_g
beta2_g = args.beta2_g
beta1_d = args.beta1_d
beta2_d = args.beta2_d
conditional = args.conditional
reg_loss = args.reg_loss
norm_reg = args.norm_reg
oversample = args.oversample
atlas_model = args.gen_config
steps = args.steps_per_epoch
lambda_gan = args.losswt_gan
lambda_reg = args.losswt_reg
lambda_tv = args.losswt_tv
lambda_gp = args.losswt_gp
g_loss_wts = [lambda_gan, lambda_reg, lambda_tv]
start_step = args.start_step
rng_seed = args.rng_seed
resume_ckpt = args.resume_ckpt
d_snout = args.d_snout
clip_bckgnd = args.clip_bckgnd
g_ch = args.g_ch
d_ch = args.d_ch
init = args.init
lazy_reg = args.lazy_reg

# Folder name --> save_folder:
save_folder = (
    ('{}_dataset_{}_eps{}_Gconfig_{}_normreg_{}_lrg{}_lrd{}_cond_{}_'
     'regloss_{}_lbdgan_{}_lbdreg_{}_lbdtv_{}_lbdgp_{}_dsnout_{}_start_{}')
    .format(exp_name, dataset, epochs, atlas_model, norm_reg, lr_g, lr_d,
            conditional, reg_loss, lambda_gan, lambda_reg, lambda_tv,
            lambda_gp, d_snout, start_step)
)

# Append to save_folder if using clip or lazy_reg settings:
if clip_bckgnd:
    save_folder = save_folder + '_clip_{}'.format(clip_bckgnd)

if lazy_reg > 1:
    save_folder = save_folder + '_lazy_{}'.format(lazy_reg)

# ----------------------------------------------------------------------------
# Set RNG seeds

seed(rng_seed)
set_random_seed(rng_seed)
random.seed(rng_seed)

# ----------------------------------------------------------------------------
# Initialize data generators

# Change these if working with new dataset:
if dataset == 'dHCP':
    fpath = './data/dHCP2/npz_files/T2/train/*.npz'
    avg_path = (
        './data/dHCP2/npz_files/T2/linearaverage_100T2_train.npz'
    )
    n_condns = 1

elif dataset == 'pHD':
    fpath = './data/predict-hd/npz_files/train_npz/*.npz'
    avg_path = './data/predict-hd/linearaverageof100.npz'
    n_condns = 3

else:
    raise ValueError('dataset expected to be dHCP or pHD')


img_paths = glob.glob(fpath)

Dtrain_data_generator = D_data_generator(
    vol_shape=(160, 192, 160),
    img_list=img_paths,
    oversample_age=oversample,
    batch_size=batch_size,
    dataset=dataset,
)

Gtrain_data_generator = G_data_generator(
    vol_shape=(160, 192, 160),
    img_list=img_paths,
    oversample_age=oversample,
    batch_size=batch_size,
    dataset=dataset,
)

avg_img = np.load(avg_path)['arr_0']  # TODO: make generic fname in npz

avg_batch = np.repeat(
    avg_img[np.newaxis, ...], batch_size, axis=0,
)[..., np.newaxis]


# ----------------------------------------------------------------------------
# Initialize networks

generator = Generator(
    ch=g_ch,
    atlas_model=atlas_model,
    conditional=conditional,
    normreg=norm_reg,
    clip_bckgnd=clip_bckgnd,
    initialization=init,
    n_condns=n_condns,
)

discriminator = Discriminator(
    ch=d_ch, conditional=conditional, sn_out=d_snout,
    initialization=init, n_condns=n_condns,
)


# If using vxm-style array of parameters for template, init. w/ linear average:
if atlas_model == 'voxelmorph' and conditional is False:
    if resume_ckpt == 0:
        if clip_bckgnd:
            # TODO: assign weights to this layer by name instead of index
            generator.layers[2].set_weights([avg_batch[0]])
        else:
            generator.layers[1].set_weights([avg_batch[0]])

# ----------------------------------------------------------------------------
# Set up optimizers

generator_optimizer, discriminator_optimizer = get_optimizers(
    lr_g, beta1_g, beta2_g, lr_d, beta1_d, beta2_d,
)

# ----------------------------------------------------------------------------
# Set up Checkpoints

checkpoint_dir = './training_checkpoints/{}/'.format(save_folder)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)

# If resuming training, restore checkpoint:
if resume_ckpt > 0:
    checkpoint.restore(
        './training_checkpoints/{}/ckpt-{}'.format(save_folder, resume_ckpt)
    ).assert_consumed()

# Set up folder for tensorboard logs:
summary_writer = tf.summary.create_file_writer(
    "logs/fit/" +
    '{}'.format(save_folder) +
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)


# ----------------------------------------------------------------------------
# Set up generator training loop

@tf.function
def get_inputs(unconditional_inputs, conditional_inputs):
    """If conditionally training, append condition tensor to network inputs."""
    if conditional:
        return unconditional_inputs + conditional_inputs
    else:
        return unconditional_inputs


@tf.function
def gen_train_step(input_images, avg_input, input_condns, epoch):
    """
    Generator training step. Args:
        input_images: tf tensor of training images.
        avg_input: tf tensor of linear average repeated 'batch_size' times.
        input_condns: tf tensor of input condns.
        epoch: tf tensor of training step.
    """
    with tf.GradientTape() as gen_tape:
        # Generator forward pass, get moved atlases, moving average of
        # displacements, generated atlases (sharp_atlas), and displacement:
        moved_atlases, disp_fields_ms, sharp_atlases, disp_fields = generator(
            get_inputs([input_images, avg_input], [input_condns]),
            training=True,
        )

        # Not used in paper. If pretraining model with registration-only:
        if epoch < start_step:
            d_logits_fake_local = tf.zeros((batch_size, 10, 12, 10, 1))
        else:
            d_logits_fake_local = discriminator(
                get_inputs([moved_atlases], [input_condns]),
                training=True,
            )

        # Get loss values. gen_tv_loss not used.
        (gen_total_loss, gen_gan_loss, gen_smoothness_loss, gen_mag_loss,
         gen_sim_loss, gen_moving_mag_loss, gen_tv_loss) = generator_loss(
            d_logits_fake_local,
            disp_fields_ms,
            disp_fields,
            moved_atlases,
            input_images,
            epoch,
            sharp_atlases,
            g_loss_wts,
            start_step=start_step,
            reg_loss_type=reg_loss,
        )

    # Get gradients:
    generator_gradients = gen_tape.gradient(
        gen_total_loss,
        generator.trainable_variables,
    )

    # Update model:
    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables),
    )

    # Tensorboard logging:
    tb_scalar = {
        'total_losses/gen_total_loss': gen_total_loss,
        'gan_losses/gen_gan_loss': gen_gan_loss,
        'regularizers/gen_smooth_loss': gen_smoothness_loss,
        'regularizers/gen_mag_loss': gen_mag_loss,
        'regularizers/gen_movmag_loss': gen_moving_mag_loss,
        'regularizers/gen_tv_loss': gen_tv_loss,  # Not used in paper
        'registration_losses/gen_sim_loss': gen_sim_loss,
    }

    tbscalarnames = list(tb_scalar.keys())

    atlasmax = tf.reduce_max(sharp_atlases)
    movedmax = tf.reduce_max(moved_atlases)

    tb_img = {
        'atlas/1': tf.nn.relu(sharp_atlases[:, 80, :, :, :]) / atlasmax,
        'atlas/2': tf.nn.relu(sharp_atlases[:, :, 96, :, :]) / atlasmax,
        'atlas/3': tf.nn.relu(sharp_atlases[:, :, :, 70, :]) / atlasmax,
        'moved_atlases/1': tf.nn.relu(moved_atlases[:, 80, :, :, :])/movedmax,
        'moved_atlases/2': tf.nn.relu(moved_atlases[:, :, 96, :, :])/movedmax,
        'moved_atlases/3': tf.nn.relu(moved_atlases[:, :, :, 70, :])/movedmax,
        'target_images/1':
            tf.image.convert_image_dtype(
                input_images[:, 80, :, :, :], dtype=tf.uint8,
            ),
        'target_images/2':
            tf.image.convert_image_dtype(
                input_images[:, :, 96, :, :], dtype=tf.uint8,
            ),
        'target_images/3':
            tf.image.convert_image_dtype(
                input_images[:, :, :, 70, :], dtype=tf.uint8,
            ),
    }

    tbimg_names = list(tb_img.keys())

    # Update tensorboard every 10 steps:
    if (epoch % 10) == 0:
        with summary_writer.as_default():
            for i in range(len(tbscalarnames)):
                tf.summary.scalar(
                    tbscalarnames[i], tb_scalar[tbscalarnames[i]], step=epoch,
                )

            for i in range(len(tbimg_names)):
                tf.summary.image(
                    tbimg_names[i], tb_img[tbimg_names[i]], step=epoch,
                )


# ----------------------------------------------------------------------------
# Set up discriminator training loop

@tf.function
def disc_train_step(
    input_images, avg_input, real_images, input_condns, real_condns, epoch,
):
    """
    Discriminator training step. Args:
        input_images: tf tensor of training images for template branch.
        avg_input: tf tensor of linear average repeated 'batch_size' times.
        input_images: tf tensor of training images for discriminator.
        input_condns: tf tensor of input condns for template branch.
        real_condns: tf tensor of input condns for discriminator.
        epoch: tf tensor of training step.
    """
    # Reorient image for more augs. Pick a flip (subset of D_4h group):
    # real_choice = tf.random.uniform((1,), 0, 4, dtype=tf.int32)
    # fake_choice = tf.random.uniform((1,), 0, 4, dtype=tf.int32)

    with tf.GradientTape() as disc_tape:
        # Generator forward pass:
        moved_atlases, _, _, _ = generator(
            get_inputs([input_images, avg_input], [input_condns]),
            training=True,
        )

        # Discriminator augmentation sequence on both fakes and reals:
        moved_atlases = disc_augment(
            moved_atlases, intensity_mods=False,
        )
        real_images = disc_augment(
            real_images, intensity_mods=False,
        )

        # Discriminator forward passes:
        d_logits_real_local = discriminator(
            get_inputs([real_images], [real_condns]),
            training=True,
        )
        d_logits_fake_local = discriminator(
            get_inputs([moved_atlases], [input_condns]),
            training=True,
        )

        # Get loss:
        disc_loss = discriminator_loss(
            d_logits_real_local,
            d_logits_fake_local,
        )

        # Get R1 gradient penalty from Mescheder, et al 2017:
        # Gradient penalty inside gradient with tf.function leads to lots of
        # if/else blocks for the tf2 graph.
        if lambda_gp > 0.0:
            # Every "lazy_reg" iterations compute the R1 gradient penalty:
            if (epoch % lazy_reg) == 0:
                new_real_batch = 1.0 * real_images
                new_label = 1.0 * real_condns
                with tf.GradientTape(persistent=True) as gp_tape:
                    gp_tape.watch(new_real_batch)
                    d_logits_real_local_new = discriminator(
                        get_inputs([new_real_batch], [new_label]),
                        training=True,
                    )

                grad = gp_tape.gradient(
                    d_logits_real_local_new, new_real_batch,
                )
                grad_sqr = tf.math.square(grad)

                grad_sqr_sum = tf.reduce_sum(
                    grad_sqr,
                    axis=np.arange(1, len(grad_sqr.shape)),
                )

                gp = (lambda_gp/2.0) * tf.reduce_mean(grad_sqr_sum)
            else:
                gp = 0.0
        else:
            gp = 0.0

        # Total loss:
        total_disc_loss = disc_loss + gp

    discriminator_gradients = disc_tape.gradient(
        total_disc_loss,
        discriminator.trainable_variables,
    )

    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables),
    )

    if (epoch % 10) == 0:
        with summary_writer.as_default():
            tf.summary.scalar(
                'total_losses/total_disc_loss', total_disc_loss, step=epoch,
            )
            tf.summary.scalar(
                'gan_losses/disc_loss', disc_loss, step=epoch,
            )
            tf.summary.scalar(
                'regularizers/gp', gp, step=epoch,
            )


# ----------------------------------------------------------------------------
# Set up overall framework training loop


def fit(epochs, disc_train_steps=1, gen_train_steps=1):
    """
    Fit framework. Args:
        epochs: Number of epochs
        disc_train_steps: Number of discriminator updates in each GAN cycle.
        gen_train_steps: Number of discriminator updates in each GAN cycle.
    """
    for epoch in range(epochs):
        start = time.time()
        print("Epoch: ", epoch)

        # Train
        for n in range(steps):
            print('.', end='')
            if (n+1) % 100 == 0:
                print()

            if (n + epoch*steps) >= start_step:
                for _ in range(disc_train_steps):
                    (target_images, real_images,
                     target_condns, real_condns) = next(
                        iter(Dtrain_data_generator),
                    )

                    disc_train_step(
                        tf.convert_to_tensor(target_images, dtype=tf.float32),
                        tf.convert_to_tensor(avg_batch, dtype=tf.float32),
                        tf.convert_to_tensor(real_images, dtype=tf.float32),
                        tf.convert_to_tensor(target_condns, dtype=tf.float32),
                        tf.convert_to_tensor(real_condns, dtype=tf.float32),
                        tf.convert_to_tensor((n + epoch*steps), dtype=tf.int64),
                    )

            target_images, target_condns = next(iter(Gtrain_data_generator))

            for _ in range(gen_train_steps):
                gen_train_step(
                    tf.convert_to_tensor(target_images, dtype=tf.float32),
                    tf.convert_to_tensor(avg_batch, dtype=tf.float32),
                    tf.convert_to_tensor(target_condns, dtype=tf.float32),
                    tf.convert_to_tensor((n + epoch*steps), dtype=tf.int64),
                )

        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    print('Time taken for epoch {} is {} sec\n'.format(
            epoch + 1, time.time() - start,
        ),
    )
    checkpoint.save(file_prefix=checkpoint_prefix)  # Save checkpoint

# ----------------------------------------------------------------------------
# Begin training


fit(
    epochs,
    disc_train_steps=args.d_train_steps,
    gen_train_steps=args.g_train_steps,
)

