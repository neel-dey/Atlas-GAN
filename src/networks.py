import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K

from tensorflow_addons.layers import SpectralNormalization
from tensorflow_addons.layers import InstanceNormalization

from .film import FiLM
from .trilinear import TrilinearResizeLayer

import sys

sys.path.append('./ext/voxelmorph/')
sys.path.append('./ext/neurite-master/')
sys.path.append('./ext/pynd-lib/')
sys.path.append('./ext/pytools-lib/')

from neurite.tf.layers import LocalParamWithInput
from neurite.tf.layers import MeanStream_old as MeanStream
from voxelmorph.tf.layers import SpatialTransformer, VecInt, RescaleTransform


# ----------------------------------------------------------------------------
# Common network block:

def conv_block(
    x_in,
    nf,
    condn_emb=None,
    mode='const',
    activation=True,
    sn=False,
    instancen=False,
    stride=1,
    kernel_size=3,
    init='default',
):
    """Convolution module including convolution followed by leakyrelu.
    Args:
        x_in: input feature map
        nf: number of filters
        condn_emb: if using FiLM, this is the condition embedding.
        mode: either upsample, downsample or leave at constant resolution.
        activation: bool indicating whether to use a leaky relu.
        sn: bool indicating whether to use spectral norm.
        instancen: bool for instance norm. Not used in paper.
        stride: convolutional stride
        kernel_size: kernel size
        init: weight initialization. either 'default' or 'orthogonal'
    """

    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims supported to up to 3. found: %d" % ndims

    maxpool = getattr(KL, 'MaxPooling%dD' % ndims)
    upsample = getattr(KL, 'UpSampling%dD' % ndims)
    conv = getattr(KL, 'Conv%dD' % ndims)

    # If unconditional and not using FiLM, then train conv layers with bias:
    if condn_emb is None:
        bias = True
    else:
        bias = False

    # Reflection pad:
    x_in = tf.pad(
        x_in,
        [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],
        "REFLECT",
    )

    # Specify initializations:
    if init == 'default' or init is None:
        initialization = None
    elif init == 'orthogonal':
        initialization = 'orthogonal'
    else:
        raise ValueError

    # If using spectral normalization:
    if sn:
        x_out = SpectralNormalization(conv(
            nf,
            kernel_size=kernel_size,
            padding='valid',
            use_bias=bias,
            strides=stride,
            kernel_initializer=initialization,
        ))(x_in)
    else:
        x_out = conv(
            nf,
            kernel_size=kernel_size,
            padding='valid',
            use_bias=bias,
            strides=stride,
            kernel_initializer=initialization,
        )(x_in)

    # If using instance normalization. Not used in paper.
    if instancen:
        x_out = InstanceNormalization(
            center=True,
            scale=True,
        )(x_out)

    # If using FiLM:
    if condn_emb is not None:
        x_out = FiLM(init=init)([x_out, condn_emb])

    # Nonlinearity:
    if activation:
        x_out = KL.LeakyReLU(0.2)(x_out)

    # Up/down sample:
    if mode == 'up':
        x_out = upsample()(x_out)
    elif mode == 'down':
        x_out = maxpool()(x_out)  # not used in paper, we used strided convs
    elif mode == 'const':
        pass
    else:
        raise ValueError('mode has to be up/down/const w.r.t. spatial res')

    return x_out


@tf.function
def const_inp(tensor):
    """Used to give a layer a constant input of 1."""
    batch_size = tf.shape(tensor)[0]
    constant = tf.constant(1.0)
    constant = tf.expand_dims(constant, axis=0)
    return tf.broadcast_to(constant, shape=(batch_size, 1))


# ----------------------------------------------------------------------------
# Generator architecture


def Generator(
    ch=32,
    full_size=False,
    conditional=True,
    normreg=False,
    atlas_model='ours',
    input_resolution=[160, 192, 160, 1],
    clip_bckgnd=True,
    initialization='default',
    n_condns=1,
):
    """
    Args:
        ch : int
            Channel multiplier.
        full_size : bool
            Flag indicating whether vel. fields estimated at half or full res.
        atlas_model : str
            One of {'ours', 'voxelmorph'}
        conditional : bool
            Flag indicating whether generator model is conditional.
        normreg : bool
            Flag indicating whether InstanceNorm is used. Not used in paper.
        input_resolution: list
            Input image dimensions.
        clip_bckgnd: bool
            Whether to use a foreground mask during training on templates.
        initialization: str
            Weight init. One of "default" or "orthogonal".
        n_condns: int
            Number of conditions if training conditionally.
    """
    image_inputs = tf.keras.layers.Input(shape=input_resolution)
    atlas_inputs = tf.keras.layers.Input(shape=input_resolution)

    if conditional:
        condn = tf.keras.layers.Input(shape=(n_condns,))

    # These are used in the template generation branch only for Atlas-HQ
    # Also for the registration sub-network.
    if initialization == 'orthogonal':
        init = 'orthogonal'
        vel_init = 'orthogonal'
    elif initialization == 'default':
        init = None
        vel_init = tf.keras.initializers.RandomNormal(
            mean=0.0,
            stddev=1e-5,
        )
    else:
        raise ValueError

    # Atlas sharpening branch:
    # TODO: Long if/else sequence, abstract out and make cleaner.
    # VXM archs taken verbatim from their repos
    if atlas_model == 'voxelmorph' and conditional is False:
        # vxm unconditional model:
        atlas_layer = LocalParamWithInput(
            name='atlas',
            shape=input_resolution,
            mult=1.0,
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=1e-7,
            ),
        )

        new_atlas = atlas_layer(atlas_inputs)

    elif atlas_model == 'voxelmorph' and conditional is True:
        # vxm conditional model:
        # TODO: fix hardcoding
        condn_emb_vxm = KL.Dense((80*96*80*8), activation='elu')(condn)
        condn_emb_vxm = KL.Reshape((80, 96, 80, 8))(condn_emb_vxm)

        # Atlas sharpening branch:
        dec_out = conv_dec(
            condn_emb_vxm,  # vxm optimized parameters
            8,
            (80, 96, 80, 8),
            2,
            [3, 3, 3],
            8,
        )
        last_tensor = dec_out
        for i in range(3):
            last_tensor = KL.Conv3D(
                8, kernel_size=3, padding='same', name='atlas_ec_%d' % i,
            )(last_tensor)

        pout = KL.Conv3D(
            1, kernel_size=3, padding='same', name='atlasmodel_c',
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1e-7,
            ),
            bias_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1e-7,
            ),
        )(last_tensor)

        new_atlas = atlas_inputs + pout

    elif atlas_model == 'ours' and conditional is True:
        """Start from a learned vector which is then conditioned."""

        # FiLM branch:
        # TODO: fix hardcoding
        condn_vec = 1.0 * condn
        for _ in range(4):
            condn_vec = KL.Dense(
                64,
                kernel_initializer=init,
            )(condn_vec)
            condn_vec = KL.LeakyReLU(0.2)(condn_vec)

        # Input layer of decoder: learned parameter vector
        const_vec = KL.Lambda(const_inp)(condn)  # use condn to get batch info
        condn_emb_vxm = KL.Dense(
            (80 * 96 * 80 * 8),
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=0.02,
            ),
            bias_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=0.02,
            ),
        )(const_vec)

        # Reshape to feature map and FiLM for convolutional processing:
        condn_emb_vxm = KL.Reshape((80, 96, 80, 8))(condn_emb_vxm)
        condn_emb_vxm = FiLM(init=init)([condn_emb_vxm, condn_vec])

        # 5 ResBlocks at lower resolution:
        s1 = conv_block(
            condn_emb_vxm, ch, condn_emb=condn_vec, sn=True, init=init,
        )
        sres = 1.0 * s1

        for _ in range(5):
            sip = sres
            sres = conv_block(
                sres, ch, condn_emb=condn_vec, sn=True, init=init,
            )
            sres = conv_block(
                sres, ch, condn_emb=condn_vec, sn=True, init=init,
            )
            sres = sres + sip

        # Upsample --> More Conv+FiLM+LeakyReLU blocks:
        dec_out = conv_dec_film(
            sres,
            condn_vec,
            8,
            (80, 96, 80, 8),
            2,
            [3, 3, 3],
            8,
            sn=True,
            init=init,
        )
        last_tensor = dec_out

        pout = KL.Conv3D(
            1, kernel_size=3, padding='same', name='atlasmodel_c',
            kernel_initializer=init,
        )(last_tensor)

        pout = KL.Activation('tanh')(pout)

        # Add to linear average:
        new_atlas = atlas_inputs + pout

    elif atlas_model == 'ours' and conditional is False:
        """Start from a learned vector."""

        condn_vec = None

        # Input layer of decoder: learned parameter vector
        const_vec = KL.Lambda(const_inp)(image_inputs)  # use ip for batch info
        condn_emb_vxm = KL.Dense(
            (80 * 96 * 80 * 8),
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=0.02,
            ),
            bias_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=0.02,
            ),
        )(const_vec)

        condn_emb_vxm = KL.Reshape((80, 96, 80, 8))(condn_emb_vxm)

        # 5 ResBlocks at lower resolution:
        s1 = conv_block(
            condn_emb_vxm, ch, condn_emb=condn_vec, sn=True, init=init,
        )
        sres = 1.0 * s1

        for _ in range(5):
            sip = sres
            sres = conv_block(
                sres, ch, condn_emb=condn_vec, sn=True, init=init,
            )
            sres = conv_block(
                sres, ch, condn_emb=condn_vec, sn=True, init=init,
            )
            sres = sres + sip

        # Upsample --> More Conv+FiLM+LeakyReLU blocks:
        dec_out = conv_dec_film(
            sres,
            condn_vec,
            8,
            (80, 96, 80, 8),
            2,
            [3, 3, 3],
            8,
            sn=True,
            conditional=False,
            init=init,
        )
        last_tensor = dec_out

        pout = KL.Conv3D(
            1, kernel_size=3, padding='same', name='atlasmodel_c',
            kernel_initializer=init,
        )(last_tensor)

        pout = KL.Activation('tanh')(pout)

        # Add to linear average:
        new_atlas = atlas_inputs + pout

    if clip_bckgnd:
        new_atlas = new_atlas * tf.cast(
            tf.math.greater(atlas_inputs, 1e-2), tf.float32,
        )

    # Registration network. Taken from vxm:
    # Encoder:
    inp = KL.concatenate([image_inputs, new_atlas])
    d1 = conv_block(inp, ch, stride=2, instancen=normreg, init=init)
    d2 = conv_block(d1, ch, stride=2, instancen=normreg, init=init)
    d3 = conv_block(d2, ch, stride=2, instancen=normreg, init=init)
    d4 = conv_block(d3, ch, stride=2, instancen=normreg, init=init)

    # Bottleneck:
    dres = conv_block(d4, ch, instancen=normreg, init=init)

    # Decoder:
    d5 = conv_block(dres, ch, mode='up', instancen=normreg, init=init)
    d5 = KL.concatenate([d5, d3])

    d6 = conv_block(d5, ch, mode='up', instancen=normreg, init=init)
    d6 = KL.concatenate([d6, d2])

    d7 = conv_block(d6, ch, mode='up', instancen=normreg, init=init)
    d7 = KL.concatenate([d7, d1])

    if full_size:  # if estimating displacements at half-resolution for speed
        d7 = conv_block(
            d7, ch, mode='up', instancen=normreg, init=init,
        )
        d7 = KL.concatenate([d7, inp])

    d7 = conv_block(
        d7, ch, mode='const', instancen=normreg, init=init,
    )
    d7 = conv_block(
        d7, ch, mode='const', instancen=normreg, init=init,
    )
    d7 = conv_block(d7, ch//2, mode='const', activation=False, init=init)

    # Get velocity field:
    d7 = tf.pad(d7, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")

    vel = KL.Conv3D(
        filters=3,
        kernel_size=3,
        padding='valid',
        use_bias=True,
        kernel_initializer=vel_init,
        name='vel_field',
    )(d7)

    # Get diffeomorphic displacement field:
    diff_field = VecInt(method='ss', int_steps=5, name='def_field')(vel)

    # Get moving average of deformations:
    diff_field_ms = MeanStream(name='mean_stream', cap=100)(diff_field)

    if full_size is False:  # i.e. rescale displacement field to full size
        # compute regularizers on diff_field_half for efficiency:
        diff_field_half = 1.0 * diff_field
        diff_field = RescaleTransform(2.0, name='flowup')(diff_field)
        moved_atlas = SpatialTransformer()([new_atlas, diff_field])
        ops = [moved_atlas, diff_field_ms, new_atlas, diff_field_half]
    else:
        moved_atlas = SpatialTransformer()([new_atlas, diff_field])
        ops = [moved_atlas, diff_field_ms, new_atlas, diff_field]

    if conditional:
        return tf.keras.Model(
            inputs=[image_inputs, atlas_inputs, condn],
            outputs=ops,
        )
    else:
        return tf.keras.Model(
            inputs=[image_inputs, atlas_inputs],
            outputs=ops,
        )


# ----------------------------------------------------------------------------
# Discriminator architecture


def Discriminator(
    ch=32,
    conditional=True,
    input_resolution=[160, 192, 160, 1],
    sn_out=True,
    initialization='orthogonal',
    n_condns=1,
):
    """
    Args:
        ch : int
            Channel multiplier.
        conditional : bool
            Flag indicating whether generator model is conditional.
        input_resolution: list
            Input image dimensions.
        sn_out: bool
            Whether to use SpectralNorm of last layer of discriminator.
        initialization: str
            Weight init. One of "default" or "orthogonal".
        n_condns: int
            Number of conditions if training conditionally.
    """
    inp = tf.keras.layers.Input(shape=input_resolution, name='input_image')

    if initialization == 'orthogonal':
        init = 'orthogonal'
    elif initialization == 'default' or initialization is None:
        init = None
    else:
        raise ValueError

    # If training conditionally:
    if conditional:
        condn = tf.keras.layers.Input(shape=(n_condns,))
        # If n_condns > 1 with both continuous and categorical attributes,
        # then continuous attributes can be linearly projected, but categorical
        # attributes need an embedding matrix, equivalently implemented here
        # with a bias-free dense layer acting on a one-hot representation:
        if n_condns == 1:
            condn_age_emb = KL.Dense(ch, kernel_initializer=init)(condn)
        elif n_condns > 1:  # Assumes that 1st idx of condn vector is cont. age
            condn_age, condn_cat = tf.split(condn, [1, -1], axis=-1)
            condn_age_emb = KL.Dense(ch, kernel_initializer=init)(condn_age)
            condn_cat_emb = KL.Dense(
                ch, use_bias=False, kernel_initializer=init,
            )(condn_cat)

    # If using spectralnorm:
    if sn_out:
        dOP = SpectralNormalization
    else:
        dOP = KL.Lambda(lambda x: x)  # basically no-op

    # Convolutional sequence:
    down1 = conv_block(inp, ch, sn=True, stride=2, init=init)
    down2 = conv_block(down1, ch*2, sn=True, stride=2, init=init)
    down3 = conv_block(down2, ch*4, sn=True, stride=2, init=init)
    fin1 = conv_block(down3, ch*8, sn=True, stride=2, init=init)

    # If using SN on discriminator output:
    fin1 = conv_block(fin1, ch, sn=sn_out, activation=False, init=init)

    if conditional:
        # Local projection discriminator feedback:
        op_age = dOP(KL.Conv3D(
            1, 1, padding='valid',
            use_bias=True, kernel_initializer=init,
        ))(fin1)

        condn_emb_spatial_age = tf.tile(
            tf.reshape(
                condn_age_emb,
                (tf.shape(condn_age_emb)[0], 1, 1,
                 1, tf.shape(condn_age_emb)[-1]),
            ),
            (1, tf.shape(op_age)[1], tf.shape(op_age)[2],
             tf.shape(op_age)[3], 1),
        )

        if n_condns > 1:
            condn_emb_spatial_cat = tf.tile(
                tf.reshape(
                    condn_cat_emb,
                    (tf.shape(condn_cat_emb)[0], 1, 1,
                     1, tf.shape(condn_cat_emb)[-1]),
                ),
                (1, tf.shape(op_age)[1], tf.shape(op_age)[2],
                 tf.shape(op_age)[3], 1),
            )

            op = (
                op_age +
                tf.reduce_sum(
                    condn_emb_spatial_cat * fin1, axis=-1, keepdims=True,
                ) +
                tf.reduce_sum(
                    condn_emb_spatial_age * fin1, axis=-1, keepdims=True,
                )
            )
        else:
            op = op_age + tf.reduce_sum(
                condn_emb_spatial_age * fin1, axis=-1, keepdims=True,
            )

        return tf.keras.Model(
            inputs=[inp, condn], outputs=[op],
        )
    else:  # Unconditional
        op = dOP(KL.Conv3D(
            1, 1, padding='valid',
            use_bias=True, kernel_initializer=init,
        ))(fin1)
        return tf.keras.Model(
            inputs=[inp], outputs=[op],
        )


# ----------------------------------------------------------------------------
# Decoder for conditional voxelmorph architectures

def conv_dec(
    input_tensor,
    nb_features,
    input_shape,
    nb_levels,
    conv_size,
    nb_labels,
    name=None,
    prefix=None,
    feat_mult=1,
    pool_size=2,
    padding='same',
    dilation_rate_mult=1,
    activation='elu',
    use_residuals=False,
    final_pred_activation='linear',
    nb_conv_per_level=2,
    layer_nb_feats=None,
    batch_norm=None,
    convL=None,
    input_model=None,
):
    """
    Decoder for conditional vxm architecture.

    Taken directly from vxm/neurite,
    https://github.com/adalca/neurite/blob/master/neurite/tf/models.py#L725

    """

    # vol size info
    ndims = len(input_shape) - 1
    input_shape = tuple(input_shape)
    if isinstance(pool_size, int):
        if ndims > 1:
            pool_size = (pool_size,) * ndims
    if ndims == 1 and isinstance(pool_size, tuple):
        pool_size = pool_size[0]  # 1D upsampling takes int not tuple

    # prepare layers
    if convL is None:
        convL = getattr(KL, 'Conv%dD' % ndims)
    conv_kwargs = {'padding': padding, 'activation': activation}
    upsample = getattr(KL, 'UpSampling%dD' % ndims)

    # up arm:
    # nb_levels - 1 layers of Deconvolution3D
    #    (approx via up + conv + ReLu) + merge + conv + ReLu + conv + ReLu
    lfidx = 0
    for level in range(nb_levels - 1):
        nb_lvl_feats = np.round(
            nb_features*feat_mult**(nb_levels - 2 - level),
        ).astype(int)
        conv_kwargs['dilation_rate'] = dilation_rate_mult**(nb_levels-2-level)

        # upsample matching the max pooling layers size
        name = '%s_up_%d' % (prefix, nb_levels + level)
        last_tensor = upsample(size=pool_size, name=name)(input_tensor)

        # convolution layers
        for conv in range(nb_conv_per_level):
            if layer_nb_feats is not None:
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += 1

            name = '%s_conv_uparm_%d_%d' % (prefix, nb_levels + level, conv)
            if conv < (nb_conv_per_level-1) or (not use_residuals):
                last_tensor = convL(
                    nb_lvl_feats, conv_size, **conv_kwargs, name=name,
                )(last_tensor)
            else:
                last_tensor = convL(
                    nb_lvl_feats, conv_size, padding=padding, name=name,
                )(last_tensor)

    pred_tensor = convL(nb_labels, 1, activation=None)(last_tensor)

    return pred_tensor


# ----------------------------------------------------------------------------
# Decoder for custom architectures

def conv_dec_film(
    condn_emb,
    condn_vec,
    nb_features,
    input_shape,
    nb_levels,
    conv_size,
    nb_labels,
    sn=False,
    conditional=True,
    name=None,
    prefix=None,
    feat_mult=1,
    pool_size=2,
    padding='valid',
    dilation_rate_mult=1,
    use_residuals=False,
    final_pred_activation='linear',
    nb_conv_per_level=2,
    layer_nb_feats=None,
    convL=None,
    init='default',
):
    """
    Modified from VXM/neurite to use FiLM layers instead of bias params,
    https://github.com/adalca/neurite/blob/master/neurite/tf/models.py#L725

    As this function was taken from VXM, it has a lot of inapplicable
    functionality which will be fixed prior to public release.
    """

    # vol size info
    ndims = len(input_shape) - 1
    input_shape = tuple(input_shape)
    if isinstance(pool_size, int):
        if ndims > 1:
            pool_size = (pool_size,) * ndims
    if ndims == 1 and isinstance(pool_size, tuple):
        pool_size = pool_size[0]  # 1D upsampling takes int not tuple

    # prepare layers
    if convL is None:
        convL = getattr(KL, 'Conv%dD' % ndims)

    # If conditional:
    if conditional is True:
        conv_kwargs = {
            'use_bias': False,
            'kernel_initializer': init,
        }
    else:
       conv_kwargs = {
            'use_bias': True,
            'kernel_initializer': init,
        }

    up_shape = tuple(2*np.array(K.int_shape(condn_emb)[1:-1]))  # TODO: ugly

    # up arm:
    # nb_levels - 1 layers of Deconvolution3D
    #    (approx via up + conv + ReLu) + merge + conv + ReLu + conv + ReLu
    lfidx = 0

    last_tensor = 1.0 * condn_emb

    for level in range(nb_levels - 1):
        nb_lvl_feats = np.round(
            nb_features*feat_mult**(nb_levels - 2 - level),
        ).astype(int)
        conv_kwargs['dilation_rate'] = dilation_rate_mult**(nb_levels-2-level)

        # upsample matching the max pooling layers size
        name = '%s_up_%d' % (prefix, nb_levels + level)
        last_tensor = TrilinearResizeLayer(
            up_shape,
            name=name,
        )(last_tensor)
        # last_tensor = upsample(size=pool_size, name=name)(condn_emb)

        up_shape = tuple(2 * np.array(up_shape))  # for next level

        # convolution layers
        for conv in range(nb_conv_per_level):
            if layer_nb_feats is not None:
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += 1

            name = '%s_conv_uparm_%d_%d' % (prefix, nb_levels + level, conv)
            if conv < (nb_conv_per_level-1) or (not use_residuals):
                last_tensor = tf.pad(
                    last_tensor,
                    [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],
                    "REFLECT",
                )
                if sn:
                    last_tensor = SpectralNormalization(convL(
                        nb_lvl_feats, conv_size, **conv_kwargs, name=name,
                    ))(last_tensor)
                else:
                    last_tensor = convL(
                        nb_lvl_feats, conv_size, **conv_kwargs, name=name,
                    )(last_tensor)

                if conditional:
                    last_tensor = FiLM(init=init)([last_tensor, condn_vec])
                last_tensor = KL.LeakyReLU(0.2)(last_tensor)
            else:
                last_tensor = tf.pad(
                    last_tensor,
                    [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],
                    "REFLECT",
                )
                if sn:
                    last_tensor = SpectralNormalization(convL(
                        nb_lvl_feats, conv_size, padding=padding, name=name,
                    ))(last_tensor)
                else:
                    last_tensor = convL(
                        nb_lvl_feats, conv_size, padding=padding, name=name,
                    )(last_tensor)

                if conditional:
                    last_tensor = FiLM(init=init)([last_tensor, condn_vec])

    if sn:
        pred_tensor = SpectralNormalization(convL(
            nb_labels, 1, activation=None,
        ))(last_tensor)
    else:
        pred_tensor = convL(nb_labels, 1, activation=None)(last_tensor)

    if conditional:
        pred_tensor = FiLM(init=init)([pred_tensor, condn_vec])

    return pred_tensor
