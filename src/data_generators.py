import numpy as np
from tensorflow.keras.utils import to_categorical


# ----------------------------------------------------------------------------
# 3D Neuroimaging data generators

def D_data_generator(
    vol_shape,
    img_list,
    oversample_age=True,
    batch_size=32,
    dataset='dHCP',
):
    """Data generator for discriminator in conditional setting.

    Args:
        vol_shape: tuple
            Shape of images (without feature axis).
        img_list: list of strings
            List of image file paths (assumed as npz files).
        oversample_age : bool
            Oversample images based on rarity w.r.t. age.
        batch_size: int
            Batch size.
        dataset: str
            Dataset choice. One of {'dHCP', 'pHD'}.
    """

    # Pick attribute of interest for dataset:
    if dataset == 'dHCP':
        age_max = 45.142857142857  # used for scaling age values
        n_condns = 1  # number of conditions
    elif dataset == 'pHD':
        attribute = 'disease'  # name of attribute in npz archive
        age_max = 87.72895277  # used for scaling age values
        n_condns = 3  # number of conditions

    # Oversample age:
    if oversample_age is True:
        ifrequencies = oversample(img_list)
    else:
        ifrequencies = None

    # prepare a zero array the size of the deformation:
    registration_images = np.zeros((batch_size,) + vol_shape + (1,))
    adversarial_images = np.zeros((batch_size,) + vol_shape + (1,))

    registration_images_condns = np.zeros((batch_size,) + (n_condns,))
    adversarial_images_condns = np.zeros((batch_size,) + (n_condns,))

    while True:
        # Sample indices for training:
        registration_idx = np.random.choice(
            len(img_list), batch_size, p=ifrequencies,
        )
        adversarial_idx = np.random.choice(
            len(img_list), batch_size, p=ifrequencies,
        )

        # TODO: looping is quite slow. however, typically 3D batches are very
        # small and it doesn't add much overhead.
        for i in range(batch_size):
            # Registration targets:
            reg_img_path = img_list[registration_idx[i]]
            reg_imgs_npz = np.load(reg_img_path)

            registration_images[i] = reg_imgs_npz['vol'][..., np.newaxis]
            registration_images_condns[i] = reg_imgs_npz['age']/age_max

            if n_condns > 1:
                registration_images_condns[i] = np.hstack((
                    reg_imgs_npz['age']/age_max,  # scale age [0, 1]
                    to_categorical(
                        reg_imgs_npz[attribute], n_condns - 1,
                    ),
                ))

            # Adversarial targets for comparison:
            adv_img_path = img_list[adversarial_idx[i]]
            advers_imgs_npz = np.load(adv_img_path)

            adversarial_images[i] = advers_imgs_npz['vol'][..., np.newaxis]
            adversarial_images_condns[i] = advers_imgs_npz['age']/age_max

            if n_condns > 1:
                adversarial_images_condns[i] = np.hstack((
                    advers_imgs_npz['age']/age_max,
                    to_categorical(
                        advers_imgs_npz[attribute], n_condns - 1,
                    ),
                ))

        yield (
            registration_images, adversarial_images,
            registration_images_condns, adversarial_images_condns,
        )


def G_data_generator(
    vol_shape,
    img_list,
    oversample_age=True,
    batch_size=32,
    dataset='dHCP',
):
    """Data generator for discriminator in conditional setting.

    Args:
        vol_shape: tuple
            Shape of images (without feature axis).
        img_list: list of strings
            List of image file paths (assumed as npz files).
        oversample_age : bool
            Oversample images based on rarity w.r.t. age.
        batch_size: int
            Batch size.
        dataset: str
            Dataset choice. One of {'dHCP', 'pHD'}.
    """

    # Pick attribute of interest for dataset:
    if dataset == 'dHCP':
        age_max = 45.142857142857  # used for scaling age values
        n_condns = 1  # number of conditions
    elif dataset == 'pHD':
        attribute = 'disease'  # name of attribute in npz archive
        age_max = 87.72895277  # used for scaling age values
        n_condns = 3  # number of conditions

    # Oversample age:
    if oversample_age is True:
        ifrequencies = oversample(img_list)
    else:
        ifrequencies = None

    # prepare a zero array the size of the deformation:
    registration_images = np.zeros((batch_size,) + vol_shape + (1,))
    registration_images_condns = np.zeros((batch_size,) + (n_condns,))

    while True:
        registration_idx = np.random.choice(
            len(img_list), batch_size, p=ifrequencies,
        )

        for i in range(batch_size):
            # Registration targets:
            reg_img_path = img_list[registration_idx[i]]
            reg_imgs_npz = np.load(
                reg_img_path,
            )

            registration_images[i] = reg_imgs_npz['vol'][..., np.newaxis]
            registration_images_condns[i] = reg_imgs_npz['age']/age_max

            if n_condns > 1:
                registration_images_condns[i] = np.hstack((
                    reg_imgs_npz['age']/age_max,
                    to_categorical(
                        reg_imgs_npz[attribute], n_condns - 1,
                    ),
                ))

        yield registration_images, registration_images_condns


# ----------------------------------------------------------------------------
# Utility functions:

def oversample(img_paths):
    """
    Takes list of file paths of npz files containing images with ages and
    outputs (rough) sampling probabilities such that the network sees
    timepoints equally frequently.

    Args:
        img_paths : str
            list of npz files containing images and attributes of interest.
    """

    frequencies = []
    for i in range(len(img_paths)):
        frequencies.append(np.load(img_paths[i])['age'])

    frequencies = np.array(frequencies).round()  # quantize (essentially bin)
    ages, counts = np.unique(frequencies, return_counts=True)  # get age hist.
    prob = counts/counts.sum()  # get age probabilities

    # Get inverted probabilities:
    iprob = 1 - prob
    iprob = iprob/iprob.sum()  # renormalize

    ifrequencies = frequencies.copy()
    for i in range(len(ages)):
        idx = np.where(frequencies == ages[i])
        ifrequencies[idx] = iprob[i]

    ifrequencies = ifrequencies/ifrequencies.sum()

    return ifrequencies
