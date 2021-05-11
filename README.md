# Generative Adversarial Template Construction
### [Project Page](https://www.neeldey.com/deformable-templates/) | [Paper](https://drive.google.com/file/d/1W29kHKU5BUY6EK1Wyuklt5y98j3nPGZE/view?usp=sharing)

![dHCP Templates](https://www.neeldey.com/deformable-templates/img/dhcp-cond.gif)

Tensorflow 2 code repository for *Generative Adversarial Registration for Improved Conditional Deformable Templates*, arXiv 2021. 

`train_script.py` is the main template construction script
that implements all methods considered in the paper for the 3D datasets. 

The current code repository will be heavily refactored (e.g., improving data loading, better abstraction). FFHQ-Aging scripts only require a 
change from 3D to 2D and will be added as well.

## Dependencies

We recommend setting up an anaconda environment and installing all dependencies as,

```bash
conda env create -f environment.yml
conda activate tf2
```

## Usage
Example training call for conditional templates:
```bash
python conditional_script.py --name phd-ours-cond --dataset pHD --oversample --nonorm_reg --clip --losswt_gp 5e-4 --gen_config ours
```

CLI args are:
```bash
usage: train_script.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE] 
                       [--dataset DATASET] [--name NAME] [--d_train_steps D_TRAIN_STEPS]
                       [--g_train_steps G_TRAIN_STEPS] [--lr_g LR_G] [--lr_d LR_D]
                       [--beta1_g BETA1_G] [--beta2_g BETA2_G] [--beta1_d BETA1_D]
                       [--beta2_d BETA2_D] [--unconditional] [--nonorm_reg] [--oversample]
                       [--d_snout] [--clip] [--reg_loss REG_LOSS] [--losswt_reg LOSSWT_REG]
                       [--losswt_gan LOSSWT_GAN] [--losswt_tv LOSSWT_TV] [--losswt_gp LOSSWT_GP]
                       [--gen_config GEN_CONFIG] [--steps_per_epoch STEPS_PER_EPOCH]
                       [--rng_seed RNG_SEED] [--start_step START_STEP] [--resume_ckpt RESUME_CKPT]
                       [--g_ch G_CH] [--d_ch D_CH] [--init INIT] [--lazy_reg LAZY_REG]
```

With verbose descriptions:
```python
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
        Not used in the paper.
    oversample: bool
        Whether to oversample rare ages during training.
    d_snout: bool
        Whether to apply Spectral Norm to the last layer of the Discriminator.
    clip: bool
        Whether to clip the template background during training.        
    reg_loss: str
        Type of registration loss. One of {'NCC', 'NonSquareNCC'}.
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
```

## Data loaders:
The training script expects data points to be in the form of npz files. To construct
a usable npz from a nifti file, the following code snippet was used:
```python
import numpy as np
import SimpleITK as sitk

simg = sitk.ReadImage('/path/to/nifti.nii.gz')
npy_img = sitk.GetArrayFromImage(simg)

# Assuming that you have 'age' and 'attribute' loaded:
np.savez_compressed(
    './data/dataset_name/train_npz/fname.npz',
    vol=npy_img,
    age=age,
    attribute=attribute,
)
```

We recommend inspecting L196-L238 of `train_script.py` and `./src/data_generators.py`
for more details of how to modify the data loaders for your use case.

## Acknowledgements:
This repo makes extensive usage of the [VoxelMorph](https://github.com/voxelmorph/voxelmorph) library.

