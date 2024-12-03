# Factorized Implicit Global Convolution Network

Computational Fluid Dynamics (CFD) is central to automotive vehicle design, which involves
studying how the car geometry affects the pressure field.
This requires very fine shapes, represented by large 3D point clouds and high accuracy,
which are currently out of reach for deep learning based methods.
As a result, the problem is typically solved with slow numerical solvers.

We propose **FIGConvUNet** [[1](#references)], a novel architecture that can efficiently
solve CFD problems for large 3D meshes and arbitrary input and output geometries.
FIGConvUNet efficiently combines U-shaped architecture, graph information gathering,
and integration, learning efficient latent representation through the representation
graph voxel layer.
We empirically validate our approach on the industry benchmark
Ahmed body [[2, 3](#references)] and the real-world DrivAerNet dataset [[4](#references)]
based on Volkswagen DrivAer [[5](#references)] dataset, with geometries composed
of 100 thousand and 1 million points, respectively.
We demonstrate a 140k× speed-up compared to GPU-accelerated
computational fluid dynamics (CFD) simulators and over 2× improvement in pressure prediction
over prior deep learning arts.

## Installation

FIGConvUNet dependencies can be installed with `pip install`, for example:

```bash
pip install -e .[figconv]
```

It is recommended to install these dependencies in a Modulus Docker container,
which provides a simple way to run Modulus.

## Training

FIGConvUNet uses [Hydra](https://hydra.cc/docs/intro/) for experiment configuration.
The following command launches the experiment defined in `drivaernet/figconv_unet` config
using default parameters with the exception of `data.every_n_data` which enables
dataset sampling.

```bash
python train.py \
    +experiment=drivaernet/figconv_unet \
    data.data_path=./datasets/drivaer/
```

A bit more interesting example demonstrates how other experiment parameters
can be overridden from the command line:

```bash
python train.py \
    +experiment=drivaernet/figconv_unet \
    data.data_path=./dataset/drivaer/ \
    'model.hidden_channels=[16, 16, 16, 16]' \
    optimizer=adamw \
    optimizer.lr=0.1 \
    seed=1 \
    train.num_epochs=10 \
    ~loggers.wandb
```

Note that for `zsh`, you need to escape `~` in front of `loggers.wandb` with `\~loggers.wandb`.

In this scenario:

* some additional dataset and model parameters are overridden.
* optimizer is changed to `AdamW` and its learning rate is set to 0.1.
* number of epochs is set to 10 and `wandb` (Weights & Biases) logger is removed.

See [Hydra documentation](https://hydra.cc/docs/intro) for more details.

For the full set of training script options, run the following command:

```bash
python train.py --help
```

In case of issues with Hydra config, you may get a Hydra error message
that is not particularly useful. In such case, use `HYDRA_FULL_ERROR=1`
environment variable:

```bash
HYDRA_FULL_ERROR=1 python train.py ...
```

### Multi-GPU Training

FIGConvUNet supports training and evaluation on multiple GPUs.
This can be done using `mpirun` or [torchrun](https://pytorch.org/docs/2.0/elastic/run.html)
utilities. For example, to train the previous example on 2 GPUs using MPI:

```bash
mpirun -np 2 python train.py \
    +experiment=drivaernet/figconv_unet \
    data.data_path=./dataset/drivaer/ \
    'model.hidden_channels=[16, 16, 16, 16]' \
    optimizer=adamw \
    optimizer.lr=0.1 \
    seed=1 \
    train.num_epochs=10 \
    ~loggers.wandb
```

## References

1. [Factorized Implicit Global Convolution for Automotive Computational Fluid Dynamics Prediction](https://arxiv.org/abs/TODO)
2. [Some Salient Features Of The Time-Averaged Ground Vehicle Wake](https://doi.org/10.4271/840300)
3. [Ahmed body wiki](https://www.cfd-online.com/Wiki/Ahmed_body)
4. [DrivAerNet: A Parametric Car Dataset for Data-Driven Aerodynamic Design and Graph-Based Drag Prediction](https://arxiv.org/abs/2403.08055)
5. [Deep Learning for Real-Time Aerodynamic Evaluations of Arbitrary Vehicle Shapes](https://arxiv.org/abs/2108.05798)
