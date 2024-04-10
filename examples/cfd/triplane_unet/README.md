# Triplane UNet for Computational Fluid Dynamics

Computational Fluid Dynamics (CFD) are central to automotive vehicle design, which involves studying how the car geometry affects the pressure field. This requires very fine shapes, represented by large 3D point clouds and high accuracy, which are currently out of reach for deep based methods. As a result, the problem is typically solved with slow numerical solvers. We propose TriPlane-UNet, a novel architecture that can efficiently solve CFD problems for very large 3D meshes and arbitrary input and output geometries. TriPlane-UNet efficiently combine U-shaped architecture, graph information gathering, and integration, learning efficient latent representation through the representation graph voxel layer. We empirically validate our approach on the industry benchmark Ahmed body and the real-world Volkswagen DrivAer datasets, with geometries composed of 100 thousand and 1 million points, respectively. We demonstrate a 140k× speed-up compared to GPU-accelerated computational fluid dynamics (CFD) simulators and over 2× improvement in pressure prediction over prior deep learning arts.

## Installation

TriplaneUNet dependencies can be installed with `pip install`, for example:

```
pip install -e .[tpunet]
```

It is recommended to install these dependencies in Modulus Docker container which has been tested and verified to work.

## Training

Triplane UNet uses [Hydra](https://hydra.cc/docs/intro/) for experiment configuration.
The following command line launches the experiment defined in `drivaer/triplane_unet` config using default parameters except `data.every_n_data` which enables dataset sampling.

```bash
python train.py \
    +experiment=drivaer/triplane_unet \
    data.data_path=./datasets/drivaer/ \
    data.every_n_data=10
```

A bit more interesting example demonstrates how other experiment parameters can be overridden from the command line:

```bash
python train.py \
    +experiment=drivaer/triplane_unet \
    data.data_path=./dataset/drivaer/ \
    'data.subsets_postfix=[nospoiler]' \
    'model.hidden_channels=[16, 16, 16, 16]' \
    optimizer=adamw \
    seed=1 \
    train.num_epochs=10 \
    ~loggers.wandb
```
In this scenario, some additional dataset and model parameters are overridden as well as optimizer is changed to `AdamW`. Additionally, number of epochs is set to 10 and W&B logger is removed.

See [Hydra documentation](https://hydra.cc/docs/intro) for more details.
