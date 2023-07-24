# Modulus (Beta)

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![GitHub](https://img.shields.io/github/license/NVIDIA/modulus)](https://github.com/NVIDIA/modulus/blob/master/LICENSE.txt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Modulus is an open source deep-learning framework for building, training, and fine-tuning deep learning models using state-of-the-art Physics-ML methods. 

Whether you are exploring the use of Neural operators like Fourier Neural Operators or interested in Physics informed Neural Networks or a hybrid approach in between, Modulus provides you with the optimized stack that will enable you to train your models at real world scale.

This package is the core module that provides the core algorithms, network architectures and utilities that cover a broad spectrum of physics-constrained and data-driven workflows to suit the diversity of use cases in the science and engineering disciplines.

Detailed information on features and capabilities can be found in the [Modulus documentation](https://docs.nvidia.com/modulus/index.html#core).

<p align="center">
  <img src=https://raw.githubusercontent.com/NVIDIA/modulus/main/docs/img/Modulus-850x720.svg alt="Modulus"/>
</p>

## Modulus Packages

- [Modulus (Beta)](https://github.com/NVIDIA/modulus)
- [Modulus Launch (Beta)](https://github.com/NVIDIA/modulus-launch)
- [Modulus Symbolic (Beta)](https://github.com/NVIDIA/modulus-sym)
- [Modulus Tool-Chain (Beta)](https://github.com/NVIDIA/modulus-toolchain)

## Installation 

### PyPi

The recommended method for installing the latest version of Modulus is using PyPi:
```Bash
pip install nvidia-modulus
```

The installation can be verified by running a simple python code snippet as shown below:
```
python
>>> import torch
>>> from modulus.models.mlp.fully_connected import FullyConnected
>>> model = FullyConnected(in_features=32, out_features=64)
>>> input = torch.randn(128, 32)
>>> output = model(input)
>>> output.shape
torch.Size([128, 64])
```

### Container

The recommended Modulus docker image can be pulled from the [NVIDIA Container Registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/containers/modulus):
```Bash
docker pull nvcr.io/nvidia/modulus/modulus:23.05
```

Inside the container you can clone the Modulus git repositories and get started with the examples. Below command show the instructions to launch the modulus container and run an example from the [Modulus Launch](https://github.com/NVIDIA/modulus-launch) repo. 
```
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia --rm -it nvcr.io/nvidia/modulus/modulus:23.05 bash
git clone https://github.com/NVIDIA/modulus-launch.git
cd modulus-launch/examples/cfd/darcy_fno/
pip install warp-lang # install NVIDIA Warp to run the darcy example
python train_fno_darcy.py
```

## From Source

### Package
For a local build of the Modulus Python package from source use:
```Bash
git clone git@github.com:NVIDIA/modulus.git && cd modulus

pip install --upgrade pip
pip install .
```

### Container

To build Modulus docker image:
```
docker build -t modulus:deploy --target deploy -f Dockerfile .
```
Alternatively, you can run `make container-deploy`

To build CI image:
```
docker build -t modulus:ci --target ci -f Dockerfile .
```
Alternatively, you can run `make container-ci`

## Contributing
Modulus is an open source collaboration and its success is rooted in community contribution to further the field of Physics-ML. Thank you for contributing to the project so others can build on your contribution.
For guidance on making a contribution to Modulus, please refer to the [contributing guidelines](https://github.com/NVIDIA/modulus/blob/main/CONTRIBUTING.md).

## Communication
* Github Discussions: Discuss new architectures, implementations, Physics-ML research, etc. 
* GitHub Issues: Bug reports, feature requests, install issues, etc.
* Modulus Forum: The [Modulus Forum](https://forums.developer.nvidia.com/c/physics-simulation/modulus-physics-ml-model-framework) hosts an audience of new to moderate level users and developers for general chat, online discussions, collaboration, etc. 

## License
Modulus is provided under the Apache License 2.0, please see [LICENSE.txt](./LICENSE.txt) for full license text.
