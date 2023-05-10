# Modulus (Beta)

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![GitHub](https://img.shields.io/github/license/NVIDIA/modulus)](https://github.com/NVIDIA/modulus/blob/master/LICENSE.txt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Modulus is a PyTorch based deep-learning toolkit for developing deep learning models for physical systems. This package aims to provide useful utilities for physics-constrained and data-driven workflows. Additional information can be found in the [Modulus documentation](https://docs.nvidia.com/modulus/index.html#core).

<p align="center">
  <img src="./docs/img/modulus-pipes.jpg" alt="NVIDIA Modulus"/>
</p>

Test

## Modulus Packages

- [Modulus (Beta)](https://github.com/NVIDIA/modulus)
- [Modulus Launch (Beta)](https://github.com/NVIDIA/modulus-launch)
- [Modulus Symbolic (Beta)](https://github.com/NVIDIA/modulus-sym)

## Installation 

### PyPi

The recommended method for installing the latest version of Modulus is using PyPi:
```Bash
pip install nvidia-modulus
```

### Container

The recommended Modulus docker image can be pulled from the [NVIDIA Container Registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/containers/modulus):
```Bash
docker pull nvcr.io/nvidia/modulus/modulus:23.05
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

Modulus is in an open-source beta. We are not accepting external contributions at this time.

## Contact

Reach out to Modulus team members and user community on the [NVIDIA developer forums](https://forums.developer.nvidia.com/c/physics-simulation/modulus-physics-ml-model-framework).

## License
Modulus is provided under the Apache License 2.0, please see [LICENSE.txt](./LICENSE.txt) for full license text.
