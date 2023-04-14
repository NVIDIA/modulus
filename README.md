# Modulus (Beta)

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![GitHub](https://img.shields.io/github/license/NVIDIA/modulus)](https://github.com/NVIDIA/modulus/blob/master/LICENSE.txt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Modulus is a PyTorch based deep-learning toolkit for developing deep learning models for physical systems. This package aims to provide useful utilities for physics-constrained and data-driven workflows.

**This is an early-access beta release**

<p align="center">
  <img src="./docs/img/modulus-pipes.jpg" alt="NVIDIA Modulus"/>
</p>

## Modulus Packages

- [Modulus (Beta)](https://github.com/NVIDIA/modulus)
- [Modulus Launch (Beta)](https://github.com/NVIDIA/modulus-launch)
- [Modulus Symbolic (Beta)](https://github.com/NVIDIA/modulus-sym)

## Installing 

Modulus is coming to PyPi soon! In the mean time the best way is to install from source:

```Bash
git clone git@github.com:NVIDIA/modulus.git && cd modulus

pip install --upgrade pip
pip install .
```

## Docker

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
