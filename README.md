# NVIDIA Modulus (Beta)

<!-- markdownlint-disable -->
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![GitHub](https://img.shields.io/github/license/NVIDIA/modulus)](https://github.com/NVIDIA/modulus/blob/master/LICENSE.txt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!-- markdownlint-enable -->

NVIDIA Modulus is an open source deep-learning framework for building, training, and fine-tuning
deep learning models using state-of-the-art SciML methods for AI4science and engineering.

Modulus provides utilities and optimized pipelines to develop AI models that combine physics-knowledge with data, enabling real-time predictions.
Whether you are exploring the use of Neural operators or GNNs or transformers or
interested in Physics informed Neural Networks or a hybrid approach in between, Modulus
provides you with the optimized stack that will enable you to train your models at scale.

<!-- toc -->

- [More About Modulus](#more-about-modulus)
    - [Scalable GPU optimized training Library](#Scalable-GPU-optimized-training-Library)
    - [AI4Science Library](#AI4Science-Library)
      - [Domain Specific Packages](#Domain-Specific-Packages) 
    - [Python First](#python-first)
- [Who is contributing to Modulus](#Who-is-contributing-to-Modulus)
- [Why use Modulus](###Why-are-they-using-Modulus)
- [Getting Started](#getting-started)
- [Resources](#resources)
- [Installation](#installation)
  - [Binaries](#PyPi)
    - [Optional dependencies](#Optional-dependencies)
    - [NVCR Container](#NVCR-Container)
  - [From Source](#from-source)
    - [Package](#Package)
      - [Source Container](#Source-Container)
    - [Docker Image](#docker-image)
      - [Using pre-built images](#using-pre-built-images)
- [Contributing](#Contributing-to-Modulus)
- [Communication](#communication)
- [License](#license)

<!-- tocstop -->

## More About Modulus

[Learn the basics of Modulus](??)

At a granular level, Modulus is a library that consists of the following components:

Component | Description |
| ---- | --- |


Usually, Modulus is used either as:

- A complementary tool to Pytorch when exploring AI for SciML and AI4Science applications.
- A deep learning research platform that provides scale and optimal performance on NVIDIA GPUs.

Elaborating Further:

### Scalable GPU optimized training Library

Modulus package is the core module that provides the core algorithms, network architectures
and utilities with a pytorch like experience. Reference samples cover a broad spectrum of physics-constrained and data-driven
workflows to suit the diversity of use cases in the science and engineering disciplines.

Detailed information on features and capabilities can be found in the [Modulus documentation](https://docs.nvidia.com/modulus/index.html#core).

## AI4Science Library

- [Modulus Symbolic (Beta)](https://github.com/NVIDIA/modulus-sym): Repository of 
  algorithms and utilities to be used with Modulus
  core to physics inform model training as well as higher level abstraction
  for domain experts.

### Domain Specific Packages
The following are packages dedicated for domain experts of specific communities catering to their unique exploration needs.  
- [Earth-2 Studio](https://github.com/NVIDIA/earth2studio): Open source project
  to enable climate researchers and scientists to explore and experiment with
  AI models for weather and climate.

### Research packages
The following are research packages that get packaged into Modulus once they are stable. 
- [Modulus Makani](https://github.com/NVIDIA/modulus-makani): Experimental library designed to enable the research and development of machine-learning based weather and climate models.
- [Earth2 Grid](https://github.com/NVlabs/earth2grid): Experimental library with utilities for working geographic data defined on various grids.
- [Earth-2 MIP](https://github.com/NVIDIA/earth2mip): Experimental library with utilities for model intercomparison for weather and climate models.

# Who is using and contributing to Modulus

Modulus is open source project and gets contributions from researchers in the SciML and AI4science field. While Modulus team works on optimizing the underlying SW stack, the community collaborates and contributes model architectures, datasets and reference applications so we can innovate in the pursuit of developing generalizable model architectures and algorithms.

Some examples of community contributors are highlighted here:

- Stanford team
- HP Labs team
- NV Research team

Examples of research teams using Modulus here:
- ORNL team Team
- TU Munich team
- CMU team

Please go to this page for a complete list of research work leveraging Modulus.

## Why are they using Modulus

Here are some of the key benefits of Modulus for SciML model development:

![Physcis based benchmarking](https://github.com/NVIDIA/modulus/tree/ram-cherukuri-patch-1/docs/img/Value%20prop/benchmarking.svg) | ![Generalized SciML recipes](https://github.com/NVIDIA/modulus/tree/ram-cherukuri-patch-1/docs/img/Value%20prop/recipe.svg) | ![Performance](https://github.com/NVIDIA/modulus/tree/ram-cherukuri-patch-1/docs/img/Value%20prop/performance.svg)
---|---|---|
Physcis based benchmarking and validation|Ease of using generalized SciML recipes with heterogenous datasets|Out of the box performance and scalability 

See what SciML reseachers are saying about Modulus


# Getting started
The following resources will help you in learning how to use Modulus. The best way is to start with a reference sample and then update it for your own use case.
* [Getting started Webinar](https://www.nvidia.com/en-us/on-demand/session/gtc24-dlit61460/?playlistId=playList-bd07f4dc-1397-4783-a959-65cec79aa985)
* [Getting started Guide](https://docs.nvidia.com/deeplearning/modulus/getting-started/index.html)
* [Reference Samples](https://github.com/NVIDIA/modulus/blob/main/examples/README.md)
* [User guide Documentation](https://docs.nvidia.com/deeplearning/modulus/modulus-core/index.html)

## Resources
* [AI4Science Bootcamp]()
* []()
  
## Installation

### PyPi

The recommended method for installing the latest version of Modulus is using PyPi:

```Bash
pip install nvidia-modulus
```

The installation can be verified by running a simple python code snippet as shown below:

```python
python
>>> import torch
>>> from modulus.models.mlp.fully_connected import FullyConnected
>>> model = FullyConnected(in_features=32, out_features=64)
>>> input = torch.randn(128, 32)
>>> output = model(input)
>>> output.shape
torch.Size([128, 64])
```

#### Optional dependencies

Modulus has many optional dependencies that are used in specific components.
When using pip, all dependencies used in Modulus can be installed with
`pip install nvidia-modulus[all]`. If you are developing Modulus, developer dependencies
can be installed using `pip install nvidia-modulus[dev]`. Otherwise, additional dependencies
can be installed on a case by case basis. A detailed information on installing the
optional dependencies can be found in the
[Getting Started Guide](https://docs.nvidia.com/deeplearning/modulus/getting-started/index.html).

### NVCR Container

The recommended Modulus docker image can be pulled from the
[NVIDIA Container Registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/containers/modulus):

```Bash
docker pull nvcr.io/nvidia/modulus/modulus:24.04
```

Inside the container you can clone the Modulus git repositories and get started with the
examples. Below command show the instructions to launch the modulus container and run an
examples from this repo.

```bash
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia \
--rm -it nvcr.io/nvidia/modulus/modulus:24.04 bash
git clone https://github.com/NVIDIA/modulus.git
cd modulus/examples/cfd/darcy_fno/
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

### Source Container

To build Modulus docker image:

```bash
docker build -t modulus:deploy \
    --build-arg TARGETPLATFORM=linux/amd64 --target deploy -f Dockerfile .
```

Alternatively, you can run `make container-deploy`

To build CI image:

```bash
docker build -t modulus:ci \
    --build-arg TARGETPLATFORM=linux/amd64 --target ci -f Dockerfile .
```

Alternatively, you can run `make container-ci`.

Currently only `linux/amd64` and `linux/arm64` platforms are supported. If using
`linux/arm64`, some dependencies like `warp-lang` might not install correctly.

## Contributing to Modulus

Modulus is an open source collaboration and its success is rooted in community
contribution to further the field of Physics-ML. Thank you for contributing to the
project so others can build on your contribution.

For guidance on making a contribution to Modulus, please refer to the
[contributing guidelines](https://github.com/NVIDIA/modulus/blob/main/CONTRIBUTING.md).

## Communication

- Github Discussions: Discuss new architectures, implementations, Physics-ML research, etc.
- GitHub Issues: Bug reports, feature requests, install issues, etc.
- Modulus Forum: The [Modulus Forum](https://forums.developer.nvidia.com/c/physics-simulation/modulus-physics-ml-model-framework)
hosts an audience of new to moderate level users and developers for general chat, online
discussions, collaboration, etc.

## License

Modulus is provided under the Apache License 2.0, please see [LICENSE.txt](./LICENSE.txt)
for full license text.
