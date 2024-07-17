# NVIDIA Modulus (Beta)

<!-- markdownlint-disable -->
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![GitHub](https://img.shields.io/github/license/NVIDIA/modulus)](https://github.com/NVIDIA/modulus/blob/master/LICENSE.txt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!-- markdownlint-enable -->
[**Getting Started**](#getting-started)
| [**Install guide**](#installation)
| [**Contributing Guidelines**](#contributing-to-modulus)
| [**Resources**](#resources)
| [**Communication**](#communication)

## What is Modulus?
NVIDIA Modulus is an open source deep-learning framework for building, training, and fine-tuning
deep learning models using state-of-the-art SciML methods for AI4science and engineering.

Modulus provides utilities and optimized pipelines to develop AI models that combine physics-knowledge with data, enabling real-time predictions.
Whether you are exploring the use of Neural operators, GNNs or transformers or are
interested in Physics-informed Neural Networks or a hybrid approach in between, Modulus
provides you with the optimized stack that will enable you to train your models at scale.

<!-- markdownlint-disable -->
<p align="center">
  <img src=docs/img/Value%20prop/Knowledge%20_guided_models.gif alt="Modulus"/>
</p>
<!-- markdownlint-enable -->

<!-- toc -->

- [More About Modulus](#more-about-modulus)
    - [Scalable GPU optimized training Library](#scalable-gpu-optimized-training-library)
    - [Suite of Physics-Informed ML Models](#suite-of-physics-informed-ml-models)
    - [Seamless PyTorch Integration](#seamless-pytorch-integration)
    - [Easy Customization and Extension](#easy-customization-and-extension)
    - [AI4Science Library](#ai4science-library)
     - [Domain Specific Packages](#domain-specific-packages) 
- [Who is contributing to Modulus](#who-is-using-and-contributing-to-modulus)
- [Why use Modulus](#why-are-they-using-modulus)
- [Getting Started](#getting-started)
- [Resources](#resources)
- [Installation](#installation)
- [Contributing](#contributing-to-modulus)
- [Communication](#communication)
- [License](#license)
  
<!-- tocstop -->

## More About Modulus

At a granular level, Modulus is a library that consists of the following components:

Component | Description |
| ---- | --- |
| [**modules.models**](https://docs.nvidia.com/deeplearning/modulus/modulus-core/api/modulus.models.html) | A collection of optimized, customizable, and easy-to-use models such as Fourier Neural Operators, Graph Neural Networks and many more|
| [**modulus.datapipes**](https://docs.nvidia.com/deeplearning/modulus/modulus-core/api/modulus.datapipes.html) | A data pipeline library, including benchmark datapipes, wheather daptapipes and graph datapipes|
| [**modulus.distributed**](https://docs.nvidia.com/deeplearning/modulus/modulus-core/api/modulus.distributed.html) | A distributed computing library build on top of troch.distributed to enable parallel training with just a few steps|
| [**modulus.utils**](https://docs.nvidia.com/deeplearning/modulus/modulus-core/api/modulus.utils.html) | Utilitis for optimization, checkpointing, logging to simplify model development and deployment|
| [**modulus.metrics**](https://docs.nvidia.com/deeplearning/modulus/modulus-core/api/modulus.metrics.html) | A library of optimized metric calculations for custom training and inference workflows, operating directly on PyTorch tensors|

Usually, Modulus is used either as:

- A complementary tool to Pytorch when exploring AI for SciML and AI4Science applications.
- A deep learning research platform that provides scale and optimal performance on NVIDIA GPUs.

Elaborating Further:

### Scalable GPU optimized training Library
Modulus provides a highly optimized and scalable training library for maximizing the power of NVIDIA GPUs.
[Distributed computing](https://docs.nvidia.com/deeplearning/modulus/modulus-core/api/modulus.distributed.html)
utilities allows for efficient scaling from a single GPU to multi-node GPU clusters with a few lines of code, ensuring that large-scale
physics-informed machine learning (ML) models can be trained quickly and effectively. The framework includes support for advanced
[optimization utilities](https://docs.nvidia.com/deeplearning/modulus/modulus-core/api/modulus.utils.html#module-modulus.utils.capture), [tailor made datapipes](https://docs.nvidia.com/deeplearning/modulus/modulus-core/api/modulus.datapipes.html), [validation utilities](https://github.com/NVIDIA/modulus-sym/tree/main/modulus/sym/eq) to enhance
the end to end training speed.

### Suite of Physics Informed ML Models
Modulus offers a comprehensive library of state-of-the-art models specifically designed for physics-ML applications.
The [Model Zoo](https://docs.nvidia.com/deeplearning/modulus/modulus-core/api/modulus.models.html#model-zoo)
includes generalizable model architectures such as [Fourier Neural Operators (FNOs)](modulus/models/fno), [Deep)Net](https://github.com/NVIDIA/modulus-sym/blob/main/modulus/sym/models/deeponet.py)
[Physics-Informed Neural Networks (PINNs)](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/foundational/1d_wave_equation.html),
[Graph Neural Networks (GNNs)](modulus/models/gnn_layers)
and generative AI models like [Diffusion Models](modulus/models/diffusion)
as well as domain-specific models such as [Deep Learning Weather Prediction (DLWP)](modulus/models/dlwp)
and [Super Resolution Network (SrNN)](modulus/models/srrn) among others.
These pretrained models are optimized for various physics domains, such as computational fluid dynamics, structural mechanics,
and electromagnetics. Users can download, customize, and build upon these models to suit their specific needs, significantly reducing the time
required to develop high-fidelity simulations.

### Seamless PyTorch Integration
Modulus is built on top of PyTorch, providing a familiar and user-friendly experience for those already proficient with PyTorch.
This includes a simple Python interface and modular design, making it easy to use Modulus with existing PyTorch workflows.
Users can leverage the extensive PyTorch ecosystem, including its libraries and tools, while benefiting from Modulus's
specialized capabilities for physics-ML. This seamless integration ensures that users can quickly adopt Modulus without a steep learning curve.

### Easy Customization and Extension
Modulus is designed to be highly extensible, allowing users to add new functionality with minimal effort. The framework provides Pythonic APIs for
defining new physics models, geometries, and constraints, making it easy to extend its capabilities to new use cases.
The adaptability of Modulus is further enhanced by key features such as [ONNX support](https://docs.nvidia.com/deeplearning/modulus/modulus-core/api/modulus.deploy.html)
for flexible model deployment, robust [logging utilities](https://docs.nvidia.com/deeplearning/modulus/modulus-core/api/modulus.launch.logging.html) for streamlined error handling,
and efficient [checkpointing](https://docs.nvidia.com/deeplearning/modulus/modulus-core/api/modulus.launch.utils.html#module-modulus.launch.utils.checkpoint) to save valuable training time. 

This extensibility ensures that Modulus can adapt to the evolving needs of researchers and engineers, facilitating the development of
innovative solutions in the field of physics-ML

You can write new neural network layers in Python using the Modulus API or PyTorch and NumPy-based libraries

Detailed information on features and capabilities can be found in the [Modulus documentation](https://docs.nvidia.com/modulus/index.html#core).

[Reference samples](examples/README.md) cover a broad spectrum of physics-constrained and data-driven
workflows to suit the diversity of use cases in the science and engineering disciplines.


### Hello world
You can start using Modulus in your PyTorch code as simple as shown here:

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

### AI4Science Library

- [Modulus Symbolic (Beta)](https://github.com/NVIDIA/modulus-sym): This repository of 
  algorithms and utilities allow SciML researchers and developers to physics inform model training and model validation. It also provides a higher level abstraction
  for domain experts that is native to science and engineering.

#### Domain Specific Packages
The following are packages dedicated for domain experts of specific communities catering to their unique exploration needs.  
- [Earth-2 Studio](https://github.com/NVIDIA/earth2studio): Open source project
  to enable climate researchers and scientists to explore and experiment with
  AI models for weather and climate.

#### Research packages
The following are research packages that get packaged into Modulus once they are stable. 
- [Modulus Makani](https://github.com/NVIDIA/modulus-makani): Experimental library designed to enable the research and development of machine-learning based weather and climate models.
- [Earth2 Grid](https://github.com/NVlabs/earth2grid): Experimental library with utilities for working geographic data defined on various grids.
- [Earth-2 MIP](https://github.com/NVIDIA/earth2mip): Experimental library with utilities for model intercomparison for weather and climate models.

## Who is using and contributing to Modulus

Modulus is open source project and gets contributions from researchers in the SciML and AI4science field. While Modulus team works on optimizing the underlying SW stack, the community collaborates and contributes model architectures, datasets and reference applications so we can innovate in the pursuit of developing generalizable model architectures and algorithms.

Some latest examples of community contributors are CMU team, Stanford Cardiovascular research team, HP Labs 3D Printing team etc.

Latest examples of research teams using Modulus are ORNL team, TU Munich team, MSOE etc.

Please go to this page for a complete list of research work leveraging Modulus. For a list of enterprises using Modulus refer [here](https://developer.nvidia.com/modulus).

## Why are they using Modulus

Here are some of the key benefits of Modulus for SciML model development:

![Benchmarking](docs/img/Value%20prop/benchmarking.svg) | ![Generalized SciML recipes](docs/img/Value%20prop/recipe.svg) | ![Performance](docs/img/Value%20prop/performance.svg)
---|---|---|
|SciML Benchmarking and validation|Ease of using generalized SciML recipes with heterogenous datasets |Out of the box performance and scalability
|Modulus enables researchers to benchmark their AI model against proven architectures for standard benchmark problems with detailed domain specific validation criteria.|Modulus enables researchers to pick from SOTA SciML architectures and use builtin data pipelines for their usecase.| Modulus provides out of the box performant training pipeline including optimized ETL pipelines for heterogrneous engineering and scientific datasets and out of the box scaling across multi-GPU and multi-node GPUs.

See what your peer SciML reseachers are saying about Modulus (Coming soon).


# Getting started
The following resources will help you in learning how to use Modulus. The best way is to start with a reference sample and then update it for your own use case.
* [Modulus: purpose and usage](https://www.nvidia.com/en-us/on-demand/session/dliteachingkit-setk5002/)
* [Using Modulus with your PyTorch model]( https://docs.nvidia.com/deeplearning/modulus/modulus-core/tutorials/simple_training_example.html#using-custom-models-in-modulus)
* [Using Modulus built-in models](https://docs.nvidia.com/deeplearning/modulus/modulus-core/tutorials/simple_training_example.html#using-built-in-models)
* [Getting started Webinar](https://www.nvidia.com/en-us/on-demand/session/gtc24-dlit61460/?playlistId=playList-bd07f4dc-1397-4783-a959-65cec79aa985)
* [Getting started Guide](https://docs.nvidia.com/deeplearning/modulus/getting-started/index.html)
* [Reference Samples](https://github.com/NVIDIA/modulus/blob/main/examples/README.md)
* [User guide Documentation](https://docs.nvidia.com/deeplearning/modulus/modulus-core/index.html)
* [Video Tutorials](https://www.nvidia.com/en-us/on-demand/search/?facet.mimetype[]=event%20session&layout=list&page=1&q=modulus&sort=relevance&sortDir=desc)


## Resources
* [AI4Science Bootcamp](https://github.com/openhackathons-org/End-to-End-AI-for-Science)
* [Pretrained models and datasets]()
* [Self-paced DLI training](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-04+V1)
* [Deep Learnning for Science and Engineering](https://www.nvidia.com/en-us/on-demand/deep-learning-for-science-and-engineering/)
  
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

## Cite Modulus
If Modulus helped your research and you would like to cite it, please refer to the [guidelines](https://github.com/NVIDIA/modulus/blob/main/CITATION.cff)

## Communication

- Github Discussions: Discuss new architectures, implementations, Physics-ML research, etc.
- GitHub Issues: Bug reports, feature requests, install issues, etc.
- Modulus Forum: The [Modulus Forum](https://forums.developer.nvidia.com/c/physics-simulation/modulus-physics-ml-model-framework)
hosts an audience of new to moderate level users and developers for general chat, online
discussions, collaboration, etc.

## Feedback

Want to suggest some improvements to Modulus? Use our feedback form
[here](https://docs.google.com/forms/d/e/1FAIpQLSfX4zZ0Lp7MMxzi3xqvzX4IQDdWbkNh5H_a_clzIhclE2oSBQ/viewform?usp=sf_link).

## License

Modulus is provided under the Apache License 2.0, please see [LICENSE.txt](./LICENSE.txt)
for full license text.
