<!-- markdownlint-disable -->
# Spherical Fourienr Neural Operator (SFNO) for Weather and Climate Forecasting (Experimental version, Modulus integration in progress)

[**Overview**](#overview) | [**Getting started**](#getting-started) | [**Further reading**](#further-reading) | [**References**](#references)

## Overview

This is a research code built for massively parallel training of SFNO for weather and climate forecasting on 100+ GPUs and to enable the development of the next generation of weather and climate models. The code is written in PyTorch and supports various forms of model- and data-parallelism, asynchronous loading of data, unpredicted channels, autoregressive training and much more.

![SFNO](https://developer-blogs.nvidia.com/wp-content/uploads/2023/07/figure_1.11-2.gif)

## Getting started

**For distributed training or inference, run `patch_pytorch.sh` in advance. This will patch the pytorch distributed utilities to support complex values.**

## Installing optional dependencies

Install the optional dependencies by running

```bash
pip install jsbeautifier xskillscore numba mpi4py
ENV NUMBA_DISABLE_CUDA=1
```

To enable MPI for HDF5 for accelerated file preprocessing, we need a custom build of h5py:

```bash
# uninstall h5py
pip uninstall h5py -y

# upgrade cmake
apt remove cmake -y
pip install cmake --upgrade

# hdf5
cd /tmp
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.14/hdf5-1.14.2/src/hdf5-1.14.2.tar.gz
gzip -cd hdf5-1.14.2.tar.gz | tar xvf -
mkdir hdf5-1.14.2/build
cd hdf5-1.14.2/build
cmake -DCMAKE_INSTALL_PREFIX=/opt/hdf5 \
    -DHDF5_ENABLE_DIRECT_VFD=1 \
    -DHDF5_ENABLE_PARALLEL=1 \
    -DHDF5_TEST_API=1 \
    -DHDF5_TEST_VFD=1 \
    ..
make -j 8 && make install

# h5py
CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/opt/hdf5 pip install --no-binary=h5py h5py

# export the environment variable
export ENABLE_H5PY_ODIRECT=True
```

If needed, you can also install some useful scripts from mlperf

```bash
pip install --ignore-installed "git+https://github.com/NVIDIA/mlperf-common.git"
```

### Training

Training is launched by calling `train.py` and passing it the necessary CLI arguments to specify the configuration file `--yaml_config` and he configuration target `--config`:

```bash
mpirun -np 8 --allow-run-as-root python -u train.py --yaml_config="config/sfnonet_devel.yaml" --config="sfno_linear_73chq_sc3_layers8_edim384_asgl2"
```

SFNO code supports various optimization to fit large models ino GPU memory and enable computationally efficient training. An overview of these features and corresponding CLI arguments is provided in the following table:

| Feature                   | CLI argument                                  | options                      |
|---------------------------|-----------------------------------------------|------------------------------|
| Automatic Mixed Precision | `--amp_mode`                                  | `none`, `fp16`, `bf16`       |
| Just-in-time compilation  | `--jit_mode`                                  | `none`, `script`, `inductor` |
| CUDA graphs               | `--cuda_graph_mode`                           | `none`, `fwdbwd`, `step`     |
| Activation checkpointing  | `--checkpointing_level`                       | 0,1,2,3                      |
| Data parallelism          | `--batch_size`                                | 1,2,3,...                    |
| Channel parallelism       | `--fin_parallel_size`, `--fout_parallel_size` | 1,2,3,...                    |
| Spatial model parallelism | `--h_parallel_size`, `--w_parallel_size`      | 1,2,3,...                    |
| Multistep training        | `--multistep_count`                           | 1,2,3,...                    |

Especially larger models are enabled by using a mix of these techniques. Spatial model parallelism splits both the model and the data onto multiple GPUs, thus reducing both the memory footprint of the model and the load on the IO as each rank only needs to read a fraction of the data. A typical "large" training run of SFNO can be launched by running

```bash
mpirun -np 256 --allow-run-as-root python -u train.py --amp_mode=bf16 --cuda_graph_mode=fwdbwd --multistep_count=1 --run_num="ngpu256_sp4" --yaml_config="config/sfnonet_devel.yaml" --config="sfno_linear_73chq_sc3_layers8_edim384_asgl2" --h_parallel_size=4 --w_parallel_size=1 --batch_size=64
```

Here we train the model on 256 GPUs, split horizontally across 4 ranks with a batch size of 64, which amounts to a local batch size of 1/4. Memory requirements are further reduced by the use of `bf16` automatic mixed precision.

### Inference

In a similar fashion to training, inference can be called from the CLI by calling `inference.py` and handled by `inferencer.py`. To launch inference on the out-of-sample dataset, we can call:

```bash
mpirun -np 256 --allow-run-as-root python -u inference.py --amp_mode=bf16 --cuda_graph_mode=fwdbwd --multistep_count=1 --run_num="ngpu256_sp4" --yaml_config="config/sfnonet_devel.yaml" --config="sfno_linear_73chq_sc3_layers8_edim384_asgl2" --h_parallel_size=4 --w_parallel_size=1 --batch_size=64
```

By default, the inference script will perform inference on the out-of-sample dataset specified

### Project structure

The project is structured as follows:

```
sfno
├── ...
├── config                  # configuration files, also known as recipes
├── data_process            # data pre-processing such as computation of statistics
├── datasets                # dataset utility scripts
├── docker                  # scripts for building a docker image for training
├── inference               # contains the inferencer
├── mpu                     # utilities for model parallelism
├── networks                # networks, contains definitions of various ML models
├── tests                   # test files
├── third_party/climt       # third party modules
│   └── zenith_angle.py     # computation of zenith angle
├── utils                   # utilities
│   ├── dataloaders         # contains various dataloaders
│   ├── metrics             # metrics folder contains routines for scoring and benchmarking.
│   ├── ...
│   ├── comm.py             # comms module for orthogonal communicator infrastructure
│   ├── dataloader.py       # dataloader interface
│   ├── metric.py           # centralized metrics handler
│   ├── trainer_profile.py  # copy of trainer.py used for profiling
│   └── trainer.py          # main file for handling training
├── ...
├── inference.py            # CLI script for launching inference
├── train.py                # CLI script for launching training
└── README.md               # this file
```

### Model and Training configuration

Model training in is specified through the use of `.yaml` files located in the `config` folder. The corresponding models are located in `networks` and registered in the `get_model` routine in `networks/models.py`. The following table lists the most important configuration options.

| Configuration Key         | Description                                             | Options                                                 |
|---------------------------|---------------------------------------------------------|---------------------------------------------------------|
| `nettype`                 | Network architecture.                                   | `sfno`, `fno`, `afno`, `unet`                           |
| `loss`                    | Loss function.                                          | `l2`, `geometric l2`, ...                               |
| `optimizer`               | Optimizer to be used.                                   | `sfno`, `fno`, `afno`, `unet`                           |
| `lr`                      | Initial learning rate.                                  | float > 0.0                                             |
| `batch_size`              | Batch size.                                             | integer > 0                                             |
| `max_epochs`              | Number of epochs to train for                           | integer                                                 |
| `scheduler`               | Learning rate scheduler to be used.                     | `None`, `CosineAnnealing`, `ReduceLROnPlateau`, `StepLR`|
| `lr_warmup_steps`         | Number of warmup steps for the learning rate scheduler. | integer >= 0                                            |
| `weight_decay`            | Weight decay.                                           | float                                                   |
| `train_data_path`         | Directory path which contains the training data.        | string                                                  |
| `test_data_path`          | Network architecture.                                   | string                                                  |
| `exp_dir`                 | Directory path for ouputs such as model checkpoints.    | string                                                  |
| `metadata_json_path`      | Path to the metadata file `data.json`.                  | string                                                  |
| `channel_names`           | Channels to be used for training.                       | List[string]                                            |

For a more comprehensive overview, we suggest looking into existing `.yaml` configurations. More details about the available configurations can be found in [this file](config/README.md).

### Training data

This model expects the training/test data in HDF5 format, where each file contains the data for an entire year. The dataloaders will then load the input `inp` and the target `tar`, which correspond to the state of the atmosphere at a given point in time and at a later time for the target. The time difference between input and target is determined by the parameter `dt`, which determines how many steps the two are apart. The physical time difference is determined by the temporal resolution `dhours` of the dataset.

The model requires a metadata file named `data.json`, which describes important properties of the dataset such as the HDF5 variable name that contains the data. Another example are channels to load in the dataloader, which arespecified via channel names. The metadata file has the following structure:

```json
{
    "dataset_name": "give this dataset a name",     # name of the dataset
    "attrs": {                                      # optional attributes, can contain anything you want
        "decription": "description of the dataset",
        "location": "location of your dataset"
    },
    "h5_path": "fields",                            # variable name of the data inside the hdf5 file
    "dims": ["time", "channel", "lat", "lon"],      # dimensions of fields contained in the dataset
    "dhours": 6,                                    # temporal resolution in hours
    "coord": {                                      # coordinates and channel descriptions
        "grid_type": "equiangular",                 # type of grid used in dataset: currently suppported choices are 'equiangular' and 'legendre-gauss'
        "lat": [0.0, 0.1, ...],                     # latitudinal grid coordinates
        "lon": [0.0, 0.1, ...],                     # longitudinal grid coordinates
        "channel": ["t2m", "u10", "v10", ...]       # names of the channels contained in the dataset
    }
}
```

### Model packages

By default, the code will save out a model package when training starts. Model packages allow easily contain all the necessary data to run the model. This includes statistics used to normalize inputs and outputs, unpredicted static channels and even the code which appends celestial features such as the cosine of the solar zenith angle. Read more about model packages [here](networks/Readme.md).

## Further reading

- [NVIDIA blog article](https://developer.nvidia.com/blog/modeling-earths-atmosphere-with-spherical-fourier-neural-operators/) on Spherical Fourier Neural Operators for ML-based weather prediction
- [torch-harmonics](https://github.com/NVIDIA/torch-harmonics), a library for differentiable Spherical Harmonics in PyTorch
- [Apex](https://github.com/NVIDIA/apex), tools for easier mixed precision
- [Dali](https://developer.nvidia.com/dali), NVIDIA data loading library
- [Modulus](https://developer.nvidia.com/modulus), NVIDIA's library for physics-ML
- [earth2mip](https://github.com/NVIDIA/earth2mip), a library for intercomparing DL based weather models

## Authors

<img src="https://www.nvidia.com/content/dam/en-zz/Solutions/about-nvidia/logo-and-brand/01-nvidia-logo-horiz-500x200-2c50-d@2x.png"  height="120px">
<img src="https://www.nersc.gov/assets/Logos/NERSClogocolor.png"  height="120px">

The code was developed by [Thorsten Kurth](https://github.com/azrael417), [Boris Bonev](https://bonevbs.github.io), [Jaideep Pathak](https://scholar.google.com/citations?user=cevw0gkAAAAJ&hl=en), [Jean Kossaifi](http://jeankossaifi.com), [Noah Brenowitz](https://www.noahbrenowitz.com), Animashree Anandkumar, Kamyar Azizzadenesheli, Ashesh Chattopadhyay, Yair Cohen, David Hall, Peter Harrington, Pedram Hassanzadeh, Christian Hundt, Karthik Kashinath, Zongyi Li, Morteza Mardani, Mike Pritchard, David Pruitt, Sanjeev Raja, Shashank Subramanian.

## References

<a id="#sfno_paper">[1]</a> 
Bonev B., Kurth T., Hundt C., Pathak, J., Baust M., Kashinath K., Anandkumar A.;
Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere;
arXiv 2306.0383, 2023.

<a id="1">[2]</a> 
Pathak J., Subramanian S., Harrington P., Raja S., Chattopadhyay A., Mardani M., Kurth T., Hall D., Li Z., Azizzadenesheli K., Hassanzadeh P., Kashinath K., Anandkumar A.;
FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators;
arXiv 2202.11214, 2022.

## Citation

If you use this package, please cite

```bibtex
@InProceedings{bonev2023sfno,
    title={Spherical {F}ourier Neural Operators: Learning Stable Dynamics on the Sphere},
    author={Bonev, Boris and Kurth, Thorsten and Hundt, Christian and Pathak, Jaideep and Baust, Maximilian and Kashinath, Karthik and Anandkumar, Anima},
    booktitle={Proceedings of the 40th International Conference on Machine Learning},
    pages={2806--2823},
    year={2023},
    volume={202},
    series={Proceedings of Machine Learning Research},
    month={23--29 Jul},
    publisher={PMLR},
}

@article{pathak2022fourcastnet,
    title={Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators},
    author={Pathak, Jaideep and Subramanian, Shashank and Harrington, Peter and Raja, Sanjeev and Chattopadhyay, Ashesh and Mardani, Morteza and Kurth, Thorsten and Hall, David and Li, Zongyi and Azizzadenesheli, Kamyar and Hassanzadeh, Pedram and Kashinath, Karthik and Anandkumar, Animashree},
    journal={arXiv preprint arXiv:2202.11214},
    year={2022}
}
```
