<!-- markdownlint-disable -->
## StormCast: Kilometer-Scale Convection Allowing Model Emulation using Generative Diffusion Modeling

**Note: this example is an initial release of the StormCast code and will be heavily refactored in future releases**

## Problem overview

Convection-allowing models (CAMs) are essential tools for forecasting severe thunderstorms and 
mesoscale convective systems, which are responsible for some of the most extreme weather events. 
By resolving kilometer-scale convective dynamics, these models provide the precision needed for 
accurate hazard prediction. However, modeling the atmosphere at this scale is both challenging
and expensive.

This example demonstrates how to run training and simple inference for [StormCast](https://arxiv.org/abs/2408.10958),
a generative diffusion model designed to emulate NOAA’s High-Resolution Rapid Refresh (HRRR) model, a 3km 
operational CAM. StormCast autoregressively predicts multiple atmospheric state variables with remarkable
accuracy, demonstrating ability to replicate storm dynamics, observed radar reflectivity, and realistic
atmospheric structure via deep learning-based CAM emulation. StormCast enables high-resolution ML-driven
regional weather forecasting and climate risk analysis.


<p align="center">
<img src="../../../docs/img/stormcast_rollout.gif"/>
</p>

## Getting started

### Preliminaries
Start by installing Modulus (if not already installed) and copying this folder (`examples/generative/stormcast`) to a system with a GPU available. Also, prepare a combined HRRR/ERA5 dataset in the form specified in `utils/data_loader_hrrr_era5.py` (**Note: subsequent versions of this example will include more detailed dataset preparation instructions**).

### Configuration basics

StormCast training is handled by `train.py` and controlled by a YAML configuration file in `config/config.yaml` and command line arguments. You can choose the configuration file using the `--config_file` option, and a specific configuration within that file with the `--config-name` option. The main configuration file specifies the training dataset, the model configuration and the training options. To change a configuration option, you can either edit the existing configurations directly or make new ones by inheriting from the existing configs and overriding specific options. For example, one could create a new config for training the diffusion model in StormCast by creating a new config that inherits from the existing `diffusion` config in `config/config.yaml`:
```
diffusion_bs64:
  <<: *diffusion
  batch_size: 1
```

The basic configuration file currently contains configurations for just the `regression` and `diffusion` components of StormCast. Note any diffusion model you train will need a pretrained regression model to use, due to how StormCast is designed (you can refer to the paper for more details), thus there are two config items that must be defined to train a diffusion model:
  1. `regression_weights` -- The path to a checkpoint with model weights for the regression model. This file should be a pytorch checkpoint saved by your training script, with the `state_dict` for the regression network saved under the `net` key.
  2. `regression_config` -- the config name used to train this regression model

All configuration items related to the dataset are also contained in `config/config.yaml`, most importantly the location on the filesystem of the prepared HRRR/ERA5 Dataset (see [Dataset section](#dataset) for details).

There is also a model registry `config/registry.json` which can be used to index different model versions to be used in inference/evaluation. For simplicity, there is just a single model version specified there currently, which matches the StormCast model used to generate results in the paper.

### Training the regression model
To train the StormCast regression model, we use the default configuration file `config.yaml` and specify the `regression` config, along with the `--outdir` argument to choose where training logs and checkpoints should be saved. 
We also can use command line options defined in `train.py` to specify other details, like a unique run ID to use for the experiment (`--run_id`). On a single GPU machine, for example, run:
```bash
python train.py --outdir rundir --config_file ./config/config.yaml --config_name regression --run_id 0
```

This will initialize training experiment and launch the main training loop, which is defined in `utils/diffusions/training_loop.py`.

### Training the diffusion model

The method for launching a diffusion model training looks almost identical, and we just have to change the configuration name appropriately. However, since we need a pre-trained regression model for the diffusion model training, this config must define `regression_pickle` to point to a compatible pickle file with network weights for the regression model. Once that is taken care of, launching diffusion training looks nearly identical as previously:
```bash
python train.py --outdir rundir --config_file ./config/config.yaml --config_name diffusion --run_id 0
```

Note that the full training pipeline for StormCast is fairly lengthy, requiring about 120 hours on 64 NVIDIA H100 GPUs. However, more lightweight trainings can still produce decent models if the diffusion model is not trained for as long. 

Both regression and diffusion training can be distributed easily with data parallelism via `torchrun`. One just needs to ensure the configuration being run has a large enough batch size to be distributed over the number of available GPUs/processes. The example `regression` and `diffusion` configs in `config/config.yaml` just use a batch size of 1 for simplicity, but new configs can be easily added [as described above](#configuration-basics). For example, distributed training over 8 GPUs on one node would look something like:
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --outdir rundir --config_file ./config/config.yaml --config_name <your_distributed_training_config> --run_id 0
```

Once the training is completed, you can enter a new model into `config/registry.json` that points to the checkpoints (`.pt` file in your training output directory), and you are ready to run inference.

### Inference

A simple demonstrative inference script is given in `inference.py`, which loads a pretrained model from a local directory named `stormcast_checkpoints`.
Yout should update this path to the checkpoints saved by your training runs that you want to run inference for.
The `inference.py` script will run a 12-hour forecast and save outputs as a `zarr` file along with a few plots saved as `png` files.

To run inference, simply do:

```bash
python inference.py
```
This inference script is configured by the contents of a model registry, which specifies config files and names to use for each of the diffusion and regression networks, along with other inference options which specify architecure types and a short description of the model. The `inference.py` script will automatically use the default file for the model registry (`config/registry.json`) and evaluate the `stormcast` example model, but you can configure it to run your desired inference case(s) with the following command-line options:
```bash
  --outdir DIR          Where to save the results
  --registry_file FILE  Path to model registry file
  --model_name MODEL    Name of model to evaluate from the registry
```

We also recommend bringing your checkpoints to [earth2studio](https://github.com/NVIDIA/earth2studio)
for further anaylysis and visualizations.


## Dataset

In this example, StormCast is trained on the [HRRR dataset](https://rapidrefresh.noaa.gov/hrrr/),
conditioned on the [ERA5 dataset](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5).
The datapipe in this example is tailored specifically for the domain and problem setting posed in the 
[original StormCast preprint](https://arxiv.org/abs/2408.10958), namely a subset of HRRR and ERA5 variables
in a region over the Central US with spatial extent 1536km x 1920km.


A custom dataset object is defined in `utils/data_loader_hrrr_era5.py`, which loads temporally-aligned samples from HRRR and ERA5, interpolated to the same grid and normalized appropriately. This data pipeline requries the HRRR and ERA5 data to abide by a specific `zarr` format and for other datasets, you will need to create a custom datapipe. The table below lists the variables used to train StormCast -- in total there are 26 ERA5 variables used and 99 HRRR variables (along with 2 static HRRR invariants, the land/water mask and orography).

#### ERA5 Variables
| Parameter                             | Pressure Levels (hPa)     | Height Levels (m) |
|---------------------------------------|---------------------------|--------------------|
| Zonal Wind (u)                        | 1000, 850, 500, 250       | 10                 |
| Meridional Wind (v)                   | 1000, 850, 500, 250       | 10                 |
| Geopotential Height (z)               | 1000, 850, 500, 250       | None               |
| Temperature (t)                       | 1000, 850, 500, 250       | 2                  |
| Humidity (q)                          | 1000, 850, 500, 250       | None               |
| Total Column of Water Vapour (tcwv)   | Integrated                | -                  |
| Mean Sea Level Pressure (mslp)        | Surface                   | -                  |
| Surface Pressure (sp)                 | Surface                   | -                  |


#### HRRR Variables
| Parameter                             | Hybrid Model Levels (Index)                               | Height Levels (m) |
|---------------------------------------|-----------------------------------------------------------|--------------------|
| Zonal Wind (u)                        | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30    | 10                 |
| Meridional Wind (v)                   | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30    | 10                 |
| Geopotential Height (z)               | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30    | None               |
| Temperature (t)                       | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30    | 2                  |
| Humidity (q)                          | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30    | None               |
| Pressure (p)                          | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20            | None               |
| Max. Composite Radar Reflectivity     | -                                                         | Integrated         |
| Mean Sea Level Pressure (mslp)        | -                                                         | Surface            |
| Orography                             | -                                                         | Surface            |
| Land/Water Mask                       | -                                                         | Surface            |


## Logging

These scripts use Weights & Biases for experiment tracking, which can be enabled by passing the `--log_to_wandb` argument to `train.py`. Academic accounts are free to create at [wandb.ai](https://wandb.ai/).
Once you have an account set up, you can adjust `entity` and `project` in `train.py` to the appropriate names for your `wandb` workspace.


## References

- [Kilometer-Scale Convection Allowing Model Emulation using Generative Diffusion Modeling](https://arxiv.org/abs/2408.10958)
- [Elucidating the design space of diffusion-based generative models](https://openreview.net/pdf?id=k7FuTOWMOc7)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/pdf/2011.13456.pdf)

