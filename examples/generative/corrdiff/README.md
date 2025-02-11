<!-- markdownlint-disable -->
# Generative Correction Diffusion Model (CorrDiff) for Km-scale Atmospheric Downscaling

## Table of Contents
- [Generative Correction Diffusion Model (CorrDiff) for Km-scale Atmospheric Downscaling](#generative-correction-diffusion-model-corrdiff-for-km-scale-atmospheric-downscaling)
  - [Table of Contents](#table-of-contents)
  - [Problem overview](#problem-overview)
  - [Getting started with the HRRR-Mini dataset](#getting-started-with-the-hrrr-mini-dataset)
    - [Preliminaries](#preliminaries)
    - [Configuration basics](#configuration-basics)
    - [Training the regression model](#training-the-regression-model)
    - [Training the diffusion model](#training-the-diffusion-model)
    - [Generation](#generation)
  - [Another example: Taiwan dataset](#another-example-taiwan-dataset)
    - [Dataset \& Datapipe](#dataset--datapipe)
    - [Training the models](#training-the-models)
    - [Sampling and Model Evaluation](#sampling-and-model-evaluation)
    - [Logging](#logging)
  - [Training CorrDiff on a Custom Dataset](#training-corrdiff-on-a-custom-dataset)
    - [Dataset Preprocessing](#dataset-preprocessing)
    - [Config Files](#config-files)
    - [FAQs](#faqs)
  - [References](#references)

## Problem overview

To improve weather hazard predictions without expensive simulations, a cost-effective
stochasticdownscaling model, [CorrDiff](https://arxiv.org/abs/2309.15214), is trained
using high-resolution weather data
and coarser ERA5 reanalysis. CorrDiff employs a two-step approach with UNet and diffusion
to address multi-scale challenges, showing strong performance in predicting weather
extremes and accurately capturing multivariate relationships like intense rainfall and
typhoon dynamics, suggesting a promising future for global-to-km-scale machine learning
weather forecasts.

<p align="center">
<img src="../../../docs/img/corrdiff_cold_front.png"/>
</p>

## Getting started with the HRRR-Mini dataset

To get familiar with CorrDiff, you can start by training the "Mini" version of
CorrDiff, which uses smaller training samples and a smaller network to reduce
training costs from thousands of GPU hours to around 10 hours on A100 GPUs
while still producing reasonable results. It also includes a simple data loader
that can be used as a baseline for training CorrDiff on custom datasets. Note
that CorrDiff-Mini is only for debugging and education purpose, and the
accuracy of its predictions should not be trusted.

### Preliminaries
Start by installing Modulus (if not already installed) and copying this folder (`examples/generative/corrdiff`) to a system with a GPU available. Also download the CorrDiff-Mini dataset from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_datasets-hrrr_mini).

### Configuration basics

CorrDiff training is handled by `train.py` and controlled by YAML configuration files handled by [Hydra](https://hydra.cc/docs/intro/). Prebuilt configuration files are found in the `conf` directory. You can choose the configuration file using the `--config-name` option. The main configuration file specifies the training dataset, the model configuration and the training options. The details of these are given in the corresponding configuration files. To change a configuration option, you can either edit the configuration files or use the Hydra command line overrides. For example, the training batch size is controlled by the option `training.hp.total_batch_size`. We can override this from the command line with the `++` syntax: `python train.py ++training.hp.total_batch_size=64` would set run the training with the batch size set to 64.

### Training the regression model
CorrDiff requires a two-step training, where a deterministic regression model
is first trained, followed by a diffusion model. The pre-trained regression
model is necessary to train the diffusion model. To train the CorrDiff-Mini
regression model, we use the main configuration file
[config_training_mini_regression.yaml](conf/config_training_mini_regression.yaml).
This includes the following components:
* The HRRR-Mini dataset: [conf/dataset/hrrrmini.yaml](conf/dataset/hrrrmini.yaml)
* The CorrDiff-Mini regression model: [conf/model/corrdiff_regression_mini.yaml](conf/model/corrdiff_regression_mini.yaml)
* The CorrDiff-Mini regression training options: [conf/training/corrdiff_regression_mini.yaml](conf/training/corrdiff_regression_mini.yaml)
  
To start the training, run:
```bash
python train.py --config-name=config_training_mini_regression.yaml ++dataset.data_path=</path/to/dataset>/hrrr_mini_train.nc ++dataset.stats_path=</path/to/dataset>/stats.json
```
where you should replace both instances of `</path/to/dataset>` with the absolute path to the directory containing the downloaded HRRR-Mini dataset.

The training will require a few hours on a single A100 GPU. If training is interrupted, it will automatically continue from the latest checkpoint when restarted. Multi-GPU and multi-node training are supported and will launch automatically when the training is run in a `torchrun` or MPI environment.

The results, including logs and checkpoints, are saved by default to `outputs/mini_generation/`. You can direct the checkpoints to be saved elsewhere by setting: `++training.io.checkpoint_dir=</path/to/checkpoints>`.

> **_Out of memory?_** CorrDiff-Mini trains by default with a batch size of 256 (set by `training.hp.total_batch_size`). If you're using a single GPU, especially one with a smaller amout of memory, you might see out-of-memory error. If that happens, set a smaller batch size per GPU, e.g.: `++training.hp.batch_size_per_gpu=64`. CorrDiff training will then automatically use gradient accumulation to train with an effective batch size of `training.hp.total_batch_size`.

### Training the diffusion model

The pre-trained regression model is needed to train the diffusion model. Assuming you trained the regression model for the default 2 million samples, the final checkpoint will be `checkpoints_regression/UNet.0.2000000.mdlus`.
Save the final regression checkpoint into a new location, then run:
```bash
python train.py --config-name=config_training_mini_diffusion.yaml ++dataset.data_path=</path/to/dataset>/hrrr_mini_train.nc ++dataset.stats_path=</path/to/dataset>/stats.json ++training.io.regression_checkpoint_path=</path/to/regression/model>
```
where `</path/to/regression/model>` should point to the saved regression checkpoint.

Once the training is completed, copy the latest checkpoint (`checkpoints_diffusion/EDMPrecondSR.0.8000000.mdlus`) to a file.

### Generation

Use the `generate.py` script to generate samples with the trained networks:
```bash
python generate.py --config-name="config_generate_mini.yaml" ++generation.io.res_ckpt_filename=</path/to/diffusion/model> ++generation.io.reg_ckpt_filename=</path/to/regression/model> ++generation.io.output_filename=</path/to/output/file>
```
where `</path/to/regression/model>` and `</path/to/diffusion/model>` should point to the regression and diffusion model checkpoints, respectively, and `</path/to/output/file>` indicates the output NetCDF4 file.

You can open the output file with e.g. the Python NetCDF4 library. The inputs are saved in the `input` group of the file, the ground truth data in the `truth` group, and the CorrDiff prediction in the `prediction` group.


## Another example: Taiwan dataset

### Dataset & Datapipe

In this example, CorrDiff training is demonstrated on the Taiwan
high-resolution dataset,
conditioned on the low-resolution [ERA5 dataset](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5).
We have made this dataset available for non-commercial use under the
[CC BY-NC-ND 4.0 license](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en)
and can be downloaded from [https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_datasets_cwa](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_datasets_cwa)
by `ngc registry resource download-version "nvidia/modulus/modulus_datasets_cwa:v1"`.


### Training the models


Similarly to the CorrDiff-Mini example, there are several models that can be
trained in this example. Those include
a regression model, a diffusion model, and a patched-based diffusion model.
The patch-based diffusion model uses small subsets of the target region during
both training and generation to enhance the scalability of the model and reduce
memory usage. The config files to train these models are:
1. [config_training_taiwan_regression.yaml](./conf/config_training_taiwan_regression.yaml)
   to train the regression model on the Taiwan dataset.
2. [config_training_taiwan_diffusion.yaml](./conf/config_training_taiwan_diffusion.yaml)
   to train the diffusion or patched-based diffusion models on the Taiwan dataset.
Additional parameters can be adjusted in [`conf/model`](./conf/model/) or [`conf/training`](./conf/training/)
These can be adjusted accordingly depending on
whether you are training the regression, diffusion, or the patch-based
diffusion model. Note that training the variants of the diffusion model
requires a trained regression checkpoint, and the path to that checkpoint should
be included in the [`conf/training/corrdiff_diffusion.yaml`](./conf/training/corrdiff_diffusion.yaml) file.

To train the regression model, run

```python train.py --config-name=config_training_taiwan_regression```

You can change the `--config-name` accordingly to train the diffusion model.

Data parallelism is supported. Use `torchrun`
To launch a multi-GPU or multi-node training:

```torchrun --standalone --nnodes=<NUM_NODES> --nproc_per_node=<NUM_GPUS_PER_NODE> train.py```

### Sampling and Model Evaluation

Model evaluation is split into two components. `generate.py` creates a netCDF file
for the generated outputs, and `score_samples.py` computes deterministic and probablistic
scores.

To generate samples and save output in a netCDF file, run:

```bash
python generate.py
```
This will use the base configs specified in the `conf/config_generate.yaml` file.

Next, to score the generated samples, run:

```bash
python score_samples.py path=<PATH_TO_NC_FILE> output=<OUTPUT_FILE>
```

Some legacy plotting scripts are also available in the `inference` directory.
You can also bring your checkpoints to [earth2studio](https://github.com/NVIDIA/earth2studio)
for further anaylysis and visualizations.

### Logging

We use TensorBoard for logging training and validation losses, as well as
the learning rate during training. To visualize TensorBoard running in a
Docker container on a remote server from your local desktop, follow these steps:

1. **Expose the Port in Docker:**
     Expose port 6006 in the Docker container by including
     `-p 6006:6006` in your docker run command.

2. **Launch TensorBoard:**
   Start TensorBoard within the Docker container:
     ```bash
     tensorboard --logdir=/path/to/logdir --port=6006
     ```

3. **Set Up SSH Tunneling:**
   Create an SSH tunnel to forward port 6006 from the remote server to your local machine:
     ```bash
     ssh -L 6006:localhost:6006 <user>@<remote-server-ip>
     ```
    Replace `<user>` with your SSH username and `<remote-server-ip>` with the IP address
    of your remote server. You can use a different port if necessary.

4. **Access TensorBoard:**
   Open your web browser and navigate to `http://localhost:6006` to view TensorBoard.

**Note:** Ensure the remote server’s firewall allows connections on port `6006`
and that your local machine’s firewall allows outgoing connections.

## Training CorrDiff on a Custom Dataset

This repository includes examples of **CorrDiff** training on specific datasets, such as **Taiwan** and **HRRR**. However, many use cases require training **CorrDiff** on a **custom high-resolution dataset**. The steps below outline the process.

### Dataset Preprocessing

Before training CorrDiff on a custom dataset, you must **preprocess the data and build a datapipe**. There are two approaches:

1. **Convert and reformat your dataset to match an existing CorrDiff datapipe.**  
   It is recommended to structure your dataset to be compatible with either:  
   - **[HRRR-Mini datapipe](./datasets/hrrrmini.py)** (uses [Zarr format](https://zarr.readthedocs.io/en/stable/))
   - **[Taiwan datapipe](./datasets/cwb.py)** (uses [NetCDF format](https://unidata.github.io/netcdf4-python/))  
   
   In this approach, you only need to write a **data conversion script**, as one of these existing datapipes can be reused.

2. **Create a custom datapipe for your specific file format.**  
   In this approach, you do not need to convert your data to a specific format,
   but you are responsible for implementing your own custom datapipe.
   The recommended starting point to implement a new datapipe is either the **[HRRR-Mini](./datasets/hrrrmini.py)** or **[Taiwan](./datasets/cwb.py)** datapipe.
   

If training a **patch-based diffusion model**, selecting the appropriate **patch size** is crucial.  
The relevant hyperparameters are `patch_shape_x` and `patch_shape_y`. These are defined in **[conf/custom/config_training.yaml](./conf/custom/config_training.yaml)**. Ideally, patch size should be based on an **auto-correlation plot**, ensuring
it corresponds to the distance where auto-correlation drops to zero. Some
helper functions (`average_power_spectrum` and `power_spectra_to_acf`) are
available in [inference/power_spectra.py](./inference/power_spectra.py) to compute the autocorrelation function.

### Config Files

The [`conf`](./conf/) directory contains preset configuration files for the model, dataset, training, and other components. These configs are written in YAML format and managed using the `omegaconf` library. Several preset configurations are available to reproduce training on specific datasets (e.g., HRRR-Mini, GEFS-HRRR).

You can specify the configuration file using the `--config-name` option.

For training on a custom dataset, it is recommended to use the configs in the [`conf/custom`](./conf/custom) directory. The [`config_training.yaml`](./conf/custom/config_training.yaml) file serves as the main entry point for custom training. It includes all required parameters while loading recommended defaults for others. In most cases, modifying [`config_training.yaml`](./conf/custom/config_training.yaml) is sufficient. 

For advanced use cases, expert users can override specific parameters in the
command line, such as adjusting the gradient clipping threshold `++training.hp.grad_clip_threshold=1e5`.
This reduces the threshold to `1e5` instead of its default value. 

To use this custom config file during training, run:

```bash
python train.py --config-path=conf/custom --config-name=config_training
```

### FAQs

1. **Is it preferable to re-train from a pre-trained checkpoint or train from scratch?**  
   Trained checkpoints are available through NVIDIA AI Enterprise. For example, a trained model for the continental United States on the GEFS-HRRR dataset is available [here](https://build.nvidia.com/nvidia/corrdiff/modelcard). It is generally recommended to start training from a checkpoint rather than from scratch if the following conditions are met:
   - Your custom dataset covers a region included in the training data of the checkpoint (e.g., a sub-region of the continental United States for the checkpoint mentioned above).
   - At most half of the variables in your dataset are also included in the training data of the checkpoint.

   Training from scratch is recommended for all other cases.

2. **How many samples are needed to train a CorrDiff model?**  
   The more, the better. As a rule of thumb, at least 50,000 samples are necessary.  
   *Note: For patch-based diffusion, each patch can be counted as a sample.*

3. **How many GPUs are required to train CorrDiff?**  
   A single GPU is sufficient as long as memory is not exhausted, but this may
   result in extremely slow training. To accelerate training, CorrDiff
   leverages distributed data parallelism. The total training wall-clock time
   roughly decreases linearly with the number of GPUs. Most CorrDiff training
   examples have been conducted with 64 A100 GPUs. If you encounter an
   out-of-memory error, reduce `batch_size_per_gpu` in the
   [`config_training.yaml`](./conf/custom/config_training.yaml) or, for
   patch-based
   diffusion models, decrease the patch size—ensuring it remains larger than
   the auto-correlation distance.

4. **How long does it take to train CorrDiff on a custom dataset?**  
   Training CorrDiff on the continental United States dataset required
   approximately 5,000 A100 GPU hours. This corresponds to roughly 80 hours of
   wall-clock time with 64 GPUs. You can expect the cost to scale
   linearly with the number of samples available.

5. **What are CorrDiff's current limitations for custom datasets?**  
   The main limitation of CorrDiff is the maximum _downscaling ratio_ it can
   achieve. For a purely spatial super-resolution task (where input and output variables are the same), CorrDiff can reliably achieve a maximum resolution scaling of ×16. If the task involves inferring new output variables, the maximum reliable spatial super-resolution is ×11.

6. **What does a successful training look like?**  
   In a successful training run, the loss function should decrease monotonically, as shown below:  
  <p align="center">
<img src="../../../docs/img/corrdiff_training_loss.png"/>
</p>

7. **Which hyperparameters are most important?**  
   One of the most crucial hyperparameters is the patch size for a patch-based
   diffusion model (`patch_shape_x` and `patch_shape_y` in
   [`config_training.yaml`](./conf/custom/config_training.yaml)). A larger
   patch size increases computational cost and GPU memory requirements, while a
   smaller patch size may lead to a loss of physical information. The patch
   size should not be smaller than the auto-correlation distance, which can be
   determined using the auto-correlation plotting utility. Other
   hyper-parameters have been thoroughly validated and should only me modified
   by expert users.


## References

- [Residual Diffusion Modeling for Km-scale Atmospheric Downscaling](https://arxiv.org/pdf/2309.15214.pdf)
- [Elucidating the design space of diffusion-based generative models](https://openreview.net/pdf?id=k7FuTOWMOc7)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/pdf/2011.13456.pdf)
