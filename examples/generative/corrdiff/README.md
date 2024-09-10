<!-- markdownlint-disable -->
## Generative Correction Diffusion Model (CorrDiff) for Km-scale Atmospheric Downscaling

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

## Configs

The `conf` directory contains the configuration files for the model, data,
training, etc. The configs are given in YAML format and use the `omegaconf`
library to manage them. Several example configs are given for training
different models that are regression, diffusion, and patched-based diffusion
models.
The default configs are set to train the regression model.
To train the other models, please adjust `conf/config_training.yaml`
according to the comments. Alternatively, you can create a new config file
and specify it using the `--config-name` option.


## Dataset & Datapipe

In this example, CorrDiff training is demonstrated on the Taiwan dataset,
conditioned on the [ERA5 dataset](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5).
We have made this dataset available for non-commercial use under the
[CC BY-NC-ND 4.0 license](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en)
and can be downloaded from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/cwa_dataset.
The datapipe in this example is tailored specifically for the Taiwan dataset.
A light-weight datapipe for the HRRR dataset is also available and can be used
with the CorrDiff-mini model.
For other datasets, you will need to create a custom datapipe.
You can use the lightweight HRRR datapipe as a starting point for developing your new one.


## Training the models


There are several models available for training in this example, including
a regression, a diffusion, and a patched-based diffusion model.
The Patch-based diffusion model uses small subsets of the target region during
both training and generation to enhance the scalability of the model.
Apart from the dataset configs the main configs for training are `model`,
`training`, and `validation`. These can be adjusted accordingly depending on
whether you are training the regression, diffusion, or the patch-based
diffusion model. Note that training the varients of the diffusion model
requres a trained regression checkpoint, and the path to that checkpoint should
be included in the `conf/training/corrdiff_diffusion.yaml ` file.
Therefore, you should start with training
a regression model, followed by training a diffusion model. To choose which model
to train, simply change the configs in `conf/config_training.yaml`.

For training the regression model, your `config_training.yaml` should be:

```
hydra:
  job:
    chdir: true
    name: regression
  run:
    dir: ./outputs/${hydra:job.name}

defaults:

  # Dataset
  - dataset/cwb_train

  # Model
  - model/corrdiff_regression

  # Training
  - training/corrdiff_regression

  # Validation
  - validation/basic
  ```

Similarly, for taining of the diffusion model, you should have:

```
hydra:
  job:
    chdir: true
    name: diffusion
  run:
    dir: ./outputs/${hydra:job.name}

defaults:

  # Dataset
  - dataset/cwb_train

  # Model
  - model/corrdiff_diffusion

  # Training
  - training/corrdiff_diffusion

  # Validation
  - validation/basic
```

To train the model, run

```python train.py```

You can monitor the training progress using TensorBoard.
Open a new terminal, navigate to the example directory, and run:

```tensorboard --logdir=outputs/<job_name>```

If using a shared cluster, you may need to forward the port to see the tensorboard logs.
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
You can also bring your checkpoints to [earth2studio]<https://github.com/NVIDIA/earth2studio>
for further anaylysis and visualizations.

## Logging

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


## References

- [Residual Diffusion Modeling for Km-scale Atmospheric Downscaling](https://arxiv.org/pdf/2309.15214.pdf)
- [Elucidating the design space of diffusion-based generative models](https://openreview.net/pdf?id=k7FuTOWMOc7)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/pdf/2011.13456.pdf)
