# Localized convection capable modeling


## Quickstart

We are using the shifter environment ```registry.nersc.gov/m4331/earth-pytorch:23.08```.

You can drop into a container by running:

``` shifter --image=docker:registry.nersc.gov/m4331/earth-pytorch:23.08 /bin/bash ```

on an interactive node

All paths in configs are interpreted relative to the $DATA_ROOT environment
variable (by default "/").

## Training

To train a model, modify one of the existing batch scripts:
``` submit_batch.sh ``` for example

use one of the existing configurations in
``` config/swin_hrrr.yaml ``` for example

## Diffusion model training

To train a diffusion model, modify one of the existing batch scripts:
``` submit_batch_diffusion.sh ``` for example

the diffusion model uses a config from config/hrrr_swin.yaml
currently, we are using the hrrr_3_1 dataset and the hrrr_3_1 with config:
``` diffusion_regression_a2s_v3_1_q: &diffusion_v3_1 ``` as the base config to inherit from.

in training a diffusion model, we specify a few command line options. Eventually we want to bring those into the config yaml file

```python train_diffusions.py --outdir rundir --tick 100 --config_file ./config/hrrr_swin.yaml --config_name diffusion_regression_a2s_v3_1_q_no_att --use_regression_net True --residual True --log_to_wandb True --run_id 0```

the run_id is used for checkpoint restart in conjunction with the rundir (which you can specify as /pscratch/sd/j/jpathak/hrrr_experiments/ for example). When you start a run, the code checks for a pre-existing run dir with matching run id and valid training checkpoint. If such a directory exists, it will load the checkpoint and resume training from there. If not, a new one will be created.

## Inference

Run the inference, making zarr files

    python3 run_inference.py --output-directory test_images/

Score it and generate movies (requires cartopy):

    python3 score_inference.py test_images/

checkpoints are located under
``` /pscratch/sd/j/jpathak/hrrr_experiments/ ```

diffusion checkpoints are currently under

```/pscratch/sd/j/jpathak/hrrr_experiments/diffusion_pickles ```

The directory structure of diffusion checkpoints might change under future release and this readme will be updated at that point

## Postprocessing
To get a vertical cross section animation run
```python vertical_section.py <path> ```
with <path> to a `data.zarr` produced by `run_inference.py`

## Dataset file structure

We have a few different datasets that we are using for this project. The file structure is as follows:

Training data directory:
``` /pscratch/sd/p/pharring/hrrr_example/ ```


## Dataset versions
We have a few versions of the HRRR dataset
```HRRR```
and the HRRR-2 dataset
```HRRR_v2```
and the HRRR-3 dataset
```HRRR_v3```

## Development

#### Container builds

To update the container env you'll need to re-build the image. On Perlmutter, first do:
```
podman-hpc login registry.nersc.gov
shifterimg login registry.nersc.gov
```
Use your NERSC username and password. You may need to file a ticket to request access the first time.

Then make your desired changes to the `Dockerfile` (e.g. new python package, increment nvcr version),
and from the top-level dir in the repo run
```
bash docker/build.sh
```
to update the image. This will build and push a new container version to the registry. Then finally run
```
shifterimg pull <image_name>
```
to make the image available to shifter and update the sbatch/launch scripts. Make sure to also freeze
the updated dependencies to `requirements.txt` by running ```make lock_deps``` from inside the container env.

#### Makefile

Generate requirements.txt:

    make lock_deps

Lint please:

    make lint

Download development data (you will be prompted for your password + OTP)

    make download_dev_data
