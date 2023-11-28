<!-- markdownlint-disable -->
## Generative Residual Diffusion Models (ResDiff)

**Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling**<br>
<br>https://arxiv.org/abs/2309.15214<br>


## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1+ high-end NVIDIA GPU for sampling and 8+ GPUs for training. We have done all testing and development using V100 and A100 GPUs.
* 64-bit Python 3.8 and PyTorch 1.12.0 (or later). See https://pytorch.org for PyTorch install instructions.
* Python libraries: See [environment.yml](./environment.yml) for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `conda env create -f environment.yml -n edm`
  - `conda activate edm`
* Docker users:
  - Ensure you have correctly installed the [NVIDIA container runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu).
  - Use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies.

## Getting started

To reproduce the main results from our paper, follow these steps:


### Data

The datasets on selene cluster

```
ERA5: /lustre/fsw/sw_climate_fno/cwb-align
CWB: /lustre/fsw/sw_climate_fno/cwb-rwrf-2211
```

### Data pre-processing
https://gitlab-master.nvidia.com/nbrenowitz/parallel-data-preprocessing/

Test by running 

```
./test.sh
```

### Training

We use runx package (https://github.com/NVIDIA/runx) to manage distributed training

```
pip install runx
```

It simply takes config files for training recpie as "./configs/<train-config>.yml". The config for data is also "./configs/<data-config>.yaml".

For batch training run

```
python3 -m runx.runx <train-config>.yml
```

For interactive jobs for testing before main training run

```
python3 -m runx.runx <train-config>.yml  -i 
```


### Sampling and Model Evaluation

Model evaluation is split into two components. generate.py creates a netCDF file
that can be further analyzed and `plot_samples.py` makes plots from it.

Generate samples, saving output in an netCDF file:

    # saves to samples.nc
    $ python3 generate.py  \
    --outdir samples.nc
    --seeds=0-63 \
    --batch=10 \
    --network=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata/satisfied-auk_2023.02.16_11.52/output/network-snapshot-087376.pkl \
    --data_config='full_field_val_crop448_grid_12inchans_fcn_4outchans_4x' \
    --data_type='era5-cwb-v3'   --task='sr'  --sample_res='full' \

Plot one-sample for all the times (row=variable, col=sample, file=time):

    $ python3 plot_single_sample.py samples.nc samples_image_dir/

Plot multiple samples for all times (row=time, col=sample, file=variable)

    $ python3 plot_multiple_samples.py samples.nc samples_image_dir/

Compute scalar scores:

    $ python3 score_samples.py samples.nc  Checkpoint: /lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata/cerulean-dolphin_2023.02.06_20.26/output/network-snapshot-014308.pkl
            maximum_radar_reflectivity  temperature_2m  eastward_wind_10m  northward_wind_10m
    metric                                                                                   
    rmse                      1.552729        0.399536           0.905391            0.927208
    crps                      0.704658        0.255652           0.488514            0.553241
