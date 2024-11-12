#!/bin/bash

checkpoint_location_eos="/lustre/fsw/coreai_climate_earth2/jpathak/hrrr-experiments/diffusion/"
config="0-diffusion_regression_a2a_v3_1_exclude_w_pstep_pos_embed-hrrr-gpus64"
ckpt="network-snapshot-003378.pkl"

destination="/home/jpathak/Data/checkpoints"
#make config directory
mkdir -p $destination/$config
#copy checkpoint
echo "Copying checkpoint $checkpoint_location_eos$config/$ckpt to $destination/$config/$ckpt"

rsync -avP login-eos:$checkpoint_location_eos$config/$ckpt $destination/$config

