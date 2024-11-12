#!/bin/bash
config="baseline_boundary_32_v2"
yaml=./config/hrrr_swin.yaml
run_num="0"
scratch="/pscratch/sd/j/jpathak/hrrr_experiments"
weights="${scratch}/${config}/${run_num}/training_checkpoints/best_ckpt.tar"

image=docker:registry.nersc.gov/m4331/earth-pytorch:23.08

export CUDA_VISIBLE_DEVICES=0
override_dir="${scratch}/${config}/inference/"
shifter --image=${image} bash -c "python inference/inference.py --yaml_config=${yaml} --config=${config} --weights=${weights} --override_dir=${override_dir}"

