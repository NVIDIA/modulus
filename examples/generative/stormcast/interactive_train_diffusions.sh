#!/bin/bash
export MASTER_ADDR=$(hostname)
export WANDB_MODE=offline
image=docker:registry.nersc.gov/m4331/earth-pytorch:23.08
ngpu=1
config="diffusion_regression_a2a_v3_1_exclude_w"
cmd="python train_diffusions.py --outdir rundir --tick 1 --config_file ./config/hrrr_swin.yaml --config_name $config --log_to_wandb False --run_id 0"
srun -u -n $ngpu --cpus-per-task=32 --gpus-per-node $ngpu shifter --image=${image} --module=gpu,nccl-2.18 bash -c "source export_DDP_vars.sh && $cmd"
