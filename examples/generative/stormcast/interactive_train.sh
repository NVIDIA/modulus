#!/bin/bash
export MASTER_ADDR=$(hostname)
image=docker:registry.nersc.gov/m4331/earth-pytorch:23.08
ngpu=4
config_file=./config/hrrr_swin.yaml
config="baseline_boundary_32_v2_chmask5"
run_num="check"
cmd="python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num"
srun -u -n $ngpu --cpus-per-task=32 --gpus-per-node $ngpu shifter --image=${image} --module=gpu,nccl-2.18 bash -c "source export_DDP_vars.sh && $cmd"
