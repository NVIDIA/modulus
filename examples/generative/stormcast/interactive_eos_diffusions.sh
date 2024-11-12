#!/bin/bash


torchrun --nnodes=1 --nproc-per-node=8 train_diffusions.py --outdir rundir --tick 10 --config_file ./config/hrrr_swin.yaml --config_name diffusion_v3_1_invariants_tendency_grid_full_field --log_to_wandb False --run_id 1


