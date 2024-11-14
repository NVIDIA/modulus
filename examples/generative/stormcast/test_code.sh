#!/bin/bash

python train.py --outdir rundir --tick 1 --config_file ./config/config.yaml --config_name regression --log_to_wandb False --run_id 0
python train.py --outdir rundir --tick 100 --config_file ./config/config.yaml --config_name diffusion --log_to_wandb False --run_id 0
python inference.py