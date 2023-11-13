#!/bin/bash
#SBATCH -A nvr_earth2_e2
#SBATCH --job-name nvr_earth2_e2-sfno:score-fcn-mip
#SBATCH -t 03:55:00
#SBATCH -p luna
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1


export model=prod_sfno_linear_73chq_sc2_layers8_edim960_wstgl2

readonly DATA="/lustre/fsw/"
readonly CODE=$(pwd)

readonly _cont_image=gitlab-master.nvidia.com/earth-2/fcn-mip:23.8.15
readonly _cont_mounts="$DATA:$DATA:rw,$HOME/era5_wind:/opt/makani:rw,$CODE:/code:ro"

export MODEL_REGISTRY=/lustre/fsw/sw_climate_fno/nbrenowitz/model_packages
export HDF=/lustre/fsw/sw_earth2_ml/test_datasets/73varQ


# run time collection with all tasks per node
export SLURM_NTASKS=1
srun \
-u \
--ntasks-per-node 1 \
--container-image="$_cont_image" \
--container-mounts="$_cont_mounts"\
--container-name="fcn-mip" \
--no-container-mount-home \
--no-container-entrypoint \
--pty \
bash -c '
ldconfig
cd /code
torchrun --nproc_per_node 8 -m earth2mip.lagged_ensembles --lags 4 --inits 664 --leads 23 --model $model --output /lustre/fsw/sw_climate_fno/nbrenowitz/scores/lagged_ensemble/$model --data=$HDF
'
