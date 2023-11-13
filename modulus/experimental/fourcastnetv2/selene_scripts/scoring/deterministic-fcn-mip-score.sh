#!/bin/bash
#SBATCH -A nvr_earth2_e2
#SBATCH --job-name nvr_earth2_e2-sfno:score-fcn-mip
#SBATCH -t 01:00:00
#SBATCH -p luna
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=8


export model=prod_sfno_linear_73chq_sc2_layers8_edim960_wstgl2

readonly DATA="/lustre/fsw/"

readonly _cont_image=gitlab-master.nvidia.com/earth-2/fcn-mip:2023.8.15
readonly CODE="/home/$USER/"
readonly _cont_mounts="$DATA:$DATA:rw,$CODE:$CODE:rw"

export MODEL_REGISTRY=/lustre/fsw/sw_climate_fno/nbrenowitz/model_packages
export HDF=/lustre/fsw/sw_earth2_ml/test_datasets/73varQ
export TIME_MEAN_73=$HDF/stats/time_means.npy


# run time collection with all tasks per node
srun \
-u \
--mpi=pmix \
--container-image="$_cont_image" \
--container-name="fcn-mip" \
--container-mounts="$_cont_mounts"\
--no-container-entrypoint \
bash -c '
ldconfig
cd /root/era5_wind
python3 -m earth2mip.inference_medium_range -n 56 $model /root/$model.nc --data $HDF'
