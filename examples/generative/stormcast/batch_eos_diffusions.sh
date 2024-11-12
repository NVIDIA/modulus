#!/bin/bash
#SBATCH -A coreai_climate_earth2
#SBATCH -J coreai_climate_earth2-research.a2a_v3_1_full
#SBATCH -t 03:58:00
#SBATCH -p batch
#SBATCH -N 8
#SBATCH -o test-%j.out
#SBATCH --dependency singleton

# be sure we define everything
set -euxo pipefail

readonly _cont_image=./hrrr.sqsh
readonly _cont_name="hrrr"
readonly _cont_mounts="/lustre:/lustre"
readonly _out_dir="/lustre/fsw/coreai_climate_earth2/jpathak/hrrr-experiments/diffusion/"

# task count and other parameters
readonly _config_file=./config/hrrr_swin.yaml
readonly _config_name=diffusion_v3_1_invariants_tendency_grid_full_field
export DGXNGPU=8
gpus_per_node=8
export TOTALGPU=$(( ${gpus_per_node} * ${SLURM_NNODES} ))
export WANDB_MODE=online
export WANDB_API_KEY=$WANDB_API_KEY
export RUN_NUM=0

RUN_CMD="python train_diffusions.py --outdir ${_out_dir} --tick 10 --config_file ${_config_file} --config_name ${_config_name} --log_to_wandb True --run_id ${RUN_NUM} --snap 1 --dump 1"


# pull image
srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${_cont_image}" --container-name="${_cont_name}" true

# run code
srun -u --wait=1800 --mpi=pmix --ntasks="${TOTALGPU}" --ntasks-per-node="${gpus_per_node}" \
     --container-name="${_cont_name}" --container-mounts="${_cont_mounts}" \
     --container-workdir /lustre/fsw/coreai_climate_earth2/jpathak/hrrr-dev-sandbox \
     bash -c "
     ldconfig
     set -x
     export WORLD_SIZE=${TOTALGPU}
     export WORLD_RANK=\${PMIX_RANK}
     export MASTER_ADDRESS=\${SLURM_LAUNCH_NODE_IPADDR}
     export MASTER_PORT=29450
     export HDF5_USE_FILE_LOCKING=FALSE
     export CUDNN_V8_API_ENABLED=1
     export OMP_NUM_THREADS=1
     unset TORCH_DISTRIBUTED_DEBUG
     ${RUN_CMD}"
