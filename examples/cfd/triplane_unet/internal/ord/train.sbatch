#!/bin/bash
#SBATCH --partition=batch_singlenode,polar
#SBATCH --time=01:00:00
#SBATCH --account coreai_climate_earth2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Above SBATCH parameters are defaults which can be overriden
# by providing arguments to sbatch command.

: "${DOCKER_IMAGE:=nvcr.io#nv-maglev/dlav/modulus-tpunet-${USER}:latest}"
: "${NUM_GPUS:=1}"
: "${NUM_CPUS_PER_TASK:=16}"
: "${DATA_PATH:=/lustre/fsw/portfolios/coreai/projects/coreai_climate_earth2/datasets/DrivAer/drivaer_webdataset}"
: "${OUTPUT_PATH:="/lustre/fsw/portfolios/coreai/users/${USER}/outputs/tpunet/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"}"

srun                    \
--nodes=1               \
--ntasks=${NUM_GPUS}    \
--gpus=${NUM_GPUS}      \
--gpus-per-node=${NUM_GPUS}             \
--cpus-per-task=${NUM_CPUS_PER_TASK}    \
--exclusive                             \
--container-image=${DOCKER_IMAGE}       \
--container-mounts=/lustre:/lustre      \
python train.py \
    +experiment=drivaer/triplane_unet_best  \
    data.data_path=${DATA_PATH}             \
    output=${OUTPUT_PATH}                   \
    "${@}"
