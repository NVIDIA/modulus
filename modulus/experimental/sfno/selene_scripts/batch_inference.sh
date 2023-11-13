#!/bin/bash
#SBATCH -A sw_earth2_ml
#SBATCH --job-name sw_earth2_ml-sfno:FNO-Transformer-Large-Scale-CWO-Forecasting_sfno_73var_inference
#SBATCH -t 02:00:00
#SBATCH -p luna
#SBATCH -N 1
#SBATCH -o sfno_73var_infer_%j.out
##SBATCH --dependency singleton

# be sure we define everything
set -euxo pipefail

matmul_parallel_size=1
model_parallel_stride=1
spatial_parallel_size=1
model_parallel_size=$(( ${matmul_parallel_size} * ${spatial_parallel_size} ))
graph_mode="none"
batch_size=8
multistep_count=1


ENABLE_PROFILING=0
readonly DATA="/lustre/fsw/sw_climate_fno/test_datasets/73var-6hourly/"
readonly OUTPUT="/lustre/fsw/nvresearch/jpathak/"
readonly CODE="/home/jpathak/era5_wind/"

# image stuff
readonly _tag=latest
readonly _cont_image=gitlab-master.nvidia.com/tkurth/era5_wind:${_tag}
readonly _cont_name="era5_wind"
if [ "${ENABLE_PROFILING:-0}" == "1" ]; then
    export WANDB_MODE=online
    suffix="_profile"
else
    readonly _data_root="${DATA}"
    suffix=""
fi

readonly _output_root="${OUTPUT}"
readonly _code_root="${CODE}"
readonly _cont_mounts="${_data_root}/stats:/stats:rw,${_data_root}/train:/train:ro,${_data_root}/test:/test:ro,${_data_root}/out_of_sample:/out_of_sample:ro,${_output_root}/runs:/runs:rw,${_code_root}/:/code:ro,${_data_root}/ifs:/ifs:ro"

# task count and other parameters
readonly _config_file=./config/sfnonet.yaml
readonly _config_name=sfno_73ch_infer
export DGXNGPU=8
gpus_per_node=8 
export TOTALGPU=$(( ${gpus_per_node} * ${SLURM_NNODES} ))
export WANDB_MODE=online
export WANDB_API_KEY=$WANDB_API_KEY
export RUN_NUM="plot_041423"
# multistep args
multistep_args="--multistep_count=${multistep_count}"

# binding
IB_BIND=''
if [[ "${SLURM_JOB_NUM_NODES}" -gt 1 ]]; then
  IB_BIND='--ib=single'
fi
BIND_CMD="bindpcie --cpu=exclusive ${IB_BIND} --"

# run command
RUN_CMD="python -u train${suffix}.py --cuda_graph_mode=${graph_mode} ${multistep_args} --amp_mode=fp16 --run_num=${RUN_NUM} --yaml_config=${_config_file} --config=${_config_name} --batch_size=${batch_size} --inference"

# pull image
srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${_cont_image}" --container-name="${_cont_name}" true

# run code
srun -u --wait=1800 --mpi=pmix --ntasks="${TOTALGPU}" --ntasks-per-node="${gpus_per_node}" \
     --container-name="${_cont_name}" --container-mounts="${_cont_mounts}" \
     --container-workdir /code/ \
     bash -c "
     set -x
     export WORLD_SIZE=${TOTALGPU}
     export WORLD_RANK=\${PMIX_RANK}
     export MASTER_ADDRESS=\${SLURM_LAUNCH_NODE_IPADDR}
     export MASTER_PORT=29450
     export HDF5_USE_FILE_LOCKING=FALSE
     export CUDNN_V8_API_ENABLED=1
     export OMP_NUM_THREADS=1
     unset TORCH_DISTRIBUTED_DEBUG
     ${BIND_CMD} ${RUN_CMD}"
