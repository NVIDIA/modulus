## to preallocate via slurm run
## salloc -p interactive -A sw_earth2_ml -J sw_earth2_ml-sfno:sfno_training -t 01:00:00 -N1 bash

#!/bin/bash
export ALLOCATED_JOBID=3632138

# be sure we define everything
set -euxo pipefail

batch_size=4
h_parallel_size=2
w_parallel_size=1
matmul_parallel_size=1
multistep_count=1
jit_mode=none
graph_mode=fwdbwd
amp_mode=none
checkpointing=0

# please change those!
ENABLE_PROFILING=0

# 73 Variable dataset
readonly TRAINING_DATA="/lustre/fsw/sw_climate_fno/test_datasets/73var-6hourly"
readonly OUTPUT="/lustre/fsw/nvresearch/jpathak/"
readonly STATS="/lustre/fsw/sw_climate_fno/test_datasets/73var-6hourly/stats"
readonly INVARIANTS="/lustre/fsw/sw_climate_fno/test_datasets/48var-6hourly/invariants"


# common folders for code and plots
readonly CODE="/home/jpathak/copies/dist/"
readonly PLOTS="/home/jpathak/copies/dist/"

# image stuff
readonly _tag=gdebug
readonly _cont_image=gitlab-master.nvidia.com/tkurth/era5_wind:${_tag}
# readonly _cont_image=/lustre/fsw/devtech/hpc-devtech/bbonev/images/tkurth+era5_wind+latest.sqsh
readonly _cont_name="era5_wind_28032023"
if [ "${ENABLE_PROFILING:-0}" == "1" ]; then
    export WANDB_MODE=offline
    suffix="_profile"
else
    export WANDB_MODE=offline
    suffix=""
fi

readonly _output_root="${OUTPUT}"
readonly _code_root="${CODE}"
readonly _data_root="${TRAINING_DATA}"
readonly _stats="${STATS}"
readonly _invariants="${INVARIANTS}"
# readonly _cont_mounts="${_output_root}/stats:/stats:rw,${_data_root}/train:/train:ro,${_data_root}/test:/test:ro,${_output_root}/runs:/runs:rw,${_code_root}/:/code:rw"
readonly _cont_mounts="${_stats}:/stats:ro,${_invariants}:/invariants:ro,${_data_root}/train:/train:ro,${_data_root}/test:/test:ro,${_data_root}/out_of_sample:/out_of_sample:ro,${_data_root}/ifs:/ifs:ro,${_output_root}/runs:/runs:rw,${_code_root}/:/code:rw"

# task count and other parameters
readonly _config_file=./config/sfnonet_distributed.yaml
readonly _config_name=distsfno_dhealy_73ch_sc3_infer
export DGXNGPU=8
gpus_per_node=${DGXNGPU}
export TOTALGPU=$(( ${gpus_per_node} * ${SLURM_NNODES} ))
# Boris API key
export WANDB_API_KEY=$WANDB_API_KEY
export RUN_NUM=04202023_01
# multistep args
multistep_args="--multistep_count=${multistep_count}"
# checkpointing args
if [ "${checkpointing:-0}" == "1" ]; then
    checkpointing_args="--enable_checkpointing"
else
    checkpointing_args=""
fi

# binding
IB_BIND=''
if [[ "${SLURM_JOB_NUM_NODES}" -gt 1 ]]; then
  IB_BIND='--ib=single'
fi
BIND_CMD="bindpcie --cpu=exclusive ${IB_BIND} --"

# run command
RUN_CMD="python -u train${suffix}.py --amp_mode=${amp_mode} --jit_mode=${jit_mode} ${checkpointing_args} --cuda_graph_mode=${graph_mode} ${multistep_args} --run_num=${RUN_NUM} --yaml_config=${_config_file} --config=${_config_name} --matmul_parallel_size=${matmul_parallel_size} --h_parallel_size=${h_parallel_size} --w_parallel_size=${w_parallel_size} --batch_size=${batch_size} --print_timings_frequency=10 --inference"
# RUN_CMD="python -u tests/test_model.py --enable_synthetic_data --amp_mode=${amp_mode} --jit_mode=${jit_mode} ${checkpointing_args} --cuda_graph_mode=${graph_mode} ${multistep_args} --run_num=${RUN_NUM} --yaml_config=${_config_file} --config=${_config_name} --matmul_parallel_size=${matmul_parallel_size} --h_parallel_size=${h_parallel_size} --w_parallel_size=${w_parallel_size} --batch_size=${batch_size} --print_timings_frequency=10"
# RUN_CMD="python -u tests/test_distributed_layer_norm.py --enable_synthetic_data --amp_mode=${amp_mode} --jit_mode=${jit_mode} ${checkpointing_args} --cuda_graph_mode=${graph_mode} ${multistep_args} --run_num=${RUN_NUM} --yaml_config=${_config_file} --config=${_config_name} --matmul_parallel_size=${matmul_parallel_size} --h_parallel_size=${h_parallel_size} --w_parallel_size=${w_parallel_size} --batch_size=${batch_size} --print_timings_frequency=10"

# pull image
srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${_cont_image}" --container-name="${_cont_name}" true

# run code
srun -u --wait=1800 --jobid=${ALLOCATED_JOBID} --mpi=pmix --ntasks="${TOTALGPU}" --ntasks-per-node="${gpus_per_node}" \
     --no-container-mount-home \
     --container-name="${_cont_name}" --container-mounts="${_cont_mounts}" \
     --container-workdir /code \
     bash -c "
     set -x
     export WORLD_SIZE=${TOTALGPU}
     export WORLD_RANK=\${SLURM_PROCID}
     export LOCAL_RANK=\$(( \${WORLD_RANK} % ${gpus_per_node} ))
     export MASTER_ADDRESS=\${SLURM_LAUNCH_NODE_IPADDR}
     export MASTER_PORT=29450
     export HDF5_USE_FILE_LOCKING=FALSE
     export CUDNN_V8_API_ENABLED=1
     export OMP_NUM_THREADS=1
     unset TORCH_DISTRIBUTED_DEBUG
     ${BIND_CMD} ${RUN_CMD}"
# ${BIND_CMD} nsys profile --output distsfno_73ch_h2w1\${SLURM_PROCID}.nsys-rep --trace cuda,mpi,nvtx --gpu-metrics-device 0 ${RUN_CMD}"