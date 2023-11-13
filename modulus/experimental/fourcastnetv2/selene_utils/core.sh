#!/bin/bash
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -p luna
#SBATCH --dependency singleton

if [[ ${ALLOCATED_JOBID} == "" ]]; then
	echo "Adding job to queue for CONFIG_NAME: $CONFIG_NAME"
else
	echo "Running on pre-ALLOCATED_JOBID: $ALLOCATED_JOBID and CONFIG_NAME: $CONFIG_NAME, INTERACTIVE: $INTERACTIVE"
fi

source ./parameters.sh
#  OTHER_OPTION --dependency=afterany:3805029
#    --job-name sw_earth2_ml-sfno:sfno_training_cp005
# ALLOCATED_JOBID=""
# -p luna

# be sure we define everything
set -euxo pipefail

if [ "${ENABLE_PROFILING:-0}" == "1" ]; then
    export WANDB_MODE=online
    suffix="_profile"
else
    suffix=""
    export WANDB_MODE=online
fi

export FULL_OUTPUT_DIR="${OUTPUT_FOLDER}/${RUNS_NAME}"
if [ ! -d "/path/to/dir" ]; then
    mkdir -p ${FULL_OUTPUT_DIR}
fi

readonly _cont_mounts="${STATS}:/stats:ro,${INVARIANTS}:/invariants:ro,${METADATA}:/metadata:ro,${TRAINING_DATA}/train:/train:ro,${TRAINING_DATA}/test:/test:ro,${TRAINING_DATA}/out_of_sample:/out_of_sample:ro,${TRAINING_DATA}/ifs:/ifs:ro,${FULL_OUTPUT_DIR}:/runs:rw,${CODE_FOLDER}/:/code:rw,${PLOTS_FOLDER}/:/plots:rw"

# readonly _output_root="${FULL_OUTPUT_DIR}"
# readonly _code_root="${CODE_FOLDER}"
# readonly _data_root="${TRAINING_DATA}"
# readonly _stats="${STATS}"
# readonly _invariants="${INVARIANTS}"
#readonly _cont_mounts="${STATS}:/stats:ro,${_invariants}:/invariants:ro,${_data_root}/train:/train:ro,${_data_root}/test:/test:ro,${_data_root}/ifs:/ifs:ro,${_output_root}:/runs:rw,${_code_root}/:/code:rw"


export TOTALGPU=$(( ${gpus_per_node} * ${SLURM_NNODES} ))
export WANDB_API_KEY=${MY_WANDB_API_KEY}
export RUN_NUM=ngpu${TOTALGPU}_mp${matmul_parallel_size}_sp${spatial_parallel_size}
# export RUN_NUM=flexible_multistep${multistep_count}

# multistep args
multistep_args="--multistep_count=${multistep_count}"

# inference args
# inference_args="--inference"
inference_args=""


ADDITIONAL_CMD_PARAMS=()
if [[ -n "${BATCH_SIZE}" ]]; then
    ADDITIONAL_CMD_PARAMS+=(--batch_size=${BATCH_SIZE})
fi


# binding
IB_BIND=''
if [[ "${SLURM_JOB_NUM_NODES}" -gt 1 ]]; then
  IB_BIND='--ib=single'
fi
BIND_CMD="bindpcie --cpu=exclusive ${IB_BIND} --"



# pull image
srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${_cont_image}" --container-name="${_cont_name}" true

ADDITIONAL_PARAMS=()
if [[ ! ${ALLOCATED_JOBID} == "" ]]; then
   echo "Adding Job ID"
    #ADDITIONAL_PARAMS+=("--jobid=${ALLOCATED_JOBID} ")
else
   echo $ALLOCATED_JOBID
    ADDITIONAL_PARAMS+=("-p luna ")
fi

if [[ -n "${INTERACTIVE}" ]]; then
   echo "Running interactvie"
    ADDITIONAL_PARAMS+=("-p interactive ")
    ADDITIONAL_PARAMS+=("--pty /opt/entrypoint.sh bash ")
    RUN_CMD=""
else
   echo "Running in non-interactive"
   RUN_CMD="python -u train.py --save_checkpoint=flexible --amp_mode=${amp_mode} --jit_mode=${jit_mode} --checkpointing_level=${checkpointing_level} --cuda_graph_mode=${graph_mode} ${multistep_args} --run_num=${RUN_NUM} --yaml_config=${_config_file} --config=${_config_name} --h_parallel_size=${h_parallel_size} --w_parallel_size=${w_parallel_size} --print_timings_frequency=10 ${ADDITIONAL_CMD_PARAMS[@]}"
   # RUN_CMD="python -u train${suffix}.py --checkpoint_format=legacy --amp_mode=${amp_mode} --jit_mode=${jit_mode} --checkpointing_level=${checkpointing_level} --cuda_graph_mode=${graph_mode} ${multistep_args} --run_num=${RUN_NUM} --yaml_config=${_config_file} --config=${_config_name} --matmul_parallel_size=${matmul_parallel_size} --h_parallel_size=${h_parallel_size} ${ADDITIONAL_PARAMS[@]}" 
	echo $INTERACTIVE
fi

echo "ADDITIONAL_PARAMS=${ADDITIONAL_PARAMS[@]}"
# run code
srun -u --wait=1800 --mpi=pmix --ntasks="${TOTALGPU}" --ntasks-per-node="${gpus_per_node}" "${ADDITIONAL_PARAMS[@]}"\
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

