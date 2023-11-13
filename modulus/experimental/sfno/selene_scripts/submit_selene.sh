#!/bin/bash
#SBATCH -A devtech
#SBATCH --job-name devtech-hpc:FNO-Transformer-Large-Scale-CWO-Forecasting
#SBATCH -t 04:00:00
#SBATCH -p luna
#SBATCH -N 16

#readonly KARTHIK_DATA="/lustre/fsw/devtech/hpc-devtech/kkashinath/ERA5/wind"
readonly KARTHIK_DATA="/lustre/fsw/devtech/hpc-devtech/kkashinath/ERA5/12Vars"
readonly THORSTEN_DATA="/lustre/fsw/devtech/hpc-devtech/tkurth/ERA5_wind/12Vars"

# image stuff
readonly _cont_image=gitlab-master.nvidia.com/tkurth/era5_wind:latest
readonly _cont_name="era5_wind"
readonly _data_root="${KARTHIK_DATA}"
readonly _output_root="${THORSTEN_DATA}"
readonly _cont_mounts="${_output_root}/stats:/stats:ro,${_data_root}/train:/train:ro,${_data_root}/test:/test:ro,${_output_root}/runs:/runs:rw"

# task count and other parameters
readonly _config_name=full_field_fno_12ch_w96
gpus_per_node=8
export DGXNGPU=8
export TOTALGPU=$(( ${gpus_per_node} * ${SLURM_NNODES} ))

# binding
IB_BIND=''
if [[ "${SLURM_JOB_NUM_NODES}" -gt 1 ]]; then
  IB_BIND='--ib=single'
fi
BIND_CMD="bindpcie --cpu=exclusive ${IB_BIND} --"

# run command
RUN_CMD="python -u train.py --enable_amp --enable_jit --run_num=ngpu${TOTALGPU} --config=${_config_name} --local_rank=\$(( \${WORLD_RANK} % ${DGXNGPU} ))"

# pull image
srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${_cont_image}" --container-name="${_cont_name}" true

# run code
srun -u --wait=300 --mpi=pmix --ntasks="${TOTALGPU}" --ntasks-per-node="${gpus_per_node}" \
     --container-name="${_cont_name}" --container-mounts="${_cont_mounts}" \
     --container-workdir /opt/makani \
     bash -c "
     set -x
     export WORLD_SIZE=${TOTALGPU}
     export WORLD_RANK=\${PMIX_RANK}
     export LOCAL_RANK=\$(( \${WORLD_RANK} % ${DGXNGPU} ))
     export MASTER_ADDRESS=\${SLURM_LAUNCH_NODE_IPADDR}
     export MASTER_PORT=29450
     ${BIND_CMD} ${RUN_CMD}"
