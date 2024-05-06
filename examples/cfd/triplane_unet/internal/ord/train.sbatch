#!/bin/bash
#SBATCH --partition=batch_singlenode,polar
#SBATCH --time=04:00:00
#SBATCH --account coreai_climate_earth2
#SBATCH --cpus-per-task=16
#SBATCH --signal=USR1@300

# Example usage:
# 
# You must specify the number of GPUs to use
#
# sbatch --gres=gpu:8 train.sbatch +experiment=drivaer/dgcnn train.num_batch=32

# Above SBATCH parameters are defaults which can be overriden
# by providing arguments to sbatch command.

: "${DOCKER_IMAGE:=gitlab-master.nvidia.com/modulus/modulus:tpunet}"
: "${NUM_CPUS_PER_TASK:=16}"

node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f)

# Print job info
echo -e "
Running a GPU job on
    Cluster: ${cluster}
    Node: ${node}
    JOB_ID ${SLURM_JOB_ID}
    Job Name: ${SLURM_JOB_NAME}
    Date: $(TZ=America/Los_Angeles date)
    User: ${user}
    Current Path: $(pwd)
    IMAGE: ${DOCKER_IMAGE}
"
 
echo ""
echo ""

# Current script
SCRIPT_PATH=$(pwd)/internal/ord/train.sbatch
ARGS="${@}"

# Print job info
echo "Current script: ${0}"
echo "Arguments: ${ARGS}"

# Print all command
set -x

srun \
    --container-image=${DOCKER_IMAGE} \
    --container-mounts=${HOME}:/root,/lustre:/lustre,$(pwd):/workspace \
    bash -c "
TZ=America/Los_Angeles date;
echo 'Job ID: ${SLURM_JOB_ID}';
cd /workspace;
torchrun --nproc-per-node=\$(nvidia-smi -L | wc -l) train.py ${ARGS};"

# The above command will run the training script inside the container
# The status file (SLURM_JOB_ID.txt) will be updated with the current status of the job. If it is STOPPED, resubmit the job to continue training.
# The training script will be run with the arguments provided to the sbatch command.

# Check if the status file exists
if [ ! -f ${SLURM_JOB_ID}.txt ]; then
    echo "Status file not found. Exiting..."
    exit 1
fi

# Read the status
STATUS=$(cat ${SLURM_JOB_ID}.txt)
# If the status is STOPPED or RUNNING, resubmit the job
# When the job killed abruptly, the status will be RUNNING
echo "Job finished with status: ${STATUS}"
if [ "${STATUS}" == "STOPPED" ] || [ "${STATUS}" == "RUNNING" ]; then
    echo "Resubmitting the job with script: ${SCRIPT_PATH} ${@}"
    scontrol requeue $SLURM_JOB_ID
fi
