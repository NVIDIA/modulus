#!/bin/bash
#SBATCH --partition=batch_singlenode,polar
#SBATCH --time=01:00:00
#SBATCH --account coreai_climate_earth2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --signal=USR1@300

# Send SIGUSR1 5 minutes before the time limit

# Above SBATCH parameters are defaults which can be overriden
# by providing arguments to sbatch command.

: "${DOCKER_IMAGE:=gitlab-master.nvidia.com/modulus/modulus:tpunet}"
: "${NUM_GPUS:=1}"
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
python train.py ${ARGS};"

# The above command will run the training script inside the container
# The status file (SLURM_JOB_ID.txt) will be updated with the current status of the job. If it is STOPPED, resubmit the job to continue training.
# The training script will be run with the arguments provided to the sbatch command.

# Read the status
STATUS=$(cat ${SLURM_JOB_ID}.txt)
# If the status is STOPPED, resubmit the job
if [ "${STATUS}" == "STOPPED" ]; then
    echo "Job stopped. Resubmitting the job with script: ${SCRIPT_PATH} ${@}"
    scontrol requeue $SLURM_JOB_ID
else
    echo "Job finished with status: ${STATUS}"
fi
