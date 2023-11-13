#!/bin/bash

# source variables file
source ./my_variables.sh

vars=$(getopt -o i:c: --long jobid:,config_name:,--interactive -- "$@")
eval set -- "$vars"

ALLOCATED_JOBID=""
INTERACTIVE=""
for opt; do
    case "$opt" in
      -i|--jobid)
	ALLOCATED_JOBID=$2
        shift 2
        ;;
      -c|--config_name)
	CONFIG_NAME=$2
        shift 2
        ;;
      --interactive)
        INTERACTIVE=true
        shift 1
	;;
    esac
done

echo "ALLOCATED_JOBID: $ALLOCATED_JOBID"
echo "CONFIG_NAME: $CONFIG_NAME"

if [[ ${ALLOCATED_JOBID} == "" ]]; then
	echo "Adding job to queue for CONFIG_NAME: $CONFIG_NAME"
else
	echo "Running on pre-ALLOCATED_JOBID: $ALLOCATED_JOBID and CONFIG_NAME: $CONFIG_NAME"
fi

if [[ ${INTERACTIVE} ]]; then
	echo "Launching interactive job"
fi


SBATCH_JOB_NAME=nvr_earth2_e2-sfno:${CONFIG_NAME}
if [ ! -d "./logs" ]; then
    mkdir logs
fi
SBATCH_OUTPUT="./logs/${CONFIG_NAME}-%j.out"
if [[ ${ALLOCATED_JOBID} == "" ]]; then
    echo "ADDING JOB TO QUEUE"
    launch="sbatch -A ${MY_SLURM_ACCOUNT} -J ${MY_SLURM_ACCOUNT}-earth2-sfno:production"
    #(export SBATCH_OUTPUT=$SBATCH_OUTPUT; export SBATCH_JOB_NAME=$SBATCH_JOB_NAME; export ALLOCATED_JOBID=$ALLOCATED_JOBID; export INTERACTIVE=$INTERACTIVE; export CONFIG_NAME=$CONFIG_NAME; sbatch core.sh)
else
    echo "LAUNCHING PREALLOCATED JOB"
    launch="bash"
    # (export SBATCH_OUTPUT=$SBATCH_OUTPUT; export SBATCH_JOB_NAME=$SBATCH_JOB_NAME; export ALLOCATED_JOBID=$ALLOCATED_JOBID; export INTERACTIVE=$INTERACTIVE; export CONFIG_NAME=$CONFIG_NAME; bash core.sh)
fi
(export SBATCH_OUTPUT=$SBATCH_OUTPUT; export SBATCH_JOB_NAME=$SBATCH_JOB_NAME; export ALLOCATED_JOBID=$ALLOCATED_JOBID; export INTERACTIVE=$INTERACTIVE; export CONFIG_NAME=$CONFIG_NAME; $launch core.sh)
