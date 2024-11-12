#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH -C gpu
#SBATCH --account=m4331
#SBATCH -q regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --image=registry.nersc.gov/m4331/earth-pytorch:23.08
#SBATCH --module=gpu,nccl-2.18
#SBATCH -o %x-%j.out

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)

export FI_MR_CACHE_MONITOR=userfaultfd
export WANDB_MODE=online

set -x
srun -u --mpi=pmi2 shifter \
    bash -c "
    source export_DDP_vars.sh
    python train_diffusions.py --outdir rundir --tick 100 --config_file ./config/hrrr_swin.yaml --config_name  --log_to_wandb True --run_id 0
    "
