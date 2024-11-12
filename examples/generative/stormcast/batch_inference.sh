#!/bin/bash -l
#SBATCH --time=01:45:00
#SBATCH -C gpu
#SBATCH --account=m4331
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --image=registry.nersc.gov/m4331/earth-pytorch:23.08
#SBATCH --module=gpu,nccl-2.18
#SBATCH -o %x-%j.out

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)

export FI_MR_CACHE_MONITOR=userfaultfd

cmd="python analysis/run_analysis.py --analysis_config analysis/analysis_config.json"

set -x
srun -u --mpi=pmi2 shifter \
    bash -c $cmd 
    