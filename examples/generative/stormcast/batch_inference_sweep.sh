#!/bin/bash -l
#SBATCH --time=01:45:00
#SBATCH -C gpu
#SBATCH --account=m4331
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --image=registry.nersc.gov/m4331/earth-pytorch:23.08
#SBATCH --module=gpu,nccl-2.18
#SBATCH -o %x-%j.out

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)

export FI_MR_CACHE_MONITOR=userfaultfd

cmd=$1 #"python analysis/run_analysis.py --analysis_config analysis/analysis_configs/analysis_diffusion_regression_a2a_v3_1_exclude_w_noema_sweep_regression_kimg_002046.json --case_studies_file analysis/case_studies.json --registry_file analysis/sweep_regression_kimg_registry.json"

set -x
srun -u --mpi=pmi2 shifter \
    bash -c "$cmd" 
    
#"python analysis/run_analysis.py --analysis_config analysis/analysis_configs/analysis_diffusion_regression_a2a_v3_1_exclude_w_noema_sweep_regression_kimg_001023.json --case_studies_file analysis/case_studies.json --registry_file analysis/sweep_regression_kimg_registry.json"
    