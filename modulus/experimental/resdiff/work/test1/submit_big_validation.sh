#!/bin/bash 
#SBATCH -A nvr_earth2_e2
#SBATCH --job-name nvr_earth2_e2-sfno:score-fcn-mip
#SBATCH -t 01:30:00
#SBATCH -p luna
#SBATCH --array 0-7
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=8


export dataconfig=validation_big
export datatype=era5-cwb-v3

readonly DATA="/lustre/fsw/"
readonly CODE="/home/$USER/"

readonly _cont_image=gitlab-master.nvidia.com/earth-2/fcn-mip:latest
readonly _cont_mounts="${DATA}:${DATA}:rw,${CODE}/:/code:rw,/lustre/fsw:/lustre/fsw:rw"

# run time collection with all tasks per node
srun \
-u \
--mpi=pmix \
--container-image="$_cont_image" \
--container-name="era5_wind" \
--container-mounts="$_cont_mounts"\
--no-container-entrypoint \
bash -c '
cd /root/diffusions-weather-forecast
source /opt/entrypoint.sh
source test.sh


outputdir=$PROJECT_ROOT/generations/$datatype/$dataconfig
echo $outputdir
mkdir -p $outputdir
ncfile=$outputdir/samples.nc

generateOut=$outputdir/ranks
mkdir -p $generateOut

export WORLD_SIZE=$SLURM_NTASKS
export RANK=${SLURM_PROCID}
export LOCAL_RANK=${RANK}
export LOCAL_RANK=$(( ${WORLD_RANK} % 8 ))
export MASTER_ADDRESS=${SLURM_LAUNCH_NODE_IPADDR}
export MASTER_PORT=29450

beginSeed=$(( $SLURM_ARRAY_TASK_ID * 32 ))
endSeed=$(( $beginSeed + 32 - 1 ))

rm -rf "$generateOut/$SLURM_ARRAY_TASK_ID.*.nc"

python generate.py --outdir "$generateOut/$SLURM_ARRAY_TASK_ID.{rank}.nc" \
--seeds=${beginSeed}-${endSeed} \
--batch=10 \
--network=$url \
--data_config=$dataconfig  \
--data_type=$datatype   \
--task=sr  \
--pretext=reg \
--sample_res=full \
--res_edm \
--network_reg=$url_reg
'
