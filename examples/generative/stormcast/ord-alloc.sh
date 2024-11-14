#!/bin/bash

readonly _lustre_user=/lustre/fsw/portfolios/nvr/users/$USER
readonly _lustre_p=/lustre/fsw/portfolios/nvr/projects/nvr_earth2_e2
readonly _cont_image=/lustre/fsw/portfolios/nvr/projects/nvr_earth2_e2/us_mesoscale/hrrr.sqsh
readonly _cont_mounts="/lustre/fsw/portfolios/nvr/projects/nvr_earth2_e2/us_mesoscale:/data:r,${_lustre_p}:${_lustre_p}:rw,${_lustre_user}:${_lustre_user}:rw,/home/$USER/.swiftstack:/root/.swiftstack"

ngpu=1

srun -A nvr_earth2_e2 \
    --partition interactive \
	--gpus ${ngpu} \
	--ntasks-per-node ${ngpu} \
	--container-image=${_cont_image} \
	-t 04:00:00 \
	--container-mounts=${_cont_mounts} \
	--container-workdir="$PWD" \
	--pty bash
