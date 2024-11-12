#!/bin/bash

srun -A coreai_climate_earth2 \
	-J coreai_climate_earth2-research.test \
	-p batch \
	-N1 \
  --verbose \
	--container-image=$PWD/hrrr.sqsh \
	--container-mounts=/lustre:/lustre \
  --container-workdir=/lustre/fsw/coreai_climate_earth2/jpathak/hrrr-dev-sandbox/ \
	--pty bash
