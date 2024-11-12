#!/bin/bash

# Launch this from the top-level dir of repo as bash docker/build.sh

# We use the NERSC private container registry here, should be shared across the m4331 group
# Set base nvcr.io pytorch container version with NVC_TAG
# To access the registry, do: podman-hpc login registry.nersc.gov
# See https://docs.nersc.gov/development/shifter/how-to-use/#using-registrynerscgov

NVC_TAG=23.08
IMAGE=registry.nersc.gov/m4331/earth-pytorch:$NVC_TAG

# build
set -x
podman-hpc build --build-arg nvc_tag=$NVC_TAG-py3 -t $IMAGE -f docker/Dockerfile .
podman-hpc push $IMAGE

