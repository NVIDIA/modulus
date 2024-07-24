#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

DEFAULT_IMAGE_NAME="nvcr.io/nv-maglev/dlav/modulus-tpunet-${USER}"
DEFAULT_IMAGE_TAG="latest"
DEFAULT_IMAGE_FULL_NAME="${DEFAULT_IMAGE_NAME}:${DEFAULT_IMAGE_TAG}"

# Use DEFAULT_IMAGE_FULL_NAME if no argument is given, otherwise use the argument
IMAGE_FULL_NAME="${1:-${DEFAULT_IMAGE_FULL_NAME}}"

TPUNET_DOCKER_DIR=$(dirname $(realpath -s $0))
MODULUS_DIR=$(realpath ${TPUNET_DOCKER_DIR}/../../../../..)

echo -e "\e[0;32m"
echo "Building image: ${IMAGE_FULL_NAME}"
echo -e "\e[0m"

docker build \
    -t ${IMAGE_FULL_NAME}   \
    --network=host          \
    -f ${TPUNET_DOCKER_DIR}/modulus_tpunet.Dockerfile \
    --target tpunet         \
    ${MODULUS_DIR}

docker push ${DEFAULT_IMAGE_FULL_NAME}
