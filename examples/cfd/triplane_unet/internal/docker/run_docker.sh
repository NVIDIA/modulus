#!/bin/bash

DEFAULT_IMAGE_FULL_NAME="nvcr.io/nv-maglev/dlav/modulus-tpunet-${USER}:latest"
: "${IMAGE_FULL_NAME:=${DEFAULT_IMAGE_FULL_NAME}}"

: "${CONTAINER_USER:=du}"

: "${CONTAINER_MOUNTS:=/data/:/data}"

: "${CONTAINER_NAME:=tpunet-transient}"

echo -e "\e[0;32m"
echo "Image name (IMAGE_FULL_NAME)       : ${IMAGE_FULL_NAME}"
echo "Container user (CONTAINER_USER)    : ${CONTAINER_USER}"
echo "Container mounts (CONTAINER_MOUNTS): ${CONTAINER_MOUNTS}"
echo "Container name (CONTAINER_NAME)    : ${CONTAINER_NAME}"
echo -e "\e[0m"

# Parse mounts.
# Split into strings and prefix with -v option.
IFS=',' read -ra MOUNT_DIRS <<< "${CONTAINER_MOUNTS}"
for s in "${MOUNT_DIRS[@]}"; do
    MOUNT_DIRS_ALL+="-v ${s} "
done

docker run -it --rm --gpus all              \
    --network=host                          \
    --ipc=host                              \
    --cap-add=SYS_PTRACE                    \
    -v /dev/shm:/dev/shm                    \
    ${MOUNT_DIRS_ALL}                       \
    -v /etc/localtime:/etc/localtime:ro     \
    -u ${CONTAINER_USER}:${CONTAINER_USER}  \
    --ulimit memlock=-1                     \
    --ulimit stack=67108864                 \
    --name=${CONTAINER_NAME}                \
    --rm                                    \
    ${IMAGE_FULL_NAME}                      \
    "${@}"
