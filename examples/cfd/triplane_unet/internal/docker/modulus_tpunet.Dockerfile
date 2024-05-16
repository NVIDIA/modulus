FROM nvcr.io/nvidia/modulus/modulus:24.04 as base

RUN apt-get update && apt-get install -y --no-install-recommends \
    htop    \
    mc      \
    tmux

RUN pip install --no-cache-dir --upgrade pip

# Add and install torch_scatter wheel, which was built using:
# TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" pip wheel --no-deps torch-scatter
ADD ./examples/cfd/triplane_unet/internal/docker/torch_scatter-2.1.2-cp310-cp310-linux_x86_64.whl /workspace
RUN pip install --no-deps /workspace/torch_scatter-2.1.2-cp310-cp310-linux_x86_64.whl

# Install TPUNet dependencies to speed up subsequent image building.
RUN pip install --no-cache-dir \
    "jaxtyping>=0.2"        \
    "torch_scatter>=2.1"    \
    "torchinfo>=1.8"        \
    "warp-lang>=1.0"        \
    "webdataset>=0.2"

# Install openpoint
RUN git clone --recurse-submodules https://github.com/guochengqian/openpoints.git; \
    cd openpoints/cpp/pointnet2_batch; \
    python setup.py install; \
    cd ../pointops/; \
    python setup.py install; \
    cd ../chamfer_dist; \
    python setup.py install

# Copy the openpoints_requirements.txt and install
COPY ./examples/cfd/triplane_unet/internal/docker/openpoints_requirements.txt /workspace/
RUN pip install --no-cache-dir -r /workspace/openpoints_requirements.txt

# Add a non-root user with a fixed UID and GID
ARG USERNAME=du
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN set -eux; \
    groupadd --gid $USER_GID $USERNAME; \
    useradd --uid $USER_UID --gid $USER_GID --no-log-init -m -G video $USERNAME

# Add sudo and allow the non-root user to execute commands as root
# without a password.
RUN set -ex; \
    apt-get install -y --no-install-recommends \
        sudo; \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME; \
    chmod 0440 /etc/sudoers.d/$USERNAME;

FROM base AS tpunet

# Add the repo and install modulus package.
ADD ./ /workspace/modulus/

WORKDIR /workspace/modulus/
RUN pip install .[tpunet]

WORKDIR /workspace/modulus/examples/cfd/triplane_unet/
