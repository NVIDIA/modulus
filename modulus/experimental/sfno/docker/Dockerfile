# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG DLFW_VERSION
FROM nvcr.io/nvidia/pytorch:${DLFW_VERSION}-py3

# update repo info
RUN apt update -y

# upgrade cmake
RUN apt remove cmake -y && \
    pip install cmake --upgrade

# install mpi4py
RUN pip install mpi4py

# hdf5
RUN cd /tmp && wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.14/hdf5-1.14.2/src/hdf5-1.14.2.tar.gz && \
    gzip -cd hdf5-1.14.2.tar.gz | tar xvf - && \
    mkdir hdf5-1.14.2/build && cd hdf5-1.14.2/build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/hdf5 \
    -DHDF5_ENABLE_DIRECT_VFD=1 \
    -DHDF5_ENABLE_PARALLEL=1 \
    -DHDF5_TEST_API=1 \
    -DHDF5_TEST_VFD=1 \
    .. && \
    make -j 8 && make install

# h5py
RUN CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/opt/hdf5 pip install --no-binary=h5py h5py

# install zarr
RUN pip install zarr netCDF4 xarray pandas

# moviepy imageio for wandb video logging
RUN pip install moviepy imageio

# other python stuff
RUN pip install --upgrade wandb ruamel.yaml tqdm jsbeautifier

# numba
RUN pip install numba
ENV NUMBA_DISABLE_CUDA=1

# scoring tools
RUN pip install xskillscore properscoring

# benchy
RUN pip install git+https://github.com/romerojosh/benchy.git

# some useful scripts from mlperf
RUN pip install --ignore-installed "git+https://github.com/NVIDIA/mlperf-common.git"

# install python version checker
RUN mkdir /opt/utils && cd /opt/utils && \
    wget https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py

# install tensorly for compression experiments
RUN cd /opt && git clone https://github.com/tensorly/tensorly && \
    cd tensorly && \
    pip install -e .

RUN cd /opt && git clone https://github.com/tensorly/torch tensorly-torch && \
    cd tensorly-torch && \
    pip install -e .

# torch-harmonics
ENV HARMONICS_VERSION 0.60.4
RUN cd /opt && git clone https://github.com/NVIDIA/torch-harmonics.git && \
    cd torch-harmonics && \
    pip install -e .

# copy source code
RUN mkdir -p /opt/makani
COPY config /opt/makani/config
COPY docker /opt/makani/docker
COPY networks /opt/makani/networks
COPY utils /opt/makani/utils
COPY inference /opt/makani/inference
COPY data_process /opt/makani/data_process
COPY datasets /opt/makani/datasets
COPY mpu /opt/makani/mpu
COPY *.py /opt/makani/
COPY third_party /opt/makani/third_party
COPY perf_tests /opt/perf_tests
COPY tests /opt/makani/tests

# temporary WAR to enable complex optimization in ADAM
# copy custom optimizers to torch
COPY third_party/torch/optim/* /usr/local/lib/python3.10/dist-packages/torch/optim/
# hack distributed
COPY third_party/torch/distributed/* /usr/local/lib/python3.10/dist-packages/torch/distributed/

# create dummy git image
RUN cd /opt/makani && git init
