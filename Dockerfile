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

ARG PYT_VER=22.12
FROM nvcr.io/nvidia/pytorch:$PYT_VER-py3 as builder

# Update pip and setuptools
RUN pip install --upgrade pip setuptools  

# Setup git lfs
RUN apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install

# Install nightly build of dgl
RUN pip install --no-deps --pre dgl -f https://data.dgl.ai/wheels/cu117/repo.html
RUN pip install --no-deps --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
ENV DGLBACKEND=pytorch

ENV _CUDA_COMPAT_TIMEOUT=90

# Install custom onnx
# TODO: Find a fix to eliminate the custom build
# Forcing numpy update to over ride numba 0.56.4 max numpy constraint
COPY . /modulus/ 
RUN if [ -e "/modulus/deps/onnxruntime_gpu-1.14.0-cp38-cp38-linux_x86_64.whl" ]; then \
	echo "Custom wheel exists, installing!" && \
	pip install --force-reinstall /modulus/deps/onnxruntime_gpu-1.14.0-cp38-cp38-linux_x86_64.whl; \
    else \
	echo "No custom wheel present, skipping" && \
	pip install numpy==1.22.4; \
    fi
# cleanup of stage
RUN rm -rf /modulus/ 

# CI image
FROM builder as ci
RUN pip install tensorflow>=2.11.0 warp-lang>=0.6.0 black==22.10.0 interrogate==1.5.0 coverage==6.5.0 protobuf==3.20.0 
# TODO remove benchy dependency
RUN pip install git+https://github.com/romerojosh/benchy.git
# TODO use torch-harmonics pip package after the upgrade
RUN pip install https://github.com/NVIDIA/torch-harmonics/archive/8826246cacf6c37b600cdd63fde210815ba238fd.tar.gz

# install libcugraphops and pylibcugraphops
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update &&\
    apt-get install -y software-properties-common &&\
    add-apt-repository ppa:ubuntu-toolchain-r/test &&\
    apt-get install -y libstdc++6
RUN mkdir -p /opt/cugraphops &&\
    cd /opt/cugraphops &&\
    wget https://anaconda.org/nvidia/libcugraphops/23.04.00/download/linux-64/libcugraphops-23.04.00-cuda11_230412_ga76892e3_0.tar.bz2 &&\
    wget https://anaconda.org/nvidia/pylibcugraphops/23.04.00/download/linux-64/pylibcugraphops-23.04.00-cuda11_py38_230412_ga76892e3_0.tar.bz2 &&\
    tar -xf libcugraphops-23.04.00-cuda11_230412_ga76892e3_0.tar.bz2 &&\
    tar -xf pylibcugraphops-23.04.00-cuda11_py38_230412_ga76892e3_0.tar.bz2 &&\
    rm libcugraphops-23.04.00-cuda11_230412_ga76892e3_0.tar.bz2 &&\
    rm pylibcugraphops-23.04.00-cuda11_py38_230412_ga76892e3_0.tar.bz2

ENV PYTHONPATH="${PYTHONPATH}:/opt/cugraphops/lib/python3.8/site-packages"

COPY . /modulus/
RUN cd /modulus/ && pip install -e . && rm -rf /modulus/

# Deployment image
FROM builder as deploy
RUN pip install protobuf==3.20.0 
COPY . /modulus/
RUN cd /modulus/ && pip install .

# Clean up
RUN rm -rf /modulus/ 

# Docs image
FROM deploy as docs
# Install CI packages
RUN pip install tensorflow>=2.11.0 warp-lang>=0.6.0 protobuf==3.20.0
# Install packages for Sphinx build
RUN pip install recommonmark==0.7.1 sphinx==5.1.1 sphinx-rtd-theme==1.0.0 pydocstyle==6.1.1 nbsphinx==0.8.9 nbconvert==6.4.3 jinja2==3.0.3
