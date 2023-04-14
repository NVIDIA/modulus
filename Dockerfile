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

# Install nightly build of dgl
RUN pip install --pre dgl -f https://data.dgl.ai/wheels/cu117/repo.html
RUN pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
ENV DGLBACKEND=pytorch

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
    wget https://anaconda.org/rapidsai-nightly/libcugraphops/23.04.00a/download/linux-64/libcugraphops-23.04.00a-cuda11_230331_g59523e85_63.tar.bz2 &&\
    wget https://anaconda.org/rapidsai-nightly/pylibcugraphops/23.04.00a/download/linux-64/pylibcugraphops-23.04.00a-cuda11_py38_230331_g59523e85_63.tar.bz2 &&\
    tar -xf libcugraphops-23.04.00a-cuda11_230331_g59523e85_63.tar.bz2 &&\
    tar -xf pylibcugraphops-23.04.00a-cuda11_py38_230331_g59523e85_63.tar.bz2 &&\
    rm libcugraphops-23.04.00a-cuda11_230331_g59523e85_63.tar.bz2 &&\
    rm pylibcugraphops-23.04.00a-cuda11_py38_230331_g59523e85_63.tar.bz2

ENV PYTHONPATH="${PYTHONPATH}:/opt/cugraphops/lib/python3.8/site-packages"

ENV _CUDA_COMPAT_TIMEOUT=90

# Install custom onnx
# TODO: Find a fix to eliminate the custom build
COPY ./deps/onnxruntime_gpu-1.14.0-cp38-cp38-linux_x86_64.whl onnxruntime_gpu-1.14.0-cp38-cp38-linux_x86_64.whl
RUN pip install --force-reinstall onnxruntime_gpu-1.14.0-cp38-cp38-linux_x86_64.whl

# CI image
FROM builder as ci
RUN pip install tensorflow>=2.11.0 warp-lang>=0.6.0 black==22.10.0 interrogate==1.5.0 coverage==6.5.0 protobuf==3.20.0 
COPY . /modulus/
RUN cd /modulus/ && pip install -e . && rm -rf /modulus/

# Deployment image
FROM builder as deploy
RUN pip install protobuf==3.20.0 
COPY . /modulus/
RUN cd /modulus/ && pip install .

# Clean up
RUN rm -rf /modulus/ \
    && rm -rf onnxruntime_gpu-1.14.0-cp38-cp38-linux_x86_64.whl
