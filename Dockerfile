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

ARG PYT_VER=23.06
FROM nvcr.io/nvidia/pytorch:$PYT_VER-py3 as builder

# Update pip and setuptools
RUN pip install --upgrade pip setuptools  

# Setup git lfs, graphviz gl1(vtk dep)
RUN apt-get update && \
    apt-get install -y git-lfs graphviz libgl1 && \
    git lfs install

ENV _CUDA_COMPAT_TIMEOUT=90

# TODO remove benchy dependency
RUN pip install git+https://github.com/romerojosh/benchy.git
# TODO use torch-harmonics pip package after the upgrade
RUN pip install https://github.com/NVIDIA/torch-harmonics/archive/8826246cacf6c37b600cdd63fde210815ba238fd.tar.gz
RUN pip install "tensorly>=0.8.1" "vtk>=9.2.6" "pyvista>=0.40.1" https://github.com/tensorly/torch/archive/715a0daa7ae0cbdb443d06780a785ae223108903.tar.gz

# Install DGL from source
ARG DGL_BACKEND=pytorch
ENV DGL_BACKEND=$DGL_BACKEND
ENV DGLBACKEND=$DGL_BACKEND

RUN git clone https://github.com/dmlc/dgl.git && cd dgl/ && git checkout tags/1.1.1 && git submodule update --init --recursive && \
	DGL_HOME="/workspace/dgl" bash script/build_dgl.sh -g && \
	cd python && \
	python setup.py install && \
	python setup.py build_ext --inplace

# Install custom onnx
# TODO: Find a fix to eliminate the custom build
# Forcing numpy update to over ride numba 0.56.4 max numpy constraint
COPY . /modulus/ 
RUN if [ -e "/modulus/deps/onnxruntime_gpu-1.15.1-cp310-cp310-linux_x86_64.whl" ]; then \
	echo "Custom wheel exists, installing!" && \
	pip install --force-reinstall /modulus/deps/onnxruntime_gpu-1.15.1-cp310-cp310-linux_x86_64.whl; \
    else \
	echo "No custom wheel present, skipping" && \
	pip install "numpy==1.22.4"; \
    fi
# cleanup of stage
RUN rm -rf /modulus/ 

# CI image
FROM builder as ci
RUN pip install "tensorflow>=2.9.0" "warp-lang>=0.6.0" "black==22.10.0" "interrogate==1.5.0" "coverage==6.5.0" "protobuf==3.20.0" 
COPY . /modulus/
RUN cd /modulus/ && pip install -e . && rm -rf /modulus/

# Deployment image
FROM builder as deploy
RUN pip install "protobuf==3.20.0"
COPY . /modulus/
RUN cd /modulus/ && pip install .

# Clean up
RUN rm -rf /modulus/ 

# Docs image
FROM deploy as docs
# Install CI packages
RUN pip install "tensorflow>=2.9.0" "warp-lang>=0.6.0" "protobuf==3.20.0"
# Install packages for Sphinx build
RUN pip install "recommonmark==0.7.1" "sphinx==5.1.1" "sphinx-rtd-theme==1.0.0" "pydocstyle==6.1.1" "nbsphinx==0.8.9" "nbconvert==6.4.3" "jinja2==3.0.3"
