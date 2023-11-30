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

ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:23.10-py3
FROM ${BASE_CONTAINER} as builder

ARG TARGETPLATFORM

# Update pip and setuptools
RUN pip install "pip==23.2.1" "setuptools==68.2.2"

# Setup git lfs, graphviz gl1(vtk dep)
RUN apt-get update && \
    apt-get install -y git-lfs graphviz libgl1 && \
    git lfs install

ENV _CUDA_COMPAT_TIMEOUT=90

# Install other dependencies
RUN wget https://github.com/mpi4py/mpi4py/releases/download/3.1.5/mpi4py-3.1.5.tar.gz && \
    tar -zxf mpi4py-3.1.5.tar.gz && cd mpi4py-3.1.5/ && \
    python3 setup.py build && python3 setup.py install && \
    cd ../ && rm -r mpi4py*
RUN pip install --no-cache-dir "h5py>=3.7.0" "netcdf4>=1.6.3" "ruamel.yaml>=0.17.22" "scikit-learn>=1.0.2" 
RUN pip install --no-cache-dir "hydra-core>=1.2.0" "termcolor>=2.1.1" "wandb>=0.13.7" "mlflow>=2.1.1" "pydantic>=1.10.2" "imageio>=2.28.1" "moviepy>=1.0.3" "tqdm>=4.60.0"

# TODO remove benchy dependency
RUN pip install --no-cache-dir git+https://github.com/romerojosh/benchy.git
# TODO use torch-harmonics pip package after the upgrade
RUN pip install --no-cache-dir https://github.com/NVIDIA/torch-harmonics/archive/8826246cacf6c37b600cdd63fde210815ba238fd.tar.gz
RUN pip install --no-cache-dir "tensorly>=0.8.1" https://github.com/tensorly/torch/archive/715a0daa7ae0cbdb443d06780a785ae223108903.tar.gz

# copy modulus source
COPY . /modulus/

# Install Numcodecs (This needs a separate install because Numcodecs ARM pip install has issues) 
# A fix is being added here: https://github.com/zarr-developers/numcodecs/pull/315 but the public release is not ready yet.
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        echo "Pip install for numcodecs for $TARGETPLATFORM exists, installing!" && \
        pip install --no-cache-dir numcodecs; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ] && [ -e "/modulus/deps/numcodecs-0.11.0-cp310-cp310-linux_aarch64.whl" ]; then \
        echo "Numcodecs wheel for $TARGETPLATFORM exists, installing!" && \
        pip install --force-reinstall --no-cache-dir /modulus/deps/numcodecs-0.11.0-cp310-cp310-linux_aarch64.whl; \
    else \
        echo "Numcodecs wheel for $TARGETPLATFORM is not present, attempting to build from pip, but might fail" && \
	pip install --no-cache-dir numcodecs; \
    fi

# install vtk and pyvista
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ] && [ -e "/modulus/deps/vtk-9.2.6.dev0-cp310-cp310-linux_aarch64.whl" ]; then \
	echo "VTK wheel for $TARGETPLATFORM exists, installing!" && \
	pip install --no-cache-dir /modulus/deps/vtk-9.2.6.dev0-cp310-cp310-linux_aarch64.whl; \
    elif [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
	echo "Installing vtk for: $TARGETPLATFORM" && \
	pip install --no-cache-dir "vtk>=9.2.6"; \ 
    else \
	echo "Installing vtk for: $TARGETPLATFORM from source" && \
	apt-get update && apt-get install -y libgl1-mesa-dev && \
	git clone https://gitlab.kitware.com/vtk/vtk.git && cd vtk && git checkout tags/v9.2.6 && git submodule update --init --recursive && \
	mkdir build && cd build && cmake -GNinja -DVTK_WHEEL_BUILD=ON -DVTK_WRAP_PYTHON=ON /workspace/vtk/ && ninja && \
	python setup.py bdist_wheel && \
	pip install --no-cache-dir dist/vtk-9.2.6.dev0-cp310-cp310-linux_aarch64.whl && \
	cd ../../ && rm -r vtk; \
    fi
RUN pip install --no-cache-dir "pyvista>=0.40.1"

# Install DGL from source
ARG DGL_BACKEND=pytorch
ENV DGL_BACKEND=$DGL_BACKEND
ENV DGLBACKEND=$DGL_BACKEND
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ] && [ -e "/modulus/deps/dgl-1.1.2-cp310-cp310-linux_x86_64.whl" ]; then \
        echo "DGL wheel for $TARGETPLATFORM exists, installing!" && \
        pip install --force-reinstall --no-cache-dir /modulus/deps/dgl-1.1.2-cp310-cp310-linux_x86_64.whl; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ] && [ -e "/modulus/deps/dgl-1.1.2-cp310-cp310-linux_aarch64.whl" ]; then \
        echo "DGL wheel for $TARGETPLATFORM exists, installing!" && \
        pip install --force-reinstall --no-cache-dir /modulus/deps/dgl-1.1.2-cp310-cp310-linux_aarch64.whl; \
    else \
        echo "No DGL wheel present, building from source" && \
	git clone https://github.com/dmlc/dgl.git && cd dgl/ && git checkout tags/1.1.2 && git submodule update --init --recursive && \
	DGL_HOME="/workspace/dgl" bash script/build_dgl.sh -g && \
	cd python && \
	python setup.py install && \
	python setup.py build_ext --inplace; \
    fi
RUN rm -rf /workspace/dgl

# Install custom onnx
# TODO: Find a fix to eliminate the custom build
# Forcing numpy update to over ride numba 0.56.4 max numpy constraint
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ] && [ -e "/modulus/deps/onnxruntime_gpu-1.15.1-cp310-cp310-linux_x86_64.whl" ]; then \
	echo "Custom onnx wheel for $TARGETPLATFORM exists, installing!" && \
	pip install --force-reinstall --no-cache-dir /modulus/deps/onnxruntime_gpu-1.15.1-cp310-cp310-linux_x86_64.whl; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ] && [ -e "/modulus/deps/onnxruntime_gpu-1.15.1-cp310-cp310-linux_aarch64.whl" ]; then \
	echo "Custom onnx wheel for $TARGETPLATFORM exists, installing!" && \
        pip install --force-reinstall --no-cache-dir /modulus/deps/onnxruntime_gpu-1.15.1-cp310-cp310-linux_aarch64.whl; \
    else \
	echo "No custom wheel present, skipping" && \
	pip install --no-cache-dir "numpy==1.22.4"; \
    fi

# cleanup of stage
RUN rm -rf /modulus/ 

# CI image
FROM builder as ci

ARG TARGETPLATFORM

COPY . /modulus/
RUN cd /modulus/ && pip install -e . && pip uninstall nvidia-modulus -y && rm -rf /modulus/
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
	echo "Installing tensorflow and warp-lang for: $TARGETPLATFORM" && \
	pip install --no-cache-dir "tensorflow==2.9.0" "warp-lang>=0.6.0"; \ 
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
	echo "Installing tensorflow and warp-lang for: $TARGETPLATFORM is not supported presently"; \
    fi
RUN pip install --no-cache-dir "black==22.10.0" "interrogate==1.5.0" "coverage==6.5.0" "protobuf==3.20.3" "mpi4py>=3.1.4"

# Deployment image
FROM builder as deploy
COPY . /modulus/
RUN cd /modulus/ && pip install .
RUN pip install --no-cache-dir "protobuf==3.20.3"

# Set Git Hash as a environment variable
ARG MODULUS_GIT_HASH
ENV MODULUS_GIT_HASH=${MODULUS_GIT_HASH:-unknown}

# Clean up
RUN rm -rf /modulus/ 

# Docs image
FROM deploy as docs

ARG TARGETPLATFORM

# Install CI packages
RUN pip install --no-cache-dir "protobuf==3.20.3"
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
	echo "Installing tensorflow and warp-lang for: $TARGETPLATFORM" && \
	pip install --no-cache-dir "tensorflow==2.9.0" "warp-lang>=0.6.0"; \ 
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
	echo "Installing tensorflow and warp-lang for: $TARGETPLATFORM is not supported presently"; \
    fi
# Install packages for Sphinx build
RUN pip install --no-cache-dir "recommonmark==0.7.1" "sphinx==5.1.1" "sphinx-rtd-theme==1.0.0" "pydocstyle==6.1.1" "nbsphinx==0.8.9" "nbconvert==6.4.3" "jinja2==3.0.3"
RUN wget https://github.com/jgm/pandoc/releases/download/3.1.6.2/pandoc-3.1.6.2-1-amd64.deb && dpkg -i pandoc-3.1.6.2-1-amd64.deb  
