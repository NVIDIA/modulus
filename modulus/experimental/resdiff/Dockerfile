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

FROM nvcr.io/nvidia/pytorch:23.01-py3

RUN apt update -y

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0 zarr==2.13.3 cftime==1.6.2

# install mpi4py
RUN pip install mpi4py

# h5py and zarr
RUN pip install h5py netCDF4

# moviepy imageio for wandb video logging
RUN pip install moviepy imageio

# other python stuff
RUN pip install wandb && \
    pip install ruamel.yaml && \
    pip install --upgrade tqdm && \
    pip install einops && \
    pip install timm



WORKDIR /workspace

RUN (printf '#!/bin/bash\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
ENTRYPOINT ["/entry.sh"]
