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
