ARG nvc_tag
FROM nvcr.io/nvidia/pytorch:$nvc_tag

# install cartopy
RUN apt-get update && apt-get install -y \
     --no-install-recommends \
    libgeos-dev \
    cdo \
    nco \
    netcdf-bin \
    && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install cartopy

# install other python stuff necessary
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir apex cartopy cdsapi black dask h5py \
matplotlib mpi4py netCDF4 numpy nvidia_modulus Pillow properscoring \
pygrib pyproj ruamel.yaml ruff scipy scikit-image timm torch torchinfo \
torchvision typer wandb xarray zarr metpy dash coverage

