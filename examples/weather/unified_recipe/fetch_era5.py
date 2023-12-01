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

# This script fetches ERA5 data from the CDS API and stores it in a Zarr store
# This Zarr store can reside on a local file system or object storage (e.g. S3)

import os
import datetime
import hydra
from omegaconf import DictConfig
import logging
import fsspec
import boto3

from modulus.datapipes.climate.era5_mirror import ERA5Mirror

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Make the date range
    date_range = (
        datetime.date(cfg.dataset.start_year, 1, 1),
        datetime.date(cfg.dataset.end_year, 12, 31),
    )

    # Make fsspec file system
    if cfg.filesystem.type == "local":
        fs = None
    elif cfg.filesystem.type == "s3":
        # Initialize a fsspec S3 client
        fs = fsspec.filesystem(cfg.filesystem.type,
                               key=cfg.filesystem.key,
                               secret=os.environ["AWS_SECRET_ACCESS_KEY"], 
                               client_kwargs={'endpoint_url': cfg.filesystem.endpoint_url,
                                              'region_name': cfg.filesystem.region_name})

        # Initialize a boto3 S3 client to create the bucket
        ##########################################################
        # TODO: This might be better to remove and do manually
        boto3_client = boto3.client(
                cfg.filesystem.type,
                aws_access_key_id=cfg.filesystem.key,
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                endpoint_url=cfg.filesystem.endpoint_url,
                region_name=cfg.filesystem.region_name)
        # Check if bucket exists
        buckets = boto3_client.list_buckets()
        bucket_names = [bucket['Name'] for bucket in buckets['Buckets']]
        if cfg.dataset.base_path not in bucket_names:
            boto3_client.create_bucket(Bucket=cfg.dataset.base_path)
            logging.info("Created bucket: ", cfg.dataset.base_path)
        ##########################################################
    else:
        raise ValueError("Currently only local and s3 filesystems are supported")

    # make ERA5 mirror
    #logging.getLogger().setLevel(logging.ERROR) # silence cdsapi logging
    mirror = ERA5Mirror(
            base_path=cfg.dataset.base_path,
            fs=fs,
            variables=cfg.dataset.variables,
            num_workers=34,
            date_range=date_range,
            dt=cfg.dataset.dt,
            progress_plot=None,
            )

if __name__ == "__main__":
    main()
