#!/bin/bash
bucket=s3://cwb-diffusions

aws s3 sync /lustre/fsw/sw_climate_fno/nbrenowitz/2023-01-24-cwb-4years.zarr $bucket/data/2023-01-24-cwb-4years.zarr
