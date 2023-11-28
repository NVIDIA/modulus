#!/bin/bash

pip install runx

#cwb
python3 -m runx.runx edm_cwb.yml  -i
#cwb-grid
python3 -m runx.runx edm_cwb_grid.yml  -i
#era5-cwb-grid-3chans
python3 -m runx.runx edm_era5_cwb.yml  -i
#era5-cwb-grid-20chans
python3 -m runx.runx edm_era5_cwb_20chans.yml  -i
#era5-cwb-grid-20chans-4x
python3 -m runx.runx edm_era5_cwb_20chans_4x.yml  -i
#era5-cwb-grid-20chans-4x-crop112
python3 -m runx.runx edm_era5_cwb_20chans_4x_crop112.yml  -i
#era5-cwb-grid-20chans-4x-crop112-zarr
python3 -m runx.runx edm_era5_cwb_20chans_4x_crop112_zarr.yml  -i
#era5-cwb-grid-20chans-4x-crop112-zarr-fulldata
python3 -m runx.runx edm_era5_cwb_20chans_4x_crop112_zarr_fulldata.yml  -i
#era5-cwb-grid-20chans-4x-crop448-zarr-fulldata
python3 -m runx.runx edm_era5_cwb_20chans_4x_crop448_zarr_fulldata.yml  -i
#era5-cwb-grid-12chans-fcn-4x-crop448-zarr-fulldata-noaug (80M)
python3 -m runx.runx edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata.yml  -i   #5layer+dim 256  --3424793: torchrun
#era5-cwb-grid-12chans-fcn-4x-crop448-zarr-fulldata-noaug (150M)
python3 -m runx.runx edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_dim512.yml  -i    #5layer + dim 512
#era5-cwb-grid-12chans-fcn-4x-crop448-zarr-fulldata-noaug-6layer (450M)
python3 -m runx.runx edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_6layer_dim1024.yml  -i    #fails after 2hrs -- 
#era5-cwb-grid-12chans-fcn-4x-crop448-zarr-fulldata-noaug + determinisitc UNet (80M), lr=2e-4
python3 -m runx.runx edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression.yml  -i   #5layer+dim 256 + lr:2e-4 
#era5-cwb-grid-12chans-fcn-4x-crop448-zarr-fulldata-noaug + determinisitc UNet (80M), lr=2e-3
python3 -m runx.runx edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression_lr2e-3.yml  -i   #5layer+dim 256 + lr:2e-3   -- not working
#era5-cwb-grid-12chans-fcn-4x-crop448-zarr-fulldata-noaug + mixture loss: denoising + regression (80M), lr=2e-4
python3 -m runx.runx edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_mixture.yml  -i   #5layer+dim 256 + lr:2e-4
#era5-cwb-grid-12chans-fcn-4x-crop448-zarr-fulldata-noaug + mixture loss: denoising + regression (80M), lr=2e-4 +  normalization v2
python3 -m runx.runx edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_mixture_normalization.yml  -i   #5layer+dim 256 + lr:2e-4
#era5-cwb-grid-12chans-fcn-4x-crop448-zarr-fulldata-noaug + mixture loss: denoising + regression (80M), lr=2e-4 +  normalization v2 + mixture v2, pmean=0.0
python3 -m runx.runx edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_mixturev2_normalization.yml  -i   #5layer+dim 256 + lr:2e-4
#era5-cwb-grid-12chans-fcn-4x-crop448-zarr-fulldata-noaug + mixture loss: denoising + regression (80M), lr=2e-4 +  normalization v3 + mixture v3, pmean=0.0
python3 -m runx.runx edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_mixturev3_normalization.yml  -i   #5layer+dim 256 + lr:2e-4
#era5-cwb-grid-12chans-fcn-4x-crop448-zarr-fulldata-noaug + mixture loss: denoising + regression (80M), lr=2e-4 +  normalization v3 + mixture v4, pmean=0.0
python3 -m runx.runx edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_mixturev4_normalization.yml  -i   #5layer+dim 256 + lr:2e-4
#era5-cwb-grid-12chans-fcn-4x-crop448-zarr-fulldata-noaug + mixture loss: denoising + regression (80M), lr=2e-4 +  normalization v3 + mixture v5, pmean=0.0
python3 -m runx.runx edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_mixturev5_normalization.yml  -i   #5layer+dim 256 + lr:2e-4
#era5-cwb-grid-12chans-fcn-4x-crop448-zarr-fulldata-noaug + mixture loss: denoising + regression (80M), lr=2e-4 +  normalization v3 + mixture v1, pmean=2.0
python3 -m runx.runx edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_mixturev1_normalization.yml  -i   #5layer+dim 256 + lr:2e-4
#era5-cwb-grid-12chans-fcn-4x-crop448-zarr-fulldata-noaug + mixture loss: denoising + regression (80M), lr=2e-4 +  normalization v3 + reslossv1, pmean=2.0
python3 -m runx.runx edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_reslossv1_normalization.yml  -i   #5layer+dim 256 + lr:2e-4
