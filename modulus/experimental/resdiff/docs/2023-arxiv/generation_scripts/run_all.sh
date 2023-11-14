#!/bin/bash


function runSamples {
(
    cd ../../
    python3 generate.py  \
    --outdir $2 \
    --n-samples 20 \
    --seeds=0-63 \
    --batch=10 \
    --network=$1 \
    --data_config='full_field_val_crop448_grid_12inchans_fcn_4outchans_4x'  \
    --data_type='era5-cwb-v3'   \
    --task='sr'  \
    --sample_res='full'
)
}

ROOT=/lustre/fsw/nvresearch/nbrenowitz/diffusions
mkdir -p $ROOT
mkdir -p $ROOT/samples


url=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata/cerulean-dolphin_2023.02.06_20.26/output/network-snapshot-014308.pkl
out=$ROOT/samples/14308.nc
[[ -f "$out" ]] || runSamples "$url" "$out"
python3 ../../score_samples.py "$out" > out

url=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata/satisfied-auk_2023.02.16_11.52/output/network-snapshot-087376.pkl
out=$ROOT/samples/87376.nc
[[ -f "$out" ]] || runSamples "$url" "$out"
python3 ../../score_samples.py "$out" >> out
