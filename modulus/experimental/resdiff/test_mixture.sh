pip install einops
pip install xskillscore

ldconfig

#generation
#residual
#url=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_reslossv1_normalization/rampant-tarantula_2023.05.02_15.03/output/network-snapshot-004516.pkl

#image
#url=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_mixturev1_normalization/happy-trout_2023.04.24_17.23/output/network-snapshot-024234.pkl
url=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_mixturev2_normalization/mustard-seagull_2023.04.18_16.51/output/network-snapshot-098345.pkl
#url=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_mixturev3_normalization/smoky-sturgeon_2023.04.18_16.52/output/network-snapshot-037482.pkl
#url=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_mixturev4_normalization/crystal-jerboa_2023.04.18_16.53/output/network-snapshot-037482.pkl
#url=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_mixturev5_normalization/auspicious-kittiwake_2023.04.22_22.13/output/network-snapshot-020872.pkl


#regression
#url_reg=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression/mindful-parakeet_2023.04.03_22.54/output/network-snapshot-042650.pkl
#url_reg=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression/complex-mayfly_2023.04.25_16.42/output/network-snapshot-041594.pkl

#test
#url_reg=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression_bs1/sage-beagle_2023.05.13_18.57/output/network-snapshot-008277.pkl
url_reg=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression_bs2/small-cormorant_2023.05.13_18.57/output/network-snapshot-034572.pkl


set -x -e

pytest training
# patch pytorch
# needed since the model checkpoints are just pickles
cp _utils.py /usr/local/lib/python3.8/dist-packages/torch/_utils.py

python3 generate.py  --outdir samples_mixture.nc \
--n-samples 2 \
--seeds=0-63 --batch=10 --network=$url_reg  --network_reg=$url_reg --data_config='full_field_val_crop448_grid_12inchans_fcn_4outchans_4x' \
--data_type='era5-cwb-v3'   --task='sr'  --sample_res='full'   --pretext 'reg'  #--res_edm

python3 plot_multiple_samples.py samples_mixture.nc samples_mixture

python3 score_samples.py samples_mixture.nc


