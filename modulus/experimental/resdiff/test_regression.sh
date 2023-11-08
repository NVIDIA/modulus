
pip install einops
pip install xskillscore

ldconfig
url=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression/mindful-parakeet_2023.04.03_22.54/output/network-snapshot-042650.pkl
#url=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression/complex-mayfly_2023.04.25_16.42/output/network-snapshot-012092.pkl


set -x -e

pytest training
# patch pytorch
# needed since the model checkpoints are just pickles
cp _utils.py /usr/local/lib/python3.8/dist-packages/torch/_utils.py

python3 generate.py  --outdir samples_regression.nc \
--n-samples 2 \
--seeds=0-63 --batch=10 --network=$url  --network_reg=$url   --data_config='full_field_train_crop448_grid_12inchans_fcn_4outchans_4x_normv2'  --data_type='era5-cwb-v3'   --task='sr'  --sample_res='full'  --pretext 'reg'

python3 plot_samples.py samples_regression.nc samples_regression

python3 score_samples.py samples_regression.nc