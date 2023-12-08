#
/bin/bash

# readonly DATA="/lustre/fsw/sw_climate_fno/34Vars/"
# readonly OUTPUT="/lustre/fsw/sw_climate_fno/34Vars/"
# readonly CODE="$HOME/era5_wind/"

# readonly _cont_image=gitlab-master.nvidia.com/tkurth/era5_wind:develop-22.07-26ch
# readonly _cont_name="era5_wind"
# readonly _data_root="${DATA}"
# readonly _output_root="${OUTPUT}"
# readonly _code_root="${CODE}"
# readonly _cont_mounts="${_data_root}/stats:/stats:ro,${_data_root}/train:/train:ro,${_data_root}/test:/test:ro,${_output_root}/runs:/runs:rw,${_code_root}/:/code:rw"

# srun -A devtech -p interactive -N1 --container-image="${_cont_image}" \
#   --container-name="${_cont_name}" --container-mounts="${_cont_mounts}"\
#   --no-container-entrypoint \
#   --job-name devtech-e2prep:FNO-Transformer-Large-Scale-CWO-Forecasting -t 00:30:00 --pty bash
# #  --container-workdir / \

# Datasets on selene
ERA5: /lustre/fsw/sw_climate_fno/cwb-align
CWB: /lustre/fsw/sw_climate_fno/cwb-rwrf-2211

ssh selene-login.nvidia.com

#adlr
#srun -A adlr -p interactive -N1   --container-image=gitlab-master.nvidia.com/tkurth/era5_wind:latest   --container-mounts=$SHARE_SOURCE,$SHARE_OUTPUT,$SHARE_DATA,/lustre/fsw/sw_climate_fno    --no-container-entrypoint    --job-name adlr-e2prep:interactive -t 00:60:00   --pty bash

#devtech-interactive
srun -A devtech -p interactive -N1   --container-image=gitlab-master.nvidia.com/tkurth/era5_wind:latest  --container-mounts=$SHARE_SOURCE,$SHARE_OUTPUT,$SHARE_DATA,/lustre/fsw    --no-container-entrypoint    --job-name devtech-e2prep:interactive -t 00:60:00   --pty bash    #sw_earth2_ml

srun -A sw_earth2_ml -p interactive -N1   --container-image=gitlab-master.nvidia.com/tkurth/era5_wind:latest  --container-mounts=$SHARE_SOURCE,$SHARE_OUTPUT,$SHARE_DATA,/lustre/fsw    --no-container-entrypoint    --job-name sw_earth2_ml-e2prep:interactive -t 00:60:00   --pty bash  

srun -A devtech -p interactive -N1   --container-image=gitlab-master.nvidia.com/mmardani/diffusions-weather-forecast:latest  --container-mounts=$SHARE_SOURCE,$SHARE_OUTPUT,$SHARE_DATA,/lustre/fsw    --no-container-entrypoint    --job-name devtech-e2prep:interactive -t 00:60:00   --pty bash


#devtech-luna
srun -A devtech -p luna -N1   --container-image=gitlab-master.nvidia.com/tkurth/era5_wind:latest  --container-mounts=$SHARE_SOURCE,$SHARE_OUTPUT,$SHARE_DATA,/lustre/fsw    --no-container-entrypoint    --job-name devtech-e2prep:interactive -t 04:00:00   --pty bash


cd $SHARE_SOURCE

rsync -a --exclude='.git/' /home/mmardani/research/elucidated-ddpm-weather-gitlab/.    mmardani@selene-login.nvidia.com:/lustre/fsw/nvresearch/mmardani/source/weather-forecast-v2

#rsync -a --exclude='.git/' /home/mmardani/research/elucidated-ddpm-weather-gitlab-v2/.  mmardani@selene-login.nvidia.com:/lustre/fsw/nvresearch/mmardani/source/weather-forecast-v2

cd $SHARE_SOURCE/weather-forecast; PYTHONPATH=$SHARE_SOURCE/weather-forecast; pip install -r requirements.txt; exec python train.py

#eval - cwb
# cd /workspace/ws-interactive/source_code; PYTHONPATH=/workspace/ws-interactive/source_code; pip install -r requirements.txt; exec python eval.py
scp -r mmardani@selene-login.nvidia.com:/lustre/fsw/nvresearch/mmardani/source/weather-forecast-v2/samples_edm   ~/research/elucidated-ddpm-weather-gitlab

#eval - ea5-cwb
scp  mmardani@selene-login.nvidia.com:/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_20chans_4x_crop112_zarr/free-goose_2023.01.23_19.53/output/stats.jsonl  ~/research/elucidated-ddpm-weather-gitlab/samples


#cwb
# no grid
python3 generate.py  --seeds=0-63 --batch=1 --network=/lustre/fsw/nvresearch/mmardani/output/logs/edm_cwb/nickel-mule_2022.12.27_11.28/output/network-snapshot-087861.pkl  --outdir=samples  --data_config='full_field_val_crop64'  --data_type='cwb'   --task='sr'
# grid
python3 generate.py  --seeds=0-63 --batch=1 --network=/lustre/fsw/nvresearch/mmardani/output/logs/edm_cwb_grid/magnetic-lynx_2022.12.30_14.21/output/network-snapshot-198151.pkl  --outdir=samples  --data_config='full_field_train_last_chans_grid_crop64'  --data_type='cwb'   --task='sr'

#era5+cwb
#grid + 2018 data
python3 generate.py  --seeds=0-63 --batch=1 --network=/lustre/fsw/nvresearch/mmardani/output/logs/edm_cwb_grid/magnetic-lynx_2022.12.30_14.21/output/network-snapshot-198151.pkl  --outdir=samples  --data_config='full_field_val_crop64_grid'  --data_type='era5-cwb'   --task='sr'

#grid + all data + 20 era5 chans
python3 generate.py  --seeds=0-63 --batch=1 --network=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_20chans/provocative-flamingo_2023.01.05_11.47/output/network-snapshot-014150.pkl  --outdir=samples  --data_config='full_field_val_crop64_grid_20inchans'  --data_type='era5-cwb-v1'   --task='sr'

#grid + all data + 20 era5 chans + 4x cwb
python3 generate.py  --seeds=0-63 --batch=1 --network=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_20chans_4x/alluring-oarfish_2023.01.05_12.02/output/network-snapshot-019720.pkl  --outdir=samples  --data_config='full_field_val_crop64_grid_20inchans_4x'  --data_type='era5-cwb-v1'   --task='sr'

#grid + all data + 20 era5 chans + 4x cwb + crop112
python3 generate.py  --seeds=0-63 --batch=1 --network=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_20chans_4x_crop112/quick-seahorse_2023.01.05_15.31/output/network-snapshot-035432.pkl  --outdir=samples  --data_config='full_field_train_crop112_grid_20inchans_4x'  --data_type='era5-cwb-v1'   --task='sr'

#grid + all data + 20 era5 chans + 4x cwb + crop112 + zarr
python3 generate.py  --seeds=0-63 --batch=1 --network=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_20chans_4x_crop112_zarr/free-goose_2023.01.23_19.53/output/network-snapshot-064504.pkl --outdir=samples  --data_config='full_field_val_crop112_grid_20inchans_4x'  --data_type='era5-cwb-v2'   --task='sr'  --sample_res='full'

#grid + all data + 20 era5 chans + 4x cwb + crop112 + zarr + fulldata + 19 out channels
python3 generate.py  --seeds=0-63 --batch=1 --network=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_20chans_4x_crop112_zarr_fulldata/lemon-okapi_2023.01.29_22.42/output/network-snapshot-075515.pkl --outdir=samples  --data_config='full_field_val_crop112_grid_20inchans_19outchans_4x'  --data_type='era5-cwb-v3'   --task='sr'  --sample_res='full'

#grid + all data + 20 era5 chans + 4x cwb + crop112 + zarr + fulldata + 4 out channels
python3 generate.py  --seeds=0-63 --batch=10 --network=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_20chans_4x_crop112_zarr_fulldata/rebel-octopus_2023.01.31_20.46/output/network-snapshot-025038.pkl  --outdir=samples  --data_config='full_field_val_crop112_grid_20inchans_4outchans_4x'  --data_type='era5-cwb-v3'   --task='sr'  --sample_res='full'

#grid + all data + 12 era5/fcn chans + 4x cwb + crop448 + zarr + fulldata + 4 out channels
python3 generate.py  --seeds=0-63 --batch=1 --network=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata/cerulean-dolphin_2023.02.06_20.26/output/network-snapshot-024264.pkl --outdir=samples  --data_config='full_field_val_crop448_grid_12inchans_fcn_4outchans_4x'  --data_type='era5-cwb-v3'   --task='sr'  --sample_res='full'

#grid + all data + 12 era5/fcn chans + 4x cwb + crop448 + zarr + fulldata + 4 out channels + new run long
python3 generate.py  --seeds=0-63 --batch=10 --network=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata/satisfied-auk_2023.02.16_11.52/output/network-snapshot-099838.pkl  --outdir=samples  --data_config='full_field_val_crop448_grid_12inchans_fcn_4outchans_4x'  --data_type='era5-cwb-v3'   --task='sr'  --sample_res='full'

#grid + all data + 12 era5/fcn chans + 4x cwb + crop448 + zarr + fulldata + 4 out channels + 6 layers + dim=1024
python3 generate.py  --seeds=0-63 --batch=1 --network=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_6layer_dim1024/just-eagle_2023.02.09_10.48/output/network-snapshot-004703.pkl    --outdir=samples  --data_config='full_field_val_crop448_grid_12inchans_fcn_4outchans_4x'  --data_type='era5-cwb-v3'   --task='sr'  --sample_res='full'

#unet regressiojn: grid + all data + 12 era5/fcn chans + 4x cwb + crop448 + zarr + fulldata + 4 out channels
python3 generate.py  --seeds=0-63 --batch=10 --network=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression/mindful-parakeet_2023.04.03_22.54/output/network-snapshot-010186.pkl  --outdir=samples  --data_config='full_field_val_crop448_grid_12inchans_fcn_4outchans_4x'  --data_type='era5-cwb-v3'   --task='sr'  --sample_res='full'


#multi-process
torchrun --standalone --nproc_per_node=2 generate.py   --seeds=0-63 --batch=64 --network=/lustre/fsw/nvresearch/mmardani/output/logs/edm_cwb/sticky-yak_2022.12.26_10.38/output/network-snapshot-031913.pkl  --outdir=/lustre/fsw/nvresearch/mmardani/output/logs/edm_cwb/sticky-yak_2022.12.26_10.38/output/samples


#git push
git add -- . ':!samples'  ':!samples_image_dir'     # :!samples_edm   :!samples_regression  :!samples_mixture      #':!**/some_deep_nested_folder/*'
git commit -m "normalization v2"
git pull origin weather-era5-cwb-sr
git push origin weather-era5-cwb-sr

#To remove this directory from Git, but not delete it entirely from the filesystem (local):
git rm -r --cached samples


rsync -a   mmardani@selene-login.nvidia.com:/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression/mindful-parakeet_2023.04.03_22.54/code/.    /home/mmardani/research/elucidated-ddpm-weather-gitlab-v1    
