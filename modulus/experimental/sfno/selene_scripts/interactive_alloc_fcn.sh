#!/bin/bash

readonly DATA="/lustre/fsw/sw_climate_fno/test_datasets/73var-6hourly/"
readonly OUTPUT="/lustre/fsw/nvresearch/jpathak/"
readonly CODE="/home/jpathak/era5_wind_refactor/"

readonly _cont_image=gitlab-master.nvidia.com/tkurth/era5_wind:latest
readonly _cont_name="era5_wind"
readonly _data_root="${DATA}"
readonly _output_root="${OUTPUT}"
readonly _code_root="${CODE}"
readonly _cont_mounts="${_data_root}/stats:/stats:ro,${_data_root}/out_of_sample:/out_of_sample:ro,${_data_root}/test:/test:ro,${_data_root}/train:/train:ro,${_output_root}/runs:/runs:rw,${_code_root}/:/code:rw,${_data_root}/ifs:/ifs:ro"

srun -A devtech -p interactive -N1 --ntasks-per-node=32 --container-image="${_cont_image}" \
  --container-name="${_cont_name}" --container-mounts="${_cont_mounts}"\
  --no-container-entrypoint \
  --job-name devtech-e2prep:FNO-Transformer-Large-Scale-CWO-Forecasting_inference -t 02:00:00 --pty bash

