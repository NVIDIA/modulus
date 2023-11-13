#!/bin/bash

readonly JAIDEEP_DATA="/lustre/fsw/devtech/CWO-data/17Vars"
readonly JAIDEEP_OUTPUT="/lustre/fsw/sw_climate_fno/17Vars"
readonly CODE="/lustre/fsw/sw_climate_fno/"
readonly _cont_image=gitlab-master.nvidia.com/tkurth/era5_wind:jaideep
readonly _cont_name="era5_wind"
readonly _data_root="${JAIDEEP_DATA}"
readonly _output_root="${JAIDEEP_OUTPUT}"
readonly _code_root="${CODE}"
readonly _cont_mounts="${_data_root}/stats:/stats:ro,${_data_root}/train:/train:ro,${_data_root}/test:/test:ro,${_output_root}/runs:/runs:rw,${_code_root}/era5_wind:/code:ro"

srun -A devtech -p interactive -N1 --container-image=gitlab-master.nvidia.com/tkurth/era5_wind:jaideep \
  --container-name="${_cont_name}" --container-mounts="${_cont_mounts}"\
  --container-workdir /opt/makani \
  --job-name devtech-e2prep:FNO-Transformer-Large-Scale-CWO-Forecasting_debug -t 00:15:00 --pty bash