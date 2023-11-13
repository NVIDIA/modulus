matmul_parallel_size=1
h_parallel_size=1
w_parallel_size=1
spatial_parallel_size=$(( ${h_parallel_size} * ${w_parallel_size} ))
model_parallel_size=$(( ${matmul_parallel_size} * ${spatial_parallel_size} ))
multistep_count=1
jit_mode=none
graph_mode=none
amp_mode=bf16
checkpointing_level=0
DGXNGPU=8

# TODO CHANGE
BATCH_SIZE=8 # int or ""

# task count and other parameters
readonly _config_file=./config/sfnonet_devel.yaml
readonly _config_name=${CONFIG_NAME}

# TODO CHANGE
export gpus_per_node=${DGXNGPU}

# please change those!
ENABLE_PROFILING=0
N_VARIABLES=73
RUNS_NAME='LR_runs'

# image stuff
readonly _tag=stable-23.05
readonly _cont_image=gitlab-master.nvidia.com/tkurth/era5_wind:${_tag}
readonly _cont_name="era5_wind_09082023"
# readonly _cont_name="era5_wind_05052023"


if [ "${N_VARIABLES}" == "73" ]; then
    # 73 Variable dataset
    # readonly TRAINING_DATA="/lustre/fsw/sw_climate_fno/test_datasets/73var-6hourly"
    # readonly STATS="/lustre/fsw/sw_climate_fno/test_datasets/73var-6hourly/stats"
    # readonly INVARIANTS="/lustre/fsw/sw_climate_fno/test_datasets/48var-6hourly/invariants"
    readonly TRAINING_DATA="/lustre/fsw/sw_earth2_ml/test_datasets/73varQ"
    readonly STATS="/lustre/fsw/devtech/hpc-devtech/bbonev/73VarQ/stats"
    readonly INVARIANTS="/lustre/fsw/sw_earth2_ml/test_datasets/73varQ/invariants"
    readonly METADATA="/lustre/fsw/sw_earth2_ml/tkurth/73varQ"
elif [ "${N_VARIABLES}" == "232" ]; then
    readonly TRAINING_DATA="/lustre/fsw/sw_earth2_ml/ERA5_prod_large"
    readonly STATS="/lustre/fsw/sw_earth2_ml/ERA5_prod_large/stats"
    readonly INVARIANTS="/lustre/fsw/sw_earth2_ml/ERA5_prod_large/invariants"
    readonly METADATA=""
elif [ "${N_VARIABLES}" == "34" ]; then
    # 34 Variable dataset
    readonly TRAINING_DATA="/lustre/fsw/sw_climate_fno/34Vars"
    readonly STATS="/lustre/fsw/sw_climate_fno/34Vars_statsv2/stats"
    readonly INVARIANTS="/lustre/fsw/sw_climate_fno/test_datasets/48var-6hourly/invariants"
    readonly METADATA=""
else
    echo "Got N_VARIABLES=${N_VARIABLES}, expected one of {73,232,34}"
    exit
fi

readonly OUTPUT_FOLDER="${MY_OUTPUT_FOLDER}"
readonly CODE_FOLDER="${MY_CODE_FOLDER}"
readonly PLOTS_FOLDER="${MY_PLOT_FOLDER}"


