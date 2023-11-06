export PATH=/opt/conda/bin:$PATH

pip install einops
pip install xskillscore

ldconfig
#url=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_20chans_4x_crop112_zarr/free-goose_2023.01.23_19.53/output/network-snapshot-087376.pkl
#url=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression/mindful-parakeet_2023.04.03_22.54/output/network-snapshot-042650.pkl
#url=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression/complex-mayfly_2023.04.25_16.42/output/network-snapshot-041594.pkl
#url=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression_bs2/small-cormorant_2023.05.13_18.57/output/network-snapshot-035776.pkl
url=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_resloss/quirky-weasel_2023.05.15_12.40/output/network-snapshot-014150.pkl

#residual
url_reg=/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression_bs2/small-cormorant_2023.05.13_18.57/output/network-snapshot-035776.pkl

# on NGC
PROJECT_ROOT=/lustre/fsw/nvresearch/nbrenowitz/diffusions
network=v2


set -x -e
pytest training

# patch pytorch
# needed since the model checkpoints are just pickles
set +e
patch --forward --batch -u /usr/local/lib/python3.8/dist-packages/torch/_utils.py < _utils.patch
set -e

# if in fcn-mip image use this
#cp -f _utils.py /opt/conda/lib/python3.8/site-packages/torch/_utils.py
set -e

export WORLD_SIZE=1

run () {
    outputdir=$PROJECT_ROOT/generations/$datatype/$dataconfig
    echo $outputdir
    mkdir -p $outputdir
    ncfile=$outputdir/samples.nc

    root=$PROJECT_ROOT/checkpoints
    url=$root/$network/diffusion.pkl
    url_reg=$root/$network/regression.pkl

    if ! [[ -f $ncfile ]] 
    then

        generateOut=$outputdir/.tmp.generate
        rm -rf $generateOut
        mkdir -p $generateOut

        python3 generate.py  --outdir "$generateOut/{rank}.nc" \
        --seeds=0-9 \
        --batch=10 \
        --network=$url \
        --data_config=$dataconfig  \
        --data_type=$datatype   \
        --task='sr'  \
        --pretext='reg' \
        --sample_res='full' \
        --res_edm \
        --network_reg=$url_reg

        python3 concat.py $generateOut/*.nc $ncfile
        rm -r "$generateOut"
    fi



    directory=$outputdir/multiple
    mkdir -p $directory
    #python plot_multiple_samples.py $ncfile $directory

    directory=$outputdir/singlke
    mkdir -p $directory
    [[ -d $directory ]] || python plot_single_sample.py $ncfile $directory
    # need to install imagemagick 
    # apt-get install imagemagick
    # convert -delay 25 -loop 0 $directory/*.sample.png $directory/sample.gif

    python score_samples.py $ncfile > $outputdir/scores.txt
    python power_spectra.py $ncfile $outputdir

}
# datatype=netcdf dataconfig=gfs run

# dataconfig='validation_small' \
# datatype='era5-cwb-v3'   \
# run

# datatype=netcdf dataconfig=fcn run
# datatype=netcdf dataconfig=era5_2022 run

export PROJECT_ROOT="$(pwd)/data"
export DATA_ROOT="s3://sw_climate_fno/nbrenowitz"
dataconfig='validation_small' \
datatype='era5-cwb-v3'   \
run
