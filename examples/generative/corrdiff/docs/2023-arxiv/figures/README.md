# CorDiff Plots

The plots for this paper are orchestrated using a makefile.

Export the project root depending on the system you are on

    # selene
    export PROJECT_ROOT=/lustre/fsw/nvresearch/nbrenowitz/diffusions
    export DATA_ROOT=/lustre/fsw/sw_climate_fno/nbrenowitz

    # PBSS
    export PROJECT_ROOT=s3://cwb-diffusions
    export DATA_ROOT=s3://sw_climate_fno/nbrenowitz

Make all figures on selene interactive session

    # -j 8 means that up to 8 plots are run in parallel
    make -j 8

To recreate an given plot, delete it and then rerun `make`.
