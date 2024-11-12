# machine specific paths
import os


# local machine
environment_name = "local"
basepath = "/data/realtime_hrrr/"
savepath_gfs = basepath + "zarrfiles/gfs/" 
savepath_hrrr_grib = basepath + "gribfiles/"
savepath_hrrr_zarr = basepath + "zarrfiles/hrrr/"
savepath_hrrr_forecast = basepath + "zarrfiles/hrrr_forecast/"
savepath_ml_forecast = basepath + "zarrfiles/ml_forecast/"
checkpoint_basepath = "/data/checkpoints/"

#
##perlmutter
#basepath = "/pscratch/sd/j/jpathak/realtime_hrrr/"
#savepath_gfs = basepath + "zarrfiles/gfs/"
#savepath_hrrr_grib = basepath + "gribfiles/"
#savepath_hrrr_zarr = basepath + "zarrfiles/hrrr/"
