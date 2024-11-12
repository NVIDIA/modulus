import pygrib
import xarray as xr
import dask
from dask import delayed
from dask.distributed import Client
from datetime import datetime, timedelta
import os
import numpy as np
from collections import OrderedDict
import time
from dask.diagnostics import ProgressBar
import calendar
import zarr
import sys

COARSE_FACTOR = 1

#extent
X_START= 579
X_END= 1219
Y_START= 273
Y_END= 785

def sequential_read_nat(grb_filename, nat_params, levels):
    """ this speeds up grib reading by like 20x compared to looping with grb.select() """

    with pygrib.open(grb_filename) as grbs:

        for grb in grbs:

            name = grb.name
            level = grb.level

            if name in nat_params.keys():

                if level in levels:

                    shortkey = nat_params[name] + str(level)

                    data, lats, lons = grb.data()

                    yield shortkey, data

def sequential_read_sfc(grb_filename, sfc_params):

    with pygrib.open(grb_filename) as grbs:

        for grb in grbs:

            name = grb.name
            level = grb.level

            if name in sfc_params.keys():

                if level == sfc_params[name]['level']:

                    shortkey = sfc_params[name]['shortkey']

                    data, lats, lons = grb.data()

                    yield shortkey, data
    
def create_keep_params():


    nat_params = OrderedDict({'U component of wind': 'u', 
                 'V component of wind': 'v', 
                 'Temperature': 't', 
                 'Specific humidity': 'q', 
                 'Geopotential Height': 'z', 
                 'Pressure': 'p',
                 'Vertical velocity': 'w'})
    
    sfc_params = OrderedDict({'10 metre U wind component': {'shortkey': 'u10m', 'level': 10},
                    '10 metre V wind component': {'shortkey': 'v10m', 'level': 10},
                    '2 metre temperature': {'shortkey': 't2m', 'level': 2},
                    'Precipitable water': {'shortkey': 'tcwv', 'level': 0},
                    'MSLP (MAPS System Reduction)': {'shortkey': 'msl', 'level': 0},
                    'Maximum/Composite radar reflectivity': {'shortkey': 'refc', 'level': 0},
                    'Vertically-integrated liquid': {'shortkey': 'vil', 'level': 0}})
    
    send_to_end = ['refc', 'vil']

    levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30, 35, 40 ]

    param_master_list = []

    for param, paramkey in sfc_params.items():

        shortkey = paramkey['shortkey']

        param_master_list.append(shortkey)

    for param, paramkey in nat_params.items():

        for level in levels:

            shortkey = paramkey + str(level)

            param_master_list.append(shortkey)
    
    for shortkey in send_to_end:

        param_master_list.remove(shortkey)
        param_master_list.append(shortkey)

    print(param_master_list) 


    return sfc_params, nat_params, levels, param_master_list

def regrid_by_factor(da, factor):
    """ Regrid the input xarray DataArray to a coarser resolution by a factor. """

    return da.coarsen(y=factor, x=factor, boundary='trim').mean()

def crop_to_domain(da, xmin = 0, xmax = 1799, ymin = 0, ymax = 1059):
    """ Crop the input xarray DataArray to the domain. """

    return da.sel(y=slice(ymin, ymax), x=slice(xmin, xmax))


def grib2_to_zarr(sfc_params, nat_params, levels, param_master_list, savepath, datestr, init_z, filename_sfc, filename_nat):
    """ Convert all parameters from a GRIB2 file to an xarray Dataset. Concatenate variables, add a time, channel dimension. """

    num_channels = len(param_master_list)


    data_array = np.zeros((num_channels, 1059, 1799))
    
    try:

        with pygrib.open(filename_sfc) as grbs:

            time_val = [grbs[1].validDate]
            lats, lons = grbs[1].latlons()

        for shortkey, data in sequential_read_sfc(filename_sfc, sfc_params):


            data_array[param_master_list.index(shortkey)] = data
    

        for shortkey, data in sequential_read_nat(filename_nat, nat_params, levels):


            data_array[param_master_list.index(shortkey)] = data
    

        channels = param_master_list
    
        data_stacked = xr.DataArray(np.array(data_array).astype(np.float32),
                                   dims=['channel', 'y', 'x'],
                                   coords={'channel': channels})


        ds = xr.Dataset({
           'HRRR': (['time', 'channel', 'y', 'x'], np.expand_dims(data_stacked, axis=0)),
           'time': time_val,
           'channel': channels,
           'latitude': (["y", "x"], lats.astype(np.float32)),
           'longitude': (["y", "x"], lons.astype(np.float32)),
            })

        #coarsen the data if requested
        if COARSE_FACTOR > 1:

            ds = regrid_by_factor(ds, COARSE_FACTOR)
            #ydim = ds['y'].shape[0]
            #xdim = ds['x'].shape[0]

        if X_START > 0 or X_END < 1799 or Y_START > 0 or Y_END < 1059:
            ds = crop_to_domain(ds, X_START, X_END, Y_START, Y_END)


        zarr_filename = savepath + f"/hrrr_{datestr}_{init_z:02d}z_f01.zarr"

        ds.to_zarr(zarr_filename, mode='a')


    except:

        print('invalid sample')


def main():


    sfc, nat, levels, param_master_list = create_keep_params()

    num_channels = len(param_master_list)

    np.save('param_master_list.npy', param_master_list)

    print('num channels: {}'.format(num_channels))

    grib2_to_zarr(sfc, nat, levels, param_master_list)


if __name__ == "__main__":

    wall_start = time.time()


    main()

    

    wall_end = time.time()
    print('wall time: {}'.format(wall_end - wall_start))
