import pygrib
import numpy as np
import xarray as xr
import datetime
import machine_info

#savepath_gfs = machine_info.savepath_gfs

def convert_grib_to_zarr(datestr, init_z, n_hours, savepath_gfs=None):

    gfs_vars = ['u10m', 
                'v10m', 
                't2m', 
                'tcwv', 
                'sp', 
                'msl', 
                'u1000', 
                'u850', 
                'u500', 
                'u250', 
                'v1000', 
                'v850', 
                'v500', 
                'v250', 
                'z1000', 
                'z850', 
                'z500', 
                'z250', 
                't1000', 
                't850', 
                't500', 
                't250', 
                'q1000', 
                'q850', 
                'q500', 
                'q250']

    mapping = {'10 metre U wind component': 'u10m',
               '10 metre V wind component': 'v10m',
               '2 metre temperature': 't2m',
               'Precipitable water': 'tcwv',
               'Surface pressure': 'sp',
               'Pressure reduced to MSL': 'msl',
               'U component of wind': 'u',
               'V component of wind': 'v',
               'Geopotential Height': 'z',
               'Temperature': 't',
               'Specific humidity': 'q'}

    
    fields = np.zeros((n_hours, len(gfs_vars), 721, 1440))
    valid_dates = []

    for forecast_hour in range(0, n_hours):

        grbs = pygrib.open(f'gribfiles/gfs_forecast_{forecast_hour:03d}.grb')

        if forecast_hour == 0:

            latitude, longitude = grbs[1].latlons()
            latitude = latitude[:, 0]
            longitude = longitude[0, :]
            print(latitude.shape, longitude.shape)
            print(latitude.min(), latitude.max())
            print(longitude.min(), longitude.max())


        for grb in grbs:

            #get the data for the first field
            data = grb.values
            level = grb.level
            name = grb.name

            #print time
            valid_date = grb.validDate

            if name in mapping:
                shortname = mapping[name]
                if level not in [0, 10, 2]:
                    shortname += str(level)
                
                if "z" in shortname:
                    data = data * 9.80665

                #get index of shortname in gfs_vars
                idx = gfs_vars.index(shortname)
                fields[forecast_hour, idx] = data

        valid_dates.append(valid_date)




    ds = xr.Dataset({'data': (['time', 'channel', 'latitude', 'longitude'], fields)},
                    coords={'time': valid_dates,
                            'channel': gfs_vars,
                            'latitude': latitude,
                            'longitude': longitude})

    #rearrange for consistency with era5 zarr
    ds = ds.reindex(latitude=ds['latitude'][::-1])
    ds['longitude'] = (ds['longitude'] + 180) % 360 - 180

    #ds.to_zarr(f"gfs_{datestr}_{init_z:02d}z.zarr", mode="w")
    ds.to_zarr(savepath_gfs + f"gfs_{datestr}_{init_z:02d}z.zarr", mode="w")