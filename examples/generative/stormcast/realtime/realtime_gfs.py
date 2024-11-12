import xarray as xr
import numpy as np
import time

def safe_open_remote(url):

    try:
        return xr.open_dataset(url)
    except:
        return None


def get_gfs_vars(datestr, init_z, lead_time, savepath):

    gfs_vars = ['u10m', 'v10m', 't2m', 'tcwv', 'sp', 'msl', 'u1000', 'u850', 'u500', 'u250', 'v1000', 'v850', 'v500', 'v250', 'z1000', 'z850', 'z500', 'z250', 't1000', 't850', 't500', 't250', 'q1000', 'q850', 'q500', 'q250']

    mapping = {'u10m': 'ugrd10m',
                    'v10m': 'vgrd10m',
                    't2m': 'tmp2m',
                    'tcwv': 'pwatclm',
                    'sp': 'pressfc',
                    'msl': 'prmslmsl',
                    'u': 'ugrdprs',
                    'v': 'vgrdprs',
                    'z': 'hgtprs',
                    't': 'tmpprs',
                    'q': 'spfhprs'}

    remote_data = safe_open_remote(f'http://nomads.ncep.noaa.gov:80/dods/gfs_0p25_1hr/gfs{datestr}/gfs_0p25_1hr_{init_z:02d}z')
    print(remote_data)

    if remote_data is None:
        print(f"Could not open remote data for {datestr} {init_z:02d}z \n")
        return False

    combined_fcst = []

    for varname in gfs_vars:
        if varname in mapping:
            varstr = mapping[varname]
            varlvl = None
        else:
            #assert that the variable is in the format 'u1000', 'v1000', etc.
            assert varname[0] in ['u', 'v', 'z', 't', 'q'] 
            assert varname[1:] in ['1000', '850', '500', '250']
            varstr = mapping[varname[0]]
            varlvl = int(varname[1:])

        print(varstr, varlvl)

        if varlvl is None:
            data = remote_data.isel(time=slice(0, lead_time))[varstr].load()
            time.sleep(2)
            #check if data has all zeros, if so, re-try
            if data.isel(time=0).values.sum() == 0 or np.isnan(data.isel(time=0).values).any():
                print(f"Data for {varname} is all zeros, retrying")
                time.sleep(5)
                remote_data = safe_open_remote(f'http://nomads.ncep.noaa.gov:80/dods/gfs_0p25_1hr/gfs{datestr}/gfs_0p25_1hr_{init_z:02d}z')
                data = remote_data.isel(time=slice(0, lead_time))[varstr].load()
            #check again
            if data.isel(time=0).values.sum() == 0 or np.isnan(data.isel(time=0).values).any():
                print(f"Data for {varname} is all zeros, skipping")
                continue

        else:
            data = remote_data.isel(time=slice(0, lead_time)).sel(lev=varlvl)[varstr].load()
            time.sleep(2)
            if varstr == 'hgtprs':
                data = data * 9.8 #check this
            if data.isel(time=0).values.sum() == 0 or np.isnan(data.isel(time=0).values).all():
                print(f"Data for {varname} is all zeros, retrying")
                #wait 5 seconds and try again
                time.sleep(5)
                #delete remote_data and try again
                del remote_data
                remote_data = safe_open_remote(f'http://nomads.ncep.noaa.gov:80/dods/gfs_0p25_1hr/gfs{datestr}/gfs_0p25_1hr_{init_z:02d}z')
                data = remote_data.isel(time=slice(0, lead_time))[varstr].load()
            #check again
            if data.isel(time=0).values.sum() == 0 or np.isnan(data.isel(time=0).values).any():
                print(f"Data for {varname} is all zeros, skipping")
                continue

        combined_fcst.append(data)

    #concatenate along new dimension
    combined_fcst = np.stack(combined_fcst, axis=1)

    ds = xr.Dataset({'data': (['time', 'channel', 'latitude', 'longitude'], combined_fcst)},
                    coords={'time': remote_data.time.isel(time=slice(0, lead_time)),
                            'channel': gfs_vars,
                            'latitude': remote_data.lat.values,
                            'longitude': remote_data.lon.values})

    #rearrange for consistency with era5 zarr
    ds = ds.reindex(latitude=ds['latitude'][::-1])
    ds['longitude'] = (ds['longitude'] + 180) % 360 - 180

    #ds.to_zarr(f"gfs_{datestr}_{init_z:02d}z.zarr", mode="w")
    ds.to_zarr(savepath + f"gfs_{datestr}_{init_z:02d}z.zarr", mode="w")

    return True


if __name__ == '__main__':

    #todays date
    import datetime
    import os
    today = datetime.datetime.utcnow()
    day = today.day
    month = today.month
    year = today.year
    hour = today.hour
    init_z = int(hour / 6) * 6
    initialization_date = datetime.datetime(year, month, day, init_z)
    datestr = f'{year}{month:02d}{day:02d}'

    for attempt in range(2):
        
        print("attempt", attempt)

        retval = get_gfs_vars(datestr, init_z, 24, savepath="./")

        if retval:
            print(f"Successfully downloaded GFS data for {datestr} {init_z:02d}z")
            break
        else:
            initialization_date -= datetime.timedelta(hours=6)
            datestr = initialization_date.strftime('%Y%m%d')
            init_z = initialization_date.hour
            fname = f"gfs_{datestr}_{init_z:02d}z.zarr"
            print(f"Failed to download GFS data for {datestr} {init_z:02d}z, checking previous initialization time which is {datestr} {init_z:02d}z")
            if os.path.exists(fname):
                print(f"Found existing file {fname}, skipping and exiting")
                break
                    


