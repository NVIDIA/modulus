import os
import sys
import re
import datetime
import time
import requests
from gfs_grib_to_zarr import convert_grib_to_zarr

inv_path = 'inventory/inv'

patterns = ["(HGT|TMP|UGRD|VGRD|SPFH):(1000|850|500|250) mb",
"(UGRD|VGRD):(10) m above ground",
"(TMP):(2) m above ground",
"(PRES):(surface)",
"(PRMSL):",
"(PWAT):(entire atmosphere)"]

def get_byteranges(inv, patterns):

    byteranges = []

    for pattern in patterns:

        for line in inv:
            if re.search(pattern, line) is not None:
                range_pattern = r'range=(\d+-\d+)'
                match = re.search(range_pattern, line)
                if match:
                    byterange = match.group(1)
                    byte_start = int(byterange.split('-')[0])
                    byte_end = int(byterange.split('-')[1])
                    byteranges.append((byte_start, byte_end))
                else:
                    Exception('No match found')

    return byteranges


def combine_byteranges(byteranges):

    #sort by byte_start
    byteranges.sort(key=lambda x: x[0])

    combined_byteranges = []

    for i, br in enumerate(byteranges):
        if i == 0:
            start = br[0]
            end = br[1]
        else:
            if br[0] == end + 1:
                end = br[1]
            else:
                combined_byteranges.append((start, end))
                start = br[0]
                end = br[1]

    combined_byteranges.append((start, end))

    return combined_byteranges

def check_url_exists(url):

    try:
        response = requests.head(url)
        return response.status_code == 200
    except:
        return False



def get_inventory(current_datetime, forecast_hour=0, inv_path="/code/realtime/inventory/inv"):

    year, month, day, hour = current_datetime.year, current_datetime.month, current_datetime.day, current_datetime.hour
    fstr = f"{forecast_hour:03d}"
    #hour = hour - (hour % 6)
    datestring = '{:04d}{:02d}{:02d}'.format(year, month, day)
    hourstring = '{:02d}'.format(hour)

    URL = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.' + datestring + '/' + hourstring + '/atmos/gfs.t' + hourstring + 'z.pgrb2.0p25.f' + fstr  + '.idx'

    if not check_url_exists(URL):
        return False

    curl_cmd = 'curl -s ' + URL + ' > ' + inv_path

    os.system(curl_cmd)

    with open(inv_path, 'r') as f:
        inv = f.read()

    inv = inv.split('\n')  # split into lines
    #remove last '\n' line
    inv = inv[:-1]

    for i, line in enumerate(inv):
        if i < len(inv)-1:
            if re.search(r'range=', line) is None:
                bytepattern = r'(?<=:)\d+(?=:d=)'
                match = re.search(bytepattern, line)
                start = match.group()
                next_line = inv[i+1]
                match = re.search(bytepattern, next_line)
                end = match.group()
                inv[i] = line + ':range=' + start + '-' + end

    # write to file
    with open(inv_path, 'w') as f:
        for line in inv:
            f.write(line + '\n')
    
    return True
            

def get_gribfiles(current_datetime, patterns, inv_path, forecast_hour=0):

    with open(inv_path, 'r') as f:
        inv = f.read()

    inv = inv.split('\n')  # split into lines

    byteranges = get_byteranges(inv, patterns)

    combined_byteranges = combine_byteranges(byteranges)

    print("current_datetime", current_datetime)

    year, month, day, hour = current_datetime.year, current_datetime.month, current_datetime.day, current_datetime.hour

    print("year, month, day, hour", year, month, day, hour)

    hour = hour - (hour % 6)

    datestring = '{:04d}{:02d}{:02d}'.format(year, month, day)
    hourstring = '{:02d}'.format(hour)

    print(datestring, hourstring)

    forecast_hour_str = f"{forecast_hour:03d}"

    URL='https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.'+datestring+'/'+hourstring+'/atmos/gfs.t'+hourstring+'z.pgrb2.0p25.f'+forecast_hour_str

    print(URL)

    range_str = ''

    for i, br in enumerate(combined_byteranges):

        range_str += '{}-{}'.format(br[0], br[1])

        if i < len(combined_byteranges) - 1:
            range_str += ','


    curl_cmd = 'curl -v -r "{}" {} -o ./gribfiles/gfs_forecast_{}.grb'.format(range_str, URL, forecast_hour_str)

    os.system(curl_cmd)

def get_forecast(datestr, init_z, lead_time, savepath):

    #datestr = "20240519"
    #init_z = 0
    inv_path = '/code/realtime/inventory/inv'

    init = datestr + '-' + str(init_z).zfill(2)

    current_datetime = datetime.datetime.strptime(init, '%Y%m%d-%H')

    print("retrieve data for", current_datetime)

    #if .grb files already exist, remove them
    for forecast_hour in range(0, lead_time):

        retval = get_inventory(current_datetime, forecast_hour=forecast_hour)
        if not retval:
            return False
    
        get_gribfiles(current_datetime, patterns, inv_path, forecast_hour=forecast_hour)

    
    # combine grib files
    convert_grib_to_zarr(datestr, init_z, lead_time, savepath_gfs=savepath)

    return True






if __name__ == '__main__':

    datestr = '20240528'

    init_z = 0

    n_hours = 24

    from machine_info import savepath_gfs

    get_forecast(datestr, init_z, n_hours, savepath_gfs)


