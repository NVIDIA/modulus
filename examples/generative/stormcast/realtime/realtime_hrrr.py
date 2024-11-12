import requests

#https://noaa-hrrr-bdp-pds.s3.amazonaws.com/index.html#hrrr.20240430/conus/
#hrrr.t00z.wrfnatf01.grib2
#hrrr.t00z.wrfsfcf01.grib2
def safe_request(url):

    try:
        return requests.get(url)
    except:
        return None

def get_hrrr(datestr, init_z, savepath):

    base_url = 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.'
    sfc_url = f'{base_url}{datestr}/conus/hrrr.t{init_z:02d}z.wrfsfcf01.grib2'
    nat_url = f'{base_url}{datestr}/conus/hrrr.t{init_z:02d}z.wrfnatf01.grib2'

    sfc = safe_request(sfc_url)
    nat = safe_request(nat_url)

    if sfc is None or nat is None:

        print(f"Could not download HRRR data for {datestr} \n")

        return False

    with open(savepath + f'hrrr_sfc_{datestr}_{init_z}z_f01.grib2', 'wb') as f:
        f.write(sfc.content)

    with open(savepath + f'hrrr_nat_{datestr}_{init_z}z_f01.grib2', 'wb') as f:
        f.write(nat.content)
    
    return True


if __name__ == '__main__':
    import datetime
    today = datetime.datetime.utcnow()
    day = today.day
    month = today.month
    year = today.year
    hour = today.hour
    datestr = f'{year}{month:02d}{day:02d}'
    init_z = int(hour / 6) * 6
    initialization_date = datetime.datetime(year, month, day, init_z)
    datestr = f'{year}{month:02d}{day:02d}'

    for attempts in range(5):

        retval = get_hrrr(datestr, init_z)

        if retval:
            print(f"successfully downloaded HRRR data for {datestr} {init_z:02d}z")
            break
        else:
            initialization_date -= datetime.timedelta(hours=6)
            print("trying previous initialization time, {}".format(initialization_date))
            datestr = initialization_date.strftime('%Y%m%d')
            init_z = initialization_date.hour
            fname_sfc = f'hrrr_sfc_{datestr}_{init_z}z.grib2'
            fname_nat = f'hrrr_nat_{datestr}_{init_z}z.grib2'
            if os.path.exists(fname_sfc) and os.path.exists(fname_nat):
                print(f"previous initialization time {datestr} {init_z:02d}z already downloaded, exiting...")
                break
