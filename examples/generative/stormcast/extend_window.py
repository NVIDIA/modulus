import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pygrib


#grib file: /global/cfs/cdirs/m3522/cmip6/HRRR/2021/20210101/hrrr.t01z.wrfsfcf01.grib2

proj_params = {'a': 6371229, 'b': 6371229, 'proj': 'lcc', 'lon_0': 262.5, 'lat_0': 38.5, 'lat_1': 38.5, 'lat_2': 38.5}

grbs = pygrib.open('/global/cfs/cdirs/m3522/cmip6/HRRR/2021/20210101/hrrr.t01z.wrfsfcf01.grib2')

hrrr_sample = xr.open_zarr('/pscratch/sd/p/pharring/hrrr_example/hrrr/valid/2021.zarr', consolidated=True).sel(time='2021-01-01T01:00:00.000000000')

lats = hrrr_sample.latitude.values
lons = hrrr_sample.longitude.values

from pyproj import Proj
p = Proj(proj_params)
#print(grb.projparams)

x, y = p(lons, lats)

dx = x[0,1] - x[0,0]
dy = y[1,0] - y[0,0]

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

#define new_x_min, new_x_max, new_y_min, new_y_max to add 10 percent padding
factor = 0.05
new_x_min = x_min - factor*(x_max - x_min)
new_x_max = x_max + factor*(x_max - x_min)
new_y_min = y_min - factor*(y_max - y_min)
new_y_max = y_max + factor*(y_max - y_min)

new_x = np.arange(new_x_min, new_x_max, dx)
new_y = np.arange(new_y_min, new_y_max, dy)

new_x, new_y = np.meshgrid(new_x, new_y)

new_lons, new_lats = p(new_x, new_y, inverse=True)

plt.figure()
plt.scatter(new_lons[0,:], new_lats[0,:], s=0.01, label = 'extended grid', color='red')
plt.scatter(new_lons[:,0], new_lats[:,0], s=0.01, color='red')
plt.scatter(new_lons[-1,:], new_lats[-1,:], s=0.01, color='red')
plt.scatter(new_lons[:,-1], new_lats[:,-1], s=0.01, color='red')
plt.scatter(lons[0,:], lats[0,:], s=0.01, label = 'original hrrr grid', color='blue')
plt.scatter(lons[:,0], lats[:,0], s=0.01, color='blue')
plt.scatter(lons[-1,:], lats[-1,:], s=0.01, color='blue')
plt.scatter(lons[:,-1], lats[:,-1], s=0.01, color='blue')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.savefig("grid_extension_{}.png".format(int(factor*100)))

hrrr_sample = xr.open_zarr('/pscratch/sd/p/pharring/hrrr_example/hrrr/valid/2021.zarr', consolidated=True).sel(time='2021-01-01T01:00:00.000000000')

era5_sample = xr.open_zarr('/pscratch/sd/p/pharring/hrrr_example/era5/valid/2021.zarr', consolidated=True).sel(time='2021-01-01T01:00:00.000000000')


hrrr_lats = hrrr_sample.latitude
hrrr_lons = hrrr_sample.longitude

#create new hrrr lats and lons using new_lats and new_lons
hrrr_lats_new = xr.DataArray(new_lats, dims=('y', 'x'))
hrrr_lons_new = xr.DataArray(new_lons, dims=('y', 'x'))
coords={'latitude': hrrr_lats_new, 'longitude': hrrr_lons_new}
#set coords of hrrr_lats_new and hrrr_lons_new to coords
hrrr_lats_new = hrrr_lats_new.assign_coords(coords)
hrrr_lons_new = hrrr_lons_new.assign_coords(coords)


era5_sample_interp = era5_sample.interp(latitude=hrrr_lats_new, longitude=hrrr_lons_new)

fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].imshow(hrrr_sample.HRRR.sel(channel="t2m").values[:,:], origin='lower', cmap='RdBu_r', vmin=260, vmax=310)
#title
ax[0].set_title('HRRR')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

ax[1].imshow(era5_sample_interp.data.sel(channel="t2m").values[:,:], origin='lower', cmap='RdBu_r', vmin=260, vmax=310)
ax[1].set_title('ERA5, boundary extended by {}'.format(int(factor*100)) + '%')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')

plt.savefig("interp_test_{}.png".format(int(factor*100)))
