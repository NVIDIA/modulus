import torch
import zarr
import xarray as xr
import numpy as np
import os


class GFSDataSet():

    def __init__(self, location, conus_dataset_name, hrrr_stats, exclude_channels, path_gfs, path_hrrr):
        self.gfs = xr.open_zarr(path_gfs)
        self.hrrr = xr.open_zarr(path_hrrr)
        self.hrrr_lon = self.hrrr.longitude
        self.hrrr_lat = self.hrrr.latitude
        self.gfs_lats, self.gfs_lons = self.construct_window()
        self.hrrr_stats = hrrr_stats
        self.means_hrrr = np.load(os.path.join(location, conus_dataset_name, hrrr_stats, 'means.npy'))[:, None, None]
        self.stds_hrrr = np.load(os.path.join(location, conus_dataset_name, hrrr_stats, 'stds.npy'))[:, None, None]
        self.means_era5 = np.load(os.path.join(location, 'era5', 'stats', 'means.npy'))[:, None, None]
        self.stds_era5 = np.load(os.path.join(location, 'era5', 'stats', 'stds.npy'))[:, None, None]
        self.hrrr_channels = list(self.hrrr.channel.values)
        self.exclude_channels = exclude_channels
        self.hrrr_latitudes = self.hrrr.latitude.values
        self.hrrr_longitudes = self.hrrr.longitude.values

    def __getitem__(self, idx):

        gfs = self.gfs.isel(time=idx + 1) #to align with HRRR f01 which is the initial condition after data assimilation

        print("gfs time: ", self.gfs.time.values[idx + 1])

        gfs = gfs.interp(latitude=self.gfs_lats, longitude=self.gfs_lons)['data'].values

        if idx == 0:

            hrrr = self.hrrr.isel(time=0)['HRRR'].values

        else:

            hrrr = None
        
        gfs, hrrr = self.normalize(gfs, hrrr)

        return gfs, hrrr
    
    def get_hrrr_channel_names(self):
        
        base_hrrr_channels = self.hrrr_channels
        kept_hrrr_channels = base_hrrr_channels

        if len(self.exclude_channels) > 0:
            kept_hrrr_channels = [x for x in base_hrrr_channels if x not in self.exclude_channels]

        return base_hrrr_channels, kept_hrrr_channels
        
    def normalize(self, gfs, hrrr=None):

        gfs -= self.means_era5
        gfs /= self.stds_era5

        if hrrr is not None:

            hrrr -= self.means_hrrr
            hrrr /= self.stds_hrrr

            if len(self.exclude_channels) > 0:
                drop_channel_indices = [self.hrrr_channels.index(x) for x in self.exclude_channels]
                hrrr = np.delete(hrrr, drop_channel_indices, axis=0)
        
        return gfs, hrrr

    
    def construct_window(self):
        '''
        Build custom indexing window to subselect HRRR region from ERA5
        '''
        from pyproj import Proj
        proj_params = {'a': 6371229, 'b': 6371229, 'proj': 'lcc', 'lon_0': 262.5, 'lat_0': 38.5, 'lat_1': 38.5, 'lat_2': 38.5} #from HRRR source grib file
        proj = Proj(proj_params)
        x, y = proj(self.hrrr_lon.values, self.hrrr_lat.values)

        dx = round(x[0,1] - x[0,0]) #grid spacing (this is pretty darn close to constant)
        dy = round(y[1,0] - y[0,0])

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        boundary_padding_pixels = 0

        new_x_min = x_min - boundary_padding_pixels*dx
        new_x_max = x_max + boundary_padding_pixels*dx
        new_y_min = y_min - boundary_padding_pixels*dy
        new_y_max = y_max + boundary_padding_pixels*dy

        new_x = np.arange(new_x_min, new_x_max, dx)

        new_y = np.arange(new_y_min, new_y_max, dy)

        new_x, new_y = np.meshgrid(new_x, new_y)

        new_lons, new_lats = proj(new_x, new_y, inverse=True)

        added_pixels_x = (new_x.shape[1] - self.hrrr_lon.shape[1]) 
        added_pixels_y = (new_x.shape[0] - self.hrrr_lon.shape[0])


        assert added_pixels_x == 2*boundary_padding_pixels, "requested {} padding pixels but got {} in x due to interpolation round off errors".format(boundary_padding_pixels, added_pixels_x)
        assert added_pixels_y == 2*boundary_padding_pixels, "requested {} padding pixels but got {} in y due to interpolation round off errors".format(boundary_padding_pixels, added_pixels_y)

        gfs_lats = xr.DataArray(new_lats, dims=('y', 'x'))
        gfs_lons = xr.DataArray(new_lons, dims=('y', 'x'))
        coords={'latitude': gfs_lats, 'longitude': gfs_lons}
        gfs_lats = gfs_lats.assign_coords(coords)
        gfs_lons = gfs_lons.assign_coords(coords)
        return gfs_lats, gfs_lons


if __name__ == "__main__":

    path_gfs = "/pscratch/sd/j/jpathak/fcn-dev-hrrr/realtime/zarrfiles/gfs/gfs_20240503_18z.zarr"
    path_hrrr = "/pscratch/sd/j/jpathak/fcn-dev-hrrr/realtime/zarrfiles/hrrr/hrrr_20240503_18z_f01.zarr"
    location = "/pscratch/sd/p/pharring/hrrr_example"
    conus_dataset_name = 'hrrr_v3'
    hrrr_stats = 'stats_v3_2019_2021'
    exclude_channels = ['u35', 'u40', 'v35', 'v40', 't35', 't40', 'q35', 'q40', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'w11', 'w13', 'w15', 'w20', 'w25', 'w30', 'w35', 'w40', 'p25', 'p30', 'p35', 'p40', 'z35', 'z40', 'tcwv', 'vil']

    ds = GFSDataSet(location, conus_dataset_name, hrrr_stats, exclude_channels, path_gfs, path_hrrr)

    import matplotlib.pyplot as plt
    gfs, hrrr = ds[0]
    print(gfs.shape, hrrr.shape)
    fig, ax = plt.subplots(1,2)
    im = ax[0].imshow(gfs[2])
    fig.colorbar(im, ax=ax[0])
    im = ax[1].imshow(hrrr[2])
    fig.colorbar(im, ax=ax[1])
    plt.savefig("test.png")