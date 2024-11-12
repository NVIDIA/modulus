import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors, ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as ticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import Normalize
from metpy.plots import ctables

def multivariate_horizontal(
         frame,
         ds_target,
         ds_edm,
         ds_noedm,
         var1,
         var2,
         var3,
         out_path,crop=False,
         left_lon_idx=None,
         right_lon_idx=None,
         bottom_lat_idx=None,
         top_lat_idx=None,
        ):
    """
    Create a 3x3 subplot of contour plots for three variables.

    Args:
    - frame (int): Time frame index.
    - ds_target (xarray.Dataset): Target dataset.
    - ds_edm (xarray.Dataset): EDM dataset.
    - ds_noedm (xarray.Dataset): No EDM dataset.
    - var1, var2, var3 (str): Variable names for plotting.
    - out_path (str): Output Path for saving figures
    - crop (bool): If True, crop the plot based on provided lon/lat values.
    - left_lon, right_lon, bottom_lat, top_lat (float): Lon/lat values for cropping.

    Returns:
    - None
    Example Usage: multivariate_horizontal(10,ds_Target,ds_edm,ds_noedm,'u_comb','v_comb','refc','/global/homes/p/piyushg/dc_figs',crop=True,left_lon=-92,right_lon=-86,bottom_lat=30.,top_lat=42.)
    """

    # Assert that crop parameters are provided if crop is True
    if crop:
        assert left_lon_idx is not None and right_lon_idx is not None and bottom_lat_idx is not None and top_lat_idx is not None, "If crop is True, provide left_lon, right_lon, bottom_lat, and top_lat."

    # Create figure and axes
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    colorbars = np.array([None, None, None, None, None, None, None, None, None]).reshape(axs.shape)

    # Extract min and max values for normalization
    var1_min = np.nanmin(ds_target[var1].values)
    var2_min = np.nanmin(ds_target[var2].values)
    var3_min = np.nanmin(ds_target[var3].values)

    var1_max = np.nanmax(ds_target[var1].values)
    var2_max = np.nanmax(ds_target[var2].values)
    var3_max = np.nanmax(ds_target[var3].values)

    longitude = ds_target.longitude.values
    latitude = ds_target.latitude.values
    
    lons_1d = longitude[0,:]
    lats_1d = latitude[:,0]
    
    left_lon = lons_1d[left_lon_idx]
    right_lon = lons_1d[right_lon_idx]
    bottom_lat = lats_1d[bottom_lat_idx]
    top_lat = lats_1d[top_lat_idx]

    datasets = ['target','prediction', 'regression'] 
    # Loop through rows and columns
    for i in range(3):
        for j in range(3):
            # Choose the correct dataset for each row
            if i == 0:
                axs[i, j].xaxis.set_visible(False)
                var_values = ds_target
            elif i == 1:
                axs[i, j].xaxis.set_visible(False)
                var_values = ds_edm
            else:
                var_values = ds_noedm

            # Extract variable values
            var1_values = var_values[var1].isel(time=frame)
            var2_values = var_values[var2].isel(time=frame)
            var3_values = var_values[var3].isel(time=frame)

            # Plot contourf
            if j == 0:
                var_plot = axs[i, j].contourf(longitude, latitude, var1_values.values, cmap=plt.cm.seismic,
                                              levels=np.linspace(var1_min, var1_max, num=100),
                                              norm=colors.TwoSlopeNorm(vmin=var1_min, vcenter=0, vmax=var1_max))
                axs[i, j].set_title(datasets[i] + ' ' + var1)  # Replace with appropriate title
                colorbars[i, j] = fig.colorbar(var_plot, ax=axs[i, j], orientation='vertical', shrink=0.85)
            elif j == 1:
                axs[i, j].yaxis.set_visible(False)
                var_plot = axs[i, j].contourf(longitude, latitude, var2_values.values, cmap=plt.cm.Reds,
                                              levels=np.linspace(var2_min, var2_max, num=100),
                                              norm=None)
                axs[i, j].set_title(datasets[i] + ' ' + var2)  # Replace with appropriate title
                # Add scientific notation to colorbar
                colorbars[i, j] = fig.colorbar(var_plot, ax=axs[i, j], orientation='vertical', shrink=0.85, format='%.0e')
            else:
                axs[i, j].yaxis.set_visible(False)
                # Use NWS Reflectivity colormap for the third column
                ctable = 'NWSReflectivity'
                norm, cmap = ctables.registry.get_with_steps(ctable, -0, 5)
                var_plot = axs[i, j].contourf(longitude, latitude, var3_values.values, cmap=cmap,
                                              levels=np.linspace(var3_min, var3_max, num=100), norm=norm)
                axs[i, j].set_title(datasets[i] + ' ' + var3)  # Replace with appropriate title
                # Add colorbar
                colorbars[i, j] = fig.colorbar(var_plot, ax=axs[i, j], orientation='vertical', shrink=0.85)

            # Add features and labels
            axs[i, j].add_feature(cfeature.STATES.with_scale('50m'))
            axs[i, j].add_feature(cfeature.LAND.with_scale('50m'))
            axs[i, j].add_feature(cfeature.OCEAN.with_scale('50m'))
            

            # Add gridlines
            gl = axs[i, j].gridlines(draw_labels=True)
            gl.top_labels = gl.right_labels = False
            if j > 0:
                gl.left_labels = False  # Disable left labels for middle and right columns
            if i < 2:
                gl.bottom_labels = False  # Disable bottom labels for top two rows
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 7}
            gl.ylabel_style = {'size': 7}


    for cbar in colorbars.flatten():
        if cbar:
            cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))  # Assuming you want 5 ticks
            for label in cbar.ax.get_yticklabels():
                label.set_fontsize(label.get_fontsize() * 0.7)

    # Set the extent of the map if cropping is enabled
    if crop:
        for i in range(3):
            for j in range(3):
                axs[i, j].set_extent([left_lon, right_lon, bottom_lat, top_lat], crs=ccrs.PlateCarree())

    # Adjust layout and save figure
    fig.suptitle('{time} : Lead Time {i} at Surface'.format(time=pd.to_datetime(ds_target.time.isel(time=frame).values), i=frame),
                 y=0.98, fontsize=16)
    fig.tight_layout()
    plt.savefig(f'{out_path}/lead_time_{frame}.png', dpi=300, bbox_inches='tight')
    plt.close()
