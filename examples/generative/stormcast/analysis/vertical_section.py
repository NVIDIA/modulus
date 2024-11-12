import os
import sys 
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.getcwd())

import numpy as np
import zarr
import xarray
import argparse
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.vertical_section_xr import interpolate_vertical
from metpy.units import units
import metpy
import matplotlib.gridspec as gridspec


def create_section(ds_targ, ds_pred_edm, ds_pred_noedm, sec_idx, file_path, variable = 'enthalpy', config=None):
    ds_targ_vert = interpolate_vertical(ds_targ, sec_idx, diagnostics=False)
    ds_pred_edm_vert = interpolate_vertical(ds_pred_edm, sec_idx, diagnostics=False)
    ds_pred_noedm_vert = interpolate_vertical(ds_pred_noedm, sec_idx, diagnostics=False)
    if config is not None:
        xmin = config['idx_west']
        xmax = config['idx_east'] + 1
        ds_targ_vert = ds_targ_vert.isel(x=slice(xmin, xmax))
        ds_pred_edm_vert = ds_pred_edm_vert.isel(x=slice(xmin, xmax))
        ds_pred_noedm_vert = ds_pred_noedm_vert.isel(x=slice(xmin, xmax))
    else:
        xmin = None
        xmax = None

    lon = ds_targ.longitude.isel(y=sec_idx, x=slice(xmin, xmax))
    refc_targ = ds_targ.refc.isel(y=sec_idx, x=slice(xmin, xmax))
    refc_pred_edm = ds_pred_edm.refc.isel(y=sec_idx, x=slice(xmin, xmax))
    refc_pred_noedm = ds_pred_noedm.refc.isel(y=sec_idx, x=slice(xmin, xmax))


    if variable == 'enthalpy':
        cp = 1004
        lv = 2.5*10**6
        mixing_ratio = metpy.calc.mixing_ratio_from_specific_humidity(ds_targ_vert['q_comb']*units('kg/kg')).metpy.convert_units('kg/kg').metpy.dequantify()
        Tv = metpy.calc.virtual_potential_temperature(ds_targ_vert['p_comb']*units.Pa,ds_targ_vert['t_comb']*units.K,mixing_ratio*units('g/kg')).metpy.convert_units('degC').metpy.dequantify()
        target_h_ = cp*Tv+lv*ds_targ_vert['q_comb']
        
        mixing_ratio = metpy.calc.mixing_ratio_from_specific_humidity(ds_pred_edm_vert['q_comb']*units('kg/kg')).metpy.convert_units('kg/kg').metpy.dequantify()
        Tv = metpy.calc.virtual_potential_temperature(ds_pred_edm_vert['p_comb']*units.Pa,ds_pred_edm_vert['t_comb']*units.K,mixing_ratio*units('g/kg')).metpy.convert_units('degC').metpy.dequantify()
        pred_edm_h_ = cp*Tv+lv*ds_pred_edm_vert['q_comb']
        
        mixing_ratio = metpy.calc.mixing_ratio_from_specific_humidity(ds_pred_noedm_vert['q_comb']*units('kg/kg')).metpy.convert_units('kg/kg').metpy.dequantify()
        Tv = metpy.calc.virtual_potential_temperature(ds_pred_noedm_vert['p_comb']*units.Pa,ds_pred_noedm_vert['t_comb']*units.K,mixing_ratio*units('g/kg')).metpy.convert_units('degC').metpy.dequantify()
        pred_noedm_h_ = cp*Tv+lv*ds_pred_noedm_vert['q_comb']
        mean_profile = np.mean(target_h_, axis=(0,2))
        target_h = target_h_[:, :, :] - mean_profile
        pred_edm_h = pred_edm_h_[:, :, :] - mean_profile
        pred_noedm_h = pred_noedm_h_[:, :, :] - mean_profile
    elif variable=='theta_eq':
        dewpoint_temp = metpy.calc.dewpoint_from_specific_humidity(ds_targ_vert['p_comb']*units.Pa,ds_targ_vert['t_comb']*units.K,
                                                                      ds_targ_vert['q_comb']*units('kg/kg')).metpy.convert_units('degC').metpy.dequantify()
        target_h = metpy.calc.equivalent_potential_temperature(ds_targ_vert['p_comb']*units.Pa,ds_targ_vert['t_comb']*units.K,
                                                                          dewpoint_temp*units.degC).metpy.convert_units('degC').metpy.dequantify()
        dewpoint_temp = metpy.calc.dewpoint_from_specific_humidity(ds_pred_edm_vert['p_comb']*units.Pa,ds_pred_edm_vert['t_comb']*units.K,
                                                                      ds_pred_edm_vert['q_comb']*units('kg/kg')).metpy.convert_units('degC').metpy.dequantify()
        pred_edm_h = metpy.calc.equivalent_potential_temperature(ds_pred_edm_vert['p_comb']*units.Pa,ds_pred_edm_vert['t_comb']*units.K,
                                                                          dewpoint_temp*units.degC).metpy.convert_units('degC').metpy.dequantify()
        dewpoint_temp = metpy.calc.dewpoint_from_specific_humidity(ds_pred_noedm_vert['p_comb']*units.Pa,ds_pred_noedm_vert['t_comb']*units.K,
                                                                      ds_pred_noedm_vert['q_comb']*units('kg/kg')).metpy.convert_units('degC').metpy.dequantify()
        pred_noedm_h = metpy.calc.equivalent_potential_temperature(ds_pred_noedm_vert['p_comb']*units.Pa,ds_pred_noedm_vert['t_comb']*units.K,
                                                                          dewpoint_temp*units.degC).metpy.convert_units('degC').metpy.dequantify()
    elif variable=='Tv':
        mixing_ratio = metpy.calc.mixing_ratio_from_specific_humidity(ds_targ_vert['q_comb']*units('kg/kg')).metpy.convert_units('kg/kg').metpy.dequantify()
        target_h_ = metpy.calc.virtual_potential_temperature(ds_targ_vert['p_comb']*units.Pa,ds_targ_vert['t_comb']*units.K,mixing_ratio*units('g/kg')).metpy.convert_units('degC').metpy.dequantify()
        mixing_ratio = metpy.calc.mixing_ratio_from_specific_humidity(ds_pred_edm_vert['q_comb']*units('kg/kg')).metpy.convert_units('kg/kg').metpy.dequantify()
        pred_edm_h_ = metpy.calc.virtual_potential_temperature(ds_pred_edm_vert['p_comb']*units.Pa,ds_pred_edm_vert['t_comb']*units.K,mixing_ratio*units('g/kg')).metpy.convert_units('degC').metpy.dequantify()
        mixing_ratio = metpy.calc.mixing_ratio_from_specific_humidity(ds_pred_noedm_vert['q_comb']*units('kg/kg')).metpy.convert_units('kg/kg').metpy.dequantify()
        pred_noedm_h_ = metpy.calc.virtual_potential_temperature(ds_pred_noedm_vert['p_comb']*units.Pa,ds_pred_noedm_vert['t_comb']*units.K,mixing_ratio*units('g/kg')).metpy.convert_units('degC').metpy.dequantify()
        mean_profile = np.mean(target_h_, axis=(0,2))
        target_h = target_h_[:, :, :] - mean_profile
        pred_edm_h = pred_edm_h_[:, :, :] - mean_profile
        pred_noedm_h = pred_noedm_h_[:, :, :] - mean_profile
    elif variable=='buoyancy':
        mixing_ratio = metpy.calc.mixing_ratio_from_specific_humidity(ds_targ_vert['q_comb']*units('kg/kg')).metpy.convert_units('kg/kg').metpy.dequantify()
        target_h_ = metpy.calc.virtual_potential_temperature(ds_targ_vert['p_comb']*units.Pa,ds_targ_vert['t_comb']*units.K,mixing_ratio*units('g/kg')).metpy.convert_units('degC').metpy.dequantify()
        mixing_ratio = metpy.calc.mixing_ratio_from_specific_humidity(ds_pred_edm_vert['q_comb']*units('kg/kg')).metpy.convert_units('kg/kg').metpy.dequantify()
        pred_edm_h_ = metpy.calc.virtual_potential_temperature(ds_pred_edm_vert['p_comb']*units.Pa,ds_pred_edm_vert['t_comb']*units.K,mixing_ratio*units('g/kg')).metpy.convert_units('degC').metpy.dequantify()
        mixing_ratio = metpy.calc.mixing_ratio_from_specific_humidity(ds_pred_noedm_vert['q_comb']*units('kg/kg')).metpy.convert_units('kg/kg').metpy.dequantify()
        pred_noedm_h_ = metpy.calc.virtual_potential_temperature(ds_pred_noedm_vert['p_comb']*units.Pa,ds_pred_noedm_vert['t_comb']*units.K,mixing_ratio*units('g/kg')).metpy.convert_units('degC').metpy.dequantify()
        mean_profile = np.mean(target_h_, axis=(0,2))
        target_h = (target_h_[:, :, :] - mean_profile)/mean_profile*9.81
        pred_edm_h = (pred_edm_h_[:, :, :] - mean_profile)/mean_profile*9.81
        pred_noedm_h = (pred_noedm_h_[:, :, :] - mean_profile)/mean_profile*9.81
    elif variable=='relative_humidity':
        target_h = metpy.calc.relative_humidity_from_specific_humidity(ds_targ_vert['p_comb']*units.Pa,ds_targ_vert['t_comb']*units.K,ds_targ_vert['q_comb']*units('kg/kg')).metpy.dequantify()
        pred_edm_h = metpy.calc.relative_humidity_from_specific_humidity(ds_pred_edm_vert['p_comb']*units.Pa,ds_pred_edm_vert['t_comb']*units.K,ds_pred_edm_vert['q_comb']*units('kg/kg')).metpy.dequantify()
        pred_noedm_h = metpy.calc.relative_humidity_from_specific_humidity(ds_pred_noedm_vert['p_comb']*units.Pa,ds_pred_noedm_vert['t_comb']*units.K,ds_pred_edm_vert['q_comb']*units('kg/kg')).metpy.dequantify()
   
    all_q_values = np.concatenate([target_h, pred_edm_h, pred_noedm_h])
    extrema_val = np.nanmax(np.abs(all_q_values))
    
    z_levels_grid, lon_grid  = np.meshgrid(target_h.z_levels, target_h.x, indexing='ij')
    max_ref = np.max([np.max(refc_targ),np.max(refc_targ),np.max(refc_targ)])
    min_ref = np.min([np.min(refc_targ),np.min(refc_targ),np.min(refc_targ)])

    fig = plt.figure(figsize=(20, 8))  # Modified for a wider figure
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1], width_ratios=[1, 1, 1.1], figure=fig)
    axes = np.array([[fig.add_subplot(gs[0, i]) for i in range(3)],
                    [fig.add_subplot(gs[1, i]) for i in range(3)]])

    colorbars = [None]*3
    def update(frame):
        for ax in axes.ravel():
            ax.clear()

        for cb, ax in zip(colorbars, axes.ravel()):
            if cb is not None:
                cb.remove()

        # Adjust figure layout to provide space for the colorbar
        fig.subplots_adjust(right=0.9)  # Adjust as needed for your specific figure size

        x_min = np.min(lon_grid)
        x_max = np.max(lon_grid)

        variables = ['target', 'diffusion', 'regression']
        datasets = [target_h, pred_edm_h, pred_noedm_h]
        titles = [f"{var} {variable} - IC" for var in variables]
        for i, (data, title) in enumerate(zip(datasets, titles)):
            pcm = axes[0, i].contourf(lon_grid, z_levels_grid, data[frame, :, :],
                                    levels=np.linspace(-extrema_val, extrema_val, num=100), cmap='seismic')
            axes[0, i].set_title(title)
            axes[0, i].set_ylabel('Vertical Height (z_levels)')
            axes[0, i].set_ylim([0, 6100])
            axes[0, i].set_xlim([x_min, x_max])
            axes[0, i].set_xticklabels([])

            # Place colorbar for the last panel
            if i == 2:
                colorbars[i] = fig.colorbar(pcm, ax=axes[:, i].ravel().tolist(), use_gridspec=True, location='right')

        # Line plots for the lower row
        ref_datasets = [refc_targ, refc_pred_edm, refc_pred_noedm]
        for i, ref_data in enumerate(ref_datasets):
            axes[1, i].plot(lon, ref_data[frame, :], color="darkorange", linewidth=1)
            axes[1, i].fill_between(lon, min_ref, ref_data[frame, :], color="darkorange", alpha =0.5)
            axes[1, i].set_xlabel('Longitude (degrees)')
            axes[1, i].set_ylabel('ref (dbz)')
            axes[1, i].set_ylim([min_ref, max_ref])
            axes[1, i].spines['top'].set_visible(False)
            axes[1, i].spines['right'].set_visible(False)
            
    num_frames = pred_edm_h.shape[0]
    anim = FuncAnimation(fig, update, frames=num_frames,interval=100, repeat=True)
    anim.save(file_path, writer='pillow', fps=2)    
    plt.close(fig)
    
    return 

def main():
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="full path to the simulation directory",        
    )
    parser.add_argument(
        "sec_idx",
        type=str,
        help="section index location",
        default = 250,
    )
    args = parser.parse_args()
    path = args.path
    sec_idx = int(args.sec_idx)
    # load netcdf 
    ds_targ = xarray.open_dataset(os.path.join(path,"ds_targ.nc"))
    ds_pred_edm = xarray.open_dataset(os.path.join(path,"ds_pred_edm.nc"))
    ds_pred_noedm = xarray.open_dataset(os.path.join(path,"ds_pred_noedm.nc"))
    file_path = os.path.join(path, 'animation.gif')
    create_section(ds_targ, ds_pred_edm, ds_pred_noedm, sec_idx, file_path)

if __name__ == "__main__":
    main()