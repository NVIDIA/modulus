import xarray as xr
import pandas as pd
import numpy as np
import metpy
from metpy.units import units
import os
import math
from pyproj import Proj, transform

def calculate_fluctuations(velocity):
    mean_velocity = velocity.mean(dim='time')
    fluctuation = velocity - mean_velocity
    return fluctuation

def calculate_tke(u, v, w):
    u_prime = calculate_fluctuations(u)
    v_prime = calculate_fluctuations(v)
    w_prime = calculate_fluctuations(w)

    tke = 0.5 * np.sqrt((u_prime**2 + v_prime**2 + w_prime**2))
    return tke

def calculate_buoyancy(ds):

    # Define constants
    g = 9.81  # gravitational acceleration (in m/s^2)

    # Get data from the dataset
    virt_temp = ds['virtual_temp']
    virt_temp_env = virt_temp.mean(dim='time')
    buoyancy = g*((virt_temp-virt_temp_env)/virt_temp_env)

    return buoyancy

def degrees_to_meters(degrees):
    earth_circumference_km = 40075.0  # Earth's circumference in kilometers
    one_degree_in_km = earth_circumference_km / 360.0  # One degree is a fraction of the Earth's circumference

    meters = degrees * (one_degree_in_km * 1000.0)  # Convert to meters
    return meters

def calculate_divergence_rh(ds):
    proj_params = {'a': 6371229, 'b': 6371229, 'proj': 'lcc', 'lon_0': 262.5, 'lat_0': 38.5, 'lat_1': 38.5, 'lat_2': 38.5} #from HRRR source grib file
    proj = Proj(proj_params)
    #lons,lats = proj(ds.longitude.isel(y=0).values,ds.latitude.isel(x=0).values)
    dy = degrees_to_meters(np.diff(ds.latitude.isel(x=0).values))
    dx = degrees_to_meters(np.diff(ds.longitude.isel(y=0).values))
    divergence = metpy.calc.divergence(ds.u_comb*units('m/s'),ds.v_comb*units('m/s'),dx=dx*units('m'),
                      dy=dy*units('m'),crs=proj.srs).metpy.dequantify()
    #ds['mixing_ratio'] = metpy.calc.mixing_ratio_from_specific_humidity(ds['q_comb']*units('kg/kg')).metpy.convert_units('kg/kg').metpy.dequantify()
    ds['relative_humidity'] = metpy.calc.relative_humidity_from_specific_humidity(ds['p_comb']*units.Pa,ds['t_comb']*units.K,ds['q_comb']*units('kg/kg')).metpy.dequantify()
    ds['divergence'] = divergence
    return ds


def add_diagnostics(ds,interpolate=False,latitude=None):
    ds['pot_temp'] = metpy.calc.potential_temperature(ds['p_comb']*units.Pa,ds['t_comb']*units.K).metpy.convert_units('degC').metpy.dequantify()
    #ds['static_stability'] = metpy.calc.static_stability(ds['p_comb']*units.Pa,ds['t_comb']*units.K,vertical_dim=1).metpy.dequantify()
    ds['dewpoint_temp'] = metpy.calc.dewpoint_from_specific_humidity(ds['p_comb']*units.Pa,ds['t_comb']*units.K,
                                                                      ds['q_comb']*units('kg/kg')).metpy.convert_units('degC').metpy.dequantify()
    ds['equivalent_theta'] = metpy.calc.equivalent_potential_temperature(ds['p_comb']*units.Pa,ds['t_comb']*units.K,
                                                                          ds['dewpoint_temp']*units.degC).metpy.convert_units('degC').metpy.dequantify()
    ds['mixing_ratio'] = metpy.calc.mixing_ratio_from_specific_humidity(ds['q_comb']*units('kg/kg')).metpy.convert_units('kg/kg').metpy.dequantify()
    #ds['moist_static_energy'] = metpy.calc.moist_static_energy(ds['z_comb']*units.m,ds['t_comb']*units.K,ds['q_comb']*units('kg/kg')).metpy.dequantify()
    ds['relative_humidity'] = metpy.calc.relative_humidity_from_specific_humidity(ds['p_comb']*units.Pa,ds['t_comb']*units.K,ds['q_comb']*units('kg/kg')).metpy.dequantify()
    ds['saturation_theta_e'] = metpy.calc.saturation_equivalent_potential_temperature(ds['p_comb']*units.Pa,ds['t_comb']*units.K).metpy.convert_units('degC').metpy.dequantify()
    ds['saturation_mixing_ratio'] = metpy.calc.saturation_mixing_ratio(ds['p_comb']*units.Pa,ds['t_comb']*units.K).metpy.convert_units('g/kg').metpy.dequantify()
    ds['theta_v'] = metpy.calc.virtual_potential_temperature(ds['p_comb']*units.Pa,ds['t_comb']*units.K,ds['mixing_ratio']*units('g/kg')).metpy.convert_units('degC').metpy.dequantify()
    #ds['geopotential'] = metpy.calc.height_to_geopotential(ds['z_comb']*units.m).metpy.dequantify()
    ds['virtual_temp'] = metpy.calc.virtual_temperature(ds['t_comb']*units.K,ds['mixing_ratio']*units('kg/kg')).metpy.dequantify()
    #ds['tke'] = calculate_tke(ds['u_comb'], ds['v_comb'], ds['w_comb'])
    #ds['buoyancy'] = calculate_buoyancy(ds)
    ds['divergence'] = calculate_divergence(ds)

    return ds

