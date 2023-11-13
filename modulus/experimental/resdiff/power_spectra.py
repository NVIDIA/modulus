# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import xarray
import numpy as np
from scipy.signal import periodogram
from scipy.fft import dctn, irfft
import matplotlib.pyplot as plt
import typer
import os


def open_data(file, group=False):

    root = xarray.open_dataset(file)
    root = root.set_coords(['lat', 'lon'])
    ds = xarray.open_dataset(file, group=group)
    ds.coords.update(root.coords)
    ds.attrs.update(root.attrs)

    return ds



def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Earth radius in meters
    earth_radius = 6371000  # Approximate value for the average Earth radius

    # Calculate differences in latitude and longitude
    dlat_rad = lat2_rad - lat1_rad
    dlon_rad = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(dlat_rad / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon_rad / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_meters = earth_radius * c

    return distance_meters


def compute_power_spectrum(data, d):
    # Compute the 2D FFT along the second dimension
    fft_data = np.fft.fft(data, axis=-2)
    
    # Compute the power spectrum by taking the absolute value and squaring
    power_spectrum = np.abs(fft_data)**2
    
    # Scale the power spectrum based on the sampling interval 'd'
    power_spectrum /= (data.shape[-1] * d)
    freqs = np.fft.fftfreq(data.shape[-1], d)
    
    return freqs, power_spectrum


def power_spectra_to_acf(f, pw):
    """Convert one sided power spectrum to autocorrelation function

    Args:
        f: frequencies
        pw: power spectral density (V ** 2 / Hz)

    Returns:
        
    """
    pw = pw.copy()
    pw[0] = 0
    # magic factor 4 comes from periodogram/irfft stuff
    # 1) a factor 2 comes from the  periodogram being one-sided.
    # I don't fully understasnd, but this ensures the acf is 1 at r=0
    sig2 = np.sum(pw * f[1]) * 4
    acf = irfft(pw) / sig2
    return acf


def average_power_spectrum(data, d):
    # Compute the power spectrum along the second dimension for each row
    freqs, power_spectra = periodogram(data, fs=1/d, axis=-1)
    
    # Average along the first dimension
    while power_spectra.ndim > 1:
        power_spectra = power_spectra.mean(axis=0)
    
    return freqs, power_spectra


def main(file, output):

    def savefig(name):
        path = os.path.join(output, name + ".png")
        plt.savefig(path)

    samples = {}
    samples["prediction"] = open_data(file, group="prediction")
    samples["prediction_mean"] = samples["prediction"].mean("ensemble")
    samples["truth"] = open_data(file, group="truth")
    samples["ERA5"] = open_data(file, group="input")

    prediction = samples["prediction"]
    lat = prediction.lat
    lon = prediction.lon

    dx = haversine(lat[0, 0], lon[0, 0], lat[1, 0], lon[1, 0])
    dy = haversine(lat[0, 0], lon[0, 0], lat[0, 1], lon[0, 1])
    print(dx, dy)
    # the approximate resolution is dx=dy=2000m

    # in km
    d = 2


    # Plot the power spectrum
    for name, data in samples.items():
        freqs, spec_x = average_power_spectrum(data.eastward_wind_10m, d=d)
        _, spec_y = average_power_spectrum(data.northward_wind_10m, d=d)
        spec = spec_x + spec_y
        plt.loglog(freqs, spec, label=name)
        plt.xlabel('Frequency (1/km)')
        plt.ylabel('Power Spectrum')
        plt.ylim(bottom=1e-1)
    plt.title("Kinetic Energy power spectra")
    plt.grid()
    plt.legend()
    savefig("ke-spectra")


    plt.figure()
    for name, data in samples.items():
        freqs, spec = average_power_spectrum(data.temperature_2m, d=d)
        plt.loglog(freqs, spec, label=name)
        plt.xlabel('Frequency (1/km)')
        plt.ylabel('Power Spectrum')
        plt.ylim(bottom=1e-1)
    plt.title("T2M Power spectra")
    plt.grid()
    plt.legend()
    savefig("t2m-spectra")

    plt.figure()
    for name, data in samples.items():
        try:
            freqs, spec = average_power_spectrum(data.maximum_radar_reflectivity, d=d)
        except AttributeError:
            continue
        plt.loglog(freqs, spec, label=name)
        plt.xlabel('Frequency (1/km)')
        plt.ylabel('Power Spectrum')
        plt.ylim(bottom=1e-1)
    plt.title("Reflectivity Power spectra")
    plt.grid()
    plt.legend()
    savefig("reflectivity-spectra")


if __name__ == "__main__":
    typer.run(main)