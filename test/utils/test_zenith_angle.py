# ignore_header_test

# climt/LICENSE
# @mcgibbon
# BSD License
# Copyright (c) 2016, Rodrigo Caballero
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED

from datetime import datetime

import numpy as np
import pytest
from pytz import utc

from modulus.utils.zenith_angle import (
    _datetime_to_julian_century,
    _timestamp_to_julian_century,
    cos_zenith_angle,
    cos_zenith_angle_from_timestamp,
    toa_incident_solar_radiation_accumulated,
)


@pytest.mark.parametrize(
    "time, lon, lat, expected",
    (
        [datetime(2020, 3, 21, 12, 0, 0), 0.0, 0.0, 0.9994836252135212],
        [datetime(2020, 3, 21, 18, 0, 0), -90.0, 0.0, 0.9994760971063111],
        [datetime(2020, 3, 21, 18, 0, 0), 270.0, 0.0, 0.99947609879941],
        [datetime(2020, 7, 6, 12, 0, 0), -90.0, 0.0, -0.019703903874316815],
        [datetime(2020, 7, 6, 9, 0, 0), 40.0, 40.0, 0.9501802266240413],
        [datetime(2020, 7, 6, 12, 0, 0), 0.0, 90.0, 0.3843918031907148],
    ),
)
def test_zenith_angle(time, lon, lat, expected):
    time = time.replace(tzinfo=utc)
    assert cos_zenith_angle(time, lon, lat) == pytest.approx(expected, abs=1e-10)
    timestamp = time.timestamp()
    assert cos_zenith_angle_from_timestamp(timestamp, lon, lat) == pytest.approx(
        expected, abs=1e-10
    )


def test_zenith_angle_array():
    timestamp = np.array([0, 1, 2])[:, None, None]
    lat = np.array([0.0, 0.0])[None, :, None]
    lon = np.array([0.0])[None, None, :]
    out = cos_zenith_angle_from_timestamp(timestamp, lon, lat)
    assert out.shape == (3, 2, 1)


@pytest.mark.parametrize(
    "t",
    [
        datetime(2020, 7, 6, 9, 0, 0),
        datetime(2000, 1, 1, 12, 0, 0),
        datetime(2000, 7, 1, 12, 0, 0),
        datetime(2000, 7, 1, 12, 0, 0, tzinfo=utc),
    ],
)
def test_timestamp_to_julian_centuries(t):
    a = _datetime_to_julian_century(t)
    b = _timestamp_to_julian_century(t.replace(tzinfo=utc).timestamp())
    assert a == b


def test_toa():
    t = datetime(2000, 7, 1, 12, 0, 0, tzinfo=utc).timestamp()
    lat, lon = 0.0, 0.0
    ans = toa_incident_solar_radiation_accumulated(t, lat, lon)
    assert ans >= 0


def test_tisr_matches_cds():
    import os

    import cdsapi
    import matplotlib.pyplot as plt

    grid = (1, 1)
    area = (90, -180, -90, 180)
    format = "netcdf"

    client = cdsapi.Client()

    if not os.path.isfile("out.nc"):
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [212],
                "year": [2018, 2019, 2020],
                "month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "day": 1,
                "time": "00:00",
                "area": area,
                "grid": grid,
                "format": format,
            },
            "out.nc",
        )

    import xarray

    ds = xarray.open_dataset("out.nc")

    def plot_comparison(ds):

        t = ds.time.values.astype("datetime64[s]").astype("int64")

        # lat = np.linspace(-90, 90, 721)[:, None]
        # lon = np.linspace(0, 360, 1440, endpoint=False)[None, :]
        t, lat, lon = np.meshgrid(t, ds.latitude, ds.longitude, indexing="ij")

        tisr = toa_incident_solar_radiation_accumulated(t, lat, lon)

        fig, axs = plt.subplots(
            t.shape[0], 3, figsize=(15, 30), sharex=True, sharey=True
        )
        for i in range(t.shape[0]):
            a, b, c = axs[i].flat
            im = a.pcolormesh(ds.longitude, ds.latitude, ds.tisr[i] / 3600)
            plt.colorbar(im)
            im = b.pcolormesh(ds.longitude, ds.latitude, tisr[i] / 3600)
            plt.colorbar(im)
            im = c.pcolormesh(ds.longitude, ds.latitude, (tisr[i] - ds.tisr[i]) / 3600)
            plt.colorbar(im)

            time_str = ds.time[i].dt.strftime("%Y-%m-%d %H:%M:%S").item()
            a.set_ylabel(time_str)

            if i == 0:

                a.set_title("ERA5")
                b.set_title("Computed")
                c.set_title("Difference")
                fig.suptitle("Solar irradiance (W/m^2)")

    plot_comparison(ds.sel(time="2018"))
    plt.savefig("tisr.png")

    def plot_global_avg(ds):

        t = ds.time.values.astype("datetime64[s]").astype("int64")

        # lat = np.linspace(-90, 90, 721)[:, None]
        # lon = np.linspace(0, 360, 1440, endpoint=False)[None, :]
        t, lat, lon = np.meshgrid(t, ds.latitude, ds.longitude, indexing="ij")

        tisr = toa_incident_solar_radiation_accumulated(t, lat, lon)
        w = np.cos(np.deg2rad(lat))
        z = np.sum(w * tisr, axis=(1, 2)) / np.sum(w, axis=(1, 2))
        z_era = np.sum(w * ds.tisr, axis=(1, 2)) / np.sum(w, axis=(1, 2))

        plt.plot(ds.time, z / 3600, label="computed")
        plt.plot(ds.time, z_era / 3600, label="era5")
        plt.legend()
        plt.xticks(rotation=45)
        plt.title("Global average solar irradiance (W/m^2)")
        plt.tight_layout()

    plt.figure()
    plot_global_avg(ds)
    plt.savefig("line.png")
