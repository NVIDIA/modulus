# ignore_header_test

"""
climt/LICENSE
@mcgibbon
BSD License
Copyright (c) 2016, Rodrigo Caballero, modified by NVIDIA 2023
All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# code taken from climt repo https://github.com/CliMT/climt
# modified 2023: vectorization over coordinates and JIT compilation added

import datetime
import numpy as np
from typing import Union, Tuple, TypeVar

# numba stuff for parallelization
import numba as nb
from numba import jit, njit

# define helper type
dtype = np.float32


@jit(forceobj=True)
def _days_from_2000(model_time: np.ndarray) -> np.ndarray:
    """Get the days since year 2000.
    """
    # compute total days
    time_diff = model_time - datetime.datetime(2000, 1, 1, 12, 0)
    result = np.asarray(time_diff).astype("timedelta64[us]") / np.timedelta64(1, "D")
    result = result.astype(dtype)
    
    return result


@jit(forceobj=True)
def _greenwich_mean_sidereal_time(model_time: np.ndarray) -> np.ndarray:
    """
    Greenwich mean sidereal time, in radians.
    Reference:
        The AIAA 2006 implementation:
            http://www.celestrak.com/publications/AIAA/2006-6753/
    """
    jul_centuries = _days_from_2000(model_time) / 36525.0
    theta = dtype(67310.54841 + jul_centuries * (
        876600 * 3600
        + 8640184.812866
        + jul_centuries * (0.093104 - jul_centuries * 6.2 * 10e-6)
    ))

    theta_radians = np.deg2rad(theta / 240.0) % (2 * np.pi)
    theta_radians = theta_radians.astype(dtype)

    return theta_radians


@jit(forceobj=True)
def _local_mean_sidereal_time(model_time: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """
    Local mean sidereal time. requires longitude in radians.
    Ref:
        http://www.setileague.org/askdr/lmst.htm
    """
    return _greenwich_mean_sidereal_time(model_time) + longitude


@jit(forceobj=True)
def _sun_ecliptic_longitude(model_time: np.ndarray) -> np.ndarray:
    """
    Ecliptic longitude of the sun.
    Reference:
        http://www.geoastro.de/elevaz/basics/meeus.htm
    """
    julian_centuries = _days_from_2000(model_time) / 36525.0

    # mean anomaly calculation
    mean_anomaly = np.deg2rad(
        357.52910
        + 35999.05030 * julian_centuries
        - 0.0001559 * julian_centuries * julian_centuries
        - 0.00000048 * julian_centuries * julian_centuries * julian_centuries,
        dtype=dtype)

    # mean longitude
    mean_longitude = np.deg2rad(
        280.46645 + 36000.76983 * julian_centuries + 0.0003032 * (julian_centuries ** 2),
        dtype=dtype)

    d_l = np.deg2rad(
        (1.914600 - 0.004817 * julian_centuries - 0.000014 * (julian_centuries ** 2))
        * np.sin(mean_anomaly)
        + (0.019993 - 0.000101 * julian_centuries) * np.sin(2 * mean_anomaly)
        + 0.000290 * np.sin(3 * mean_anomaly),
        dtype=dtype)

    # true longitude
    return mean_longitude + d_l


@jit(forceobj=True)
def _obliquity_star(julian_centuries: np.ndarray) -> np.ndarray:
    """
    return obliquity of the sun
    Use 5th order equation from
    https://en.wikipedia.org/wiki/Ecliptic#Obliquity_of_the_ecliptic
    """
    return np.deg2rad(
        23.0
        + 26.0 / 60
        + 21.406 / 3600.0
        - (
            46.836769 * julian_centuries
            - 0.0001831 * (julian_centuries ** 2)
            + 0.00200340 * (julian_centuries ** 3)
            - 0.576e-6 * (julian_centuries ** 4)
            - 4.34e-8 * (julian_centuries ** 5)
        )
        / 3600.0,
        dtype=dtype)


@jit(forceobj=True)
def _right_ascension_declination(model_time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Right ascension and declination of the sun.
    Ref:
        http://www.geoastro.de/elevaz/basics/meeus.htm
    """
    julian_centuries = _days_from_2000(model_time) / 36525.0
    eps = _obliquity_star(julian_centuries.astype(dtype))

    eclon = _sun_ecliptic_longitude(model_time)
    x = np.cos(eclon)
    y = np.cos(eps) * np.sin(eclon)
    z = np.sin(eps) * np.sin(eclon)
    r = np.sqrt(1.0 - z * z)
    # sun declination
    declination = np.arctan2(z, r)
    # right ascension
    right_ascension = dtype(2. * np.arctan2(y, (x + r)))
    return right_ascension, declination


@jit(forceobj=True)
def _local_hour_angle(model_time: np.ndarray, longitude: np.ndarray, right_ascension: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hour angle at model_time for the given longitude and right_ascension
    longitude in radians. Return shape: [t, lon]
    Ref:
        https://en.wikipedia.org/wiki/Hour_angle#Relation_with_the_right_ascension
    """
    loc_mean = _local_mean_sidereal_time(model_time, longitude)

    # take the diff
    loc_hour_angle = loc_mean - right_ascension
    
    return loc_hour_angle

@jit(forceobj=True, cache=True)
def _star_cos_zenith(model_time: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    Return cosine of star zenith angle
    lon,lat in radians. Return shape: [t, lat, lon]
    Ref:
        Azimuth:
            https://en.wikipedia.org/wiki/Solar_azimuth_angle#Formulas
        Zenith:
            https://en.wikipedia.org/wiki/Solar_zenith_angle
    """
    # right ascension, only dependent on model times
    ra, dec = _right_ascension_declination(model_time)
    
    # compute local hour angle
    h_angle = _local_hour_angle(model_time, lon, ra)
    
    # compute zenith:
    cosine_zenith = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(h_angle)

    return cosine_zenith


@jit(forceobj=True, cache=True)   
def cos_zenith_angle(
    time: np.ndarray, lon: np.ndarray, lat: np.ndarray,
) -> np.ndarray:
    """
    Cosine of sun-zenith angle for lon, lat at time (UTC).
    If DataArrays are provided for the lat and lon arguments, their units will
    be assumed to be in degrees, unless they have a units attribute that
    contains "rad"; in that case they will automatically be converted to having
    units of degrees.
    Args:
        time: time in UTC
        lon: float or np.ndarray in degrees (E/W)
        lat: float or np.ndarray in degrees (N/S)
    Returns:
        float, np.ndarray
    """
    # convert deg -> rad
    lon_rad = np.deg2rad(lon, dtype=dtype)
    lat_rad = np.deg2rad(lat, dtype=dtype)

    # reshape all inputs
    lon_rad = np.expand_dims(lon_rad, axis=0)
    lat_rad = np.expand_dims(lat_rad, axis=0)
    time = np.reshape(time, (-1, 1, 1))

    result = _star_cos_zenith(time, lon_rad, lat_rad)

    return result


if __name__ == "__main__":

    # create grid
    lon = np.arange(0, 360, 20.)
    lat = np.arange(-90, 90.25, 10.)
    lat = lat[::-1]
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # model time
    model_time = np.asarray([datetime.datetime(2002, 1, 1, 12, 0, 0),
                             datetime.datetime(2002, 6, 1, 12, 0, 0),
                             datetime.datetime(2003, 1, 1, 12, 0, 0)])
     
    #test _days_from_2000
    days = _days_from_2000(model_time)
    print(days)

    # test _greenwich_mean_sidereal_time
    theta = _greenwich_mean_sidereal_time(model_time)
    print(theta)

    # test _local_mean_sidereal_time
    theta = _local_mean_sidereal_time(np.reshape(model_time, (-1,1,1)), np.expand_dims(lon_grid, axis=0))
    print(theta)

    # test _sun_ecliptic_longitude    
    eclon = _sun_ecliptic_longitude(model_time)
    print(eclon)

    # test _obliquity_star
    julian_centuries = _days_from_2000(model_time) / 36525.0
    eps = _obliquity_star(julian_centuries)
    print(eps)

    # test _right_ascension_declination
    ra, dec = _right_ascension_declination(model_time)
    print(ra, dec)

    # test cos_zenith_angle
    za = cos_zenith_angle(model_time, lat=lat_grid, lon=lon_grid)
    print("zenith angle", za)
