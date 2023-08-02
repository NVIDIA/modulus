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
# OF THE POSSIBILITY OF SUCH DAMAGE.


import datetime
import numpy as np
from typing import Union, TypeVar

# helper type
dtype = np.float32

T = TypeVar("T", np.ndarray, float)


def _ensure_units_of_degrees(da):  # pragma: no cover
    """Ensure that the units of the DataArray are in degrees."""
    units = da.attrs.get("units", "").lower()
    if "rad" in units:
        return np.rad2deg(da).assign_attrs(units="degrees")
    else:
        return da


def cos_zenith_angle(
    time: Union[T, datetime.datetime],
    lon: T,
    lat: T,
) -> T:  # pragma: no cover
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

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0)
    >>> angle = cos_zenith_angle(model_time, lat=360, lon=120)
    >>> abs(angle - -0.447817277) < 1e-6
    True
    """
    lon_rad = np.deg2rad(lon, dtype=dtype)
    lat_rad = np.deg2rad(lat, dtype=dtype)
    return _star_cos_zenith(time, lon_rad, lat_rad)


def _days_from_2000(model_time):  # pragma: no cover
    """Get the days since year 2000.

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0)
    >>> _days_from_2000(model_time)
    731.0
    """
    date_type = type(np.asarray(model_time).ravel()[0])
    if date_type not in [datetime.datetime]:
        raise ValueError(
            f"model_time has an invalid date type. It must be "
            f"datetime.datetime. Got {date_type}."
        )
    return _total_days(model_time - date_type(2000, 1, 1, 12, 0))


def _total_days(time_diff):  # pragma: no cover
    """
    Total time in units of days
    """
    return np.asarray(time_diff).astype("timedelta64[us]") / np.timedelta64(1, "D")


def _greenwich_mean_sidereal_time(model_time):  # pragma: no cover
    """
    Greenwich mean sidereal time, in radians.
    Reference:
        The AIAA 2006 implementation:
            http://www.celestrak.com/publications/AIAA/2006-6753/

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0)
    >>> g_time = _greenwich_mean_sidereal_time(model_time)
    >>> abs(g_time - 4.903831411) < 1e-8
    True
    """
    jul_centuries = _days_from_2000(model_time) / 36525.0
    theta = 67310.54841 + jul_centuries * (
        876600 * 3600
        + 8640184.812866
        + jul_centuries * (0.093104 - jul_centuries * 6.2 * 10e-6)
    )

    theta_radians = np.deg2rad(theta / 240.0) % (2 * np.pi)

    return theta_radians


def _local_mean_sidereal_time(model_time, longitude):  # pragma: no cover
    """
    Local mean sidereal time. requires longitude in radians.
    Ref:
        http://www.setileague.org/askdr/lmst.htm


    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0)
    >>> l_time = _local_mean_sidereal_time(model_time, np.deg2rad(90))
    >>> abs(l_time - 6.474627737) < 1e-8
    True
    """
    return _greenwich_mean_sidereal_time(model_time) + longitude


def _sun_ecliptic_longitude(model_time):  # pragma: no cover
    """
    Ecliptic longitude of the sun.
    Reference:
        http://www.geoastro.de/elevaz/basics/meeus.htm

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0)
    >>> lon = _sun_ecliptic_longitude(model_time)
    >>> abs(lon - 17.469114444) < 1e-8
    True
    """
    julian_centuries = _days_from_2000(model_time) / 36525.0

    # mean anomaly calculation
    mean_anomaly = np.deg2rad(
        357.52910
        + 35999.05030 * julian_centuries
        - 0.0001559 * julian_centuries * julian_centuries
        - 0.00000048 * julian_centuries * julian_centuries * julian_centuries
    )

    # mean longitude
    mean_longitude = np.deg2rad(
        280.46645 + 36000.76983 * julian_centuries + 0.0003032 * (julian_centuries**2)
    )

    d_l = np.deg2rad(
        (1.914600 - 0.004817 * julian_centuries - 0.000014 * (julian_centuries**2))
        * np.sin(mean_anomaly)
        + (0.019993 - 0.000101 * julian_centuries) * np.sin(2 * mean_anomaly)
        + 0.000290 * np.sin(3 * mean_anomaly)
    )

    # true longitude
    return mean_longitude + d_l


def _obliquity_star(julian_centuries):  # pragma: no cover
    """
    return obliquity of the sun
    Use 5th order equation from
    https://en.wikipedia.org/wiki/Ecliptic#Obliquity_of_the_ecliptic

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0)
    >>> julian_centuries = _days_from_2000(model_time) / 36525.0
    >>> obl = _obliquity_star(julian_centuries)
    >>> abs(obl - 0.409088056) < 1e-8
    True
    """
    return np.deg2rad(
        23.0
        + 26.0 / 60
        + 21.406 / 3600.0
        - (
            46.836769 * julian_centuries
            - 0.0001831 * (julian_centuries**2)
            + 0.00200340 * (julian_centuries**3)
            - 0.576e-6 * (julian_centuries**4)
            - 4.34e-8 * (julian_centuries**5)
        )
        / 3600.0
    )


def _right_ascension_declination(model_time):  # pragma: no cover
    """
    Right ascension and declination of the sun.
    Ref:
        http://www.geoastro.de/elevaz/basics/meeus.htm

    Example:
    --------
    >>> model_time = datetime.datetime(2002, 1, 1, 12, 0, 0)
    >>> out1, out2 = _right_ascension_declination(model_time)
    >>> abs(out1 - -1.363787213) < 1e-8
    True
    >>> abs(out2 - -0.401270126) < 1e-8
    True
    """
    julian_centuries = _days_from_2000(model_time) / 36525.0
    eps = _obliquity_star(julian_centuries)

    eclon = _sun_ecliptic_longitude(model_time)
    x = np.cos(eclon)
    y = np.cos(eps) * np.sin(eclon)
    z = np.sin(eps) * np.sin(eclon)
    r = np.sqrt(1.0 - z * z)
    # sun declination
    declination = np.arctan2(z, r)
    # right ascension
    right_ascension = 2 * np.arctan2(y, (x + r))
    return right_ascension, declination


def _local_hour_angle(model_time, longitude, right_ascension):  # pragma: no cover
    """
    Hour angle at model_time for the given longitude and right_ascension
    longitude in radians
    Ref:
        https://en.wikipedia.org/wiki/Hour_angle#Relation_with_the_right_ascension
    """
    return _local_mean_sidereal_time(model_time, longitude) - right_ascension


def _star_cos_zenith(model_time, lon, lat):  # pragma: no cover
    """
    Return cosine of star zenith angle
    lon,lat in radians
    Ref:
        Azimuth:
            https://en.wikipedia.org/wiki/Solar_azimuth_angle#Formulas
        Zenith:
            https://en.wikipedia.org/wiki/Solar_zenith_angle
    """

    ra, dec = _right_ascension_declination(model_time)
    h_angle = _local_hour_angle(model_time, lon, ra)

    cosine_zenith = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(
        h_angle
    )

    return cosine_zenith
