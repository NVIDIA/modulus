from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
import xarray as xr


def latlon_grid(
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (90, -90),
        (0, 360),
    ),
    shape: Tuple[int, int] = (1440,721)    
) -> np.ndarray :
    """Infer latitude and longitude coordinates from bounds and data shape."""

    # get latitudes and longitudes from data shape
    lat = np.linspace(*bounds[0], shape[0], dtype=np.float32)

    # does longitude wrap around the globe?
    lon_wraparound = ((bounds[1][0] % 360) == (bounds[1][1] % 360))
    if lon_wraparound:
        # treat differently from lat due to wrap-around
        lon = np.linspace(*bounds[1], shape[1]+1, dtype=np.float32)[:-1]
    else:
        lon = np.linspace(*bounds[1], shape[1], dtype=np.float32)

    return np.meshgrid(lat, lon, indexing='ij')


class Invariant(ABC):
    @abstractmethod
    def __call__(self, latlon: np.ndarray):
        pass


class LatLon(Invariant):
    def __init__(
        self,
        outputs: List[str] = ("sin_lat", "cos_lat", "sin_lon", "cos_lon")
    ):
        """
        Outputs latitude and longitude and their trigonometric functions.

        Parameters
        ----------
        outputs: List[str]
            List of outputs. Supported values are
            `{"lat", "lon", "sin_lat", "cos_lat", "sin_lon", "cos_lon"}`
        """
        self.outputs = outputs

    def __call__(self, latlon: np.ndarray):
        (lat, lon) = latlon

        vars = {"lat": lat, "lon": lon}

        # cos/sin latitudes and longitudes
        if "sin_lat" in self.outputs:
            vars["sin_lat"] = np.sin(np.deg2rad(lat))
        if "cos_lat" in self.outputs:
            vars["cos_lat"] = np.cos(np.deg2rad(lat))
        if "sin_lon" in self.outputs:
            vars["sin_lon"] = np.sin(np.deg2rad(lon))
        if "cos_lon" in self.outputs:
            vars["cos_lon"] = np.cos(np.deg2rad(lon))

        return np.stack([vars[o] for o in self.outputs], axis=0)


class FileInvariant(Invariant):
    def __init__(
        self,
        filename: str,
        var_name: str,
        normalize=False,
        interp_method='linear',
    ):
        """
        Loads an time-invariant variable from a NetCDF4 file. The file should
        contain one or more data variables of dimensions 
        `(channels, latitude, longitude)` as well as variables `latitude` and
        `longitude` specifying these coordinates. `latitude` and `longitude`
        can be either 2D or 1D.

        Parameters
        ----------
        filename: str
            Path to the file containing the variable
        var_name: str
            The variable in the file containing the data
        normalize: bool, optional
            If True, normalize the data by to zero-mean and unit variance.
            Default False.
        interp_method: str, optional
            Any argument accepted by xarray.DataArray.interp.
            Default 'linear'.
        """

        with xr.open_dataset(filename) as ds:
            self.data = ds[var_name].astype(np.float32)
            self.lat = ds["latitude"].to_numpy().astype(np.float32)
            self.lon = ds["longitude"].to_numpy().astype(np.float32)

        if (self.lat.ndim == 1):
            (self.lat, self.lon) = np.meshgrid(self.lat, self.lon, indexing='ij')

        if normalize:
            self.data = (self.data - self.data.mean()) / self.data.std()
        
        self.interp_method = interp_method

    def __call__(self, latlon: np.ndarray):
        (lat, lon) = latlon
        lat = xr.DataArray(lat, dims=["latitude", "longitude"])
        lon = xr.DataArray(lon, dims=["latitude", "longitude"])
        return self.data.interp(
            method=self.interp_method, latitude=lat, longitude=lon
        ).to_numpy()
