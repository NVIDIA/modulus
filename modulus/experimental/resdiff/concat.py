import xarray

import sys
import dask.diagnostics

base = sys.argv[1:-1]
out = sys.argv[-1]

with dask.diagnostics.ProgressBar():
    t = xarray.open_mfdataset(base, group='prediction' , concat_dim='ensemble', combine='nested', chunks={"time": 1, "ensemble": 10})
    t.to_zarr(out, group='prediction')

    t = xarray.open_dataset(base[0], group='input', chunks={"time":1 })
    t.to_zarr(out, group='input', mode='a')

    t = xarray.open_dataset(base[0], group='truth', chunks={"time":1 })
    t.to_zarr(out, group='truth', mode='a')

    t = xarray.open_dataset(base[0])
    t.to_zarr(out, mode='a')
