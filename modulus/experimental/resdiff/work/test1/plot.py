# %%
import xarray as xr

def open_samples(f):
    root = xr.open_dataset(f)
    pred = xr.open_dataset(f, group="prediction")
    truth = xr.open_dataset(f, group="truth")

    pred = pred.merge(root)
    truth = truth.merge(root)

    truth = truth.set_coords(["lon", "lat"])
    pred = pred.set_coords(["lon", "lat"])
    return truth, pred, root

f = "/lustre/fsw/nvresearch/nbrenowitz/diffusions/samples/87376.nc"
truth, pred, root = open_samples(f)

# import matplotlib.pyplot as plt
# for v in pred:
#     plt.figure()
#     pred[v].plot(col="ensemble", row="time")

# %%
y = truth.assign_coords(ensemble='truth').expand_dims("ensemble")
plotme = xr.concat([pred, y], dim='ensemble')


pred['maximum_radar_reflectivity'][:5, :].plot(col='ensemble', row='time')

# %%
