
import hydra
from climate_hdf5 import ClimateHDF5Datapipe
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
import datetime

from modulus.datapipes.climate import ERA5HDF5Datapipe
from modulus.distributed import DistributedManager

from numpy_zenith_angle import cos_zenith_angle

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    # configs
    data_dir = '/data/CMIP6_2D_TAS/train'
    stats_dir = '/data/CMIP6_2D_TAS/stats'
    channels = None
    batch_size = 4
    stride = 1
    dt = 6
    start_year = 2016
    num_steps = 6
    lsm_filename = './static/land_sea_mask.nc'
    geopotential_filename = './static/geopotential.nc'
    use_cos_zenith = True
    use_latlon = True
    
    # create data pipe
    dp = ClimateHDF5Datapipe(
        data_dir=data_dir,
        stats_dir=stats_dir,
        channels=channels,
        batch_size=batch_size,
        stride=stride,
        dt=dt,
        start_year=start_year,
        num_steps=num_steps,
        lsm_filename=lsm_filename,
        geopotential_filename=geopotential_filename,
        use_cos_zenith=use_cos_zenith,
        use_latlon=use_latlon,
        shuffle=True,
    )
    
    for data in dp:

        # check timestamp
        #for i in range(data[0]["timestamps"].shape[1]):
        #    tp = data[0]["timestamps"][0, i].detach().cpu().numpy()
        #    timestamp = datetime.datetime.fromtimestamp(tp.astype(int))
        #    print(timestamp)

        for i in range(num_steps):
            fig, ax = plt.subplots(1, 7, figsize=(20, 4))
            ax[0].imshow(data[0]["state_seq"][0, i, 0, :, :].detach().cpu().numpy(), origin="lower")
            ax[1].imshow(data[0]["land_sea_mask"][0, 0, :, :].detach().cpu().numpy(), origin="lower")
            ax[2].imshow(data[0]["geopotential"][0, 0, :, :].detach().cpu().numpy(), origin="lower")
            ax[3].imshow(data[0]["latlon"][0, 0, :, :].detach().cpu().numpy(), origin="lower")
            ax[4].imshow(data[0]["cos_latlon"][0, 0, :, :].detach().cpu().numpy(), origin="lower")
            ax[5].imshow(data[0]["cos_zenith"][0, i, 0, :, :].detach().cpu().numpy(), origin="lower")

            # get numpy zenith angle
            timestamp = datetime.datetime.fromtimestamp(data[0]["timestamps"][0, i].detach().cpu().numpy().astype(int))
            lat = data[0]["latlon"][0, 0, :, :].detach().cpu().numpy()
            lon = data[0]["latlon"][0, 1, :, :].detach().cpu().numpy()
            print((1.0 * 3600 + data[0]["timestamps"][0, i].detach().cpu().numpy() - 946756800.0) / (24 * 3600) / 36525.0)
            cos_zenith = cos_zenith_angle(timestamp, lon, lat)
            ax[6].imshow(cos_zenith, origin="lower")

            plt.show()
            plt.close()
   
    """
    datapipe = ERA5HDF5Datapipe(
        data_dir=data_dir,
        stats_dir=stats_dir,
        channels=channels,
        num_samples_per_year=1456,  # Need better shard fix
        batch_size=2,
        patch_size=(8, 8),
        num_workers=8,
        device=dist.device,
        world_size=dist.world_size,
    )
    
    for data in datapipe:
        print(data)
        print(data['invar'].shape)
    """

if __name__ == "__main__":
    main()
