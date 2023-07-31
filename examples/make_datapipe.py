
import hydra
from climate_hdf5 import ClimateHDF5Datapipe
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
import datetime

from modulus.datapipes.climate import ERA5HDF5Datapipe
from modulus.distributed import DistributedManager

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

        fig, ax = plt.subplots(1, 6, figsize=(20, 4))
        ax[0].imshow(data[0]["state_seq"][0, 0, 0, :, :].detach().cpu().numpy(), origin="lower")
        ax[1].imshow(data[0]["land_sea_mask"][0, 0, :, :].detach().cpu().numpy(), origin="lower")
        ax[2].imshow(data[0]["geopotential"][0, 0, :, :].detach().cpu().numpy(), origin="lower")
        ax[3].imshow(data[0]["latlon"][0, 0, :, :].detach().cpu().numpy(), origin="lower")
        ax[4].imshow(data[0]["cos_latlon"][0, 0, :, :].detach().cpu().numpy(), origin="lower")
        ax[5].imshow(data[0]["cos_zenith"][0, :, :].detach().cpu().numpy(), origin="lower")
        plt.savefig("test.png")
        exit()
   
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
