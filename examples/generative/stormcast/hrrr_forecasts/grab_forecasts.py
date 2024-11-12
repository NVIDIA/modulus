import s3fs
import xarray as xr
import datetime
import dataclasses
import metpy
import cartopy.crs as ccrs
import cartopy.feature
from metpy.plots import ctables
import numpy as np
import matplotlib.pyplot as plt
import json


X_START= 579
X_END= 1219
Y_START= 273
Y_END= 785



projection = ccrs.LambertConformal(central_longitude=262.5, 
                                   central_latitude=38.5, 
                                   standard_parallels=(38.5, 38.5),
                                    globe=ccrs.Globe(semimajor_axis=6371229,
                                                     semiminor_axis=6371229))
fs = s3fs.S3FileSystem(anon=True)

#https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/python_data_loading.html
@dataclasses.dataclass
class ZarrId:
    run_hour: datetime.datetime
    level_type: str
    var_level: str
    var_name: str
    model_type: str
        
    def format_chunk_id(self, chunk_id):
        if self.model_type == "fcst": 
            # Extra id part since forecasts have an additional (time) dimension
            return "0." + str(chunk_id)
        else:
            return chunk_id

def create_hrrr_zarr_explorer_url(level_type, model_type, run_hour):
    url = "https://hrrrzarr.s3.amazonaws.com/index.html"
    url += run_hour.strftime(
        f"#{level_type}/%Y%m%d/%Y%m%d_%Hz_{model_type}.zarr/")
    return url

def load_dataset(urls):
    fs = s3fs.S3FileSystem(anon=True)    
    ds = xr.open_mfdataset([s3fs.S3Map(url, s3=fs) for url in urls], engine='zarr')
    
    #add the projection data
    ds = ds.rename(projection_x_coordinate="x", projection_y_coordinate="y")
    ds = ds.metpy.assign_crs(projection.to_cf())
    ds = ds.metpy.assign_latitude_longitude()    
    return ds

def create_s3_group_url(zarr_id, prefix=True):
    url = "s3://hrrrzarr/" if prefix else "" # Skip when using boto3
    url += zarr_id.run_hour.strftime(
        f"{zarr_id.level_type}/%Y%m%d/%Y%m%d_%Hz_{zarr_id.model_type}.zarr/")
    url += f"{zarr_id.var_level}/{zarr_id.var_name}"
    return url

def create_s3_subgroup_url(zarr_id, prefix=True):
    url = create_s3_group_url(zarr_id, prefix)
    url += f"/{zarr_id.var_level}"
    return url


if __name__ == "__main__":

    with open("analysis/case_studies.json") as f:
        case_studies = json.load(f)

    print(case_studies)



    for case_study_name in case_studies.keys():
        print(case_study_name)

        import os

        #make dir
        os.makedirs(case_study_name, exist_ok=True)

        run_hour=case_studies[case_study_name]["initial_time"]
        run_hour = datetime.datetime.strptime(run_hour, "%Y-%m-%d %H:%M")
        idx_west = case_studies[case_study_name]["idx_west"]
        idx_east = case_studies[case_study_name]["idx_east"]
        idx_north = case_studies[case_study_name]["idx_north"]
        idx_south = case_studies[case_study_name]["idx_south"]


        zarr_id = ZarrId(
                        run_hour=run_hour, #datetime.datetime(2020, 8, 1, 0), # Aug 1, 0Z
                        level_type="sfc",
                        var_level="entire_atmosphere",
                        var_name="REFC",
                        model_type="fcst"
                        )

        #ds = load_dataset([create_s3_group_url(zarr_id)])
        ds = load_dataset([create_s3_group_url(zarr_id), create_s3_subgroup_url(zarr_id)])

        #print(ds)
        ds = ds.isel(x=slice(X_START, X_END), y=slice(Y_START, Y_END))
        ds = ds.isel(x=slice(idx_west, idx_east), y=slice(idx_north, idx_south))

        #save dataset to zarr
        zarr_path = "/pscratch/sd/j/jpathak/hrrr_forecasts/" + case_study_name + ".zarr"
        os.makedirs(zarr_path, exist_ok=True)

        new_ds = xr.Dataset(
            data_vars=dict(
                REFC=(["time", "y", "x"], ds.REFC.values.astype(np.float32)),
            ),
            coords=dict(
                time=ds.time.values,
                longitude=( ["y", "x"], ds.longitude.values),
                latitude=( ["y", "x"], ds.latitude.values),
            ),
        )

        new_ds.to_zarr(zarr_path, mode="w")

        for t in range(12):
    
            norm, cmap = ctables.registry.get_with_steps('NWSReflectivity', -0, 5)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': projection})
            refc = new_ds.REFC.isel(time=t).squeeze()
            c = ax.contourf(new_ds.longitude.values, new_ds.latitude.values, refc, cmap=cmap, transform=ccrs.PlateCarree(), norm=norm, zorder=0)
            #state boundaries
            ax.add_feature(cartopy.feature.STATES.with_scale('50m'), linewidth=0.5, edgecolor='black', zorder=2)
            #colorbar
            fig.colorbar(c, ax=ax, orientation='horizontal', label='Reflectivity (dBZ)', pad=0.05, aspect=50)
    
            #plot title
            ax.set_title(f"HRRR Reflectivity (dBZ), forecast initialization time: {run_hour}, forecast lead: {t+1}, valid time: {new_ds.time.values[t]}")
    
            #save figure
            plt.savefig(f"{case_study_name}/z_{t}.png")
    
