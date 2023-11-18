# Order of running the scripts

1. Run `data_download_simple.py`.
2. Run `post_processing.py`.

## Note

To run the `post_processing.py` script, you need the map files.
These map files can be generated using
[TempestRemap](https://github.com/ClimateGlobalChange/tempestremap) library.
Once the library is installed, the below sequence of commands can be run to generate the
map files.

```bash
GenerateRLLMesh \
    --lat 721 \
    --lon 1440 \
    --file out_latlon.g \
    --lat_begin 90 \
    --lat_end -90 \
    --out_format Netcdf4
GenerateCSMesh \
    --res <desired-res> \
    --file out_cubedsphere.g \
    --out_format Netcdf4
GenerateOverlapMesh \
    --a out_latlon.g \
    --b out_cubedsphere.g \
    --out overlap_latlon_cubedsphere.g \
    --out_format Netcdf4
GenerateOfflineMap \
    --in_mesh out_latlon.g \
    --out_mesh out_cubedsphere.g \
    --ov_mesh overlap_latlon_cubedsphere.g \
    --in_np 1 \
    --in_type FV \
    --out_type FV \
    --out_map map_LL_CS.nc \
    --out_format Netcdf4
GenerateOverlapMesh \
    --a out_cubedsphere.g \
    --b out_latlon.g \
    --out overlap_cubedsphere_latlon.g \
    --out_format Netcdf4
GenerateOfflineMap \
    --in_mesh out_cubedsphere.g \
    --out_mesh out_latlon.g \
    --ov_mesh overlap_cubedsphere_latlon.g \
    --in_np 1 \
    --in_type FV \
    --out_type FV \
    --out_map map_CS_LL.nc \
    --out_format Netcdf4
```
