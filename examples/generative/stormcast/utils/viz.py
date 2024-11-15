
import zarr
import matplotlib.pyplot as plt
import datetime
import cartopy.crs
import cartopy.feature
import os
import pathlib
import numpy as np



def make_movie(path: str, dest: str, output_format: str = "gif", output_name: str = "out"):    
    time_str = pathlib.Path(path).parent.name.replace('_', ':')
    initial_time = datetime.datetime.fromisoformat(time_str)

    g = zarr.open(path, mode="r")
    target = g["target"]["refc"]
    prediction = g["edm_prediction"]["refc"]
    lat = g["latitude"][:]
    lon = g["longitude"][:]

    i = 1
    projection = cartopy.crs.LambertConformal(
        central_longitude=-95, central_latitude=35
    )
    src_crs = cartopy.crs.PlateCarree()

    def plot_latlon(a, z):
        z = np.where(z > 0, z, np.nan)
        im = a.pcolormesh(lon, lat, z, transform=src_crs, vmax=60, cmap="gist_ncar")
        a.coastlines()
        a.add_feature(cartopy.feature.STATES, edgecolor="0.5")
        return im

    image_root = os.path.join(dest, "images")

    os.makedirs(image_root, exist_ok=True)
    n = target.shape[0]
    for i in range(n):
        image_path = os.path.join(image_root, f"{i:06}.png")
        if os.path.exists(image_path):
            continue

        fig, (a, b) = plt.subplots(
            1, 2, figsize=(12, 6), subplot_kw=dict(projection=projection)
        )
        im = plot_latlon(a, prediction[i])
        plot_latlon(b, target[i])
        a.set_title('StormCast')
        b.set_title('HRRR Analysis State')
        time = initial_time + i * datetime.timedelta(hours=1)
        fig.suptitle(f"Valid Time: {time.isoformat()} \n Tag: {output_name}")
        cb = plt.colorbar(im, ax=[a, b], orientation="horizontal", shrink=0.8)
        cb.set_label("Composite Reflectivity dBZ")
        plt.savefig(image_path)
        plt.close(fig)

    if output_format == "gif":
        from PIL import Image

        images = []
        for i in range(n):
            image_path = os.path.join(image_root, f"{i:06}.png")
            images.append(Image.open(image_path))
        images[0].save(
            os.path.join(dest, f"{output_name}.gif"),
            save_all=True,
            append_images=images[1:],
            duration=200,
            loop=0,
        )
        
    else:
        os.system(
            f"ffmpeg -y -framerate 5 -i {image_root}/%06d.png -c:v libx264 -pix_fmt yuv420p {dest}/out.gif"
        )

