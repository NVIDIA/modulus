from typing import List
import zarr
import matplotlib.pyplot as plt
import datetime
import cartopy.crs
import cartopy.feature
import os
import pathlib
from scipy.fft import dctn
import numpy as np


def dct_power(y, d=3, bins=100, axes=(-2, -1)):
    """

    Args:
        y: the signal to compute the powe spectra of
        d: the sampling rate (grid box)
        bins: the number of bins to average the 2d power spectra over

    Returns:
        f, pw: the total frequency (1/2d) being the smallest and power spectra
            f = sqrt(f1^2 + f2^2)
    """
    # see https://journals.ametsoc.org/view/journals/mwre/130/7/1520-0493_2002_130_1812_sdotda_2.0.co_2.xml
    # IDCT is same as DCT-3 which is x_n cos(pi n (2 k + 1) / 2N) for k = 0, ..., N-1
    # x_k =  (2 k + 1) / 2N varies from 0 to 1 (not inclusive)
    # D = d N is the domain size
    # so n = 1 corresponds to a wavelength of 2 D
    # n = 2 is a wavelength of D
    # n = 3 (0, 3 pi) is a wavelength of 2 D /3
    # so wavelength of n is  2 D / n
    d = 3  # km
    fy = dctn(y, axes=axes)

    N = y.shape[0]
    D = d * N
    f1 = np.arange(N) / 2 / D

    N = y.shape[1]
    D = d * N
    f2 = np.arange(N) / 2 / D
    f_total = np.sqrt(f1[:, None] ** 2 + f2[None, :] ** 2)

    f_total_grid = np.linspace(0, f_total.max(), bins)
    binned_power, edges = np.histogram(
        f_total.ravel(), bins=f_total_grid, weights=(fy**2).ravel()
    )
    f_total = (edges[1:] + edges[:-1]) / 2
    return f_total, binned_power


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


def compute_decorrelation_scale(
    fields: List[np.ndarray],
    bins: int = 200,
    threshold: float = 0.95,
) -> np.ndarray:
    """Compute the decorrelation scale as a function of lead time

    This is the scale below which the fields are decorrelated

    Args:
        *fields: arrays with shape=(time, y, x)
        bins: the number of bins to bin the total wave number over
        threshold: the threshold above the fields are considered decorrelated

    Returns:
        decorrelation_scale: shape=(time,) units = km.

    Note:

        Define in this paper:

        Surcel, M., Zawadzki, I., & Yau, M. K. (2015). A Study on the Scale
        Dependence of the Predictability of Precipitation Patterns. Journal of the
        Atmospheric Sciences, 72(1), 216–235.
        https://doi.org/10.1175/JAS-D-14-0071.1


    """
    lams = []
    nt = fields[0].shape[0]
    # TODO should be possible to vectorize the binning
    for i in range(nt):
        f, R = power_ratio([a[i] for a in fields], bins=bins)
        i0 = np.searchsorted(R, threshold)
        i0 = min([i0, R.size - 1])
        lam0 = 1 / f[i0]
        lams.append(lam0)
    return lams


def power_ratio(fields: List[np.ndarray], bins: int):
    """Power ratio

    R(lambda) = Σ(P_xi(lambda)) / P_xT(lambda)

    """
    A = sum([dct_power(a, bins=bins)[1] for a in fields])
    total = sum([a for a in fields])
    f, B = dct_power(total, bins=bins)
    return f, A / B


def save_decorrelation_scale(path, output_path):
    g = zarr.open(path, mode="r")
    target = g["target"]["refc"]
    prediction = g["edm_prediction"]["refc"]
    lam = compute_decorrelation_scale([target, prediction])
    np.savetxt(os.path.join(output_path, "decorrelation_scale.txt"), lam)

    fig, ax = plt.subplots()
    ax.plot(lam)
    ax.set_xlabel("Lead Time (hours)")
    ax.set_ylabel("km")
    ax.grid()
    fig.savefig(os.path.join(output_path, "decorrelation_scale.png"))
    plt.close(fig)

    f, R = power_ratio([target[1], prediction[1]], bins=100)
    fig, ax = plt.subplots()
    ax.semilogx(1 / f, R)
    ax.set_xlabel("Wavelength (km)")
    ax.set_ylabel("Power ratio")
    ax.set_title("t = 1 hour.")
    ax.grid()
    fig.savefig(os.path.join(output_path, "power_ratio.png"))
    plt.close(fig)


if __name__ == "__main__":
    import sys

    root = sys.argv[1]
    for dir in os.listdir(root):
        try:
            time = datetime.datetime.fromisoformat(dir)
        except ValueError:
            pass
        else:
            try:
                zarr_path = os.path.join(root, dir, "data.zarr")
                make_movie(zarr_path, os.path.join(root, dir))
                save_decorrelation_scale(zarr_path, os.path.join(root, dir))
            except:
                pass
