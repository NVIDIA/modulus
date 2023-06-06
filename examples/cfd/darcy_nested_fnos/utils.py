# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os.path
import mlflow
import warp as wp
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Dict
from torch import FloatTensor, Tensor
from torch.nn import MSELoss
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger
from modulus.datapipes.benchmarks.darcy import Darcy2D
from modulus.datapipes.benchmarks.kernels.initialization import init_uniform_random_4d
from modulus.datapipes.benchmarks.kernels.utils import (
    fourier_to_array_batched_2d,
    threshold_3d,
    bilinear_upsample_batched_2d,
)


class NestedDarcyDataset:
    """Nested Darcy Dataset

    A Dataset class for loading nested Darcy data generated with generate_nested_darcy.py
    during training. The method takes care of loading the correct level and associated
    information from its parent level.

    Parameters
    ----------
    data_path : str
        Path to numpy dict file containing the data
    level : int, optional
        Refinement level which shall be loaded
    norm : Dict, optional
        mean and standard deviation for each channel to normalise input and target
    log : PythonLogger
        logger for command line output

    """

    def __init__(
        self,
        mode: str,
        data_path: str = None,
        level: int = None,
        norm: dict = {"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        log: PythonLogger = None,
        parent_prediction: FloatTensor = None,
    ) -> None:
        self.dist = DistributedManager()
        self.data_path = os.path.abspath(data_path)
        self.level = level
        self.norm = norm
        self.log = log
        self.mode = mode
        assert self.mode in [
            "train",
            "eval",
        ], "mode in NestedDarcyDataset must be train or eval."
        self.load_dataset(parent_prediction)

    def load_dataset(self, parent_prediction: FloatTensor = None) -> None:
        try:
            dat = np.load(self.data_path, allow_pickle=True)
        except IOError as err:
            self.log.error(f"Unable to find or load file {self.data_path}")
            exit()

        # load input varibales, copy to device and normalise
        self.invars = dat.item()[f"permeability_{self.level}"]
        self.invars = torch.from_numpy(self.invars).float().to(self.dist.device)
        self.invars = (self.invars - self.norm["permeability"][0]) / self.norm[
            "permeability"
        ][1]

        # load target, copy to device and normalise
        self.outvars = dat.item()[f"darcy_{self.level}"]
        self.outvars = torch.from_numpy(self.outvars).float().to(self.dist.device)
        self.outvars = (self.outvars - self.norm["darcy"][0]) / self.norm["darcy"][1]

        self.length = self.invars.shape[0]
        xy_size = self.invars.shape[-1]
        norm = self.norm

        # get parent info for refined regions
        if self.level > 0:
            # during training, read parent data from file and normalise, for use result from parent
            if self.mode == "train":
                coarse_full = dat.item()[f"darcy_{self.level-1}"]
                coarse_full = torch.from_numpy(coarse_full).float().to(self.dist.device)
                coarse_full = (coarse_full - self.norm["darcy"][0]) / self.norm[
                    "darcy"
                ][1]
            elif self.mode == "eval":
                assert (
                    parent_prediction is not None
                ), f"pass parent result to evaluate level {level}"
                coarse_full = parent_prediction.float()

            pos = dat.item()[f"position"]  # smallest index of x,y in inset
            self.position = pos
            coarse_dat = torch.zeros(
                (self.length, 1, xy_size, xy_size),
                dtype=torch.float,
                device=self.dist.device,
            )

            for ii in range(self.length):
                coarse_dat[ii, 0, ...] = coarse_full[
                    ii,
                    0,
                    pos[ii, 0] : pos[ii, 0] + xy_size,
                    pos[ii, 1] : pos[ii, 1] + xy_size,
                ]
            self.invars = torch.cat([coarse_dat, self.invars], axis=1)

    def __getitem__(self, idx: int):
        return {"permeability": self.invars[idx, ...], "darcy": self.outvars[idx, ...]}

    def __len__(self):
        return self.length


class GridValidator:
    """Grid Validator

    The validator compares model output and target, inverts normalisation and plots a sample

    Parameters
    ----------
    loss_fun : MSELoss
        loss function for assessing validation error
    norm : Dict, optional
        mean and standard deviation for each channel to normalise input and target
    out_dir : str, optional
        directory to which plots shall be stored
    font_size : float, optional
        font size used in figures

    """

    def __init__(
        self,
        loss_fun: MSELoss,
        norm: dict = {"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        out_dir: str = "./outputs/validators",
        font_size: float = 28.0,
    ) -> None:
        self.norm = norm
        self.criterion = loss_fun
        self.font_size = font_size
        self.headers = ("invar", "truth", "prediction", "relative error")
        self.out_dir = os.path.abspath(os.path.join(os.getcwd(), out_dir))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def compare(
        self,
        invar: FloatTensor,
        target: FloatTensor,
        prediction: FloatTensor,
        step: int,
    ) -> float:
        """compares model output, target and plots everything

        Parameters
        ----------
        invar : FloatTensor
            input to model
        target : FloatTensor
            ground truth
        prediction : FloatTensor
            model output
        step : int
            iteration counter

        Returns
        -------
        float
            validation error
        """
        loss = self.criterion(prediction, target)
        norm = self.norm

        # pick first sample from batch
        invar = invar * norm["permeability"][1] + norm["permeability"][0]
        target = target * norm["darcy"][1] + norm["darcy"][0]
        prediction = prediction * norm["darcy"][1] + norm["darcy"][0]
        invar = invar.cpu().numpy()[0, -1, :, :]
        target = target.cpu().numpy()[0, 0, :, :]
        prediction = prediction.detach().cpu().numpy()[0, 0, :, :]

        plt.close("all")
        plt.rcParams.update({"font.size": self.font_size})
        fig, ax = plt.subplots(1, 4, figsize=(15 * 4, 15), sharey=True)
        im = []
        im.append(ax[0].imshow(invar))
        im.append(ax[1].imshow(target))
        im.append(ax[2].imshow(prediction))
        im.append(ax[3].imshow((prediction - target) / norm["darcy"][1]))

        for ii in range(len(im)):
            fig.colorbar(im[ii], ax=ax[ii], location="bottom", fraction=0.046, pad=0.04)
            ax[ii].set_title(self.headers[ii])

        mlflow.log_figure(fig, f"val_step_{step}.png")
        fig.savefig(os.path.join(self.out_dir, f"validation_step_{step}.png"))

        return loss


@wp.kernel
def fourier_to_array_batched_2d_cropped(
    array: wp.array3d(dtype=float),
    fourier: wp.array4d(dtype=float),
    nr_freq: int,
    lx: int,
    ly: int,
    bounds: wp.array2d(dtype=int),
):  # pragma: no cover
    """Array of Fourier amplitudes to batched 2d spatial array

    Parameters
    ----------
    array : wp.array3d
        Spatial array
    fourier : wp.array4d
        Array of Fourier amplitudes
    nr_freq : int
        Number of frequencies in Fourier array
    lx : int
        Grid size x
    ly : int
        Grid size y
    x_start : int
        lowest x-index
    y_start : int
        lowest y-index
    """
    b, x, y = wp.tid()

    x += bounds[b, 0]
    y += bounds[b, 1]

    array[b, x, y] = 0.0
    dx = 6.28318 / wp.float32(lx)
    dy = 6.28318 / wp.float32(ly)
    rx = dx * wp.float32(x)
    ry = dy * wp.float32(y)
    for i in range(nr_freq):
        for j in range(nr_freq):
            ri = wp.float32(i)
            rj = wp.float32(j)
            ss = fourier[0, b, i, j] * wp.sin(ri * rx) * wp.sin(rj * ry)
            cs = fourier[1, b, i, j] * wp.cos(ri * rx) * wp.sin(rj * ry)
            sc = fourier[2, b, i, j] * wp.sin(ri * rx) * wp.cos(rj * ry)
            cc = fourier[3, b, i, j] * wp.cos(ri * rx) * wp.cos(rj * ry)
            wp.atomic_add(
                array, b, x, y, 1.0 / (wp.float32(nr_freq) ** 2.0) * (ss + cs + sc + cc)
            )


class DarcyInset2D(Darcy2D):
    """2D Darcy flow benchmark problem datapipe.

    This datapipe continuously generates solutions to the 2D Darcy equation with variable
    permeability. All samples are generated on the fly and is meant to be a benchmark
    problem for testing data driven models. Permeability is drawn from a random Fourier
    series and threshold it to give a piecewise constant function. The solution is obtained
    using a GPU enabled multi-grid Jacobi iterative method.

    Parameters
    ----------
    resolution : int, optional
        Resolution to run simulation at, by default 256
    batch_size : int, optional
        Batch size of simulations, by default 64
    nr_permeability_freq : int, optional
        Number of frequencies to use for generating random permeability. Higher values
        will give higher freq permeability fields., by default 5
    max_permeability : float, optional
        Max permeability, by default 2.0
    min_permeability : float, optional
        Min permeability, by default 0.5
    max_iterations : int, optional
        Maximum iterations to use for each multi-grid, by default 30000
    convergence_threshold : float, optional
        Solver L-Infinity convergence threshold, by default 1e-6
    iterations_per_convergence_check : int, optional
        Number of Jacobi iterations to run before checking convergence, by default 1000
    nr_multigrids : int, optional
        Number of multi-grid levels, by default 4
    normaliser : Union[Dict[str, Tuple[float, float]], None], optional
        Dictionary with keys `permeability` and `darcy`. The values for these keys are two floats corresponding to mean and std `(mean, std)`.
    device : Union[str, torch.device], optional
        Device for datapipe to run place data on, by default "cuda"

    Raises
    ------
    ValueError
        Incompatable multi-grid and resolution settings
    """

    def __init__(
        self,
        resolution: int = 256,
        batch_size: int = 64,
        nr_permeability_freq: int = 5,
        max_permeability: float = 2.0,
        min_permeability: float = 0.5,
        max_iterations: int = 30000,
        convergence_threshold: float = 1e-6,
        iterations_per_convergence_check: int = 1000,
        nr_multigrids: int = 4,
        normaliser: Union[Dict[str, Tuple[float, float]], None] = None,
        device: Union[str, torch.device] = "cuda",
        fine_res: int = 32,
        fine_permeability_freq: int = 10,
        min_offset: int = 48,
        ref_fac: int = None,
    ):
        super().__init__(
            resolution,
            batch_size,
            nr_permeability_freq,
            max_permeability,
            min_permeability,
            max_iterations,
            convergence_threshold,
            iterations_per_convergence_check,
            nr_multigrids,
            normaliser,
            device,
        )

        self.fine_res = fine_res
        self.fine_freq = fine_permeability_freq
        self.ref_fac = ref_fac
        assert (
            resolution % self.ref_fac == 0
        ), "simulation res must be multiple of ref_fac"

        # force inset on coarse grid
        if not min_offset % self.ref_fac == 0:
            min_offset += self.ref_fac - min_offset % self.ref_fac
        self.beg_min = min_offset
        self.beg_max = resolution - min_offset - fine_res - self.ref_fac
        self.bounds = None
        # self.mask = wp.zeros(self.dim, dtype=bool, device=self.device)

        assert (self.beg_max - self.beg_min) % ref_fac == 0, "lsdhfgn3x!!!!"

    def initialize_batch(self) -> None:
        """Initializes arrays for new batch of simulations"""

        # initialize permeability
        self.permeability.zero_()
        seed = np.random.randint(np.iinfo(np.uint64).max, dtype=np.uint64)
        wp.launch(
            kernel=init_uniform_random_4d,
            dim=self.fourier_dim,
            inputs=[self.rand_fourier, -1.0, 1.0, seed],
            device=self.device,
        )
        wp.launch(
            kernel=fourier_to_array_batched_2d,
            dim=self.dim,
            inputs=[
                self.permeability,
                self.rand_fourier,
                self.nr_permeability_freq,
                self.resolution,
                self.resolution,
            ],
            device=self.device,
        )

        rr = np.random.randint(
            low=0,
            high=(self.beg_max - self.beg_min) // self.ref_fac,
            size=(self.batch_size, 2),
        )
        # print(rr.min(), rr.max(), self.beg_max, self.beg_min, (self.beg_max-self.beg_min)//self.ref_fac)
        self.bounds = wp.array(
            (rr * self.ref_fac) + self.beg_min, dtype=int, device=self.device
        )

        wp.launch(
            kernel=fourier_to_array_batched_2d_cropped,
            dim=(self.batch_size, self.fine_res, self.fine_res),
            inputs=[
                self.permeability,
                self.rand_fourier,
                self.fine_freq,
                self.fine_res,
                self.fine_res,
                self.bounds,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=threshold_3d,
            dim=self.dim,
            inputs=[
                self.permeability,
                0.0,
                self.min_permeability,
                self.max_permeability,
            ],
            device=self.device,
        )

        # zero darcy arrays
        self.darcy0.zero_()
        self.darcy1.zero_()

    def batch_generator(self) -> Tuple[Tensor, Tensor]:
        # run simulation
        self.generate_batch()

        # convert warp arrays to pytorch
        permeability = wp.to_torch(self.permeability)
        darcy = wp.to_torch(self.darcy0)

        # add channel dims
        permeability = torch.unsqueeze(permeability, axis=1)
        darcy = torch.unsqueeze(darcy, axis=1)

        # crop edges by 1 from multi-grid TODO messy
        permeability = permeability[:, :, : self.resolution, : self.resolution]
        darcy = darcy[:, :, : self.resolution, : self.resolution]

        # normalize values
        if self.normaliser is not None:
            permeability = (
                permeability - self.normaliser["permeability"][0]
            ) / self.normaliser["permeability"][1]
            darcy = (darcy - self.normaliser["darcy"][0]) / self.normaliser["darcy"][1]

        # CUDA graphs static copies
        if self.output_k is None:
            self.output_k = permeability
            self.output_p = darcy
        else:
            self.output_k.data.copy_(permeability)
            self.output_p.data.copy_(darcy)

        return {"permeability": self.output_k, "darcy": self.output_p}

    def __iter__(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Yields
        ------
        Iterator[Tuple[Tensor, Tensor, Tensor]]
            Infinite iterator that returns a batch of (permeability, darcy pressure)
            fields of size [batch, resolution, resolution]
        """
        # infinite generator
        while True:
            batch = self.batch_generator()
            batch["inset_pos"] = wp.to_torch(self.bounds)
            yield batch
