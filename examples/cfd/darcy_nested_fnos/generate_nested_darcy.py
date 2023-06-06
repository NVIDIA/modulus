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

import hydra
import time
from os.path import isdir
from os import mkdir
from utils import DarcyInset2D
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
import torch


def nested_darcy_generator() -> None:
    """Dataset Generator for the nested Darcy Problem

    This script generates the training, validation and out-of-sample data sets
    for the nested FNO problem and stores them in ./data, where trainer and
    inferencer will find it.
    """
    out_dir = "./data/"
    file_names = ["training_data.npy", "validation_data.npy", "out_of_sample.npy"]
    # sample_size = [2048, 256, 128]
    sample_size = [8192, 256, 128]
    max_batch_size = 128
    resolution = 1024
    glob_res = 256
    fine_res = 128
    buffer = 8
    permea_freq = 5
    fine_permeability_freq = 3
    device = "cuda"
    plot = False

    perm_norm = (0.0, 1.0)
    darc_norm = (0.0, 1.0)

    if not isdir(out_dir):
        mkdir(out_dir)

    assert resolution % glob_res == 0, "resolution needs to be multiple of glob_res"
    ref_fac = resolution // glob_res
    inset_size = fine_res + 2 * buffer
    min_offset = (fine_res * (ref_fac - 1) + 1) // 2 + buffer * ref_fac

    # force inset on coarse grid
    if not min_offset % ref_fac == 0:
        min_offset += ref_fac - min_offset % ref_fac

    for dset in range(len(file_names)):
        # compute batch size and number of iterations
        batch_size = min(max_batch_size, sample_size[dset])
        nr_iterations = (sample_size[dset] - 1) // max_batch_size + 1

        datapipe = DarcyInset2D(
            resolution=resolution,
            batch_size=batch_size,
            nr_permeability_freq=permea_freq,
            max_permeability=2.0,
            min_permeability=0.5,
            max_iterations=300,
            convergence_threshold=1e-4,
            iterations_per_convergence_check=10,
            nr_multigrids=3,
            normaliser={"permeability": perm_norm, "darcy": darc_norm},
            device=device,
            fine_res=fine_res,
            fine_permeability_freq=fine_permeability_freq,
            min_offset=min_offset,
            ref_fac=ref_fac,
        )

        perm_std, perm_mean, darc_std, darc_mean = 0.0, 0.0, 0.0, 0.0
        permeability_0, darcy_0, permeability_1, darcy_1, position = [], [], [], [], []
        for jj, sample in zip(range(nr_iterations), datapipe):
            perm_std, perm_mean = torch.std_mean(sample["permeability"])
            darc_std, darc_mean = torch.std_mean(sample["darcy"])

            permea = sample["permeability"].cpu().detach().numpy()
            darcy = sample["darcy"].cpu().detach().numpy()
            pos = (sample["inset_pos"].cpu().detach().numpy()).astype(int)
            assert (pos % ref_fac).sum() == 0, "inset off coarse grid"

            # crop out refined region, allow for sourrounding area, save in extra array
            permea_fine = np.zeros((batch_size, 1, inset_size, inset_size), dtype=float)
            darcy_fine = np.zeros_like(permea_fine)
            for ii in range(batch_size):
                xs = pos[ii, 0] - buffer
                ys = pos[ii, 1] - buffer
                permea_fine[ii, 0, :, :] = permea[
                    ii, 0, xs : xs + inset_size, ys : ys + inset_size
                ]
                darcy_fine[ii, 0, :, :] = darcy[
                    ii, 0, xs : xs + inset_size, ys : ys + inset_size
                ]

            # downsample resolution of global field
            permea_glob = permea[:, :, ::ref_fac, ::ref_fac]
            darcy_glob = darcy[:, :, ::ref_fac, ::ref_fac]

            # save those three arrays to numpy dict, translate pos from simulation grid to coarser parent
            res = np.array([resolution // ref_fac, inset_size])
            pos = (pos - min_offset) // ref_fac

            permeability_0.append(permea_glob)
            darcy_0.append(darcy_glob)
            permeability_1.append(permea_fine)
            darcy_1.append(darcy_fine)
            position.append(pos)

        # concatenate arrays, then store to file
        permeability_0 = np.concatenate(permeability_0, axis=0)[
            : sample_size[dset], ...
        ]
        darcy_0 = np.concatenate(darcy_0, axis=0)[: sample_size[dset], ...]
        permeability_1 = np.concatenate(permeability_1, axis=0)[
            : sample_size[dset], ...
        ]
        darcy_1 = np.concatenate(darcy_1, axis=0)[: sample_size[dset], ...]
        position = np.concatenate(position, axis=0)[: sample_size[dset], ...]

        np.save(
            out_dir + file_names[dset],
            {
                "permeability_0": permeability_0,
                "darcy_0": darcy_0,
                "permeability_1": permeability_1,
                "darcy_1": darcy_1,
                "position": position,
                "resolution": res,
            },
        )

        assert pos.min() >= 0, f"too small min, {pos.min()}, {pos.max()}"
        assert (
            pos.max()
            <= (resolution - 2 * min_offset - fine_res) * glob_res / resolution
        ), f"too large max, {pos.min()}, {pos.max()}"

        print(
            f"    position: min={pos.min()}, max={pos.max()}, "
            + f"upper bound={int((resolution-2*min_offset-fine_res)*(glob_res/resolution))}"
        )
        print(f"permeability: mean={perm_mean}, std={perm_std}")
        print(f"       darcy: mean={darc_mean}, std={darc_std}")

        # plot coef and solution
        if plot:
            for ii in range(20):
                fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(
                    3, 2, figsize=(10, 15)
                )
                ax0.imshow(permea_glob[ii, 0, :, :])
                ax0.set_title("permeability glob")
                ax1.imshow(darcy_glob[ii, 0, :, :])
                ax1.set_title("darcy glob")
                ax2.imshow(permea_fine[ii, 0, :, :])
                ax2.set_title("permeability fine")
                ax3.imshow(darcy_fine[ii, 0, :, :])
                ax3.set_title("darcy fine")
                ax4.imshow(
                    permea_glob[
                        ii,
                        0,
                        pos[ii, 0] : pos[ii, 0] + inset_size,
                        pos[ii, 1] : pos[ii, 1] + inset_size,
                    ]
                )
                ax4.set_title("permeability zoomed")
                ax5.imshow(
                    darcy_glob[
                        ii,
                        0,
                        pos[ii, 0] : pos[ii, 0] + inset_size,
                        pos[ii, 1] : pos[ii, 1] + inset_size,
                    ]
                )
                ax5.set_title("darcy zoomed")
                fig.tight_layout()
                plt.savefig(f"test_{ii}.png")


if __name__ == "__main__":
    nested_darcy_generator()
