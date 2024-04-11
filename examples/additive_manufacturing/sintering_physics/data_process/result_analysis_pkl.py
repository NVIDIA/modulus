# © Copyright 2023 HP Development Company, L.P.
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


"""
This is the code to plot the error from the rollout.pkl files
"""

import numpy as np
import matplotlib.pyplot as plt
import glob, re, os
import json
from natsort import natsorted
import pickle

from absl import app
from absl import flags


flags.DEFINE_string("rollout_path", None, help="Path to rollout pickle file")
flags.DEFINE_string("rollout_folder_path", None, help="Path to rollout pickle file")
flags.DEFINE_string("meta_path", None, help="Path to metadata file")

FLAGS = flags.FLAGS


def plot_pkl(rollout_path, metadata):
    """Compute the ground-truth acceleration, and prediction acceleration comparison"""
    if not rollout_path:
        raise ValueError("A `rollout_path` must be passed.")
    with open(rollout_path, "rb") as file:
        rollout_data = pickle.load(file)

    pos_mean = metadata["pos_mean"]
    pos_std = metadata["pos_std"]

    # Read data and denormalize
    gt_data = rollout_data["ground_truth_rollout"] * pos_std + pos_mean
    pred_data = rollout_data["predicted_rollout"] * pos_std + pos_mean

    const_acc_mean, const_acc_max = [], []
    diff_dl = []
    sol_index = []
    for step in range(3, gt_data.shape[0]):
        # Compute for ground-truth data
        pos_1, pos_2, pos_3 = gt_data[step - 1], gt_data[step - 2], gt_data[step - 3]
        acc_ = pos_1 - pos_2 - (pos_2 - pos_3)
        vel_ = pos_1 - pos_2
        diff_ = gt_data[step] - (pos_1 + (vel_ + acc_))

        const_acc_mean.append(np.mean(np.abs(diff_)))

        diff_dl_ = pred_data[step] - gt_data[step]
        diff_dl.append(np.mean(np.abs(diff_dl_)))
        sol_index.append(step)

        print(
            f"{step}: const acc mean={np.mean(np.abs(diff_))}, dl_diff = {np.mean(np.abs(diff_dl_))}"
        )

    return const_acc_mean, diff_dl, sol_index


def main(unused_argv):
    """Visualize the ground-truth acceleration, and prediction acceleration comparison"""

    with open(os.path.join(FLAGS.meta_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    const_acc_mean, diff_dl, sol_index = plot_pkl(FLAGS.rollout_path, metadata)

    fig, ax = plt.subplots()

    ax.plot(sol_index, const_acc_mean, "g-", linewidth=1, label="const acce")
    ax.plot(sol_index, diff_dl, "y-", linewidth=1, label="dl pred")
    ax.set_ylabel(" max acc diff(mm)", color="blue", fontsize=14)
    ax.set_ylim(0, 3e-6)
    # add the cut-off lines
    cutoff_line_cycle10 = [3e-7 for i in range(len(sol_index))]
    cutoff_line_cycle100 = [7e-7 for i in range(len(sol_index))]
    plt.plot(sol_index, cutoff_line_cycle10, "r.", markersize=1)
    plt.plot(sol_index, cutoff_line_cycle100, "r.", markersize=1)

    fig_name = "check_acc_max"
    # ax.set_title('p'+str(pid), fontsize=14)
    ax.legend(loc="upper left")
    fig.savefig(fig_name + ".jpg", format="png", dpi=100, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    app.run(main)
