# © Copyright 2023 HP Development Company, L.P.
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
File to visualize and check the acceleration profile
"""

import numpy as np
import matplotlib.pyplot as plt
import glob, re, os
import pyvista as pv
from absl import app
from absl import flags

from utils import read_raw_folder, get_solution_id, time_diff, get_data_position

from constants import Constants

C = Constants()
def main():
    solution_list = read_raw_folder(C.raw_data_dir)

    pos_list, pos_p_list, pos_max_list = [], [], []
    pos_list_axis = []
    step = 100
    for i in range(0, len(solution_list), step):
        # Read the raw solution file with step size
        solution_data = pv.read(solution_list[i])
        # pos_array: np array stores displacement,
        # dim: (num_nodes, 3)
        pos_array, _ = get_data_position(solution_data)
        pos_list.append(pos_array)

        pos_list_axis.append(np.mean(pos_array, axis=0))
        pos_max_list.append(np.max(pos_array))

    # Compute the velovity, acceleration for each node
    velocity_array = time_diff(np.array(pos_list))
    acceleration_array = time_diff(velocity_array)

    acc_3d_mean = np.mean(acceleration_array, axis=1)

    # Visualize
    build_name = os.path.basename(C.raw_data_dir)
    fig, ax = plt.subplots()
    sol_index = [i for i in range(acc_3d_mean.shape[0])]
    ax.plot(sol_index, acc_3d_mean[:,0], "b-", linewidth=1, label='x-dim velocity')
    ax.plot(sol_index, acc_3d_mean[:,1], "y-", linewidth=1, label='y-dim velocity')
    ax.plot(sol_index, acc_3d_mean[:,2], "g-", linewidth=1, label='z-dim velocity')
    ax.set_xlabel("time steps", color="blue", fontsize=14)
    ax.set_ylabel("acce", color="blue", fontsize=14)
    # ax.set_ylim(0, 3e-6)

    ax.legend(loc="lower right")
    fig_name = 'acc_3d_'+build_name+'_step'+str(step)
    fig.savefig(fig_name+'.jpg',
                format='png',
                dpi=100,
                bbox_inches='tight')


if __name__ == "__main__":
    app.run(main)