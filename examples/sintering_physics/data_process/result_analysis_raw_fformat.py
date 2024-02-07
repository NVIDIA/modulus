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
This is the code to plot the error of Forward-projection model version , v.s. the VFGN prediction
directly from the /out folders
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from utils import read_raw_folder, get_data_position


# read solution files from solver-only
solution_list_solver = read_raw_folder(physics_solver_path)
# read solution files from VFGN
solution_list_vfgn = read_raw_folder(vfgn_prediction_path)

pos_list, pos_p_list, pos_max_list = [], [], []
pos_list_axis = []

pos_list_vfgn, pos_p_list_vfgn, pos_max_list_vfgn = [], [], []
pos_list_axis_vfgn = []

n = len(solution_list_solver)
# set random voxel id for visualization
pid = 50000
sol_index = []
for i in range(0, len(solution_list_solver), 3):
    print("process solution ", os.path.basename(solution_list_solver[i]))
    solution_data = pv.read(solution_list_solver[i])
    pos_array, _ = get_data_position(solution_data)
    pos_list.append(np.mean(pos_array))
    pos_list_axis.append(np.mean(pos_array, axis=0))
    pos_p_list.append(pos_array[pid])
    pos_max_list.append(np.max(pos_array))

    solution_data_vfgn = pv.read(solution_list_vfgn[i])
    pos_array_vfgn, _ = get_data_position(solution_data_vfgn)
    pos_list_vfgn.append(np.mean(pos_array_vfgn))
    pos_p_list_vfgn.append(pos_array_vfgn[pid])
    pos_max_list_vfgn.append(np.max(pos_array_vfgn))
    pos_list_axis_vfgn.append(np.mean(pos_array_vfgn, axis=0))

    sol_index.append(i)


fig, ax = plt.subplots()

ax.plot(sol_index, pos_list_axis, "b-", linewidth=1, label='solver mean')
ax.plot(sol_index, pos_list_axis_vfgn, "y-", linewidth=1, label='vfgn')
ax.set_ylabel(" mean deformation (mm)", color="blue", fontsize=14)

fig_name = 'check_consistency_axis'
# ax.set_title('p'+str(pid), fontsize=14)
ax.legend(loc="upper left")
fig.savefig(fig_name+'.jpg',
            format='png',
            dpi=100,
            bbox_inches='tight')
plt.close()

