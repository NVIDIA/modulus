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
Include some basic functions for simulation data processings.
Convert simulation output displacement-*.pvtu files to tfRecord,
Store for model training

usage:
python rawdata2tfrecord.py data-root meta-file-root
i.e.
python rawdata2tfrecord.py "/home/rachel_chen/dataset/Virtual-Foundry" "./"
python rawdata2tfrecord.py "/home/lopezca/repos/sintervox-models/dl-models"
"""

import sys, os, glob
import csv
import pyvista as pv
import numpy as np
import json
import tensorflow as tf
from sklearn import neighbors

import utils
from utils import get_solution_id, time_diff

from constants import Constants

C = Constants()
# Create a description of the features.
_FEATURE_DESCRIPTION = {
    'position': tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(
    tf.string)

_FEATURE_DTYPES = {
    'position': {
        'in': np.float64,
        'out': tf.float64
    },
    'step_context': {
        'in': np.float64,
        'out': tf.float64
    }
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string),
    'senders': tf.io.VarLenFeature(tf.string),
    'receivers': tf.io.VarLenFeature(tf.string),
    }


def arrange_data(data):
    arranged_data = {}

    name2array = {}
    for array_name in data.array_names:
        name2array[array_name] = data[array_name]

    # Get all points in the solution file
    points = data.points

    # Number of points in one solution file
    n_points = points.shape[0]
    for point_index in range(n_points):
        point = data.GetPoint(point_index)
        if point not in arranged_data:
            data_item = {}
            for array_name in name2array:
                data_item[array_name] = name2array[array_name][point_index]
            arranged_data[point] = data_item
    return arranged_data


def get_data_position(data):
    """
    For the data read from one displacement-id.pvtu file,
    iterate each point data, filter out the points in existed physical xyz-location
    store the non-repeating point's uvw_values in
    Args:
        data: data read from displacement-id.pvtu file

    Returns: array of non-repeating nodes' current physical location (original location + displacement)

    """
    # Construct a dictionary, store physical location {xyz: boolean}
    arranged_data = {}

    points = data.points
    n_points = points.shape[0]

    # uvw_values = data['displacement_U']
    uvw_values = data['u__v__w']
    print("get_data_position read uvw_values: ", uvw_values.shape)

    pos_list = []
    index_list = []

    for point_index in range(n_points):
        point = data.GetPoint(point_index)
        # point_array=points[point_index]
        if point not in arranged_data:
            uvw = uvw_values[point_index]

            # Compute the deformed physical location from original physical location
            pos = point + uvw

            index_list.append(point_index)

            pos_list.append(pos)

            arranged_data[point] = True

    return np.array(pos_list),index_list

def read_sol_time(series_file):
    """
    Read the solution.pvtu.series file
    Returns:

    """
    dict_sol_time = {}
    time_list = []
    with open(series_file, 'r') as fobj:
        data = json.load(fobj)

    for idx, item in enumerate(data['files']):
        time_list.append(item['time'])
        dict_sol_time[item['name']] = [item['time']]

    return dict_sol_time


def read_temperature(temp_file):
    """
    Open and read the temperature profile from params.prm file under each build data.
    Save Temp for each time solution file.
    Returns:
        list of temperature value, corresponding to each solution file (sorted with time)
    """
    # reading csv file
    print("read temperature file from path: ", temp_file)
    with open(temp_file, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting each data row one by one
        for idx, row in enumerate(csvreader):
            # print(row)
            if row[0].find('temperature_curve') != -1:
                print("find sintering_temperature_curve: ", row[0], row[0].find('temperature_curve'))
                temp_row = row[1:]
                break

    # Add temperature data to each solution file
    temp_row.insert(0, "0")
    return temp_row


def get_solution_temperature_customer(solution_path, dict_sol_time, temp_curve_list):
    """
    This function read each solution file, determine the time, Temperature of this file
    """
    t_list = []
    temp_list = []
    for idx in range(len(temp_curve_list) // 2):
        t_list.append(int(temp_curve_list[idx*2]) / 3600)
        temp_list.append(int(temp_curve_list[idx*2+1]))

    sol_name = os.path.basename(solution_path)
    sol_time = int(dict_sol_time[sol_name][0])

    # search range
    sol_time = sol_time / 3600
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    nearest_id, nearest_value = find_nearest(t_list, sol_time)

    # determine left / right / ==
    if nearest_value == sol_time:
        return temp_list[nearest_id]
    elif nearest_value < sol_time:
        start_temp, end_temp = temp_list[nearest_id], temp_list[nearest_id+1]
        start_time, end_time = nearest_value, t_list[nearest_id+1]
    else:
        start_temp, end_temp = temp_list[nearest_id-1], temp_list[nearest_id]
        start_time, end_time = t_list[nearest_id-1], nearest_value

    temp = ((end_temp - start_temp) / (end_time - start_time)) * (sol_time-start_time) + start_temp

    print("solname | soltime | temp: ", sol_name, sol_time, temp)
    return temp


def get_solution_temperature(solution_path, dict_sol_time, temp_curve_list):
    """
    This temperature point compute only works for 1st-version temp curve
    :param solution_path:
    :param dict_sol_time:
    :param temp_curve_list:
    :return:
    """
    sol_name = os.path.basename(solution_path)
    sol_time = int(dict_sol_time[sol_name][0])

    start_time, start_temp = int(temp_curve_list[0]), int(temp_curve_list[1])
    equil_time, equil_temp = int(temp_curve_list[2]), int(temp_curve_list[3])

    if sol_time >= equil_time:
        return equil_temp
    else:
        temp = ((equil_temp - start_temp) / (equil_time - start_time)) * sol_time + start_temp
    return temp


def get_solution_time(solution_path, dict_sol_time):
    sol_name = os.path.basename(solution_path)
    sol_time = int(dict_sol_time[sol_name][0])
    return sol_time

def _compute_connectivity(positions, radius):
    """Get the indices of connected edges with radius connectivity.

    Args:
      positions: Positions of nodes in the graph. Shape:
        [num_nodes_in_graph, num_dims].
      radius: Radius of connectivity.

    Returns:
      senders indices [num_edges_in_graph]
      receiver indices [num_edges_in_graph]

    """
    tree = neighbors.KDTree(positions)
    receivers_list = tree.query_radius(positions, r=radius)

    num_nodes = len(positions)
    senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
    receivers = np.concatenate(receivers_list, axis=0)
    return senders, receivers


def get_anchor_point(ds):
    """
    pvtu_file_name:
    with the given build files, return the index of point, that is the anchor point of the build
    Returns:
        anchor point index: int
    """
    # Read the Digital sintering software raw data, field version name: "u_v_w"
    ds1=ds['u__v__w']
    ## return index of the anchor point
    ds1_ax = np.where(ds1[:, 0] == 0)[0]
    ds1_ay = np.where(ds1[:, 1] == 0)[0]
    ds1_az = np.where(ds1[:, 2] == 0)[0]

    listx_as_set = set(ds1_ax)
    intersection = listx_as_set.intersection(ds1_ay)
    anchor_pset = intersection.intersection(ds1_az)

    assert len(anchor_pset) == 1, print(f"Find anchor point number {len(anchor_pset)}!")
    return list(anchor_pset)[0]


def get_anchor_zplane(ds):
    """
    with the given build files, return the index of point that are on the anchor z-plane of the build
    Returns:
        anchor plane points index: List[int]
    """
    print("find the anchor plane")
    n_points=ds.points.shape[0]
    uvw_values = ds['u__v__w']
    print("ds.points: ", ds.points.shape)

    z_plane = []
    for ip in range(n_points):
        point = ds.GetPoint(ip)
        if point[2] == 0:
            # for all the points fall on the z-plane
            z_plane.append(ip)
            uvw = uvw_values[ip]

            # set non-0 threshold
            threshold_z = 0.1
            assert uvw[2]<threshold_z, "wrong anchor point: {} {}".format(ip,uvw[2])

    z_plane = set(z_plane)
    print("cnt of z anchor plane points:", len(z_plane))
    return z_plane

def read_solutions_data_temp_range(raw_data_path=None, init_idx=0, metadata=None):
    build_path = os.path.join(raw_data_path, 'out')
    solution_list = glob.glob(build_path + '/solution-*.pvtu')
    solution_list = sorted(solution_list, key=get_solution_id)
    assert len(solution_list) >= 3, "Need to have at least 3 solution files as input to start prediction or analysis!"

    # For each build, read the displacement.pvtu.series file
    series_file = os.path.join(raw_data_path, 'out', "solution.pvtu.series")
    # print("Find and read series file: ", series_file)
    assert os.path.exists(series_file), "volume-deformation.pvtu.series not exists!"
    dict_sol_time = read_sol_time(series_file)

    # For each build, read the temperature profile at every time step
    temp_profile_path = os.path.join(raw_data_path, "params.prm")
    assert os.path.exists(temp_profile_path), "Temperature profile file params.prm not exists!"
    temp_curve_list = read_temperature(temp_profile_path)

    # Get the stage separation time-temperature pairs
    stage_t_list = []
    stage_temp_list = []
    for idx in range(len(temp_curve_list) // 2):
        stage_t_list.append(int(temp_curve_list[idx*2]))
        stage_temp_list.append(int(temp_curve_list[idx*2+1]))

    # Record stage ids
    # For example, for an entire sintering duration of 3393 simulation steps, can choose the data process window
    # can either be the entire sintering duration, for process window for detailed accuracy
    # for the default sintering profile, there are 3393 simulation steps, with
    #   stage 1: temperature increase window [0, 596],
    #   stage 2: temperature stable window [596,1321]
    #   stage 3: T increase: [> 1325]
    #   if to consider the transition stage separately: [1200, 1700]

    # set the start and end file index for data processing
    start_index, end_index = 0, 3393

    particles_list = []
    key_list = []

    radius = 0
    anchors = []
    anchor_point = None
    temp_list = []

    # index for the step 100 model, 2 stages
    step = 100

    solution_data_end = pv.read(solution_list[-1])
    solution_data_begin = pv.read(solution_list[0])

    radius_begin = utils.get_radius(solution_data_begin)
    print("get simulation radius: ", radius_begin) # i.e. radius: 1.2

    pos_array_begin, index_list = get_data_position(solution_data_begin)
    senders_graph_i, receivers_graph_i = _compute_connectivity(pos_array_begin, radius_begin)
    print("number of edge: ", senders_graph_i.shape[0])

    if C.ADD_ANCHOR:
        zplane_anchors = get_anchor_zplane(solution_data_end)
        anchor_0 = get_anchor_point(solution_data_end)
        # print("\n\nzplane_anchors: ", zplane_anchors)
        print("anchor_0: ", anchor_0)

        n_index = len(index_list)
        for pi in range(n_index):
            if index_list[pi] in zplane_anchors:
                anchors.append(pi)
            if index_list[pi] == anchor_0:
                anchor_point = pi
                print("anchor point: ",pi)

        print("anchors: ", anchors, len(anchors))

    solution_step_list=solution_list[init_idx::step]
    print("\nprocess with start idx: ", init_idx)
    print("solution list len: ", len(solution_step_list))
    for solution_path in solution_step_list:
        # For each displacement-*.pvtu file, get the displacement-id, read data points
        # filter out the repeated data
        time_ = int(dict_sol_time[os.path.basename(solution_path)][0])

        solution_id = get_solution_id(solution_path)
        if start_index <= solution_id <= end_index:
            solution_temp = get_solution_temperature_customer(solution_path, dict_sol_time, temp_curve_list)
            print(f"process {solution_path}, time: {time_}")
            print(f"solution time: {time_}, solution_temp: {solution_temp}")

            solution_data = pv.read(solution_path)

            pos_array, index_list = get_data_position(solution_data)
            print("data type: ", pos_array.dtype)

            if radius == 0:
                radius = utils.get_radius(solution_data)
                senders_graph_i, receivers_graph_i = _compute_connectivity(pos_array, radius)
                print("number of edge: ", senders_graph_i.shape[0])

            particles_list.append(pos_array)
            key_list.append(solution_id)
            temp_list.append(solution_temp)

    # ensure all sequences have same length
    if init_idx != 0 and len(particles_list) != metadata['sequence_length']:
        skip = True
    else:
        skip = False

    return key_list, particles_list, temp_list, senders_graph_i, receivers_graph_i, \
           radius, anchors, anchor_point, skip


def compute_metadata_stats(metadata, particles_list_builds, velocity_list_builds,
                           acceleration_list_builds, radius_list, temp_list_builds):
    # Compute position mean, std
    # todo: check why use different norm dimension
    # todo: change the pos stats to 3d as well
    position_stats_array=np.concatenate(particles_list_builds)
    position_std=position_stats_array.std()
    metadata['pos_mean'] = position_stats_array.mean()
    metadata['pos_std'] = position_stats_array.std()

    # Compute velocity mean, std
    velocity_stats_array=np.concatenate(velocity_list_builds)
    velocity_stats_array = velocity_stats_array/position_std
    metadata['vel_mean'] = [i for i in velocity_stats_array.mean(axis=0).tolist()]
    metadata['vel_std'] = [i for i in velocity_stats_array.std(axis=0).tolist()]

    # Compute acceleration mean, std
    acceleration_stats_array = np.concatenate(acceleration_list_builds)
    acceleration_stats_array = acceleration_stats_array/position_std
    metadata['acc_mean'] = [i for i in acceleration_stats_array.mean(axis=0).tolist()]
    metadata['acc_std'] = [i for i in acceleration_stats_array.std(axis=0).tolist()]

    # Compute radius mean, std
    radius_array = np.array(radius_list)/position_std
    metadata['default_connectivity_radius'] = radius_array.mean()

    # Compute temperature mean, std
    metadata['context_mean'] = [np.array(temp_list_builds).mean()]
    metadata['context_std'] = [np.array(temp_list_builds).std()]
    print(np.array(temp_list_builds).shape)
    if np.array(temp_list_builds).ndim >1:
        metadata['context_feat_len'] = np.array(temp_list_builds).shape[1]
    else:
        metadata['context_feat_len'] = 1

    return metadata


def write_tfrecord_entry(writer, features, particles_array, times_array):
    tf_sequence_example = tf.train.SequenceExample(context=features)
    position_list = tf_sequence_example.feature_lists.feature_list["position"]
    timestep_list = tf_sequence_example.feature_lists.feature_list["step_context"]

    for i in range(len(particles_array)):
        position_list.feature.add().bytes_list.value.append(particles_array[i].tobytes())
        timestep_list.feature.add().bytes_list.value.append(times_array[i].tobytes())

    writer.write(tf_sequence_example.SerializeToString())


def main(argv):
    """
    Args:
        raw_data_dir:  raw data directory (direct output from the Physics simulation engine)
        metadata_json_path: path of metadata.json (metadata computed from the train builds)
        mode: there are three mode [train, test, stats]
    """
    raw_data_dir, metadata_json_path, mode = argv

    with open(os.path.join(metadata_json_path, "metadata.json"), 'r') as f:
        metadata = json.load(f)
        print("meta: ", metadata)
    if mode != "stats":
        writer = tf.io.TFRecordWriter(os.path.join(metadata_json_path, mode+'.tfrecord'))

    # State the build names
    # i.e. choose from ['busbar', 'USB_casing', 'pushing_grip', 'ExtrusionScrew' or other customize builds]
    build_list = []
    if mode == 'train' or mode == 'stats':
        # sample build list data names
        build_list = []
    elif mode == "test":
        # test data for validation data - NVIDIA
        build_list = ['busbar', 'USB_casing', 'pushing_grip', 'ExtrusionScrew']
    else:
        print("Mode not implemented")
        exit()

    key_i = 0
    n_steps = 0
    temp_list_builds = []
    particles_list_builds = []
    velocity_list_builds = []
    acceleration_list_builds = []

    radius_list = []
    # Read and process each build data set
    for build_name in build_list:
        print("\n process build: ", build_name)
        # Get information for each build, the params 100 as step size,
        # 30 for sampling frequency is for testing purpose
        for init_idx in range(0, 100, 30):
            key_list, particles_list, temp_list, \
            senders_graph_i, receivers_graph_i, \
            radius, anchors, anchor_point, skip = read_solutions_data_temp_range(os.path.join(raw_data_dir, build_name),
                                                                                 init_idx=init_idx, metadata=metadata)

            # Gather information across builds, prep for builds stats calculation
            particles_list_builds += particles_list

            velocity_array = time_diff(np.array(particles_list))
            velocity_list = [velocity_array[i] for i in range(velocity_array.shape[0])]
            velocity_list_builds += velocity_list

            acceleration_array = time_diff(velocity_array)
            acceleration_list = [acceleration_array[i] for i in range(acceleration_array.shape[0])]
            acceleration_list_builds += acceleration_list

            radius_list.append(radius)

            temp_list_builds += temp_list

            # particles_array.shape(num_time_steps, nodes_per_build, xyz-dim) i.e. (12, 1107, 3)
            particles_array = np.array(particles_list)
            n_steps = particles_array.shape[0]
            n_particles = particles_array.shape[1]
            if init_idx == 0:
                metadata['sequence_length'] = n_steps

            # Write to TFRECORD
            if mode == 'train' or mode == 'test' or mode == 'generalization':
                # Reshape: append all nodes in one timestep to same array, i.e. (12, 3321)
                particles_array = particles_array.reshape((n_steps, -1)).astype(np.float64)

                # normalize data
                particles_mean = metadata['pos_mean']
                particles_std = metadata['pos_std']
                particles_array = (particles_array - particles_mean) / particles_std

                # set same particle type for now, can vary the type as an additional feature
                # shape: num_particles, i.e. 1107
                particles_type = np.repeat(2, n_particles)
                # set the anchoring nodes particle type differently
                if C.ADD_ANCHOR:
                    particles_type[anchors] = 1
                    particles_type[anchor_point] = 0
                particles_type = particles_type.tobytes()

                # temperature array shape: (num_steps, ), i.e. (83,)
                temps_array = np.array(temp_list)

                # Add global features
                context_features = {
                    'particle_type': utils._bytes_feature(particles_type),
                    'key': utils.create_int_feature(key_i),
                    'senders': utils._bytes_feature(senders_graph_i.tobytes()),
                    'receivers': utils._bytes_feature(receivers_graph_i.tobytes()),
                }

                key_i += 1
                features = tf.train.Features(feature=context_features)

                for idx_build in range(1):
                    start_idx, end_idx = 0, particles_array.shape[0]
                    print(f"write range: {start_idx}-{end_idx}")
                    write_tfrecord_entry(writer, features, particles_array[start_idx: end_idx],
                                         temps_array[start_idx: end_idx])

                print("Finale feature shape: ", particles_array.shape)

    # Write metadata file
    if mode == 'stats':
        metadata = compute_metadata_stats(metadata, particles_list_builds, velocity_list_builds,
                                          acceleration_list_builds, radius_list, temp_list_builds)
        metadata['sequence_length'] = n_steps - 1
        metadata['dim'] = 3

        with open(os.path.join(metadata_json_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
    elif mode == "test" or mode == "generalization":
        print("Finale feature shape: ", particles_array.shape)
        metadata['sequence_length'] = particles_array.shape[0] - 1
        with open(os.path.join(metadata_json_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f)


if __name__ == '__main__':
    argv = sys.argv[1:]
    main(argv)
