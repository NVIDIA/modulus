# adapted from https://github.com/RuoyanLi2002/FluidDiffusion/blob/main/data/t2p.py

import os
import json
import pickle
import functools
import tensorflow as tf
# import reading_utils

# ls_folder = ["Sand-3D", "SandRamps", "Water-3D", "WaterDrop", "WaterDrop-XL", "WaterRamps"]
path = "/media/wumming/HHD/HHD_data/graph/"

ls_folder = ["WaterRamps"]
ls_type = ["test", "train", "valid"]
part = 0
count = 0

for folder in ls_folder:
    print(folder)
    with open(path + f'/data/{folder}/metadata.json', 'r') as file:
        metadata = json.load(file)

    for type in ls_type:
        data = []

        ds = tf.data.TFRecordDataset(
            [os.path.join(path + f"/data/{folder}", f'{type}.tfrecord')])
        ds = ds.map(functools.partial(
            reading_utils.parse_serialized_simulation_example, metadata=metadata))

        for element in ds.as_numpy_iterator():
            print(element)
            exit()
            # data.append(element)
            # count += 1
            # print(count)


        with open(path + f"/data/{folder}/{type}.pkl", 'wb') as file:
            pickle.dump(data, file)
