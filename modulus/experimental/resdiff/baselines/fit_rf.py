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

#%%
import generate
import netCDF4
import joblib
import numpy as np
import torch
import typer
import sys
from training.YParams import YParams
from training.dataset import get_zarr_dataset
import tqdm
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

datatype, config_name, output_pkl = sys.argv[1:]
params = YParams(datatype, config_name)
dataset = get_zarr_dataset(params, train=True)

def dataset_to_xy(dataset, n, samples_per_index):
    inds = list(range(len(dataset)))
    random.shuffle(inds)

    Xs = []
    ys = []
    print("Loading data")
    for i in tqdm.tqdm(range(n)):
        y, x, _ = dataset[inds[i]]

        xx = x.numpy().reshape([16, -1]).T
        yy = y.numpy().reshape([4, -1]).T

        N = xx.shape[0]

        samples = np.random.choice(N, samples_per_index)
        xx = xx[samples]
        yy = yy[samples]

        Xs.append(xx)
        ys.append(yy)

    X = np.concatenate(Xs)
    y = np.concatenate(ys)

    return X, y

X, y = dataset_to_xy(dataset, n=200, samples_per_index=200)

base_estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=16)
multi_output_model = MultiOutputRegressor(base_estimator)

print(f"Fitting {base_estimator}")
multi_output_model.fit(X, y)
print(f"Saving to {output_pkl}")
joblib.dump(multi_output_model, output_pkl)
