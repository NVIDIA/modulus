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

# %%
import random
import sys

import joblib
import numpy as np
import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from training.dataset import get_zarr_dataset
from training.YParams import YParams

datatype, config_name, output_pkl = sys.argv[1:]
params = YParams(datatype, config_name)
dataset = get_zarr_dataset(params, train=True)


def dataset_to_xy(dataset, n, samples_per_index):
    """
    Converts a dataset into X and y arrays for machine learning models.

    Randomly selects 'n' samples from the dataset and extracts corresponding X and y
    data, sampling 'samples_per_index' points from each selected sample.

    Args:
        dataset: The dataset to process.
        n: Number of samples to select from the dataset.
        samples_per_index: Number of points to sample from each selected sample.

    Returns:
        X: Concatenated features from the selected samples.
        y: Concatenated labels corresponding to the features.
    """
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
