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

import numpy as np
from scipy.linalg import sqrtm


def calculate_fid_from_inception_stats(
    mu: np.ndarray, sigma: np.ndarray, mu_ref: np.ndarray, sigma_ref: np.ndarray
) -> float:
    """
    Calculate the Fréchet Inception Distance (FID) between two sets
    of Inception statistics.

    The Fréchet Inception Distance is a measure of the similarity between two datasets
    based on their Inception features (mu and sigma). It is commonly used to evaluate
    the quality of generated images in generative models.

    Parameters
    ----------
    mu:  np.ndarray:
        Mean of Inception statistics for the generated dataset.
    sigma: np.ndarray:
        Covariance matrix of Inception statistics for the generated dataset.
    mu_ref: np.ndarray
        Mean of Inception statistics for the reference dataset.
    sigma_ref: np.ndarray
        Covariance matrix of Inception statistics for the reference dataset.

    Returns
    -------
    float
        The Fréchet Inception Distance (FID) between the two datasets.
    """
    m = np.square(mu - mu_ref).sum()
    s, _ = sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))
