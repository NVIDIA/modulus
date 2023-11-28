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

import pkg_resources
import pytest
from pytest_utils import _import_or_fail


@pytest.mark.parametrize(
    "model_name",
    [
        "AFNO",
        "DLWP",
        "FNO",
        "GraphCastNet",
        "MeshGraphNet",
        "FullyConnected",
        "Pix2Pix",
        "One2ManyRNN",
        #'SphericalFourierNeuralOperatorNet',
        "SRResNet",
    ],
)
def test_model_entry_points(model_name, pytestconfig):
    """Test model entry points"""

    if model_name == "GraphCastNet" or model_name == "MeshGraphNet":
        _import_or_fail("dgl", pytestconfig)

    # Get all the models exposed by the package
    models = {
        entry_point.name: entry_point
        for entry_point in pkg_resources.iter_entry_points("modulus.models")
    }

    # Assert that the model is among them
    assert model_name in models

    # Try loading the model
    try:
        models[model_name].load()
    except Exception as e:
        pytest.fail(f"Failed to load {model_name}: {e}")
