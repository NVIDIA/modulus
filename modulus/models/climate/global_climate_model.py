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

import math
from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn

import modulus  # noqa: F401 for docs
from modulus.models.layers import get_activation
from modulus.models.meta import ModelMetaData
from modulus.models.module import Module

Tensor = torch.Tensor


class GlobalClimateModel(Module):
    """
    Global Climate Model. This model is a wrapper around a regridder, normalizer, and model.
    It performs regridding and normalization on the input, runs the model, and then denormalizes
    and regrids the output. 

    Parameters
    ----------
    regridder : modulus.models.Module
        Regridder module. This model should have a `forward` method that takes in a tensor of
        the expected input grid and returns a tensor of the expected output grid.
    normalizer : modulus.models.Module
        Normalizer module. This model should have a `forward` method that takes in a tensor of
        the expected input grid and returns a tensor of the expected output grid.
    model : modulus.models.Module
        Model module. This model should have a `forward` method that takes in a tensor of
        the expected input grid and returns a tensor of the expected output grid.
    static_variables : Tuple[str]
        Strings of variable names ('land_mask', 'land_fraction', etc..)
    static_dynamic_variables : Tuple[str]
        Strings of variable names ('total_incoming_solar_radiation', 'total_outgoing_longwave_radiation', etc..)
    dynamic_variables : Tuple[str]
        Strings of variable names ('air_temperature', 'specific_humidity', etc..)
    expected_input_grid : str
        Expected input grid. Defaults to "latlon_721x1440".
        Options are: "latlon_721x1440", "latlon_512x256", "gaussian_grid_xyz"
    """

    def __init__(
        self,
        regridder: modulus.models.Module,
        normalizer: modulus.models.Module,
        model: modulus.models.Module,
        static_variables: Tuple[str], # strings of variable names ('land_mask', 'land_fraction', etc..)
        static_dynamic_variables: Tuple[str],
        dynamic_variables: Tuple[str], # strings of variable names ('air_temperature', 'specific_humidity', etc..)
        expected_input_grid: str = "latlon_721x1440",
    ):
        super().__init__(meta=model.meta)

        # Store the regridder, normalizer, and model
        self.regridder = regridder
        self.normalizer = normalizer
        self.model = model

        # Store variable mappings
        self.static_variables = static_variables
        self.static_dynamic_variables = static_dynamic_variables
        self.dynamic_variables = dynamic_variables
        self.expected_input_grid = expected_input_grid

    def forward(
            self,
            input: Tensor,
            normalize_output: bool = True,
            regrid_output: bool = True,
        ) -> Tensor:

        # Perform regridding
        regridded_input = self.regridder(input, direction="forward")

        # Normalize the input
        normalized_input = self.normalizer(regridded_input, direction="forward")

        # Run the model
        output = self.model(normalized_input)

        # Denormalize the output
        if normalize_output:
            output = self.normalizer(output, direction="inverse")

        # Perform regridding
        if regrid_output:
            output = self.regridder(output, direction="inverse")

        return output
