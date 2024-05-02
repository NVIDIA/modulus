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


import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticWeatherDataLoader(DataLoader):
    """
    This custom DataLoader initializes the SyntheticWeatherDataset with given arguments.
    """

    def __init__(self, *args, **kwargs):
        dataset = SyntheticWeatherDataset(*args, **kwargs)
        super().__init__(
            dataset=dataset,
            batch_size=kwargs.get("batch_size", 1),
            shuffle=kwargs.get("shuffle", False),
            num_workers=kwargs.get("num_workers", 0),
            pin_memory=kwargs.get("pin_memory", False),
            drop_last=kwargs.get("drop_last", False),
        )


class SyntheticWeatherDataset(Dataset):
    """
    A dataset for generating synthetic temperature data on a latitude-longitude grid for multiple atmospheric layers.

    Args:
        channels (list): List of channels representing different atmospheric layers.
        num_samples_per_year (int): Total number of days to simulate per year.
        num_steps (int): Number of consecutive days in each training sample.
        grid_size (tuple): Latitude by longitude dimensions of the temperature grid.
        base_temp (float): Base temperature around which variations are simulated.
        amplitude (float): Amplitude of the sinusoidal temperature variation.
        noise_level (float): Standard deviation of the noise added to temperature data.
        **kwargs: Additional keyword arguments for advanced configurations.
    """

    def __init__(
        self,
        channels: List[int],
        num_samples_per_year: int,
        num_steps: int,
        device: str | torch.device = "cuda",
        grid_size: Tuple[int, int] = (721, 1440),
        base_temp: float = 15,
        amplitude: float = 10,
        noise_level: float = 2,
        **kwargs: Any,
    ):
        self.num_days: int = num_samples_per_year
        self.num_steps: int = num_steps
        self.num_channels: int = len(channels)
        self.device = device
        self.grid_size: Tuple[int, int] = grid_size
        start_time = time.time()
        self.temperatures: np.ndarray = self.generate_data(
            self.num_days,
            self.num_channels,
            self.grid_size,
            base_temp,
            amplitude,
            noise_level,
        )
        print(
            f"Generated synthetic temperature data in {time.time() - start_time:.2f} seconds."
        )
        self.extra_args: Dict[str, Any] = kwargs

    def generate_data(
        self,
        num_days: int,
        num_channels: int,
        grid_size: Tuple[int, int],
        base_temp: float,
        amplitude: float,
        noise_level: float,
    ) -> np.ndarray:
        """
        Generates synthetic temperature data over a specified number of days for multiple atmospheric layers.

        Args:
            num_days (int): Number of days to generate data for.
            num_channels (int): Number of channels representing different layers.
            grid_size (tuple): Grid size (latitude, longitude).
            base_temp (float): Base mean temperature for the data.
            amplitude (float): Amplitude of temperature variations.
            noise_level (float): Noise level to add stochasticity to the temperature.

        Returns:
            numpy.ndarray: A 4D array of temperature values across days, channels, latitudes, and longitudes.
        """
        days: np.ndarray = np.arange(num_days)
        latitudes, longitudes = grid_size
        daily_temps: np.ndarray = np.zeros(
            (num_days, num_channels, latitudes, longitudes)
        )

        for day in days:
            for channel in range(num_channels):
                altitude_effect: float = (
                    channel * -0.5
                )  # Temperature decreases with altitude
                lat_variation: np.ndarray = np.linspace(
                    -amplitude, amplitude, latitudes
                )
                daily_temps[day, channel] = (
                    base_temp
                    + altitude_effect
                    + np.outer(lat_variation, np.sin(2 * np.pi * day / 365))
                    + np.random.normal(scale=noise_level, size=(latitudes, longitudes))
                )
        return daily_temps

    def __len__(self) -> int:
        """
        Returns the number of samples available in the dataset.
        """
        return self.num_days - self.num_steps

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a sample from the dataset at the specified index.
        """
        return [
            {
                "invar": torch.tensor(self.temperatures[idx], dtype=torch.float32).to(
                    self.device
                ),
                "outvar": torch.tensor(
                    self.temperatures[idx + 1 : idx + self.num_steps + 1],
                    dtype=torch.float32,
                ).to(self.device),
            }
        ]
