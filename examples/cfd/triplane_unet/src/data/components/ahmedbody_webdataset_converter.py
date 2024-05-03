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

from typing import Dict, Optional, Literal, List

import random
import glob
import sys
from pathlib import Path
import numpy as np
import tarfile
from multiprocessing import Pool

from torch.multiprocessing import set_start_method

from src.data.ahmedbody_datamodule import AhmedBodyDataset


def compute_dataset_statistics(dataset) -> Dict[str, float]:
    """
    Compute the mean, min, max of p and velocity in the dataset
    """
    # compute the mean, min, max of data["p"], data["case_infp"]["Velocity"]
    # Use the running sum, running count method
    count_p = 0
    running_sum_p = 0
    running_square_sum_p = 0
    running_min_p = np.inf
    running_max_p = -np.inf
    count_vel = 0
    running_sum_vel = 0
    running_min_vel = np.inf
    running_max_vel = -np.inf

    print("Computing mean, min, max of p and vel")
    for i in range(len(dataset)):
        if i % 100 == 0:
            print(f"Processing {i}/{len(dataset)}")
        data = dataset[i]
        point_data = data["point_data"]
        p = point_data["p"]
        vel = data["case_info"]["velocity"]
        running_sum_p += p.sum()
        running_square_sum_p += (p**2).sum()
        running_min_p = min(running_min_p, p.min())
        running_max_p = max(running_max_p, p.max())
        count_p += len(p)

        running_sum_vel += vel
        running_min_vel = min(running_min_vel, vel)
        running_max_vel = max(running_max_vel, vel)
        count_vel += 1

    # Save the mean, min, max of p and vel into a file
    mean_p = running_sum_p / count_p
    std_p = np.sqrt((running_square_sum_p / count_p) - mean_p**2)
    mean_vel = running_sum_vel / count_vel
    # Create a return dictionary
    return {
        "mean_p": mean_p,
        "min_p": running_min_p,
        "max_p": running_max_p,
        "std_p": std_p,
        "mean_vel": mean_vel,
        "min_vel": running_min_vel,
        "max_vel": running_max_vel,
    }


class AhmedBodyToWebDataset:
    """
    Convert AhmedBodyDataset to a webdataset
    """

    def __init__(self, dataset: AhmedBodyDataset, output_path: Path):
        self.dataset = dataset
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _save_item(self, idx: int):
        try:
            item = self.dataset[idx]
        except:
            print(f"Failed to load item {idx}")
            return

        # Save to 0 padded index
        print(f"Saving item {idx}/{len(self.dataset)}")
        np.savez_compressed(
            self.output_path / f"case_{item['case_id']:03d}.npz",
            **item,
        )

    def save(self, num_processes: int = 1):
        # Save each item in as a numpy file in parallel
        assert (
            num_processes > 0
        ), f"num_processes should be greater than 0, got {num_processes}"
        if num_processes < 2:
            for idx in range(len(self.dataset)):
                print(f"Saving item {idx}/{len(self.dataset)}")
                self._save_item(idx)
        elif num_processes > 1:
            with Pool(num_processes) as p:
                p.map(self._save_item, range(len(self.dataset)))

        # Compress the dataset to a tar file
        print("Compressing the dataset to a tar file")
        self.output_path = self.output_path.expanduser()
        self.output_path.mkdir(exist_ok=True)

    def to_tars(self):
        case_files = glob.glob(str(self.output_path / "*.npz"))

        case_files.sort()
        # len(case_files) == 423

        def to_tar(out_tar_file: str, files: List[str]):
            print(f"Creating tar file {out_tar_file}")
            tar_path = self.output_path / out_tar_file
            with tarfile.open(tar_path, "w") as tar:
                for file in files:
                    tar.add(file, arcname=Path(file).name)

        # split into train/val/test with 0.7, 0.1, 0.2
        # random permute case_files
        random.seed(42)

        # Shuffle the list randomly
        random.shuffle(case_files)

        # Compute split indices for 70% train, 10% validation, 20% test
        n = len(case_files)
        idx_train = int(n * 0.7)
        idx_val = idx_train + int(n * 0.1)

        # Split files into train, val, and test
        train_files = case_files[:idx_train]
        val_files = case_files[idx_train:idx_val]
        test_files = case_files[idx_val:]

        # Create tar files for each dataset
        to_tar("train.tar", train_files)
        to_tar("val.tar", val_files)
        to_tar("test.tar", test_files)


def convert_to_webdataset(
    data_path: str,
    out_path: str,
    num_processes: Optional[int] = 8,
):
    set_start_method("spawn", force=True)

    # Create separate paths for train/text, with and without spoiler
    dataset = AhmedBodyDataset(data_path)
    output_path = Path(out_path).expanduser()
    output_path.mkdir(exist_ok=True)
    # Save to numpy
    converter = AhmedBodyToWebDataset(
        dataset,
        output_path,
    )
    converter.save(num_processes=num_processes)
    converter.to_tars()


if __name__ == "__main__":
    import fire

    fire.Fire(convert_to_webdataset)
