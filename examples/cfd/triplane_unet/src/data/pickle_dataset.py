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

import pickle
from pathlib import Path
from typing import Dict, Optional, Union


class PickleDataset:
    """Dataset wrapper for Python pickle files."""

    def __init__(
        self,
        root_path: Union[str, Path],
        file_format: Optional[str] = None,
        extension: str = "pkl",
    ) -> None:
        if isinstance(root_path, str):
            root_path = Path(root_path)
        self.root_path = root_path
        # Get the number of files in the directory with the specified extension
        if file_format is not None:
            glob_filter = f"{file_format}.{extension}"
        else:
            glob_filter = f"*.{extension}"
        self.pkl_files = list(self.root_path.glob(glob_filter))
        # Sort files
        self.pkl_files.sort()

    def __len__(self) -> int:
        return len(self.pkl_files)

    def __getitem__(self, index) -> Dict:
        # Get the file path
        file_path = self.pkl_files[index]
        # Load pickle file
        data = pickle.load(open(file_path, "rb"))
        return data


if __name__ == "__main__":
    # Test the NumpyDataset on the provided path
    import sys

    path = sys.argv[-1]
    dataset = PickleDataset(path)
    print(len(dataset))
    print(dataset[0])
