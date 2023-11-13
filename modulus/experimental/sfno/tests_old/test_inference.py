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

import glob
import pathlib
import subprocess
import hashlib

import pytest


def test_inference(tmp_path: pathlib.Path, regtest):
    model_weights = pathlib.Path("/runs/baseline_afno_v1/model_weights/26var_0.tar")

    if not model_weights.exists():
        pytest.skip("Model weights not found.")

    cmd = [
        "python",
        "inference/inference.py",
        "--yaml_config=config/AFNO2.yaml",
        "--config=afno_26_ch_inference",
        "--run_num=0",
        "--vis",
        "--weights",
        model_weights.as_posix(),
        "--override_dir",
        tmp_path.as_posix(),
    ]
    subprocess.check_call(cmd)
    (output_filename,) = tmp_path.glob("*.h5")
    checksum = hashlib.md5(output_filename.read_bytes())
    print(checksum.hexdigest(), file=regtest)
