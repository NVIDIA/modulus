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

import shutil
from typing import Callable

import boto3
import pytest
import torch
import torch.nn as nn
from moto import mock_aws
from pytest_utils import import_or_fail

from modulus.distributed import DistributedManager
from modulus.models.mlp import FullyConnected

# @pytest.fixture(scope="function")
# def aws_credentials():
#     """Mocked AWS Credentials for moto."""
#     os.environ["AWS_ACCESS_KEY_ID"] = "testing"
#     os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
#     os.environ["AWS_SECURITY_TOKEN"] = "testing"
#     os.environ["AWS_SESSION_TOKEN"] = "testing"
#     os.environ["AWS_DEFAULT_REGION"] = "us-west-1"
#
#
# @pytest.fixture(scope="function")
# def s3(aws_credentials):
#     """
#     Return a mocked S3 client
#     """
#     with mock_aws():
#         yield boto3.client("s3", region_name="us-west-1")


# "./checkpoints",
# ["./checkpoints", "msc://checkpoint-test/checkpoints"]
@pytest.fixture(params=["./checkpoints", "msc://checkpoint-test/checkpoints"])
def checkpoint_folder(request) -> str:
    return request.param


# , "pytorch"
@pytest.fixture(params=["modulus"])
def model_generator(request) -> Callable:
    # Create fully-connected NN generator function
    if request.param == "modulus":

        def model(x):
            return FullyConnected(
                in_features=x,
                out_features=x,
                num_layers=2,
                layer_size=8,
            )

    else:

        def model(x):
            return nn.Sequential(
                nn.Linear(x, 8),
                nn.ReLU(),
                nn.Linear(8, x),
            )

    return model


@mock_aws
@import_or_fail(["wandb", "mlflow"])
@pytest.mark.parametrize("device", [("cpu")])
def test_model_checkpointing(
    device,
    model_generator,
    checkpoint_folder,
    pytestconfig,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    """Test checkpointing util for model"""

    import os

    os.environ["AWS_ACCESS_KEY_ID"] = "access-key-id"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "secret-access-key"
    # os.environ["AWS_SECURITY_TOKEN"] = "testing"
    # os.environ["AWS_SESSION_TOKEN"] = "testing"
    # os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["MSC_CONFIG"] = "./msc_config.yaml"
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket="checkpoint-test-bucket")

    from modulus.launch.utils import load_checkpoint, save_checkpoint

    # Initialize DistributedManager first since save_checkpoint instantiates it
    DistributedManager.initialize()

    mlp_model_1 = model_generator(8).to(device)
    mlp_model_2 = model_generator(4).to(device)

    input_1 = torch.randn(4, 8).to(device)
    input_2 = torch.randn(4, 4).to(device)

    output_1 = mlp_model_1(input_1)
    output_2 = mlp_model_2(input_2)
    # Save model weights to checkpoint
    print("About to save")
    save_checkpoint(
        checkpoint_folder,
        models=[mlp_model_1, mlp_model_2],
        metadata={"model_type": "MLP"},
    )

    print("saved")

    import fsspec

    fs = fsspec.filesystem("msc")
    files = fs.glob("msc://checkpoint-test/checkpoints/*")
    print(f"files: {files}")

    # Load twin set of models for importing weights
    mlp_model_1 = model_generator(8).to(device)
    mlp_model_2 = model_generator(4).to(device)

    new_output_1 = mlp_model_1(input_1)
    new_output_2 = mlp_model_2(input_2)
    # Assert models are now different
    assert not torch.allclose(output_1, new_output_1, rtol, atol)
    assert not torch.allclose(output_2, new_output_2, rtol, atol)

    print("About to load 1")
    # Load model weights from checkpoint
    load_checkpoint(checkpoint_folder, models=[mlp_model_1, mlp_model_2], device=device)

    print("Finished loading 2")
    loaded_output_1 = mlp_model_1(input_1)
    loaded_output_2 = mlp_model_2(input_2)

    print(f"output_1: {output_1}")
    print(f"new_output_1: {new_output_1}")
    print(f"loaded_output_1: {loaded_output_1}")
    print()
    print(f"output_2: {output_2}")
    print(f"new_output_2: {new_output_2}")
    print(f"loaded_output_2: {loaded_output_2}")

    assert torch.allclose(output_1, loaded_output_1, rtol, atol)
    assert torch.allclose(output_2, loaded_output_2, rtol, atol)

    print("About to load")
    # Also load the model with metadata
    metadata_dict = {}
    epoch = load_checkpoint(
        checkpoint_folder,
        models=[mlp_model_1, mlp_model_2],
        metadata_dict=metadata_dict,
        device=device,
    )
    print("Loaded")

    assert epoch == 0
    assert metadata_dict["model_type"] == "MLP"

    # Clean up if writing to local file system - object storage files will disappear along with the mock.
    if fsspec.utils.get_protocol(checkpoint_folder) == "file":
        shutil.rmtree(checkpoint_folder)
    else:
        # For non-file systems, the local cache must be cleared to allow multiple test runs
        local_cache = os.environ["HOME"] + "/.cache/modulus"
        shutil.rmtree(local_cache)
