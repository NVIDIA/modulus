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

import logging
import pytest
import torch
import torch.nn as nn

try:
    import onnxruntime as ort
except:
    ort = None

from pathlib import Path
from modulus.models.mlp import FullyConnected
from modulus.deploy.onnx import export_to_onnx_stream, run_onnx_inference

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


# TODO(akamenev): remove once the bug below is fixed.
# Version "1.14.0" is the custom local build where the bug is fixed.
def check_ort_version():
    if ort is None:
        return pytest.mark.skipif(
            True,
            reason="Proper ONNX runtime is not installed. 'pip install onnxruntime onnxruntime_gpu'",
        )
    elif ort.__version__ != "1.14.0":
        return pytest.mark.skipif(
            True,
            reason="Must install custom ORT 1.14.0. Other versions do not work \
        due to bug in IRFFT: https://github.com/microsoft/onnxruntime/issues/13236",
        )
    else:
        return pytest.mark.skipif(False, reason="")


@pytest.fixture(params=["modulus", "pytorch"])
def model(request) -> str:
    # Create fully-connected NN to test exporting
    if request.param == "modulus":
        # Modulus version with meta data
        model = FullyConnected(
            in_features=32,
            out_features=8,
            num_layers=1,
            layer_size=8,
        )
    else:
        # PyTorch version
        model = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
        )
    return model


@check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_onnx_bytestream(device, model, rtol: float = 1e-3, atol: float = 1e-3):
    """Test Modulus' export onnx stream function is consistent with file saving"""

    model = model.to(device)
    bsize = 8
    invar = torch.randn(bsize, 32).to(device)
    outvar = model(invar)

    onnx_name = "model.onnx"
    # Run ONNX using standard export to file approach
    model = model.eval().cpu()
    onnx_in_args = invar.cpu()
    torch.onnx.export(
        model.cpu(),
        onnx_in_args,
        onnx_name,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        opset_version=15,
        verbose=False,
    )
    outvar_ort_file = run_onnx_inference(onnx_name, invar, device=device)
    assert len(outvar_ort_file) == 1
    outvar_ort_file = torch.Tensor(outvar_ort_file[0]).to(device)
    # Run ONNX using built in stream util in Modulus
    onnx_stream = export_to_onnx_stream(model, invar, verbose=False)
    outvar_ort = run_onnx_inference(onnx_stream, invar, device=device)
    assert len(outvar_ort) == 1
    outvar_ort = torch.Tensor(outvar_ort[0]).to(device)

    # Delete onnx model file
    Path(onnx_name).unlink(missing_ok=False)

    assert torch.allclose(outvar, outvar_ort_file, rtol, atol)
    assert torch.allclose(outvar, outvar_ort, rtol, atol)
