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

from setuptools import setup

setup(
    entry_points={
        "modulus.models": [
            "AFNO = modulus.models.afno:AFNO",
            "DLWP = modulus.models.dlwp:DLWP",
            "FNO = modulus.models.fno:FNO",
            "GraphCastNet = modulus.models.graphcast:GraphCastNet",
            "MeshGraphNet = modulus.models.meshgraphnet:MeshGraphNet",
            "FullyConnected = modulus.models.mlp:FullyConnected",
            "Pix2Pix = modulus.models.pix2pix:Pix2Pix",
            "One2ManyRNN = modulus.models.rnn:One2ManyRNN",
            "SFNO = modulus.experimental.models.sfno:SFNO",
            "SRResNet = modulus.models.srrn:SRResNet",
        ],
    }
)
