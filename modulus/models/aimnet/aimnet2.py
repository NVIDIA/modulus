# ignore_header_test
# ruff: noqa: E402,S101

""""""
"""
AIMNet model. This code was modified from,
https://github.com/isayevlab/aimnetcentral

The following license is provided from their source,

MIT License

Copyright (c) 2024, Roman Zubatyuk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from modulus.models.aimnet import nbops, ops
from modulus.models.aimnet.base import AIMNet2Base
from modulus.models.aimnet.modules import AEVSV, MLP, ConvSV, Embedding
from modulus.models.meta import ModelMetaData


@dataclass
class MetaData(ModelMetaData):
    name: str = "AIMNet2"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class AIMNet2(AIMNet2Base):
    """
    Atoms-In-Molecules Neural Network.

    Reference
    ----------
    Dylan Anstine,Roman Zubatyuk, Olexandr Isayev, 2023
    AIMNet2: A Neural Network Potential to Meet your Neutral, Charged,
    Organic, and Elemental-Organic Needs.

    https://chemrxiv.org/engage/chemrxiv/article-details/6525b39e8bab5d2055123f75

    Parameters
    ----------
    aev: Dict
        Atomic Environment Vector (AEV) parameters.
    nfeature: int
        Number of features.
    d2features: bool
        Whether to use 2D features.
    ncomb_v: int
        Number of combinations in vector quantities.
    hidden: Tuple[List[int]]
        Hidden layers for the network.
    aim_size: int
        Size of the output.
    outputs: Union[List[nn.Module], Dict[str, nn.Module]]
        Output modules.
    num_charge_channels: int, optional
        Number of charge channels (1 for closed-shell, 2 for open-shell). Defaults to 1.

    Raises
    ------
    ValueError
        If num_charge_channels is not 1 or 2.
    TypeError
        If outputs is not a list or a dict.

    Notes
    ----
    Equivalent to the original implementation from:
    https://github.com/isayevlab/aimnetcentral
    """

    def __init__(
        self,
        aev: Dict,
        nfeature: int,
        d2features: bool,
        ncomb_v: int,
        hidden: Tuple[List[int]],
        aim_size: int,
        outputs: Union[List[nn.Module], Dict[str, nn.Module]],
        num_charge_channels: int = 1,
    ):
        super().__init__(meta=MetaData())

        if num_charge_channels not in [1, 2]:
            raise ValueError(
                "num_charge_channels must be 1 (closed shell) or 2 (NSE for open-shell)."
            )
        self.num_charge_channels = num_charge_channels

        self.aev = AEVSV(**aev)
        nshifts_s = aev["nshifts_s"]
        nshifts_v = aev.get("nshitfs_v") or nshifts_s
        if d2features:
            if nshifts_s != nshifts_v:
                raise ValueError("nshifts_s must be equal to nshifts_v for d2features")
            nfeature_tot = nshifts_s * nfeature
        else:
            nfeature_tot = nfeature
        self.nfeature = nfeature
        self.nshifts_s = nshifts_s
        self.d2features = d2features

        self.afv = Embedding(num_embeddings=64, embedding_dim=nfeature, padding_idx=0)

        with torch.no_grad():
            nn.init.orthogonal_(self.afv.weight[1:])
            if d2features:
                self.afv.weight = nn.Parameter(
                    self.afv.weight.clone()
                    .unsqueeze(-1)
                    .expand(64, nfeature, nshifts_s)
                    .flatten(-2, -1)
                )

        conv_param = {
            "nshifts_s": nshifts_s,
            "nshifts_v": nshifts_v,
            "ncomb_v": ncomb_v,
            "do_vector": True,
        }
        self.conv_a = ConvSV(nchannel=nfeature, d2features=d2features, **conv_param)
        self.conv_q = ConvSV(
            nchannel=num_charge_channels, d2features=False, **conv_param
        )

        mlp_param = {"activation_fn": nn.GELU(), "last_linear": True}
        mlps = [
            MLP(
                n_in=self.conv_a.output_size() + nfeature_tot,
                n_out=nfeature_tot + 2 * num_charge_channels,
                hidden=hidden[0],
                **mlp_param,
            )
        ]
        mlp_param = {"activation_fn": nn.GELU(), "last_linear": False}
        for h in hidden[1:-1]:
            mlps.append(  # noqa: PERF401
                MLP(
                    n_in=self.conv_a.output_size()
                    + self.conv_q.output_size()
                    + nfeature_tot
                    + num_charge_channels,
                    n_out=nfeature_tot + 2 * num_charge_channels,
                    hidden=h,
                    **mlp_param,
                )
            )
        mlp_param = {"activation_fn": nn.GELU(), "last_linear": False}
        mlps.append(
            MLP(
                n_in=self.conv_a.output_size()
                + self.conv_q.output_size()
                + nfeature_tot
                + num_charge_channels,
                n_out=aim_size,
                hidden=hidden[-1],
                **mlp_param,
            )
        )
        self.mlps = nn.ModuleList(mlps)

        if isinstance(outputs, Sequence):
            self.outputs = nn.ModuleList(outputs)
        elif isinstance(outputs, Mapping):
            self.outputs = nn.ModuleDict(outputs)
        else:
            raise TypeError("`outputs` is not either list or dict")

    def _preprocess_spin_polarized_charge(
        self, data: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        if "mult" not in data:
            raise ValueError(
                "mult key is required for NSE if two channels for charge are not provided"
            )
        _half_spin = 0.5 * (data["mult"] - 1.0)
        _half_q = 0.5 * data["charge"]
        data["charge"] = torch.stack(
            [_half_q + _half_spin, _half_q - _half_spin], dim=-1
        )
        return data

    def _postprocess_spin_polarized_charge(
        self, data: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        data["spin_charges"] = data["charges"][..., 0] - data["charges"][..., 1]
        data["charges"] = data["charges"].sum(dim=-1)
        data["charge"] = data["charge"].sum(dim=-1)
        return data

    def _prepare_in_a(self, data: Dict[str, Tensor]) -> Tensor:
        a_i, a_j = nbops.get_ij(data["a"], data)
        avf_a = self.conv_a(a_j, data["gs"], data["gv"])
        if self.d2features:
            a_i = a_i.flatten(-2, -1)
        _in = torch.cat([a_i.squeeze(-2), avf_a], dim=-1)
        return _in

    def _prepare_in_q(self, data: Dict[str, Tensor]) -> Tensor:
        q_i, q_j = nbops.get_ij(data["charges"], data)
        avf_q = self.conv_q(q_j, data["gs"], data["gv"])
        _in = torch.cat([q_i.squeeze(-2), avf_q], dim=-1)
        return _in

    def _update_q(
        self, data: Dict[str, Tensor], x: Tensor, delta_q: bool = True
    ) -> Dict[str, Tensor]:
        _q, _f, delta_a = x.split(
            [
                self.num_charge_channels,
                self.num_charge_channels,
                x.shape[-1] - 2 * self.num_charge_channels,
            ],
            dim=-1,
        )
        # for loss
        data["_delta_Q"] = data["charge"] - nbops.mol_sum(_q, data)
        q = data["charges"] + _q if delta_q else _q
        f = _f.pow(2)
        q = ops.nse(data["charge"], q, f, data, epsilon=1.0e-6)
        data["charges"] = q
        data["a"] = data["a"] + delta_a.view_as(data["a"])
        return data

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data = self.prepare_input(data)

        # initial features
        a: Tensor = self.afv(data["numbers"])
        if self.d2features:
            a = a.unflatten(-1, (self.nfeature, self.nshifts_s))
        data["a"] = a

        # NSE case
        if self.num_charge_channels == 2:
            data = self._preprocess_spin_polarized_charge(data)
        else:
            # make sure that charge has channel dimension
            data["charge"] = data["charge"].unsqueeze(-1)

        # AEV
        data = self.aev(data)

        # MP iterations
        _npass = len(self.mlps)
        for ipass, mlp in enumerate(self.mlps):
            if ipass == 0:
                _in = self._prepare_in_a(data)
            else:
                _in = torch.cat(
                    [self._prepare_in_a(data), self._prepare_in_q(data)], dim=-1
                )

            _out = mlp(_in)
            if data["_input_padded"].item():
                _out = nbops.mask_i_(_out, data, mask_value=0.0)

            if ipass == 0:
                data = self._update_q(data, _out, delta_q=False)
            elif ipass < _npass - 1:
                data = self._update_q(data, _out, delta_q=True)
            else:
                data["aim"] = _out

        # squeeze charges
        if self.num_charge_channels == 2:
            data = self._postprocess_spin_polarized_charge(data)
        else:
            data["charges"] = data["charges"].squeeze(-1)
            data["charge"] = data["charge"].squeeze(-1)

        # readout
        for m in self.outputs.children():
            data = m(data)

        return data
