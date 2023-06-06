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

import hydra
from torch import cat, std_mean, FloatTensor
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from modulus.models.mlp import FullyConnected
from modulus.models.fno import FNO
from modulus.utils import StaticCaptureEvaluateNoGrad
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger
from modulus.launch.utils import load_checkpoint

from utils import NestedDarcyDataset


def plot_result(idx: int, perm: dict, darcy: dict, pos: dict) -> None:
    """Plot Results

    Takes results from inference and assembles all levels to a single plot.

    Parameters
    ----------
    idx : int
        index of sample which shall be plotted
    perm : dict
        dictionary containing permeability field for each level
    darcy :
        dictionary containing results of inference for each level
    pos : dict
        dictionary containing information about position of inset
    """
    ref_fac = 4  # TODO include in dataset generator so can be read from file
    buffer = 8
    fine_res = 128
    min_offset = (fine_res * (ref_fac - 1) + 1) // 2 + buffer * ref_fac

    headers = ["permeability", "darcy"]
    invar = perm["ref0"][idx, -1, :, :]
    prediction = darcy["ref0"][idx, 0, :, :]

    # add refined region
    loc = pos["ref1"][idx] * ref_fac + min_offset
    pred_ref = darcy["ref1"][idx, -1, :, :]
    expanded = np.ones((1024, 1024), dtype=float)
    mask = np.ones((1024, 1024), dtype=bool)
    expanded[
        loc[0] : loc[0] + pred_ref.shape[-2], loc[1] : loc[1] + pred_ref.shape[-1]
    ] = pred_ref
    mask[
        loc[0] : loc[0] + pred_ref.shape[-2], loc[1] : loc[1] + pred_ref.shape[-1]
    ] = False
    expanded = np.ma.array(expanded, mask=mask)
    vmin, vmax = prediction.min(), prediction.max()

    prediction = np.kron(
        prediction, np.ones((ref_fac, ref_fac), dtype=prediction.dtype)
    )
    invar = np.kron(invar, np.ones((ref_fac, ref_fac), dtype=invar.dtype))

    plt.close("all")
    plt.rcParams.update({"font.size": 28})
    fig, ax = plt.subplots(1, 2, figsize=(15 * 2, 15), sharey=True)
    im = []
    im.append(ax[0].imshow(invar))
    im.append(ax[1].imshow(prediction, cmap="viridis", vmin=vmin, vmax=vmax))
    ax[1].imshow(expanded, cmap="viridis", vmin=vmin, vmax=vmax)

    for ii in range(len(im)):
        fig.colorbar(im[ii], ax=ax[ii], location="bottom", fraction=0.046, pad=0.04)
        ax[ii].set_title(headers[ii])

    fig.savefig(join("./", f"inferred_{idx:03d}.png"))


def EvaluateModel(
    cfg: DictConfig,
    model_name: str,
    norm: dict = {"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
    parent_result: FloatTensor = None,
    log: PythonLogger = None,
):
    dist = DistributedManager()
    # define model
    log.info(f"evaluating model {model_name}")
    model_cfg = cfg.arch[model_name]
    level = int(model_name[-1])
    decoder = FullyConnected(
        in_features=model_cfg.fno.latent_channels,
        out_features=model_cfg.decoder.out_features,
        num_layers=model_cfg.decoder.layers,
        layer_size=model_cfg.decoder.layer_size,
    )
    model = FNO(
        decoder_net=decoder,
        in_channels=model_cfg.fno.in_channels,
        dimension=model_cfg.fno.dimension,
        latent_channels=model_cfg.fno.latent_channels,
        num_fno_layers=model_cfg.fno.fno_layers,
        num_fno_modes=model_cfg.fno.fno_modes,
        padding=model_cfg.fno.padding,
    ).to(dist.device)
    load_checkpoint(
        path=f"./checkpoints/{model_name}", device=dist.device, models=model
    )

    # prepare data for inference
    dataset = NestedDarcyDataset(
        mode="eval",
        data_path=cfg.inference.inference_set,
        level=level,
        norm=norm,
        log=log,
        parent_prediction=parent_result,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.inference.batch_size, shuffle=False)

    # store positions of insets
    if level > 0:
        pos = dataset.position
    else:
        pos = None

    # define forward method
    @StaticCaptureEvaluateNoGrad(
        model=model, logger=log, use_amp=False, use_graphs=False
    )
    def forward_eval(invars):
        return model(invars)

    # evaluate and invert normalisation
    invars, result = [], []
    for batch in dataloader:
        invars.append(batch["permeability"])
        result.append(forward_eval(batch["permeability"]))
    invars = cat(invars, dim=0).detach()
    result = cat(result, dim=0).detach()

    return pos, invars, result


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def nested_darcy_evaluation(cfg: DictConfig) -> None:
    """Inference of the nested 2D Darcy flow benchmark problem.

    This inference script consecutively evaluates the models of nested FNO for the
    nested Darcy problem, taking into account the result of the model associated
    with the parent level. All results are stored in a numpy file and a selection
    of samples can be plotted in the end.
    """
    # initialize monitoring
    n_plots = 10

    DistributedManager.initialize()  # Only call this once in the entire script!
    log = PythonLogger(name="darcy_fno")

    model_names = sorted(list(cfg.arch.keys()))
    norm = {
        "permeability": (
            cfg.normaliser.permeability.mean,
            cfg.normaliser.permeability.std,
        ),
        "darcy": (cfg.normaliser.darcy.mean, cfg.normaliser.darcy.std),
    }

    # loop over models, evaluate, revoke normalisation and write results to file
    perm, darcy, pos, result = {}, {}, {}, None
    for name in model_names:
        position, invars, result = EvaluateModel(cfg, name, norm, result, log)
        perm[name] = (
            (invars * norm["permeability"][1] + norm["permeability"][0])
            .detach()
            .cpu()
            .numpy()
        )
        darcy[name] = (
            (result * norm["darcy"][1] + norm["darcy"][0]).detach().cpu().numpy()
        )
        pos[name] = position

    # save to file
    np.save(
        "nested_darcy_results.npy",
        {"permeability": perm, "darcy": darcy, "position": pos},
    )

    for idx in range(n_plots):
        plot_result(idx, perm, darcy, pos)


if __name__ == "__main__":
    nested_darcy_evaluation()
