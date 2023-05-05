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

import io
import numpy as np
import concurrent.futures as cf
from PIL import Image
from moviepy.editor import ImageSequenceClip
import wandb

import torch


def plot_comparison(
    pred,
    truth,
    pred_title="Prediction",
    truth_title="Ground truth",
    cmap="twilight_shifted",
    projection="mollweide",
    diverging=False,
    figsize=(8, 9),
    vmax=None,
):
    """
    Visualization tool to plot a comparison between ground truth and prediction
    pred: 2d array
    truth: 2d array
    cmap: colormap
    projection: 'mollweide', 'hammer', 'aitoff' or None
    """
    import matplotlib.pyplot as plt

    assert len(pred.shape) == 2
    assert len(truth.shape) == 2
    assert pred.shape == truth.shape

    H, W = pred.shape
    lon = np.linspace(-np.pi, np.pi, W)
    lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, H)
    Lon, Lat = np.meshgrid(lon, lat)

    # only normalize with the truth
    vmax = vmax or np.abs(truth).max()
    # vmax = vmax or max(np.abs(pred).max(), np.abs(truth).max())
    if diverging:
        vmin = -vmax
    else:
        vmin = 0.0

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(2, 1, 1, projection=projection)  # can also be Mollweide

    ax.pcolormesh(Lon, Lat, pred, cmap=cmap, vmax=vmax, vmin=vmin)
    ax.set_title(pred_title)
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax = fig.add_subplot(2, 1, 2, projection=projection)  # can also be Mollweide

    ax.pcolormesh(Lon, Lat, truth, cmap=cmap, vmax=vmax, vmin=vmin)
    ax.set_title(truth_title)
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()

    # save into memory buffer
    buf = io.BytesIO()
    plt.savefig(buf)
    plt.close(fig)
    buf.seek(0)

    # create image
    image = Image.open(buf)

    return image


def plot_ifs_acc_comparison(acc_curve, params, epoch):

    import os

    ifs_comparison_dict = {
        "u10m": "u10_2018_acc.npy",
        "v10m": "v10_2018_acc.npy",
        "z500": "z500_2018_acc.npy",
        "t2m": "t2m_2018_acc.npy",
        "t850": "t850_2018_acc.npy",
    }

    for comparison_var, comparison_file in ifs_comparison_dict.items():

        ifs_acc_file = os.path.join(
            params.ifs_acc_path, comparison_var, comparison_file
        )

        ifs_acc = np.mean(np.load(ifs_acc_file), axis=0)[0 : acc_curve.shape[1] + 1, 0]

        channel_names = params.channel_names

        fcn_acc = acc_curve[channel_names.index(comparison_var), :].cpu().numpy()

        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        var_name = comparison_var

        fig, ax = plt.subplots()
        t = np.arange(1, len(ifs_acc), 1) * 6
        ax.plot(t, ifs_acc[1:], ".-", label="IFS")
        ax.plot(t, fcn_acc, ".-", label="S-FNO")
        xticks = np.arange(0, len(ifs_acc), 1) * 6
        x_locator = ticker.FixedLocator(xticks)
        ax.xaxis.set_major_locator(x_locator)
        y_locator = ticker.MaxNLocator(nbins=20)
        ax.yaxis.set_major_locator(y_locator)
        ax.grid(which="major", alpha=0.5)
        ax.legend()
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("ACC " + var_name)
        ax.set_title(params.wandb_name)
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
        fig.savefig(os.path.join(params.experiment_dir, "acc_" + var_name + ".png"))
        # push to wandb
        if params.log_to_wandb:
            wandb.log({"ACC " + var_name: wandb.Image(fig)}, step=epoch)


def visualize_field(tag, func_string, prediction, target, scale, bias, diverging):

    torch.cuda.nvtx.range_push("visualize_field")

    # get func handle:
    func_handle = eval(func_string)

    # unscale:
    pred = scale * prediction + bias
    targ = scale * target + bias

    # apply functor:
    pred = func_handle(pred)
    targ = func_handle(targ)

    # generate image
    image = plot_comparison(
        pred,
        targ,
        pred_title="Prediction",
        truth_title="Ground truth",
        projection="mollweide",
        diverging=diverging,
    )

    torch.cuda.nvtx.range_pop()

    return tag, image


class VisualizationWrapper(object):
    def __init__(
        self, log_to_wandb, path, prefix, plot_list, scale=1.0, bias=0.0, num_workers=1
    ):
        self.log_to_wandb = log_to_wandb
        self.generate_video = True
        self.path = path
        self.prefix = prefix
        self.plot_list = plot_list

        # normalization
        self.scale = scale
        self.bias = bias

        # this is for parallel processing
        self.executor = cf.ProcessPoolExecutor(max_workers=num_workers)
        self.requests = []

    def reset(self):
        self.requests = []

    def add(self, tag, prediction, target):
        # go through the plot list
        for item in self.plot_list:
            field_name = item["name"]
            func_string = item["functor"]
            plot_diverge = item["diverging"]
            self.requests.append(
                self.executor.submit(
                    visualize_field,
                    (tag, field_name),
                    func_string,
                    np.copy(prediction),
                    np.copy(target),
                    self.scale,
                    self.bias,
                    plot_diverge,
                )
            )

        return

    def finalize(self):

        torch.cuda.nvtx.range_push("VisualizationWrapper:finalize")

        results = {}

        for request in cf.as_completed(self.requests):
            token, image = request.result()
            tag, field_name = token
            prefix = field_name + "_" + tag
            # results.append(wandb.Image(image, caption=prefix))
            results[prefix] = image

        if self.generate_video:

            if self.log_to_wandb:
                video = []

                # draw stuff that goes on every frame here
                for prefix, image in sorted(results.items()):
                    video.append(np.transpose(np.asarray(image), (2, 0, 1)))

                video = np.stack(video)
                results = [wandb.Video(video, fps=1, format="gif")]
            else:
                video = []

                # draw stuff that goes on every frame here
                for prefix, image in sorted(results.items()):
                    video.append(np.asarray(image))

                video = ImageSequenceClip(video, fps=1)
                video.write_gif("video_output.gif")

        else:
            results = [
                wandb.Image(image, caption=prefix) for prefix, image in results.items()
            ]

        if self.log_to_wandb and results:
            wandb.log({"Inference samples": results})

        # reset requests
        self.reset()

        torch.cuda.nvtx.range_pop()

        return
