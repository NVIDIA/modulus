# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import matplotlib.pyplot as plt
from torch import FloatTensor
from modulus.launch.logging import LaunchLogger


class GridValidator:
    """Grid Validator

    The validator compares model output and target, inverts normalisation and plots a sample

    Parameters
    ----------
    loss_fun : MSELoss
        loss function for assessing validation error
    norm : Dict, optional
        mean and standard deviation for each channel to normalise input and target
    font_size : float, optional
        font size used in figures

    """

    def __init__(
        self,
        loss_fun,
        norm,
        font_size: float = 28.0,
    ):
        self.norm = norm
        self.criterion = loss_fun
        self.font_size = font_size
        self.headers = ("invar", "truth", "prediction", "relative error")

    def compare(
        self,
        prediction: FloatTensor,
        target: FloatTensor,
        step: int,
        logger: LaunchLogger,
    ) -> float:
        """compares model output, target and plots everything

        Parameters
        ----------
        invar : FloatTensor
            input to model
        target : FloatTensor
            ground truth
        prediction : FloatTensor
            model output
        step : int
            iteration counter
        logger : LaunchLogger
            logger to which figure is passed

        Returns
        -------
        float
            validation error
        """
        loss = self.criterion(prediction, target)
        # print(f"target.shape: {target.shape}, prediction.shape: {prediction.shape}")
        target = target.cpu().numpy()[0, :, :]
        prediction = prediction.reshape(-1, 85, 85).detach().cpu().numpy()[0, :, :]

        plt.close("all")
        plt.rcParams.update({"font.size": self.font_size})
        fig, ax = plt.subplots(1, 3, figsize=(15 * 3, 15), sharey=True)
        im = []
        im.append(ax[0].imshow(target))
        im.append(ax[1].imshow(prediction))
        im.append(ax[2].imshow((prediction - target)))

        for ii in range(len(im)):
            fig.colorbar(im[ii], ax=ax[ii], location="bottom", fraction=0.046, pad=0.04)
            ax[ii].set_title(self.headers[ii])

        logger.log_figure(figure=fig, artifact_file=f"validation_step_{step:03d}.png")

        return loss
