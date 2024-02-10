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

import torch
import matplotlib.pyplot as plt
from .utils import get_grid3d


def plot_predictions(
    key,
    key_t,
    test_x,
    test_y,
    preds_y,
    print_index=False,
    save_path=None,
    font_size=None,
):

    if font_size is not None:
        plt.rcParams.update({"font.size": font_size})

    Nsamples, Nx, Ny, Nt, Nfields = preds_y.shape

    a = test_x[key]
    # Nt, Nx, _ = a.shape
    h0 = a[..., 0, -1]
    # v0 = a[..., 0, -1]
    pred_h = preds_y[key, ..., key_t, 0]
    pred_u = preds_y[key, ..., key_t, 1]
    pred_v = preds_y[key, ..., key_t, 2]
    true_h = test_y[key, ..., key_t, 0]
    true_u = test_y[key, ..., key_t, 1]
    true_v = test_y[key, ..., key_t, 2]

    # T = a[:,:,2]
    # X = a[:,:,1]
    # x = X[0]
    x = torch.linspace(0, 1, Nx + 1)[:-1]
    y = torch.linspace(0, 1, Nx + 1)[:-1]
    X, Y = torch.meshgrid(x, y, indexing="ij")
    u0 = torch.zeros_like(X)
    v0 = torch.zeros_like(X)
    t = a[0, 0, key_t, 2]
    grid_x, grid_y, grid_t = get_grid3d(Nx, Nt)

    fig = plt.figure(figsize=(24, 15))
    plt.subplot(3, 4, 1)

    plt.pcolormesh(X, Y, h0, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    plt.title("Intial Condition $\eta(x,y)$")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 2)
    # plt.pcolor(XX,TT, S_test, cmap='jet')
    plt.pcolormesh(X, Y, true_h, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    #     plt.title(f'Exact $\eta(x,y,t={t:.2f})$')
    plt.title(f"Exact $\eta(x,y,t={int(t)})$")

    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 3)
    # plt.pcolor(XX,TT, S_pred, cmap='jet')
    plt.pcolormesh(X, Y, pred_h, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    #     plt.title(f'Predict $\eta(x,y,t={t:.2f})$')
    plt.title(f"Predict $\eta(x,y,t={int(t)})$")
    plt.axis("square")

    plt.tight_layout()

    plt.subplot(3, 4, 4)
    # plt.pcolor(XX,TT, S_pred - S_test, cmap='jet')
    plt.pcolormesh(X, Y, pred_h - true_h, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    plt.title("Absolute Error $\eta$")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 5)
    plt.pcolormesh(X, Y, u0, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    plt.title("Intial Condition $u(x,y)$")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 6)
    # plt.pcolor(XX,TT, S_test, cmap='jet')
    plt.pcolormesh(X, Y, true_u, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    #     plt.title(f'Exact $u(x,y,t={t:.2f})$')
    plt.title(f"Exact $u(x,y,t={int(t)})$")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 7)
    # plt.pcolor(XX,TT, S_pred, cmap='jet')
    plt.pcolormesh(X, Y, pred_u, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    #     plt.title(f'Predict $u(x,y,t={t:.2f})$')
    plt.title(f"Predict $u(x,y,t={int(t)})$")
    plt.axis("square")

    plt.tight_layout()

    plt.subplot(3, 4, 8)
    # plt.pcolor(XX,TT, S_pred - S_test, cmap='jet')
    plt.pcolormesh(X, Y, pred_u - true_u, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    plt.title("Absolute Error u")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 9)
    plt.pcolormesh(X, Y, v0, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    plt.title("Intial Condition $v(x,y)$")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 10)
    # plt.pcolor(XX,TT, S_test, cmap='jet')
    plt.pcolormesh(X, Y, true_v, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    #     plt.title(f'Exact $v(x,y,t={t:.2f})$')
    plt.title(f"Exact $v(x,y,t={int(t)})$")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 11)
    # plt.pcolor(XX,TT, S_pred, cmap='jet')
    plt.pcolormesh(X, Y, pred_v, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    #     plt.title(f'Predict $v(x,y,t={t:.2f})$')
    plt.title(f"Predict $v(x,y,t={int(t)})$")
    plt.axis("square")

    plt.tight_layout()

    plt.subplot(3, 4, 12)
    # plt.pcolor(XX,TT, S_pred - S_test, cmap='jet')
    plt.pcolormesh(X, Y, pred_v - true_v, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    plt.title("Absolute Error v")
    plt.tight_layout()
    plt.axis("square")

    if save_path is not None:
        plt.savefig(f"{save_path}.png", bbox_inches="tight")
    plt.show()
