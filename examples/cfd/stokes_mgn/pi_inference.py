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

import torch
import numpy as np

import time, os
import wandb as wb

try:
    import apex
except:
    pass

try:
    import pyvista as pv
except:
    raise ImportError(
        "Stokes Dataset requires the pyvista library. Install with "
        + "pip install pyvista"
    )

from modulus.models.mlp.fully_connected import FullyConnected

from modulus.launch.logging import (
    PythonLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
from utils import relative_lp_error, get_dataset
from constants import Constants

# Instantiate constants
C = Constants()


class PhysicsInformedInferencer:
    def __init__(
        self, wb, device, gnn_u, gnn_v, gnn_p, coords, coords_inflow, coords_noslip, nu
    ):
        super(PhysicsInformedInferencer, self).__init__()

        self.wb = wb
        self.device = device

        self.gnn_u = torch.tensor(gnn_u).float().to(self.device)
        self.gnn_v = torch.tensor(gnn_v).float().to(self.device)
        self.gnn_p = torch.tensor(gnn_p).float().to(self.device)

        self.coords = torch.tensor(coords, requires_grad=True).float().to(self.device)
        self.coords_inflow = torch.tensor(coords_inflow).float().to(self.device)
        self.coords_noslip = torch.tensor(coords_noslip).float().to(self.device)

        self.nu = nu

        # instantiate the model
        self.model = FullyConnected(
            C.mlp_input_dim, C.mlp_hidden_dim, C.mlp_output_dim, C.mlp_num_layers
        )
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=C.pi_lr)
        self.criterion = torch.nn.MSELoss()

    def parabolic_inflow(self, y, U_max=0.3):
        u = 4 * U_max * y * (0.4 - y) / (0.4**2)
        v = torch.zeros_like(y)
        return u, v

    def net_uvp(self, x, y):
        x = x / 1.5  # rescale x into [0, 1]
        y = y / 0.4  # rescale y into [0, 1]

        uvp = self.model(torch.cat([x, y], dim=1))
        u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]
        return u, v, p

    # Define the loss function incorporating the physics
    def net_r(self, x, y):
        u, v, p = self.net_uvp(x, y)

        # Compute gradients
        u_x = torch.autograd.grad(
            u, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u)
        )[0]
        u_y = torch.autograd.grad(
            u, y, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u)
        )[0]

        v_x = torch.autograd.grad(
            v, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v)
        )[0]
        v_y = torch.autograd.grad(
            v, y, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v)
        )[0]

        p_x = torch.autograd.grad(
            p, x, create_graph=True, grad_outputs=torch.ones_like(p)
        )[0]
        p_y = torch.autograd.grad(
            p, y, create_graph=True, grad_outputs=torch.ones_like(p)
        )[0]

        u_xx = torch.autograd.grad(
            u_x,
            x,
            create_graph=True,
            retain_graph=True,
            grad_outputs=torch.ones_like(u_x),
        )[0]
        u_yy = torch.autograd.grad(
            u_y,
            y,
            create_graph=True,
            retain_graph=True,
            grad_outputs=torch.ones_like(u_y),
        )[0]

        v_xx = torch.autograd.grad(
            v_x,
            x,
            create_graph=True,
            retain_graph=True,
            grad_outputs=torch.ones_like(u_x),
        )[0]
        v_yy = torch.autograd.grad(
            v_y,
            y,
            create_graph=True,
            retain_graph=True,
            grad_outputs=torch.ones_like(u_y),
        )[0]

        # Stokes equations
        mom_u = -self.nu * (u_xx + u_yy) + p_x
        mom_v = -self.nu * (v_xx + v_yy) + p_y
        cont = u_x + v_y

        return mom_u, mom_v, cont

    def loss(self):
        pred_u, pred_v, pred_p = self.net_uvp(self.coords[:, 0:1], self.coords[:, 1:2])
        pred_u_in, pred_v_in, _ = self.net_uvp(
            self.coords_inflow[:, 0:1], self.coords_inflow[:, 1:2]
        )
        pred_u_noslip, pred_v_noslip, _ = self.net_uvp(
            self.coords_noslip[:, 0:1], self.coords_noslip[:, 1:2]
        )
        pred_mom_u, pred_mom_v, pred_cont = self.net_r(
            self.coords[:, 0:1], self.coords[:, 1:2]
        )
        u_in, v_in = self.parabolic_inflow(self.coords_inflow[:, 1:2])

        # Compute losses

        # data loss
        loss_u = torch.mean((self.gnn_u - pred_u) ** 2)
        loss_v = torch.mean((self.gnn_v - pred_v) ** 2)
        loss_p = torch.mean((self.gnn_p - pred_p) ** 2)

        # inflow boundary condition loss
        loss_u_in = torch.mean((u_in - pred_u_in) ** 2)
        loss_v_in = torch.mean((v_in - pred_v_in) ** 2)

        # noslip boundary condition loss
        loss_u_noslip = torch.mean(pred_u_noslip**2)
        loss_v_noslip = torch.mean(pred_v_noslip**2)

        # pde loss
        loss_mom_u = torch.mean(pred_mom_u**2)
        loss_mom_v = torch.mean(pred_mom_v**2)
        loss_cont = torch.mean(pred_cont**2)

        return (
            loss_u,
            loss_v,
            loss_p,
            loss_u_in,
            loss_v_in,
            loss_u_noslip,
            loss_v_noslip,
            loss_mom_u,
            loss_mom_v,
            loss_cont,
        )

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        (
            loss_u,
            loss_v,
            loss_p,
            loss_u_in,
            loss_v_in,
            loss_u_noslip,
            loss_v_noslip,
            loss_mom_u,
            loss_mom_v,
            loss_cont,
        ) = self.loss()
        loss = (
            loss_u
            + loss_v
            + loss_p
            + 10 * loss_u_in
            + 10 * loss_v_in
            + 10 * loss_u_noslip
            + 10 * loss_v_noslip
            + loss_mom_u
            + loss_mom_v
            + 10 * loss_cont
        )
        loss.backward()
        self.optimizer.step()

        return (
            loss_u,
            loss_v,
            loss_p,
            loss_u_in,
            loss_v_in,
            loss_u_noslip,
            loss_v_noslip,
            loss_mom_u,
            loss_mom_v,
            loss_cont,
        )

    @torch.no_grad()
    def validation(self):
        self.model.eval()

        pred_u, pred_v, pred_p = self.net_uvp(self.coords[:, 0:1], self.coords[:, 1:2])

        pred_u = pred_u.detach().cpu().numpy()
        pred_v = pred_v.detach().cpu().numpy()
        pred_p = pred_p.detach().cpu().numpy()

        error_u = np.linalg.norm(ref_u - pred_u) / np.linalg.norm(ref_u)
        error_v = np.linalg.norm(ref_v - pred_v) / np.linalg.norm(ref_v)
        error_p = np.linalg.norm(ref_p - pred_p) / np.linalg.norm(ref_p)

        self.wb.log(
            {
                "test_u_error (%)": error_u,
                "test_v_error (%)": error_v,
                "test_p_error (%)": error_p,
            }
        )

        return error_u, error_v, error_p


if __name__ == "__main__":
    # CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # initialize loggers
    initialize_wandb(
        project="Modulus-Launch",
        entity="Modulus",
        name="Stokes-Physics-Informed-Inference",
        group="Stokes-DDP-Group",
        mode=C.wandb_mode,
    )

    logger = PythonLogger("main")  # General python logger
    logger.file_logging()

    # Get dataset
    path = os.path.join(C.results_dir, C.graph_path)

    (
        ref_u,
        ref_v,
        ref_p,
        gnn_u,
        gnn_v,
        gnn_p,
        coords,
        coords_inflow,
        coords_outflow,
        coords_wall,
        coords_polygon,
        nu,
    ) = get_dataset(path)
    coords_noslip = np.concatenate([coords_wall, coords_polygon], axis=0)

    # Initialize model
    pi_inferencer = PhysicsInformedInferencer(
        wb, device, gnn_u, gnn_v, gnn_p, coords, coords_inflow, coords_noslip, nu
    )

    logger.info("Physics-informed inference started...")
    for iters in range(C.pi_iters):
        # Start timing the iteration
        start_iter_time = time.time()

        (
            loss_u,
            loss_v,
            loss_p,
            loss_u_in,
            loss_v_in,
            loss_u_noslip,
            loss_v_noslip,
            loss_mom_u,
            loss_mom_v,
            loss_cont,
        ) = pi_inferencer.train()

        if iters % 100 == 0:
            error_u, error_v, error_p = pi_inferencer.validation()

            # Print losses
            logger.info(f"Iteration: {iters}")
            logger.info(f"Loss u: {loss_u.detach().cpu().numpy():.3e}")
            logger.info(f"Loss v: {loss_v.detach().cpu().numpy():.3e}")
            logger.info(f"Loss p: {loss_p.detach().cpu().numpy():.3e}")
            logger.info(f"Loss u_in: {loss_u_in.detach().cpu().numpy():.3e}")
            logger.info(f"Loss v_in: {loss_v_in.detach().cpu().numpy():.3e}")
            logger.info(f"Loss u noslip: {loss_u_noslip.detach().cpu().numpy():.3e}")
            logger.info(f"Loss v noslip: {loss_v_noslip.detach().cpu().numpy():.3e}")
            logger.info(f"Loss momentum u: {loss_mom_u.detach().cpu().numpy():.3e}")
            logger.info(f"Loss momentum v: {loss_mom_v.detach().cpu().numpy():.3e}")
            logger.info(f"Loss continuity: {loss_cont.detach().cpu().numpy():.3e}")

            # Print errors
            logger.info(f"Error u: {error_u:.3e}")
            logger.info(f"Error v: {error_v:.3e}")
            logger.info(f"Error p: {error_p:.3e}")

            # Print iteration time
            end_iter_time = time.time()
            logger.info(
                f"This iteration took {end_iter_time - start_iter_time:.2f} seconds"
            )
            logger.info("-" * 50)  # Add a separator for clarity

    logger.info("Training completed!")

    # Save results
    pred_u, pred_v, pred_p = pi_inferencer.net_uvp(
        pi_inferencer.coords[:, 0:1], pi_inferencer.coords[:, 1:2]
    )

    pred_u = pred_u.detach().cpu().numpy()
    pred_v = pred_v.detach().cpu().numpy()
    pred_p = pred_p.detach().cpu().numpy()

    polydata = pv.read(path)
    polydata["filtered_u"] = pred_u
    polydata["filtered_v"] = pred_v
    polydata["filtered_p"] = pred_p
    polydata.save(path)
