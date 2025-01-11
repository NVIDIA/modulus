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

"""
This code contains the DoMINO model architecture.
The DoMINO class contains an architecture to model both surface and 
volume quantities together as well as separately (controlled using 
the config.yaml file)
"""

# from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from modulus.models.layers.ball_query import BallQueryLayer

# from modulus.models.meta import ModelMetaData
# from modulus.models.module import Module


def calculate_pos_encoding(nx, d=8):
    """Function to caluculate positional encoding"""
    vec = []
    for k in range(int(d / 2)):
        vec.append(torch.sin(nx / 10000 ** (2 * (k) / d)))
        vec.append(torch.cos(nx / 10000 ** (2 * (k) / d)))
    return vec


def scale_sdf(sdf):
    """Function to scale SDF"""
    return sdf / (0.4 + abs(sdf))


def calculate_gradient(sdf):
    """Function to calculate the gradients of SDF"""
    m, n, o = sdf.shape[2], sdf.shape[3], sdf.shape[4]
    sdf_x = sdf[:, :, 2:m, :, :] - sdf[:, :, 0 : m - 2, :, :]
    sdf_y = sdf[:, :, :, 2:n, :] - sdf[:, :, :, 0 : n - 2, :]
    sdf_z = sdf[:, :, :, :, 2:o] - sdf[:, :, :, :, 0 : o - 2]

    sdf_x = F.pad(input=sdf_x, pad=(0, 0, 0, 0, 0, 1), mode="constant", value=0.0)
    sdf_x = F.pad(input=sdf_x, pad=(0, 0, 0, 0, 1, 0), mode="constant", value=0.0)
    sdf_y = F.pad(input=sdf_y, pad=(0, 0, 0, 1, 0, 0), mode="constant", value=0.0)
    sdf_y = F.pad(input=sdf_y, pad=(0, 0, 1, 0, 0, 0), mode="constant", value=0.0)
    sdf_z = F.pad(input=sdf_z, pad=(0, 1, 0, 0, 0, 0), mode="constant", value=0.0)
    sdf_z = F.pad(input=sdf_z, pad=(1, 0, 0, 0, 0, 0), mode="constant", value=0.0)

    return sdf_x, sdf_y, sdf_z


def binarize_sdf(sdf):
    """Function to calculate the binarize the SDF"""
    sdf = torch.where(sdf >= 0, 0.0, 1.0)
    return sdf


class BQWarp(nn.Module):
    """Warp based ball-query layer"""

    def __init__(
        self,
        input_features,
        grid_resolution=[256, 96, 64],
        radius=0.25,
        neighbors_in_radius=10,
    ):
        super().__init__()
        self.ball_query_layer = BallQueryLayer(neighbors_in_radius, radius)
        self.grid_resolution = grid_resolution

    def forward(self, x, p_grid, reverse_mapping=True):
        batch_size = x.shape[0]
        nx, ny, nz = (
            self.grid_resolution[0],
            self.grid_resolution[1],
            self.grid_resolution[2],
        )

        p_grid = torch.reshape(p_grid, (batch_size, nx * ny * nz, 3))
        p1 = nx * ny * nz
        p2 = x.shape[1]

        if reverse_mapping:
            lengths1 = torch.full((batch_size,), p1, dtype=torch.int32)
            lengths2 = torch.full((batch_size,), p2, dtype=torch.int32)
            mapping, num_neighbors, outputs = self.ball_query_layer(
                p_grid,
                x,
                lengths1,
                lengths2,
            )
        else:
            lengths1 = torch.full((batch_size,), p2, dtype=torch.int32)
            lengths2 = torch.full((batch_size,), p1, dtype=torch.int32)
            mapping, num_neighbors, outputs = self.ball_query_layer(
                x,
                p_grid,
                lengths1,
                lengths2,
            )

        return mapping, outputs


class GeoConvOut(nn.Module):
    """Geometry layer to project STLs on grids"""

    def __init__(self, input_features, model_parameters, grid_resolution=[256, 96, 64]):
        super().__init__()
        base_neurons = model_parameters.base_neurons

        self.fc1 = nn.Linear(input_features, base_neurons)
        self.fc2 = nn.Linear(base_neurons, int(base_neurons / 2))
        self.fc3 = nn.Linear(int(base_neurons / 2), model_parameters.base_neurons_out)

        self.grid_resolution = grid_resolution

        self.activation = F.relu

    def forward(self, x, radius=0.025, neighbors_in_radius=10):
        batch_size = x.shape[0]
        nx, ny, nz = (
            self.grid_resolution[0],
            self.grid_resolution[1],
            self.grid_resolution[2],
        )

        mask = abs(x - 0) > 1e-6

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = F.tanh(self.fc3(x))
        mask = mask[:, :, :, 0:1].expand(
            mask.shape[0], mask.shape[1], mask.shape[2], x.shape[-1]
        )

        x = torch.sum(x * mask, 2)

        x = torch.reshape(x, (batch_size, x.shape[-1], nx, ny, nz))
        return x


class GeoProcessor(nn.Module):
    """Geometry processing layer using CNNs"""

    def __init__(self, input_filters, model_parameters):
        super().__init__()
        base_filters = model_parameters.base_filters
        self.conv1 = nn.Conv3d(
            input_filters, base_filters, kernel_size=3, padding="same"
        )
        self.conv_bn1 = nn.BatchNorm3d(int(base_filters))
        self.conv2 = nn.Conv3d(
            base_filters, 2 * base_filters, kernel_size=3, padding="same"
        )
        self.conv_bn2 = nn.BatchNorm3d(int(2 * base_filters))
        self.conv3 = nn.Conv3d(
            2 * base_filters, 4 * base_filters, kernel_size=3, padding="same"
        )
        self.conv_bn3 = nn.BatchNorm3d(int(4 * base_filters))
        self.conv3_1 = nn.Conv3d(
            4 * base_filters, 4 * base_filters, kernel_size=3, padding="same"
        )
        self.conv4 = nn.Conv3d(
            4 * base_filters, 2 * base_filters, kernel_size=3, padding="same"
        )
        self.conv_bn4 = nn.BatchNorm3d(int(2 * base_filters))
        self.conv5 = nn.Conv3d(
            4 * base_filters, base_filters, kernel_size=3, padding="same"
        )
        self.conv_bn5 = nn.BatchNorm3d(int(base_filters))
        self.conv6 = nn.Conv3d(
            2 * base_filters, input_filters, kernel_size=3, padding="same"
        )
        self.conv_bn6 = nn.BatchNorm3d(int(input_filters))
        self.conv7 = nn.Conv3d(
            2 * input_filters, input_filters, kernel_size=3, padding="same"
        )
        self.conv8 = nn.Conv3d(input_filters, 1, kernel_size=3, padding="same")
        self.avg_pool = torch.nn.AvgPool3d((2, 2, 2))
        self.max_pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.activation = F.relu
        self.batch_norm = False

    def forward(self, x):
        # Encoder
        x0 = x
        if self.batch_norm:
            x = self.activation(self.conv_bn1(self.conv1(x)))
        else:
            x = self.activation(self.conv1(x))
        x = self.max_pool(x)
        x1 = x
        if self.batch_norm:
            x = self.activation(self.conv_bn2(self.conv2(x)))
        else:
            x = self.activation((self.conv2(x)))
        x = self.max_pool(x)

        x2 = x
        if self.batch_norm:
            x = self.activation(self.conv_bn3(self.conv2(x)))
        else:
            x = self.activation((self.conv3(x)))
        x = self.max_pool(x)

        # Processor loop
        x = F.relu(self.conv3_1(x))

        # Decoder
        if self.batch_norm:
            x = self.activation(self.conv_bn4(self.conv4(x)))
        else:
            x = self.activation((self.conv4(x)))
        x = self.upsample(x)
        x = torch.cat((x, x2), axis=1)

        if self.batch_norm:
            x = self.activation(self.conv_bn5(self.conv5(x)))
        else:
            x = self.activation((self.conv5(x)))
        x = self.upsample(x)
        x = torch.cat((x, x1), axis=1)
        if self.batch_norm:
            x = self.activation(self.conv_bn6(self.conv6(x)))
        else:
            x = self.activation((self.conv6(x)))
        x = self.upsample(x)
        x = torch.cat((x, x0), axis=1)

        x = self.activation(self.conv7(x))
        x = self.conv8(x)

        return x


class GeometryRep(nn.Module):
    """Geometry representation from STLs block"""

    def __init__(self, input_features, model_parameters=None):
        super().__init__()
        geometry_rep = model_parameters.geometry_rep

        self.bq_warp_short = BQWarp(
            input_features=input_features,
            grid_resolution=model_parameters.interp_res,
            radius=geometry_rep.geo_conv.radius_short,
        )

        self.bq_warp_long = BQWarp(
            input_features=input_features,
            grid_resolution=model_parameters.interp_res,
            radius=geometry_rep.geo_conv.radius_long,
        )

        self.geo_conv_out = GeoConvOut(
            input_features=input_features,
            model_parameters=geometry_rep.geo_conv,
            grid_resolution=model_parameters.interp_res,
        )

        self.geo_processor_short_range = GeoProcessor(
            input_filters=geometry_rep.geo_conv.base_neurons_out,
            model_parameters=geometry_rep.geo_processor,
        )
        self.geo_processor_long_range = GeoProcessor(
            input_filters=geometry_rep.geo_conv.base_neurons_out,
            model_parameters=geometry_rep.geo_processor,
        )
        self.geo_processor_sdf = GeoProcessor(
            input_filters=6, model_parameters=geometry_rep.geo_processor
        )
        self.activation = F.relu
        self.radius_short = geometry_rep.geo_conv.radius_short
        self.radius_long = geometry_rep.geo_conv.radius_long
        self.hops = geometry_rep.geo_conv.hops

    def forward(self, x, p_grid, sdf):

        # Expand SDF
        sdf = torch.unsqueeze(sdf, 1)

        # Calculate short-range geoemtry dependency
        mapping, k_short = self.bq_warp_short(x, p_grid)
        x_encoding_short = self.geo_conv_out(k_short)

        # Calculate long-range geometry dependency
        mapping, k_long = self.bq_warp_long(x, p_grid)
        x_encoding_long = self.geo_conv_out(k_long)

        # Scaled sdf to emphasis on surface
        scaled_sdf = scale_sdf(sdf)
        # Binary sdf
        binary_sdf = binarize_sdf(sdf)
        # Gradients of SDF
        sdf_x, sdf_y, sdf_z = calculate_gradient(sdf)

        # Propagate information in the geometry enclosed BBox
        for _ in range(self.hops):
            dx = self.geo_processor_short_range(x_encoding_short) / self.hops
            x_encoding_short = x_encoding_short + dx

        # Propagate information in the computational domain BBox
        for _ in range(self.hops):
            dx = self.geo_processor_long_range(x_encoding_long) / self.hops
            x_encoding_long = x_encoding_long + dx

        # Process SDF and its computed features
        sdf = torch.cat((sdf, scaled_sdf, binary_sdf, sdf_x, sdf_y, sdf_z), 1)
        sdf_encoding = self.geo_processor_sdf(sdf)

        # Geometry encoding comprised of short-range, long-range and SDF features
        encoding_g = torch.cat((x_encoding_short, sdf_encoding, x_encoding_long), 1)

        return encoding_g


class NNBasisFunctions(nn.Module):
    """Basis function layer for point clouds"""

    def __init__(self, input_features, model_parameters=None):
        super(NNBasisFunctions, self).__init__()
        self.input_features = input_features

        base_layer = model_parameters.base_layer
        self.fc1 = nn.Linear(self.input_features, base_layer)
        self.fc2 = nn.Linear(base_layer, int(base_layer))
        self.fc3 = nn.Linear(int(base_layer), int(base_layer))
        self.bn1 = nn.BatchNorm1d(base_layer)
        self.bn2 = nn.BatchNorm1d(int(base_layer))
        self.bn3 = nn.BatchNorm1d(int(base_layer))

        self.activation = F.gelu

    def forward(self, x, padded_value=-10):
        facets = x
        facets = self.activation(self.fc1(facets))
        facets = self.activation(self.fc2(facets))
        facets = self.fc3(facets)

        return facets


class AggregationModel(nn.Module):
    """Layer to aggregate local geometry encoding with basis functions"""

    def __init__(
        self, input_features, output_features, model_parameters=None, new_change=True
    ):
        super(AggregationModel, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.new_change = new_change
        base_layer = model_parameters.base_layer
        self.fc1 = nn.Linear(self.input_features, base_layer)
        self.fc2 = nn.Linear(base_layer, int(base_layer / 2))
        self.fc3 = nn.Linear(int(base_layer / 2), int(base_layer / 4))
        self.fc4 = nn.Linear(int(base_layer / 4), int(base_layer / 8))
        self.fc5 = nn.Linear(int(base_layer / 8), self.output_features)
        self.bn1 = nn.BatchNorm1d(base_layer)
        self.bn2 = nn.BatchNorm1d(int(base_layer / 2))
        self.bn3 = nn.BatchNorm1d(int(base_layer / 4))
        self.bn4 = nn.BatchNorm1d(int(base_layer / 8))
        self.activation = F.gelu

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        out = self.activation(self.fc3(out))
        out = self.activation(self.fc4(out))

        out = self.fc5(out)

        return out


# @dataclass
# class MetaData(ModelMetaData):
#     name: str = "DoMINO"
#     # Optimization
#     jit: bool = False
#     cuda_graphs: bool = True
#     amp: bool = True
#     # Inference
#     onnx_cpu: bool = True
#     onnx_gpu: bool = True
#     onnx_runtime: bool = True
#     # Physics informed
#     var_dim: int = 1
#     func_torch: bool = False
#     auto_grad: bool = False


class DoMINO(nn.Module):
    """DoMINO model architecture
    Parameters
    ----------
    input_features : int
        Number of point input features
    output_features_vol : int
        Number of output features in volume
    output_features_surf : int
        Number of output features on surface
    model_parameters: dict
        Dictionary of model parameters controlled by config.yaml

    Example
    -------
    >>> from modulus.models.domino.model import DoMINO
    >>> import torch, os
    >>> from hydra import compose, initialize
    >>> from omegaconf import OmegaConf
    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> cfg = OmegaConf.register_new_resolver("eval", eval)
    >>> with initialize(version_base="1.3", config_path="examples/cfd/external_aerodynamics/domino/src/conf"):
    ...    cfg = compose(config_name="config")
    >>> cfg.model.model_type = "combined"
    >>> model = DoMINO(
    ...         input_features=3,
    ...         output_features_vol=5,
    ...         output_features_surf=4,
    ...         model_parameters=cfg.model
    ...     ).to(device)

    Warp ...
    >>> bsize = 1
    >>> nx, ny, nz = 128, 64, 48
    >>> num_neigh = 7
    >>> pos_normals_closest_vol = torch.randn(bsize, 100, 3).to(device)
    >>> pos_normals_com_vol = torch.randn(bsize, 100, 3).to(device)
    >>> pos_normals_com_surface = torch.randn(bsize, 100, 3).to(device)
    >>> geom_centers = torch.randn(bsize, 100, 3).to(device)
    >>> grid = torch.randn(bsize, nx, ny, nz, 3).to(device)
    >>> surf_grid = torch.randn(bsize, nx, ny, nz, 3).to(device)
    >>> sdf_grid = torch.randn(bsize, nx, ny, nz).to(device)
    >>> sdf_surf_grid = torch.randn(bsize, nx, ny, nz).to(device)
    >>> sdf_nodes = torch.randn(bsize, 100, 1).to(device)
    >>> surface_coordinates = torch.randn(bsize, 100, 3).to(device)
    >>> surface_neighbors = torch.randn(bsize, 100, num_neigh, 3).to(device)
    >>> surface_normals = torch.randn(bsize, 100, 3).to(device)
    >>> surface_neighbors_normals = torch.randn(bsize, 100, num_neigh, 3).to(device)
    >>> surface_sizes = torch.randn(bsize, 100, 3).to(device)
    >>> surface_neighbors_sizes = torch.randn(bsize, 100, num_neigh, 3).to(device)
    >>> volume_coordinates = torch.randn(bsize, 100, 3).to(device)
    >>> vol_grid_max_min = torch.randn(bsize, 2, 3).to(device)
    >>> surf_grid_max_min = torch.randn(bsize, 2, 3).to(device)
    >>> input_dict = {
    ...            "pos_volume_closest": pos_normals_closest_vol,
    ...            "pos_volume_center_of_mass": pos_normals_com_vol,
    ...            "pos_surface_center_of_mass": pos_normals_com_surface,
    ...            "geometry_coordinates": geom_centers,
    ...            "grid": grid,
    ...            "surf_grid": surf_grid,
    ...            "sdf_grid": sdf_grid,
    ...            "sdf_surf_grid": sdf_surf_grid,
    ...            "sdf_nodes": sdf_nodes,
    ...            "surface_mesh_centers": surface_coordinates,
    ...            "surface_mesh_neighbors": surface_neighbors,
    ...            "surface_normals": surface_normals,
    ...            "surface_neighbors_normals": surface_neighbors_normals,
    ...            "surface_areas": surface_sizes,
    ...            "surface_neighbors_areas": surface_neighbors_sizes,
    ...            "volume_mesh_centers": volume_coordinates,
    ...            "volume_min_max": vol_grid_max_min,
    ...            "surface_min_max": surf_grid_max_min,
    ...        }
    >>> output = model(input_dict)
    Module ...
    >>> print(f"{output[0].shape}, {output[1].shape}")
    torch.Size([1, 100, 5]), torch.Size([1, 100, 4])
    """

    def __init__(
        self,
        input_features,
        output_features_vol=None,
        output_features_surf=None,
        model_parameters=None,
    ):
        super(DoMINO, self).__init__()
        self.input_features = input_features
        self.output_features_vol = output_features_vol
        self.output_features_surf = output_features_surf

        if self.output_features_vol is None and self.output_features_surf is None:
            raise ValueError("Need to specify number of volume or surface features")

        self.num_variables_vol = output_features_vol
        self.num_variables_surf = output_features_surf
        self.grid_resolution = model_parameters.interp_res
        self.surface_neighbors = model_parameters.surface_neighbors
        self.use_surface_normals = model_parameters.use_surface_normals
        self.use_only_normals = model_parameters.use_only_normals

        if self.use_surface_normals:
            if self.use_only_normals:
                input_features_surface = input_features + 3
            else:
                input_features_surface = input_features + 4
        else:
            input_features_surface = input_features

        self.geo_rep = GeometryRep(
            input_features=input_features,
            model_parameters=model_parameters,
        )

        # Basis functions for surface and volume
        base_layer_nn = model_parameters.nn_basis_functions.base_layer
        if self.output_features_surf is not None:
            self.nn_basis_surf = nn.ModuleList()
            for _ in range(self.num_variables_surf):
                self.nn_basis_surf.append(
                    NNBasisFunctions(
                        input_features=input_features_surface,
                        model_parameters=model_parameters.nn_basis_functions,
                    )
                )

        if self.output_features_vol is not None:
            self.nn_basis_vol = nn.ModuleList()
            for _ in range(self.num_variables_vol):
                self.nn_basis_vol.append(
                    NNBasisFunctions(
                        input_features=input_features,
                        model_parameters=model_parameters.nn_basis_functions,
                    )
                )

        # Positional encoding
        position_encoder_base_neurons = model_parameters.position_encoder.base_neurons
        if self.output_features_vol is not None:
            if model_parameters.positional_encoding:
                inp_pos_vol = 25 if model_parameters.use_sdf_in_basis_func else 12
            else:
                inp_pos_vol = 7 if model_parameters.use_sdf_in_basis_func else 3

            self.fc_p_vol = nn.Linear(inp_pos_vol, position_encoder_base_neurons)

        if self.output_features_surf is not None:
            if model_parameters.positional_encoding:
                inp_pos_surf = 12
            else:
                inp_pos_surf = 3

            self.fc_p_surf = nn.Linear(inp_pos_surf, position_encoder_base_neurons)

        # Positional encoding hidden layers
        self.fc_p1 = nn.Linear(
            position_encoder_base_neurons, position_encoder_base_neurons
        )
        self.fc_p2 = nn.Linear(
            position_encoder_base_neurons, position_encoder_base_neurons
        )

        # BQ for surface and volume
        self.neighbors_in_radius = model_parameters.geometry_local.neighbors_in_radius
        self.radius = model_parameters.geometry_local.radius
        self.bq_warp = BQWarp(
            input_features=input_features,
            grid_resolution=model_parameters.interp_res,
            radius=self.radius,
            neighbors_in_radius=self.neighbors_in_radius,
        )

        base_layer_geo = model_parameters.geometry_local.base_layer
        self.fc_1 = nn.Linear(self.neighbors_in_radius * 3, base_layer_geo)
        self.fc_2 = nn.Linear(base_layer_geo, base_layer_geo)
        self.activation = F.gelu

        # Aggregation model
        if self.output_features_surf is not None:
            # Surface
            self.agg_model_surf = nn.ModuleList()
            for _ in range(self.num_variables_surf):
                self.agg_model_surf.append(
                    AggregationModel(
                        input_features=position_encoder_base_neurons
                        + base_layer_nn
                        + base_layer_geo,
                        output_features=1,
                        model_parameters=model_parameters.aggregation_model,
                    )
                )

        if self.output_features_vol is not None:
            # Volume
            self.agg_model_vol = nn.ModuleList()
            for _ in range(self.num_variables_vol):
                self.agg_model_vol.append(
                    AggregationModel(
                        input_features=position_encoder_base_neurons
                        + base_layer_nn
                        + base_layer_geo,
                        output_features=1,
                        model_parameters=model_parameters.aggregation_model,
                    )
                )

    def geometry_encoder(self, geo_centers, p_grid, sdf):
        """Function to return local geometry encoding"""
        return self.geo_rep(geo_centers, p_grid, sdf)

    def position_encoder(self, encoding_node, eval_mode="volume"):
        """Function to calculate positional encoding"""
        if eval_mode == "volume":
            x = self.activation(self.fc_p_vol(encoding_node))
        elif eval_mode == "surface":
            x = self.activation(self.fc_p_surf(encoding_node))
        x = self.activation(self.fc_p1(x))
        x = self.fc_p2(x)
        return x

    def geo_encoding_local_surface(self, encoding_g, volume_mesh_centers, p_grid):
        """Function to calculate local geometry encoding from global encoding for surface"""
        batch_size = volume_mesh_centers.shape[0]
        nx, ny, nz = (
            self.grid_resolution[0],
            self.grid_resolution[1],
            self.grid_resolution[2],
        )
        p_grid = torch.reshape(p_grid, (batch_size, nx * ny * nz, 3))
        mapping, outputs = self.bq_warp(
            volume_mesh_centers, p_grid, reverse_mapping=False
        )
        mapping = mapping.type(torch.int64)
        mask = mapping != 0

        geo_encoding = torch.reshape(encoding_g[:, 0], (batch_size, 1, nx * ny * nz))
        geo_encoding = geo_encoding.expand(
            batch_size, volume_mesh_centers.shape[1], geo_encoding.shape[2]
        )
        sdf_encoding = torch.reshape(encoding_g[:, 1], (batch_size, 1, nx * ny * nz))
        sdf_encoding = sdf_encoding.expand(
            batch_size, volume_mesh_centers.shape[1], sdf_encoding.shape[2]
        )
        geo_encoding_long = torch.reshape(
            encoding_g[:, 2], (batch_size, 1, nx * ny * nz)
        )
        geo_encoding_long = geo_encoding_long.expand(
            batch_size, volume_mesh_centers.shape[1], geo_encoding_long.shape[2]
        )

        geo_encoding_sampled = torch.gather(geo_encoding, 2, mapping) * mask
        sdf_encoding_sampled = torch.gather(sdf_encoding, 2, mapping) * mask
        geo_encoding_long_sampled = torch.gather(geo_encoding_long, 2, mapping) * mask

        encoding_g = torch.cat(
            (geo_encoding_sampled, sdf_encoding_sampled, geo_encoding_long_sampled),
            axis=2,
        )
        encoding_g = self.activation(self.fc_1(encoding_g))
        encoding_g = self.fc_2(encoding_g)

        return encoding_g

    def geo_encoding_local(self, encoding_g, volume_mesh_centers, p_grid):
        """Function to calculate local geometry encoding from global encoding"""
        batch_size = volume_mesh_centers.shape[0]
        nx, ny, nz = (
            self.grid_resolution[0],
            self.grid_resolution[1],
            self.grid_resolution[2],
        )
        p_grid = torch.reshape(p_grid, (batch_size, nx * ny * nz, 3))
        mapping, outputs = self.bq_warp(
            volume_mesh_centers, p_grid, reverse_mapping=False
        )
        mapping = mapping.type(torch.int64)
        mask = mapping != 0

        geo_encoding = torch.reshape(encoding_g[:, 0], (batch_size, 1, nx * ny * nz))
        geo_encoding = geo_encoding.expand(
            batch_size, volume_mesh_centers.shape[1], geo_encoding.shape[2]
        )
        sdf_encoding = torch.reshape(encoding_g[:, 1], (batch_size, 1, nx * ny * nz))
        sdf_encoding = sdf_encoding.expand(
            batch_size, volume_mesh_centers.shape[1], sdf_encoding.shape[2]
        )
        geo_encoding_long = torch.reshape(
            encoding_g[:, 2], (batch_size, 1, nx * ny * nz)
        )
        geo_encoding_long = geo_encoding_long.expand(
            batch_size, volume_mesh_centers.shape[1], geo_encoding_long.shape[2]
        )

        geo_encoding_sampled = torch.gather(geo_encoding, 2, mapping) * mask
        sdf_encoding_sampled = torch.gather(sdf_encoding, 2, mapping) * mask
        geo_encoding_long_sampled = torch.gather(geo_encoding_long, 2, mapping) * mask

        encoding_g = torch.cat(
            (geo_encoding_sampled, sdf_encoding_sampled, geo_encoding_long_sampled),
            axis=2,
        )
        encoding_g = self.activation(self.fc_1(encoding_g))
        encoding_g = self.fc_2(encoding_g)

        return encoding_g

    def calculate_solution_with_neighbors(
        self,
        surface_mesh_centers,
        encoding_g,
        encoding_node,
        surface_mesh_neighbors,
        surface_normals,
        surface_neighbors_normals,
        surface_areas,
        surface_neighbors_areas,
    ):
        """Function to approximate solution given the neighborhood information"""
        num_variables = self.num_variables_surf
        nn_basis = self.nn_basis_surf
        agg_model = self.agg_model_surf
        num_sample_points = surface_mesh_neighbors.shape[2] + 1

        if self.use_surface_normals:
            if self.use_only_normals:
                surface_mesh_centers = torch.cat(
                    (surface_mesh_centers, surface_normals),
                    axis=-1,
                )
                surface_mesh_neighbors = torch.cat(
                    (
                        surface_mesh_neighbors,
                        surface_neighbors_normals,
                    ),
                    axis=-1,
                )

            else:
                surface_mesh_centers = torch.cat(
                    (surface_mesh_centers, surface_normals, 10**5 * surface_areas),
                    axis=-1,
                )
                surface_mesh_neighbors = torch.cat(
                    (
                        surface_mesh_neighbors,
                        surface_neighbors_normals,
                        10**5 * surface_neighbors_areas,
                    ),
                    axis=-1,
                )

        for f in range(num_variables):
            for p in range(num_sample_points):
                if p == 0:
                    volume_m_c = surface_mesh_centers
                else:
                    volume_m_c = surface_mesh_neighbors[:, :, p - 1]
                    noise = surface_mesh_centers - volume_m_c
                    dist = torch.sqrt(
                        noise[:, :, 0:1] ** 2.0
                        + noise[:, :, 1:2] ** 2.0
                        + noise[:, :, 2:3] ** 2.0
                    )
                basis_f = nn_basis[f](volume_m_c)
                output = torch.cat((basis_f, encoding_node, encoding_g), axis=-1)
                if p == 0:
                    output_center = agg_model[f](output)
                else:
                    if p == 1:
                        output_neighbor = agg_model[f](output) * (1.0 / dist)
                        dist_sum = 1.0 / dist
                    else:
                        output_neighbor += agg_model[f](output) * (1.0 / dist)
                        dist_sum += 1.0 / dist
            if num_sample_points > 1:
                output_res = 0.5 * output_center + 0.5 * output_neighbor / dist_sum
            else:
                output_res = output_center
            if f == 0:
                output_all = output_res
            else:
                output_all = torch.cat((output_all, output_res), axis=-1)

        return output_all

    def calculate_solution(
        self,
        volume_mesh_centers,
        encoding_g,
        encoding_node,
        eval_mode,
        num_sample_points=20,
        noise_intensity=50,
    ):
        """Function to approximate solution sampling the neighborhood information"""
        if eval_mode == "volume":
            num_variables = self.num_variables_vol
            nn_basis = self.nn_basis_vol
            agg_model = self.agg_model_vol
        elif eval_mode == "surface":
            num_variables = self.num_variables_surf
            nn_basis = self.nn_basis_surf
            agg_model = self.agg_model_surf
        for f in range(num_variables):
            for p in range(num_sample_points):
                if p == 0:
                    volume_m_c = volume_mesh_centers
                else:
                    noise = torch.rand_like(volume_mesh_centers)
                    noise = 2 * (noise - 0.5)
                    noise = noise / noise_intensity
                    dist = torch.sqrt(
                        noise[:, :, 0:1] ** 2.0
                        + noise[:, :, 1:2] ** 2.0
                        + noise[:, :, 2:3] ** 2.0
                    )
                    volume_m_c = volume_mesh_centers + noise
                basis_f = nn_basis[f](volume_m_c)
                output = torch.cat((basis_f, encoding_node, encoding_g), axis=-1)
                if p == 0:
                    output_center = agg_model[f](output)
                else:
                    if p == 1:
                        output_neighbor = agg_model[f](output) * (1.0 / dist)
                        dist_sum = 1.0 / dist
                    else:
                        output_neighbor += agg_model[f](output) * (1.0 / dist)
                        dist_sum += 1.0 / dist
            if num_sample_points > 1:
                output_res = 0.5 * output_center + 0.5 * output_neighbor / dist_sum
            else:
                output_res = output_center
            if f == 0:
                output_all = output_res
            else:
                output_all = torch.cat((output_all, output_res), axis=-1)

        return output_all

    def forward(
        self,
        data_dict,
    ):
        # Loading STL inputs, bounding box grids, precomputed SDF and scaling factors

        # STL nodes
        geo_centers = data_dict["geometry_coordinates"]

        # Bounding box grid
        s_grid = data_dict["surf_grid"]
        sdf_surf_grid = data_dict["sdf_surf_grid"]
        # Scaling factors
        surf_max = data_dict["surface_min_max"][:, 1]
        surf_min = data_dict["surface_min_max"][:, 0]

        if self.output_features_vol is not None:
            # Represent geometry on computational grid
            # Computational domain grid
            p_grid = data_dict["grid"]
            sdf_grid = data_dict["sdf_grid"]
            # Scaling factors
            vol_max = data_dict["volume_min_max"][:, 1]
            vol_min = data_dict["volume_min_max"][:, 0]

            # Normalize based on computational domain
            geo_centers_vol = 2.0 * (geo_centers - vol_min) / (vol_max - vol_min) - 1
            encoding_g_vol = self.geo_rep(geo_centers_vol, p_grid, sdf_grid)

            # Normalize based on BBox around surface (car)
            geo_centers_surf = (
                2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
            )
            encoding_g_surf = self.geo_rep(geo_centers_surf, s_grid, sdf_surf_grid)

            # SDF on volume mesh nodes
            sdf_nodes = data_dict["sdf_nodes"]
            # Positional encoding based on closest point on surface to a volume node
            pos_volume_closest = data_dict["pos_volume_closest"]
            # Positional encoding based on center of mass of geometry to volume node
            pos_volume_center_of_mass = data_dict["pos_volume_center_of_mass"]
            encoding_node_vol = torch.cat(
                (sdf_nodes, pos_volume_closest, pos_volume_center_of_mass), axis=-1
            )

            # Calculate positional encoding on volume nodes
            encoding_node_vol = self.position_encoder(
                encoding_node_vol, eval_mode="volume"
            )

        if self.output_features_surf is not None:
            # Represent geometry on bounding box
            geo_centers_surf = (
                2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
            )
            encoding_g_surf = self.geo_rep(geo_centers_surf, s_grid, sdf_surf_grid)

            # Positional encoding based on center of mass of geometry to surface node
            pos_surface_center_of_mass = data_dict["pos_surface_center_of_mass"]
            encoding_node_surf = pos_surface_center_of_mass

            # Calculate positional encoding on surface centers
            encoding_node_surf = self.position_encoder(
                encoding_node_surf, eval_mode="surface"
            )

        encoding_g = 0.5 * encoding_g_surf
        # Average the encodings
        if self.output_features_vol is not None:
            encoding_g += 0.5 * encoding_g_vol

        if self.output_features_vol is not None:
            # Calculate local geometry encoding for volume
            # Sampled points on volume
            volume_mesh_centers = data_dict["volume_mesh_centers"]
            encoding_g_vol = self.geo_encoding_local(
                encoding_g, volume_mesh_centers, p_grid
            )

            # Approximate solution on volume node
            output_vol = self.calculate_solution(
                volume_mesh_centers,
                encoding_g_vol,
                encoding_node_vol,
                eval_mode="volume",
            )
        else:
            output_vol = None

        if self.output_features_surf is not None:
            # Sampled points on surface
            surface_mesh_centers = data_dict["surface_mesh_centers"]
            surface_normals = data_dict["surface_normals"]
            surface_areas = data_dict["surface_areas"]

            # Neighbors of sampled points on surface
            surface_mesh_neighbors = data_dict["surface_mesh_neighbors"]
            surface_neighbors_normals = data_dict["surface_neighbors_normals"]
            surface_neighbors_areas = data_dict["surface_neighbors_areas"]
            surface_areas = torch.unsqueeze(surface_areas, -1)
            surface_neighbors_areas = torch.unsqueeze(surface_neighbors_areas, -1)
            # Calculate local geometry encoding for surface
            encoding_g_surf = self.geo_encoding_local_surface(
                encoding_g, surface_mesh_centers, s_grid
            )

            # Approximate solution on surface cell center
            if not self.surface_neighbors:
                output_surf = self.calculate_solution(
                    surface_mesh_centers,
                    encoding_g_surf,
                    encoding_node_surf,
                    eval_mode="surface",
                    num_sample_points=1,
                    noise_intensity=500,
                )
            else:
                output_surf = self.calculate_solution_with_neighbors(
                    surface_mesh_centers,
                    encoding_g_surf,
                    encoding_node_surf,
                    surface_mesh_neighbors,
                    surface_normals,
                    surface_neighbors_normals,
                    surface_areas,
                    surface_neighbors_areas,
                )
        else:
            output_surf = None

        return output_vol, output_surf
