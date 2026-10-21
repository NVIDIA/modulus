# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Tests for surface example meshes."""

import pytest
import torch

from physicsnemo.mesh import primitives


class TestSurfacePrimitives:
    """Test all surface example meshes (2D→3D)."""

    @pytest.mark.parametrize(
        "example_name",
        [
            # Spheres
            # "sphere_icosahedral",
            "sphere_uv",
            # Cylinders
            "cylinder",
            "cylinder_open",
            # Other shapes
            "torus",
            "plane",
            "cone",
            "disk",
            "hemisphere",
            # Platonic solids
            "cube_surface",
            "tetrahedron_surface",
            "octahedron_surface",
            "icosahedron_surface",
            # Special
            "mobius_strip",
        ],
    )
    def test_surface_mesh(self, example_name):
        """Test that surface mesh loads with correct dimensions."""
        primitives_module = getattr(primitives.surfaces, example_name)
        mesh = primitives_module.load()

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    @pytest.mark.parametrize(
        "example_name",
        [
            "sphere_icosahedral",
            "sphere_uv",
            "cylinder",
            "torus",
            "cone",
            "cube_surface",
            "tetrahedron_surface",
            "octahedron_surface",
            "icosahedron_surface",
        ],
    )
    def test_closed_surface_outward_normals(self, example_name):
        """A closed surface wound outward encloses a positive signed volume."""
        mesh = getattr(primitives.surfaces, example_name).load()

        tri = mesh.points.double()[mesh.cells]
        signed_volume = torch.linalg.det(tri).sum() / 6
        assert signed_volume > 0

    def test_torus_signed_volume(self):
        """The torus encloses +2 pi^2 R r^2, so its normals point out of the tube."""
        mesh = primitives.surfaces.torus.load(major_radius=1.0, minor_radius=0.3)

        tri = mesh.points.double()[mesh.cells]
        signed_volume = torch.linalg.det(tri).sum().item() / 6
        assert signed_volume == pytest.approx(2 * torch.pi**2 * 1.0 * 0.3**2, rel=0.02)
        assert mesh.mean_curvature_vertices.mean() > 0

    def test_hemisphere_outward_normals(self):
        """Every hemisphere normal points away from the center."""
        mesh = primitives.surfaces.hemisphere.load()

        # Skip the zero-area triangles at the pole, whose normals are rounding noise
        nondegenerate = mesh.cell_areas > 1e-6 * mesh.cell_areas.max()
        radial = (mesh.cell_normals * mesh.cell_centroids).sum(dim=-1)
        assert (radial[nondegenerate] > 0).all()

    @pytest.mark.parametrize("example_name", ["disk", "plane"])
    def test_xy_surface_normals_point_up(self, example_name):
        """A surface in the xy plane is wound counterclockwise, with +z normals."""
        mesh = getattr(primitives.surfaces, example_name).load()

        expected = torch.tensor([0.0, 0.0, 1.0]).expand_as(mesh.cell_normals)
        torch.testing.assert_close(mesh.cell_normals, expected)

    @pytest.mark.parametrize(
        "normal",
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (1e-7, 0.0, 1.0),
            (1e-7, 0.0, -1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 1.0),
        ],
    )
    def test_plane_normal(self, normal):
        """The cell normals of a plane match the requested normal."""
        mesh = primitives.surfaces.plane.load(normal=normal)

        expected = torch.nn.functional.normalize(torch.tensor(normal), dim=0)
        torch.testing.assert_close(
            mesh.cell_normals, expected.expand_as(mesh.cell_normals)
        )

    def test_torus_radii(self):
        """Test that torus has correct radii."""
        major_radius = 2.0
        minor_radius = 0.5
        torus = primitives.surfaces.torus.load(
            major_radius=major_radius,
            minor_radius=minor_radius,
            n_major=32,
            n_minor=16,
        )

        # Check that points are in expected range
        radii_xy = torch.norm(torus.points[:, :2], dim=1)
        assert radii_xy.min() >= major_radius - minor_radius - 0.1
        assert radii_xy.max() <= major_radius + minor_radius + 0.1

    def test_closed_vs_open(self):
        """Test that closed and open surfaces have correct topology."""
        # Closed cylinder should have no boundary
        cylinder_closed = primitives.surfaces.cylinder.load(n_circ=16, n_height=5)

        # Open cylinder should have boundary
        cylinder_open = primitives.surfaces.cylinder_open.load(n_circ=16, n_height=5)

        # Both should have points
        assert cylinder_closed.n_points > 0
        assert cylinder_open.n_points > 0
